import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import litserve as ls
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class EmbeddingsAPI(ls.LitAPI):
    def setup(
        self,
        device,
        pretrained_model_name_or_path="facebook/dinov3-vitb16-pretrain-lvd1689m",
    ):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.feature_extractor = pipeline(
            model=pretrained_model_name_or_path,
            task="image-feature-extraction",
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def decode_request(self, request):
        image_data = request["image_data"]
        embedding_types = request.get("embedding_types", ["cls_token"])

        if isinstance(embedding_types, str):
            embedding_types = [embedding_types]

        return {
            "image_data": image_data,
            "embedding_types": embedding_types,
        }

    def batch(self, inputs):
        def process_input(item):
            image_data = item["image_data"]
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            return {
                "image": pil_image,
                "embedding_types": item["embedding_types"],
            }

        processed_items = list(self.pool.map(process_input, inputs))

        images = [item["image"] for item in processed_items]
        embedding_typess = [item["embedding_types"] for item in processed_items]

        return images, embedding_typess

    def predict(self, batch_input):
        images, embedding_typess = batch_input

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
        ):
            resized_images = []
            for image in images:
                width, height = image.size
                new_width = ((width + 15) // 16) * 16
                new_height = ((height + 15) // 16) * 16
                resized_image = image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                resized_images.append(resized_image)

            features = self.feature_extractor(resized_images)
            features = np.asarray(
                features
            )  # (batch_size, 1, 1 + num_register_tokens + num_patches, hidden_size)

            results = []
            for i, embedding_types in enumerate(embedding_typess):
                last_hidden_state = features[
                    i, 0, :, :
                ]  # (1 + num_register_tokens + num_patches, hidden_size)

                result = {}
                for embedding_type in embedding_types:
                    if embedding_type == "cls_token":
                        # CLS token is the first token
                        result["cls_token"] = last_hidden_state[0, :]
                    elif embedding_type == "patch_features":
                        # Skip CLS token (index 0) and register tokens
                        num_register_tokens = 4
                        result["patch_features"] = last_hidden_state[
                            1 + num_register_tokens :, :
                        ]
                    else:
                        raise ValueError(
                            f"Unknown embedding_type: {embedding_type}. Use 'cls_token' or 'patch_features'"
                        )

                results.append(result)

            return results

    def unbatch(self, output):
        return output

    def encode_response(self, result):
        response = {}
        for embedding_type, embedding in result.items():
            response[embedding_type] = embedding.tolist()
        return response


if __name__ == "__main__":
    api = EmbeddingsAPI(max_batch_size=16, batch_timeout=0.01)
    server = ls.LitServer(api)
    server.run(port=8000, num_api_servers=4, generate_client_file=False)
