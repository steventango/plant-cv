import base64
import io
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

    def decode_request(self, request):
        image_data = request["image_data"]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        if width % 16 != 0 or height % 16 != 0:
            new_width = ((width + 15) // 16) * 16
            new_height = ((height + 15) // 16) * 16
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return {
            "image": image,
            "embedding_types": request.get("embedding_types", ["cls_token"]),
        }

    def batch(self, inputs):
        images = [item["image"] for item in inputs]
        embedding_typess = [item["embedding_types"] for item in inputs]
        return images, embedding_typess

    def predict(self, batch_data):
        if isinstance(batch_data, dict):
            # LitServe skipped batch() because max_batch_size=1
            images = [batch_data["image"]]
            embedding_typess = [batch_data["embedding_types"]]
        else:
            # LitServe called batch()
            images, embedding_typess = batch_data

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
        ):
            features = self.feature_extractor(images)

        features = np.asarray(features)
        results = []
        for last_hidden_state, embedding_types in zip(features[:, 0], embedding_typess):
            result = {}
            for embedding_type in embedding_types:
                if embedding_type == "cls_token":
                    # CLS token is the first token
                    result["cls_token"] = last_hidden_state[0, :].tolist()
                elif embedding_type == "patch_features":
                    # Skip CLS token (index 0) and register tokens
                    num_register_tokens = 4
                    result["patch_features"] = last_hidden_state[
                        1 + num_register_tokens :, :
                    ].tolist()
                else:
                    raise ValueError(
                        f"Unknown embedding_type: {embedding_type}. Use 'cls_token' or 'patch_features'"
                    )
            results.append(result)

        return results

    def encode_response(self, result):
        if isinstance(result, list):
            # LitServe skipped unbatch() because max_batch_size=1
            return result[0]
        return result


if __name__ == "__main__":
    api = EmbeddingsAPI(max_batch_size=1, batch_timeout=0.01)
    server = ls.LitServer(api, accelerator="gpu", devices=1, workers_per_device=1)
    server.run(port=8803, num_api_servers=2, generate_client_file=False)
