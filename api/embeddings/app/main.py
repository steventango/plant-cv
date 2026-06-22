import base64
import io
from contextlib import nullcontext

import litserve as ls
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _decode_and_resize(image_data: str) -> Image.Image:
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    if width % 16 != 0 or height % 16 != 0:
        new_width = ((width + 15) // 16) * 16
        new_height = ((height + 15) // 16) * 16
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return image


class EmbeddingsAPI(ls.LitAPI):
    def setup(
        self,
        device,
        pretrained_model_name_or_path=_MODEL_NAME,
    ):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = (
            AutoModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        )

    def decode_request(self, request):
        image_data = request["image_data"]
        embedding_types = request.get("embedding_types", ["cls_token"])
        if isinstance(image_data, list):
            images = [_decode_and_resize(d) for d in image_data]
            return {"images": images, "embedding_types": embedding_types, "is_batch": True}
        image = _decode_and_resize(image_data)
        return {"images": [image], "embedding_types": embedding_types, "is_batch": False}

    def predict(self, inputs):
        images = inputs["images"]
        embedding_types = inputs["embedding_types"]

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
        ):
            processor_inputs = self.processor(images=images, return_tensors="pt").to(
                self.device
            )
            outputs = self.model(**processor_inputs)

        # (N, seq_len, hidden_dim) on CPU
        last_hidden_states = outputs.last_hidden_state.cpu().float()

        results = []
        for last_hidden_state in last_hidden_states:
            result = {}
            for embedding_type in embedding_types:
                if embedding_type == "cls_token":
                    result["cls_token"] = last_hidden_state[0, :].tolist()
                elif embedding_type == "patch_features":
                    num_register_tokens = 4
                    result["patch_features"] = last_hidden_state[
                        1 + num_register_tokens :, :
                    ].tolist()
                else:
                    raise ValueError(
                        f"Unknown embedding_type: {embedding_type}. Use 'cls_token' or 'patch_features'"
                    )
            results.append(result)

        return {"results": results, "is_batch": inputs["is_batch"]}

    def encode_response(self, output):
        if output["is_batch"]:
            return output["results"]
        return output["results"][0]


if __name__ == "__main__":
    api = EmbeddingsAPI()
    server = ls.LitServer(
        api, accelerator="gpu", devices=1, timeout=False
    )
    server.run(port=8803, generate_client_file=False)
