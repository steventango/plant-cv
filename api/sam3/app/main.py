import base64
import gc
import io
import zlib
from contextlib import nullcontext

import cv2
import litserve as ls
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoInferenceSession

from .utils import _move_to_device, decode_image, mask_to_contour


def prune_session(session: Sam3VideoInferenceSession, keep_frames: int = 2):
    """
    Prune session state to keep it small, only keeping the most recent frames and features.
    """
    if not hasattr(session, "processed_frames"):
        return

    all_frame_indices = sorted(session.processed_frames.keys())
    if len(all_frame_indices) <= keep_frames:
        return

    # Indices to remove
    to_remove = all_frame_indices[:-keep_frames]

    # Prune processed_frames
    for idx in to_remove:
        if idx in session.processed_frames:
            del session.processed_frames[idx]

    # Prune cache features
    if hasattr(session, "cache") and hasattr(session.cache, "_vision_features"):
        for idx in to_remove:
            if idx in session.cache._vision_features:
                del session.cache._vision_features[idx]

    # Prune output_buffer and other temporary state
    if hasattr(session, "output_buffer"):
        session.output_buffer = []


def serialize_state(inference_session: Sam3VideoInferenceSession) -> str:
    """Serialize the entire inference session object to compressed base64 string."""
    prune_session(inference_session)

    buffer = io.BytesIO()
    torch.save(inference_session, buffer)
    compressed = zlib.compress(buffer.getvalue())
    return base64.b64encode(compressed).decode("utf-8")


def deserialize_state(state_str: str, device: str) -> Sam3VideoInferenceSession:
    """Deserialize compressed state and restore the inference session object."""
    compressed_bytes = base64.b64decode(state_str)
    state_bytes = zlib.decompress(compressed_bytes)

    buffer = io.BytesIO(state_bytes)
    # Load to CPU first to avoid intermediate OOM
    session = torch.load(buffer, map_location="cpu", weights_only=False)

    # Update inference device
    session.inference_device = str(device)

    # Move attributes to device, excluding large offloaded data
    for k, v in session.__dict__.items():
        if k in ["processed_frames", "video_data"]:
            continue
        session.__dict__[k] = _move_to_device(v, device)

    if hasattr(session, "cache"):
        session.cache = _move_to_device(session.cache, device)

    return session


def process_outputs(
    processor: Sam3VideoProcessor,
    session: Sam3VideoInferenceSession,
    model_outputs: dict,
    inference_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list:
    """Process model outputs into mask info list with manual upscaling."""
    inf_h, inf_w = inference_size
    orig_h, orig_w = original_size

    # Process to inference resolution first
    # This ensures internal padding/offset correction is relative to the 480p image
    processed = processor.postprocess_outputs(
        session,
        model_outputs,
        original_sizes=[[inf_h, inf_w]],
    )

    masks_list = []
    obj_ids = processed.get("object_ids", [])
    masks = processed.get("masks", torch.tensor([]))
    boxes = processed.get("boxes", torch.tensor([]))
    scores = processed.get("scores", torch.tensor([]))

    if len(obj_ids) > 0:
        masks_np = masks.to(torch.float32).cpu().numpy().astype(bool)
        boxes_np = boxes.to(torch.float32).cpu().numpy()
        scores_np = scores.to(torch.float32).cpu().numpy()
        ids_np = (
            obj_ids.cpu().numpy().astype(int)
            if isinstance(obj_ids, torch.Tensor)
            else np.array(obj_ids)
        )

        # Calculate manual scale factors to map back to true original image
        scale_x = orig_w / inf_w
        scale_y = orig_h / inf_h

        for i, (mask, box, score, obj_id) in enumerate(
            zip(masks_np, boxes_np, scores_np, ids_np)
        ):
            if mask.ndim == 3:
                mask = mask.squeeze(0)

            # Manually upscale mask to original resolution
            mask_high = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                > 0.5
            )

            # Manually upscale box to original resolution
            box_high = [
                float(box[0] * scale_x),
                float(box[1] * scale_y),
                float(box[2] * scale_x),
                float(box[3] * scale_y),
            ]

            contour = mask_to_contour(mask_high)
            masks_list.append(
                {
                    "object_id": int(obj_id),
                    "contour": contour,
                    "box": box_high,
                    "score": float(score),
                }
            )

    return masks_list


class SAM3API(ls.LitAPI):
    """SAM3 Video Tracking API with stateless operation."""

    def setup(self, device: str):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load SAM3 Video Model
        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            device, dtype=torch.bfloat16
        )
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        # Inference resolution setting
        self.inference_height = 480

    def decode_request(self, request: dict):
        return request

    def predict(self, request: dict):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        endpoint = request.get("endpoint", "detect")

        if endpoint == "detect":
            return self._detect(request)
        elif endpoint == "propagate":
            return self._propagate(request)
        elif endpoint == "health":
            return {"status": "healthy"}
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}

    def _detect(self, request: dict):
        """Initialize tracking session for a single prompt."""
        image_data = request["image_data"]
        text_prompt = request.get("text_prompt")

        if not text_prompt:
            return {"error": "text_prompt is required for detect"}

        image_np = decode_image(image_data)
        h, w = image_np.shape[:2]

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
        ):
            # Create fresh session for this prompt
            session = self.processor.init_video_session(
                inference_device=self.device,
                inference_state_device="cpu",
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
                max_vision_features_cache_size=1,
            )

            # Add text prompt
            self.processor.add_text_prompt(
                inference_session=session,
                text=text_prompt,
            )

            # Process first frame - resize for memory efficiency
            inference_width = int(w * (self.inference_height / h))
            image_resized = cv2.resize(
                image_np, (inference_width, self.inference_height)
            )

            inputs = self.processor(
                images=image_resized,
                device=self.device,
                return_tensors="pt",
            )

            # Run inference
            model_outputs = self.model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )

            # Process outputs
            masks_list = process_outputs(
                self.processor,
                session,
                model_outputs,
                inference_size=(self.inference_height, inference_width),
                original_size=(h, w),
            )

            # Serialize state
            state_str = serialize_state(session)

            results = {"state": state_str, "masks": masks_list}

            del session, model_outputs, inputs

        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _propagate(self, request: dict):
        """Propagate tracking to next frame for a single state."""
        image_data = request["image_data"]
        state = request.get("state")

        if not state:
            return {"error": "state is required for propagate"}

        image_np = decode_image(image_data)
        h, w = image_np.shape[:2]

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
        ):
            session = deserialize_state(state, self.device)

            # Process frame - resize for memory efficiency
            inference_width = int(w * (self.inference_height / h))
            image_resized = cv2.resize(
                image_np, (inference_width, self.inference_height)
            )

            inputs = self.processor(
                images=image_resized,
                device=self.device,
                return_tensors="pt",
            )
            model_outputs = self.model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )
            masks_list = process_outputs(
                self.processor,
                session,
                model_outputs,
                inference_size=(self.inference_height, inference_width),
                original_size=(h, w),
            )

            state_str = serialize_state(session)

            results = {"state": state_str, "masks": masks_list}

            del session, model_outputs, inputs

        gc.collect()
        torch.cuda.empty_cache()

        return results

    def encode_response(self, result: dict):
        return result


if __name__ == "__main__":
    api = SAM3API()
    server = ls.LitServer(api)
    server.run(port=8805, num_api_servers=1, generate_client_file=False)
