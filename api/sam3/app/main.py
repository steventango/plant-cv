import base64
import gc
import io
import time
import uuid
import zlib
from collections import OrderedDict
from contextlib import nullcontext

import cv2
import litserve as ls
import numpy as np
import torch
from transformers import (
    Sam3VideoConfig,
    Sam3VideoInferenceSession,
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from utils import _move_to_device, decode_image, mask_to_contour


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
    t0 = time.time()
    prune_session(inference_session)
    t_prune = time.time() - t0

    buffer = io.BytesIO()
    torch.save(inference_session, buffer)
    t_save = time.time() - t0 - t_prune

    raw_bytes = buffer.getvalue()
    raw_size = len(raw_bytes)

    compressed = zlib.compress(raw_bytes)
    t_compress = time.time() - t0 - t_prune - t_save
    comp_size = len(compressed)

    b64_str = base64.b64encode(compressed).decode("utf-8")
    t_b64 = time.time() - t0 - t_prune - t_save - t_compress

    print(
        f"DEBUG: serialize raw={raw_size / 1e6:.2f}MB, comp={comp_size / 1e6:.2f}MB, timings: prune:{t_prune:.3f}s, save:{t_save:.3f}s, compress:{t_compress:.3f}s, b64:{t_b64:.3f}s",
        flush=True,
    )
    return b64_str


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


class SessionCache:
    """In-memory cache for SAM3 inference sessions."""

    def __init__(self, max_size: int):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, session_id: str) -> Sam3VideoInferenceSession:
        if session_id in self.cache:
            self.cache.move_to_end(session_id)
            return self.cache[session_id]
        # TODO: load from checkpoint
        return None

    def set(self, session_id: str, session: Sam3VideoInferenceSession):
        if session_id in self.cache:
            self.cache.move_to_end(session_id)
        self.cache[session_id] = session
        # TODO: save checkpoint
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            gc.collect()
            torch.cuda.empty_cache()


def process_outputs(
    processor: Sam3VideoProcessor,
    session: Sam3VideoInferenceSession,
    model_outputs: dict,
    inference_size: tuple[int, int],
    original_size: tuple[int, int],
) -> dict:
    """Process model outputs into mask info dictionary grouped by prompt."""
    inf_h, inf_w = inference_size
    orig_h, orig_w = original_size

    # Process to inference resolution first
    processed = processor.postprocess_outputs(
        session,
        model_outputs,
        original_sizes=[[inf_h, inf_w]],
    )

    obj_ids = processed.get("object_ids", [])
    masks = processed.get("masks", torch.tensor([]))
    boxes = processed.get("boxes", torch.tensor([]))
    scores = processed.get("scores", torch.tensor([]))
    prompt_to_obj_ids = processed.get("prompt_to_obj_ids", {})

    total_masks_list = []
    if len(obj_ids) > 0:
        masks_np = masks.to(torch.float32).cpu().numpy().astype(bool)
        boxes_np = boxes.to(torch.float32).cpu().numpy()
        scores_np = scores.to(torch.float32).cpu().numpy()
        ids_np = (
            obj_ids.cpu().numpy().astype(int)
            if isinstance(obj_ids, torch.Tensor)
            else np.array(obj_ids)
        )

        print(
            f"DEBUG: SAM3 reporting {len(ids_np)} objects: {ids_np.tolist()} with scores: {scores_np.tolist()}",
            flush=True,
        )

        scale_x = orig_w / inf_w
        scale_y = orig_h / inf_h

        for i, (mask, box, score, obj_id) in enumerate(
            zip(masks_np, boxes_np, scores_np, ids_np)
        ):
            if mask.ndim == 3:
                mask = mask.squeeze(0)

            # Find contour on low-res mask
            contour_low = mask_to_contour(mask)

            # Rescale contour points mathematically (faster than resizing mask)
            contour_high = []
            if contour_low:
                contour_np = np.array(contour_low)
                contour_scaled = contour_np * [scale_x, scale_y]
                contour_high = contour_scaled.tolist()

            box_high = [
                float(box[0] * scale_x),
                float(box[1] * scale_y),
                float(box[2] * scale_x),
                float(box[3] * scale_y),
            ]

            total_masks_list.append(
                {
                    "object_id": int(obj_id),
                    "contour": contour_high,
                    "box": box_high,
                    "score": float(score),
                }
            )

    return {"masks": total_masks_list, "prompt_to_obj_ids": prompt_to_obj_ids}


class SAM3API(ls.LitAPI):
    """SAM3 Video Tracking API with stateless operation."""

    def setup(self, device: str):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load SAM3 Video Model
        self.config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        self.model = Sam3VideoModel.from_pretrained(
            "facebook/sam3", config=self.config
        ).to(device, dtype=torch.bfloat16)
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        # Inference resolution setting
        self.inference_height = 1008

        # Initialize session cache
        self.session_cache = SessionCache(max_size=24)

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

    def _update_config(self, request: dict):
        self.model.score_threshold_detection = float(
            request.get(
                "score_threshold_detection", self.config.score_threshold_detection
            )
        )
        self.model.new_det_thresh = float(
            request.get("new_det_thresh", self.config.new_det_thresh)
        )
        self.model.high_conf_thresh = float(
            request.get("high_conf_thresh", self.config.high_conf_thresh)
        )
        self.model.high_iou_thresh = float(
            request.get("high_iou_thresh", self.config.high_iou_thresh)
        )
        self.model.recondition_every_nth_frame = int(
            request.get(
                "recondition_every_nth_frame", self.config.recondition_every_nth_frame
            )
        )
        self.model.max_num_objects = int(
            request.get("max_num_objects", self.config.max_num_objects)
        )
        self.model.recondition_on_trk_masks = bool(
            request.get(
                "recondition_on_trk_masks", self.config.recondition_on_trk_masks
            )
        )
        self.model.det_nms_thresh = float(
            request.get("det_nms_thresh", self.config.det_nms_thresh)
        )
        self.model.assoc_iou_thresh = float(
            request.get("assoc_iou_thresh", self.config.assoc_iou_thresh)
        )
        self.model.trk_assoc_iou_thresh = float(
            request.get("trk_assoc_iou_thresh", self.config.trk_assoc_iou_thresh)
        )
        self.model.init_trk_keep_alive = int(
            request.get("init_trk_keep_alive", self.config.init_trk_keep_alive)
        )
        self.model.max_trk_keep_alive = int(
            request.get("max_trk_keep_alive", self.config.max_trk_keep_alive)
        )
        self.model.min_trk_keep_alive = int(
            request.get("min_trk_keep_alive", self.config.min_trk_keep_alive)
        )
        self.model.fill_hole_area = int(
            request.get("fill_hole_area", self.config.fill_hole_area)
        )
        self.model.low_res_mask_size = int(
            request.get("low_res_mask_size", self.config.low_res_mask_size)
        )
        self.model.hotstart_delay = int(
            request.get("hotstart_delay", self.config.hotstart_delay)
        )
        self.model.suppress_overlapping_based_on_recent_occlusion_threshold = float(
            request.get(
                "suppress_overlapping_based_on_recent_occlusion_threshold",
                self.config.suppress_overlapping_based_on_recent_occlusion_threshold,
            )
        )

    def _detect(self, request: dict):
        """Initialize tracking session for a single prompt."""
        t_start = time.time()
        image_data = request["image_data"]
        text_prompt = request.get("text_prompt")

        if not text_prompt:
            return {"error": "text_prompt is required for detect"}

        self._update_config(request)

        image_np = decode_image(image_data)
        h, w = image_np.shape[:2]
        t_decode = time.time() - t_start

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
            t_pre = time.time() - t_start - t_decode

            # Run inference
            model_outputs = self.model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )
            t_model = time.time() - t_start - t_decode - t_pre

            processed_results = process_outputs(
                self.processor,
                session,
                model_outputs,
                inference_size=(self.inference_height, inference_width),
                original_size=(h, w),
            )
            t_post = time.time() - t_start - t_decode - t_pre - t_model

            # Update cache and return session_id
            session_id = str(uuid.uuid4())
            self.session_cache.set(session_id, session)
            t_serialize = time.time() - t_start - t_decode - t_pre - t_model - t_post

            results = {"session_id": session_id, **processed_results}
            del model_outputs, inputs

        gc.collect()
        torch.cuda.empty_cache()

        t_total = time.time() - t_start
        print(
            f"TIMING: detect total {t_total:.4f}s (decode:{t_decode:.4f}, pre:{t_pre:.4f}, model:{t_model:.4f}, post:{t_post:.4f}, serialize:{t_serialize:.4f})",
            flush=True,
        )

        return results

    def _propagate(self, request: dict):
        """Propagate tracking to next frame for a single state."""
        t_start = time.time()
        image_data = request["image_data"]
        session_id = request["session_id"]

        self._update_config(request)

        image_np = decode_image(image_data)
        h, w = image_np.shape[:2]
        t_decode = time.time() - t_start

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
        ):
            # Load session
            t_deserialize = 0
            session = self.session_cache.get(session_id)
            if session is None:
                return {"error": "Session expired or not found in cache"}
            t_deserialize = time.time() - t_start - t_decode

            # Resize frame
            inference_width = int(w * (self.inference_height / h))
            image_resized = cv2.resize(
                image_np, (inference_width, self.inference_height)
            )

            inputs = self.processor(
                images=image_resized,
                device=self.device,
                return_tensors="pt",
            )
            t_pre = time.time() - t_start - t_decode - t_deserialize

            # Clear GPU memory and prune session before inference to limit memory growth
            torch.cuda.empty_cache()
            prune_session(session)

            model_outputs = self.model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )
            t_model = time.time() - t_start - t_decode - t_deserialize - t_pre

            processed_results = process_outputs(
                self.processor,
                session,
                model_outputs,
                inference_size=(self.inference_height, inference_width),
                original_size=(h, w),
            )
            t_post = time.time() - t_start - t_decode - t_deserialize - t_pre - t_model

            # Update cache
            t_cache_update = 0
            t_cache_start = time.time()
            self.session_cache.set(session_id, session)
            t_cache_update = time.time() - t_cache_start

            results = {"session_id": session_id, **processed_results}
            del model_outputs, inputs

        gc.collect()
        torch.cuda.empty_cache()

        t_total = time.time() - t_start
        print(
            f"TIMING: propagate total {t_total:.4f}s (decode:{t_decode:.4f}, deserialize:{t_deserialize:.4f}, pre:{t_pre:.4f}, model:{t_model:.4f}, post:{t_post:.4f}, cache_update:{t_cache_update:.4f})",
            flush=True,
        )

        return results

    def encode_response(self, result: dict):
        return result


if __name__ == "__main__":
    api = SAM3API()
    server = ls.LitServer(api)
    server.run(port=8805, num_api_servers=1, generate_client_file=False)
