import time
import uuid
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
from utils import decode_image, mask_to_contour, _move_to_device
from caching import SessionCache, PersistenceManager, prune_session


PATCH_SIZE = 14
DEFAULT_INFERENCE_HEIGHT = 1008
MAX_RESOLUTIONS = 4


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
        config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        self.model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
            device, dtype=torch.bfloat16
        )
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        # Model per inference resolution, all sharing the base model's weights.
        self.models = {DEFAULT_INFERENCE_HEIGHT: self.model}

        # Initialize session cache
        self.persistence = PersistenceManager(persistence_dir="checkpoints")
        self.session_cache = SessionCache(
            max_size=24, persistence_manager=self.persistence, device=device
        )

    def decode_request(self, request: dict):
        return request

    def predict(self, request: dict):
        # Load-shedding: if the client's deadline has already passed, skip this
        # request without touching the session. During a CV-request storm this
        # sheds work whose zone has already timed out, so SAM3 spends its serial
        # capacity on requests still within their deadline and the effective
        # queue stays short. Returning before any session load/mutation means
        # tracking state is not advanced (equivalent to a skipped frame).
        deadline = request.get("deadline")
        if deadline is not None and time.time() > float(deadline):
            print("DEBUG: skipping request past client deadline", flush=True)
            return {
                "skipped": True,
                "session_id": request.get("session_id"),
                "masks": [],
            }

        endpoint = request.get("endpoint", "detect")

        if endpoint == "detect":
            return self._detect(request)
        elif endpoint == "propagate":
            return self._propagate(request)
        elif endpoint == "health":
            return {"status": "healthy"}
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}

    def _model_for(self, height) -> Sam3VideoModel:
        height = int(height)
        if height in self.models:
            return self.models[height]

        if height % (2 * PATCH_SIZE):
            raise ValueError(
                f"inference_height must be a multiple of {2 * PATCH_SIZE} (patch size "
                f"{PATCH_SIZE}), got {height}"
            )
        if len(self.models) >= MAX_RESOLUTIONS:
            raise ValueError(
                f"already serving {sorted(self.models)}; refusing to build a model for "
                f"{height} as well (MAX_RESOLUTIONS={MAX_RESOLUTIONS})"
            )

        t0 = time.time()
        config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        config.image_size = height
        config.low_res_mask_size = 4 * height // PATCH_SIZE

        model = Sam3VideoModel(config)
        model.load_state_dict(self.model.state_dict(), assign=True)
        self.models[height] = model.to(self.device, dtype=torch.bfloat16).eval()
        print(
            f"DEBUG: built SAM3 model for {height}x{height} in {time.time() - t0:.2f}s",
            flush=True,
        )
        return self.models[height]

    def _update_config(self, model: Sam3VideoModel, request: dict):
        model.score_threshold_detection = float(
            request.get(
                "score_threshold_detection", model.config.score_threshold_detection
            )
        )
        model.new_det_thresh = float(
            request.get("new_det_thresh", model.config.new_det_thresh)
        )
        model.high_conf_thresh = float(
            request.get("high_conf_thresh", model.config.high_conf_thresh)
        )
        model.high_iou_thresh = float(
            request.get("high_iou_thresh", model.config.high_iou_thresh)
        )
        model.recondition_every_nth_frame = int(
            request.get(
                "recondition_every_nth_frame", model.config.recondition_every_nth_frame
            )
        )
        model.max_num_objects = int(
            request.get("max_num_objects", model.config.max_num_objects)
        )
        model.recondition_on_trk_masks = bool(
            request.get(
                "recondition_on_trk_masks", model.config.recondition_on_trk_masks
            )
        )
        model.det_nms_thresh = float(
            request.get("det_nms_thresh", model.config.det_nms_thresh)
        )
        model.assoc_iou_thresh = float(
            request.get("assoc_iou_thresh", model.config.assoc_iou_thresh)
        )
        model.trk_assoc_iou_thresh = float(
            request.get("trk_assoc_iou_thresh", model.config.trk_assoc_iou_thresh)
        )
        model.init_trk_keep_alive = int(
            request.get("init_trk_keep_alive", model.config.init_trk_keep_alive)
        )
        model.max_trk_keep_alive = int(
            request.get("max_trk_keep_alive", model.config.max_trk_keep_alive)
        )
        model.min_trk_keep_alive = int(
            request.get("min_trk_keep_alive", model.config.min_trk_keep_alive)
        )
        model.fill_hole_area = int(
            request.get("fill_hole_area", model.config.fill_hole_area)
        )
        model.low_res_mask_size = int(
            request.get("low_res_mask_size", model.config.low_res_mask_size)
        )
        model.hotstart_delay = int(
            request.get("hotstart_delay", model.config.hotstart_delay)
        )
        model.suppress_overlapping_based_on_recent_occlusion_threshold = float(
            request.get(
                "suppress_overlapping_based_on_recent_occlusion_threshold",
                model.config.suppress_overlapping_based_on_recent_occlusion_threshold,
            )
        )

    def _detect(self, request: dict):
        """Initialize tracking session for a single prompt."""
        t_start = time.time()
        image_data = request["image_data"]
        text_prompt = request.get("text_prompt")

        if not text_prompt:
            return {"error": "text_prompt is required for detect"}

        model = self._model_for(
            request.get("inference_height") or DEFAULT_INFERENCE_HEIGHT
        )
        inference_height = model.config.image_size
        self._update_config(model, request)

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

            # This pre-resize does not change the shape the model sees, but the
            # resample perturbs pixels slightly so we are keeping it for now.
            inference_width = int(w * (inference_height / h))
            image_resized = cv2.resize(image_np, (inference_width, inference_height))
            inputs = self.processor(
                images=image_resized,
                size={"height": inference_height, "width": inference_height},
                device=self.device,
                return_tensors="pt",
            )
            t_pre = time.time() - t_start - t_decode

            # Run inference
            model_outputs = model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )
            t_model = time.time() - t_start - t_decode - t_pre

            processed_results = process_outputs(
                self.processor,
                session,
                model_outputs,
                inference_size=(inference_height, inference_width),
                original_size=(h, w),
            )
            t_post = time.time() - t_start - t_decode - t_pre - t_model

            # Move session tensors back to CPU before caching/serializing.
            _move_to_device(session, "cpu")

            session.plantcv_inference_height = inference_height

            # Update cache and return session_id
            session_id = str(uuid.uuid4())
            self.session_cache.set(session_id, session)
            t_serialize = time.time() - t_start - t_decode - t_pre - t_model - t_post

            results = {"session_id": session_id, **processed_results}
            del model_outputs, inputs

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

            model = self._model_for(
                getattr(session, "plantcv_inference_height", DEFAULT_INFERENCE_HEIGHT)
            )
            inference_height = model.config.image_size
            self._update_config(model, request)

            inference_width = int(w * (inference_height / h))
            image_resized = cv2.resize(image_np, (inference_width, inference_height))
            inputs = self.processor(
                images=image_resized,
                size={"height": inference_height, "width": inference_height},
                device=self.device,
                return_tensors="pt",
            )
            t_pre = time.time() - t_start - t_decode - t_deserialize

            # Sessions stay on CPU between requests; bring them to GPU just for
            # the forward pass.
            t0 = time.time()
            prune_session(session, keep_frames=int(request.get("prune_keep_frames", 2)))
            t_prune = time.time() - t0
            t0 = time.time()
            _move_to_device(session, self.device)
            t_h2d = time.time() - t0

            t0 = time.time()
            model_outputs = model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.time() - t0
            t_model = time.time() - t_start - t_decode - t_deserialize - t_pre

            processed_results = process_outputs(
                self.processor,
                session,
                model_outputs,
                inference_size=(inference_height, inference_width),
                original_size=(h, w),
            )
            t_post = time.time() - t_start - t_decode - t_deserialize - t_pre - t_model

            # Move session back to CPU before caching so VRAM isn't held between
            # requests and disk-serialized sessions always have CPU tensors.
            t0 = time.time()
            _move_to_device(session, "cpu")
            t_d2h = time.time() - t0

            # Update cache
            t_cache_update = 0
            t_cache_start = time.time()
            self.session_cache.set(session_id, session)
            t_cache_update = time.time() - t_cache_start

            results = {"session_id": session_id, **processed_results}
            del model_outputs, inputs

        t_total = time.time() - t_start
        print(
            f"TIMING: propagate total {t_total:.4f}s (decode:{t_decode:.4f}, deserialize:{t_deserialize:.4f}, pre:{t_pre:.4f}, prune:{t_prune:.4f}, h2d:{t_h2d:.4f}, forward:{t_fwd:.4f}, post:{t_post:.4f}, d2h:{t_d2h:.4f}, cache_update:{t_cache_update:.4f})",
            flush=True,
        )

        return results

    def encode_response(self, result: dict):
        return result


if __name__ == "__main__":
    api = SAM3API()
    server = ls.LitServer(api, timeout=False)
    server.run(port=8805, generate_client_file=False)
