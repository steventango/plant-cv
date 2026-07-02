"""SAM 3.1 (native facebookresearch/sam3) online video-tracking service.

Stateless HTTP contract (unchanged from the SAM 3 / transformers version): a `detect`
seeds a tracking session from one frame + a text prompt and returns a `session_id`;
each `propagate` advances that session by ONE more frame. Object IDs stay stable across
frames so the pipeline's id-mapping keeps working.

Internally this drives the native multiplex tracker one frame at a time via
`model._run_single_frame_inference` (the public whole-video `start_session`/
`propagate_in_video` API can't ingest frames incrementally). Tracking memory lives in the
`inference_state` dict, cached resident and written through to disk for restart durability
(see caching.py). See the migration plan + memory note "sam31-online-tracking-technique".
"""

import gc
import os
import time
import uuid
from contextlib import nullcontext

import cv2
import litserve as ls
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_multiplex_video_predictor
from sam3.model.data_misc import BatchedPointer, FindStage, convert_my_tensors
from sam3.model.sam3_multiplex_tracking import recursive_to

from caching import PersistenceManager, SessionCache, prune_session, _trim_image_buffer
from utils import _move_to_device, decode_image, mask_to_contour

# Model attributes the pipeline may send that have native equivalents. Anything absent
# (high_conf_thresh, high_iou_thresh, recondition_on_trk_masks, low_res_mask_size, ...) is
# skipped via hasattr below rather than assumed. score_threshold_detection maps to the
# detector's grounding threshold; the rest are runtime tracker knobs.
_INT_KNOBS = (
    "recondition_every_nth_frame",
    "init_trk_keep_alive",
    "max_trk_keep_alive",
    "min_trk_keep_alive",
    "fill_hole_area",
    "hotstart_delay",
)
_FLOAT_KNOBS = (
    "score_threshold_detection",
    "new_det_thresh",
    "suppress_overlapping_based_on_recent_occlusion_threshold",
    # The pipeline's SAM3-era plant tuning mostly transfers to SAM3.1 (per-knob ablation):
    #   det_nms_thresh=0.01      -> keeps more detections, BOOSTS recall (60->75), stable
    #   trk_assoc_iou_thresh=0.0 -> neutral/safe (60->67, == defaults)
    # so we honor those. Only assoc_iou_thresh is handled specially below.
    "det_nms_thresh",
    "trk_assoc_iou_thresh",
)
# assoc_iou_thresh has different semantics in SAM3.1's association: the pipeline's SAM3-era
# value 0.0 is degenerate and progressively collapses tracks (ablation: alone 60->57, and
# it poisons the combination 60->46). We FLOOR it to SAM3.1's default 0.1 so the old
# tuning is honored everywhere it works, without the track loss.
_ASSOC_IOU_FLOOR = 0.1


def _append_frame(model, state: dict, pil_image: Image.Image) -> int:
    """Append one new frame to an existing inference_state and return its absolute index.

    Mirrors `_construct_initial_input_batch`'s per-frame entry: the detector indexes
    `img_batch.tensors[frame_idx]`, so we grow the image tensor and every per-frame list,
    and bump num_frames (top-level + each bucket state). Past pixels are never re-read
    (grounding runs one frame at a time), so the buffer can be trimmed between requests.
    """
    from sam3.model.io_utils import load_resource_as_video_frames

    new_idx = state["num_frames"]
    imgs, _, _ = load_resource_as_video_frames(
        resource_path=[pil_image],
        image_size=model.image_size,
        offload_video_to_cpu=False,
        img_mean=model.image_mean,
        img_std=model.image_std,
    )
    input_batch = state["input_batch"]
    input_batch.img_batch.tensors = torch.cat(
        [input_batch.img_batch.tensors, imgs[0][None]], dim=0
    )

    dummy_ptrs = BatchedPointer(
        stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
    )
    stage = FindStage(
        img_ids=[new_idx],
        img_ids_np=np.array([new_idx]),
        text_ids=[0],
        input_boxes=[torch.zeros(258)],
        input_boxes_before_embed=[torch.empty(0, 4)],
        input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
        input_boxes_label=[torch.empty(0, dtype=torch.long)],
        input_points=[torch.empty(0, 257)],
        input_points_before_embed=[torch.empty(0, 3)],
        input_points_mask=[torch.empty(0)],
        ptrs=dummy_ptrs,
        ptrs_seg=dummy_ptrs,
        object_ids=[],
    )
    stage = recursive_to(convert_my_tensors(stage), state["device"], non_blocking=True)
    input_batch.find_inputs.append(stage)
    input_batch.find_targets.append(None)
    input_batch.find_metadatas.append(None)
    for key in (
        "previous_stages_out",
        "per_frame_raw_point_input",
        "per_frame_raw_box_input",
        "per_frame_visual_prompt",
        "per_frame_geometric_prompt",
    ):
        state[key].append(None)
    state["per_frame_cur_step"].append(0)
    state["num_frames"] += 1
    for sub in state["sam2_inference_states"]:
        if "num_frames" in sub:
            sub["num_frames"] += 1
    return new_idx


# Output masks are produced at the resolution of the image we hand to the tracker. The
# model resizes whatever we pass to image_size x image_size internally, so downsizing the
# input first is near-lossless for the model but keeps the *output* masks small. This is
# essential: the non-overlapping-constraint postprocess materializes several
# [num_objects, H, W] tensors, so full 5MP frames x ~64 objects OOMs the GPU. We track at
# this height and scale contours/boxes back to the true frame resolution (as the previous
# transformers-based service did).
_INFERENCE_HEIGHT = int(os.environ.get("SAM3_INFERENCE_HEIGHT", "1008"))


def _resize_for_inference(image_np: np.ndarray) -> tuple[Image.Image, float, float]:
    """Downsize to _INFERENCE_HEIGHT (preserving aspect); return PIL + scale-up factors."""
    h, w = image_np.shape[:2]
    if h <= _INFERENCE_HEIGHT:
        return Image.fromarray(image_np), 1.0, 1.0
    new_h = _INFERENCE_HEIGHT
    new_w = max(1, round(w * _INFERENCE_HEIGHT / h))
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized), w / new_w, h / new_h


def _extract_masks(
    post_out: dict, scale_x: float = 1.0, scale_y: float = 1.0
) -> list[dict]:
    """Convert native per-frame outputs into the pipeline's mask dicts.

    Native: out_obj_ids (int), out_probs (score), out_boxes_xywh (normalized xywh),
    out_binary_masks (bool at the tracking resolution). Contours/boxes are scaled by
    (scale_x, scale_y) back up to the true original frame resolution.
    """
    obj_ids = np.asarray(post_out.get("out_obj_ids", []))
    if len(obj_ids) == 0:
        return []
    probs = np.asarray(post_out["out_probs"])
    boxes = np.asarray(post_out["out_boxes_xywh"])
    masks = np.asarray(post_out["out_binary_masks"])

    results = []
    for i, obj_id in enumerate(obj_ids):
        mask = masks[i]
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        h, w = mask.shape[:2]
        contour = [[px * scale_x, py * scale_y] for px, py in mask_to_contour(mask.astype(bool))]
        x, y, bw, bh = (float(v) for v in boxes[i].tolist())
        box = [x * w * scale_x, y * h * scale_y, (x + bw) * w * scale_x, (y + bh) * h * scale_y]
        results.append(
            {
                "object_id": int(obj_id),
                "contour": contour,
                "box": box,
                "score": float(probs[i]),
            }
        )
    return results


class SAM3API(ls.LitAPI):
    """SAM 3.1 online video-tracking API (stateless HTTP, resident+durable sessions)."""

    def setup(self, device: str):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=os.environ.get("SAM3_CKPT_PATH"),  # None -> download from HF
            max_num_objects=int(os.environ.get("SAM3_MAX_OBJECTS", "100")),
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
        )
        self.model = self.predictor.model
        # Online-friendly settings: emit results immediately (no hotstart/confirmation
        # buffering) and ground one frame at a time so only the current frame is indexed.
        self.model.hotstart_delay = 0
        self.model.masklet_confirmation_enable = False
        self.model.postprocess_batch_size = 1
        self.model.use_batched_grounding = False

        self.persistence = PersistenceManager(persistence_dir="checkpoints")
        self.session_cache = SessionCache(
            max_size=24, persistence_manager=self.persistence, device=device
        )

    def decode_request(self, request: dict):
        return request

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _update_config(self, request: dict):
        for name in _INT_KNOBS:
            if name in request and hasattr(self.model, name):
                setattr(self.model, name, int(request[name]))
        for name in _FLOAT_KNOBS:
            if name in request and hasattr(self.model, name):
                setattr(self.model, name, float(request[name]))
        # assoc_iou_thresh: honor the request but floor it (0.0 is degenerate in SAM3.1).
        if "assoc_iou_thresh" in request and hasattr(self.model, "assoc_iou_thresh"):
            self.model.assoc_iou_thresh = max(
                float(request["assoc_iou_thresh"]), _ASSOC_IOU_FLOOR
            )

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
        return {"error": f"Unknown endpoint: {endpoint}"}

    def _prepare_for_inference(self, state: dict):
        """Move a cached/loaded state onto the GPU and restore the image-buffer length."""
        _move_to_device(state, self.device)
        from caching import _relink_after_load

        _relink_after_load(state, self.device)

    def _finalize(self, session_id: str, state: dict, masks: list, t_start: float):
        """Prune + offload to CPU (frees VRAM), cache, and write through to disk."""
        prune_session(state)
        _trim_image_buffer(state)
        _move_to_device(state, "cpu")
        self.session_cache.set(session_id, state)
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(
            f"TIMING: total {time.time() - t_start:.3f}s, objects={len(masks)}",
            flush=True,
        )

    def _detect(self, request: dict):
        t_start = time.time()
        image_data = request.get("image_data")
        text_prompt = request.get("text_prompt")
        if not text_prompt:
            return {"error": "text_prompt is required for detect"}

        self._update_config(request)
        image_np = decode_image(image_data)  # RGB, true resolution
        pil, scale_x, scale_y = _resize_for_inference(image_np)

        with torch.inference_mode(), self._autocast():
            state = self.model.init_state(resource_path=[pil])
            _, post_out = self.model.add_prompt(state, frame_idx=0, text_str=text_prompt)
            masks = _extract_masks(post_out, scale_x, scale_y)

        session_id = str(uuid.uuid4())
        self._finalize(session_id, state, masks, t_start)
        return {"session_id": session_id, "masks": masks}

    def _propagate(self, request: dict):
        t_start = time.time()
        image_data = request.get("image_data")
        session_id = request.get("session_id")
        if not session_id:
            return {"error": "session_id is required for propagate"}

        self._update_config(request)
        state = self.session_cache.get(session_id)
        if state is None:
            return {"error": "Session expired or not found in cache"}

        image_np = decode_image(image_data)
        pil, scale_x, scale_y = _resize_for_inference(image_np)

        with torch.inference_mode(), self._autocast():
            self._prepare_for_inference(state)
            frame_idx = _append_frame(self.model, state, pil)
            raw = self.model._run_single_frame_inference(
                state, frame_idx, reverse=False
            )
            post_out = self.model._postprocess_output(
                state, raw, suppressed_obj_ids=raw.get("suppressed_obj_ids")
            )
            masks = _extract_masks(post_out, scale_x, scale_y)
            del raw

        self._finalize(session_id, state, masks, t_start)
        return {"session_id": session_id, "masks": masks}

    def encode_response(self, result: dict):
        return result


if __name__ == "__main__":
    api = SAM3API()
    server = ls.LitServer(api, timeout=False)
    server.run(port=8805, generate_client_file=False)
