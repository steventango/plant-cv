"""Session cache + durable persistence for the native SAM 3.1 online tracker.

A "session" here is the native `inference_state` dict produced by
`Sam3MultiplexTrackingWithInteractivity.init_state` and driven one frame at a time
(see app/main.py). The tracking MEMORY lives in `state["sam2_inference_states"]`
(per-bucket `output_dict` of plain tensors) plus `tracker_metadata` and counters.

Two facts drive the (de)serialization below, both validated on real GPU runs:

1. Never move the state with sam3's `recursive_to` — it rebuilds containers via
   `type(d)()` and silently strips `defaultdict.default_factory` (e.g.
   `tracker_metadata["obj_id_to_sam2_score_frame_wise"]`), which then KeyErrors on the
   next frame. We move tensors with the in-place `_move_to_device` (preserves container
   types) and rely on `torch.load(map_location=...)` for device placement.

2. On reload the by-reference link between `state["feature_cache"]` and each
   `sam2_inference_states[*]["cached_features"]` must be re-established (pickling splits
   them into separate dicts; the tracker then misses its per-frame feature lookup and
   falls back to an absent `images` key). See `_relink_after_load`.

Unpruned state is ~1.45 GB and OOMs within ~10 frames, so we prune every frame to the
cond frames + the last `KEEP_NONCOND_FRAMES` non-cond frames per bucket (the native
memory attention only looks back `num_maskmem-1 = 6` non-cond frames).
"""

import gc
import glob
import io
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import lz4.frame
import torch

from utils import _move_to_device

# Keep cond frames + this many recent non-cond frames per bucket. The model attends to
# at most num_maskmem-1 = 6 non-cond frames, so 6 preserves tracking quality; lower
# (e.g. 2-3) shrinks the persisted blob further at some quality cost.
KEEP_NONCOND_FRAMES = 6


def prune_session(inference_state: dict, keep_frames: int = KEEP_NONCOND_FRAMES) -> None:
    """Drop old non-cond frame memory and transient feature caches to bound size/VRAM."""
    for sub in inference_state.get("sam2_inference_states", []):
        output_dict = sub.get("output_dict", {})
        non_cond = output_dict.get("non_cond_frame_outputs", {})
        if len(non_cond) > keep_frames:
            for idx in sorted(non_cond)[:-keep_frames]:
                del non_cond[idx]
        # output_dict_per_obj holds views into the same per-frame dicts; trim to match.
        for obj_dict in sub.get("output_dict_per_obj", {}).values():
            obj_non_cond = obj_dict.get("non_cond_frame_outputs", {})
            if len(obj_non_cond) > keep_frames:
                for idx in sorted(obj_non_cond)[:-keep_frames]:
                    del obj_non_cond[idx]
        # Transient backbone features are recomputed per frame; never persist them.
        for cache_key in ("cached_features", "feature_cache"):
            cache = sub.get(cache_key)
            if isinstance(cache, dict):
                cache.clear()
    top_cache = inference_state.get("feature_cache")
    if isinstance(top_cache, dict):
        top_cache.clear()


def _trim_image_buffer(inference_state: dict) -> None:
    """Keep only the most recent raw frame; past pixels are never re-read.

    The detector indexes `img_batch.tensors[frame_idx]` by absolute index, so on reload
    we pad the buffer back to `num_frames` rows (see `_relink_after_load`). Past rows are
    never read when grounding runs one frame at a time (use_batched_grounding=False).
    """
    input_batch = inference_state.get("input_batch")
    if input_batch is None:
        return
    tensors = input_batch.img_batch.tensors
    if tensors is not None and tensors.shape[0] > 1:
        input_batch.img_batch.tensors = tensors[-1:].clone()


def _relink_after_load(inference_state: dict, device: str) -> None:
    """Restore by-reference cache link, device fields, and the image-buffer length."""
    inference_state["device"] = torch.device(device)
    fresh_cache: dict = {}
    inference_state["feature_cache"] = fresh_cache
    for sub in inference_state.get("sam2_inference_states", []):
        sub["cached_features"] = fresh_cache
        sub["device"] = torch.device(device)
        if "storage_device" in sub:
            sub["storage_device"] = torch.device(device)

    # Pad img_batch.tensors back to num_frames so absolute frame_idx indexing stays valid.
    input_batch = inference_state.get("input_batch")
    num_frames = inference_state.get("num_frames", 0)
    if input_batch is not None and num_frames:
        tensors = input_batch.img_batch.tensors
        if tensors is not None and tensors.shape[0] < num_frames:
            pad = tensors[-1:].repeat(num_frames - tensors.shape[0], 1, 1, 1)
            input_batch.img_batch.tensors = torch.cat([pad, tensors], dim=0)


def serialize_state(inference_state: dict) -> bytes:
    """Prune, drop the raw image buffer, and lz4-compress the inference state."""
    t0 = time.time()
    prune_session(inference_state)
    _trim_image_buffer(inference_state)
    # Move tensors to CPU in place (preserves defaultdicts, unlike sam3.recursive_to).
    _move_to_device(inference_state, "cpu")

    buffer = io.BytesIO()
    torch.save(inference_state, buffer)
    raw = buffer.getvalue()
    compressed = lz4.frame.compress(raw)
    print(
        f"DEBUG: serialize raw={len(raw) / 1e6:.2f}MB comp={len(compressed) / 1e6:.2f}MB "
        f"in {time.time() - t0:.3f}s",
        flush=True,
    )
    return compressed


def deserialize_state(state_bytes: bytes, device: str) -> dict:
    """Decompress, load onto `device`, and re-link references for continued tracking."""
    raw = lz4.frame.decompress(state_bytes)
    inference_state = torch.load(
        io.BytesIO(raw), map_location=device, weights_only=False
    )
    _relink_after_load(inference_state, device)
    return inference_state


class PersistenceManager:
    def __init__(self, persistence_dir: str, max_size: int = 24):
        self.persistence_dir = persistence_dir
        self.max_size = max_size
        os.makedirs(self.persistence_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.saving_sessions = set()
        self.lock = threading.Lock()

    def save_session_async(self, session_id: str, session: dict):
        with self.lock:
            if session_id in self.saving_sessions:
                return
            self.saving_sessions.add(session_id)
        self.executor.submit(self._write_to_disk, session_id, session)

    def _write_to_disk(self, session_id: str, session: dict):
        try:
            compressed_bytes = serialize_state(session)
            temp_path = os.path.join(self.persistence_dir, f"{session_id}.tmp")
            final_path = os.path.join(self.persistence_dir, f"{session_id}.bin")
            with open(temp_path, "wb") as f:
                f.write(compressed_bytes)
                os.fsync(f.fileno())
            os.rename(temp_path, final_path)
            print(f"DEBUG: Persisted session {session_id} to disk", flush=True)
            self._enforce_limit()
        except Exception as e:
            print(f"ERROR: Failed to persist session {session_id}: {e}", flush=True)
        finally:
            with self.lock:
                self.saving_sessions.discard(session_id)

    def _enforce_limit(self):
        """Keep at most max_size persisted sessions (LRU by mtime)."""
        try:
            files = glob.glob(os.path.join(self.persistence_dir, "*.bin"))
            if len(files) <= self.max_size:
                return
            files.sort(key=os.path.getmtime)
            for fpath in files[: len(files) - self.max_size]:
                try:
                    os.remove(fpath)
                    print(
                        f"DEBUG: Pruned old session file {os.path.basename(fpath)}",
                        flush=True,
                    )
                except OSError as e:
                    print(f"ERROR: Failed to prune {fpath}: {e}", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to enforce disk limit: {e}", flush=True)

    def load_session(self, session_id: str, device: str) -> dict | None:
        path = os.path.join(self.persistence_dir, f"{session_id}.bin")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = f.read()
            return deserialize_state(data, device)
        except Exception as e:
            print(f"ERROR: Failed to load session {session_id}: {e}", flush=True)
            return None


class SessionCache:
    """In-memory LRU cache of native inference states with write-through persistence."""

    def __init__(
        self, max_size: int, persistence_manager: PersistenceManager, device: str
    ):
        self.cache = OrderedDict()
        self.access_times = {}
        self.persisted_sessions = set()
        self.max_size = max_size
        self.persistence = persistence_manager
        self.device = device
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def get(self, session_id: str) -> dict | None:
        self.access_times[session_id] = time.time()
        if session_id in self.cache:
            self.cache.move_to_end(session_id)
            return self.cache[session_id]
        session = self.persistence.load_session(session_id, self.device)
        if session is not None:
            self.cache[session_id] = session
            self.cache.move_to_end(session_id)
            self.persisted_sessions.discard(session_id)
            return session
        return None

    def set(self, session_id: str, session: dict):
        self.access_times[session_id] = time.time()
        self.persisted_sessions.discard(session_id)
        if session_id in self.cache:
            self.cache.move_to_end(session_id)
        self.cache[session_id] = session
        # Write-through so a restart (or another worker) can reload the session.
        self.persistence.save_session_async(session_id, session)
        if len(self.cache) > self.max_size:
            old_id, _ = self.cache.popitem(last=False)
            self.access_times.pop(old_id, None)
            self.persisted_sessions.discard(old_id)
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _cleanup_loop(self):
        while self.running:
            time.sleep(60)
            now = time.time()
            for session_id, last_access in list(self.access_times.items()):
                if now - last_access > 300:
                    if (
                        session_id in self.cache
                        and session_id not in self.persisted_sessions
                    ):
                        print(
                            f"DEBUG: Session {session_id} inactive; persisting to disk.",
                            flush=True,
                        )
                        self.persistence.save_session_async(
                            session_id, self.cache[session_id]
                        )
                        self.persisted_sessions.add(session_id)
