import io
import time
import zlib
import gc
import os
import threading
import glob
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import Sam3VideoInferenceSession
from utils import _move_to_device


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


def serialize_state(inference_session: Sam3VideoInferenceSession) -> bytes:
    """Serialize the entire inference session object to compressed bytes."""
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

    print(
        f"DEBUG: serialize raw={raw_size / 1e6:.2f}MB, comp={comp_size / 1e6:.2f}MB, timings: prune:{t_prune:.3f}s, save:{t_save:.3f}s, compress:{t_compress:.3f}s",
        flush=True,
    )
    return compressed


def deserialize_state(state_bytes: bytes, device: str) -> Sam3VideoInferenceSession:
    """Deserialize compressed state bytes and restore the inference session object."""
    decompressed_bytes = zlib.decompress(state_bytes)

    buffer = io.BytesIO(decompressed_bytes)
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


class PersistenceManager:
    def __init__(self, persistence_dir: str, max_size: int = 24):
        self.persistence_dir = persistence_dir
        self.max_size = max_size
        os.makedirs(self.persistence_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.saving_sessions = set()
        self.lock = threading.Lock()

    def save_session_async(self, session_id: str, session: Sam3VideoInferenceSession):
        with self.lock:
            if session_id in self.saving_sessions:
                return
            self.saving_sessions.add(session_id)

        # Prune session in main thread (fast) before offloading
        prune_session(session)
        self.executor.submit(self._write_to_disk, session_id, session)

    def _write_to_disk(self, session_id: str, session: Sam3VideoInferenceSession):
        try:
            # Serialize
            compressed_bytes = serialize_state(session)

            # Atomic write
            temp_path = os.path.join(self.persistence_dir, f"{session_id}.tmp")
            final_path = os.path.join(self.persistence_dir, f"{session_id}.bin")

            with open(temp_path, "wb") as f:
                f.write(compressed_bytes)
                os.fsync(f.fileno())

            os.rename(temp_path, final_path)

            print(f"DEBUG: Persisted session {session_id} to disk", flush=True)

            # Enforce disk limit
            self._enforce_limit()

        except Exception as e:
            print(f"ERROR: Failed to persist session {session_id}: {e}", flush=True)
        finally:
            with self.lock:
                self.saving_sessions.discard(session_id)

    def _enforce_limit(self):
        """Ensure persistence directory does not exceed max_size files (LRU based on mtime)."""
        try:
            files = glob.glob(os.path.join(self.persistence_dir, "*.bin"))
            if len(files) <= self.max_size:
                return

            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)

            # Delete oldest
            to_delete = files[: len(files) - self.max_size]
            for fpath in to_delete:
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

    def load_session(self, session_id: str, device: str) -> Sam3VideoInferenceSession:
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
    """In-memory cache for SAM3 inference sessions with LRU and persistence."""

    def __init__(
        self, max_size: int, persistence_manager: PersistenceManager, device: str
    ):
        self.cache = OrderedDict()
        self.access_times = {}  # session_id -> timestamp
        self.persisted_sessions = set()  # session_id
        self.max_size = max_size
        self.persistence = persistence_manager
        self.device = device
        self.running = True

        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def get(self, session_id: str) -> Sam3VideoInferenceSession:
        self.access_times[session_id] = time.time()

        if session_id in self.cache:
            self.cache.move_to_end(session_id)
            return self.cache[session_id]

        # Try loading from disk
        session = self.persistence.load_session(session_id, self.device)
        if session:
            self.cache[session_id] = session
            self.cache.move_to_end(session_id)
            self.persisted_sessions.discard(
                session_id
            )  # Mark as loaded, no longer just persisted (assumed active/dirty)
            return session

        return None

    def set(self, session_id: str, session: Sam3VideoInferenceSession):
        self.access_times[session_id] = time.time()
        self.persisted_sessions.discard(session_id)  # Mark as dirty/not persisted

        if session_id in self.cache:
            self.cache.move_to_end(session_id)
        self.cache[session_id] = session

        if len(self.cache) > self.max_size:
            # Force persist and evict LRU
            oldan_id, old_session = self.cache.popitem(last=False)
            self.persistence.save_session_async(oldan_id, old_session)

            # Clean up memory
            if oldan_id in self.access_times:
                del self.access_times[oldan_id]
            self.persisted_sessions.discard(oldan_id)

            gc.collect()
            torch.cuda.empty_cache()

    def _cleanup_loop(self):
        """Background thread to persist inactive sessions."""
        while self.running:
            # Check every minute
            time.sleep(60)
            now = time.time()

            # Identify inactive sessions
            # We iterate over a copy of keys to avoid modification issues
            for session_id, last_access in list(self.access_times.items()):
                if now - last_access > 300:  # 5 minutes
                    if (
                        session_id in self.cache
                        and session_id not in self.persisted_sessions
                    ):
                        print(
                            f"DEBUG: Session {session_id} inactive for {now - last_access:.0f}s. Persisting to disk (keeping in RAM).",
                            flush=True,
                        )
                        session = self.cache[session_id]
                        self.persistence.save_session_async(session_id, session)
                        self.persisted_sessions.add(session_id)
