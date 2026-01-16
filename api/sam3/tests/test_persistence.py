import unittest
import time
import shutil
import os
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies BEFORE importing caching
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["utils"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["litserve"] = MagicMock()
sys.modules["glob"] = MagicMock()

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
import caching


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_checkpoints"
        os.makedirs(self.test_dir, exist_ok=True)

        self.prune_patcher = patch("caching.prune_session")
        self.serialize_patcher = patch(
            "caching.serialize_state", return_value=b"dummy_bytes"
        )
        self.deserialize_patcher = patch(
            "caching.deserialize_state", return_value="dummy_session"
        )

        self.mock_prune = self.prune_patcher.start()
        self.mock_serialize = self.serialize_patcher.start()
        self.mock_deserialize = self.deserialize_patcher.start()

    def tearDown(self):
        self.prune_patcher.stop()
        self.serialize_patcher.stop()
        self.deserialize_patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_persistence_manager_atomic_write(self):
        pm = caching.PersistenceManager(self.test_dir)
        session = MagicMock()
        session_id = "sess_atomic"

        # Dispatch async
        pm.save_session_async(session_id, session)

        # Wait a bit for thread
        time.sleep(0.5)

        final_path = os.path.join(self.test_dir, f"{session_id}.bin")
        self.assertTrue(os.path.exists(final_path))

        # Verify content
        with open(final_path, "rb") as f:
            content = f.read()
            self.assertEqual(content, b"dummy_bytes")

    def test_session_cache_lazy_load(self):
        pm = caching.PersistenceManager(self.test_dir)
        # Prevent thread start
        with patch("threading.Thread"):
            cache = caching.SessionCache(
                max_size=5, persistence_manager=pm, device="cpu"
            )

        session_id = "cached_sess"

        # Simulate file on disk
        with open(os.path.join(self.test_dir, f"{session_id}.bin"), "wb") as f:
            f.write(b"stored_data")

        # Get should load from disk
        loaded = cache.get(session_id)

        # Since file exists, it should call load_session -> deserialize_state
        self.assertEqual(loaded, "dummy_session")
        self.assertIn(session_id, cache.cache)

    def test_cleanup_logic(self):
        pm = caching.PersistenceManager(self.test_dir)
        # Prevent thread start
        with patch("threading.Thread"):
            cache = caching.SessionCache(
                max_size=5, persistence_manager=pm, device="cpu"
            )

        session_id = "old_sess"
        cache.cache[session_id] = "session_obj"
        # Force old timestamp
        cache.access_times[session_id] = time.time() - 400

        # Run cleanup logic manually
        now = time.time()

        # Simulate one iteration of cleanup
        for sid, last in list(cache.access_times.items()):
            if now - last > 300:
                if sid in cache.cache and sid not in cache.persisted_sessions:
                    cache.persistence.save_session_async(sid, cache.cache[sid])
                    cache.persisted_sessions.add(sid)

        # Assertions
        # 1. Should NOT be evicted
        self.assertIn(session_id, cache.cache)
        # 2. Should be marked persisted
        self.assertIn(session_id, cache.persisted_sessions)
        # 3. Should have triggered save
        time.sleep(0.5)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, f"{session_id}.bin"))
        )

    def test_persistence_manager_enforce_limit(self):
        pm = caching.PersistenceManager(self.test_dir, max_size=2)

        # Mocks
        mock_glob = caching.glob.glob

        # Setup scenario: 4 files exist
        files = ["f1.bin", "f2.bin", "f3.bin", "f4.bin"]
        mock_glob.return_value = files

        # Mock os.path.getmtime to return ordered times
        # f1=10, f2=20, f3=30, f4=40. Oldest are f1, f2.
        def mock_getmtime(path):
            mapping = {"f1.bin": 10, "f2.bin": 20, "f3.bin": 30, "f4.bin": 40}
            return mapping.get(path, 0)

        with (
            patch("os.path.getmtime", side_effect=mock_getmtime),
            patch("os.remove") as mock_remove,
        ):
            pm._enforce_limit()

            # Assertions
            self.assertEqual(mock_remove.call_count, 2)
            calls = [c[0][0] for c in mock_remove.call_args_list]
            self.assertIn("f1.bin", calls)
            self.assertIn("f2.bin", calls)
            self.assertNotIn("f3.bin", calls)


if __name__ == "__main__":
    unittest.main()
