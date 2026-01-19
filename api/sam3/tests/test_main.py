import pytest
import requests
import numpy as np
from PIL import Image
from transformers.video_utils import load_video
from ..app.utils import encode_pil_image


SAM3_ENDPOINT = "http://localhost:8805/predict"
VIDEO_URL = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
TEXT_PROMPT = "person"


def check_service_available():
    """Check if SAM3 service is available."""
    try:
        requests.post(
            SAM3_ENDPOINT,
            json={"endpoint": "health"},
            timeout=5,
        )
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


@pytest.fixture(scope="module")
def service_available():
    """Check if SAM3 service is running."""
    if not check_service_available():
        pytest.skip(
            "SAM3 service not available. "
            "Start with: docker compose up -d sam3, "
            "then run: docker compose exec sam3 uv run python -m pytest tests -v"
        )
    return True


@pytest.fixture(scope="module")
def video_frames():
    """Load video frames once for the module."""
    try:
        frames, _ = load_video(VIDEO_URL)
        return frames
    except ImportError as e:
        pytest.skip(f"Failed to import video loading dependencies: {e}")
    except Exception as e:
        pytest.skip(f"Failed to load video from {VIDEO_URL}: {e}")


class TestSAM3API:
    def test_detect(self, service_available, video_frames, benchmark):
        """Test detect endpoint with first frame of video."""
        if video_frames is None or len(video_frames) == 0:
            pytest.skip("No video frames loaded")

        # Use first frame
        frame = video_frames[0]
        # Transformers load_video might return PIL images or numpy arrays
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        print(f"Testing detect on frame size: {frame.size}")

        payload = {
            "endpoint": "detect",
            "image_data": encode_pil_image(frame),
            "text_prompt": TEXT_PROMPT,
        }

        def detect():
            return requests.post(SAM3_ENDPOINT, json=payload, timeout=30)

        response = benchmark(detect)
        assert response.status_code == 200, f"Detect failed: {response.text}"

        result = response.json()
        assert "session_id" in result, "Response missing session_id"
        assert "masks" in result, "Response missing masks"

        masks = result["masks"]
        print(f"Detected {len(masks)} objects for prompt '{TEXT_PROMPT}'")
        assert len(masks) > 0, "No objects detected"

        # Verify mask structure basic check
        mask = masks[0]
        assert "box" in mask
        assert "score" in mask

    def test_propagate(self, service_available, video_frames, benchmark):
        """Test propagate endpoint with second frame."""
        if video_frames is None or len(video_frames) < 2:
            pytest.skip("Not enough video frames")

        frame1 = video_frames[0]
        frame2 = video_frames[1]

        if isinstance(frame1, np.ndarray):
            frame1 = Image.fromarray(frame1)
        if isinstance(frame2, np.ndarray):
            frame2 = Image.fromarray(frame2)

        # 1. Detect on frame 1 to get session_id
        init_payload = {
            "endpoint": "detect",
            "image_data": encode_pil_image(frame1),
            "text_prompt": TEXT_PROMPT,
        }
        resp1 = requests.post(SAM3_ENDPOINT, json=init_payload, timeout=30)
        assert resp1.status_code == 200
        result1 = resp1.json()
        session_id = result1["session_id"]
        num_detected = len(result1["masks"])

        if num_detected == 0:
            pytest.skip("No objects detected in first frame, cannot test propagation")

        # 2. Propagate to frame 2
        print(f"Propagating {num_detected} objects to frame 2")
        prop_payload = {
            "endpoint": "propagate",
            "session_id": session_id,
            "image_data": encode_pil_image(frame2),
        }

        def propagate():
            return requests.post(SAM3_ENDPOINT, json=prop_payload, timeout=30)

        resp2 = benchmark(propagate)
        assert resp2.status_code == 200, f"Propagate failed: {resp2.text}"
        result2 = resp2.json()

        assert "session_id" in result2
        assert "masks" in result2

        propagated_masks = result2["masks"]
        print(f"Propagated {len(propagated_masks)} objects")

        # We expect to track at least something if detection worked
        assert len(propagated_masks) > 0
