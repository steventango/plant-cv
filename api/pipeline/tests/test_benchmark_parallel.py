import base64
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

# Path to test tracking frames (relative to this file)
TRACKING_FRAMES_DIR = Path(__file__).parent / "test_data" / "tracking_frames"


def get_service_url():
    """Try localhost then pipeline hostname."""
    for host in ["localhost", "pipeline"]:
        url = f"http://{host}:8800/pipeline"
        try:
            requests.get(f"http://{host}:8800/health", timeout=1)
            return url
        except Exception:
            continue
    return "http://localhost:8800/pipeline"


PIPELINE_BASE_URL = get_service_url()
DETECT_ENDPOINT = f"{PIPELINE_BASE_URL}/detect"
PROPAGATE_ENDPOINT = f"{PIPELINE_BASE_URL}/propagate"


def check_service_available():
    """Check if Pipeline service is available."""
    try:
        requests.get(
            f"{PIPELINE_BASE_URL}/health".replace("/pipeline/health", "/health"),
            timeout=2,
        )
        return True
    except Exception:
        return False


def encode_image_from_path(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_frame_pair():
    """Return a pair of frame paths for benchmark, or None if unavailable."""
    if not TRACKING_FRAMES_DIR.exists():
        return None

    for dataset_dir in sorted(TRACKING_FRAMES_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        frames = sorted(dataset_dir.glob("*.jpg"))
        if len(frames) >= 2:
            return frames[0], frames[1]

    return None


def post_detect(image_data, profile=False):
    """POST a detect request."""
    payload = {"image_data": image_data, "profile": profile}
    return requests.post(DETECT_ENDPOINT, json=payload, timeout=120)


def post_propagate(image_data, state, profile=False):
    """POST a propagate request."""
    payload = {"image_data": image_data, "state": state, "profile": profile}
    return requests.post(PROPAGATE_ENDPOINT, json=payload, timeout=180)


def summarize_profiles(profiles):
    """Flatten nested timing dicts and average them across responses."""
    sums = {}
    counts = {}

    def add(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else k
                add(key, v)
        elif isinstance(value, (int, float)):
            sums[prefix] = sums.get(prefix, 0.0) + value
            counts[prefix] = counts.get(prefix, 0) + 1

    for profile in profiles:
        add("", profile)

    return {k: sums[k] / counts[k] for k in sums}


class TestPipelineParallelBenchmark:
    """Benchmark parallel detect/propagate calls using the Pipeline API."""

    @pytest.fixture(autouse=True)
    def check_service(self):
        """Check if Pipeline service is running before each test."""
        if not check_service_available():
            pytest.skip(
                "Pipeline service not available. Run: docker compose up -d pipeline"
            )

    def test_parallel_detect_and_propagate(self):
        frame_pair = find_frame_pair()
        if not frame_pair:
            pytest.skip("Tracking frames not available for benchmark")

        first_frame, second_frame = frame_pair
        image_first = encode_image_from_path(first_frame)
        image_second = encode_image_from_path(second_frame)

        # Warm-up detect to get a state object for propagate.
        warmup_resp = post_detect(image_first)
        assert warmup_resp.status_code == 200, warmup_resp.text
        warmup_state = warmup_resp.json().get("state")
        assert warmup_state is not None, "Warm-up detect missing state"

        # Benchmark 12 parallel detect calls.
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=12) as executor:
            detect_futures = [
                executor.submit(post_detect, image_first, True) for _ in range(12)
            ]
            detect_results = [f.result() for f in detect_futures]
        detect_elapsed = time.perf_counter() - start

        detect_profiles = []
        detect_states = []
        for idx, resp in enumerate(detect_results):
            assert resp.status_code == 200, f"Detect {idx} failed: {resp.text}"
            result = resp.json()
            assert result.get("state") is not None, f"Detect {idx} missing state"
            detect_states.append(result["state"])
            if result.get("profile"):
                detect_profiles.append(result["profile"])

        # Benchmark 12 parallel propagate calls, each using its own independent state.
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=12) as executor:
            prop_futures = [
                executor.submit(post_propagate, image_second, detect_states[i], True)
                for i in range(12)
            ]
            prop_results = [f.result() for f in prop_futures]
        propagate_elapsed = time.perf_counter() - start

        propagate_profiles = []
        for idx, resp in enumerate(prop_results):
            assert resp.status_code == 200, f"Propagate {idx} failed: {resp.text}"
            result = resp.json()
            assert result.get("state") is not None, f"Propagate {idx} missing state"
            if result.get("profile"):
                propagate_profiles.append(result["profile"])

        print(
            f"Parallel detect (12 calls): {detect_elapsed:.2f}s | "
            f"Parallel propagate (12 calls): {propagate_elapsed:.2f}s"
        )
        print("Target (eventual): both <= 30s")

        if detect_profiles:
            detect_summary = summarize_profiles(detect_profiles)
            print("Detect profile (avg, seconds):")
            for key in sorted(detect_summary.keys()):
                print(f"  {key}: {detect_summary[key]:.3f}")

        if propagate_profiles:
            propagate_summary = summarize_profiles(propagate_profiles)
            print("Propagate profile (avg, seconds):")
            for key in sorted(propagate_summary.keys()):
                print(f"  {key}: {propagate_summary[key]:.3f}")
