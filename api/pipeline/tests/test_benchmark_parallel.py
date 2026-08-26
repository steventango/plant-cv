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
NUM_CHAMBERS = 11


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


def load_frames():
    """Return sorted E14Z1 tracking frames, or None if unavailable."""
    dataset_dir = TRACKING_FRAMES_DIR / "E14Z1"
    if not dataset_dir.is_dir():
        return None
    frames = sorted(dataset_dir.glob("*.jpg"))
    if len(frames) < 2:
        return None
    return frames


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
        frames = load_frames()
        if not frames:
            pytest.skip("Tracking frames not available for benchmark")

        encoded = [encode_image_from_path(p) for p in frames]

        # Warm-up detect to load models / sessions.
        warmup_resp = post_detect(encoded[0])
        assert warmup_resp.status_code == 200, warmup_resp.text
        assert warmup_resp.json().get("state") is not None, (
            "Warm-up detect missing state"
        )

        # Benchmark parallel detect calls on the first frame.
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_CHAMBERS) as executor:
            detect_futures = [
                executor.submit(post_detect, encoded[0], True)
                for _ in range(NUM_CHAMBERS)
            ]
            detect_results = [f.result() for f in detect_futures]
        detect_elapsed = time.perf_counter() - start

        detect_profiles = []
        chamber_states = []
        for idx, resp in enumerate(detect_results):
            assert resp.status_code == 200, f"Detect {idx} failed: {resp.text}"
            result = resp.json()
            assert result.get("state") is not None, f"Detect {idx} missing state"
            chamber_states.append(result["state"])
            if result.get("profile"):
                detect_profiles.append(result["profile"])

        print(f"Parallel detect ({frames[0].name}): {detect_elapsed:.2f}s")
        print("Target (eventual): both <= 30s")
        if detect_profiles:
            detect_summary = summarize_profiles(detect_profiles)
            print("Detect profile (avg, seconds):")
            for key in sorted(detect_summary.keys()):
                print(f"  {key}: {detect_summary[key]:.3f}")

        # Each later frame: chambers propagate in parallel, then keep their states.
        all_propagate_profiles = []
        propagate_walls = []
        for step, frame in enumerate(frames[1:], start=1):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=NUM_CHAMBERS) as executor:
                prop_futures = [
                    executor.submit(
                        post_propagate, encoded[step], chamber_states[i], True
                    )
                    for i in range(NUM_CHAMBERS)
                ]
                prop_results = [f.result() for f in prop_futures]
            elapsed = time.perf_counter() - start
            propagate_walls.append(elapsed)

            next_states = []
            step_profiles = []
            for idx, resp in enumerate(prop_results):
                assert resp.status_code == 200, (
                    f"Propagate frame {step} chamber {idx} failed: {resp.text}"
                )
                result = resp.json()
                assert result.get("state") is not None, (
                    f"Propagate frame {step} chamber {idx} missing state"
                )
                next_states.append(result["state"])
                if result.get("profile"):
                    step_profiles.append(result["profile"])
                    all_propagate_profiles.append(result["profile"])
            chamber_states = next_states

            plant = None
            if step_profiles:
                summary = summarize_profiles(step_profiles)
                plant = summary.get("sam3_plant_propagate")
            plant_s = f" plant={plant:.2f}s" if plant is not None else ""
            print(
                f"Parallel propagate (12 calls, step {step}/{len(frames) - 1} "
                f"{frame.name}): {elapsed:.2f}s{plant_s}"
            )

        print(
            f"Propagate walls over {len(propagate_walls)} E14Z1 frames: "
            f"min={min(propagate_walls):.2f}s mean={sum(propagate_walls) / len(propagate_walls):.2f}s "
            f"max={max(propagate_walls):.2f}s"
        )
        if all_propagate_profiles:
            propagate_summary = summarize_profiles(all_propagate_profiles)
            print("Propagate profile (avg across all frames, seconds):")
            for key in sorted(propagate_summary.keys()):
                print(f"  {key}: {propagate_summary[key]:.3f}")
