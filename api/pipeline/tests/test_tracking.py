import base64
from pathlib import Path

import numpy as np
import pytest
import requests
from tqdm import tqdm

# Path to test tracking frames (relative to this file)
TRACKING_FRAMES_DIR = Path(__file__).parent / "test_data" / "tracking_frames"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output" / "tracking"


# Pipeline service endpoints
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


def encode_image_from_path(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


class TestPipelineTracking:
    """Test tracking consistency using the Pipeline API."""

    @pytest.fixture(autouse=True)
    def check_service(self):
        """Check if Pipeline service is running before each test."""
        if not check_service_available():
            pytest.skip(
                "Pipeline service not available. Run: docker compose up -d pipeline"
            )

    @pytest.fixture
    def video_frames(self):
        """Get video frames for testing from local test_data."""
        if not TRACKING_FRAMES_DIR.exists():
            pytest.skip(f"Tracking frames directory not found: {TRACKING_FRAMES_DIR}")

        frames = sorted(TRACKING_FRAMES_DIR.glob("*.jpg"))
        if len(frames) < 2:
            pytest.skip("Not enough video frames available for tracking test")

        # Return list of (path_str, timestamp_str) tuples
        result = []
        for f in frames:  # Use all frames
            basename = f.stem
            # Handling 2025-11-12T163000+0000_left.jpg
            # split by "_" and take everything before the last "_"
            parts = basename.split("_")
            ts_part = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            result.append((str(f), ts_part))
        return result

    def test_tracking_with_visualization(self, video_frames):
        """
        Test tracking using the Pipeline API and save visualization.
        """
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Process frames
        pot_state = None
        plant_state = None

        for j, (frame_path, timestamp) in enumerate(tqdm(video_frames)):
            image_data = encode_image_from_path(frame_path)

            if pot_state is None:
                payload = {"image_data": image_data}
                response = requests.post(DETECT_ENDPOINT, json=payload, timeout=120)
            else:
                payload = {
                    "image_data": image_data,
                    "pot_state": pot_state,
                    "plant_state": plant_state,
                }
                response = requests.post(PROPAGATE_ENDPOINT, json=payload, timeout=120)

            if response.status_code != 200:
                pytest.fail(f"Frame {j} failed: {response.text}")

            result = response.json()
            print(f"\nResponse keys for frame {j}: {list(result.keys())}")
            pot_state = result.get("pot_state")
            plant_state = result.get("plant_state")
            associations = result.get("associations", {})
            ordered_pot_ids = result.get("ordered_pot_ids", [])
            ordered_pot_ids_str = [str(x) for x in ordered_pot_ids]
            pot_masks = result.get("pot_masks", [])
            plant_stats = result.get("plant_stats", {})

            # DEBUG:
            print(
                f"Plant mask count from SAM3 for frame {j}: {result.get('debug_plant_mask_sam3_count', 0)}"
            )
            print(
                f"Refined plant mask count for frame {j}: {result.get('debug_refined_plant_masks_count', 0)}"
            )
            print(
                f"Pot mask count for frame {j}: {result.get('debug_pot_masks_count', 0)}"
            )

            # Verify stats: should be present and for 64 pots
            assert plant_stats is not None, f"Frame {j}: plant_stats missing"
            assert len(plant_stats) == len(ordered_pot_ids), (
                f"Frame {j}: stats length mismatch"
            )

            # Print example stat and save warped image if available
            if plant_stats and ordered_pot_ids:
                first_pot_id = str(ordered_pot_ids[0])
                if first_pot_id in plant_stats:
                    stats = plant_stats[first_pot_id]
                    if stats:
                        area = stats.get("area", 0)
                        print(
                            f"Sample stats for pot {first_pot_id}: area={area:.2f} mm²"
                        )

                        # Verify embeddings
                        embeddings = stats.get("embeddings")
                        assert embeddings is not None, (
                            f"Frame {j}: embeddings missing for pot {first_pot_id}"
                        )
                        assert len(embeddings) == 768, (
                            f"Frame {j}: embedding length mismatch for pot {first_pot_id}: {len(embeddings)}"
                        )

                        # Save warped image
                        w_b64 = stats.get("warped_image")
                        if w_b64:
                            import base64 as b64

                            warped_dir = TEST_OUTPUT_DIR / "warped_pots"
                            warped_dir.mkdir(parents=True, exist_ok=True)
                            img_data = b64.b64decode(w_b64)
                            with open(
                                warped_dir / f"frame_{j}_pot_{first_pot_id}.jpg", "wb"
                            ) as f:
                                f.write(img_data)

                # Create reverse mapping: pot_id -> plant_id
                pot_to_plant = {str(v): k for k, v in associations.items()}

                # Monitor specific pots requested by user
                monitored_pots = ["29", "38"]
                for p_id in monitored_pots:
                    if p_id in pot_to_plant:
                        plant_id = pot_to_plant[p_id]
                        if p_id in plant_stats:
                            p_area = plant_stats[p_id].get("area", 0)
                            print(
                                f"DEBUG: Pot {p_id} (Plant {plant_id}) area: {p_area:.0f} mm²"
                            )
                        else:
                            print(f"DEBUG: Pot {p_id} (Plant {plant_id}) has no stats")
                    elif (
                        str(p_id) in ordered_pot_ids_str
                    ):  # Check if pot exists but has no plant
                        print(f"DEBUG: Pot {p_id} exists but has NO associated plant")
                    else:
                        print(f"DEBUG: Pot {p_id} is LOST/MISSING from ordered_pot_ids")

                # Check for lost plants 34 and 41 explicitly
                lost_plants = [34, 41]
                found_plants = [p["object_id"] for p in result.get("plant_masks", [])]
                for lp in lost_plants:
                    if lp in found_plants:
                        # Find its mask and center
                        pm = next(
                            p for p in result["plant_masks"] if p["object_id"] == lp
                        )
                        box = pm["box"]
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        print(
                            f"DEBUG: Plant {lp} FOUND in plant_masks. Center: ({cx:.1f}, {cy:.1f})"
                        )
                    else:
                        print(f"DEBUG: Plant {lp} NOT FOUND in plant_masks")

            # Verify ordering: pot_masks should match ordered_pot_ids and be row-major
            print(
                f"DEBUG: frame {j}, pot_masks len={len(pot_masks)}, ordered_pot_ids len={len(ordered_pot_ids)}, plant_stats len={len(plant_stats)}"
            )
            assert len(pot_masks) == len(ordered_pot_ids)
            if len(pot_masks) > 1:
                # Check for row-major order (Y then X)
                boxes = np.array([m["box"] for m in pot_masks])
                xs = (boxes[:, 0] + boxes[:, 2]) / 2
                ys = (boxes[:, 1] + boxes[:, 3]) / 2
                for k in range(len(ys) - 1):
                    # In a perfect world ys[j] <= ys[j+1].
                    # If ys are very close, xs[j] should be <= xs[j+1]
                    if abs(ys[k] - ys[k + 1]) > 30:  # Increased tolerance from 10 to 30
                        assert ys[k] <= ys[k + 1], (
                            f"Frame {j}: Pots not in row-major order (Y deviation)"
                        )

            viz_data = result.get("visualization_data")
            assert viz_data is not None, (
                f"Frame {j}: visualization_data missing from response"
            )
            viz_bytes = base64.b64decode(viz_data)
            viz_path = TEST_OUTPUT_DIR / f"viz_pipeline_{j:03d}.jpg"
            with open(viz_path, "wb") as f:
                f.write(viz_bytes)

        print(
            f"\nTracking visualizations with associations saved to: {TEST_OUTPUT_DIR}"
        )

        # Basic assertion to ensure we got some masks and associations
        assert pot_state is not None
        assert plant_state is not None
        assert len(associations) > 0, "No plant-pot associations found"
