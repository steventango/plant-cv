import base64
import io
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

# Configuration
BASE_URL = "http://pipeline:8800"
TEST_IMAGE_PATH = Path(
    "/data/online/E13/P1/Dirichlet1/alliance-zone01/images/2025-10-08T153000+0000_left.jpg"
)
OUTPUT_DIR = Path("/tmp/plant-cv/api/pipeline/test_output")


@pytest.fixture
def output_dir():
    """Fixture to create and return output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture
def test_image():
    """Fixture to check if test image exists and return path."""
    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")
    return TEST_IMAGE_PATH


def encode_image(image_path):
    """Load and encode image to base64."""
    image = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_image(image_data):
    """Decode base64 to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def decode_masks(masks_b64):
    """Decode base64 numpy array."""
    masks_bytes = io.BytesIO(base64.b64decode(masks_b64))
    return np.load(masks_bytes)


def save_image(image, path):
    """Save PIL Image to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image.save(path)


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "healthy"


def test_pipeline_flow(test_image, output_dir):
    """Run full pipeline flow on the test image."""

    output_dir = output_dir / test_image.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Detect
    print(f"\nTesting pot/detect on {test_image.name}")
    image_data = encode_image(test_image)
    detect_payload = {
        "image_data": image_data,
        "text_prompt": "pot",
        "visualize": True,
    }
    resp = requests.post(f"{BASE_URL}/pot/detect", json=detect_payload)
    assert resp.status_code == 200, resp.text
    detect_result = resp.json()

    boxes = detect_result["boxes"]
    assert len(boxes) > 0, "Should detect at least one pot"

    if "visualization" in detect_result:
        vis_image = decode_image(detect_result["visualization"])
        save_image(vis_image, output_dir / f"{test_image.stem}_detect.jpg")

    # 2. Segment
    print(f"Testing pot/segment on {test_image.name}")
    segment_payload = {
        "image_data": image_data,
        "boxes": boxes,
        "visualize": True,
    }
    resp = requests.post(f"{BASE_URL}/pot/segment", json=segment_payload)
    assert resp.status_code == 200
    segment_result = resp.json()

    masks_b64 = segment_result["masks"]
    masks = decode_masks(masks_b64)
    assert masks.shape[0] == len(boxes)

    if "visualization" in segment_result:
        vis_image = decode_image(segment_result["visualization"])
        save_image(vis_image, output_dir / f"{test_image.stem}_segment.jpg")

    # 3. Quad
    print(f"Testing pot/quad on {test_image.name}")
    quad_payload = {"masks": masks_b64, "visualize": True, "image_data": image_data}
    resp = requests.post(f"{BASE_URL}/pot/quad", json=quad_payload)
    assert resp.status_code == 200
    quad_result = resp.json()

    quads = quad_result["quadrilaterals"]
    assert len(quads) == len(boxes)

    if "visualization" in quad_result:
        vis_image = decode_image(quad_result["visualization"])
        save_image(vis_image, output_dir / f"{test_image.stem}_quad.jpg")

    # 4. Warp
    print(f"Testing pot/warp on {test_image.name}")
    warp_payload = {
        "image_data": image_data,
        "quadrilaterals": quads,
        "margin": 0.25,
        "output_size": 256,
    }
    resp = requests.post(f"{BASE_URL}/pot/warp", json=warp_payload)
    assert resp.status_code == 200
    warp_result = resp.json()

    warped_images = warp_result["warped_images"]
    assert len(warped_images) == len(quads)

    # 5. Plant Detect and Segment
    print("Testing plant/detect and plant/segment on warped images")
    for i, warped in enumerate(warped_images):
        if not warped:
            continue

        plant_dir = output_dir / f"plant_{i:02d}"
        plant_dir.mkdir(parents=True, exist_ok=True)

        # 5a. Detect
        plant_detect_payload = {
            "image_data": warped,
        }
        resp = requests.post(f"{BASE_URL}/plant/detect", json=plant_detect_payload)
        assert resp.status_code == 200
        detect_result = resp.json()

        boxes = detect_result["boxes"]

        if len(boxes) > 0:
            # 5b. Segment
            plant_segment_payload = {
                "image_data": warped,
                "boxes": boxes,
                "confidences": detect_result["confidences"],
            }
            resp = requests.post(
                f"{BASE_URL}/plant/segment", json=plant_segment_payload
            )
            assert resp.status_code == 200
            segment_result = resp.json()
            stats = None

            if segment_result["success"]:
                assert "mask" in segment_result

                # 6. Plant Stats
                print("Testing plant/stats on first warped image")
                stats_payload = {
                    "warped_image": warped,
                    "mask": segment_result["mask"],
                    "pot_size_mm": 60.0,
                    "margin": 0.25,
                }
                resp = requests.post(f"{BASE_URL}/plant/stats", json=stats_payload)
                assert resp.status_code == 200, resp.text
                stats_result = resp.json()

                assert "stats" in stats_result
                stats = stats_result["stats"]
                assert "area" in stats
                assert "width" in stats
                assert "height" in stats
            else:
                print(f"Plant segmentation failed: {segment_result.get('note')}")
            # Visualize
            visualize_payload = {
                "image_data": warped,
                "boxes": boxes,
                "confidences": detect_result["confidences"],
                "masks": segment_result.get("masks"),
                "stats": stats is not None,
                "selected_index": segment_result.get("selected_index"),
                "mask_scores": segment_result.get("mask_scores"),
                "combined_scores": segment_result.get("combined_scores"),
                "pot_size_mm": 60.0,
                "margin": 0.25,
            }
            resp = requests.post(f"{BASE_URL}/plant/visualize", json=visualize_payload)
            assert resp.status_code == 200
            visualize_result = resp.json()

            if "visualization" in visualize_result:
                vis_image = decode_image(visualize_result["visualization"])
                save_image(vis_image, plant_dir / f"{i:02d}_visualize.jpg")
        else:
            print("No plants detected (expected if crop is empty)")
