import re
from pathlib import Path

import pytest
import requests
from tests.utils import decode_image, encode_image, save_image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://pipeline:8800"
REFERENCE_IMAGES_DIR = Path(__file__).parent / "test_data" / "reference_images"
OUTPUT_DIR = Path(__file__).parent / "test_output"


@pytest.fixture
def reference_images():
    """Yield all jpg images from reference directory."""
    if not REFERENCE_IMAGES_DIR.exists():
        return []

    all_images = sorted(list(REFERENCE_IMAGES_DIR.glob("*.jpg")))
    return all_images


def get_expected_pot_count(filename):
    """Determine expected pot count based on experiment ID in filename."""
    match = re.search(r"E(\d+)", filename)
    if not match:
        return None

    exp_id = int(match.group(1))
    if 11 <= exp_id <= 13:
        return 18
    elif exp_id > 13:
        return 64
    return None


# Specific overrides for images where pots might be out of frame
SPECIFIC_COUNTS = {
    "E14Z05_2025-11-12T163000+0000_left.jpg": 60,
    "E14Z08_2025-11-12T163000+0000_left.jpg": 59,
}


def check_image(image_path):
    """
    Process a single image and return an error message if the check fails,
    or None if it succeeds.
    """
    # Check if there is a specific override for this image
    if image_path.name in SPECIFIC_COUNTS:
        expected_count = SPECIFIC_COUNTS[image_path.name]
    else:
        expected_count = get_expected_pot_count(image_path.name)

    if expected_count is None:
        return None

    try:
        image_data = encode_image(image_path)
        detect_payload = {
            "image_data": image_data,
        }

        resp = requests.post(
            f"{BASE_URL}/pipeline/detect", json=detect_payload, timeout=120
        )

        # Save visualization if available
        try:
            result_json = resp.json()
            if "visualization_data" in result_json:
                # We can't use save_image directly with bytes if it expects PIL,
                # but decode_image decodes base64 string to PIL.
                # visualization_data is base64 string.
                vis_image = decode_image(result_json["visualization_data"])
                save_image(vis_image, OUTPUT_DIR / f"{image_path.stem}_detect.jpg")
        except Exception:
            pass

        if resp.status_code != 200:
            return f"Error for {image_path.name}: {resp.text}", None
        resp.raise_for_status()
        detect_result = resp.json()

        pot_masks = detect_result.get("pot_masks", [])
        detected_count = len(pot_masks)

        assert detected_count == expected_count, (
            f"Expected {expected_count} pots, but detected {detected_count} for {image_path.name}"
        )

    except requests.exceptions.ConnectionError:
        return "Could not connect to pipeline service. Is it running?"
    except Exception as e:
        return (
            f"Exception for {image_path.name}: {str(e)}",
            detected_count - expected_count,
        )

    return None, 0


def test_pot_detection(reference_images):
    """
    Test SAM3 pot detection on all reference images.
    """

    if not reference_images:
        pytest.skip("No reference images available")

    errors = []
    successes = []

    mae = 0

    for img in (pbar := tqdm(reference_images)):
        error, diff = check_image(img)
        if error:
            errors.append(error)
            logger.error(error)
        else:
            successes.append(img.name)
        if diff is not None:
            mae += abs(diff)
        pbar.set_postfix(
            {"MAE": mae, "Successes": len(successes), "Errors": len(errors)}
        )

    if errors:
        pytest.fail("\n".join(errors))
