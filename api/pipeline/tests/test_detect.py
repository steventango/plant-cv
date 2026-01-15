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
    problem_images = [
        "E11Z09_2025-08-20T153000+0000_left.jpg",
        "E12Z06_2025-09-17T153000+0000_left.jpg",
        "E13Z01_2025-10-08T153000+0000_left.jpg",
    ]
    return [img for img in all_images if img.name in problem_images]


# Specific overrides for images where pots might be out of frame
SPECIFIC_POT_COUNTS = {
    "E14Z05_2025-11-12T163000+0000_left.jpg": 60,
    "E14Z08_2025-11-12T163000+0000_left.jpg": 59,
}
SPECIFIC_PLANT_COUNTS = {
    "E14Z05_2025-11-12T163000+0000_left.jpg": 60,
    "E14Z08_2025-11-12T163000+0000_left.jpg": 59,
}


def get_expected_count(filename, override):
    """Determine expected pot count based on experiment ID in filename."""
    if filename in override:
        return override[filename]
    match = re.search(r"E(\d+)", filename)
    if not match:
        return None

    exp_id = int(match.group(1))
    if 11 <= exp_id <= 13:
        return 18
    elif exp_id > 13:
        return 64
    return None


def check_image(image_path):
    """
    Process a single image and return an error message if the check fails,
    or None if it succeeds.
    """
    # Check if there is a specific override for this image
    expected_pot_count = get_expected_count(image_path.name, SPECIFIC_POT_COUNTS)
    expected_plant_count = get_expected_count(image_path.name, SPECIFIC_PLANT_COUNTS)

    if expected_pot_count is None:
        return None

    pot_count = 0
    plant_count = 0
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
        pot_count = len(pot_masks)

        assert pot_count == expected_pot_count, (
            f"Expected {expected_pot_count} pots, but detected {pot_count} for {image_path.name}"
        )

        plant_masks = detect_result.get("plant_masks", [])
        plant_count = len(plant_masks)

        assert plant_count == expected_plant_count, (
            f"Expected {expected_plant_count} plants, but detected {plant_count} for {image_path.name}"
        )

    except requests.exceptions.ConnectionError:
        return "Could not connect to pipeline service. Is it running?", None, None
    except Exception as e:
        return (
            f"Exception for {image_path.name}: {str(e)}",
            pot_count - expected_pot_count,
            plant_count - expected_plant_count,
        )

    return None, 0, 0


def test_pot_detection(reference_images):
    """
    Test SAM3 pot detection on all reference images.
    """

    if not reference_images:
        pytest.skip("No reference images available")

    errors = []
    successes = []

    pot_mae = 0
    plant_mae = 0

    for img in (pbar := tqdm(reference_images)):
        error, pot_diff, plant_diff = check_image(img)
        if error:
            errors.append(error)
            logger.error(error)
        else:
            successes.append(img.name)
        if pot_diff is not None:
            pot_mae += abs(pot_diff)
        if plant_diff is not None:
            plant_mae += abs(plant_diff)
        pbar.set_postfix(
            {
                "POT MAE": pot_mae,
                "PLANT MAE": plant_mae,
                "Successes": len(successes),
                "Errors": len(errors),
            }
        )

    if errors:
        pytest.fail("\n".join(errors))
