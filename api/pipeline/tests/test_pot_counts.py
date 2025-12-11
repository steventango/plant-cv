import re
from pathlib import Path

import pytest
import requests

from tests.utils import encode_image, decode_image, save_image

# Configuration
BASE_URL = "http://pipeline:8800"
BASE_URL = "http://pipeline:8800"
REFERENCE_IMAGES_DIR = Path(__file__).parent / "test_data" / "reference_images"
OUTPUT_DIR = Path("test_output")


def get_reference_images():
    """Yield all jpg images from reference directory."""
    if not REFERENCE_IMAGES_DIR.exists():
        return []
    return sorted(list(REFERENCE_IMAGES_DIR.glob("*.jpg")))


def get_expected_pot_count(filename):
    """Determine expected pot count based on experiment ID in filename."""
    match = re.search(r"E(\d+)", filename)
    if not match:
        return None

    exp_id = int(match.group(1))
    if 11 <= exp_id <= 13:
        return 18
    elif exp_id == 14:
        return 64
    return None


# Specific overrides for images where pots might be out of frame
SPECIFIC_COUNTS = {
    "E14Z05_2025-11-12T163000+0000_left.jpg": 61,
    "E14Z08_2025-11-12T163000+0000_left.jpg": 57,
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
            "text_prompt": "pot",
            "visualize": True,
        }

        resp = requests.post(f"{BASE_URL}/pot/detect", json=detect_payload)

        # Save visualization if available, regardless of status code
        try:
            result_json = resp.json()
            if "visualization" in result_json:
                vis_image = decode_image(result_json["visualization"])
                save_image(vis_image, OUTPUT_DIR / f"{image_path.stem}_detect.jpg")
        except Exception:
            pass  # Don't fail test if visualization saving fails

        if resp.status_code != 200:
            return f"Error for {image_path.name}: {resp.text}"
        resp.raise_for_status()
        detect_result = resp.json()

        boxes = detect_result["boxes"]
        detected_count = len(boxes)

        if detected_count != expected_count:
            return f"Expected {expected_count} pots, but detected {detected_count} for {image_path.name}"

    except requests.exceptions.ConnectionError:
        return "Could not connect to pipeline service. Is it running?"
    except Exception as e:
        return f"Exception for {image_path.name}: {str(e)}"

    return None


def test_pot_detection_count_parallel():
    """
    Test that the pot detect endpoint finds the correct number of pots
    for reference images, running checks in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    image_paths = get_reference_images()
    errors = []

    # Run in parallel to speed up testing
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_image = {
            executor.submit(check_image, img): img for img in image_paths
        }

        for future in as_completed(future_to_image):
            error = future.result()
            if error:
                errors.append(error)

    if errors:
        pytest.fail(f"Errors: {len(errors)}\n\n" + "\n".join(errors))
