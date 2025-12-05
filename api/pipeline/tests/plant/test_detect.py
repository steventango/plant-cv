
from pathlib import Path
from PIL import Image
import pytest

from app.plant.detect import (
    detect_plant,
    select_top_boxes,
    filter_boxes_by_area,
    filter_boxes_by_aspect_ratio,
    filter_boxes_by_centroid_in_pot,
)

TEST_DATA_DIR = Path(__file__).parents[1] / "test_data"


def test_empty_pot_edge_case():
    image_path = TEST_DATA_DIR / "empty_pot_edge_case_E14_P1_Z1_2025-12-01T163000.jpg"
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    image = Image.open(image_path)

    boxes, confidences, class_names = detect_plant(
        image,
        text_prompt="plant",
        threshold=0.21,
        text_threshold=0.0,
    )

    crop_shape = (image.height, image.width)

    if len(boxes) > 0:
        boxes, confidences, class_names, _ = filter_boxes_by_area(
            boxes, confidences, class_names, crop_shape, area_ratio_threshold=0.90
        )

    if len(boxes) > 0:
        boxes, confidences, class_names, _ = filter_boxes_by_aspect_ratio(
            boxes, confidences, class_names, crop_shape, aspect_ratio_threshold=2
        )

    if len(boxes) > 0:
        boxes, confidences, class_names, _ = filter_boxes_by_centroid_in_pot(
            boxes, confidences, class_names, crop_shape, margin=0.25
        )

    boxes, confidences, class_names = select_top_boxes(
        boxes, confidences, class_names, crop_shape, k=5
    )

    assert len(boxes) == 0, f"Expected 0 boxes, got {len(boxes)}"
