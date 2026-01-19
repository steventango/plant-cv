import numpy as np
from ..app.plant.visualize import visualize_annotation
from PIL import Image
from pathlib import Path

OUTPUT_DIR = Path("/tmp/plant-cv/api/pipeline/plant/test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_visualize_annotation_boxes():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10, 10, 50, 50]]

    annotated = visualize_annotation(image, boxes=boxes)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    Image.fromarray(annotated).save(OUTPUT_DIR / "test_visualize_annotation_boxes.png")


def test_visualize_annotation_mask():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    masks = np.zeros((1, 100, 100), dtype=np.uint8)
    masks[0, 20:40, 20:40] = 1

    annotated = visualize_annotation(image, masks=masks)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    Image.fromarray(annotated).save(OUTPUT_DIR / "test_visualize_annotation_mask.png")


def test_visualize_annotation_stats():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    masks = np.zeros((1, 100, 100), dtype=np.uint8)
    masks[0, 30:70, 30:70] = 1

    annotated = visualize_annotation(image, masks=masks, stats=True)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    Image.fromarray(annotated).save(OUTPUT_DIR / "test_visualize_annotation_stats.png")


def test_visualize_annotation_combined():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[30, 30, 70, 70]]
    masks = np.zeros((1, 100, 100), dtype=np.uint8)
    masks[0, 30:70, 30:70] = 1

    annotated = visualize_annotation(
        image, boxes=boxes, masks=masks, stats=True, selected_index=0
    )
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    Image.fromarray(annotated).save(
        OUTPUT_DIR / "test_visualize_annotation_combined.png"
    )
