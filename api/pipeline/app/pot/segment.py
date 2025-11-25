import cv2
import numpy as np
import requests
from PIL import Image

from .utils import encode_image


def contours_to_masks(contours: list[list[list[int]]], image_size: tuple[int, int]):
    h, w = image_size
    masks = np.zeros((len(contours), h, w), dtype=np.uint8)
    for i, contour in enumerate(contours):
        if contour:
            contour_np = np.array(contour, dtype=np.int32)
            cv2.fillPoly(masks[i], [contour_np], 1)
    return masks


def segment_pot_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    multimask_output: bool = False,
    server_url: str = "http://segment-anything:8000/predict",
):
    """
    Call Segment Anything API for the given boxes and return masks, scores.

    Args:
        image: PIL Image
        boxes: Numpy array of boxes [x1, y1, x2, y2]
        multimask_output: Whether to output multiple masks per box
        server_url: URL of the Segment Anything API server

    Returns:
        masks: Numpy array of masks (M, H, W)
        scores: Numpy array of scores
    """
    payload = {
        "image_data": encode_image(image),
        "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
        "multimask_output": multimask_output,
    }

    resp = requests.post(server_url, json=payload)
    resp.raise_for_status()
    result = resp.json()

    h, w = np.array(image).shape[:2]
    contours = result.get("contours", [])
    masks = contours_to_masks(contours, (h, w))

    scores = np.array(result.get("scores", []))

    return masks, scores
