import base64
import io

import cv2
import numpy as np
import requests
from PIL import Image


def encode_image(image: Image.Image | np.ndarray) -> str:
    """Encode image to base64 string."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def call_grounding_dino_api(
    image: Image.Image,
    text_prompt: str,
    threshold: float = 0.05,
    text_threshold: float = 0.05,
    server_url: str = "http://grounding-dino:8000/predict",
):
    """
    Call the Grounding DINO API with an image and text prompt.

    Args:
        image: PIL Image to be processed
        text_prompt: Text prompt for object detection
        threshold: Confidence threshold for boxes
        text_threshold: Text threshold
        server_url: URL of the Grounding DINO API server

    Returns:
        boxes: Numpy array of boxes in format [x1, y1, x2, y2]
        scores: Numpy array of confidence scores
        text_labels: List of detected class names
    """
    payload = {
        "image_data": encode_image(image),
        "text_prompt": text_prompt,
        "threshold": threshold,
        "text_threshold": text_threshold,
    }

    response = requests.post(server_url, json=payload)
    response.raise_for_status()
    result = response.json()
    boxes = np.array(result["boxes"])
    scores = np.array(result["scores"])
    text_labels = np.array(result["text_labels"])

    return boxes, scores, text_labels


def contours_to_masks(contours: list[list[list[int]]], image_size: tuple[int, int]):
    """
    Convert contours to binary masks.

    Args:
        contours: List of contours, each contour is a list of [x, y] points
        image_size: Tuple (H, W) of image size

    Returns:
        masks: Numpy array of masks (M, H, W) with values 0 or 1
    """
    h, w = image_size
    masks = np.zeros((len(contours), h, w), dtype=np.uint8)
    for i, contour in enumerate(contours):
        if contour:
            contour_np = np.array(contour, dtype=np.int32)
            cv2.fillPoly(masks[i], [contour_np], 1)
    return masks


def call_segment_anything_api(
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
