import base64
import io


import numpy as np
import requests
from PIL import Image


def encode_image(image: Image.Image) -> str:
    """Encode image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str


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
    text_labels = result["text_labels"]

    return boxes, scores, text_labels
