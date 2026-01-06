import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image


def encode_pil_image(image):
    """Encodes a PIL image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array (RGB)."""
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def mask_to_contour(mask: np.ndarray) -> list:
    """Convert binary mask to contour points."""
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    return biggest[:, 0, :].tolist()


def _move_to_device(obj, device):
    """Recursively move tensors in a nested structure to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        # We must mutation-safe update for dicts/lists
        for k, v in obj.items():
            obj[k] = _move_to_device(v, device)
        return obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _move_to_device(v, device)
        return obj
    elif hasattr(obj, "__dict__"):
        # Handle custom objects (like cache)
        for k, v in obj.__dict__.items():
            if not k.startswith("__"):
                obj.__dict__[k] = _move_to_device(v, device)
        return obj
    return obj
