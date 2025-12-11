import base64
import io
import numpy as np
from PIL import Image
from pathlib import Path


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image.save(path)
