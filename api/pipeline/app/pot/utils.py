import base64
import io

from PIL import Image


def encode_image(image: Image.Image) -> str:
    """Encode image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str