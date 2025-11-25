"""Warp pot boxes to square images."""

import cv2
import numpy as np
from PIL import Image


def warp_quad_to_square(
    image_np,
    quad,
    margin=0.25,
    output_size=None,
    border_value=(0, 0, 0),
):
    """
    Warp a quadrilateral region to a square image with uniform margin.

    Args:
        image_np: Input image as numpy array
        quad: 4x2 array of quadrilateral corners (TL, TR, BR, BL)
        margin: Padding margin within the output square (default 0.25 = 25%)
        output_size: Optional fixed output size
        border_value: Border color for areas outside the image

    Returns:
        (warped_square, homography_matrix)
    """
    quad = np.asarray(quad, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError("quad must have shape (4,2)")

    def _edge_len(a, b):
        return float(np.hypot(*(b - a)))

    tl, tr, br, bl = quad
    max_edge = max(
        _edge_len(tl, tr), _edge_len(tr, br), _edge_len(br, bl), _edge_len(bl, tl)
    )
    if max_edge < 1:
        max_edge = 1.0

    side = int(round((1.0 + 2.0 * float(margin)) * max_edge))
    side = max(side, 8)
    pad = int(round(max_edge * float(margin)))

    pad = max(0, min(pad, side // 4))

    inner = max(side - 2 * pad, 1)
    dst = np.array(
        [
            [pad, pad],
            [pad + inner - 1, pad],
            [pad + inner - 1, pad + inner - 1],
            [pad, pad + inner - 1],
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(quad, dst)
    warped = cv2.warpPerspective(
        image_np,
        H,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    if output_size is not None and output_size != side:
        warped = cv2.resize(warped, (output_size, output_size))

    return warped, H


def warp_pots(image: Image.Image, quads: np.ndarray, margin=0.25, output_size=256):
    """
    Warp multiple pot regions to square images.

    Args:
        image: PIL Image
        quads: Numpy array of quadrilaterals (N, 4, 2) float32
        margin: Padding margin within the output square
        output_size: Optional fixed output size

    Returns:
        List of (warped_image, homography) tuples
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    results = []
    for quad in quads:
        try:
            warped, H = warp_quad_to_square(
                image_np, quad, margin=margin, output_size=output_size
            )
            results.append((warped, H))
        except Exception as e:
            print(f"Failed to warp quadrilateral: {e}")
            results.append((None, None))
    return results
