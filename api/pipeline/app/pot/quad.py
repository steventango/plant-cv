import cv2
import numpy as np


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL for stable warps."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("_order_quad expects shape (4,2)")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def mask_to_quadrilateral(mask: np.ndarray) -> np.ndarray:
    """
    Compute a bounding quadrilateral around the pot mask using cv2.approxPolyN.

    Args:
        mask: 2D binary mask

    Returns:
        (4,2) float32 points representing the bounding quadrilateral (TL, TR, BR, BL).
    """
    if mask.ndim != 2:
        raise ValueError("mask_to_quadrilateral expects a 2D mask")
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        raise ValueError("mask_to_quadrilateral received an empty mask")

    # Remove small protrusions using morphological opening
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # Find largest contour
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask after morphology")
    largest = max(contours, key=cv2.contourArea)

    # Approximate the contour to a quadrilateral using approxPolyN
    # ensure_convex=True (default) approximates with a convex hull
    quad = cv2.approxPolyN(largest, nsides=4)

    # Reshape from (4, 1, 2) to (4, 2)
    quad = quad.reshape(4, 2).astype(np.float32)

    # Order corners as TL, TR, BR, BL
    return order_quad(quad)


def compute_quadrilaterals(masks: np.ndarray) -> np.ndarray:
    """
    Compute quadrilaterals for multiple masks.

    Args:
        masks: Numpy array of masks (M, H, W)

    Returns:
        Numpy array of quadrilaterals, each (M, 4, 2) float32
    """
    quads = np.zeros((masks.shape[0], 4, 2), dtype=np.float32)
    for i in range(masks.shape[0]):
        try:
            quad = mask_to_quadrilateral(masks[i])
            quads[i] = quad
        except Exception as e:
            print(f"Failed to compute quadrilateral for mask {i}: {e}")
    return quads
