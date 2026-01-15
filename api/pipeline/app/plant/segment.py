import logging

import cv2
import numpy as np
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)


def preprocess_for_otsu(image_rgb: np.ndarray) -> np.ndarray:
    """
    Pre-process image for Otsu refinement: Grayscale + fast OpenCV CLAHE.
    """
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def refine_mask_with_otsu(
    mask: np.ndarray,
    preprocessed_gray: np.ndarray,
) -> np.ndarray:
    """
    Refine a binary mask using Otsu thresholding on the equalized grayscale image.
    This helps to better separate the plant from the background within the SAM mask.

    Args:
        image_rgb: RGB image as numpy array (H, W, 3)
        mask: Binary mask as numpy array (H, W)
        pot_width_px: Approximate width of the pot in pixels (for kernel size).
                      If None, defaults to 1/5 of image width.

    Returns:
        Refined binary mask (H, W)
    """
    if not np.any(mask):
        return mask

    # Calculate Otsu threshold only on pixels within the initial mask
    masked_pixels = preprocessed_gray[mask > 0]
    if masked_pixels.size == 0:
        return mask

    try:
        # threshold_otsu expects 1D array
        otsu_thresh = threshold_otsu(masked_pixels)
    except ValueError:
        return mask

    # dilate mask
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Create binary mask from threshold and intersect with original mask
    # Assuming object_type="light" (plant is usually lighter than the pot/background in CLAHE)
    refined_mask_bool = (preprocessed_gray > otsu_thresh) & (dilated_mask > 0)

    return refined_mask_bool.astype(np.uint8) * 255
