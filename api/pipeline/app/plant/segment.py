import logging

import cv2
import numpy as np
from plantcv import plantcv as pcv
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


def filter_masks_by_area(
    masks: np.ndarray,
    image_shape: tuple[int, int],
    near_full_threshold: float = 0.92,
    median_multiplier: float = 10.0,
):
    """
    Filter plant masks by relative area to avoid full-image or tiny masks.

    Args:
        masks: (M, H, W) with values >= 0
        image_shape: Tuple (H, W) of image shape
        near_full_threshold: Ratio above which mask is considered too large (default: 0.92)
        median_multiplier: Masks > this multiple of median area are rejected (default: 10.0)

    Returns:
        valid_indices: Numpy array of valid mask indices
        mask_areas: Numpy array of mask areas
        area_ratios: Numpy array of area ratios
        reason_codes: Dict mapping mask index to reason code (902 = too large, 903 = too small)
    """
    if len(masks) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            {},
        )

    H, W = image_shape[:2]
    crop_area = float(H * W)

    # Compute areas and ratios
    mask_areas = np.array([np.sum(mask > 0) for mask in masks], dtype=float)
    area_ratios = mask_areas / max(crop_area, 1.0)

    # Compute filters
    near_full_mask = (area_ratios > near_full_threshold).astype(bool)
    valid_mask = (~near_full_mask).astype(bool)

    # Map reason codes
    reason_codes = {}
    for idx in range(len(masks)):
        if near_full_mask[idx]:
            reason_codes[idx] = 902  # too large

    valid_indices = np.where(valid_mask)[0]

    return valid_indices, mask_areas, area_ratios, reason_codes


def select_best_mask(
    masks,
    confidences,
    boxes,
    image_shape,
    valid_indices=None,
):
    """
    Select the best plant mask using heuristic scoring.

    Combines:
    1. Grounding DINO confidence
    2. Proximity to center (RBF scoring)

    Args:
        masks: Numpy array of masks (M, H, W)
        confidences: Numpy array of confidence scores
        boxes: Numpy array of boxes [x1, y1, x2, y2]
        image_shape: Tuple (H, W) of image shape
        valid_indices: Optional indices of valid masks to consider

    Returns:
        best_idx_original: Index in the original masks array
        combined_scores: Numpy array of combined scores
    """
    if valid_indices is None:
        valid_indices = np.arange(len(masks), dtype=int)
        valid_mask = np.ones(len(masks), dtype=bool)
    else:
        valid_mask = np.zeros(len(masks), dtype=bool)
        valid_mask[valid_indices] = True

    confidences[~valid_mask] = 0.0

    H, W = image_shape[:2]
    center_x = W / 2.0
    center_y = H / 2.0
    # The crop width W is 1.5 * pot_width, so sigma should be equal to W.
    sigma = min(W, H)

    valid_boxes = boxes
    valid_confidences = confidences

    box_centers_x = (valid_boxes[:, 0] + valid_boxes[:, 2]) / 2
    box_centers_y = (valid_boxes[:, 1] + valid_boxes[:, 3]) / 2

    # RBF score: higher for boxes closer to center
    center_score = np.exp(
        -((center_x - box_centers_x) ** 2 + (center_y - box_centers_y) ** 2)
        / (sigma / 2) ** 2
    )

    mask_areas = np.array([np.sum(mask > 0) for mask in masks], dtype=float)
    box_widths = valid_boxes[:, 2] - valid_boxes[:, 0]
    box_heights = valid_boxes[:, 3] - valid_boxes[:, 1]
    box_areas = box_widths * box_heights

    # AREA ratio score: mask to box areas
    area_ratio_score = mask_areas / box_areas
    # Penalize high area ratios
    area_ratio_score = 1 - np.clip(area_ratio_score, 0.0, 1.0)
    area_ratio_score[area_ratio_score > 0.4] = 1

    combined_scores = valid_confidences * center_score * area_ratio_score
    combined_scores = np.exp(combined_scores) / np.sum(np.exp(combined_scores))

    best_relative_idx = int(np.argmax(combined_scores))
    best_score = combined_scores[best_relative_idx]

    return best_relative_idx, best_score, combined_scores
