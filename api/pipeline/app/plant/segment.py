import numpy as np


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
    median_area = float(np.median(mask_areas)) if mask_areas.size else 0.0
    too_big_vs_median = (
        (median_area > 0.0) & (mask_areas > median_multiplier * median_area)
    ).astype(bool)

    valid_mask = ((~near_full_mask) & (~too_big_vs_median)).astype(bool)

    # Map reason codes
    reason_codes = {}
    for idx in range(len(masks)):
        if near_full_mask[idx] or too_big_vs_median[idx]:
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
    elif len(valid_indices) == 0:
        return None, np.array([])

    H, W = image_shape[:2]
    center_x = W / 2.0
    center_y = H / 2.0
    sigma = min(W, H) * 0.6

    valid_boxes = boxes[valid_indices]
    valid_confidences = confidences[valid_indices].copy()

    box_centers_x = (valid_boxes[:, 0] + valid_boxes[:, 2]) / 2
    box_centers_y = (valid_boxes[:, 1] + valid_boxes[:, 3]) / 2

    # RBF score: higher for boxes closer to center
    center_score = np.exp(
        -((center_x - box_centers_x) ** 2 + (center_y - box_centers_y) ** 2)
        / (sigma / 2) ** 2
    )

    combined_scores = valid_confidences * center_score
    best_relative_idx = int(np.argmax(combined_scores))
    best_idx_original = int(valid_indices[int(best_relative_idx)])

    return best_idx_original, combined_scores
