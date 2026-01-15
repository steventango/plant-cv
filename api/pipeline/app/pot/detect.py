import logging

import numpy as np
from PIL import Image

from app.utils import call_sam3_api

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_by_aspect_ratio(boxes, low: float = 0.5, high: float = 1.5):
    """
    Filter boxes by aspect ratio.
    Default range [0.5, 1.5] relaxes the lower bound (was ~0.66)
    while maintaining the upper bound (1.5) to avoid false positives like E11Z03 (AR 1.51).
    """
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = widths / np.maximum(heights, 1e-6)
    ar_mask = ((aspect >= low) & (aspect <= high)).astype(bool)
    if np.sum(ar_mask) < len(boxes):
        dropped = ~ar_mask
        logger.debug(
            f"AR Filter dropped {np.sum(dropped)} detected boxes. Aspects of dropped: {aspect[dropped]}"
        )
    return ar_mask


def filter_by_areas(
    boxes, areas, image_np, confidences=None, stats_conf_threshold=0.03
):
    # Filter by area (remove outliers using IQR and median for both small and large)
    img_area = image_np.shape[0] * image_np.shape[1]

    # Use only "reasonable" boxes to calculate statistics:
    # 1. Exclude boxes > 20% of image area
    # 2. If confidences provided, only use boxes with confidence > stats_conf_threshold
    #    This prevents noise boxes from skewing statistics when using low detection thresholds
    size_mask = areas <= 0.2 * img_area
    if confidences is not None:
        conf_mask = confidences >= stats_conf_threshold
        stats_mask = size_mask & conf_mask
    else:
        stats_mask = size_mask

    stats_areas = areas[stats_mask]
    if len(stats_areas) < 2:
        stats_areas = areas[size_mask]  # Fallback to size-only filter
    if len(stats_areas) < 2:
        stats_areas = areas  # Fallback if everything is huge or empty

    if len(stats_areas) >= 2:
        q1, q3 = np.percentile(stats_areas, [25, 75])
        iqr = q3 - q1
        median_area = float(np.median(stats_areas))
    else:
        q1, q3, median_area = 0, 0, 0

    # Upper threshold (remove large outliers)
    tukey_hi = q3 + 1.5 * iqr
    median_hi = median_area * 1.5 if median_area > 0 else tukey_hi
    hi_thresh = min(tukey_hi, median_hi)
    hi_thresh = min(hi_thresh, 0.2 * img_area)

    # Lower threshold (remove small outliers)
    tukey_lo = q1 - 2.5 * iqr
    median_lo = median_area * 0.4 if median_area > 0 else tukey_lo
    lo_thresh = max(tukey_lo, median_lo, 0)

    inlier_mask = ((areas >= lo_thresh) & (areas <= hi_thresh)).astype(bool)
    if np.sum(inlier_mask) < len(boxes):
        dropped = ~inlier_mask
    if np.sum(inlier_mask) < len(boxes):
        dropped = ~inlier_mask
        logger.debug(
            f"Area Filter dropped {np.sum(dropped)} boxes. Areas of dropped: {areas[dropped]}. Limits: Lo={lo_thresh}, Hi={hi_thresh}"
        )

    # Log kept areas using log_debug
    kept_areas = areas[inlier_mask]
    if len(kept_areas) > 0:
        # Calculate aspect ratios and confidences for kept boxes for debugging
        kept_boxes = boxes[inlier_mask]
        kept_widths = np.clip(kept_boxes[:, 2] - kept_boxes[:, 0], a_min=0, a_max=None)
        kept_heights = np.clip(kept_boxes[:, 3] - kept_boxes[:, 1], a_min=0, a_max=None)
        with np.errstate(divide="ignore", invalid="ignore"):
            kept_aspects = kept_widths / np.maximum(kept_heights, 1e-6)

        logger.debug(
            f"Kept {len(kept_areas)} boxes.\nAreas: {np.sort(kept_areas)}\nAspects: {kept_aspects}"
        )
    if not inlier_mask.any():
        # Fallback: keep box closest to median area
        median_idx = int(np.argmin(np.abs(areas - median_area)))
        inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
        inlier_mask[median_idx] = True

    return inlier_mask


def apply_nms(boxes, confidences, iou_threshold=0.55):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
    Keeps box with higher confidence when IoU > threshold.
    """
    if len(boxes) == 0:
        return np.array([], dtype=bool)

    # Sort by confidence descending
    indices = np.argsort(confidences)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Compare current box with rest
        rest_indices = indices[1:]

        # Calculate IoU
        # Coordinates
        xx1 = np.maximum(boxes[current, 0], boxes[rest_indices, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest_indices, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest_indices, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest_indices, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter_area = w * h

        # Area of boxes
        area_current = (boxes[current, 2] - boxes[current, 0]) * (
            boxes[current, 3] - boxes[current, 1]
        )
        area_rest = (boxes[rest_indices, 2] - boxes[rest_indices, 0]) * (
            boxes[rest_indices, 3] - boxes[rest_indices, 1]
        )

        union_area = area_current + area_rest - inter_area
        iou = inter_area / np.maximum(union_area, 1e-6)

        # Keep boxes with small IoU (not overlapping significantly)
        keep_mask = iou < iou_threshold
        dropped_indices = rest_indices[~keep_mask]
        if len(dropped_indices) > 0:
            logger.debug(
                f"NMS Suppressed indices {dropped_indices} due to IoU > {iou_threshold} (Max IoU {[f'{x:.2f}' for x in iou[~keep_mask]]}) with box {current}"
            )

        indices = rest_indices[keep_mask]

    # Create mask
    mask = np.zeros(len(boxes), dtype=bool)
    mask[keep] = True
    return mask


def order_boxes(boxes):
    xs = (boxes[:, 0] + boxes[:, 2]) / 2
    ys = (boxes[:, 1] + boxes[:, 3]) / 2
    order = np.lexsort((xs, ys))
    return order


def filter_clipped_pots(boxes, areas, image_shape, margin=1, area_threshold=0.7):
    """
    Remove boxes that are touching the edge AND have an area significantly smaller than the median.
    """
    h, w = image_shape[:2]
    # boxes are [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Check if touching edge
    touching_edge = (
        (x1 < margin) | (y1 < margin) | (x2 > w - margin) | (y2 > h - margin)
    )

    if not np.any(touching_edge):
        return np.ones(len(boxes), dtype=bool)

    median_area = np.median(areas)
    # Filter if touching edge AND area < area_threshold * median
    small_area = areas < (area_threshold * median_area)

    # We want to DROP if (touching_edge AND small_area)
    # So we KEEP if NOT (touching_edge AND small_area)
    keep_mask = ~(touching_edge & small_area)

    if np.sum(keep_mask) < len(boxes):
        dropped = ~keep_mask
        logger.debug(
            f"Clipped Pot Filter dropped {np.sum(dropped)} boxes. Areas: {areas[dropped]}, Median: {median_area}"
        )

    return keep_mask


def detect_pots_sam3(image, state=None, extra_prompts=None, **kwargs):
    """
    Detect pots in an image using SAM3 and filter outliers.

    Args:
        image: PIL Image or numpy array
        state: Previous pot tracking state (for propagate mode)
        extra_prompts: Optional list of additional text prompts to detect simultaneously
        **kwargs: Additional parameters for SAM3 (e.g., threshold, recondition_every_nth_frame)


    Returns:
        boxes: Filtered numpy array of boxes [x1, y1, x2, y2] in grid order
        confidences: Filtered confidence scores
        class_names: Filtered class names
        state: Updated pot tracking state
        masks: Filtered list of mask dictionaries
        raw_result: The full result from SAM3 API (useful for extra_prompts)
    """

    if isinstance(image, np.ndarray):
        image_np = image
        image = Image.fromarray(image)
    else:
        image_np = np.array(image)

    # Default threshold for pots if not provided
    if "score_threshold_detection" not in kwargs:
        kwargs["score_threshold_detection"] = 0.3

    # Determine endpoint based on whether we have previous state
    if state is None:
        endpoint = "detect"
        prompts = ["plant pot"]
        if extra_prompts:
            if isinstance(extra_prompts, list):
                prompts.extend(extra_prompts)
            else:
                prompts.append(extra_prompts)

        result = call_sam3_api(image, endpoint=endpoint, text_prompt=prompts, **kwargs)
    else:
        endpoint = "propagate"
        result = call_sam3_api(image, endpoint=endpoint, state=state, **kwargs)

    # Extract masks from result
    # If multiple prompts were used, masks are grouped by prompt
    prompt_masks = result.get("prompt_masks", {})
    if prompt_masks:
        # Use "plant pot" masks if available, otherwise fallback to all masks
        masks = prompt_masks.get("plant pot", result.get("masks", []))
    else:
        masks = result.get("masks", [])

    if len(masks) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            result.get("session_id"),
            [],
        )

    # Convert mask info to boxes and confidences
    boxes = np.array([m["box"] for m in masks])
    confidences = np.array([m["score"] for m in masks])
    class_names = np.array(["pot"] * len(masks))

    logger.debug(f"SAM3 detected {len(boxes)} pot boxes.")

    if len(boxes) == 0:
        return (
            boxes,
            confidences,
            class_names,
            result.get("session_id"),
            [],
        )

    # Helper to filter masks list
    masks_np = np.array(masks)

    # Apply same filtering logic as detect_pots
    ar_mask = filter_by_aspect_ratio(boxes, low=0.5, high=1.5)
    boxes = boxes[ar_mask]
    confidences = confidences[ar_mask]
    class_names = class_names[ar_mask]
    masks_np = masks_np[ar_mask]

    # Filter clipped pots
    # Access areas from indices that survived previous filters
    # We need to recompute areas for the currently surviving boxes
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    current_areas = widths * heights

    edge_mask = filter_clipped_pots(boxes, current_areas, image_np.shape, margin=1)
    boxes = boxes[edge_mask]
    confidences = confidences[edge_mask]
    class_names = class_names[edge_mask]
    masks_np = masks_np[edge_mask]

    if len(boxes) == 0:
        return (
            boxes,
            confidences,
            class_names,
            result.get("session_id"),
            [],
        )

    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    areas = widths * heights

    inlier_mask = filter_by_areas(boxes, areas, image_np, confidences=confidences)
    boxes = boxes[inlier_mask]
    confidences = confidences[inlier_mask]
    class_names = class_names[inlier_mask]
    masks_np = masks_np[inlier_mask]

    if len(boxes) == 0:
        return (
            boxes,
            confidences,
            class_names,
            result.get("session_id"),
            [],
        )

    # Apply NMS
    kept = apply_nms(boxes, confidences, iou_threshold=0.55)
    boxes = boxes[kept]
    confidences = confidences[kept]
    class_names = class_names[kept]
    masks_np = masks_np[kept]

    # Order boxes in grid order
    if len(boxes) > 0:
        order = order_boxes(boxes)
        boxes = boxes[order]
        confidences = confidences[order]
        class_names = class_names[order]
        masks_np = masks_np[order]

    return (
        boxes,
        confidences,
        class_names,
        result.get("session_id"),
        masks_np.tolist(),
        result,
    )
