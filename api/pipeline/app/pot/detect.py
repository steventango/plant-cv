import logging

import numpy as np
from PIL import Image

from app.utils import call_sam3_api

logger = logging.getLogger(__name__)


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
        q1, q3, iqr, median_area = 0, 0, 0, 0

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


def order_boxes(boxes):
    xs = (boxes[:, 0] + boxes[:, 2]) / 2
    ys = (boxes[:, 1] + boxes[:, 3]) / 2
    order = np.lexsort((xs, ys))
    return order


def filter_clipped_pots(boxes, areas, image_shape, margin=1, area_threshold=0.725):
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


def filter_by_aspect_ratio(boxes, min_ratio=0.5, max_ratio=2.0):
    """
    Remove boxes that have an unusual aspect ratio.
    Pots should be roughly square or slightly rectangular.
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Avoid division by zero
    heights = np.maximum(heights, 1e-6)
    aspect_ratios = widths / heights

    keep_mask = (aspect_ratios >= min_ratio) & (aspect_ratios <= max_ratio)

    if np.sum(~keep_mask) > 0:
        logger.debug(
            f"Aspect Ratio Filter dropped {np.sum(~keep_mask)} boxes. Ratios: {aspect_ratios[~keep_mask]}"
        )

    return keep_mask


def filter_pot_masks(masks, image_np):
    """
    Apply shared filtering logic to pot masks.
    """
    if not masks:
        return []

    # Convert mask info to boxes and confidences
    boxes = np.array([m["box"] for m in masks])
    confidences = np.array([m["score"] for m in masks])
    masks_np = np.array(masks)

    # 2. Filter clipped pots
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    current_areas = widths * heights

    edge_mask = filter_clipped_pots(boxes, current_areas, image_np.shape, margin=1)
    logger.info(f"Clipped Filter: kept {np.sum(edge_mask)}/{len(boxes)}")
    boxes = boxes[edge_mask]
    confidences = confidences[edge_mask]
    masks_np = masks_np[edge_mask]

    if len(boxes) == 0:
        return []

    # 3. Area filter
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    areas = widths * heights

    inlier_mask = filter_by_areas(boxes, areas, image_np, confidences=confidences)
    logger.info(f"Area Filter: kept {np.sum(inlier_mask)}/{len(boxes)}")
    boxes = boxes[inlier_mask]
    confidences = confidences[inlier_mask]
    masks_np = masks_np[inlier_mask]

    if len(boxes) == 0:
        return []

    # 4. Aspect ratio filter
    ratio_mask = filter_by_aspect_ratio(boxes)
    logger.info(f"Aspect Ratio Filter: kept {np.sum(ratio_mask)}/{len(boxes)}")
    boxes = boxes[ratio_mask]
    confidences = confidences[ratio_mask]
    masks_np = masks_np[ratio_mask]

    if len(boxes) == 0:
        return []

    logger.debug(f"filter_pot_masks: finished with {len(masks_np)} masks")
    return masks_np.tolist()


def detect_pots_sam3(image, state=None, extra_prompts=None, **kwargs):
    """
    Detect pots in an image using SAM3 and filter outliers.

    Args:
        image: PIL Image or numpy array
        state: Previous pot tracking state (for propagate mode)
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

    if "score_threshold_detection" not in kwargs:
        kwargs["score_threshold_detection"] = 0.15

    # Determine endpoint based on whether we have previous state
    if state is None:
        result = call_sam3_api(
            image, endpoint="detect", text_prompt="soil plant pot", **kwargs
        )
    else:
        result = call_sam3_api(image, endpoint="propagate", state=state, **kwargs)

    masks = result.get("masks", [])

    filtered_masks = filter_pot_masks(masks, image_np)

    if not filtered_masks:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            result.get("session_id"),
            [],
            result,
        )

    # Convert back to arrays for the expected return format
    boxes = np.array([m["box"] for m in filtered_masks])
    confidences = np.array([m["score"] for m in filtered_masks])
    class_names = np.array(["pot"] * len(filtered_masks))

    return (
        boxes,
        confidences,
        class_names,
        result.get("session_id"),
        filtered_masks,
        result,
    )
