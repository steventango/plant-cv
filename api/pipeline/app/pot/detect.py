import numpy as np
from PIL import Image
from app.utils import call_grounding_dino_api


def log_debug(message):
    with open("/tmp/pot_detect_debug.log", "a") as f:
        f.write(message + "\n")


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
        log_debug(
            f"DEBUG: AR Filter dropped {np.sum(dropped)} detected boxes. Aspects of dropped: {aspect[dropped]}"
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
        log_debug(
            f"DEBUG: Area Filter dropped {np.sum(dropped)} boxes. Areas of dropped: {areas[dropped]}. Limits: Lo={lo_thresh}, Hi={hi_thresh}"
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

        # We need original confidences here.
        # Note: 'confidences' in this scope is already filtered by previous steps (AR filter),
        # but not yet by area filter (inlier_mask is applied to 'boxes' inside this function logic flow in caller?)
        # Wait, filter_by_areas receives 'boxes' and returns 'inlier_mask'.
        # It doesn't have access to 'confidences' array passed to it.
        # So we can't log confidences HERE easily without changing function signature.
        # Let's just log Areas and Aspects here. Confidences are logged in the caller or we can pass them.
        # Check signature: def filter_by_areas(boxes, areas, image_np):

        log_debug(
            f"DEBUG: Kept {len(kept_areas)} boxes.\nAreas: {np.sort(kept_areas)}\nAspects: {kept_aspects}"
        )
    if not inlier_mask.any():
        # Fallback: keep box closest to median area
        median_idx = int(np.argmin(np.abs(areas - median_area)))
        inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
        inlier_mask[median_idx] = True

    return inlier_mask


def filter_by_confidence(confidences):
    gap = 0.08
    # Filter by confidence - find first gap >= 0.05 between sorted confidences
    conf_mask = np.ones(len(confidences), dtype=bool)  # Initialize to keep all

    if len(confidences) >= 2:
        sorted_confidences = np.sort(confidences)
        # Calculate gaps between consecutive sorted confidences
        gaps = np.diff(sorted_confidences)

        # Find the first gap >= gap
        gap_indices = np.where(gaps >= gap)[0]

        if len(gap_indices) > 0:
            # Set threshold to the confidence value after the first large gap
            first_gap_idx = gap_indices[0]
            threshold = sorted_confidences[first_gap_idx + 1]
            conf_mask = (confidences >= threshold).astype(bool)

    if not conf_mask.any():
        # Fallback: keep highest confidence box
        best_idx = int(np.argmax(confidences))
        conf_mask = np.zeros_like(conf_mask, dtype=bool)
        conf_mask[best_idx] = True
    return conf_mask


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
            log_debug(
                f"DEBUG: NMS Suppressed indices {dropped_indices} due to IoU > {iou_threshold} (Max IoU {[f'{x:.2f}' for x in iou[~keep_mask]]}) with box {current}"
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


def detect_pots(image, text_prompt="pot", threshold=0.01, text_threshold=0):
    """
    Detect pots in an image and filter outliers.

    Args:
        image: PIL Image or numpy array
        text_prompt: Text prompt for detection (default: "pot")
        threshold: Confidence threshold
        text_threshold: Text threshold

    Returns:
        boxes: Filtered numpy array of boxes [x1, y1, x2, y2] in grid order
        confidences: Filtered confidence scores
        class_names: Filtered class names
    """
    if isinstance(image, np.ndarray):
        image_np = image
        image = Image.fromarray(image)
    else:
        image_np = np.array(image)

    boxes, confidences, class_names = call_grounding_dino_api(
        image=image,
        text_prompt=text_prompt,
        threshold=threshold,
        text_threshold=text_threshold,
    )
    boxes, confidences, class_names = call_grounding_dino_api(
        image=image,
        text_prompt=text_prompt,
        threshold=threshold,
        text_threshold=text_threshold,
    )
    log_debug(f"DEBUG: Model returned {len(boxes)} boxes.")
    # print(f"DEBUG: Raw confidences: {confidences}", flush=True)

    if len(boxes) == 0:
        return boxes, confidences, class_names

    ar_mask = filter_by_aspect_ratio(boxes, low=0.5, high=1.5)
    boxes = boxes[ar_mask]
    confidences = confidences[ar_mask]
    class_names = class_names[ar_mask]

    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    areas = widths * heights

    inlier_mask = filter_by_areas(boxes, areas, image_np, confidences=confidences)
    boxes = boxes[inlier_mask]
    confidences = confidences[inlier_mask]

    if len(confidences) > 0:
        log_debug(f"DEBUG: Post-Area Filter Confidences: {np.sort(confidences)}")

    class_names = (
        class_names[inlier_mask]
        if class_names.shape[0] == inlier_mask.shape[0]
        else class_names
    )

    conf_mask = filter_by_confidence(confidences)
    if np.sum(conf_mask) < len(boxes):
        dropped = ~conf_mask
        log_debug(
            f"DEBUG: Confidence Filter dropped {np.sum(dropped)} boxes. Confidences of dropped: {confidences[dropped]}"
        )

    boxes = boxes[conf_mask]
    confidences = confidences[conf_mask]
    class_names = (
        class_names[conf_mask]
        if class_names.shape[0] == conf_mask.shape[0]
        else class_names
    )

    # Hybrid Filter: Drop boxes that are BOTH small AND low confidence
    # This targets "weed" false positives that look somewhat like pots but are statistically weak
    if len(boxes) >= 2:
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
        areas = widths * heights
        img_area = image_np.shape[0] * image_np.shape[1]

        # Recalculate robust median of surviving boxes
        stats_areas = areas[areas <= 0.2 * img_area]
        if len(stats_areas) < 2:
            stats_areas = areas

        if len(stats_areas) > 0:
            median_area = float(np.median(stats_areas))

            # Condition: Area < 60% of Median AND Confidence < 0.08
            is_small = areas < (median_area * 0.6)
            is_low_conf = confidences < 0.08
            to_drop = is_small & is_low_conf

            if np.any(to_drop):
                log_debug(
                    f"DEBUG: Hybrid Filter dropped {np.sum(to_drop)} boxes. Areas: {areas[to_drop]}, Confidences: {confidences[to_drop]}"
                )
                keep_mask = ~to_drop
                boxes = boxes[keep_mask]
                confidences = confidences[keep_mask]
                class_names = class_names[keep_mask]

    kept = apply_nms(boxes, confidences, iou_threshold=0.55)
    if np.sum(kept) < len(boxes):
        dropped = ~kept
        log_debug(f"DEBUG: NMS Filter dropped {np.sum(dropped)} boxes.")

    boxes = boxes[kept]
    confidences = confidences[kept]
    class_names = class_names[kept]

    if len(boxes) > 0:
        order = order_boxes(boxes)
        boxes = boxes[order]
        confidences = confidences[order]
        class_names = class_names[order]

    return boxes, confidences, class_names
