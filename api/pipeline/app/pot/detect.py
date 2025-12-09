import numpy as np
from PIL import Image
from app.utils import call_grounding_dino_api


def filter_by_aspect_ratio(boxes, ratio: float = 3 / 2):
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = widths / np.maximum(heights, 1e-6)
    ar_mask = ((aspect >= 1 / ratio) & (aspect <= ratio)).astype(bool)
    return ar_mask


def filter_by_areas(boxes, areas, image_np):
    # Filter by area (remove outliers using IQR and median for both small and large)
    if len(boxes) >= 2:
        q1, q3 = np.percentile(areas, [25, 75])
        iqr = q3 - q1
        median_area = float(np.median(areas))

    # Upper threshold (remove large outliers)
    tukey_hi = q3 + 1.5 * iqr
    median_hi = median_area * 1.5 if median_area > 0 else tukey_hi
    hi_thresh = min(tukey_hi, median_hi)
    img_area = image_np.shape[0] * image_np.shape[1]
    hi_thresh = min(hi_thresh, 0.2 * img_area)

    # Lower threshold (remove small outliers)
    tukey_lo = q1 - 1.5 * iqr
    median_lo = median_area * 0.5 if median_area > 0 else tukey_lo
    lo_thresh = max(tukey_lo, median_lo, 0)

    inlier_mask = ((areas >= lo_thresh) & (areas <= hi_thresh)).astype(bool)
    if not inlier_mask.any():
        # Fallback: keep box closest to median area
        median_idx = int(np.argmin(np.abs(areas - median_area)))
        inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
        inlier_mask[median_idx] = True

    return inlier_mask


def filter_by_confidence(confidences):
    gap = 0.1
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


def remove_contained_boxes(boxes):
    # Remove boxes fully contained in larger boxes
    if len(boxes) >= 2:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    contains = (
        (x1[:, None] <= x1[None, :])
        & (y1[:, None] <= y1[None, :])
        & (x2[:, None] >= x2[None, :])
        & (y2[:, None] >= y2[None, :])
    )
    np.fill_diagonal(contains, False)

    larger = areas[:, None] > areas[None, :]
    contained_smaller = contains & larger
    to_remove = np.asarray(contained_smaller.any(axis=0), dtype=bool)

    if to_remove.any():
        kept = ~to_remove
        return kept
    else:
        return np.ones(len(boxes), dtype=bool)


def order_boxes(boxes):
    xs = (boxes[:, 0] + boxes[:, 2]) / 2
    ys = (boxes[:, 1] + boxes[:, 3]) / 2
    order = np.lexsort((xs, ys))
    return order


def detect_pots(image, text_prompt="pot", threshold=0.03, text_threshold=0):
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

    if len(boxes) == 0:
        return boxes, confidences, class_names

    ar_mask = filter_by_aspect_ratio(boxes)
    boxes = boxes[ar_mask]
    confidences = confidences[ar_mask]
    class_names = class_names[ar_mask]

    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    areas = widths * heights

    inlier_mask = filter_by_areas(boxes, areas, image_np)
    boxes = boxes[inlier_mask]
    confidences = confidences[inlier_mask]
    class_names = (
        class_names[inlier_mask]
        if class_names.shape[0] == inlier_mask.shape[0]
        else class_names
    )

    conf_mask = filter_by_confidence(confidences)
    boxes = boxes[conf_mask]
    confidences = confidences[conf_mask]
    class_names = (
        class_names[conf_mask]
        if class_names.shape[0] == conf_mask.shape[0]
        else class_names
    )

    kept = remove_contained_boxes(boxes)
    boxes = boxes[kept]
    confidences = confidences[kept]
    class_names = class_names[kept]

    if len(boxes) > 0:
        order = order_boxes(boxes)
        boxes = boxes[order]
        confidences = confidences[order]
        class_names = class_names[order]

    return boxes, confidences, class_names
