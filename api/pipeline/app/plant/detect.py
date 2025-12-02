import numpy as np
from PIL import Image
from utils import call_grounding_dino_api


def detect_plant(
    crop_image: Image,
    text_prompt: str = "plant",
    threshold: float = 0.05,
    text_threshold: float = 0.05,
):
    """
    Detect plants in a warped pot crop image.

    Args:
        crop_image: PIL Image of the warped pot crop
        text_prompt: Text prompt for detection (default: "plant")
        threshold: Confidence threshold
        text_threshold: Text threshold

    Returns:
        boxes: Numpy array of boxes [x1, y1, x2, y2]
        confidences: Numpy array of confidence scores
        class_names: List of class names
    """
    return call_grounding_dino_api(
        image=crop_image,
        text_prompt=text_prompt,
        threshold=threshold,
        text_threshold=text_threshold,
    )


def filter_boxes_by_area(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    image_shape: tuple[int, int],
    area_ratio_threshold: float = 0.9,
):
    """
    Filter boxes that are too large relative to the image area.

    Args:
        boxes: Numpy array of boxes [x1, y1, x2, y2]
        confidences: Numpy array of confidence scores
        class_names: List of class names
        image_shape: Tuple (H, W) of image shape
        area_ratio_threshold: Maximum ratio of box area to image area (default: 0.90)

    Returns:
        filtered_boxes, filtered_confidences, filtered_class_names, valid_mask
    """
    if len(boxes) == 0:
        return boxes, confidences, class_names, np.array([], dtype=bool)

    H, W = image_shape[:2]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    area_ratios = areas / float(H * W)

    valid_masks = (area_ratios <= area_ratio_threshold).astype(bool)

    filtered_boxes = boxes[valid_masks]
    filtered_confidences = confidences[valid_masks]

    # Ensure class_names is a numpy array for boolean indexing
    if isinstance(class_names, list):
        class_names = np.array(class_names)
    filtered_class_names = class_names[valid_masks]

    return filtered_boxes, filtered_confidences, filtered_class_names, valid_masks


def filter_boxes_by_aspect_ratio(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    image_shape: tuple[int, int],
    aspect_ratio_threshold: float = 1.5,
):
    if len(boxes) == 0:
        return boxes, confidences, class_names, np.array([], dtype=bool)

    H, W = image_shape[:2]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    aspect_ratios = widths / heights

    valid_masks = (1 / aspect_ratio_threshold <= aspect_ratios).astype(bool)
    valid_masks &= (aspect_ratios <= aspect_ratio_threshold).astype(bool)

    filtered_boxes = boxes[valid_masks]
    filtered_confidences = confidences[valid_masks]

    # Ensure class_names is a numpy array for boolean indexing
    if isinstance(class_names, list):
        class_names = np.array(class_names)
    filtered_class_names = class_names[valid_masks]

    return filtered_boxes, filtered_confidences, filtered_class_names, valid_masks


def select_top_boxes(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str] | np.ndarray,
    image_shape: tuple[int, int],
    k: int = 5,
):
    """
    Select top k boxes based on a heuristic score (Confidence * (1 - Area_Ratio)).
    This penalizes very large boxes (like the pot itself) in favor of smaller plant boxes.

    Args:
        boxes: Numpy array of boxes [x1, y1, x2, y2]
        confidences: Numpy array of confidence scores
        class_names: List or array of class names
        image_shape: Tuple (H, W) of image shape
        k: Number of boxes to keep

    Returns:
        selected_boxes, selected_confidences, selected_class_names
    """
    if len(boxes) <= k:
        return boxes, confidences, class_names

    H, W = image_shape[:2]

    sorted_indices = np.argsort(confidences)[::-1]
    top_indices = sorted_indices[:k]

    # Handle class_names being list or array
    if isinstance(class_names, list):
        class_names = np.array(class_names)

    return boxes[top_indices], confidences[top_indices], class_names[top_indices]
