import logging
import numpy as np
import supervision as sv

from app.plant.stats import analyze_plant_mask

logger = logging.getLogger(__name__)


def visualize_plant_detections(
    image_np: np.ndarray, boxes: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
    """
    Visualize plant detections with boxes and labels.
    """
    ids = np.arange(len(boxes), dtype=int)
    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=ids,
        tracker_id=ids,
    )

    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(scene=image_np.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    labels = [f"#{tid} {conf:.2f}" for tid, conf in zip(ids, confidences)]
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    return annotated


def visualize_plant_segmentation(
    image_np: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    masks: np.ndarray,
    best_idx: int,
    reason_codes: dict = None,
    combined_score: float = None,
) -> np.ndarray:
    """
    Visualize plant segmentation results.
    If best_idx is provided, highlights the selected mask in green and others in gray.
    If reason_codes is provided (and best_idx is None/invalid), visualizes rejected masks.
    """
    annotated = image_np.copy()

    if best_idx is not None:
        # 2. Annotate selected (Green)
        selected_detection = sv.Detections(
            xyxy=boxes[[best_idx]],
            confidence=confidences[[best_idx]],
            class_id=np.array([best_idx]),
        )
        selected_detection.mask = masks[[best_idx]].astype(bool)

        mask_annotator_green = sv.MaskAnnotator(color=sv.Color(r=0, g=255, b=0))
        annotated = mask_annotator_green.annotate(
            scene=annotated, detections=selected_detection
        )

        label_annotator = sv.LabelAnnotator()
        label = f"#{best_idx} {confidences[best_idx]:.2f}"
        if combined_score is not None:
            label += f" | {combined_score:.2f}"
        labels = [label]
        annotated = label_annotator.annotate(
            scene=annotated, detections=selected_detection, labels=labels
        )

    return annotated


def visualize_annotation(
    image: np.ndarray,
    boxes: list = None,
    confidences: list = None,
    masks: list = None,
    labels: list = None,
    stats: bool = True,
    selected_index: int = None,
    mask_scores: list = None,
    combined_scores: list = None,
    pot_size_mm: float = 60.0,
    margin: float = 0.25,
) -> np.ndarray:
    """
    Unified visualization function.

    Args:
        image: Input image (H, W, C)
        boxes: List of [x1, y1, x2, y2]
        confidences: List of confidence scores
        masks: List of binary masks (H, W) or (H, W, 1)
        labels: List of strings corresponding to boxes
        stats: If True and mask is provided, draw PlantCV stats.
        selected_index: Index of selected mask
        mask_scores: List of mask scores
        combined_scores: List of combined scores
        pot_size_mm: Physical size of pot in mm (default: 60mm)
        margin: Margin ratio in warped image (default: 0.25 = 25%)
    """
    annotated = image.copy()

    length = len(boxes) if boxes is not None else 0
    if length == 0:
        length = len(masks) if masks is not None else 0

    xyxy = np.array(boxes) if boxes is not None else np.empty((length, 4))
    mask = np.array(masks) if masks is not None else np.empty((length, image.shape[0], image.shape[1]))

    detections = sv.Detections(
        xyxy=xyxy,
        mask=mask,
        class_id=np.arange(length),
    )

    if boxes is not None:
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)

    if masks is not None:
        mask_annotator = sv.MaskAnnotator()
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    if stats and selected_index is not None:
        mask = masks[selected_index]
        stat, annotated = analyze_plant_mask(
            annotated, mask, pot_size_mm=pot_size_mm, margin=margin
        )

    if labels is None:
        labels = [
            f"{i}"
            for i in range(len(detections))
        ]

        if confidences is not None:
            labels = [
                f"{labels[i]} B:{confidences[i]:.2f}"
                for i in range(len(detections))
            ]

        if mask_scores is not None:
            labels = [
                f"{labels[i]} M:{mask_scores[i]:.2f}"
                for i in range(len(detections))
            ]

        if combined_scores is not None:
            labels = [
                f"{labels[i]} C:{combined_scores[i]:.2f}"
                for i in range(len(detections))
            ]

        if selected_index is not None:
            labels[selected_index] = (
                f"*{labels[selected_index]} {stat['area']:.0f} mm^2"
            )

    label_annotator = sv.LabelAnnotator()
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    return annotated
