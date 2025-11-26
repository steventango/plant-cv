import numpy as np
import supervision as sv


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
) -> np.ndarray:
    """
    Visualize plant segmentation results.
    If best_idx is provided, highlights the selected mask in green and others in gray.
    If reason_codes is provided (and best_idx is None/invalid), visualizes rejected masks.
    """
    annotated = image_np.copy()

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
    labels = [f"#{best_idx} {confidences[best_idx]:.2f}"]
    annotated = label_annotator.annotate(
        scene=annotated, detections=selected_detection, labels=labels
    )

    return annotated
