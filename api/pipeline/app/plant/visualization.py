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

    # Case 1: Visualization for rejected masks (when no valid mask found)
    if best_idx is None and reason_codes:
        mask_detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=np.arange(len(boxes), dtype=int),
        )
        mask_detections.mask = masks.astype(bool)

        # Mark invalid masks
        for idx, code in reason_codes.items():
            mask_detections.class_id[idx] = code

        mask_annotator = sv.MaskAnnotator()
        annotated = mask_annotator.annotate(scene=annotated, detections=mask_detections)
        return annotated

    # Case 2: Visualization for successful selection
    # 1. Annotate others (Gray)
    other_indices = [i for i in range(len(masks)) if i != best_idx]
    if other_indices:
        other_detections = sv.Detections(
            xyxy=boxes[other_indices],
            confidence=confidences[other_indices],
            class_id=np.array(other_indices),
        )
        other_detections.mask = masks[other_indices].astype(bool)

        mask_annotator_gray = sv.MaskAnnotator(color=sv.Color(r=128, g=128, b=128))
        annotated = mask_annotator_gray.annotate(
            scene=annotated, detections=other_detections
        )

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

    # 3. Add labels
    all_detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=np.arange(len(boxes), dtype=int),
    )

    label_annotator = sv.LabelAnnotator()
    labels = [
        f"#{i} {conf:.2f}{' *' if i == best_idx else ''}"
        for i, conf in enumerate(confidences)
    ]
    annotated = label_annotator.annotate(
        scene=annotated, detections=all_detections, labels=labels
    )

    return annotated
