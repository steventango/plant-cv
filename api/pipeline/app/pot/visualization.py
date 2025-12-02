import cv2
import numpy as np
import supervision as sv


def visualize_pot_detections(
    image_np: np.ndarray, boxes: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
    """
    Visualize pot detections with boxes and labels.
    """
    ids = np.arange(len(boxes), dtype=int)
    # Create supervision detections
    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=ids,
        tracker_id=ids,
    )

    # Annotate with boxes
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(scene=image_np.copy(), detections=detections)

    # Annotate with labels (ID and confidence)
    label_annotator = sv.LabelAnnotator()
    labels = [f"#{tid} {conf:.2f}" for tid, conf in zip(ids, confidences)]
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    return annotated


def visualize_pot_segmentation(
    image_np: np.ndarray, boxes: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """
    Visualize pot segmentation with masks and boxes.
    """
    # Create supervision detections with masks
    detections = sv.Detections(
        xyxy=boxes, mask=masks.astype(bool), class_id=np.arange(len(masks))
    )

    # Annotate with masks
    mask_annotator = sv.MaskAnnotator()
    annotated = mask_annotator.annotate(scene=image_np.copy(), detections=detections)

    # Add boxes on top
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    return annotated


def visualize_quadrilaterals(image_np: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """
    Visualize quadrilaterals on the image.
    """
    annotated = image_np.copy()

    # Draw quadrilaterals
    for i, quad in enumerate(quads):
        if quad is not None:
            quad_int = quad.astype(np.int32).reshape((-1, 1, 2))
            # Use yellow for quads
            cv2.polylines(
                annotated,
                [quad_int],
                isClosed=True,
                color=(0, 255, 255),
                thickness=3,
            )
            # Add corner markers
            for pt in quad:
                cv2.circle(annotated, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

    return annotated
