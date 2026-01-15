import cv2
import numpy as np
import supervision as sv
from app.pot.quad import mask_to_quadrilateral


def visualize_pipeline_tracking(
    image_np: np.ndarray,
    plant_masks_info: list,
    pot_masks_info: list,
    associations: dict,
    plant_stats: dict | None = None,
) -> np.ndarray:
    """
    Visualize plant masks and pot quadrilaterals with associations.
    Plants are colored based on the pot they belong to.
    """
    h, w = image_np.shape[:2]
    annotated = image_np.copy()

    if not pot_masks_info:
        return annotated

    # 1. Visualize Pot Quadrilaterals with Supervision
    if pot_masks_info:
        pot_boxes = []
        pot_ids = []
        pot_quads = []

        for pot in pot_masks_info:
            if "contour" in pot and pot["contour"]:
                contour_array = np.array(pot["contour"], dtype=np.int32)
                # Skip if contour is empty or has too few points
                if contour_array.size == 0 or len(contour_array) < 3:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [contour_array], 1)
                try:
                    quad = mask_to_quadrilateral(mask)  # (4, 2)
                    pot_quads.append(quad)
                    pot_boxes.append(pot["box"])
                    pot_ids.append(pot["object_id"])
                except Exception:
                    continue

        if pot_quads:
            pot_detections = sv.Detections(
                xyxy=np.array(pot_boxes),
                class_id=np.array(pot_ids),
                data={"xyxyxyxy": np.array(pot_quads)},
            )

            # Use OrientedBoxAnnotator for quads
            obb_annotator = sv.OrientedBoxAnnotator(thickness=2)
            annotated = obb_annotator.annotate(
                scene=annotated, detections=pot_detections
            )

            # Add labels for pots
            pot_label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=2)
            pot_labels = [f"{pid}" for pid in pot_ids]
            annotated = pot_label_annotator.annotate(
                scene=annotated, detections=pot_detections, labels=pot_labels
            )

    # 2. Visualize Plant Masks with Supervision
    if plant_masks_info:
        # Create masks manually to handle multiple contours per plant
        masks_list = []
        for p in plant_masks_info:
            mask = np.zeros((h, w), dtype=np.uint8)
            try:
                if "contours" in p and p["contours"]:
                    # Filter for valid polygons (at least 3 points is safer for fillPoly, but simple points might pass if handled as lines)
                    # We'll rely on numpy shape check.
                    polys = []
                    for c in p["contours"]:
                        poly = np.array(c, dtype=np.int32)
                        # Ensure it's (N, 2) and N > 0
                        if poly.ndim == 2 and poly.shape[0] > 0 and poly.shape[1] == 2:
                            polys.append(poly)
                    if polys:
                        cv2.fillPoly(mask, polys, 1)
                elif "contour" in p:
                    poly = np.array(p["contour"], dtype=np.int32)
                    if poly.ndim == 2 and poly.shape[0] > 0:
                        cv2.fillPoly(mask, [poly], 1)
            except Exception:
                # Log or ignore error to prevent crash
                # In this context we can't easily log w/o logger, but preventing crash is key
                pass
            masks_list.append(mask)

        plant_masks = np.array(masks_list).astype(bool)
        plant_boxes = np.array([p["box"] for p in plant_masks_info])
        plant_ids = np.array([p["object_id"] for p in plant_masks_info])

        # Use associated pot_id as class_id for consistent coloring
        class_ids = []
        labels = []
        for p_id in plant_ids:
            pot_id = associations.get(str(p_id))
            if pot_id is not None:
                class_ids.append(int(pot_id))
                area_text = ""
                if plant_stats:
                    pot_stat = plant_stats.get(str(pot_id))
                    if pot_stat and "area" in pot_stat:
                        area = pot_stat["area"]
                        area_text = f" {area:.0f} mm^2"
                labels.append(area_text)
            else:
                # Fallback to a high ID to avoid overlap with pot IDs if possible
                class_ids.append(0)
                labels.append("NULL")

        detections = sv.Detections(
            xyxy=plant_boxes,
            mask=plant_masks,
            class_id=np.array(class_ids),
        )

        mask_annotator = sv.MaskAnnotator()
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

        label_annotator = sv.LabelAnnotator(text_scale=0.4)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

    return annotated
