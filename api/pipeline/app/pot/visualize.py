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
    target_height: int = 1080,
) -> np.ndarray:
    """
    Visualize plant masks and pot quadrilaterals with associations.
    Plants are colored based on the pot they belong to.

    Args:
        target_height: Height to resize the image to for visualization.
                      This significantly improves performance by reducing mask memory footprint.
    """
    h_orig, w_orig = image_np.shape[:2]

    # Calculate scale factor
    scale = target_height / h_orig if h_orig > target_height else 1.0

    if scale < 1.0:
        h_new = int(h_orig * scale)
        w_new = int(w_orig * scale)
        image_vis = cv2.resize(image_np, (w_new, h_new), interpolation=cv2.INTER_AREA)
    else:
        image_vis = image_np.copy()
        h_new, w_new = h_orig, w_orig

    annotated = image_vis.copy()

    if not pot_masks_info and not plant_masks_info:
        return annotated

    # Helper to scale boxes
    def scale_box(box):
        return [coord * scale for coord in box]

    # Helper to scale contours
    def scale_contour(contour):
        return (np.array(contour) * scale).astype(np.int32)

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

                mask = np.zeros((h_new, w_new), dtype=np.uint8)

                scaled_poly = scale_contour(contour_array)
                cv2.fillPoly(mask, [scaled_poly], 1)

                try:
                    quad = mask_to_quadrilateral(mask)  # (4, 2)
                    pot_quads.append(
                        quad
                    )  # Logic handles scaled mask, so quad is scaled
                    pot_boxes.append(scale_box(pot["box"]))
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
        masks_list = []

        for idx, p in enumerate(plant_masks_info):
            mask = np.zeros((h_new, w_new), dtype=np.uint8)
            try:
                if "contours" in p and p["contours"]:
                    polys = []
                    for c in p["contours"]:
                        poly = np.array(c, dtype=np.float32)  # float for scaling
                        if poly.ndim == 2 and poly.shape[0] > 0 and poly.shape[1] == 2:
                            # Scale
                            poly_scaled = (poly * scale).astype(np.int32)
                            polys.append(poly_scaled)
                    if polys:
                        cv2.fillPoly(mask, polys, 1)
                elif "contour" in p:
                    poly = np.array(p["contour"], dtype=np.float32)
                    if poly.ndim == 2 and poly.shape[0] > 0:
                        poly_scaled = (poly * scale).astype(np.int32)
                        cv2.fillPoly(mask, [poly_scaled], 1)
            except Exception:
                pass

            masks_list.append(mask)

        if masks_list:
            plant_masks = np.array(masks_list).astype(bool)
            plant_boxes = np.array([scale_box(p["box"]) for p in plant_masks_info])
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
                        if pot_stat and "clean_area" in pot_stat:
                            area = pot_stat["clean_area"]
                            current_area = pot_stat.get("area", 0)
                            if area is not None:
                                area_text = (
                                    f"{'*' if current_area != area else ''}{area:.0f}"
                                )
                            else:
                                area_text = "N/A"
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

            label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=2)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

    return annotated
