import logging

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from app.plant.segment import preprocess_for_otsu, refine_mask_with_otsu
from app.pot.detect import (
    filter_by_areas,
    filter_by_aspect_ratio,
    filter_clipped_pots,
)
from app.pot.visualize import visualize_pipeline_tracking
from app.utils import (
    apply_id_mapping,
    associate_plants_to_pots,
    call_sam3_api,
    decode_image,
    encode_image,
    order_masks_row_major,
    process_pot_stats,
    refine_pot_masks_with_plants,
    unwrap_state,
    wrap_state,
)

logger = logging.getLogger(__name__)

propagate_blueprint = Blueprint("propagate", __name__, url_prefix="/pipeline")


@propagate_blueprint.route("/propagate", methods=["POST"])
def propagate():
    """
    Track pot and plant in the next frame.

    Input JSON:
        {
            "image_data": "base64_encoded_image",
            "pot_state": "serialized_state",
            "plant_state": "serialized_state"
        }

    Output JSON:
        {
            "pot_state": "serialized_state",
            "plant_state": "serialized_state",
            "pot_masks": [...],
            "plant_masks": [...]
        }
    """
    try:
        data = request.json
        image_data = data["image_data"]
        pot_state = data.get("pot_state")
        plant_state = data.get("plant_state")
        image = decode_image(image_data)
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        response = {}
        plant_masks = []
        pot_masks = []

        sam3_pot_state, id_map = unwrap_state(pot_state)

        if pot_state:
            pot_result = call_sam3_api(
                image, endpoint="propagate", state=sam3_pot_state, score_threshold_detection=0.15
            )

            # Apply ID mapping
            p_masks_raw = pot_result.get("masks", [])

            # --- FILTERING START ---
            if p_masks_raw:
                p_boxes = np.array([m["box"] for m in p_masks_raw])
                p_confs = np.array([m["score"] for m in p_masks_raw])
                p_widths = p_boxes[:, 2] - p_boxes[:, 0]
                p_heights = p_boxes[:, 3] - p_boxes[:, 1]
                p_areas = p_widths * p_heights

                # Aspect ratio filter
                ar_mask = filter_by_aspect_ratio(p_boxes, low=0.5, high=1.5)
                # Clipped pot filter
                edge_mask = filter_clipped_pots(
                    p_boxes[ar_mask], p_areas[ar_mask], image_np.shape, margin=1
                )
                # Area filter
                inlier_mask = filter_by_areas(
                    p_boxes[ar_mask][edge_mask],
                    p_areas[ar_mask][edge_mask],
                    image_np,
                    confidences=p_confs[ar_mask][edge_mask],
                )

                # Combine masks
                final_keep = np.zeros(len(p_masks_raw), dtype=bool)
                indices_ar = np.where(ar_mask)[0]
                indices_edge = indices_ar[edge_mask]
                final_keep[indices_edge[inlier_mask]] = True

                # Filter p_masks_raw
                p_masks_raw = [
                    p_masks_raw[i] for i in range(len(p_masks_raw)) if final_keep[i]
                ]

            # --- FILTERING END ---

            remapped_pot_masks, updated_id_map = apply_id_mapping(p_masks_raw, id_map)
            pot_masks = remapped_pot_masks

            response["pot_state"] = wrap_state(pot_result.get("session_id"), updated_id_map)

        if plant_state:
            # For the plant SAM3 request, we want to run mask reconditioning every frame
            plant_result = call_sam3_api(
                image,
                endpoint="propagate",
                state=plant_state,
                recondition_on_trk_masks=True,
                recondition_every_nth_frame=1,
                score_threshold_detection=0.15,
                high_conf_thresh=0.15,
                high_iou_thresh=0.0001,
                new_det_thresh=0.7,
                det_nms_thresh=0.01,
                assoc_iou_thresh=0.0001,
                trk_assoc_iou_thresh=0.0001,
            )
            plant_masks = plant_result.get("masks", [])
            response["debug_plant_mask_sam3_count"] = len(plant_masks)
            response["plant_state"] = plant_result.get("session_id")

        response["debug_pot_masks_count"] = len(pot_masks)

        # Refine plant masks (if any) - requires pot_masks for quad-based expansion
        if plant_masks and pot_masks:
            # First pass: associate plants to pots
            temp_associations = associate_plants_to_pots(plant_masks, pot_masks)

            preprocessed_gray = preprocess_for_otsu(image_np)
            refined_plant_masks = []
            for p in plant_masks:
                plant_id = str(p["object_id"])
                pot_id = temp_associations.get(plant_id)

                # Remove plants not associated with a pot
                if pot_id is None:
                    continue

                if "contour" in p and p["contour"]:
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(m, [np.array(p["contour"], dtype=np.int32)], 1)
                    orig_area = np.sum(m)

                    refined_m = refine_mask_with_otsu(m, preprocessed_gray)

                    refined_area = np.sum(refined_m > 0)

                    if refined_area > 0.05 * orig_area:
                        contours, _ = cv2.findContours(
                            refined_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            p["contours"] = [
                                c.reshape(-1, 2).tolist() for c in contours
                            ]
                            largest_contour = max(contours, key=cv2.contourArea)
                            p["contour"] = largest_contour.reshape(-1, 2).tolist()

                            all_pts = np.concatenate(
                                [c.reshape(-1, 2) for c in contours]
                            )
                            x, y, w_b, h_b = cv2.boundingRect(all_pts)
                            p["box"] = [
                                float(x),
                                float(y),
                                float(x + w_b),
                                float(y + h_b),
                            ]

                refined_plant_masks.append(p)

            plant_masks = refined_plant_masks

        response["debug_refined_plant_masks_count"] = len(plant_masks)

        # Associate and Refine Pot Masks
        if plant_masks and pot_masks:
            associations = associate_plants_to_pots(plant_masks, pot_masks)

            # Refine pot masks by subtracting plant masks
            refine_pot_masks_with_plants(
                pot_masks,
                plant_masks,
                associations,
                image_np.shape,
            )

            # Re-sort pots after refinement for stability
            ordered_pot_masks = order_masks_row_major(pot_masks)
            response["pot_masks"] = ordered_pot_masks
            response["ordered_pot_ids"] = [m["object_id"] for m in ordered_pot_masks]
            response["plant_masks"] = plant_masks

            plant_stats = process_pot_stats(
                image_np,
                ordered_pot_masks,
                plant_masks,
                associations,
            )

            response["associations"] = associations
            response["plant_stats"] = plant_stats
            response["visualization_data"] = encode_image(
                visualize_pipeline_tracking(
                    image_np,
                    plant_masks,
                    ordered_pot_masks,
                    associations,
                    plant_stats=plant_stats,
                )
            )

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in pipeline track: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
