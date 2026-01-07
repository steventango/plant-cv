import logging
import numpy as np
import cv2
from flask import Blueprint, jsonify, request
from app.utils import call_sam3_api, decode_image, encode_image
from app.pot.visualize import visualize_pipeline_tracking
from app.plant.segment import refine_mask_with_otsu
from app.utils import (
    order_masks_row_major,
    associate_plants_to_pots,
    wrap_state,
    unwrap_state,
    apply_id_mapping,
    process_pot_stats,
    refine_pot_masks_with_plants,
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

        response = {}

        if pot_state:
            sam3_state, id_map = unwrap_state(pot_state)
            pot_result = call_sam3_api(image, endpoint="propagate", state=sam3_state)

            # Apply ID mapping and then sort row-major list (consistency)
            remapped_masks, updated_id_map = apply_id_mapping(
                pot_result.get("masks", []), id_map
            )
            ordered_pot_masks = order_masks_row_major(remapped_masks)

            response["pot_state"] = wrap_state(pot_result.get("state"), updated_id_map)
            response["pot_masks"] = ordered_pot_masks
            response["ordered_pot_ids"] = [m["object_id"] for m in ordered_pot_masks]

        if plant_state:
            plant_result = call_sam3_api(image, endpoint="propagate", state=plant_state)
            plant_masks = plant_result.get("masks", [])

            # Refine plant masks
            h, w = np.array(image).shape[:2]
            image_np = np.array(image)
            for p in plant_masks:
                if "contour" in p and p["contour"]:
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(m, [np.array(p["contour"], dtype=np.int32)], 1)
                    orig_area = np.sum(m)

                    refined_m = refine_mask_with_otsu(image_np, m)
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

            response["plant_state"] = plant_result.get("state")
            response["plant_masks"] = plant_masks

        # Associate if both updated
        if "plant_masks" in response and "pot_masks" in response:
            associations = associate_plants_to_pots(
                response["plant_masks"], response["pot_masks"]
            )

            # Refine pot masks by subtracting plant masks
            refine_pot_masks_with_plants(
                response["pot_masks"],
                response["plant_masks"],
                associations,
                np.array(image).shape,
            )

            plant_stats = process_pot_stats(
                np.array(image),
                response["pot_masks"],
                response["plant_masks"],
                associations,
            )

            response["associations"] = associations
            response["plant_stats"] = plant_stats
            response["visualization_data"] = encode_image(
                visualize_pipeline_tracking(
                    np.array(image),
                    response["plant_masks"],
                    response["pot_masks"],
                    associations,
                    plant_stats=plant_stats,
                )
            )

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in pipeline track: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
