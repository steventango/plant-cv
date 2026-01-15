import logging

import numpy as np
from flask import Blueprint, jsonify, request

from app.pot.detect import filter_pot_masks
from app.utils import (
    call_sam3_api,
    decode_image,
    process_pipeline_outputs,
    unwrap_state,
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
        sam3_pot_state, id_map = unwrap_state(pot_state)

        if pot_state:
            pot_result = call_sam3_api(
                image,
                endpoint="propagate",
                state=sam3_pot_state,
                score_threshold_detection=0.15,
            )

            # Filter recomputed masks
            p_masks_raw = filter_pot_masks(pot_result.get("masks", []), image_np)

            # --- PLANT DETECTION ---
            plant_masks = []
            plant_session_id = None
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
                plant_session_id = plant_result.get("session_id")

            # Process all outputs using the shared utility
            response = process_pipeline_outputs(
                image_np,
                plant_masks,
                p_masks_raw,
                id_map=id_map,
                sam3_session_id=pot_result.get("session_id"),
            )
            response["plant_state"] = plant_session_id
            response["debug_plant_mask_sam3_count"] = len(plant_masks)
            response["debug_pot_masks_count"] = len(p_masks_raw)

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in pipeline track: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
