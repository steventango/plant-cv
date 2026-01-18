import logging

import numpy as np
from flask import Blueprint, jsonify, request

from app.pot.detect import detect_pots_sam3
from app.utils import (
    call_sam3_api,
    decode_image,
    process_pipeline_outputs,
)

logger = logging.getLogger(__name__)

detect_blueprint = Blueprint("detect", __name__, url_prefix="/pipeline")


@detect_blueprint.route("/detect", methods=["POST"])
def detect():
    """
    detect tracking for pot and plant.

    Input JSON:
        {
            "image_data": "base64_encoded_image"
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
        state_in = data.get("state", {})
        image_data = data["image_data"]
        image = decode_image(image_data)

        cleaning_state = state_in.get("cleaning_state")

        _, _, _, pot_state_from_sam3, sorted_pot_masks, _ = detect_pots_sam3(image)

        # For the plant SAM3 request, we want to run mask reconditioning every frame
        plant_result = call_sam3_api(
            image,
            endpoint="detect",
            text_prompt="plant",
            recondition_on_trk_masks=False,
            recondition_every_nth_frame=1,
            score_threshold_detection=0.165,
            high_conf_thresh=0.3,
            high_iou_thresh=0.5,
            new_det_thresh=0.5,
            det_nms_thresh=0.5,
            assoc_iou_thresh=0.1,
            trk_assoc_iou_thresh=0.5,
            suppress_overlapping_based_on_recent_occlusion_threshold=1.0,
        )
        plant_state_from_sam3 = plant_result["session_id"]
        plant_masks = plant_result.get("masks", [])

        # Process all outputs using the shared utility
        response = process_pipeline_outputs(
            np.array(image),
            plant_masks,
            sorted_pot_masks,
            sam3_session_id=pot_state_from_sam3,
            cleaning_state=cleaning_state,
        )
        response["state"]["plant_state"] = plant_state_from_sam3

        logger.info(
            f"detect: pot_masks={len(response['pot_masks'])}, ordered_pot_ids={len(response['ordered_pot_ids'])}"
        )
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in pipeline detect: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
