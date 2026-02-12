import logging

import numpy as np
from flask import Blueprint, jsonify, request

import cv2
from app.pot.detect import filter_pot_masks
from app.utils import (
    associate_plants_to_pots,
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
        state_in = data.get("state", {})

        image_data = data["image_data"]
        pot_state = state_in.get("pot_state")
        plant_state = state_in.get("plant_state")
        cleaning_state = state_in.get("cleaning_state")
        image = decode_image(image_data)
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        response = {}
        plant_masks = []
        pot_session_id, id_map, p_masks_raw_old = unwrap_state(pot_state)

        if pot_state:
            # Get options from request
            enable_outgrowth_lock = data.get("enable_outgrowth_lock", True)
            outgrowth_pixel_threshold = data.get("outgrowth_pixel_threshold", 50)

            # --- POT TRACKING ---
            # Forward extra SAM3 params from request (overriding defaults if provided)
            pot_params = {"score_threshold_detection": 0.15}
            for k, v in data.items():
                if k not in [
                    "image_data",
                    "state",
                    "enable_outgrowth_lock",
                    "outgrowth_pixel_threshold",
                ]:
                    pot_params[k] = v

            pot_result = call_sam3_api(
                image,
                endpoint="propagate",
                state=pot_session_id,
                **pot_params,
            )
            p_masks_raw_new = filter_pot_masks(pot_result.get("masks", []), image_np)

            # Mapping new masks back to stable IDs and handling outgrowth
            old_mask_lookup = (
                {m["object_id"]: m for m in p_masks_raw_old} if p_masks_raw_old else {}
            )

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
                    score_threshold_detection=0.165,
                    high_conf_thresh=0.165,
                    high_iou_thresh=0.0001,
                    new_det_thresh=0.2,
                    det_nms_thresh=0.01,
                    assoc_iou_thresh=0.0001,
                    trk_assoc_iou_thresh=0.0001,
                )
                plant_masks = plant_result.get("masks", [])
                plant_session_id = plant_result.get("session_id")

            # --- OUTGROWTH CHECK (Plant t vs Pot t-1) ---
            # We need to temporarily associate current plants with previous pots to check containment
            # Note: process_pipeline_outputs will do the FINAL association for the response
            # p_masks_raw_old has stable IDs, plant_masks has SAM3 IDs
            newly_outgrown_stable_ids = set()
            if enable_outgrowth_lock:
                temp_associations = associate_plants_to_pots(
                    plant_masks, p_masks_raw_old
                )

                # Record which stable IDs have outgrown based on Plant(t) vs Pot(t-1)
                for p_sam3_id_str, stable_pot_id in temp_associations.items():
                    plant_mask_obj = next(
                        (
                            m
                            for m in plant_masks
                            if str(m["object_id"]) == p_sam3_id_str
                        ),
                        None,
                    )
                    pot_mask_old_obj = old_mask_lookup.get(stable_pot_id)

                    if plant_mask_obj and pot_mask_old_obj:
                        # Create binary masks for comparison
                        p_mask = np.zeros((h, w), dtype=np.uint8)
                        if "contour" in plant_mask_obj:
                            cv2.fillPoly(
                                p_mask,
                                [np.array(plant_mask_obj["contour"], dtype=np.int32)],
                                1,
                            )

                        pot_mask_old = np.zeros((h, w), dtype=np.uint8)
                        if "contour" in pot_mask_old_obj:
                            cv2.fillPoly(
                                pot_mask_old,
                                [np.array(pot_mask_old_obj["contour"], dtype=np.int32)],
                                1,
                            )

                        # Check if plant (t) is contained in pot (t-1)
                        outside_pixels = np.sum((p_mask > 0) & (pot_mask_old == 0))
                        if outside_pixels > outgrowth_pixel_threshold:
                            newly_outgrown_stable_ids.add(stable_pot_id)
                            logger.info(
                                f"Pot {stable_pot_id} detection: Plant at t outgrew Pot at t-1. Outside pixels: {outside_pixels}"
                            )

            new_mask_by_stable_id = {}
            for m_new in p_masks_raw_new:
                sam3_id = str(m_new.get("object_id"))
                stable_id = id_map.get(sam3_id)
                if stable_id is not None:
                    new_mask_by_stable_id[stable_id] = m_new

            final_p_masks_raw = []
            # We want to maintain all known pots
            for stable_id, old_mask in old_mask_lookup.items():
                was_outgrown = old_mask.get("is_outgrown", False)
                is_now_outgrown = enable_outgrowth_lock and (
                    was_outgrown or (stable_id in newly_outgrown_stable_ids)
                )

                if is_now_outgrown:
                    # If outgrown, keep the old mask (lock it)
                    locked_mask = old_mask.copy()
                    locked_mask["is_outgrown"] = True
                    final_p_masks_raw.append(locked_mask)
                    logger.debug(f"Pot {stable_id} is LOCKED (outgrown)")
                else:
                    new_mask = new_mask_by_stable_id.get(stable_id)
                    if new_mask:
                        # Update with new mask from SAM3
                        new_mask_updated = new_mask.copy()
                        new_mask_updated["object_id"] = stable_id
                        new_mask_updated["is_outgrown"] = False
                        final_p_masks_raw.append(new_mask_updated)
                    else:
                        # Lost track or filtered out: keep old one as fallback
                        final_p_masks_raw.append(old_mask)

            # Process all outputs using the shared utility
            response = process_pipeline_outputs(
                image_np,
                plant_masks,
                final_p_masks_raw,
                id_map=id_map,
                sam3_session_id=pot_session_id,
                cleaning_state=cleaning_state,
            )
            response["state"]["plant_state"] = plant_session_id
            response["debug_plant_mask_sam3_count"] = len(plant_masks)
            response["debug_pot_masks_count"] = len(final_p_masks_raw)

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in pipeline track: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
