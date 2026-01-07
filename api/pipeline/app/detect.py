import logging

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from app.plant.segment import refine_mask_with_otsu
from app.pot.detect import detect_pots_sam3
from app.pot.visualize import visualize_pipeline_tracking

from app.utils import (
    associate_plants_to_pots,
    call_sam3_api,
    decode_image,
    encode_image,
    process_pot_stats,
    refine_pot_masks_with_plants,
    wrap_state,
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
        image_data = data["image_data"]
        image = decode_image(image_data)

        # detect Pot
        _, _, _, pot_state_from_sam3, p_masks = detect_pots_sam3(image)
        # detect_pots_sam3 returns filtered and ordered masks

        # detect Plant
        plant_result = call_sam3_api(image, endpoint="detect", text_prompt="plant", threshold=0.3)

        # Order pots row-major and assign new sequential IDs
        # p_masks are already ordered by detect_pots_sam3
        sorted_pot_masks = p_masks

        # Create initial ID map for pots based on row-major order
        id_map = {}
        ordered_pot_masks = []
        for i, m in enumerate(sorted_pot_masks):
            old_id = str(m["object_id"])
            new_id = i + 1
            id_map[old_id] = new_id
            m_new = m.copy()
            m_new["object_id"] = new_id
            ordered_pot_masks.append(m_new)

        # Plant masks
        plant_masks = plant_result.get("masks", [])

        # Refine plant masks
        h, w = image.size[1], image.size[0]  # PIL size is (W, H)
        image_np = np.array(image)
        for p in plant_masks:
            if "contour" in p and p["contour"]:
                m = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(p["contour"], dtype=np.int32)
                if pts.size > 0:
                    cv2.fillPoly(m, [pts], 1)

                orig_area = np.sum(m)

                refined_m = refine_mask_with_otsu(image_np, m)
                refined_area = np.sum(refined_m > 0)

                # Safety check: if refined mask is too small, it might have failed.
                # Use it only if it's at least 5% of the original mask area.
                if refined_area > 0.05 * orig_area:
                    contours, _ = cv2.findContours(
                        refined_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # Return all contours as a list of lists of [x, y]
                        # Each contour is reshaped to (-1, 2)
                        p["contours"] = [c.reshape(-1, 2).tolist() for c in contours]
                        # Compatibility: update "contour" to be the largest one for non-updated consumers
                        largest_contour = max(contours, key=cv2.contourArea)
                        p["contour"] = largest_contour.reshape(-1, 2).tolist()

                        # Update box based on all components
                        all_pts = np.concatenate([c.reshape(-1, 2) for c in contours])
                        x, y, w_b, h_b = cv2.boundingRect(all_pts)
                        p["box"] = [float(x), float(y), float(x + w_b), float(y + h_b)]

        # Associate using new IDs
        associations = associate_plants_to_pots(plant_masks, ordered_pot_masks)

        # Refine pot masks by subtracting plant masks
        refine_pot_masks_with_plants(
            ordered_pot_masks, plant_masks, associations, image_np.shape
        )

        plant_stats = process_pot_stats(
            image_np, ordered_pot_masks, plant_masks, associations
        )

        response = {
            "pot_state": wrap_state(pot_state_from_sam3, id_map),
            "plant_state": plant_result.get("state"),
            "pot_masks": ordered_pot_masks,
            "plant_masks": plant_masks,
            "associations": associations,
            "ordered_pot_ids": [m["object_id"] for m in ordered_pot_masks],
            "plant_stats": plant_stats,
            "visualization_data": encode_image(
                visualize_pipeline_tracking(
                    np.array(image),
                    plant_masks,
                    ordered_pot_masks,
                    associations,
                    plant_stats=plant_stats,
                )
            ),
        }

        logger.info(
            f"detect: pot_masks={len(ordered_pot_masks)}, ordered_pot_ids={len(response['ordered_pot_ids'])}"
        )
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in pipeline detect: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
