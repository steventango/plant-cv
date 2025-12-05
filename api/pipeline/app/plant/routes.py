import base64
import io

import numpy as np
from flask import Blueprint, jsonify, request
from utils import call_segment_anything_api, decode_image, encode_image

from plant.detect import (
    detect_plant,
    filter_boxes_by_area,
    filter_boxes_by_aspect_ratio,
    filter_boxes_by_centroid_in_pot,
    select_top_boxes,
)
from plant.segment import (
    filter_masks_by_area,
    refine_mask_with_otsu,
    select_best_mask,
)
from plant.stats import analyze_plant_mask
from plant.visualization import (
    visualize_plant_detections,
    visualize_plant_segmentation,
)
import logging

logger = logging.getLogger(__name__)

plant_blueprint = Blueprint("plant", __name__, url_prefix="/plant")


@plant_blueprint.route("/detect", methods=["POST"])
def plant_detect():
    """
    Detect plants in a crop image.

    Input JSON:
        {
            "image_data": "base64_encoded_image",
            "text_prompt": "plant" (optional),
            "detection_threshold": 0.05 (optional),
            "detection_text_threshold": 0.05 (optional),
            "area_ratio_threshold": 0.90 (optional),
            "visualize": false (optional)
        }

    Output JSON:
        {
            "boxes": [[x1, y1, x2, y2], ...],
            "confidences": [0.95, ...],
            "class_names": ["plant", ...],
            "visualization": "base64_encoded_image" (if visualize=true)
        }
    """
    try:
        data = request.json
        image = decode_image(data["image_data"])
        crop_np = np.array(image)
        crop_shape = crop_np.shape[:2]

        text_prompt = data.get("text_prompt", "plant")
        detection_threshold = data.get("detection_threshold", 0.21)
        detection_text_threshold = data.get("detection_text_threshold", 0.0)
        area_ratio_threshold = data.get("area_ratio_threshold", 0.90)
        aspect_ratio_threshold = data.get("aspect_ratio_threshold", 2)
        margin = data.get("margin", 0.25)
        visualize = data.get("visualize", False)

        # Step 1: Detect plants
        boxes, confidences, class_names = detect_plant(
            image,
            text_prompt=text_prompt,
            threshold=detection_threshold,
            text_threshold=detection_text_threshold,
        )

        # Step 2: Filter boxes by area
        if len(boxes) > 0:
            boxes, confidences, class_names, valid_box_mask = filter_boxes_by_area(
                boxes, confidences, class_names, crop_shape, area_ratio_threshold
            )

        # Step 3: Filter boxes by aspect ratio
        if len(boxes) > 0:
            boxes, confidences, class_names, valid_box_mask = (
                filter_boxes_by_aspect_ratio(
                    boxes, confidences, class_names, crop_shape, aspect_ratio_threshold
                )
            )

        # Step 4: Filter boxes by centroid in pot
        if len(boxes) > 0:
            boxes, confidences, class_names, valid_box_mask = (
                filter_boxes_by_centroid_in_pot(
                    boxes, confidences, class_names, crop_shape, margin
                )
            )

        # Step 5: Keep k-most confident boxes
        boxes, confidences, class_names = select_top_boxes(
            boxes, confidences, class_names, crop_shape, k=5
        )

        response = {
            "boxes": boxes.tolist(),
            "confidences": confidences.tolist(),
            "class_names": class_names.tolist()
            if isinstance(class_names, np.ndarray)
            else class_names,
        }

        if visualize and len(boxes) > 0:
            annotated = visualize_plant_detections(crop_np, boxes, confidences)
            response["visualization"] = encode_image(annotated)

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in plant_detect: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@plant_blueprint.route("/segment", methods=["POST"])
def plant_segment():
    """
    Segment plants in a crop image using detected boxes.

    Input JSON:
        {
            "image_data": "base64_encoded_image",
            "boxes": [[x1, y1, x2, y2], ...],
            "confidences": [0.1, 0.2, ...],
            "mask_near_full_threshold": 0.92 (optional),
            "mask_median_multiplier": 10.0 (optional),
            "visualize": false (optional)
        }

    Output JSON:
        {
            "success": bool,
            "mask": "base64_encoded_mask",
            "mask_score": float,
            "selected_index": int,
            "visualization": "base64_encoded_image" (if visualize=true)
        }
    """
    try:
        data = request.json
        image = decode_image(data["image_data"])
        crop_np = np.array(image)
        crop_shape = crop_np.shape[:2]

        boxes = np.array(data["boxes"])
        confidences = np.array(data["confidences"])
        mask_near_full_threshold = data.get("mask_near_full_threshold", 0.60)
        mask_median_multiplier = data.get("mask_median_multiplier", 10.0)
        visualize = data.get("visualize", False)

        result = {
            "success": False,
            "mask": None,
            "mask_score": None,
            "selected_index": None,
        }

        if len(boxes) == 0:
            result["note"] = "No boxes provided"
            return jsonify(result)

        # Step 1: Segment with SAM
        masks, scores = call_segment_anything_api(image, boxes)

        if masks.size == 0:
            result["note"] = "No mask returned from SAM"
            return jsonify(result)

        # Step 2: Filter masks by area
        valid_indices, mask_areas, area_ratios, reason_codes = filter_masks_by_area(
            masks,
            crop_shape,
            near_full_threshold=mask_near_full_threshold,
            median_multiplier=mask_median_multiplier,
        )

        # Step 3: Select best mask
        best_combined_score = None
        logger.info(f"Confidences: {confidences}")
        if len(valid_indices) == 0:
            # Fallback: keep largest non-full-size mask
            non_full_idx = [i for i in range(len(masks)) if i not in reason_codes]
            if non_full_idx:
                best_idx_original = non_full_idx[
                    int(np.argmax(area_ratios[non_full_idx]))
                ]
            else:
                # All rejected
                result["note"] = "No valid masks after relative-area filtering"
                result["reason_codes"] = {
                    k: "too large" if v == 902 else "too small"
                    for k, v in reason_codes.items()
                }

                if visualize:
                    annotated = visualize_plant_segmentation(
                        crop_np,
                        boxes,
                        confidences,
                        masks,
                        best_idx=None,
                        reason_codes=reason_codes,
                    )
                    result["visualization"] = encode_image(annotated)

                return jsonify(result)
        else:
            best_idx_original, best_combined_score = select_best_mask(
                masks, confidences, boxes, crop_shape, valid_indices
            )
            best_combined_score = float(best_combined_score)
            logger.info(f"Best combined score: {best_combined_score}")

        # Step 4: Prepare result
        best_mask = masks[best_idx_original]
        mask_binary = (best_mask > 0).astype(np.uint8) * 255

        # Refine mask with Otsu thresholding
        mask_binary = refine_mask_with_otsu(crop_np, mask_binary)

        best_mask_score = (
            float(scores[best_idx_original])
            if len(scores) > best_idx_original
            else None
        )

        # Encode mask
        mask_bytes = io.BytesIO()
        np.save(mask_bytes, mask_binary)
        mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode("utf-8")

        result["success"] = True
        result["mask"] = mask_b64
        result["mask_score"] = best_mask_score
        result["combined_score"] = best_combined_score
        result["selected_index"] = int(best_idx_original)

        if visualize:
            annotated = visualize_plant_segmentation(
                crop_np,
                boxes,
                confidences,
                masks,
                best_idx=best_idx_original,
                combined_score=best_combined_score,
            )
            result["visualization"] = encode_image(annotated)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in plant_segment: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@plant_blueprint.route("/stats", methods=["POST"])
def plant_stats():
    """
    Analyze plant mask to compute morphological attributes using PlantCV.

    Input JSON:
        {
            "warped_image": "base64_encoded_image",  # Warped pot image
            "mask": "base64_encoded_mask",  # Binary plant mask (0 or 255)
            "pot_size_mm": 60.0 (optional),
            "margin": 0.25 (optional)
        }

    Output JSON:
        {
            "stats": {
                "area": 123.45,  # mm²
                "convex_hull_area": 150.0,
                "solidity": 0.823,
                "perimeter": 45.6,  # mm
                "width": 12.3,  # mm
                "height": 15.2,
                ...
            }
        }
    """
    try:
        data = request.json

        # Decode warped image
        warped_image = decode_image(data["warped_image"])
        warped_image_np = np.array(warped_image)

        # Decode mask
        mask_bytes = io.BytesIO(base64.b64decode(data["mask"]))
        mask_np = np.load(mask_bytes)

        # Get optional parameters
        pot_size_mm = data.get("pot_size_mm", 60.0)
        margin = data.get("margin", 0.25)
        visualize = data.get("visualize", False)

        # Analyze plant mask
        stats, visualization = analyze_plant_mask(
            warped_image_np,
            mask_np,
            pot_size_mm=pot_size_mm,
            margin=margin,
            visualize=visualize,
        )

        response = {"stats": stats}
        if visualize and visualization is not None:
            response["visualization"] = encode_image(visualization)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
