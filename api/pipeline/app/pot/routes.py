import base64
import io

import numpy as np
from flask import Blueprint, jsonify, request
from utils import call_segment_anything_api, decode_image, encode_image

from pot.detect import detect_pots
from pot.quad import compute_quadrilaterals
from pot.warp import warp_pots
from pot.visualization import (
    visualize_pot_detections,
    visualize_pot_segmentation,
    visualize_quadrilaterals,
)

pot_blueprint = Blueprint("pot", __name__, url_prefix="/pot")


@pot_blueprint.route("/detect", methods=["POST"])
def detect():
    """
    Detect pot boxes in an image.

    Input JSON:
        {
            "image_data": "base64_encoded_image",
            "text_prompt": "pot" (optional),
            "threshold": 0.03 (optional),
            "visualize": false (optional)
        }

    Output JSON:
        {
            "boxes": [[x1, y1, x2, y2], ...],
            "confidences": [0.95, ...],
            "class_names": ["pot", ...],
            "visualization": "base64_encoded_image" (if visualize=true)
        }
    """
    try:
        data = request.json
        image = decode_image(data["image_data"])
        text_prompt = data.get("text_prompt", "pot")
        threshold = data.get("threshold", 0.03)
        visualize = data.get("visualize", False)

        boxes, confidences, class_names = detect_pots(
            image, text_prompt=text_prompt, threshold=threshold
        )

        response = {
            "boxes": boxes.tolist(),
            "confidences": confidences.tolist(),
            "class_names": class_names.tolist()
            if isinstance(class_names, np.ndarray)
            else class_names,
        }

        if visualize and len(boxes) > 0:
            image_np = np.array(image)
            annotated = visualize_pot_detections(image_np, boxes, confidences)
            response["visualization"] = encode_image(annotated)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pot_blueprint.route("/segment", methods=["POST"])
def segment():
    """
    Segment pot boxes in an image.

    Input JSON:
        {
            "image_data": "base64_encoded_image",
            "boxes": [[x1, y1, x2, y2], ...],
            "visualize": false (optional)
        }

    Output JSON:
        {
            "masks": "base64_encoded_masks_array",  # (M, H, W) uint8
            "scores": [0.95, ...],
            "visualization": "base64_encoded_image" (if visualize=true)
        }
    """
    try:
        data = request.json
        image = decode_image(data["image_data"])
        boxes = np.array(data["boxes"])
        visualize = data.get("visualize", False)

        masks, scores = call_segment_anything_api(image, boxes)

        # Encode masks as base64 numpy array
        # TODO: maybe return contours instead
        masks_bytes = io.BytesIO()
        np.save(masks_bytes, masks)
        masks_b64 = base64.b64encode(masks_bytes.getvalue()).decode("utf-8")

        response = {
            "masks": masks_b64,
            "scores": scores.tolist(),
        }

        if visualize and len(masks) > 0:
            image_np = np.array(image)
            annotated = visualize_pot_segmentation(image_np, boxes, masks)
            response["visualization"] = encode_image(annotated)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pot_blueprint.route("/quad", methods=["POST"])
def quad():
    """
    Compute quadrilaterals from masks.

    Input JSON:
        {
            "masks": "base64_encoded_masks_array",  # (M, H, W) uint8
            "image_data": "base64_encoded_image" (optional, for visualization),
            "visualize": false (optional)
        }

    Output JSON:
        {
            "quadrilaterals": [[[x, y], [x, y], [x, y], [x, y]], ...],
            "visualization": "base64_encoded_image" (if visualize=true)
        }
    """
    try:
        data = request.json

        # Decode masks
        masks_bytes = io.BytesIO(base64.b64decode(data["masks"]))
        masks = np.load(masks_bytes)

        visualize = data.get("visualize", False)

        quads = compute_quadrilaterals(masks)

        response = {
            "quadrilaterals": quads.tolist(),
        }

        if visualize and data.get("image_data"):
            image = decode_image(data["image_data"])
            image_np = np.array(image)
            annotated = visualize_quadrilaterals(image_np, quads)
            response["visualization"] = encode_image(annotated)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pot_blueprint.route("/warp", methods=["POST"])
def warp():
    """
    Warp pot regions to square images.

    Input JSON:
        {
            "image_data": "base64_encoded_image",
            "quadrilaterals": [[[x, y], [x, y], [x, y], [x, y]], ...],
            "margin": 0.25 (optional),
            "output_size": null (optional)
        }

    Output JSON:
        {
            "warped_images": ["base64_encoded_image", ...],
            "homographies": [[[...], [...], [...]], ...]
        }
    """
    try:
        data = request.json
        image = decode_image(data["image_data"])
        quads = np.array(data["quadrilaterals"])
        margin = data.get("margin", 0.25)
        output_size = data.get("output_size")

        results = warp_pots(image, quads, margin=margin, output_size=output_size)

        warped_images = []
        homographies = []
        for warped, H in results:
            if warped is not None:
                warped_images.append(encode_image(warped))
                homographies.append(H.tolist())
            else:
                warped_images.append(None)
                homographies.append(None)

        return jsonify(
            {
                "warped_images": warped_images,
                "homographies": homographies,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
