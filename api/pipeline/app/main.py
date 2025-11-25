import base64
import io

import cv2
import numpy as np
import supervision as sv
from flask import Flask, jsonify, request
from PIL import Image
from pot.detect import detect_pots
from pot.quad import compute_quadrilaterals
from pot.segment import segment_pot_boxes
from pot.warp import warp_pots

app = Flask(__name__)


def decode_image(image_data):
    """Decode base64 image to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(image_np):
    """Encode numpy image to base64 JPEG."""
    image = Image.fromarray(image_np.astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/pot/detect", methods=["POST"])
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

            ids = np.arange(len(boxes), dtype=int)
            # Create supervision detections
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=ids,
                tracker_id=ids,
            )

            # Annotate with boxes
            box_annotator = sv.BoxAnnotator()
            annotated = box_annotator.annotate(
                scene=image_np.copy(), detections=detections
            )

            # Annotate with labels (ID and confidence)
            label_annotator = sv.LabelAnnotator()
            labels = [f"#{tid} {conf:.2f}" for tid, conf in zip(ids, confidences)]
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

            response["visualization"] = encode_image(annotated)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pot/segment", methods=["POST"])
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

        masks, scores = segment_pot_boxes(image, boxes)

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

            # Create supervision detections with masks
            detections = sv.Detections(
                xyxy=boxes, mask=masks.astype(bool), class_id=np.arange(len(masks))
            )

            # Annotate with masks
            mask_annotator = sv.MaskAnnotator()
            annotated = mask_annotator.annotate(
                scene=image_np.copy(), detections=detections
            )

            # Add boxes on top
            box_annotator = sv.BoxAnnotator()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)

            response["visualization"] = encode_image(annotated)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pot/quad", methods=["POST"])
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

            # Draw quadrilaterals
            for i, quad in enumerate(quads):
                if quad is not None:
                    quad_int = quad.astype(np.int32).reshape((-1, 1, 2))
                    # Use yellow for quads
                    cv2.polylines(
                        image_np,
                        [quad_int],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=3,
                    )
                    # Add corner markers
                    for pt in quad:
                        cv2.circle(image_np, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

            response["visualization"] = encode_image(image_np)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pot/warp", methods=["POST"])
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



@app.route("/plant/stats", methods=["POST"])
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
        from .plant.stats import analyze_plant_mask

        data = request.json

        # Decode warped image
        warped_image = decode_image(data["warped_image"])
        warped_image_np = np.array(warped_image)

        # Decode mask
        mask = decode_image(data["mask"])
        mask_np = np.array(mask)

        # Convert mask to grayscale if needed
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]

        # Get optional parameters
        pot_size_mm = data.get("pot_size_mm", 60.0)
        margin = data.get("margin", 0.25)

        # Analyze plant mask
        stats = analyze_plant_mask(
            warped_image_np, mask_np, pot_size_mm=pot_size_mm, margin=margin
        )

        return jsonify({"stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
