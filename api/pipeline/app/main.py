import base64
import io

import numpy as np
import supervision as sv
from flask import Flask, jsonify, request
from pot.routes import pot_bp
from utils import call_segment_anything_api, decode_image, encode_image

app = Flask(__name__)
app.register_blueprint(pot_bp)



@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


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
