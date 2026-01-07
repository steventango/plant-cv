import base64
import io
import json
import logging

import cv2
import numpy as np
import requests
from PIL import Image

from app.plant.stats import analyze_plant_mask
from app.pot.quad import mask_to_quadrilateral
from app.pot.warp import warp_quad_to_square

logger = logging.getLogger(__name__)


def encode_image(image: Image.Image | np.ndarray) -> str:
    """Encode image to base64 string."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def call_sam3_api(
    image: Image.Image,
    endpoint: str = "detect",
    text_prompt: str | None = None,
    state: str | None = None,
    threshold: float | None = None,
    server_url: str = "http://sam3:8805/predict",
):
    """
    Call SAM3 API for detection and tracking.

    Args:
        image: PIL Image to be processed
        endpoint: Either "detect" or "propagate"
        text_prompt: Text prompt for detection (required for detect)
        state: State from previous call (required for propagate)
        server_url: URL of the SAM3 API server

    Returns:
        dict with keys:
            - state: Serialized state for tracking
            - masks: List of mask info dicts
    """
    payload = {
        "endpoint": endpoint,
        "image_data": encode_image(image),
    }

    if threshold is not None:
        payload["threshold"] = threshold

    if endpoint == "detect":
        if text_prompt:
            payload["text_prompt"] = text_prompt
    elif endpoint == "propagate":
        if state:
            payload["state"] = state

    response = requests.post(server_url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def order_masks_row_major(masks):
    """Order masks in row-major order (top-to-bottom, left-to-right)."""
    if not masks:
        return []

    # Extract centers or top-left for sorting
    boxes = np.array([m["box"] for m in masks])
    # centers
    xs = (boxes[:, 0] + boxes[:, 2]) / 2
    ys = (boxes[:, 1] + boxes[:, 3]) / 2

    # Row-major: Primary Y, Secondary X.
    # lexsort sorts by last element first.
    indices = np.lexsort((xs, ys))
    return [masks[i] for i in indices]


def associate_plants_to_pots(plant_masks, pot_masks):
    """
    Associate each plant with a pot ID.
    Returns a mapping of plant_object_id -> pot_object_id.
    """
    associations = {}
    if not plant_masks or not pot_masks:
        return associations

    for plant in plant_masks:
        p_box = plant["box"]
        p_center = ((p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2)

        best_pot_id = None
        # Could use IoU, but center-in-box is usually sufficient for plant-in-pot matching
        for pot in pot_masks:
            b = pot["box"]
            # Check if plant center is inside pot box
            if b[0] <= p_center[0] <= b[2] and b[1] <= p_center[1] <= b[3]:
                best_pot_id = pot["object_id"]
                break

        if best_pot_id is not None:
            associations[str(plant["object_id"])] = best_pot_id

    return associations


def wrap_state(sam3_state, id_map):
    """Wrap SAM3 state and ID map into a single base64 string."""
    data = {"sam3_state": sam3_state, "id_map": id_map}
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def unwrap_state(wrapped_state):
    """Unwrap SAM3 state and ID map from a base64 string."""
    if not wrapped_state:
        return None, {}
    try:
        # Check if it's our JSON wrapper or raw SAM3 state
        decoded = base64.b64decode(wrapped_state).decode("utf-8")
        data = json.loads(decoded)
        if isinstance(data, dict) and "sam3_state" in data:
            return data["sam3_state"], data.get("id_map", {})
    except Exception:
        # If decoding fails, assume it's a raw SAM3 state
        pass
    return wrapped_state, {}


def apply_id_mapping(masks, id_map):
    """Apply ID mapping to masks. Assign new IDs if not present."""
    remapped_masks = []
    # Find next available ID for new objects
    next_id = max(id_map.values()) + 1 if id_map else 1

    for m in masks:
        old_id = str(m["object_id"])
        if old_id not in id_map:
            id_map[old_id] = next_id
            next_id += 1

        # Create a copy and update ID
        m_new = m.copy()
        m_new["object_id"] = id_map[old_id]
        remapped_masks.append(m_new)

    return remapped_masks, id_map


def process_pot_stats(image_np, pot_masks, plant_masks, associations):
    """
    Compute statistics for each plant in its respective pot.
    """
    h, w = image_np.shape[:2]
    stats_dict = {}

    # Compute quads for all pots
    # We can use mask_to_quadrilateral for each pot mask
    for pot in pot_masks:
        pot_id = pot["object_id"]
        plant_id = None
        # Find which plant (if any) is in this pot
        # The associations dict has plant_id -> pot_id
        for p_id_str, pt_id in associations.items():
            if pt_id == pot_id:
                plant_id = int(p_id_str)
                break

        if plant_id is None:
            stats_dict[str(pot_id)] = None
            continue

        # Get the plant mask info
        plant_info = next((p for p in plant_masks if p["object_id"] == plant_id), None)
        if not plant_info:
            stats_dict[str(pot_id)] = None
            continue

        try:
            # 1. Get Pot Quad
            pot_contour = np.array(pot["contour"], dtype=np.int32)
            pot_mask_single = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(pot_mask_single, [pot_contour], 1)
            quad = mask_to_quadrilateral(pot_mask_single)

            # 2. Warp Pot Image
            output_size = 224  # Target 224x224 as per feedback
            warped_pot, H = warp_quad_to_square(image_np, quad, output_size=output_size)

            # 3. Get Plant Mask and Warp it
            # Use all refined contours if available
            p_contours = plant_info.get("contours")
            if p_contours:
                plant_mask_single = np.zeros((h, w), dtype=np.uint8)
                for c in p_contours:
                    cv2.fillPoly(plant_mask_single, [np.array(c, dtype=np.int32)], 255)
            else:
                plant_contour = np.array(plant_info["contour"], dtype=np.int32)
                plant_mask_single = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(plant_mask_single, [plant_contour], 255)

            warped_plant_mask = cv2.warpPerspective(
                plant_mask_single,
                H,
                (output_size, output_size),
                flags=cv2.INTER_NEAREST,
            )

            # 4. Analyze
            plant_stats, _ = analyze_plant_mask(warped_pot, warped_plant_mask)

            # Include base64 warped image in stats
            import io

            from PIL import Image

            pill_warped = Image.fromarray(warped_pot)
            buffered = io.BytesIO()
            pill_warped.save(buffered, format="JPEG")
            warped_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            plant_stats["warped_image"] = warped_b64
            stats_dict[str(pot_id)] = plant_stats

        except Exception as e:
            logger.error(f"Error processing stats for pot {pot_id}: {e}")
            stats_dict[str(pot_id)] = {"error": str(e)}

    return stats_dict


def refine_pot_masks_with_plants(pot_masks, plant_masks, associations, image_shape):
    """
    Refine pot masks by subtracting associated plant masks.
    This prevents plant leaves from being treated as part of the pot.
    """
    h, w = image_shape[:2]
    plant_map = {str(p["object_id"]): p for p in plant_masks}

    # Invert associations for easier lookup: pot_id -> plant_id
    pot_to_plant = {}
    for p_id_str, pot_id in associations.items():
        pot_to_plant[pot_id] = p_id_str

    for pot in pot_masks:
        pot_id = pot["object_id"]
        plant_id_str = pot_to_plant.get(pot_id)

        if not plant_id_str:
            continue

        plant = plant_map.get(plant_id_str)
        if not plant:
            continue

        if "contour" not in pot or not pot["contour"]:
            continue

        # Create pot mask
        pot_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pot_mask, [np.array(pot["contour"], dtype=np.int32)], 255)

        # Create plant mask
        plant_mask = np.zeros((h, w), dtype=np.uint8)
        if "contours" in plant and plant["contours"]:
            polys = []
            for c in plant["contours"]:
                poly = np.array(c, dtype=np.int32)
                if poly.ndim == 2 and poly.shape[0] > 0:
                    polys.append(poly)
            if polys:
                cv2.fillPoly(plant_mask, polys, 255)
        elif "contour" in plant and plant["contour"]:
            poly = np.array(plant["contour"], dtype=np.int32)
            if poly.ndim == 2 and poly.shape[0] > 0:
                cv2.fillPoly(plant_mask, [poly], 255)

        # Subtract plant from pot
        refined_mask = cv2.bitwise_and(pot_mask, cv2.bitwise_not(plant_mask))

        # Determine new largest contour for pot
        if np.sum(refined_mask) > 0:
            contours, _ = cv2.findContours(
                refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)

                # Check if the refined area is too small compared to original (excessive subtraction)
                if cv2.contourArea(largest) < 0.3 * cv2.contourArea(
                    np.array(pot["contour"], dtype=np.int32)
                ):
                    continue

                # Use convex hull to maintain pot-like shape and avoid artifacts (triangles/slivers)
                hull = cv2.convexHull(largest)

                # Update pot info if hull is valid
                if len(hull) >= 3:
                    pot["contour"] = hull.reshape(-1, 2).tolist()
                    x, y, wb, hb = cv2.boundingRect(hull)
                    pot["box"] = [float(x), float(y), float(x + wb), float(y + hb)]
