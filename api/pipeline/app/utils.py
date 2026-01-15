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
from app.pot.visualize import visualize_pipeline_tracking
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
    server_url: str = "http://sam3:8805/predict",
    **kwargs,
):
    """
    Call SAM3 API for detection and tracking.

    Args:
        image: PIL Image to be processed
        endpoint: Either "detect" or "propagate"
        text_prompt: Text prompt for detection (required for detect)
        state: State from previous call (required for propagate)
        server_url: URL of the SAM3 API server
        **kwargs: Additional parameters for SAM3 (e.g., threshold, recondition_every_nth_frame)

    Returns:
        dict with keys:
            - state: Serialized state for tracking
            - masks: List of mask info dicts
    """
    payload = {
        "endpoint": endpoint,
        "image_data": encode_image(image),
    }

    if "new_det_thresh" not in kwargs:
        kwargs["new_det_thresh"] = 0.5
    if "high_conf_thresh" not in kwargs:
        kwargs["high_conf_thresh"] = 0.4
    if "high_iou_thresh" not in kwargs:
        kwargs["high_iou_thresh"] = 0.4
    if "recondition_on_trk_masks" not in kwargs:
        kwargs["recondition_on_trk_masks"] = True
    if "recondition_every_nth_frame" not in kwargs:
        kwargs["recondition_every_nth_frame"] = 10000
    if "max_num_objects" not in kwargs:
        kwargs["max_num_objects"] = 100

    payload.update(kwargs)

    if endpoint == "detect":
        if text_prompt:
            payload["text_prompt"] = text_prompt
    elif endpoint == "propagate":
        if state:
            payload["session_id"] = state

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


def associate_plants_to_pots(plant_masks, pot_masks, greedy=True):
    """
    Associate plants with pots based on bounding box IOU.
    Returns a mapping of plant_object_id -> pot_object_id.

    Args:
        plant_masks: List of plant mask dictionaries.
        pot_masks: List of pot mask dictionaries.
        greedy: If True, ensures a one-to-one mapping.
                If False, assigns each plant to its best pot (many-to-one possible).
    """
    associations = {}
    if not plant_masks or not pot_masks:
        return associations

    import sys

    print(
        f"DEBUG: STARTING associate_plants_to_pots with {len(plant_masks)} plants and {len(pot_masks)} pots",
        file=sys.stderr,
        flush=True,
    )
    # Calculate all possible plant-pot IOUs
    import sys

    logger.error(
        f"DEBUG: associate_plants_to_pots: {len(plant_masks)} plants, {len(pot_masks)} pots, greedy={greedy}"
    )
    for p in pot_masks:
        logger.error(f"DEBUG: Pot {p['object_id']} box: {p['box']}")
    pairs = []
    for plant in plant_masks:
        p_box = plant["box"]
        p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])

        best_pot_id = None
        max_iou = 0.0

        for pot in pot_masks:
            b = pot["box"]
            b_area = (b[2] - b[0]) * (b[3] - b[1])

            # Calculate intersection
            ix1 = max(p_box[0], b[0])
            iy1 = max(p_box[1], b[1])
            ix2 = min(p_box[2], b[2])
            iy2 = min(p_box[3], b[3])

            iou = 0.0
            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                union_area = p_area + b_area - intersection_area
                iou = intersection_area / max(union_area, 1e-6)

            if pot["object_id"] in [15, 38]:
                logger.error(
                    f"DEBUG: Pot {pot['object_id']} (box {b}) with Plant {plant['object_id']} (box {p_box}) -> IOU: {iou:.4f}"
                )

            if iou > max_iou:
                max_iou = iou
                best_pot_id = pot["object_id"]

            if greedy and iou > 0:
                pairs.append(
                    {
                        "plant_id": str(plant["object_id"]),
                        "pot_id": pot["object_id"],
                        "iou": iou,
                    }
                )

        if not greedy and best_pot_id is not None:
            associations[str(plant["object_id"])] = best_pot_id

    if greedy:
        # Sort pairs by IOU descending
        pairs.sort(key=lambda x: x["iou"], reverse=True)

        # Greedy assignment
        used_plants = set()
        used_pots = set()

        for pair in pairs:
            p_id = pair["plant_id"]
            pt_id = pair["pot_id"]
            if p_id not in used_plants and pt_id not in used_pots:
                associations[p_id] = pt_id
                used_plants.add(p_id)
                used_pots.add(pt_id)
                if pt_id in [15, 38]:
                    logger.error(
                        f"DEBUG: Pot {pt_id} ASSIGNED to Plant {p_id} (IOU: {pair.get('iou', 'N/A'):.4f})"
                    )
            elif pt_id in [15, 38] and pt_id not in used_pots:
                logger.error(
                    f"DEBUG: Pot {pt_id} NOT ASSIGNED to Plant {p_id} (IOU: {pair.get('iou', 'N/A'):.4f}) because plant {p_id} already used"
                )

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


def refine_plant_masks(image_np, plant_masks, pot_masks):
    """
    Refine plant masks using Otsu thresholding and associate them with pots.
    """
    from app.plant.segment import preprocess_for_otsu, refine_mask_with_otsu

    if not plant_masks or not pot_masks:
        return plant_masks, {}

    h, w = image_np.shape[:2]

    # First pass: associate plants to pots (non-greedy to allow all candidates to be refined)
    temp_associations = associate_plants_to_pots(plant_masks, pot_masks, greedy=False)

    logger.debug(f"Refining {len(plant_masks)} plant masks")
    preprocessed_gray = preprocess_for_otsu(image_np)
    refined_plant_masks = []

    for p in plant_masks:
        plant_id = str(p["object_id"])
        pot_id = temp_associations.get(plant_id)

        # Remove plants not associated with a pot
        if pot_id is None:
            logger.debug(f"Plant {plant_id} not associated with pot, removing")
            continue

        if "contour" in p and p["contour"]:
            m = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(p["contour"], dtype=np.int32)
            if pts.size > 0:
                cv2.fillPoly(m, [pts], 1)

            orig_area = np.sum(m)
            refined_m = refine_mask_with_otsu(m, preprocessed_gray)
            refined_area = np.sum(refined_m > 0)

            # Safety check: if refined mask is too small, it might have failed.
            if refined_area > 0.05 * orig_area:
                contours, _ = cv2.findContours(
                    refined_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    p["contours"] = [c.reshape(-1, 2).tolist() for c in contours]
                    largest_contour = max(contours, key=cv2.contourArea)
                    p["contour"] = largest_contour.reshape(-1, 2).tolist()

                    all_pts = np.concatenate([c.reshape(-1, 2) for c in contours])
                    x, y, w_b, h_b = cv2.boundingRect(all_pts)
                    p["box"] = [
                        float(x),
                        float(y),
                        float(x + w_b),
                        float(y + h_b),
                    ]

        refined_plant_masks.append(p)

    # Final association pass with refined masks
    final_associations = associate_plants_to_pots(refined_plant_masks, pot_masks)
    return refined_plant_masks, final_associations


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
    Refine pot masks by subtracting associated plant masks and applying morphological cleanup.

    This handles plants that outgrow the true pot boundaries by:
    1. Subtracting the plant mask from the pot mask
    2. Applying morphological closing (to fill small holes)
    3. Applying morphological opening (to remove noise/artifacts)
    4. Using convex hull for a clean pot shape
    """
    h, w = image_shape[:2]
    plant_map = {str(p["object_id"]): p for p in plant_masks}

    # Invert associations for easier lookup: pot_id -> plant_id
    pot_to_plant = {}
    for p_id_str, pot_id in associations.items():
        pot_to_plant[pot_id] = p_id_str

    # Morphological kernels
    close_kernel = np.ones((15, 15), np.uint8)  # Closing to fill small holes
    open_kernel = np.ones((7, 7), np.uint8)  # Opening to remove noise

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

        # Step 1: Subtract plant from pot
        refined_mask = cv2.bitwise_and(pot_mask, cv2.bitwise_not(plant_mask))

        # Step 2: Apply morphological closing to fill small holes created by subtraction
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, close_kernel)

        # Step 3: Apply morphological opening to remove noise/artifacts
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, open_kernel)

        # Determine new largest contour for pot
        if np.sum(refined_mask) > 0:
            contours, _ = cv2.findContours(
                refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)

                # Check if the refined area is too small compared to original (excessive subtraction)
                original_area = cv2.contourArea(
                    np.array(pot["contour"], dtype=np.int32)
                )
                if original_area > 0 and cv2.contourArea(largest) < 0.1 * original_area:
                    continue

                # Use convex hull to maintain pot-like shape and avoid artifacts
                hull = cv2.convexHull(largest)

                # Update pot info if hull is valid and has enough points/area
                if len(hull) >= 4 and cv2.contourArea(hull) > 100:
                    pot["contour"] = hull.reshape(-1, 2).tolist()
                    x, y, wb, hb = cv2.boundingRect(hull)
                    pot["box"] = [float(x), float(y), float(x + wb), float(y + hb)]


def process_pipeline_outputs(
    image_np,
    plant_masks,
    pot_masks_raw,
    id_map=None,
    sam3_session_id=None,
):
    """
    Common post-processing for both detect and propagate endpoints.
    """
    response = {}

    # 1. ID Mapping
    if id_map is not None:
        pot_masks, updated_id_map = apply_id_mapping(pot_masks_raw, id_map)
    else:
        # Initial detection case: create ID map based on row-major order
        sorted_pot_masks = order_masks_row_major(pot_masks_raw)
        updated_id_map = {}
        pot_masks = []
        for new_id, m in enumerate(sorted_pot_masks):
            old_id = str(m["object_id"])
            updated_id_map[old_id] = new_id
            m_new = m.copy()
            m_new["object_id"] = new_id
            pot_masks.append(m_new)

    if sam3_session_id is not None:
        response["pot_state"] = wrap_state(sam3_session_id, updated_id_map)

    # 2. Refine plant masks
    plant_masks, associations = refine_plant_masks(image_np, plant_masks, pot_masks)

    # 2.5 Filter plant masks to only include associated ones (ensures 1-to-1 matching in response)
    plant_masks = [p for p in plant_masks if str(p["object_id"]) in associations]

    # 3. Refine pot masks by subtracting plant masks
    refine_pot_masks_with_plants(pot_masks, plant_masks, associations, image_np.shape)

    # 4. Final ordering and results
    ordered_pot_masks = order_masks_row_major(pot_masks)
    plant_stats = process_pot_stats(
        image_np, ordered_pot_masks, plant_masks, associations
    )

    response.update(
        {
            "pot_masks": ordered_pot_masks,
            "plant_masks": plant_masks,
            "associations": associations,
            "ordered_pot_ids": [m["object_id"] for m in ordered_pot_masks],
            "plant_stats": plant_stats,
            "visualization_data": encode_image(
                visualize_pipeline_tracking(
                    image_np,
                    plant_masks,
                    ordered_pot_masks,
                    associations,
                    plant_stats=plant_stats,
                )
            ),
        }
    )

    return response
