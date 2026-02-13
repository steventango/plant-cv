import asyncio
import base64
import io
import json
import logging

import cv2
import httpx
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry

from app.plant.stats import analyze_plant_mask
from app.pot.quad import mask_to_quadrilateral
from app.pot.visualize import visualize_pipeline_tracking
from app.pot.warp import warp_quad_to_square
from app.analysis import (
    apply_tukey_outlier_detection_frame,
    update_plant_cleaning_state,
)

from app.cv_utils import safe_fill_poly

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

    session = requests.Session()
    retries = Retry(
        total=kwargs.pop("max_retries", 5),
        backoff_factor=kwargs.pop("backoff_factor", 1.0),
        status_forcelist=[502, 503, 504],
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    response = session.post(server_url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


async def async_get_embeddings(
    warped_images_b64: list[str],
    server_url: str = "http://embeddings:8803/predict",
    timeout: int = 60,
) -> list[dict]:
    """
    Call Embeddings API asynchronously for a list of images.
    """
    if not warped_images_b64:
        return []

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = []
        for img_b64 in warped_images_b64:
            payload = {
                "image_data": img_b64,
                "embedding_types": ["cls_token"],
            }
            tasks.append(client.post(server_url, json=payload))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Error calling embeddings API for image {i}: {resp}")
                results.append({"error": str(resp)})
            elif resp.status_code != 200:
                logger.error(
                    f"Embeddings API returned status {resp.status_code} for image {i}: {resp.text}"
                )
                results.append({"error": f"Status {resp.status_code}"})
            else:
                results.append(resp.json())

        return results


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
    Associate plants with pots based on a composite score of:
    1. Centroid Distance (normalized by pot size)
    2. Detection Confidence

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

    logger.debug(
        f"associate_plants_to_pots: {len(plant_masks)} plants, {len(pot_masks)} pots, greedy={greedy}"
    )

    # Convert to numpy arrays for vectorization
    plant_boxes = np.array([m["box"] for m in plant_masks])
    plant_confs = np.array([m.get("score", 1.0) for m in plant_masks])
    plant_centers = np.column_stack(
        [
            (plant_boxes[:, 0] + plant_boxes[:, 2]) / 2,
            (plant_boxes[:, 1] + plant_boxes[:, 3]) / 2,
        ]
    )

    pot_boxes = np.array([m["box"] for m in pot_masks])
    pot_centers = np.column_stack(
        [
            (pot_boxes[:, 0] + pot_boxes[:, 2]) / 2,
            (pot_boxes[:, 1] + pot_boxes[:, 3]) / 2,
        ]
    )
    pot_widths = pot_boxes[:, 2] - pot_boxes[:, 0]
    pot_heights = pot_boxes[:, 3] - pot_boxes[:, 1]
    pot_norms = np.maximum(np.maximum(pot_widths, pot_heights), 1e-6)

    # Compute distances: N_plants x M_pots
    # diffs shape: (N, M, 2)
    diffs = plant_centers[:, np.newaxis, :] - pot_centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (N, M)

    # check if plant center is inside pot box
    # plant_centers: (N, 2), pot_boxes: (M, 4) [x1, y1, x2, y2]
    # x_inside: (N, M), y_inside: (N, M)
    x_inside = (plant_centers[:, 0][:, np.newaxis] >= pot_boxes[np.newaxis, :, 0]) & (
        plant_centers[:, 0][:, np.newaxis] <= pot_boxes[np.newaxis, :, 2]
    )
    y_inside = (plant_centers[:, 1][:, np.newaxis] >= pot_boxes[np.newaxis, :, 1]) & (
        plant_centers[:, 1][:, np.newaxis] <= pot_boxes[np.newaxis, :, 3]
    )
    inside = x_inside & y_inside

    # Compute dist_score: (N, M)
    dist_scores = np.maximum(0, 1.0 - (dists / pot_norms[np.newaxis, :]))

    # Compute final score: (N, M)
    scores = plant_confs[:, np.newaxis] * dist_scores * inside

    if not greedy:
        # For each plant, find the best pot
        best_pot_indices = np.argmax(scores, axis=1)
        for i, plant in enumerate(plant_masks):
            score = scores[i, best_pot_indices[i]]
            if score > 0.05:
                associations[str(plant["object_id"])] = pot_masks[best_pot_indices[i]][
                    "object_id"
                ]
    else:
        # Greedy assignment
        # Get all scores above threshold
        plant_indices, pot_indices = np.where(scores > 0.05)
        # Flattened scores for sorting
        pair_scores = scores[plant_indices, pot_indices]

        # Sort indices by score descending
        sorted_indices = np.argsort(pair_scores)[::-1]

        used_plants = set()
        used_pots = set()

        for idx in sorted_indices:
            pi = plant_indices[idx]
            pti = pot_indices[idx]

            p_id = str(plant_masks[pi]["object_id"])
            pot_obj_id = pot_masks[pti]["object_id"]

            if p_id not in used_plants and pot_obj_id not in used_pots:
                associations[p_id] = pot_obj_id
                used_plants.add(p_id)
                used_pots.add(pot_obj_id)

    return associations


def refine_pot_masks_with_plants(pot_masks, plant_masks, associations, image_shape):
    """
    Refine pot masks by subtracting plant masks (rim/soil view).
    """
    h, w = image_shape[:2]
    # Map pot_id -> plant_masks list
    pot_to_plants = {}
    for p_id_str, pot_id in associations.items():
        if pot_id not in pot_to_plants:
            pot_to_plants[pot_id] = []
        plant_mask_obj = next(
            (pm for pm in plant_masks if str(pm["object_id"]) == p_id_str), None
        )
        if plant_mask_obj:
            pot_to_plants[pot_id].append(plant_mask_obj)

    for pot in pot_masks:
        pot_id = pot["object_id"]
        plants = pot_to_plants.get(pot_id, [])

        if not plants:
            continue

        # Build full pot mask
        pot_mask = np.zeros((h, w), dtype=np.uint8)
        if "contour" in pot:
            safe_fill_poly(pot_mask, pot["contour"], 1)

        # Build combined plant mask for this pot
        plant_mask_combined = np.zeros((h, w), dtype=np.uint8)
        for p in plants:
            if "contours" in p:
                safe_fill_poly(plant_mask_combined, p["contours"], 1)
            elif "contour" in p:
                safe_fill_poly(plant_mask_combined, p["contour"], 1)

        # Refine pot mask by subtracting plant mask (rim/soil view)
        # Note: We don't overwrite the main contour here because SAM3 needs the full pot.
        # This function can be extended if we need refined masks for soil analysis.

    return pot_masks


def wrap_state(sam3_state, id_map, pot_masks=None):
    """Wrap SAM3 state, ID map, and pot masks into a single base64 string."""
    data = {"sam3_state": sam3_state, "id_map": id_map, "pot_masks": pot_masks}
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def unwrap_state(wrapped_state):
    """Unwrap SAM3 state, ID map, and pot masks from a base64 string."""
    if not wrapped_state:
        return None, {}, None
    try:
        # Check if it's our JSON wrapper or raw SAM3 state
        decoded = base64.b64decode(wrapped_state).decode("utf-8")
        data = json.loads(decoded)
        if isinstance(data, dict) and "sam3_state" in data:
            return (
                data["sam3_state"],
                data.get("id_map", {}),
                data.get("pot_masks"),
            )
    except Exception:
        # If decoding fails, assume it's a raw SAM3 state
        pass
    return wrapped_state, {}, None


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
            safe_fill_poly(m, p["contour"], 1)

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

    # Calculate next_id from all values currently in use
    all_assigned_ids = list(id_map.values())
    for m in masks:
        if isinstance(m.get("object_id"), int):
            all_assigned_ids.append(m["object_id"])

    next_id = max(all_assigned_ids) + 1 if all_assigned_ids else 0

    for m in masks:
        old_id = m.get("object_id")

        # If already a stable ID (integer), keep it as is
        if isinstance(old_id, int):
            remapped_masks.append(m.copy())
            continue

        old_id_str = str(old_id)
        if old_id_str not in id_map:
            id_map[old_id_str] = next_id
            next_id += 1

        # Create a copy and update ID
        m_new = m.copy()
        m_new["object_id"] = id_map[old_id_str]
        remapped_masks.append(m_new)

    return remapped_masks, id_map


def process_pot_stats(image_np, pot_masks, plant_masks, associations):
    """
    Compute statistics for each plant in its respective pot.
    """
    h, w = image_np.shape[:2]
    stats_dict = {}

    # Collect all metadata for processing
    processing_tasks = []

    for pot in pot_masks:
        pot_id = pot["object_id"]
        plant_id = None
        for p_id_str, pt_id in associations.items():
            if pt_id == pot_id:
                plant_id = int(p_id_str)
                break

        if plant_id is None:
            stats_dict[str(pot_id)] = None
            continue

        plant_info = next((p for p in plant_masks if p["object_id"] == plant_id), None)
        if not plant_info:
            stats_dict[str(pot_id)] = None
            continue

        processing_tasks.append(
            {"pot_id": pot_id, "pot": pot, "plant_info": plant_info}
        )

    # 1. Generate warped images (synchronously for now as CV2 is fast)
    warped_images_info = []
    for task in processing_tasks:
        try:
            pot_id = task["pot_id"]
            pot = task["pot"]
            plant_info = task["plant_info"]

            # Pot Quad
            pot_mask_single = np.zeros((h, w), dtype=np.uint8)
            if "contour" in pot:
                safe_fill_poly(pot_mask_single, pot["contour"], 1)
            quad = mask_to_quadrilateral(pot_mask_single)

            # Warp Pot Image
            output_size = 224
            warped_pot, H = warp_quad_to_square(image_np, quad, output_size=output_size)

            # Warp Plant Mask
            p_contours = plant_info.get("contours")
            plant_mask_single = np.zeros((h, w), dtype=np.uint8)
            if p_contours:
                safe_fill_poly(plant_mask_single, p_contours, 255)
            else:
                plant_contour = plant_info.get("contour")
                if plant_contour:
                    safe_fill_poly(plant_mask_single, plant_contour, 255)

            warped_plant_mask = cv2.warpPerspective(
                plant_mask_single,
                H,
                (output_size, output_size),
                flags=cv2.INTER_NEAREST,
            )

            # Encode to B64
            pill_warped = Image.fromarray(warped_pot)
            buffered = io.BytesIO()
            pill_warped.save(buffered, format="JPEG")
            warped_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            warped_images_info.append(
                {
                    "pot_id": pot_id,
                    "warped_pot": warped_pot,
                    "warped_plant_mask": warped_plant_mask,
                    "warped_b64": warped_b64,
                    "plant_id": plant_info["object_id"],
                }
            )
        except Exception as e:
            logger.error(f"Error preparing warp for pot {task['pot_id']}: {e}")
            stats_dict[str(task["pot_id"])] = {"error": str(e)}

    # 2. Get embeddings asynchronously
    if warped_images_info:
        warped_b64_list = [info["warped_b64"] for info in warped_images_info]
        embeddings_results = asyncio.run(async_get_embeddings(warped_b64_list))

        for info, emb_res in zip(warped_images_info, embeddings_results):
            pot_id = info["pot_id"]
            try:
                # 3. Analyze Plant Stats
                plant_stats, _ = analyze_plant_mask(
                    info["warped_pot"], info["warped_plant_mask"]
                )
                plant_stats["warped_image"] = info["warped_b64"]

                # 4. Integrate Embeddings
                if "cls_token" in emb_res:
                    plant_stats["cls_token"] = emb_res["cls_token"]
                elif "error" in emb_res:
                    plant_stats["cls_token_error"] = emb_res["error"]

                plant_stats["plant_id"] = info["plant_id"]
                plant_stats["pot_id"] = pot_id
                stats_dict[str(pot_id)] = plant_stats
            except Exception as e:
                logger.error(f"Error finalising stats for pot {pot_id}: {e}")
                stats_dict[str(pot_id)] = {"error": str(e)}

    return stats_dict


def process_pipeline_outputs(
    image_np,
    plant_masks,
    pot_masks_raw,
    id_map=None,
    sam3_session_id=None,
    cleaning_state=None,
):
    """
    Common post-processing for both detect and propagate endpoints.
    """
    response = {}

    # 1. ID Mapping
    state = {}
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
        state["pot_state"] = wrap_state(
            sam3_session_id, updated_id_map, pot_masks=pot_masks
        )

    # 2. Refine plant masks
    plant_masks, associations = refine_plant_masks(image_np, plant_masks, pot_masks)

    # 3. Refine pot masks and check for outgrowth
    refine_pot_masks_with_plants(pot_masks, plant_masks, associations, image_np.shape)

    # 2.5 Filter plant masks to only include associated ones (ensures 1-to-1 matching in response)
    plant_masks = [p for p in plant_masks if str(p["object_id"]) in associations]

    # 4. Final ordering and results
    ordered_pot_masks = order_masks_row_major(pot_masks)
    plant_stats = process_pot_stats(
        image_np, ordered_pot_masks, plant_masks, associations
    )

    # 5. Apply Stat Cleaning
    # Collect valid stats for Tukey
    valid_stats_list = []
    for s in plant_stats.values():
        if s and "error" not in s:
            valid_stats_list.append(s)

    apply_tukey_outlier_detection_frame(valid_stats_list)

    if cleaning_state is None:
        cleaning_state = {}

    new_cleaning_state = {}

    # Map plant_id -> stats
    pid_to_stats = {}
    for s in valid_stats_list:
        pid = str(s.get("plant_id"))
        if pid:
            pid_to_stats[pid] = s

    # Identify all plant IDs to process (current + stored)
    all_pids = set(pid_to_stats.keys()).union(cleaning_state.keys())
    logger.info(
        f"DEBUG: Processing pids: {all_pids} (cleaning keys: {list(cleaning_state.keys())})"
    )

    for p_id in all_pids:
        s = pid_to_stats.get(p_id)
        current_state = cleaning_state.get(p_id)

        p_mask_obj = next(
            (m for m in plant_masks if str(m.get("object_id")) == p_id), None
        )
        updated_state, clean_values, clean_mask = update_plant_cleaning_state(
            s, current_state, p_mask_obj
        )

        if s:
            updated_state["last_pot_id"] = s.get("pot_id")

        new_cleaning_state[p_id] = updated_state

        if s:
            s.update(clean_values)
            if p_mask_obj and clean_mask and clean_mask is not p_mask_obj:
                p_mask_obj.update(clean_mask)

        elif clean_values.get("is_missing"):
            last_pot_id = updated_state.get("last_pot_id")
            prev_mask = updated_state.get("prev_clean_mask")
            logger.info(
                f"DEBUG: Plant {p_id} missing. Last pot: {last_pot_id}, Has mask: {prev_mask is not None}"
            )

            if last_pot_id is not None:
                full_stats = {
                    "plant_id": int(p_id),
                    "pot_id": last_pot_id,
                    "area": None,
                    **clean_values,
                }

                pid_str = str(last_pot_id)
                if pid_str not in plant_stats or plant_stats[pid_str] is None:
                    plant_stats[pid_str] = full_stats

                if prev_mask:
                    plant_masks.append(prev_mask)
                    if last_pot_id is not None:
                        associations[str(p_id)] = last_pot_id

    final_cleaning_state = cleaning_state.copy()
    final_cleaning_state.update(new_cleaning_state)
    state["cleaning_state"] = final_cleaning_state
    response.update(
        {
            "state": state,
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
