import base64
import io
import json
from math import ceil
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import requests
import supervision as sv
from PIL import Image
from supervision.draw.color import DEFAULT_COLOR_PALETTE, ColorPalette

from api import call_grounding_dino_api
from api.pipeline.app.plant import detect_and_segment_plants_in_crop

# Custom color palette with gray colors for ignored/rejected detections
CUSTOM_COLOR_PALETTE = (
    DEFAULT_COLOR_PALETTE * ceil(128 / len(DEFAULT_COLOR_PALETTE)) + ["#808080"] * 1000
)
color_palette_custom = ColorPalette.from_hex(CUSTOM_COLOR_PALETTE)

# Paths
TEST_IMAGES_DIR = Path("test_images")
OUTPUT_DIR = TEST_IMAGES_DIR / "grounding_dino"
OUTPUT_DIR.mkdir(exist_ok=True)

# Get all image files
image_files = sorted(TEST_IMAGES_DIR.glob("*.jpg"))

# Simple persistent state for cross-image ID association
_PREV_BOXES: np.ndarray | None = None
_PREV_IDS: np.ndarray | None = None
_NEXT_ID: int = 0

# Track mask areas per plant ID across frames for adaptive filtering
# Structure: {tracker_id: [area1, area2, ...]}
_PLANT_MASK_HISTORY: dict[int, list[float]] = {}
_HISTORY_LENGTH: int = 5  # Keep last N frames for each plant

# --- Pot mask caching (reuse initial pot masks across all timesteps) ---
# These globals store the first image's pot segmentation results. For subsequent
# images we skip calling Segment Anything and simply reuse these, assuming a
# fixed camera and stable pot positions. This avoids later frames' plant growth
# distorting pot masks.
_INITIAL_POT_MASKS: np.ndarray | None = None  # shape (M,H,W) binary masks (uint8/0/1)
_INITIAL_POT_BOXES: np.ndarray | None = None  # shape (M,4) xyxy boxes corresponding to masks
_INITIAL_POT_IDS: np.ndarray | None = None    # shape (M,) tracker ids for consistent naming
_INITIAL_POT_QUADS: list[np.ndarray] | None = None  # list of (4,2) float32 quadrilaterals per pot


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL for stable warps."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("_order_quad expects shape (4,2)")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def mask_to_quadrilateral(mask: np.ndarray, debug_image_path: Path | None = None) -> np.ndarray:
    """
    Compute a bounding quadrilateral around the pot mask using cv2.approxPolyN.

    Args:
        mask: 2D binary mask
        debug_image_path: Optional path to save debug visualization

    Returns (4,2) float32 points representing the bounding quadrilateral (TL, TR, BR, BL).
    """
    if mask.ndim != 2:
        raise ValueError("mask_to_quadrilateral expects a 2D mask")
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        raise ValueError("mask_to_quadrilateral received an empty mask")

    # Find largest contour
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    largest = max(contours, key=cv2.contourArea)

    # Approximate the contour to a quadrilateral using approxPolyN
    # nsides=4 ensures we get a 4-sided polygon
    # epsilon_percentage=-1 means no area limit (get best 4-sided approximation)
    # ensure_convex=True ensures the result is convex
    quad = cv2.approxPolyN(largest, nsides=4)

    # Reshape from (4, 1, 2) to (4, 2)
    quad = quad.reshape(4, 2).astype(np.float32)

    # Create debug visualization if requested
    if debug_image_path:
        debug_img = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        # Draw original contour in yellow
        cv2.drawContours(debug_img, [largest], -1, (0, 255, 255), 2)
        # Draw quadrilateral in green
        quad_int = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_img, [quad_int], isClosed=True, color=(0, 255, 0), thickness=3)
        # Draw corners as circles
        for i, pt in enumerate(quad):
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
        Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)).save(debug_image_path)
        print(f"  Saved debug visualization to: {debug_image_path}")

    # Order corners as TL, TR, BR, BL
    return _order_quad(quad)
# --- Plant vs Pot heuristic features ---
def _compute_mask_solidity(mask_bin: np.ndarray) -> float:
    m = (mask_bin.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-6:
        return 1.0
    return float(np.clip(area / hull_area, 0.0, 1.0))


def _compute_edge_density(image_rgb: np.ndarray, mask_bin: np.ndarray) -> float:
    if mask_bin.sum() == 0:
        return 0.0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, L2gradient=True)
    e = float(edges[mask_bin].mean()) / 255.0  # fraction of edge pixels in mask
    # Normalize to 0..1 with gentle cap (typical useful range up to ~0.15)
    return float(np.clip(e / 0.15, 0.0, 1.0))

def _compute_circularity(mask_bin: np.ndarray) -> float:
    m = (mask_bin.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    peri = float(cv2.arcLength(largest, True))
    if peri <= 1e-6:
        return 0.0
    circ = 4.0 * np.pi * area / (peri * peri)
    return float(np.clip(circ, 0.0, 1.0))


def warp_quad_to_square(
    image_np: np.ndarray,
    quad: np.ndarray,
    margin: float = 0.25,
    output_size: int | None = None,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp a quadrilateral region to a square image with uniform margin.

    Args:
        image_np: Input image
        quad: 4x2 array of quadrilateral corners (TL, TR, BR, BL)
        margin: Padding margin within the output square (default 0.25 = 25%)
        output_size: Optional fixed output size
        border_value: Border color for areas outside the image

    Returns (warped_square, homography_matrix).
    """
    quad = np.asarray(quad, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError("quad must have shape (4,2)")

    def _edge_len(a, b):
        return float(np.hypot(*(b - a)))

    tl, tr, br, bl = quad
    max_edge = max(
        _edge_len(tl, tr), _edge_len(tr, br), _edge_len(br, bl), _edge_len(bl, tl)
    )
    if max_edge < 1:
        max_edge = 1.0

    if output_size is None:
        side = int(round((1.0 + 2.0 * float(margin)) * max_edge))
        side = max(side, 8)
        pad = int(round(max_edge * float(margin)))
    else:
        side = int(output_size)
        pad = int(round(side * float(margin)))
    pad = max(0, min(pad, side // 4))

    inner = max(side - 2 * pad, 1)
    dst = np.array(
        [
            [pad, pad],
            [pad + inner - 1, pad],
            [pad + inner - 1, pad + inner - 1],
            [pad, pad + inner - 1],
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(quad, dst)
    warped = cv2.warpPerspective(
        image_np,
        H,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return warped, H


def detect_plants_in_cropped_pots(
    base_stem: str, warps_meta: list[dict]
) -> None:
    """
    For each warped pot crop, detect a plant with Grounding DINO and segment it.

    Uses the pipeline API in api.pipeline.app.plant to handle detection and segmentation.

    Saves per-crop artifacts: annotated box, mask image, mask overlay, and a JSON.

    Reason codes for ignored detections:
    - 901: Box too large (near full image size)
    - 902: Mask too large (near full crop area)
    - 903: Mask too small (tiny leaf fragment)
    """
    global _PLANT_MASK_HISTORY

    for meta in warps_meta:
        warp_name = meta.get("warp_image")
        if not warp_name:
            continue
        warp_path = OUTPUT_DIR / warp_name
        if not warp_path.exists():
            continue

        crop = Image.open(warp_path)
        tracker_id = meta.get("tracker_id")
        warp_suffix = Path(warp_name).stem

        # Call the pipeline API
        result = detect_and_segment_plants_in_crop(
            crop_image=crop,
            output_dir=OUTPUT_DIR,
            base_stem=base_stem,
            warp_suffix=warp_suffix,
            warp_name=warp_name,
            segment_fn=segment_pot_boxes,
            tracker_id=tracker_id,
            text_prompt="plant",
            detection_threshold=0.05,
            detection_text_threshold=0.05,
            area_ratio_threshold=0.90,
            mask_near_full_threshold=0.92,
            mask_median_multiplier=10.0,
        )

        # Save JSON result
        plant_dir = OUTPUT_DIR / "plants" / base_stem
        plant_json = plant_dir / f"{base_stem}_{warp_suffix}_plant.json"
        with open(plant_json, "w") as f:
            json.dump(result["detections_json"], f, indent=2)

        # Update mask history if plant was selected
        if result["success"] and tracker_id is not None:
            mask_area = np.sum(result["mask_binary"] > 0)
            if tracker_id not in _PLANT_MASK_HISTORY:
                _PLANT_MASK_HISTORY[tracker_id] = []
            _PLANT_MASK_HISTORY[tracker_id].append(float(mask_area))
            if len(_PLANT_MASK_HISTORY[tracker_id]) > _HISTORY_LENGTH:
                _PLANT_MASK_HISTORY[tracker_id] = _PLANT_MASK_HISTORY[tracker_id][-_HISTORY_LENGTH:]



def _infer_grid(n: int, target_aspect: float = 2.0) -> tuple[int, int]:
    """Infer (rows, cols) from tile count using factor pairs.

    - If n is a perfect square, prefer square (sqrt x sqrt).
    - Else choose factor pair (r,c), r<=c, that minimizes |c/r - target_aspect|.
    - Fallback: (rows, cols) = (int(sqrt(n)), ceil(n/rows)).
    """
    if n <= 0:
        return 1, 1
    r0 = int(np.floor(np.sqrt(n)))
    if r0 * r0 == n:
        return r0, r0
    pairs: list[tuple[int, int]] = []
    for r in range(1, r0 + 1):
        if n % r == 0:
            c = n // r
            if r <= c:
                pairs.append((r, c))
            else:
                pairs.append((c, r))
    if pairs:
        best = min(pairs, key=lambda rc: abs(rc[1] / rc[0] - float(target_aspect)))
        return best
    # Fallback if no factor pairs found (shouldn't happen): near-square grid
    rows = max(1, r0)
    cols = int(np.ceil(n / rows))
    return rows, cols


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Simple 1D k-means clustering.

    Returns (labels, centers) with centers sorted ascending and labels remapped accordingly.
    """
    v = np.asarray(values, dtype=float).ravel()
    n = v.size
    if n == 0 or k <= 1:
        return np.zeros(n, dtype=int), np.array([v.mean() if n else 0.0], dtype=float)
    k = int(max(1, min(int(k), n)))
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin <= 1e-9:
        return np.zeros(n, dtype=int), np.array([vmin], dtype=float)
    # Initialize centers at quantiles
    qs = np.linspace(0.0, 1.0, k, endpoint=False) + 0.5 / k
    centers = np.quantile(v, np.clip(qs, 0.0, 1.0))
    labels = np.zeros(n, dtype=int)
    for _ in range(max(1, int(iters))):
        # Assign
        # Use broadcasting for speeds on small n
        d = np.abs(v[:, None] - centers[None, :])
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = float(v[mask].mean())
            else:
                # Reinitialize empty cluster near global mean with small jitter
                centers[i] = float(v.mean())
    # Sort centers and remap labels
    order = np.argsort(centers)
    inv = np.zeros_like(order)
    inv[order] = np.arange(order.size)
    centers = centers[order]
    labels = inv[labels]
    return labels.astype(int, copy=False), centers.astype(float, copy=False)


def compute_row_major_grid_order(xyxy: np.ndarray, major: str = "row") -> np.ndarray:
    """Compute a perspective-robust grid ordering for tray-like layouts.

    Args:
        xyxy: (N,4) boxes.
        major: "row" (default) for row-major, "col" for column-major.

    Method:
        - Estimates grid axes via PCA on box centers
        - Orients axes to align monotonically with image x/y via correlation
        - Infers (rows, cols) from count and PCA aspect ratio
        - Clusters along the major axis (1D k-means) to assign major indices
        - Orders lexicographically: major index, then minor axis position
    """
    boxes = np.asarray(xyxy, dtype=float)
    n = boxes.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    P = np.stack([cx, cy], axis=1)

    # PCA via SVD on centered points
    mean = P.mean(axis=0)
    Q = P - mean
    try:
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        v0 = Vt[0, :]
        v1 = Vt[1, :]
    except Exception:
        # Fallback to simple y, then x
        return np.lexsort((boxes[:, 0], boxes[:, 1]))

    # Projections along PCA axes
    s = Q @ v0
    t = Q @ v1

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        try:
            if a.size < 2 or b.size < 2:
                return 0.0
            c = np.corrcoef(a, b)[0, 1]
            return float(c) if np.isfinite(c) else 0.0
        except Exception:
            return 0.0

    # Decide which axis is x-like (columns) vs y-like (rows) by correlation magnitude
    corr_x_s = abs(_safe_corr(s, cx))
    corr_x_t = abs(_safe_corr(t, cx))
    if corr_x_s >= corr_x_t:
        u = s  # x-like (left→right)
        v = t  # y-like (top→bottom)
    else:
        u = t
        v = s

    # Orient u to increase with x and v to increase with y
    if _safe_corr(u, cx) < 0:
        u = -u
    if _safe_corr(v, cy) < 0:
        v = -v

    # Estimate grid shape from aspect in PCA space
    range_s = float(s.max() - s.min())
    range_t = float(t.max() - t.min())
    # Use u for width and v for height when estimating aspect
    range_s = float(u.max() - u.min())
    range_t = float(v.max() - v.min())
    aspect = (range_s / (range_t + 1e-9)) if range_t > 1e-9 else 10.0
    rows, cols = _infer_grid(n=int(n), target_aspect=float(max(0.5, min(5.0, aspect))))

    major = (major or "row").lower()
    if major == "col":
        # Cluster columns along u (left→right)
        col_labels, col_centers = _kmeans_1d(u, max(1, cols))
        col_order = np.argsort(col_centers)  # left to right
        col_map = np.zeros_like(col_order)
        col_map[col_order] = np.arange(col_order.size)
        col_idx = col_map[col_labels]
        # Within each column, sort by v (top→bottom)
        order = np.lexsort((v, col_idx))
    else:
        # Row-major (default): cluster rows along v (top→bottom)
        row_labels, row_centers = _kmeans_1d(v, max(1, rows))
        row_order = np.argsort(row_centers)  # top to bottom
        row_map = np.zeros_like(row_order)
        row_map[row_order] = np.arange(row_order.size)
        row_idx = row_map[row_labels]
        # Within each row, sort by u (left→right)
        order = np.lexsort((u, row_idx))
    return order.astype(int, copy=False)


def _compose_grid(image_paths: list[Path], tile_size: int = 256) -> Image.Image:
    """Compose a grid from image paths based on inferred rows/cols.

    Returns a PIL Image. Skips non-existent paths.
    """
    paths = [p for p in image_paths if p.exists()]
    if not paths:
        return Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))

    n = len(paths)
    rows, cols = _infer_grid(n)

    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(0, 0, 0))
    try:
        resample_bilinear = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    except Exception:
        resample_bilinear = 2  # PIL bilinear fallback
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im = im.resize((tile_size, tile_size), resample_bilinear)
        r = i // cols
        c = i % cols
        canvas.paste(im, (c * tile_size, r * tile_size))
    return canvas


def create_warp_composites(base_stem: str, warps_meta: list[dict], tile_size: int = 256) -> None:
    """Create composites for warped pots with plant boxes and mask overlays.

    Saves two images: `<base>_composite_plant_boxes.jpg` and
    `<base>_composite_plant_masks.jpg` in `OUTPUT_DIR`.
    """
    def sort_key(m: dict):
        tid = m.get("tracker_id")
        idx = m.get("index", 0)
        return (0, int(tid)) if tid is not None else (1, int(idx))

    warps_sorted = sorted(warps_meta, key=sort_key)
    plant_dir = OUTPUT_DIR / "plants" / base_stem
    box_tiles: list[Path] = []
    mask_tiles: list[Path] = []
    for meta in warps_sorted:
        warp_name = meta.get("warp_image")
        if not warp_name:
            continue
        suffix = Path(warp_name).stem
        # Prefer subdirectory paths; fallback to root; else raw warp
        box_path_sub = plant_dir / f"{base_stem}_{suffix}_plant_box.jpg"
        overlay_path_sub = plant_dir / f"{base_stem}_{suffix}_plant_mask_overlay.jpg"
        box_path_root = OUTPUT_DIR / f"{base_stem}_{suffix}_plant_box.jpg"
        overlay_path_root = OUTPUT_DIR / f"{base_stem}_{suffix}_plant_mask_overlay.jpg"
        warp_path = OUTPUT_DIR / warp_name
        # Prefer annotation images in subdir; fallback to root; else raw warp
        box_tiles.append(
            box_path_sub if box_path_sub.exists() else (
                box_path_root if box_path_root.exists() else warp_path
            )
        )
        mask_tiles.append(
            overlay_path_sub if overlay_path_sub.exists() else (
                overlay_path_root if overlay_path_root.exists() else warp_path
            )
        )

    if box_tiles:
        comp_boxes = _compose_grid(box_tiles, tile_size=tile_size)
        comp_boxes_path = OUTPUT_DIR / f"{base_stem}_composite_plant_boxes.jpg"
        comp_boxes.save(comp_boxes_path)
    if mask_tiles:
        comp_masks = _compose_grid(mask_tiles, tile_size=tile_size)
        comp_masks_path = OUTPUT_DIR / f"{base_stem}_composite_plant_masks.jpg"
        comp_masks.save(comp_masks_path)

def detect_pots(image: Image.Image, image_np: np.ndarray, text_prompt: str = "pot"):
    """Detect pots with Grounding DINO and remove large-area outliers."""
    boxes, confidences, class_names = call_grounding_dino_api(
        image=image,
        text_prompt=text_prompt,
        threshold=0.03,
        text_threshold=0,
    )

    boxes = np.asarray(boxes)
    confidences = np.asarray(confidences)
    class_names = np.asarray(class_names)

    print(f"  Detected {len(boxes)} objects")

    if len(boxes) >= 1:
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
        areas = widths * heights

        # Heuristic: keep aspect ratio between 1:2 and 2:1 (0.5..2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            aspect = widths / np.maximum(heights, 1e-6)
        ar_mask = (aspect >= 0.5) & (aspect <= 2.0)
        if not ar_mask.all():
            removed = int((~ar_mask).sum())
            boxes = boxes[ar_mask]
            confidences = confidences[ar_mask] if len(confidences) == len(aspect) else confidences
            class_names = class_names[ar_mask] if len(class_names) == len(aspect) else class_names
            widths = widths[ar_mask]
            heights = heights[ar_mask]
            areas = areas[ar_mask]
            print(f"  Removed {removed} aspect-ratio outlier(s) (<0.5 or >2.0)")

        if len(areas) >= 2:
            q1, q3 = np.percentile(areas, [25, 75])
            iqr = q3 - q1
            median_area = float(np.median(areas))
            tukey_hi = q3 + 1.5 * iqr
            median_hi = median_area * 1.5 if median_area > 0 else tukey_hi
            hi_thresh = min(tukey_hi, median_hi)
            img_area = image_np.shape[0] * image_np.shape[1]
            hi_thresh = min(hi_thresh, 0.2 * img_area)

            inlier_mask = areas <= hi_thresh
            if not inlier_mask.any():
                smallest_idx = int(np.argmin(areas))
                inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
                inlier_mask[smallest_idx] = True

            removed = int(len(areas) - inlier_mask.sum())
            if removed > 0:
                boxes = boxes[inlier_mask]
                confidences = (
                    confidences[inlier_mask]
                    if confidences.shape[0] == inlier_mask.shape[0]
                    else confidences
                )
                class_names = (
                    class_names[inlier_mask]
                    if class_names.shape[0] == inlier_mask.shape[0]
                    else class_names
                )
                print(f"  Removed {removed} large-area outlier(s)")

    # After outlier filtering, drop any box fully contained in a larger box
    if len(boxes) >= 2:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

        # contains[i, j] => box i contains box j (including shared edges)
        contains = (
            (x1[:, None] <= x1[None, :])
            & (y1[:, None] <= y1[None, :])
            & (x2[:, None] >= x2[None, :])
            & (y2[:, None] >= y2[None, :])
        )
        # Exclude self comparisons
        np.fill_diagonal(contains, False)

        # Only consider cases where container strictly larger by area
        larger = areas[:, None] > areas[None, :]
        contained_smaller = contains & larger

        # Mark any box j that is contained by any strictly larger box i
        to_remove = np.asarray(contained_smaller.any(axis=0), dtype=bool)

        if to_remove.any():
            kept = ~to_remove
            removed = int(to_remove.sum())
            orig_n = int(len(to_remove))
            kept_n = orig_n - removed
            boxes = boxes[kept]
            if len(confidences) == orig_n:
                confidences = confidences[kept]
            if len(class_names) == orig_n:
                class_names = class_names[kept]
            print(f"  Suppressed {removed} contained smaller box(es)")

    return boxes, confidences, class_names


def segment_pot_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    multimask_output: bool = False,
    server_url: str = "http://segment-anything:8000/predict",
):
    """Call Segment Anything API for the given boxes and return masks, scores.

    The API is expected to return {contours: List[List[[x,y], ...]], scores: List[float]}.
    """
    # Encode image
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = json.dumps(
        {
            "image_data": img_str,
            "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            "multimask_output": multimask_output,
        }
    )
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(server_url, data=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        # Build masks from contours
        h, w = np.array(image).shape[:2]
        contours = result.get("contours", [])
        masks = np.zeros((len(contours), h, w), dtype=np.uint8)
        for i, contour in enumerate(contours):
            if contour:
                cnt = np.array(contour, dtype=np.int32)
                # Rasterize polygon: simple fill (scanline)
                # Use bounding box and point-in-polygon (vectorized) to avoid OpenCV dependency
                minx = np.clip(cnt[:, 0].min(), 0, w - 1)
                maxx = np.clip(cnt[:, 0].max(), 0, w - 1)
                miny = np.clip(cnt[:, 1].min(), 0, h - 1)
                maxy = np.clip(cnt[:, 1].max(), 0, h - 1)
                xs = np.arange(minx, maxx + 1)
                ys = np.arange(miny, maxy + 1)
                grid_x, grid_y = np.meshgrid(xs, ys)
                pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

                # Ray casting point-in-polygon
                x = pts[:, 0]
                y = pts[:, 1]
                x0 = cnt[:, 0]
                y0 = cnt[:, 1]
                j = len(cnt) - 1
                inside = np.zeros(len(pts), dtype=bool)
                for i2 in range(len(cnt)):
                    xi, yi = x0[i2], y0[i2]
                    xj, yj = x0[j], y0[j]
                    intersect = ((yi > y) != (yj > y)) & (
                        x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
                    )
                    inside ^= intersect
                    j = i2
                patch = inside.reshape(len(ys), len(xs))
                masks[i, ys[:, None], xs[None, :]] = patch.astype(np.uint8)

        scores = np.array(result.get("scores", []))
        return masks, scores
    except Exception as e:
        print(f"Error calling Segment Anything API: {e}")
        return np.array([]), np.array([])


def process_image_file(image_path: Path):
    print(f"\nProcessing: {image_path.name}")
    image = Image.open(image_path)
    image_np = np.array(image)

    # Detect pots and filter
    boxes, confidences, class_names = detect_pots(image, image_np, text_prompt="pot")

    # Visualize detections
    if len(boxes) > 0:
        class_ids = np.arange(len(class_names), dtype=int)
        detections: sv.Detections = sv.Detections(
            xyxy=boxes, confidence=confidences, class_id=class_ids
        )

        # Perspective-aware row-major ordering across the tray grid
        # Top-left is 0, then go left→right within a row, top→bottom across rows
        order = compute_row_major_grid_order(detections.xyxy, major="row")
        detections = cast(sv.Detections, detections[order])
        detections.tracker_id = np.full(len(detections), -1, dtype=int)

        print(f"Number of detections: {len(detections)}")

        # Build persistent IDs across images (fallback if ByteTrack can't match)
        global _PREV_BOXES, _PREV_IDS, _NEXT_ID
        used: sv.Detections = detections
        n = len(used)
        ids = np.full(n, -1, dtype=int)

        if _PREV_BOXES is None or _PREV_IDS is None:
            # First image across the run: ensure top-left gets id 0 and consecutive ordering
            # Row-major: left→right within a row, then next row top→bottom
            order_init = compute_row_major_grid_order(used.xyxy, major="row")
            assigned = 0
            for di in order_init:
                if ids[di] == -1:
                    ids[di] = _NEXT_ID
                    _NEXT_ID += 1
                    assigned += 1
            _PREV_BOXES = used.xyxy.copy()
            _PREV_IDS = ids.copy()
        else:
            # Associate remaining -1s to previous IDs via IoU >= 0.2 using greedy matching
            ious = sv.box_iou_batch(_PREV_BOXES, used.xyxy)  # type: ignore
            # Greedy highest IoU assignment ensuring 1-1
            pairs = []
            for pi in range(ious.shape[0]):
                for di in range(ious.shape[1]):
                    pairs.append((ious[pi, di], pi, di))
            pairs.sort(reverse=True)
            taken_prev = set()
            taken_det = set(i for i in range(n) if ids[i] != -1)
            for v, pi, di in pairs:
                if v < 0.2:
                    break
                if pi in taken_prev or di in taken_det:
                    continue
                ids[di] = int(_PREV_IDS[pi])
                taken_prev.add(pi)
                taken_det.add(di)
            # Any still -1 get new IDs in top-left row-major order
            grid_order = compute_row_major_grid_order(used.xyxy, major="row")
            remaining = [di for di in grid_order if ids[di] == -1]
            for di in remaining:
                ids[di] = _NEXT_ID
                _NEXT_ID += 1
            _PREV_BOXES = used.xyxy.copy()
            _PREV_IDS = ids.copy()


        # Ensure tracker IDs are unique within this image
        # If duplicates exist, assign new unique IDs to the later occurrences
        seen = {}
        for i, tid in enumerate(ids):
            if tid in seen:
                # Duplicate found, assign a new unique ID
                ids[i] = _NEXT_ID
                _NEXT_ID += 1
            else:
                seen[tid] = i
        used.tracker_id = ids

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image_np.copy(), detections=used
        )

        label_annotator = sv.LabelAnnotator()
        n_used: int = len(used)
        conf_arr: np.ndarray = np.asarray(
            used.confidence if getattr(used, "confidence", None) is not None else np.zeros(n_used, dtype=float),
            dtype=float,
        )
        tid_attr = getattr(used, "tracker_id", None)
        if isinstance(tid_attr, np.ndarray) and len(tid_attr) == n_used and n_used > 0:
            tid_arr_use = tid_attr.astype(int, copy=False)
            labels = [f"#{int(t)} {c:.2f}" for t, c in zip(tid_arr_use, conf_arr)]
        else:
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(class_names, confidences)
            ]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=used, labels=labels
        )

        output_image_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
        Image.fromarray(annotated_frame).save(output_image_path)
        print(f"  Saved annotated image to: {output_image_path}")

        # Save detection JSON
        output_json_path = OUTPUT_DIR / f"{image_path.stem}_detections.json"
        detection_data = {
            "image": image_path.name,
            "detections": [
                {
                    "box": box.tolist(),
                    "confidence": float(conf),
                    "class_name": class_name,
                    "tracker_id": (int(tid) if tid is not None else None),
                }
                for (box, conf, class_name, tid) in zip(
                    used.xyxy,
                    conf_arr,
                    class_names[: n_used],
                    (
                        tid_attr.astype(object, copy=False)
                        if isinstance(tid_attr, np.ndarray) and len(tid_attr) == n_used
                        else np.array([None] * n_used, dtype=object)
                    ),
                )
            ],
        }
        with open(output_json_path, "w") as f:
            json.dump(detection_data, f, indent=2)
        print(f"  Saved detection data to: {output_json_path}")

        # Segment Anything over pot boxes
        # For segmentation, prefer tracked boxes for ID consistency
        seg_boxes = used.xyxy if len(used) else boxes
        seg_conf = used.confidence if len(used) else confidences
        seg_conf = np.asarray(seg_conf)
        seg_ids = getattr(used, "tracker_id", None)
        # ------------------------------------------------------------------
        # Pot segmentation (cached first frame):
        # We only run Segment Anything on the first image to obtain pot masks.
        # For subsequent images we reuse the initial masks, boxes, quads and ids
        # to avoid plant overgrowth distorting pot segmentation.
        global _INITIAL_POT_MASKS, _INITIAL_POT_BOXES, _INITIAL_POT_IDS, _INITIAL_POT_QUADS
        global _INITIAL_POT_SCORES  # store SAM scores if available
        try:
            _ = _INITIAL_POT_SCORES  # type: ignore[name-defined]
        except NameError:
            _INITIAL_POT_SCORES = None  # lazy declare

        if _INITIAL_POT_MASKS is None:
            # First frame: perform SAM segmentation
            masks, scores = segment_pot_boxes(image, seg_boxes)
            if masks.size > 0:
                # Compute quadrilaterals once per mask and cache
                quads: list[np.ndarray] = []
                for mi in range(masks.shape[0]):
                    mask_bin = masks[mi] > 0
                    try:
                        quad = mask_to_quadrilateral(mask_bin)
                        quads.append(quad)
                    except Exception as e:
                        print(f"  Initial quad computation failed for mask {mi}: {e}")
                        continue
                _INITIAL_POT_MASKS = masks.copy()
                _INITIAL_POT_BOXES = seg_boxes.copy()
                _INITIAL_POT_IDS = np.asarray(seg_ids).copy() if isinstance(seg_ids, np.ndarray) else None
                _INITIAL_POT_QUADS = quads
                _INITIAL_POT_SCORES = scores.copy() if scores is not None and scores.size else None
                print(f"  Cached {len(quads)} initial pot mask(s) for reuse in later frames")
        else:
            # Subsequent frames: reuse cached data
            masks = _INITIAL_POT_MASKS.copy()
            scores = _INITIAL_POT_SCORES if _INITIAL_POT_SCORES is not None else np.zeros(len(masks), dtype=float)
            seg_boxes = _INITIAL_POT_BOXES.copy() if _INITIAL_POT_BOXES is not None else seg_boxes
            seg_ids = _INITIAL_POT_IDS.copy() if _INITIAL_POT_IDS is not None else seg_ids
            quads = _INITIAL_POT_QUADS if _INITIAL_POT_QUADS is not None else []
            print(f"  Reused {len(quads)} cached pot mask(s) from first frame")
        # ------------------------------------------------------------------
        if masks.size > 0:
            # Save labeled mask where pixel = index+1
            labeled = np.zeros(image_np.shape[:2], dtype=np.uint16)
            for idx in range(min(masks.shape[0], 65534)):
                labeled[masks[idx] > 0] = idx + 1
            mask_path = OUTPUT_DIR / f"{image_path.stem}_masks.png"
            Image.fromarray(labeled).save(mask_path)
            print(f"  Saved masks to: {mask_path}")


            # Visualize masks on top using Supervision
            m = masks.shape[0]
            mask_detections = sv.Detections(
                xyxy=seg_boxes[:m],
                confidence=seg_conf[:m],
                class_id=np.arange(min(len(seg_boxes), m), dtype=int),
            )
            mask_detections.mask = masks.astype(bool)

            mask_annotator = sv.MaskAnnotator()
            overlay = mask_annotator.annotate(
                scene=image_np.copy(),
                detections=mask_detections,
            )

            # Draw quadrilaterals and IDs on a single overlay
            overlay_with_quads_and_ids = overlay.copy()
            for idx in range(m):
                mask_bin = masks[idx] > 0

                # Create debug path for this mask
                if seg_ids is not None and len(seg_ids) > idx and seg_ids[idx] is not None:
                    suffix = f"id{int(seg_ids[idx])}"
                else:
                    suffix = f"idx{idx:02d}"
                debug_path = OUTPUT_DIR / f"{image_path.stem}_mask_{suffix}_lines_debug.jpg"

                try:
                    quad = mask_to_quadrilateral(mask_bin)
                    quad_int = np.round(quad).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(overlay_with_quads_and_ids, [quad_int], isClosed=True, color=(255, 255, 0), thickness=1)
                except Exception as e:
                    print(f"  Could not draw quad for mask {idx}: {e}")
                    import traceback
                    traceback.print_exc()

            # Add tracked labels on the same overlay
            if seg_ids is not None and len(seg_ids):
                id_labels = [f"#{int(tid)}" for tid in seg_ids[:m]]
                overlay_with_quads_and_ids = sv.LabelAnnotator().annotate(
                    scene=overlay_with_quads_and_ids,
                    detections=mask_detections,
                    labels=id_labels,
                )

            overlay_path = OUTPUT_DIR / f"{image_path.stem}_masks_overlay.jpg"
            Image.fromarray(overlay_with_quads_and_ids).save(overlay_path)
            print(f"  Saved mask overlay with quadrilaterals and IDs to: {overlay_path}")

            # Warp each mask region to a square with margin to correct perspective
            # Save per-id warps into plants/<base_stem>/ subdirectory
            plant_dir = OUTPUT_DIR / "plants" / image_path.stem
            plant_dir.mkdir(parents=True, exist_ok=True)
            warps_meta: list[dict] = []
            for idx in range(m):
                # Use cached quad if available (ensures consistent pot crop across frames)
                if _INITIAL_POT_QUADS is not None and idx < len(_INITIAL_POT_QUADS):
                    quad = _INITIAL_POT_QUADS[idx]
                else:
                    # Fallback: recompute quad from current (cached or freshly segmented) mask
                    mask_bin = masks[idx] > 0
                    try:
                        quad = mask_to_quadrilateral(mask_bin)
                    except Exception as e:
                        print(f"  Quad fallback failed for mask {idx}: {e}")
                        continue
                try:
                    warped, H = warp_quad_to_square(image_np, quad, margin=0.25)
                except Exception as e:
                    print(f"  Warp failed for mask {idx}: {e}")
                    continue

                # Name outputs using initial tracker id when available (seg_ids reused)
                if seg_ids is not None and len(seg_ids) > idx and seg_ids[idx] is not None:
                    suffix = f"id{int(seg_ids[idx])}"
                else:
                    suffix = f"idx{idx:02d}"
                warp_path = plant_dir / f"{image_path.stem}_warp_{suffix}.jpg"
                Image.fromarray(warped).save(warp_path)

                warps_meta.append(
                    {
                        "index": int(idx),
                        "tracker_id": (
                            int(seg_ids[idx]) if seg_ids is not None and len(seg_ids) > idx and seg_ids[idx] is not None else None
                        ),
                        "quad": np.asarray(quad, dtype=float).tolist(),
                        "homography": np.asarray(H, dtype=float).tolist(),
                        "warp_image": str(warp_path.relative_to(OUTPUT_DIR)),
                    }
                )

            # Save a single JSON manifest for this image's warps
            if warps_meta:
                warps_json = OUTPUT_DIR / f"{image_path.stem}_warps.json"
                with open(warps_json, "w") as f:
                    json.dump(
                        {
                            "image": image_path.name,
                            "count": len(warps_meta),
                            "warps": warps_meta,
                        },
                        f,
                        indent=2,
                    )
                print(f"  Saved {len(warps_meta)} warped crops to: {warps_json}")
                # New step: detect and segment plant within each warped pot crop
                detect_plants_in_cropped_pots(image_path.stem, warps_meta)
                # Create composite images for quick review
                create_warp_composites(image_path.stem, warps_meta, tile_size=256)
        else:
            print("  No masks returned from Segment Anything")
    else:
        print("  No detections found")
        output_image_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
        image.save(output_image_path)


def main():
    print(f"Found {len(image_files)} images to process")
    for image_path in image_files:
        process_image_file(image_path)
    print(f"\n✓ Processing complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
