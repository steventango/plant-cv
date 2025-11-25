import numpy as np
import requests
from PIL import Image

from .utils import encode_image


def call_grounding_dino_api(
    image: Image.Image,
    text_prompt: str,
    threshold: float = 0.05,
    text_threshold: float = 0.05,
    server_url: str = "http://grounding-dino:8000/predict",
):
    """
    Call the Grounding DINO API with an image and text prompt.

    Args:
        image: PIL Image to be processed
        text_prompt: Text prompt for object detection
        threshold: Confidence threshold for boxes
        text_threshold: Text threshold
        server_url: URL of the Grounding DINO API server

    Returns:
        boxes: Numpy array of boxes in format [x1, y1, x2, y2]
        scores: Numpy array of confidence scores
        text_labels: List of detected class names
    """
    payload = {
        "image_data": encode_image(image),
        "text_prompt": text_prompt,
        "threshold": threshold,
        "text_threshold": text_threshold,
    }

    response = requests.post(server_url, json=payload)
    response.raise_for_status()
    result = response.json()
    boxes = np.array(result["boxes"])
    scores = np.array(result["scores"])
    text_labels = result["text_labels"]

    return boxes, scores, text_labels


# def _infer_grid(n: int, target_aspect: float = 2.0) -> tuple[int, int]:
#     """Infer (rows, cols) from tile count using factor pairs.

#     - If n is a perfect square, prefer square (sqrt x sqrt).
#     - Else choose factor pair (r,c), r<=c, that minimizes |c/r - target_aspect|.
#     - Fallback: (rows, cols) = (int(sqrt(n)), ceil(n/rows)).
#     """
#     if n <= 0:
#         return (1, 1)
#     r0 = int(np.floor(np.sqrt(n)))
#     if r0 * r0 == n:
#         return (r0, r0)
#     pairs: list[tuple[int, int]] = []
#     for r in range(1, r0 + 1):
#         if n % r == 0:
#             pairs.append((r, n // r))
#     if pairs:
#         best = min(pairs, key=lambda rc: abs(rc[1] / rc[0] - float(target_aspect)))
#         return best
#     # Fallback if no factor pairs found (shouldn't happen): near-square grid
#     rows = max(1, r0)
#     cols = int(np.ceil(n / rows))
#     return rows, cols


# def _kmeans_1d(
#     values: np.ndarray, k: int, iters: int = 20
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Simple 1D k-means clustering.

#     Returns (labels, centers) with centers sorted ascending and labels remapped accordingly.
#     """
#     v = np.asarray(values, dtype=float).ravel()
#     n = v.size
#     if n == 0 or k <= 1:
#         return np.zeros(n, dtype=int), np.array([np.mean(v)] if n > 0 else [0.0])
#     k = int(max(1, min(int(k), n)))
#     vmin, vmax = float(v.min()), float(v.max())
#     if vmax - vmin <= 1e-9:
#         return np.zeros(n, dtype=int), np.array([np.mean(v)])
#     # Initialize centers at quantiles
#     qs = np.linspace(0.0, 1.0, k, endpoint=False) + 0.5 / k
#     centers = np.quantile(v, np.clip(qs, 0.0, 1.0))
#     labels = np.zeros(n, dtype=int)
#     for _ in range(max(1, int(iters))):
#         # Assign
#         d = np.abs(v[:, None] - centers[None, :])
#         new_labels = np.argmin(d, axis=1)
#         if np.array_equal(new_labels, labels):
#             break
#         labels = new_labels
#         # Update
#         for i in range(k):
#             mask = labels == i
#             if mask.any():
#                 centers[i] = v[mask].mean()
#     # Sort centers and remap labels
#     order = np.argsort(centers)
#     inv = np.zeros_like(order)
#     inv[order] = np.arange(order.size)
#     centers = centers[order]
#     labels = inv[labels]
#     return labels.astype(int, copy=False), centers.astype(float, copy=False)


# def compute_row_major_grid_order(xyxy: np.ndarray, major: str = "row") -> np.ndarray:
#     """Compute a perspective-robust grid ordering for tray-like layouts.

#     Args:
#         xyxy: (N,4) boxes.
#         major: "row" (default) for row-major, "col" for column-major.

#     Method:
#         - Estimates grid axes via PCA on box centers
#         - Orients axes to align monotonically with image x/y via correlation
#         - Infers (rows, cols) from count and PCA aspect ratio
#         - Clusters along the major axis (1D k-means) to assign major indices
#         - Orders lexicographically: major index, then minor axis position
#     """
#     boxes = np.asarray(xyxy, dtype=float)
#     n = boxes.shape[0]
#     if n <= 1:
#         return np.arange(n, dtype=int)

#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     P = np.stack([cx, cy], axis=1)

#     # PCA via SVD on centered points
#     mean = P.mean(axis=0)
#     Q = P - mean
#     try:
#         _, _, Vt = np.linalg.svd(Q, full_matrices=False)
#         v0 = Vt[0, :]
#         v1 = Vt[1, :]
#     except Exception:
#         # Fallback to simple y, then x
#         return np.lexsort((boxes[:, 0], boxes[:, 1]))

#     # Projections along PCA axes
#     s = Q @ v0
#     t = Q @ v1

#     def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
#         try:
#             a_std = np.std(a)
#             b_std = np.std(b)
#             if a_std < 1e-9 or b_std < 1e-9:
#                 return 0.0
#             return float(np.corrcoef(a, b)[0, 1])
#         except Exception:
#             return 0.0

#     # Decide which axis is x-like (columns) vs y-like (rows) by correlation magnitude
#     corr_x_s = abs(_safe_corr(s, cx))
#     corr_x_t = abs(_safe_corr(t, cx))
#     if corr_x_s >= corr_x_t:
#         u = s  # x-like (left→right)
#         v = t  # y-like (top→bottom)
#     else:
#         u = t
#         v = s

#     # Orient u to increase with x and v to increase with y
#     if _safe_corr(u, cx) < 0:
#         u = -u
#     if _safe_corr(v, cy) < 0:
#         v = -v

#     # Estimate grid shape from aspect in PCA space
#     range_s = float(u.max() - u.min())
#     range_t = float(v.max() - v.min())
#     aspect = (range_s / (range_t + 1e-9)) if range_t > 1e-9 else 10.0
#     rows, cols = _infer_grid(n=int(n), target_aspect=float(max(0.5, min(5.0, aspect))))

#     major = (major or "row").lower()
#     if major == "col":
#         # Cluster columns along u (left→right)
#         col_labels, col_centers = _kmeans_1d(u, max(1, cols))
#         col_order = np.argsort(col_centers)  # left to right
#         col_map = np.zeros_like(col_order)
#         col_map[col_order] = np.arange(col_order.size)
#         col_idx = col_map[col_labels]
#         # Within each column, sort by v (top→bottom)
#         order = np.lexsort((v, col_idx))
#     else:
#         # Row-major (default): cluster rows along v (top→bottom)
#         row_labels, row_centers = _kmeans_1d(v, max(1, rows))
#         row_order = np.argsort(row_centers)  # top to bottom
#         row_map = np.zeros_like(row_order)
#         row_map[row_order] = np.arange(row_order.size)
#         row_idx = row_map[row_labels]
#         # Within each row, sort by u (left→right)
#         order = np.lexsort((u, row_idx))
#     return order.astype(int, copy=False)


def detect_pots(image, text_prompt="pot", threshold=0.03, text_threshold=0):
    """
    Detect pots in an image and filter outliers.

    Args:
        image: PIL Image or numpy array
        text_prompt: Text prompt for detection (default: "pot")
        threshold: Confidence threshold
        text_threshold: Text threshold

    Returns:
        boxes: Filtered numpy array of boxes [x1, y1, x2, y2] in grid order
        confidences: Filtered confidence scores
        class_names: Filtered class names
    """
    if isinstance(image, np.ndarray):
        image_np = image
        image = Image.fromarray(image)
    else:
        image_np = np.array(image)

    boxes, confidences, class_names = call_grounding_dino_api(
        image=image,
        text_prompt=text_prompt,
        threshold=threshold,
        text_threshold=text_threshold,
    )

    boxes = np.asarray(boxes)
    confidences = np.asarray(confidences)
    class_names = np.asarray(class_names)

    if len(boxes) == 0:
        return boxes, confidences, class_names, np.array([], dtype=int)

    # Filter by aspect ratio (0.5 to 2.0)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    areas = widths * heights

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = widths / np.maximum(heights, 1e-6)
    ar_mask = (aspect >= 0.5) & (aspect <= 2.0)

    boxes = boxes[ar_mask]
    confidences = (
        confidences[ar_mask] if len(confidences) == len(aspect) else confidences
    )
    class_names = (
        class_names[ar_mask] if len(class_names) == len(aspect) else class_names
    )
    areas = areas[ar_mask]

    # Filter by area (remove outliers using IQR and median for both small and large)
    if len(areas) >= 2:
        q1, q3 = np.percentile(areas, [25, 75])
        iqr = q3 - q1
        median_area = float(np.median(areas))

        # Upper threshold (remove large outliers)
        tukey_hi = q3 + 1.5 * iqr
        median_hi = median_area * 1.5 if median_area > 0 else tukey_hi
        hi_thresh = min(tukey_hi, median_hi)
        img_area = image_np.shape[0] * image_np.shape[1]
        hi_thresh = min(hi_thresh, 0.2 * img_area)

        # Lower threshold (remove small outliers)
        tukey_lo = q1 - 1.5 * iqr
        median_lo = median_area * 0.5 if median_area > 0 else tukey_lo
        lo_thresh = max(tukey_lo, median_lo, 0)

        inlier_mask = (areas >= lo_thresh) & (areas <= hi_thresh)
        if not inlier_mask.any():
            # Fallback: keep box closest to median area
            median_idx = int(np.argmin(np.abs(areas - median_area)))
            inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
            inlier_mask[median_idx] = True

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

    # Filter by confidence (remove low-confidence outliers)
    if len(confidences) >= 2:
        q1_conf, q3_conf = np.percentile(confidences, [25, 75])
        iqr_conf = q3_conf - q1_conf
        conf_lo_thresh = max(q1_conf - 1.5 * iqr_conf, threshold)

        conf_mask = confidences >= conf_lo_thresh
        if not conf_mask.any():
            # Fallback: keep highest confidence box
            best_idx = int(np.argmax(confidences))
            conf_mask = np.zeros_like(conf_mask, dtype=bool)
            conf_mask[best_idx] = True

        boxes = boxes[conf_mask]
        confidences = confidences[conf_mask]
        class_names = (
            class_names[conf_mask]
            if class_names.shape[0] == conf_mask.shape[0]
            else class_names
        )

    # Remove boxes fully contained in larger boxes
    if len(boxes) >= 2:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

        contains = (
            (x1[:, None] <= x1[None, :])
            & (y1[:, None] <= y1[None, :])
            & (x2[:, None] >= x2[None, :])
            & (y2[:, None] >= y2[None, :])
        )
        np.fill_diagonal(contains, False)

        larger = areas[:, None] > areas[None, :]
        contained_smaller = contains & larger
        to_remove = np.asarray(contained_smaller.any(axis=0), dtype=bool)

        if to_remove.any():
            kept = ~to_remove
            boxes = boxes[kept]
            if len(confidences) == len(to_remove):
                confidences = confidences[kept]
            if len(class_names) == len(to_remove):
                class_names = class_names[kept]

    if len(boxes) > 0:
        # Apply perspective-aware row-major ordering
        # order = compute_row_major_grid_order(boxes, major="row")
        xs = (boxes[:, 0] + boxes[:, 2]) / 2
        ys = (boxes[:, 1] + boxes[:, 3]) / 2
        order = np.lexsort((xs, ys))
        boxes = boxes[order]
        confidences = (
            confidences[order] if len(confidences) == len(order) else confidences
        )
        class_names = (
            class_names[order] if len(class_names) == len(order) else class_names
        )

    return boxes, confidences, class_names
