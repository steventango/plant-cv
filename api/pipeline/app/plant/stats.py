import logging
import time
from contextlib import contextmanager

import cv2
import numpy as np
from plantcv import plantcv as pcv
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@contextmanager
def _timed_accumulate(profile: dict | None, key: str):
    """Accumulate elapsed seconds into profile['analyze_plant_mask_detail'][key]."""
    if profile is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings = profile.setdefault("analyze_plant_mask_detail", {})
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)


def _increment_profile_count(profile: dict | None) -> None:
    if profile is None:
        return
    timings = profile.setdefault("analyze_plant_mask_detail", {})
    timings["count"] = timings.get("count", 0) + 1


def calculate_scale_factor(
    image_width: int, pot_size_mm: float = 60.0, margin: float = 0.25
) -> float:
    """
    Calculate the scale factor (pixels per mm) for a warped image.
    """
    # Warped image includes margin, so total size = pot_size * (1 + 2*margin)
    total_size_mm = pot_size_mm * (1.0 + 2.0 * margin)
    # Assume square warped image
    return image_width / total_size_mm


def convert_px_to_mm(metrics: dict, scale: float) -> dict:
    """
    Convert pixel-based measurements to mm.
    """
    area_metrics = ["area", "convex_hull_area"]
    linear_metrics = [
        "perimeter",
        "width",
        "height",
        "longest_path",
        "ellipse_major_axis",
        "ellipse_minor_axis",
    ]

    for metric in area_metrics:
        if metric in metrics:
            metrics[metric] = metrics[metric] / (scale**2)

    for metric in linear_metrics:
        if metric in metrics:
            metrics[metric] = metrics[metric] / scale

    return metrics


def calculate_histogram_mean(histogram: list) -> float:
    """
    Calculate the mean value from a frequency histogram.
    """
    hist = np.array(histogram)
    if hist.sum() == 0:
        return 0.0

    values = np.arange(len(hist))
    repeated_values = np.repeat(values, hist.astype(int))

    if len(repeated_values) == 0:
        return 0.0

    return float(np.mean(repeated_values))


_COLOR_MEAN_KEYS = (
    "red_frequencies_mean",
    "green_frequencies_mean",
    "blue_frequencies_mean",
    "green-magenta_frequencies_mean",
    "blue-yellow_frequencies_mean",
    "hue_frequencies_mean",
    "hue_circular_mean",
    "hue_circular_std",
    "saturation_frequencies_mean",
    "value_frequencies_mean",
    "lightness_frequencies_mean",
)


def _circular_hue_mean_and_std(hue: np.ndarray) -> tuple[float, float]:
    """Circular mean/std of OpenCV-encoded hue (0-179), reported in degrees (0-359).

    Mirrors plantcv.analyze.color: hue==0 marks red/undefined-hue (saturated)
    pixels and is excluded, and an all-zero/empty selection yields NaN (not 0.0),
    matching scipy.stats.circmean/circstd on an empty array.
    """
    nonzero = hue[hue > 0]
    if nonzero.size == 0:
        return float("nan"), float("nan")
    mean = float(scipy_stats.circmean(nonzero, high=179, low=0) * 2)
    std = float(scipy_stats.circstd(nonzero, high=179, low=0) * 2)
    return mean, std


def _color_means_for_masked_region(
    warped_image_np: np.ndarray, mask: np.ndarray
) -> dict:
    """Mean colour stats over the masked region, matching plantcv.analyze.color.

    plantcv treats the input array as BGR: it splits the channels as ``b, g, r``
    and converts with ``COLOR_BGR2LAB`` / ``COLOR_BGR2HSV``. The pipeline feeds
    RGB images, so we replicate that exact (BGR-interpreting) derivation and key
    mapping so these stats mean the same thing they always have — i.e. plantcv's
    ``red_frequencies`` channel is array channel 2, ``blue_frequencies`` is array
    channel 0, and LAB/HSV are computed on the BGR-interpreted array. Scales also
    follow plantcv's observation labels: RGB→0-255, L/S/V→0-100 (percent_values),
    a*/b*→-128..127 (diverging_values), hue→degrees.
    """
    idx = mask > 0
    if not np.any(idx):
        means = {k: 0.0 for k in _COLOR_MEAN_KEYS}
        # plantcv reports NaN hue circular stats when there are no hue>0 pixels.
        means["hue_circular_mean"] = float("nan")
        means["hue_circular_std"] = float("nan")
        return means

    # Crop to the mask bounding box so the colour-space conversions and the
    # per-channel reductions only touch the plant region, not the full frame.
    ys, xs = np.nonzero(idx)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    img = warped_image_np[y0:y1, x0:x1]
    sel = idx[y0:y1, x0:x1]

    # plantcv labels the split channels b, g, r assuming a BGR array; index first,
    # then convert only the surviving masked pixels to float.
    blue = img[..., 0][sel].astype(np.float32)
    green = img[..., 1][sel].astype(np.float32)
    red = img[..., 2][sel].astype(np.float32)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0][sel].astype(np.float32) * 100.0 / 255.0
    green_magenta = lab[..., 1][sel].astype(np.float32) - 128.0
    blue_yellow = lab[..., 2][sel].astype(np.float32) - 128.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0][sel].astype(np.float32)
    # plantcv labels saturation and value with percent_values (0-100), exactly
    # as it does lightness, so rescale both from the encoded 0-255 range.
    saturation = hsv[..., 1][sel].astype(np.float32) * 100.0 / 255.0
    value = hsv[..., 2][sel].astype(np.float32) * 100.0 / 255.0

    hue_circular_mean, hue_circular_std = _circular_hue_mean_and_std(hue)

    return {
        "red_frequencies_mean": float(red.mean()),
        "green_frequencies_mean": float(green.mean()),
        "blue_frequencies_mean": float(blue.mean()),
        "green-magenta_frequencies_mean": float(green_magenta.mean()),
        "blue-yellow_frequencies_mean": float(blue_yellow.mean()),
        # plantcv labels hue bin i with the interval midpoint i*2+1 (degrees),
        # so the label-weighted mean is 2*mean(hue) + 1.
        "hue_frequencies_mean": float((hue * 2.0 + 1.0).mean()),
        "hue_circular_mean": hue_circular_mean,
        "hue_circular_std": hue_circular_std,
        "saturation_frequencies_mean": float(saturation.mean()),
        "value_frequencies_mean": float(value.mean()),
        "lightness_frequencies_mean": float(lightness.mean()),
    }


def analyze_plant_mask(
    warped_image_np: np.ndarray,
    mask_binary: np.ndarray,
    pot_size_mm: float = 60.0,
    margin: float = 0.25,
    profile: dict | None = None,
):
    """
    Analyze a plant mask using PlantCV to extract morphological features.

    Args:
        warped_image_np: Warped pot image as numpy array (RGB)
        mask_binary: Binary plant mask (0 or 255) as numpy array
        pot_size_mm: Physical size of pot in mm (default: 60mm)
        margin: Margin ratio in warped image (default: 0.25 = 25%)

    Returns:
        stats: Dict with plant statistics
        visualization: Numpy array of visualization image (or None)
    """

    with _timed_accumulate(profile, "total"):
        _increment_profile_count(profile)

        # Calculate scale factor: pixels per mm
        image_height, image_width = warped_image_np.shape[:2]
        scale = calculate_scale_factor(image_width, pot_size_mm, margin)

        labeled_mask = np.where(mask_binary > 0, 1, 0).astype(np.uint8)
        n_pixels = np.sum(labeled_mask)
        print(
            f"DEBUG: Analyze plant mask: scale={scale:.4f}, plant_pixels={n_pixels}",
            flush=True,
        )
        logger.debug(f"Analyze plant mask: scale={scale:.4f}, plant_pixels={n_pixels}")

        # Set PlantCV parameters
        pcv.params.line_thickness = 2
        pcv.params.debug = None

        # Clear previous observations
        pcv.outputs.clear()

        try:
            # Size analysis
            with _timed_accumulate(profile, "size_analysis"):
                analysis_image = pcv.analyze.size(
                    img=warped_image_np, labeled_mask=labeled_mask, n_labels=1
                )

            # Fast custom color means analysis
            with _timed_accumulate(profile, "color_analysis"):
                color_stats = _color_means_for_masked_region(
                    warped_image_np, labeled_mask > 0
                )

        except Exception as e:
            # Return empty stats if analysis fails
            logger.error(f"PlantCV analysis failed: {e}")
            return {
                "area": None,
                "convex_hull_area": None,
                "solidity": None,
                "perimeter": None,
                "width": None,
                "height": None,
                "longest_path": None,
                "center_of_mass_x": None,
                "center_of_mass_y": None,
                "convex_hull_vertices": 0,
                "object_in_frame": 0,
                "ellipse_center_x": None,
                "ellipse_center_y": None,
                "ellipse_major_axis": None,
                "ellipse_minor_axis": None,
                "ellipse_angle": None,
                "ellipse_eccentricity": None,
                **{k: None for k in _COLOR_MEAN_KEYS},
                "error": str(e),
            }, None

        stats = {}

        # pcv.outputs.observations is a dict of observations from analyze.size
        # (color stats are computed separately via _color_means_for_masked_region).
        with _timed_accumulate(profile, "observations"):
            for observation_label, observation in pcv.outputs.observations.items():
                for variable, value in observation.items():
                    if variable in ["center_of_mass", "ellipse_center"]:
                        stats[variable + "_x"], stats[variable + "_y"] = value["value"]
                    elif "frequencies" in variable:
                        stats[f"{variable}_mean"] = calculate_histogram_mean(
                            value["value"]
                        )
                    else:
                        stats[variable] = value["value"]

        stats.update(color_stats)

        with _timed_accumulate(profile, "convert_px_to_mm"):
            stats = convert_px_to_mm(stats, scale)

    return stats, analysis_image
