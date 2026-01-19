import numpy as np
from plantcv import plantcv as pcv
import logging

logger = logging.getLogger(__name__)


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


def analyze_plant_mask(
    warped_image_np: np.ndarray,
    mask_binary: np.ndarray,
    pot_size_mm: float = 60.0,
    margin: float = 0.25,
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
        analysis_image = pcv.analyze.size(
            img=warped_image_np, labeled_mask=labeled_mask, n_labels=1
        )

        # Color analysis
        pcv.analyze.color(
            rgb_img=warped_image_np,
            labeled_mask=labeled_mask,
            n_labels=1,
            colorspaces="all",
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
            "blue-yellow_frequencies_mean": None,
            "blue_frequencies_mean": None,
            "green-magenta_frequencies_mean": None,
            "green_frequencies_mean": None,
            "hue_circular_mean": None,
            "hue_circular_std": None,
            "hue_frequencies_mean": None,
            "lightness_frequencies_mean": None,
            "red_frequencies_mean": None,
            "saturation_frequencies_mean": None,
            "value_frequencies_mean": None,
            "error": str(e),
        }, None

    stats = {}

    # pcv.outputs.observations is a dict of observations
    # For analyze.size and analyze.color, they usually share the same sample name if n_labels=1
    for observation_label, observation in pcv.outputs.observations.items():
        for variable, value in observation.items():
            if variable in ["center_of_mass", "ellipse_center"]:
                stats[variable + "_x"], stats[variable + "_y"] = value["value"]
            elif "frequencies" in variable:
                stats[f"{variable}_mean"] = calculate_histogram_mean(value["value"])
            else:
                stats[variable] = value["value"]

    stats = convert_px_to_mm(stats, scale)

    return stats, analysis_image
