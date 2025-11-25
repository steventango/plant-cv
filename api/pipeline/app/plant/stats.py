import numpy as np
from plantcv import plantcv as pcv


def calculate_scale_factor(image_width: int, pot_size_mm: float = 60.0, margin: float = 0.25) -> float:
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


def analyze_plant_mask(
    warped_image_np: np.ndarray,
    mask_binary: np.ndarray,
    pot_size_mm: float = 60.0,
    margin: float = 0.25,
    visualize: bool = False,
):
    """
    Analyze a plant mask using PlantCV to extract morphological features.

    Args:
        warped_image_np: Warped pot image as numpy array (RGB)
        mask_binary: Binary plant mask (0 or 255) as numpy array
        pot_size_mm: Physical size of pot in mm (default: 60mm)
        margin: Margin ratio in warped image (default: 0.25 = 25%)
        visualize: Whether to return a visualization image

    Returns:
        stats: Dict with plant statistics
        visualization: Numpy array of visualization image (or None)
    """

    # Calculate scale factor: pixels per mm
    image_height, image_width = warped_image_np.shape[:2]
    scale = calculate_scale_factor(image_width, pot_size_mm, margin)

    # Prepare labeled mask for PlantCV (label 1 for plant)
    labeled_mask = np.where(mask_binary > 0, 1, 0).astype(np.uint8)

    # Set PlantCV parameters
    pcv.params.line_thickness = 2
    pcv.params.debug = "print" if visualize else None

    # Clear previous observations
    pcv.outputs.clear()

    visualization = None

    try:
        analysis_image = pcv.analyze.size(
            img=warped_image_np, labeled_mask=labeled_mask, n_labels=1
        )

        if visualize and analysis_image is not None:
            visualization = analysis_image

    except Exception as e:
        # Return empty stats if analysis fails
        return {
            "area": 0.0,
            "convex_hull_area": 0.0,
            "solidity": 0.0,
            "perimeter": 0.0,
            "width": 0.0,
            "height": 0.0,
            "longest_path": 0.0,
            "center_of_mass_x": 0.0,
            "center_of_mass_y": 0.0,
            "convex_hull_vertices": 0,
            "object_in_frame": 0,
            "ellipse_center_x": 0.0,
            "ellipse_center_y": 0.0,
            "ellipse_major_axis": 0.0,
            "ellipse_minor_axis": 0.0,
            "ellipse_angle": 0.0,
            "ellipse_eccentricity": 0.0,
            "error": str(e),
        }, None

    stats = {}

    observation = next(iter(pcv.outputs.observations.values()))

    for variable, value in observation.items():
        if variable in ["center_of_mass", "ellipse_center"]:
            stats[variable + "_x"], stats[variable + "_y"] = value["value"]
        else:
            stats[variable] = value["value"]

    stats = convert_px_to_mm(stats, scale)

    return stats, visualization