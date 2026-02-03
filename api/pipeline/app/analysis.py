import numpy as np
import logging

logger = logging.getLogger(__name__)

# Configuration
TUKEY_K_UPPER = 4.0
CLEAN_AREA_LOWER_THRESHOLD = 0.1
CLEAN_AREA_UPPER_THRESHOLD = 1.5
MINIMUM_AREA_COUNT = 1
EWM_BETA = 0.1

MORPHOLOGY_FEATURES = [
    "in_bounds",
    "area",
    "convex_hull_area",
    "solidity",
    "perimeter",
    "width",
    "height",
    "longest_path",
    "center_of_mass_x",
    "center_of_mass_y",
    "convex_hull_vertices",
    "object_in_frame",
    "ellipse_center_x",
    "ellipse_center_y",
    "ellipse_major_axis",
    "ellipse_minor_axis",
    "ellipse_angle",
    "ellipse_eccentricity",
    "blue-yellow_frequencies_mean",
    "blue_frequencies_mean",
    "green-magenta_frequencies_mean",
    "green_frequencies_mean",
    "hue_circular_mean",
    "hue_circular_std",
    "hue_frequencies_mean",
    "lightness_frequencies_mean",
    "red_frequencies_mean",
    "saturation_frequencies_mean",
    "value_frequencies_mean",
]


def apply_tukey_outlier_detection_frame(plant_stats_list):
    """
    Apply Tukey outlier detection to a list of plant stats dictionaries for a single frame.

    Modifies the dictionaries in-place to add 'tukey_outlier' and 'area_after_tukey'.
    """
    if not plant_stats_list:
        return plant_stats_list

    # Extract valid areas
    areas = []
    indices = []

    for i, stats in enumerate(plant_stats_list):
        if stats and "area" in stats and stats["area"] is not None:
            areas.append(stats["area"])
            indices.append(i)

    if len(areas) < 4:
        # Not enough data for reliable stats, assume no outliers
        for stats in plant_stats_list:
            if stats:
                stats["tukey_outlier"] = False
                # If area is None, area_after_tukey should be None
                stats["area_after_tukey"] = stats.get("area")
        return plant_stats_list

    # Compute quartiles
    q1 = np.percentile(areas, 25)
    q3 = np.percentile(areas, 75)
    iqr = q3 - q1
    upper_fence = q3 + TUKEY_K_UPPER * iqr

    # Apply fences
    for i, stats in enumerate(plant_stats_list):
        if not stats:
            continue

        area = stats.get("area")
        is_outlier = False
        area_after = None

        if area is not None:
            if area > upper_fence:
                is_outlier = True
                area_after = None
            else:
                is_outlier = False
                area_after = area

        stats["tukey_outlier"] = is_outlier
        stats["area_after_tukey"] = area_after

    return plant_stats_list


def update_plant_cleaning_state(stats, state, current_mask=None):
    """
    Update cleaning state for a single plant using EWM logic.

    Args:
        stats: Current plant statistics dictionary
        state: Previous cleaning state dictionary
        current_mask: Optional current mask dictionary (box/contour) to persist

    Returns:
        updated_state: New state dictionary
        clean_stats: Dictionary containing cleaned features
    """
    # Initialize state if empty
    if not state:
        state = {
            "ewm_sum": 0.0,
            "ewm_weight": 0.0,
            "area_count": 0,
            "prev_clean_values": {f: 0.0 for f in MORPHOLOGY_FEATURES},
            "prev_clean_area": None,
            "prev_clean_mask": None,
        }
        # Ensure clean_area is present in prev_clean_values if not in MORPHOLOGY_FEATURES
        if "clean_area" not in state["prev_clean_values"]:
            state["prev_clean_values"]["clean_area"] = 0.0

    # Extract current values
    if stats is None:
        area_tukey = None
        is_missing = True
    else:
        area_tukey = stats.get("area_after_tukey")
        is_missing = False

    # Prepare output stats with clean prefix
    clean_stats = {}

    # State variables
    ewm_sum = state.get("ewm_sum", 0.0)
    ewm_weight = state.get("ewm_weight", 0.0)
    area_count = state.get("area_count", 0)
    prev_clean_values = state.get("prev_clean_values", {})
    prev_clean_area = state.get("prev_clean_area", 0.0)
    prev_clean_mask = state.get("prev_clean_mask")

    alpha = 1.0 - EWM_BETA

    is_invalid = area_tukey is None

    # Calculate current EWM (for outlier check)
    if ewm_weight > 0:
        current_ewm = ewm_sum / ewm_weight
    else:
        # First observation or reset
        current_ewm = area_tukey if not is_invalid else 0.0

    is_outlier = False

    if is_invalid:
        is_outlier = True
    elif area_count >= MINIMUM_AREA_COUNT and prev_clean_area is not None:
        lower_bound = (1.0 - CLEAN_AREA_LOWER_THRESHOLD) * current_ewm
        upper_bound = (1.0 + CLEAN_AREA_UPPER_THRESHOLD) * current_ewm
        is_outlier = area_tukey < lower_bound or area_tukey > upper_bound

    if is_outlier:
        # use previous clean values
        for f in MORPHOLOGY_FEATURES:
            clean_stats[f"clean_{f}"] = prev_clean_values.get(f, 0.0)

        new_state = state.copy()
        clean_stats["uema_area"] = current_ewm
        new_state = state.copy()
        clean_stats["uema_area"] = current_ewm
        clean_stats["is_outlier"] = True
        if is_missing:
            clean_stats["is_missing"] = True

        # If we have a stored clean mask, return it to be reused
        clean_mask = prev_clean_mask
    else:
        # accept new values
        new_ewm_sum = ewm_sum * (1.0 - alpha) + area_tukey
        new_ewm_weight = ewm_weight * (1.0 - alpha) + 1.0
        new_area_count = area_count + 1

        # Update prev clean values
        new_prev_clean_values = prev_clean_values.copy()

        for f in MORPHOLOGY_FEATURES:
            val = stats.get(f)
            clean_stats[f"clean_{f}"] = val
            new_prev_clean_values[f] = val

        clean_stats["uema_area"] = new_ewm_sum / new_ewm_weight
        clean_stats["is_outlier"] = False

        new_state = {
            "ewm_sum": new_ewm_sum,
            "ewm_weight": new_ewm_weight,
            "area_count": new_area_count,
            "prev_clean_values": new_prev_clean_values,
            "prev_clean_area": area_tukey,
            "prev_clean_mask": current_mask if current_mask else prev_clean_mask,
        }

        # For valid observations, the "clean mask" is just the current one
        clean_mask = current_mask

    return new_state, clean_stats, clean_mask
