import cv2
import numpy as np


def safe_fill_poly(mask, contours, color=1):
    """
    Safely call cv2.fillPoly by checking for empty or invalid contours.
    Handles both a single contour [[x,y],...] or list of contours [[[x,y],...],...].
    """
    if contours is None or len(contours) == 0:
        return mask

    # Standardize to a list of numpy arrays
    # If it's already a numpy array (e.g. from scale_contour), it's a single contour
    if isinstance(contours, np.ndarray):
        if contours.size > 0:
            cv2.fillPoly(mask, [contours.astype(np.int32)], color)
        return mask

    # If it's a list, peek at the first element to see if it's a point [x, y]
    first = contours[0]
    if (
        isinstance(first, (list, tuple, np.ndarray))
        and len(first) == 2
        and np.isscalar(first[0])
    ):
        # Single contour: [[x,y], [x,y], ...]
        pts = np.array(contours, dtype=np.int32)
        if pts.size > 0:
            cv2.fillPoly(mask, [pts], color)
    else:
        # List of contours: [[[x,y],...], [[x,y],...]]
        valid = []
        for c in contours:
            if c is not None:
                pts = np.array(c, dtype=np.int32)
                if pts.size > 0:
                    valid.append(pts)
        if valid:
            cv2.fillPoly(mask, valid, color)

    return mask
