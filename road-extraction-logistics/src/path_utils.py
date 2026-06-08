import numpy as np
from typing import List, Tuple, Optional


def _perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Perpendicular distance from a point to a line segment."""
    if np.allclose(line_start, line_end):
        return float(np.linalg.norm(point - line_start))
    d = line_end - line_start
    t = np.dot(point - line_start, d) / np.dot(d, d)
    t = np.clip(t, 0, 1)
    projection = line_start + t * d
    return float(np.linalg.norm(point - projection))


def _rdp(points: np.ndarray, epsilon: float, start: int, end: int, keep: list):
    """Recursive Ramer-Douglas-Peucker over index range [start, end]."""
    if end <= start + 1:
        return
    max_dist = 0.0
    max_idx  = start
    for i in range(start + 1, end):
        d = _perpendicular_distance(points[i], points[start], points[end])
        if d > max_dist:
            max_dist = d
            max_idx  = i
    if max_dist > epsilon:
        keep.append(max_idx)
        _rdp(points, epsilon, start, max_idx, keep)
        _rdp(points, epsilon, max_idx, end, keep)


def smooth_path(path: List[Tuple[int, int]], epsilon: float = 2.0) -> List[Tuple[int, int]]:
    """
    Simplify a pixel-level path using the Ramer-Douglas-Peucker algorithm.

    Removes intermediate points that deviate less than `epsilon` pixels from
    the straight line between their neighbours, eliminating the staircase
    pattern produced by 8-connected A*.

    Args:
        path:    List of (x, y) pixel coordinates.
        epsilon: Maximum allowed deviation in pixels (default 2.0).

    Returns:
        Simplified list of (x, y) coordinates (always includes endpoints).
    """
    if len(path) <= 2:
        return path

    pts = np.array(path, dtype=float)
    keep = {0, len(pts) - 1}
    _rdp(pts, epsilon, 0, len(pts) - 1, list(keep))

    # Re-collect all kept indices then sort
    final_keep: list = [0, len(pts) - 1]
    _rdp(pts, epsilon, 0, len(pts) - 1, final_keep)
    final_keep = sorted(set(final_keep))

    return [path[i] for i in final_keep]


def compute_path_distance(
    path: List[Tuple[int, int]],
    meters_per_pixel: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    """
    Compute the cumulative pixel length of a path, optionally converting to metres.

    Args:
        path:             List of (x, y) coordinates.
        meters_per_pixel: Ground sampling distance in m/px.
                          Pass config.PIXEL_RESOLUTION_METERS for real-world distances.
                          If None, only pixel distance is returned.

    Returns:
        (pixel_distance, meter_distance) — meter_distance is None when
        meters_per_pixel is not provided.
    """
    if len(path) < 2:
        return 0.0, None

    arr   = np.array(path, dtype=float)
    diffs = np.diff(arr, axis=0)
    px    = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
    m     = px * meters_per_pixel if meters_per_pixel is not None else None
    return px, m
