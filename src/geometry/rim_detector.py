"""
Gradient-based rim detector for datum holes.

Task 1b Part B4: Fixed-radius circle search on gradient magnitude.

The key insight: the top rim of a through-hole creates a circular edge in gradient
magnitude. Unlike blob detection, this works whether the hole interior is bright
(reflective bore wall) or dark (see-through opening).

Procedure:
1. Compute gradient magnitude
2. For each candidate radius r, score centres by mean gradient on circle of radius r
3. Take the highest-scoring centre (Hough transform with fixed radius)
4. Refine with ellipse fit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class RimFitResult:
    """Result of gradient-based rim detection."""
    # Centre in pixel coordinates
    centre_px: tuple[float, float]   # (u, v) in the source image

    # Hough search result
    hough_score: float               # Mean gradient on the circle
    contrast_ratio: float            # Score / median score in window

    # Free-fit ellipse
    free_fit_centre: tuple[float, float]
    free_fit_axes: tuple[float, float]  # (major, minor) in pixels
    free_fit_angle: float            # Rotation angle in degrees
    free_fit_diameter_mm: float      # Estimated diameter from major axis

    # Fixed-radius refit
    fixed_fit_centre: tuple[float, float]
    fixed_fit_residual: float

    # Quality metrics
    n_edge_points: int
    accepted: bool


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel filters."""
    # Use float to avoid overflow
    gray_f = gray.astype(np.float32)

    # Sobel gradients
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude
    mag = np.sqrt(gx**2 + gy**2)

    return mag


def score_circle_at_centre(
    gradient_mag: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_samples: int = 64,
) -> float:
    """Score a circle centre by mean gradient magnitude on the circle.

    Args:
        gradient_mag: Gradient magnitude image
        cx, cy: Centre coordinates (x=col, y=row)
        radius: Circle radius in pixels
        n_samples: Number of points to sample on circle

    Returns:
        Mean gradient magnitude on the circle
    """
    h, w = gradient_mag.shape

    # Sample points on circle
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)

    # Bilinear interpolation
    total = 0.0
    count = 0

    for x, y in zip(xs, ys):
        if 0 <= x < w - 1 and 0 <= y < h - 1:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            fx, fy = x - x0, y - y0

            val = (
                gradient_mag[y0, x0] * (1 - fx) * (1 - fy) +
                gradient_mag[y0, x1] * fx * (1 - fy) +
                gradient_mag[y1, x0] * (1 - fx) * fy +
                gradient_mag[y1, x1] * fx * fy
            )
            total += val
            count += 1

    return total / count if count > 0 else 0.0


def find_best_centre_hough(
    gradient_mag: np.ndarray,
    expected_radius: float,
    window_centre: tuple[int, int],
    window_radius_px: int,
    step_px: float = 0.5,
) -> tuple[tuple[float, float], float, np.ndarray]:
    """Find the best circle centre using Hough-like voting on gradient.

    Args:
        gradient_mag: Gradient magnitude image
        expected_radius: Expected hole radius in pixels
        window_centre: Centre of search window (x, y)
        window_radius_px: Radius of search window in pixels
        step_px: Step size for grid search

    Returns:
        (best_centre, best_score, score_grid)
    """
    cx0, cy0 = window_centre
    h, w = gradient_mag.shape

    # Create search grid
    xs = np.arange(cx0 - window_radius_px, cx0 + window_radius_px + step_px, step_px)
    ys = np.arange(cy0 - window_radius_px, cy0 + window_radius_px + step_px, step_px)

    # Clip to image bounds
    xs = xs[(xs >= expected_radius) & (xs < w - expected_radius)]
    ys = ys[(ys >= expected_radius) & (ys < h - expected_radius)]

    best_score = 0
    best_centre = (float(cx0), float(cy0))
    scores = []

    for y in ys:
        for x in xs:
            dist = np.sqrt((x - cx0)**2 + (y - cy0)**2)
            if dist <= window_radius_px:
                score = score_circle_at_centre(gradient_mag, x, y, expected_radius)
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_centre = (x, y)

    return best_centre, best_score, np.array(scores)


def refine_with_edge_points(
    gradient_mag: np.ndarray,
    centre: tuple[float, float],
    expected_radius: float,
    radius_tolerance: float = 0.3,  # ±30%
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sub-pixel edge points along radial rays from centre.

    Args:
        gradient_mag: Gradient magnitude image
        centre: Initial centre estimate
        expected_radius: Expected radius in pixels
        radius_tolerance: Fractional tolerance for edge search

    Returns:
        (edge_points, edge_gradients) - Nx2 arrays
    """
    cx, cy = centre
    h, w = gradient_mag.shape

    r_min = expected_radius * (1 - radius_tolerance)
    r_max = expected_radius * (1 + radius_tolerance)

    # Cast radial rays
    n_rays = 72
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    edge_points = []
    edge_gradients = []

    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)

        # Sample along ray
        radii = np.linspace(r_min, r_max, 20)
        ray_grads = []

        for r in radii:
            x, y = cx + r * dx, cy + r * dy
            if 0 <= x < w - 1 and 0 <= y < h - 1:
                # Bilinear interpolation
                x0, y0 = int(x), int(y)
                fx, fy = x - x0, y - y0
                val = (
                    gradient_mag[y0, x0] * (1 - fx) * (1 - fy) +
                    gradient_mag[y0, min(w-1, x0+1)] * fx * (1 - fy) +
                    gradient_mag[min(h-1, y0+1), x0] * (1 - fx) * fy +
                    gradient_mag[min(h-1, y0+1), min(w-1, x0+1)] * fx * fy
                )
                ray_grads.append((r, val))

        if ray_grads:
            # Find peak gradient along ray
            ray_grads = np.array(ray_grads)
            peak_idx = np.argmax(ray_grads[:, 1])
            peak_r = ray_grads[peak_idx, 0]
            peak_grad = ray_grads[peak_idx, 1]

            edge_x = cx + peak_r * dx
            edge_y = cy + peak_r * dy

            edge_points.append([edge_x, edge_y])
            edge_gradients.append(peak_grad)

    return np.array(edge_points), np.array(edge_gradients)


def fit_ellipse_to_points(points: np.ndarray) -> tuple | None:
    """Fit ellipse to edge points.

    Returns:
        ((cx, cy), (major, minor), angle) or None if fit fails
    """
    if len(points) < 5:
        return None

    try:
        # OpenCV fitEllipse expects Nx1x2 or Nx2 integer points
        pts = points.astype(np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(pts)
        return ellipse
    except cv2.error:
        return None


def fit_circle_fixed_radius_from_points(
    points: np.ndarray,
    fixed_radius: float,
    max_iterations: int = 10,
) -> tuple[tuple[float, float], float]:
    """Fit circle with fixed radius, solving for centre only.

    Uses iterative least squares.

    Returns:
        (centre, residual)
    """
    if len(points) < 3:
        return (0, 0), float('inf')

    x, y = points[:, 0], points[:, 1]

    # Initial estimate: centroid
    cx, cy = np.mean(x), np.mean(y)

    for _ in range(max_iterations):
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances = np.maximum(distances, 1e-6)

        # Gradient step
        dcx = np.mean((distances - fixed_radius) * (x - cx) / distances)
        dcy = np.mean((distances - fixed_radius) * (y - cy) / distances)

        cx -= dcx
        cy -= dcy

        if abs(dcx) < 0.01 and abs(dcy) < 0.01:
            break

    # Final residual
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    residual = np.sqrt(np.mean((distances - fixed_radius)**2))

    return (cx, cy), residual


def detect_rim_in_window(
    gray: np.ndarray,
    window_centre: tuple[int, int],
    window_radius_px: int,
    expected_diameter_mm: float,
    depth_mm: float,
    fx: float,
    contrast_threshold: float = 1.5,
) -> RimFitResult | None:
    """Detect hole rim in a search window using gradient-based Hough transform.

    Args:
        gray: Grayscale image
        window_centre: Centre of search window (u, v)
        window_radius_px: Radius of search window in pixels
        expected_diameter_mm: Expected hole diameter in mm
        depth_mm: Depth at hole location in mm
        fx: Colour camera focal length
        contrast_threshold: Minimum contrast ratio for acceptance

    Returns:
        RimFitResult or None if no hole found
    """
    # Expected radius in pixels
    expected_radius_px = (expected_diameter_mm / 2) * fx / depth_mm

    # Compute gradient magnitude
    gradient_mag = compute_gradient_magnitude(gray)

    # Find best centre
    best_centre, best_score, all_scores = find_best_centre_hough(
        gradient_mag,
        expected_radius_px,
        window_centre,
        window_radius_px,
    )

    # Compute contrast ratio
    median_score = np.median(all_scores) if len(all_scores) > 0 else 1.0
    contrast_ratio = best_score / median_score if median_score > 0 else 0.0

    # Accept/reject
    accepted = contrast_ratio >= contrast_threshold

    # Refine with edge points
    edge_points, edge_grads = refine_with_edge_points(
        gradient_mag, best_centre, expected_radius_px
    )

    # Free-fit ellipse
    ellipse = fit_ellipse_to_points(edge_points)

    if ellipse is not None:
        (free_cx, free_cy), (minor, major), angle = ellipse
        free_fit_centre = (free_cx, free_cy)
        free_fit_axes = (max(major, minor), min(major, minor))  # Ensure major > minor
        free_fit_angle = angle
        free_fit_diameter_mm = free_fit_axes[0] * depth_mm / fx  # major axis is already diameter in px
    else:
        free_fit_centre = best_centre
        free_fit_axes = (expected_radius_px * 2, expected_radius_px * 2)
        free_fit_angle = 0.0
        free_fit_diameter_mm = expected_diameter_mm

    # Fixed-radius refit
    if len(edge_points) >= 3:
        fixed_centre, fixed_residual = fit_circle_fixed_radius_from_points(
            edge_points, expected_radius_px
        )
    else:
        fixed_centre = best_centre
        fixed_residual = float('inf')

    return RimFitResult(
        centre_px=fixed_centre,
        hough_score=best_score,
        contrast_ratio=contrast_ratio,
        free_fit_centre=free_fit_centre,
        free_fit_axes=free_fit_axes,
        free_fit_angle=free_fit_angle,
        free_fit_diameter_mm=free_fit_diameter_mm,
        fixed_fit_centre=fixed_centre,
        fixed_fit_residual=fixed_residual,
        n_edge_points=len(edge_points),
        accepted=accepted,
    )


def detect_rim_from_seed(
    gray: np.ndarray,
    seed_px: tuple[int, int],
    expected_diameter_mm: float,
    depth_mm: float,
    fx: float,
    window_radius_mm: float = 12.0,
    contrast_threshold: float = 1.5,
) -> RimFitResult | None:
    """Detect hole rim starting from a manual seed point.

    This is the seeded path (B6 in Task 1b): the seed provides the search window,
    and the algorithm refines the centre and measures the rim.

    Args:
        gray: Grayscale image
        seed_px: Seed point (u, v) from manual clicking
        expected_diameter_mm: Expected hole diameter (6 or 8)
        depth_mm: Depth at this location in mm
        fx: Colour camera focal length
        window_radius_mm: Search window radius in mm (default 12mm)
        contrast_threshold: Minimum contrast ratio for acceptance

    Returns:
        RimFitResult or None
    """
    # Convert window radius to pixels
    window_radius_px = int(window_radius_mm * fx / depth_mm)

    return detect_rim_in_window(
        gray,
        window_centre=seed_px,
        window_radius_px=window_radius_px,
        expected_diameter_mm=expected_diameter_mm,
        depth_mm=depth_mm,
        fx=fx,
        contrast_threshold=contrast_threshold,
    )
