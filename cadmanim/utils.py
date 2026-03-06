"""Utility helpers for triangle-free conversion between AdaptiveCAD and Manim.

All geometry stays analytic: parametric surfaces and 2D contour tracing.
No marching cubes, no triangle meshes.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def vec3_to_array(v) -> np.ndarray:
    """Convert an AdaptiveCAD Vec3 to a numpy array usable by Manim."""
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def array_to_vec3(a: np.ndarray):
    """Convert a 3-element numpy array to an AdaptiveCAD Vec3."""
    from adaptivecad.linalg import Vec3

    return Vec3(float(a[0]), float(a[1]), float(a[2]))


def normalize_points(
    points: np.ndarray,
    fit_radius: float = 2.0,
) -> np.ndarray:
    """Center and scale a point array so it fits in a sphere of *fit_radius*."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    max_extent = np.max(np.linalg.norm(centered, axis=1))
    if max_extent > 0:
        centered *= fit_radius / max_extent
    return centered


def sdf_contours_at_z(
    sdf_func: Callable,
    z_level: float,
    bounds: Tuple[float, float] = (-2.0, 2.0),
    resolution: int = 128,
) -> List[np.ndarray]:
    """Extract 2D zero-contour paths of an SDF at a given z-level.

    Returns a list of (N, 3) arrays, each a contour path in world
    coordinates.  Uses ``skimage.measure.find_contours`` (2D marching-
    squares contour tracing) — produces polyline curves, NOT triangles.
    """
    from skimage.measure import find_contours

    lo, hi = bounds
    x = np.linspace(lo, hi, resolution)
    y = np.linspace(lo, hi, resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")

    try:
        Z = sdf_func(X, Y, np.full_like(X, z_level))
    except Exception:
        Z = np.empty_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = sdf_func(X[i, j], Y[i, j], z_level)

    contours_2d = find_contours(Z, level=0.0)
    step = (hi - lo) / (resolution - 1)
    contours: List[np.ndarray] = []
    for c in contours_2d:
        world_xy = c * step + lo
        pts_3d = np.column_stack([world_xy, np.full(len(world_xy), z_level)])
        contours.append(pts_3d)
    return contours


# ---- Parametric shape factories -------------------------------------------
# Each returns a callable  (u, v) -> np.array([x, y, z])
# All use a *standard* parameter domain  [0, TAU] × [0, TAU]
# so they are directly compatible with MorphBetweenSDFs interpolation.

_PI = np.pi
_TAU = 2.0 * np.pi


def sphere_parametric(radius: float = 1.0) -> Callable:
    """Parametric sphere.  u, v ∈ [0, TAU]."""

    def func(u: float, v: float) -> np.ndarray:
        theta = u * _PI / _TAU          # [0, TAU] → [0, PI]
        return np.array([
            radius * np.sin(theta) * np.cos(v),
            radius * np.cos(theta),
            radius * np.sin(theta) * np.sin(v),
        ])

    return func


def torus_parametric(R: float = 1.0, r: float = 0.4) -> Callable:
    """Parametric torus: major radius *R*, tube radius *r*.  u, v ∈ [0, TAU]."""

    def func(u: float, v: float) -> np.ndarray:
        return np.array([
            (R + r * np.cos(u)) * np.cos(v),
            r * np.sin(u),
            (R + r * np.cos(u)) * np.sin(v),
        ])

    return func


def cylinder_parametric(
    radius: float = 0.5,
    height: float = 2.0,
) -> Callable:
    """Parametric cylinder along the Y axis.  u, v ∈ [0, TAU]."""

    def func(u: float, v: float) -> np.ndarray:
        y = height * (v / _TAU - 0.5)   # [0, TAU] → [-h/2, h/2]
        return np.array([
            radius * np.cos(u),
            y,
            radius * np.sin(u),
        ])

    return func


def bezier_control_points_to_manim(
    control_points,
    close: bool = False,
) -> np.ndarray:
    """Convert a list of AdaptiveCAD control points to a Manim-ready (N, 3) array."""
    pts = np.array([[p.x, p.y, p.z] for p in control_points], dtype=np.float64)
    if close and len(pts) > 1:
        pts = np.vstack([pts, pts[:1]])
    return pts
