"""Custom Manim Mobjects powered by AdaptiveCAD geometry — triangle-free.

All 3D shapes use analytic parametric surfaces (Manim ``Surface``) or
SDF contour slicing (``SDFContourStack``).  No marching cubes, no
explicit triangle meshes.  This aligns with AdaptiveCAD's philosophy
of representing geometry as analytic SDF functions and rendering via
ray-marching, not tessellated meshes.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
from manim import (
    BLUE,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    TAU,
    PI,
    VGroup,
    VMobject,
)
from manim.mobject.three_d.three_dimensions import Surface

from .utils import (
    bezier_control_points_to_manim,
    normalize_points,
    sdf_contours_at_z,
    sphere_parametric,
    torus_parametric,
    cylinder_parametric,
)


# ---------------------------------------------------------------------------
# SDFSurface — analytic parametric surface for SDF-defined shapes
# ---------------------------------------------------------------------------

class SDFSurface(Surface):
    """Analytic parametric surface for SDF-defined shapes.

    Uses Manim's ``Surface`` with smooth parametric functions — no triangle
    meshes.  Aligned with AdaptiveCAD's philosophy: geometry is defined
    analytically, and the renderer handles visualisation.

    For known shapes use the factory classmethods:
        ``SDFSurface.sphere()``, ``SDFSurface.torus()``,
        ``SDFSurface.cylinder()``.

    For custom shapes pass any ``(u, v) -> [x, y, z]`` function directly.
    """

    def __init__(
        self,
        func: Callable,
        u_range=(0, TAU),
        v_range=(0, PI),
        resolution=(32, 32),
        color=BLUE,
        opacity: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            func,
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            fill_color=color,
            fill_opacity=opacity,
            stroke_width=0.5,
            stroke_color=WHITE,
            **kwargs,
        )

    @classmethod
    def sphere(
        cls,
        radius: float = 1.0,
        color=BLUE,
        opacity: float = 0.7,
        resolution=(32, 32),
        **kwargs: Any,
    ) -> "SDFSurface":
        """Analytic sphere of given *radius*."""
        return cls(
            sphere_parametric(radius),
            u_range=[0.01, TAU - 0.01],
            v_range=[0, TAU],
            resolution=resolution,
            color=color,
            opacity=opacity,
            **kwargs,
        )

    @classmethod
    def torus(
        cls,
        R: float = 1.0,
        r: float = 0.4,
        color=GREEN,
        opacity: float = 0.7,
        resolution=(32, 32),
        **kwargs: Any,
    ) -> "SDFSurface":
        """Analytic torus with major radius *R* and tube radius *r*."""
        return cls(
            torus_parametric(R, r),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=resolution,
            color=color,
            opacity=opacity,
            **kwargs,
        )

    @classmethod
    def cylinder(
        cls,
        radius: float = 0.5,
        height: float = 2.0,
        color=YELLOW,
        opacity: float = 0.7,
        resolution=(32, 16),
        **kwargs: Any,
    ) -> "SDFSurface":
        """Analytic cylinder along the Y axis."""
        return cls(
            cylinder_parametric(radius, height),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=resolution,
            color=color,
            opacity=opacity,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# SDFContourStack — stacked contour slices (triangle-free)
# ---------------------------------------------------------------------------

class SDFContourStack(VGroup):
    """Visualise an arbitrary SDF as stacked contour slices — triangle-free.

    Extracts zero-level contours at multiple z-levels through the SDF
    volume and renders each as a smooth ``VMobject`` curve.  The result
    is a layered cross-section visualisation that preserves the analytic
    / implicit nature of the SDF — no triangulation.
    """

    def __init__(
        self,
        sdf_func: Callable,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        n_slices: int = 24,
        resolution: int = 128,
        color=BLUE,
        opacity: float = 0.8,
        stroke_width: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        lo, hi = bounds
        z_levels = np.linspace(lo * 0.95, hi * 0.95, n_slices)

        for z in z_levels:
            contours = sdf_contours_at_z(sdf_func, z, bounds, resolution)
            for pts in contours:
                if len(pts) < 3:
                    continue
                curve = VMobject(
                    color=color,
                    stroke_width=stroke_width,
                    fill_opacity=opacity * 0.3,
                    fill_color=color,
                )
                curve.set_points_smoothly(pts)
                self.add(curve)


# ---------------------------------------------------------------------------
# Bezier / BSpline Curve Mobject
# ---------------------------------------------------------------------------

class BezierCurveMobject(VMobject):
    """Render an AdaptiveCAD ``BezierCurve`` or ``BSplineCurve`` as a Manim path.

    Parameters
    ----------
    curve
        An AdaptiveCAD curve object that has a ``.evaluate(t)`` method
        returning Vec3 or an ``.control_points`` attribute.
    samples : int
        Number of evaluation samples along the curve.
    """

    def __init__(
        self,
        curve,
        samples: int = 200,
        color=YELLOW,
        stroke_width: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, stroke_width=stroke_width, **kwargs)
        ts = np.linspace(0, 1, samples)
        points = []
        for t in ts:
            pt = curve.evaluate(t)
            points.append([pt.x, pt.y, pt.z])
        self.set_points_smoothly(np.array(points))

    @classmethod
    def from_control_points(
        cls,
        control_points,
        **kwargs: Any,
    ) -> "BezierCurveMobject":
        """Build from an AdaptiveCAD BezierCurve's control points."""
        from adaptivecad.geom import BezierCurve

        curve = BezierCurve(list(control_points))
        return cls(curve, **kwargs)


# ---------------------------------------------------------------------------
# 2D Sketch Mobject
# ---------------------------------------------------------------------------

class SketchMobject(VGroup):
    """Render an AdaptiveCAD ``SketchDocument`` as a 2D Manim overlay.

    Converts lines, arcs, circles, and polylines from a sketch into
    Manim ``VMobject`` paths.
    """

    def __init__(
        self,
        sketch_doc,
        color=WHITE,
        stroke_width: float = 2.0,
        scale_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for entity in sketch_doc.entities:
            mob = self._entity_to_mobject(entity, color, stroke_width, scale_factor)
            if mob is not None:
                self.add(mob)

    @staticmethod
    def _entity_to_mobject(entity, color, stroke_width, scale_factor):
        from manim import Line as MLine, Circle as MCircle, Arc as MArc

        kind = type(entity).__name__
        if kind == "Line":
            p1 = np.array([entity.start.x, entity.start.y, 0]) * scale_factor
            p2 = np.array([entity.end.x, entity.end.y, 0]) * scale_factor
            return MLine(p1, p2, color=color, stroke_width=stroke_width)
        elif kind == "Circle":
            c = MCircle(radius=entity.radius * scale_factor, color=color, stroke_width=stroke_width)
            c.move_to(np.array([entity.center.x, entity.center.y, 0]) * scale_factor)
            return c
        elif kind == "Polyline":
            pts = np.array([[p.x, p.y, 0] for p in entity.points]) * scale_factor
            mob = VMobject(color=color, stroke_width=stroke_width)
            mob.set_points_smoothly(pts)
            return mob
        return None


# ---------------------------------------------------------------------------
# Toolpath Mobject — animate CNC / 3D-print paths
# ---------------------------------------------------------------------------

class ToolpathMobject(VGroup):
    """Visualise a G-code or linear toolpath as a coloured Manim trail.

    Parameters
    ----------
    points : sequence of Vec3 or (N, 3) array
        Ordered toolpath positions.
    color_start, color_end
        Gradient start/end colours.
    """

    def __init__(
        self,
        points,
        color_start=GREEN,
        color_end=RED,
        stroke_width: float = 2.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hasattr(points[0], "x"):
            pts = np.array([[p.x, p.y, p.z] for p in points])
        else:
            pts = np.asarray(points, dtype=np.float64)

        # Normalise to Manim coordinate range
        pts = normalize_points(pts, fit_radius=3.0)

        path = VMobject(stroke_width=stroke_width)
        path.set_points_smoothly(pts)
        path.set_color_by_gradient(color_start, color_end)
        self.add(path)
