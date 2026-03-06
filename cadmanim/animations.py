"""Custom Manim animations tailored for CAD visualisation workflows.

Includes exploded-view animations, toolpath tracing, parametric SDF
morphing, assembly sequences, and curve growth — all triangle-free.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
from manim import (
    Animation,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Mobject,
    ReplacementTransform,
    Succession,
    Transform,
    VGroup,
    VMobject,
    rate_functions,
    BLUE,
    PI,
    TAU,
)
from manim.mobject.three_d.three_dimensions import Surface


# ---------------------------------------------------------------------------
# Exploded View
# ---------------------------------------------------------------------------

class AnimateExplodedView(Animation):
    """Push parts of a VGroup outward from a common centre.

    Parameters
    ----------
    group : VGroup
        The collection of CAD parts (each a sub-Mobject).
    scale_factor : float
        How far to push parts outward (1.0 = original, 2.0 = twice as far).
    """

    def __init__(
        self,
        group: VGroup,
        scale_factor: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self.scale_factor = scale_factor
        self.center = group.get_center()
        self.original_positions = [m.get_center().copy() for m in group]
        super().__init__(group, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        for mob, orig_pos in zip(self.mobject, self.original_positions):
            direction = orig_pos - self.center
            target = self.center + direction * (1 + (self.scale_factor - 1) * alpha)
            mob.move_to(target)


# ---------------------------------------------------------------------------
# Assembly (reverse exploded view, snap parts together)
# ---------------------------------------------------------------------------

class AnimateAssembly(Animation):
    """Animate parts converging to their assembled positions.

    Provide a VGroup whose sub-mobjects are already in their *assembled*
    positions.  The animation starts them spread out and snaps them together.
    """

    def __init__(
        self,
        group: VGroup,
        spread: float = 3.0,
        **kwargs: Any,
    ) -> None:
        self.spread = spread
        self.center = group.get_center()
        self.final_positions = [m.get_center().copy() for m in group]
        # Push parts outward before starting
        for mob, pos in zip(group, self.final_positions):
            direction = pos - self.center
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                mob.move_to(self.center + direction / norm * self.spread)
        super().__init__(group, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        for mob, final_pos in zip(self.mobject, self.final_positions):
            current_start = mob.get_center()
            # Smoothly interpolate toward final position
            t = rate_functions.smooth(alpha)
            mob.move_to(current_start + (final_pos - current_start) * t)


# ---------------------------------------------------------------------------
# Toolpath trace
# ---------------------------------------------------------------------------

class AnimateToolpath(Animation):
    """Progressively reveal a toolpath (like a CNC cutter moving).

    Parameters
    ----------
    toolpath_mob : VMobject
        A ``ToolpathMobject`` or any VMobject representing the path.
    """

    def __init__(self, toolpath_mob: VMobject, **kwargs: Any) -> None:
        self._full_points = toolpath_mob.points.copy()
        super().__init__(toolpath_mob, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        n = len(self._full_points)
        end_idx = max(1, int(alpha * n))
        self.mobject.points = self._full_points[:end_idx].copy()


# ---------------------------------------------------------------------------
# Curve Growth — draw an AdaptiveCAD curve over time
# ---------------------------------------------------------------------------

class AnimateCurveGrowth(Animation):
    """Grow a ``BezierCurveMobject`` from t=0 to t=1 over the animation.

    Similar to Manim's ``Create`` but specifically tuned for parametric curves
    so intermediate frames are geometrically accurate.
    """

    def __init__(
        self,
        curve_mob: VMobject,
        curve_evaluator: Optional[Callable] = None,
        samples: int = 200,
        **kwargs: Any,
    ) -> None:
        self._curve_eval = curve_evaluator
        self._samples = samples
        self._full_points = curve_mob.points.copy()
        super().__init__(curve_mob, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        if self._curve_eval is not None:
            n = max(2, int(alpha * self._samples))
            ts = np.linspace(0, alpha, n)
            pts = []
            for t in ts:
                p = self._curve_eval(t)
                pts.append([p.x, p.y, p.z])
            self.mobject.set_points_smoothly(np.array(pts))
        else:
            # Fall back to slicing pre-computed points
            total = len(self._full_points)
            end_idx = max(1, int(alpha * total))
            self.mobject.points = self._full_points[:end_idx].copy()


# ---------------------------------------------------------------------------
# SDF Morph — parametric interpolation between two shapes (triangle-free)
# ---------------------------------------------------------------------------

class MorphBetweenSDFs(Animation):
    """Morph between two shapes using parametric surface interpolation.

    Triangle-free: builds intermediate ``Surface`` mobjects by blending
    two parametric functions analytically:

        surface(u, v, t) = (1 - t) × param_a(u, v)  +  t × param_b(u, v)

    This produces geometrically smooth intermediates without any
    triangulation — aligned with AdaptiveCAD's analytic philosophy.

    Parameters
    ----------
    target_mob : Surface
        The Manim mobject to animate.
    param_a, param_b : callable
        Parametric functions ``(u, v) -> np.array([x, y, z])``.
        Use helpers from ``cadmanim.utils``:
        ``sphere_parametric(r)``, ``torus_parametric(R, r)``, etc.
    u_range, v_range : list
        Parameter ranges for the Surface (default ``[0, TAU]``).
    keyframes : int
        Number of pre-computed intermediate surfaces.
    """

    def __init__(
        self,
        target_mob: Mobject,
        param_a: Callable,
        param_b: Callable,
        u_range: list | None = None,
        v_range: list | None = None,
        resolution: tuple = (24, 24),
        keyframes: int = 12,
        color=None,
        opacity: float = 0.7,
        **kwargs: Any,
    ) -> None:
        if u_range is None:
            u_range = [0, TAU]
        if v_range is None:
            v_range = [0, TAU]
        if color is None:
            try:
                color = target_mob.get_color()
            except Exception:
                color = BLUE

        self._keyframe_surfaces: list[Surface] = []
        for i in range(keyframes + 1):
            t = i / keyframes

            def blended(u, v, _t=t):
                a = np.asarray(param_a(u, v), dtype=np.float64)
                b = np.asarray(param_b(u, v), dtype=np.float64)
                return (1 - _t) * a + _t * b

            surf = Surface(
                blended,
                u_range=u_range,
                v_range=v_range,
                resolution=resolution,
                fill_color=color,
                fill_opacity=opacity,
                stroke_width=0.5,
            )
            self._keyframe_surfaces.append(surf)

        super().__init__(target_mob, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        idx = int(alpha * (len(self._keyframe_surfaces) - 1))
        idx = min(idx, len(self._keyframe_surfaces) - 1)
        self.mobject.become(self._keyframe_surfaces[idx])


# ---------------------------------------------------------------------------
# Convenience: lagged build for multiple parts
# ---------------------------------------------------------------------------

def lagged_cad_build(parts: VGroup, lag_ratio: float = 0.3, **kwargs) -> LaggedStart:
    """Convenience wrapper: ``LaggedStart(GrowFromCenter(...))`` for each part."""
    return LaggedStart(
        *[GrowFromCenter(p, **kwargs) for p in parts],
        lag_ratio=lag_ratio,
    )
