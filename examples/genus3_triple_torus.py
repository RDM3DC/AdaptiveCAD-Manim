"""Genus-3 triple torus from cylinder array mergers under decaying curvature.

Three cylindrical tubes arranged at 120° merge into a genus-3 surface
as the blending radius (curvature) decays.  All geometry uses SDF
contour stacking and parametric surfaces — no triangles.

Run:
    manim -pql examples/genus3_triple_torus.py CylinderMerger
    manim -pql examples/genus3_triple_torus.py Genus3CurvatureDecay
"""

from __future__ import annotations

import sys
import os
import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    PI,
    RED,
    RIGHT,
    TAU,
    UP,
    WHITE,
    YELLOW,
    BLUE_E,
    TEAL,
    Create,
    FadeIn,
    FadeOut,
    MathTex,
    Rotate,
    Text,
    ThreeDScene,
    Transform,
    VGroup,
    Write,
    interpolate_color,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface, SDFContourStack
from cadmanim.animations import AnimateExplodedView, MorphBetweenSDFs
from cadmanim.utils import sphere_parametric, torus_parametric, cylinder_parametric


# ---- SDF primitives --------------------------------------------------------

def sdf_cylinder_z(x, y, z, cx, cy, r=0.35):
    """Infinite cylinder along Z at centre (cx, cy)."""
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r


def sdf_torus_xy(x, y, z, cx, cy, R=0.6, r=0.2):
    """Torus lying flat in XY plane centred at (cx, cy)."""
    dx, dy = x - cx, y - cy
    q = np.sqrt(dx ** 2 + dy ** 2) - R
    return np.sqrt(q ** 2 + z ** 2) - r


def smooth_union(d1, d2, k=0.4):
    """Smooth (polynomial) union of two SDF fields."""
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def sdf_triple_cylinder(x, y, z, blend=0.6):
    """Three cylinders at 120° merged with smooth union."""
    # Centres at 120° on a circle of radius 0.7
    ang = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    R = 0.7
    d = sdf_cylinder_z(x, y, z, R * np.cos(ang[0]), R * np.sin(ang[0]))
    for a in ang[1:]:
        d2 = sdf_cylinder_z(x, y, z, R * np.cos(a), R * np.sin(a))
        d = smooth_union(d, d2, k=blend)
    return d


def sdf_genus3(x, y, z, blend=0.15):
    """Genus-3 surface: three tori at 120° smoothly merged."""
    ang = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    R_place = 0.65
    d = sdf_torus_xy(x, y, z, R_place * np.cos(ang[0]), R_place * np.sin(ang[0]))
    for a in ang[1:]:
        d2 = sdf_torus_xy(x, y, z, R_place * np.cos(a), R_place * np.sin(a))
        d = smooth_union(d, d2, k=blend)
    return d


def sdf_interpolated(x, y, z, alpha, blend_start=0.6, blend_end=0.08):
    """Interpolate from triple-cylinder to genus-3 with decaying curvature."""
    d_cyl = sdf_triple_cylinder(x, y, z, blend=blend_start)
    d_g3 = sdf_genus3(x, y, z, blend=blend_end)
    return (1 - alpha) * d_cyl + alpha * d_g3


# ---- Scene 1: Cylinder Merger ----------------------------------------------

class CylinderMerger(ThreeDScene):
    """Three separate cylinders merge under smooth-union blending."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)

        title = Text("Cylinder Array → Genus-3 Merger", font_size=32).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"d_{\mathrm{blend}} = \mathrm{smin}(d_1, d_2, k)",
            font_size=28,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Stage 1: three separate cylinders (large blend → almost merged)
        blend_values = [0.8, 0.5, 0.25, 0.1]
        colors = [BLUE, TEAL, GREEN, YELLOW]

        stack = SDFContourStack(
            lambda x, y, z: sdf_triple_cylinder(x, y, z, blend=blend_values[0]),
            bounds=(-2.0, 2.0),
            n_slices=20,
            resolution=100,
            color=colors[0],
        )
        self.play(FadeIn(stack), run_time=2)
        self.play(Rotate(stack, angle=PI / 4, axis=UP), run_time=1.5)

        # Sweep blend parameter — curvature decays
        for bv, col in zip(blend_values[1:], colors[1:]):
            new_stack = SDFContourStack(
                lambda x, y, z, _b=bv: sdf_triple_cylinder(x, y, z, blend=_b),
                bounds=(-2.0, 2.0),
                n_slices=20,
                resolution=100,
                color=col,
            )
            label = MathTex(f"k = {bv:.2f}", font_size=24).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(label)
            self.play(Transform(stack, new_stack), FadeIn(label), run_time=2)
            self.play(Rotate(stack, angle=PI / 3, axis=UP), run_time=1)
            self.remove(label)

        self.play(FadeOut(stack), FadeOut(eq), FadeOut(title), run_time=1)
        self.wait(0.5)


# ---- Scene 2: Full Curvature Decay to Genus-3 ------------------------------

class Genus3CurvatureDecay(ThreeDScene):
    """Full morphological transition: cylinders → genus-3 triple torus.

    The curvature (blend radius k) decays while the SDF field transitions
    from three smooth-unioned cylinders to three merged tori forming a
    genus-3 surface.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES)

        title = Text("Genus-3 Triple Torus under Decaying Curvature", font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\Sigma_{g=3} \;:\; \chi = 2 - 2g = -4",
            font_size=28,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Start with cylinder array
        start_stack = SDFContourStack(
            lambda x, y, z: sdf_interpolated(x, y, z, alpha=0.0),
            bounds=(-2.0, 2.0),
            n_slices=24,
            resolution=100,
            color=BLUE,
        )
        self.play(FadeIn(start_stack), run_time=2)
        self.play(Rotate(start_stack, angle=PI / 3, axis=UP), run_time=1.5)

        # Morph through intermediate alpha values
        alphas = [0.25, 0.5, 0.75, 1.0]
        colors = [BLUE_E, TEAL, GREEN, ORANGE]

        for alpha, col in zip(alphas, colors):
            target = SDFContourStack(
                lambda x, y, z, _a=alpha: sdf_interpolated(x, y, z, alpha=_a),
                bounds=(-2.0, 2.0),
                n_slices=24,
                resolution=100,
                color=col,
            )
            a_label = MathTex(rf"\alpha = {alpha:.2f}", font_size=24).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(a_label)
            self.play(Transform(start_stack, target), FadeIn(a_label), run_time=2.5)
            self.play(Rotate(start_stack, angle=PI / 2, axis=UP), run_time=1.5)
            self.remove(a_label)

        # Final genus-3 — exploded view of contour slices
        self.play(AnimateExplodedView(start_stack, scale_factor=1.8), run_time=3)
        self.play(AnimateExplodedView(start_stack, scale_factor=1 / 1.8), run_time=2)

        self.play(FadeOut(start_stack), FadeOut(eq), FadeOut(title))
        self.wait(0.5)
