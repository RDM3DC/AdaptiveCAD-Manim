"""Example: Animate AdaptiveCAD SDF shapes morphing and rotating — triangle-free.

All 3D shapes use analytic parametric surfaces (SDFSurface) or
SDF contour slicing (SDFContourStack).  No marching cubes, no triangles.

Run with:
    manim -pql examples/sdf_shapes_demo.py SDFShapesDemo
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    GREEN,
    LEFT,
    PI,
    RED,
    RIGHT,
    TAU,
    UP,
    WHITE,
    YELLOW,
    Create,
    FadeIn,
    FadeOut,
    MathTex,
    Rotate,
    Text,
    ThreeDScene,
    VGroup,
    Write,
    config,
)
from manim.mobject.three_d.three_dimensions import Surface

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadmanim.mobjects import SDFSurface, SDFContourStack
from cadmanim.animations import AnimateExplodedView, MorphBetweenSDFs
from cadmanim.utils import sphere_parametric, torus_parametric


# ---- SDF primitives (kept for SDFContourStack) ----------------------------

def sdf_gyroid(x, y, z, scale=3.0, thickness=0.3):
    sx, sy, sz = x * scale, y * scale, z * scale
    return np.abs(np.sin(sx) * np.cos(sy) + np.sin(sy) * np.cos(sz) + np.sin(sz) * np.cos(sx)) - thickness


# ---- Scene -----------------------------------------------------------------

class SDFShapesDemo(ThreeDScene):
    """Showcase AdaptiveCAD SDF shapes animated with Manim — triangle-free."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        title = Text("AdaptiveCAD × Manim", font_size=36).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait(0.5)

        # --- 1. Analytic sphere from SDF ---
        subtitle = Text("SDF → Analytic Surface → Animation", font_size=24).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(subtitle)
        self.play(FadeIn(subtitle))

        sphere_surf = SDFSurface.sphere(radius=1.0, color=BLUE, opacity=0.7)
        self.play(FadeIn(sphere_surf), run_time=2)
        self.play(Rotate(sphere_surf, angle=PI, axis=UP), run_time=2)
        self.wait(0.5)

        # --- 2. Parametric morph sphere → torus ---
        morph_label = Text("Parametric Morph: Sphere → Torus", font_size=24).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(morph_label)
        self.play(FadeOut(subtitle), FadeIn(morph_label))

        self.play(
            MorphBetweenSDFs(
                sphere_surf,
                sphere_parametric(1.0),
                torus_parametric(1.0, 0.4),
                keyframes=10,
            ),
            run_time=4,
        )
        self.wait(0.5)

        # --- 3. Gyroid as contour stack ---
        gyroid_label = Text("Gyroid (TPMS) — contour slices", font_size=24).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(gyroid_label)
        self.play(FadeOut(morph_label), FadeIn(gyroid_label))

        gyroid_stack = SDFContourStack(
            sdf_gyroid, bounds=(-2.0, 2.0), n_slices=28,
            color=GREEN, opacity=0.6,
        )
        self.play(FadeOut(sphere_surf), FadeIn(gyroid_stack), run_time=2)
        self.play(Rotate(gyroid_stack, angle=TAU, axis=np.array([1, 1, 0])), run_time=4)

        # --- 4. Exploded view (contour layers separate) ---
        explode_label = Text("Exploded View", font_size=24).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(explode_label)
        self.play(FadeOut(gyroid_label), FadeIn(explode_label))

        n = len(gyroid_stack)
        chunk = max(1, n // 5)
        parts = VGroup(*[gyroid_stack[i:i + chunk] for i in range(0, n, chunk)])
        self.play(AnimateExplodedView(parts, scale_factor=2.5), run_time=3)
        self.wait(1)

        self.play(FadeOut(parts), FadeOut(title), FadeOut(explode_label))


# ---- Scene 2: Parametric surface using AdaptiveCAD pi_a --------------------

class PiAdaptiveSurface(ThreeDScene):
    """Animate a surface whose radius is modulated by AdaptiveCAD's π_a ratio."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        title = Text("Adaptive π Surface", font_size=36).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        try:
            from adaptivecad.geom import pi_a_over_pi
        except ImportError:
            pi_a_over_pi = lambda r, kappa: 1.0  # fallback

        kappa = 1.5

        def pi_surface(u, v):
            r = 0.5 + 0.5 * u  # radius varies from 0.5 to 1.0
            ratio = pi_a_over_pi(r, kappa)
            x = r * ratio * np.cos(v)
            y = r * ratio * np.sin(v)
            z = (u - 0.5) * 3
            return np.array([x, y, z])

        surface = Surface(
            pi_surface,
            u_range=[0, 1],
            v_range=[0, TAU],
            resolution=(30, 30),
            fill_color=YELLOW,
            fill_opacity=0.7,
        )

        self.play(Create(surface), run_time=3)
        self.play(Rotate(surface, angle=TAU, axis=UP), run_time=5)
        self.wait(1)
        self.play(FadeOut(surface), FadeOut(title))
