"""Gyroid ↔ Schwarz-D TPMS phase transition via alpha sweep.

Two triply-periodic minimal surfaces (TPMS) are defined as implicit
SDF functions.  An alpha parameter sweeps between them, visualised
through SDF contour stacking — no triangles.

Gyroid:    sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
Schwarz D: cos(x)cos(y)cos(z) - sin(x)sin(y)sin(z) = 0

Run:
    manim -pql examples/gyroid_schwarz_d.py GyroidSchwarzDTransition
    manim -pql examples/gyroid_schwarz_d.py TPMSAlphaSweep
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
    PURPLE,
    RED,
    RIGHT,
    TAU,
    UP,
    WHITE,
    YELLOW,
    TEAL,
    BLUE_E,
    GREEN_E,
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
from cadmanim.mobjects import SDFContourStack
from cadmanim.animations import AnimateExplodedView


# ---- SDF definitions -------------------------------------------------------

def sdf_gyroid(x, y, z, scale=2.5, thickness=0.25):
    """Gyroid TPMS level-set."""
    sx, sy, sz = x * scale, y * scale, z * scale
    val = np.sin(sx) * np.cos(sy) + np.sin(sy) * np.cos(sz) + np.sin(sz) * np.cos(sx)
    return np.abs(val) - thickness


def sdf_schwarz_d(x, y, z, scale=2.5, thickness=0.25):
    """Schwarz D (Diamond) TPMS level-set."""
    sx, sy, sz = x * scale, y * scale, z * scale
    val = np.cos(sx) * np.cos(sy) * np.cos(sz) - np.sin(sx) * np.sin(sy) * np.sin(sz)
    return np.abs(val) - thickness


def sdf_schwarz_p(x, y, z, scale=2.5, thickness=0.25):
    """Schwarz P (Primitive) TPMS level-set — intermediate reference."""
    sx, sy, sz = x * scale, y * scale, z * scale
    val = np.cos(sx) + np.cos(sy) + np.cos(sz)
    return np.abs(val) - thickness


def sdf_alpha_blend(x, y, z, alpha, scale=2.5, thickness=0.25):
    """Linear alpha blend between Gyroid (alpha=0) and Schwarz D (alpha=1)."""
    g = sdf_gyroid(x, y, z, scale, thickness)
    d = sdf_schwarz_d(x, y, z, scale, thickness)
    return (1 - alpha) * g + alpha * d


# ---- Scene 1: Side-by-side then morph -------------------------------------

class GyroidSchwarzDTransition(ThreeDScene):
    """Show Gyroid and Schwarz D separately, then morph between them."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = Text("TPMS Phase Transition: Gyroid ↔ Schwarz D", font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # ---- Gyroid ----
        gyroid_eq = MathTex(
            r"\sin x \cos y + \sin y \cos z + \sin z \cos x = 0",
            font_size=22,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(gyroid_eq)
        self.play(FadeIn(gyroid_eq))

        gyroid = SDFContourStack(
            sdf_gyroid,
            bounds=(-1.8, 1.8),
            n_slices=22,
            resolution=100,
            color=BLUE,
            opacity=0.8,
        )
        self.play(FadeIn(gyroid), run_time=2)
        self.play(Rotate(gyroid, angle=PI / 3, axis=UP), run_time=2)
        self.wait(0.5)

        # ---- Schwarz D ----
        schwarz_eq = MathTex(
            r"\cos x \cos y \cos z - \sin x \sin y \sin z = 0",
            font_size=22,
        ).next_to(title, DOWN)

        schwarz_d = SDFContourStack(
            sdf_schwarz_d,
            bounds=(-1.8, 1.8),
            n_slices=22,
            resolution=100,
            color=GREEN,
            opacity=0.8,
        )

        self.remove(gyroid_eq)
        self.add_fixed_in_frame_mobjects(schwarz_eq)
        self.play(
            Transform(gyroid, schwarz_d),
            FadeIn(schwarz_eq),
            run_time=3,
        )
        self.play(Rotate(gyroid, angle=PI / 3, axis=UP), run_time=2)

        # ---- Morph back to Gyroid via alpha sweep ----
        alpha_label = MathTex(
            r"\mathcal{S}(\alpha) = (1-\alpha)\,G + \alpha\,D",
            font_size=24,
        ).next_to(title, DOWN)
        self.remove(schwarz_eq)
        self.add_fixed_in_frame_mobjects(alpha_label)
        self.play(FadeIn(alpha_label))

        for alpha in [0.75, 0.5, 0.25, 0.0]:
            col = interpolate_color(GREEN, BLUE, 1 - alpha)
            target = SDFContourStack(
                lambda x, y, z, _a=alpha: sdf_alpha_blend(x, y, z, _a),
                bounds=(-1.8, 1.8),
                n_slices=22,
                resolution=100,
                color=col,
                opacity=0.8,
            )
            val_label = MathTex(rf"\alpha = {alpha:.2f}", font_size=22).next_to(alpha_label, DOWN)
            self.add_fixed_in_frame_mobjects(val_label)
            self.play(Transform(gyroid, target), FadeIn(val_label), run_time=2)
            self.play(Rotate(gyroid, angle=PI / 4, axis=UP), run_time=1)
            self.remove(val_label)

        self.play(FadeOut(gyroid), FadeOut(alpha_label), FadeOut(title))
        self.wait(0.5)


# ---- Scene 2: Alpha Sweep with exploded view -------------------------------

class TPMSAlphaSweep(ThreeDScene):
    """Continuous alpha sweep with exploded contour inspection at midpoint."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)

        title = Text("TPMS α-Sweep: Contour Evolution", font_size=30).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\alpha \in [0, 1] \;:\; "
            r"\text{Gyroid} \xrightarrow{\;\alpha\;} \text{Schwarz D}",
            font_size=24,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Forward sweep 0 → 1
        n_steps = 8
        alphas = np.linspace(0, 1, n_steps)

        stack = SDFContourStack(
            lambda x, y, z: sdf_alpha_blend(x, y, z, 0.0),
            bounds=(-1.8, 1.8),
            n_slices=20,
            resolution=90,
            color=BLUE,
        )
        self.play(FadeIn(stack), run_time=1.5)

        for i, alpha in enumerate(alphas[1:], 1):
            col = interpolate_color(BLUE, GREEN, alpha)
            target = SDFContourStack(
                lambda x, y, z, _a=alpha: sdf_alpha_blend(x, y, z, _a),
                bounds=(-1.8, 1.8),
                n_slices=20,
                resolution=90,
                color=col,
            )
            self.play(Transform(stack, target), run_time=1.5)

            # At midpoint, do exploded view
            if i == n_steps // 2:
                mid_label = MathTex(
                    r"\alpha = 0.5 \;\text{(critical point)}",
                    font_size=22,
                ).next_to(eq, DOWN)
                self.add_fixed_in_frame_mobjects(mid_label)
                self.play(FadeIn(mid_label))
                self.play(AnimateExplodedView(stack, scale_factor=1.6), run_time=2)
                self.play(AnimateExplodedView(stack, scale_factor=1 / 1.6), run_time=1.5)
                self.remove(mid_label)

        # Final rotation at Schwarz D
        self.play(Rotate(stack, angle=TAU / 2, axis=UP), run_time=3)
        self.play(FadeOut(stack), FadeOut(eq), FadeOut(title))
        self.wait(0.5)
