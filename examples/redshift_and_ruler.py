"""TopEquations #8 + #17 -- ARP Redshift Law & Adaptive pi_a Ruler with AdaptiveCAD 3D.

    z(t) = z0 * (1 - e^{-gamma*t})             (#8  Redshift)
    pi_a_dot = alpha_pi*S - mu_pi*(pi_a - pi0)  (#17 Adaptive Ruler)

Analytic SDF sphere expanding (redshift stretches space) and SDF cylinder
as adaptive ruler breathing with entropy events — all triangle-free.

Run:
    manim -pql examples/redshift_and_ruler.py ARPRedshiftLaw
    manim -pql examples/redshift_and_ruler.py AdaptiveRulerDynamics
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
    Create,
    Cube,
    FadeIn,
    FadeOut,
    MathTex,
    Rotate,
    Text,
    ThreeDScene,
    VGroup,
    Write,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface
from cadmanim.animations import MorphBetweenSDFs
from cadmanim.utils import sphere_parametric, cylinder_parametric


# ---- Scene 1 ---------------------------------------------------------------

class ARPRedshiftLaw(ThreeDScene):
    """SDF sphere expanding from tiny to z0 size — space stretching.

    Parametric morph animates the expansion inside a reference Cube frame.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        title = MathTex(
            r"z(t) = z_0(1 - e^{-\gamma t})", font_size=44,
        ).to_edge(UP, buff=0.3)
        diff = MathTex(
            r"\dot{z} = \gamma(z_0 - z)", font_size=30, color=BLUE,
        ).next_to(title, DOWN, buff=0.12)
        ref = Text(
            "TopEquations #8 -- ARP Redshift Law", font_size=18, color=YELLOW,
        ).next_to(diff, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, diff, ref)
        self.play(Write(title), Write(diff), FadeIn(ref))

        # Reference frame — Manim Cube (no triangles)
        frame_box = Cube(
            side_length=3.6, fill_color=WHITE, fill_opacity=0.08,
            stroke_width=0.5,
        )
        self.play(FadeIn(frame_box, scale=0.8), run_time=1)

        # Small sphere (z ~ 0)
        cosmos = SDFSurface.sphere(radius=0.5, color=BLUE, opacity=0.7)
        z_label = MathTex(
            r"z \approx 0", font_size=24, color=RED,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(z_label)
        self.play(FadeIn(cosmos, scale=0.5), FadeIn(z_label), run_time=2)

        # Morph small -> large (redshift stretching space)
        z_final = MathTex(
            r"z \to z_0", font_size=24, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(z_final)
        self.play(FadeOut(z_label))
        self.play(
            MorphBetweenSDFs(
                cosmos,
                sphere_parametric(0.5),
                sphere_parametric(2.0),
                keyframes=10,
            ),
            FadeIn(z_final),
            run_time=5,
        )

        # Camera orbit
        self.play(
            Rotate(VGroup(frame_box, cosmos), angle=TAU / 2, axis=UP), run_time=4,
        )
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---- Scene 2 ---------------------------------------------------------------

class AdaptiveRulerDynamics(ThreeDScene):
    """SDF cylinder as the adaptive pi_a ruler — thin at rest, thick when excited.

    Parametric morph animates the ruler breathing with entropy events,
    then relaxing back — all triangle-free.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-50 * DEGREES)

        title = MathTex(
            r"\dot{\pi}_a = \alpha_\pi S - \mu_\pi(\pi_a - \pi_0)", font_size=40,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #17 -- Adaptive Angular Ruler",
            font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.12)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # Thin cylinder at rest
        ruler = SDFSurface.cylinder(radius=0.2, height=3.0, color=GREEN, opacity=0.7)
        rest = MathTex(
            r"\pi_a \approx \pi_0", font_size=24, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(rest)
        self.play(FadeIn(ruler), FadeIn(rest), run_time=2)
        self.play(Rotate(ruler, angle=PI / 3, axis=UP), run_time=1.5)

        # Morph thin -> thick (entropy event expands ruler)
        excited = MathTex(
            r"\pi_a \gg \pi_0\;\text{(entropy event!)}", font_size=24, color=ORANGE,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(excited)
        self.play(FadeOut(rest))
        self.play(
            MorphBetweenSDFs(
                ruler,
                cylinder_parametric(0.2, 3.0),
                cylinder_parametric(0.8, 3.0),
                keyframes=8,
            ),
            FadeIn(excited),
            run_time=4,
        )

        # Morph back thick -> thin (relaxation)
        relax = MathTex(
            r"\pi_a \to \pi_0\;\text{(relaxing)}", font_size=24, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(relax)
        self.play(FadeOut(excited))
        self.play(
            MorphBetweenSDFs(
                ruler,
                cylinder_parametric(0.8, 3.0),
                cylinder_parametric(0.2, 3.0),
                keyframes=8,
            ),
            FadeIn(relax),
            run_time=4,
        )

        self.play(Rotate(ruler, angle=TAU / 2, axis=UP), run_time=3)

        eq = MathTex(
            r"\pi_a^\star = \pi_0 + \frac{\alpha_\pi}{\mu_\pi}\mathbb{E}[S_k]",
            font_size=28, color=YELLOW,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(Write(eq))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
