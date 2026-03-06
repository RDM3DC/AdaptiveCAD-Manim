"""Example: Animate AdaptiveCAD Bezier curves, sketches, and toolpaths.

Run with:
    manim -pql examples/curves_and_toolpaths.py CurvesAndToolpaths
"""

from __future__ import annotations

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
    FadeIn,
    FadeOut,
    Scene,
    Text,
    ThreeDScene,
    VGroup,
    Write,
)

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadmanim.mobjects import BezierCurveMobject, ToolpathMobject
from cadmanim.animations import AnimateCurveGrowth, AnimateToolpath


class CurvesAndToolpaths(ThreeDScene):
    """Demonstrate Bezier curve growth and toolpath tracing."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        title = Text("AdaptiveCAD Curves & Toolpaths", font_size=32).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # --- 1. Bezier curve ---
        try:
            from adaptivecad.linalg import Vec3
            from adaptivecad.geom import BezierCurve

            control_pts = [
                Vec3(-3, -1, 0),
                Vec3(-1, 2, 1),
                Vec3(1, -2, 1),
                Vec3(3, 1, 0),
            ]
            curve = BezierCurve(control_pts)
            curve_mob = BezierCurveMobject(curve, color=YELLOW, stroke_width=4)
        except ImportError:
            # Fallback: create a parametric stand-in
            from manim import ParametricFunction

            curve_mob = ParametricFunction(
                lambda t: np.array([
                    -3 + 6 * t,
                    2 * np.sin(t * PI),
                    np.sin(t * PI * 2) * 0.5,
                ]),
                t_range=[0, 1],
                color=YELLOW,
                stroke_width=4,
            )
            curve = None

        label1 = Text("Bézier Curve Growth", font_size=24).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(label1)
        self.play(FadeIn(label1))

        if curve is not None:
            self.play(
                AnimateCurveGrowth(curve_mob, curve_evaluator=curve.evaluate, samples=200),
                run_time=3,
            )
        else:
            self.play(Create(curve_mob), run_time=3)

        self.wait(1)

        # --- 2. Helical toolpath ---
        label2 = Text("CNC Toolpath Trace", font_size=24).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(label2)
        self.play(FadeOut(label1), FadeIn(label2))

        n_pts = 300
        t = np.linspace(0, 6 * PI, n_pts)
        helix_pts = np.column_stack([
            np.cos(t) * (1 + t / (6 * PI)),
            np.sin(t) * (1 + t / (6 * PI)),
            t / (6 * PI) * 3 - 1.5,
        ])

        toolpath = ToolpathMobject(helix_pts, color_start=GREEN, color_end=RED)
        # Get the inner VMobject path for the trace animation
        path_mob = toolpath[0]

        self.play(FadeOut(curve_mob))
        self.play(AnimateToolpath(path_mob), run_time=4)
        self.wait(1)

        self.play(FadeOut(toolpath), FadeOut(title), FadeOut(label2))


class AssemblyDemo(ThreeDScene):
    """Show parts assembling from scattered positions."""

    def construct(self):
        from cadmanim.animations import AnimateAssembly

        self.set_camera_orientation(phi=70 * DEGREES, theta=-35 * DEGREES)

        title = Text("Assembly Animation", font_size=32).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Create simple stand-in parts (cubes at different positions)
        from manim import Cube

        parts = VGroup(
            Cube(side_length=0.8, fill_color=RED, fill_opacity=0.8).shift(LEFT * 1.5),
            Cube(side_length=0.8, fill_color=GREEN, fill_opacity=0.8).shift(RIGHT * 1.5),
            Cube(side_length=0.8, fill_color=BLUE, fill_opacity=0.8).shift(UP * 1.5),
            Cube(side_length=0.8, fill_color=ORANGE, fill_opacity=0.8).shift(DOWN * 1.5),
        )

        self.play(AnimateAssembly(parts, spread=5.0), run_time=4)
        self.wait(1)

        from cadmanim.animations import AnimateExplodedView

        self.play(AnimateExplodedView(parts, scale_factor=3.0), run_time=3)
        self.wait(1)

        self.play(FadeOut(parts), FadeOut(title))
