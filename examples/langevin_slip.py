"""TopEquations #22 + #12 -- Langevin Phase Dynamics & Slip Asymptote with AdaptiveCAD 3D.

    phi_dot = Delta - lambdaG * sin(phi) + sqrt(2D) * eta(t)    (#22)
    r_b     = |Delta| / pi                                       (#12)

3D washboard potential surface with analytic SDF sphere particles.
Torus -> gyroid contour-stack morph represents the lock/chaos transition.
All triangle-free.

Run:
    manim -pql examples/langevin_slip.py LangevinPhaseDynamics
    manim -pql examples/langevin_slip.py SlipAsymptoteUniversality
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
    FadeIn,
    FadeOut,
    MathTex,
    Rotate,
    Text,
    ThreeDScene,
    Transform,
    VGroup,
    Write,
)
from manim.mobject.three_d.three_dimensions import Surface

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface, SDFContourStack
from cadmanim.animations import AnimateExplodedView


# ---- SDF primitives --------------------------------------------------------

def sdf_sphere(x, y, z, r=1.0):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2) - r


def sdf_torus(x, y, z, R=1.0, r=0.35):
    q = np.sqrt(x ** 2 + z ** 2) - R
    return np.sqrt(q ** 2 + y ** 2) - r


def sdf_gyroid(x, y, z, scale=3.0, thickness=0.3):
    sx, sy, sz = x * scale, y * scale, z * scale
    return (
        np.abs(
            np.sin(sx) * np.cos(sy)
            + np.sin(sy) * np.cos(sz)
            + np.sin(sz) * np.cos(sx)
        )
        - thickness
    )


# ---- Scene 1 ---------------------------------------------------------------

class LangevinPhaseDynamics(ThreeDScene):
    """3D washboard potential with SDF sphere particles at different noise levels.

    The washboard V(phi) = -Delta*phi + lambdaG*(1-cos phi) is a tilted
    sinusoidal landscape. SDF spheres represent particles at D=0, D=0.01, D=0.1.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        title = MathTex(
            r"\dot{\varphi} = \Delta - \lambda G\sin\varphi + \sqrt{2D}\,\eta(t)",
            font_size=34,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #22 -- Langevin Phase Dynamics",
            font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # 3D washboard potential
        Delta, lambdaG = 0.3, 1.0

        def washboard(u, v):
            V = -Delta * u + lambdaG * (1 - np.cos(u))
            return np.array([u * 0.2, v * 0.5, V * 0.3])

        surface = Surface(
            washboard, u_range=[-3 * PI, 3 * PI], v_range=[-1, 1],
            resolution=(80, 8), fill_color=BLUE, fill_opacity=0.5,
            stroke_width=0.2,
        )
        self.play(Create(surface), run_time=2)

        # Analytic SDF spheres representing particles at different noise levels
        colors = [GREEN, ORANGE, RED]
        labels = [r"D=0", r"D=0.01", r"D=0.1"]
        positions = [
            np.array([-0.8, -0.4, 0.5]),
            np.array([-0.4, 0.0, 0.7]),
            np.array([0.0, 0.4, 0.9]),
        ]

        particles = VGroup()
        for col, pos in zip(colors, positions):
            p = SDFSurface.sphere(
                radius=0.2, color=col, opacity=0.9,
                resolution=(16, 16),
            )
            p.move_to(pos)
            particles.add(p)

        d_labels = VGroup()
        for i, (lab_text, col) in enumerate(zip(labels, colors)):
            lab = MathTex(lab_text, font_size=18, color=col)
            lab.to_corner(DOWN + LEFT).shift(RIGHT * i * 1.8)
            self.add_fixed_in_frame_mobjects(lab)
            d_labels.add(lab)

        self.play(FadeIn(particles), run_time=1.5)

        # Camera orbit
        self.play(
            Rotate(VGroup(surface, particles), angle=PI, axis=UP), run_time=4,
        )

        kramers = MathTex(
            r"\tau_{\mathrm{escape}} \sim e^{\Delta U/D}",
            font_size=28, color=GREEN,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(kramers)
        self.play(Write(kramers))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---- Scene 2 ---------------------------------------------------------------

class SlipAsymptoteUniversality(ThreeDScene):
    """SDF morph: torus (locked) -> gyroid (chaotic) at the bifurcation.

    The geometry change represents the lock/slip transition.
    Exploded view reveals the chaotic internal structure.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)

        title = MathTex(
            r"r_b = \frac{|\Delta|}{\pi}", font_size=48,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #12 -- Slip Asymptote", font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # Torus = locked state (ordered) — analytic parametric surface
        shape = SDFSurface.torus(R=1.0, r=0.35, color=GREEN, opacity=0.7)
        lock_label = MathTex(
            r"\text{LOCKED}\;(r_b = 0)", font_size=24, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(lock_label)
        self.play(FadeIn(shape), FadeIn(lock_label), run_time=2)
        self.play(Rotate(shape, angle=PI / 3, axis=UP), run_time=1.5)

        # Morph torus -> gyroid (chaotic) via contour-stack Transform
        slip_label = MathTex(
            r"\text{SLIP}\;(r_b \to |\Delta|/\pi)", font_size=24, color=RED,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(slip_label)
        self.play(FadeOut(lock_label))

        gyroid_target = SDFContourStack(
            sdf_gyroid, bounds=(-2, 2), n_slices=28,
            color=RED, opacity=0.6,
        )
        self.play(
            FadeOut(shape),
            FadeIn(gyroid_target),
            FadeIn(slip_label),
            run_time=3,
        )

        self.play(Rotate(gyroid_target, angle=TAU / 2, axis=UP), run_time=3)

        # Exploded view of the chaotic contour layers
        n = len(gyroid_target)
        chunk = max(1, n // 8)
        parts = VGroup(*[gyroid_target[i:i + chunk] for i in range(0, n, chunk)])
        self.play(AnimateExplodedView(parts, scale_factor=2.0), run_time=3)

        pi_note = MathTex(
            r"\frac{1}{\pi}\;\text{is universal}",
            font_size=32, color=YELLOW,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(pi_note)
        self.play(Write(pi_note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
