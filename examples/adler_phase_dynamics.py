"""TopEquations #3 -- Phase (Adler/RSJ) Dynamics with AdaptiveCAD 3D.

    phi_dot = Delta - lambda*G * sin(phi)

3D washboard potential landscape with analytic SDF sphere marble.
Torus morph shows lock/slip bifurcation — all triangle-free.

Run:
    manim -pql examples/adler_phase_dynamics.py AdlerPhaseDynamics
    manim -pql examples/adler_phase_dynamics.py ParityLockBifurcation
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
from cadmanim.mobjects import SDFSurface
from cadmanim.animations import MorphBetweenSDFs
from cadmanim.utils import torus_parametric


# ---- SDF primitives (definitions kept for documentation) -------------------

def sdf_sphere(x, y, z, r=1.0):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2) - r


# ---- Scene 1 ---------------------------------------------------------------

class AdlerPhaseDynamics(ThreeDScene):
    """3D washboard potential with an AdaptiveCAD SDF marble.

    A parametric Surface shows V(phi) = -Delta*phi + lambdaG*(1 - cos phi).
    An SDF-meshed sphere sits in the potential well (locked), then the
    landscape tilts to show the slip regime.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = MathTex(
            r"\dot{\phi} = \Delta - \lambda G \sin\phi", font_size=44,
        ).to_edge(UP, buff=0.4)
        ref = Text(
            "TopEquations #3 -- Adler / RSJ Phase Dynamics",
            font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # 3D washboard potential surface
        Delta, lambdaG = 0.3, 1.0

        def washboard(u, v):
            V = -Delta * u + lambdaG * (1 - np.cos(u))
            return np.array([u * 0.25, v * 0.6, V * 0.4])

        surface = Surface(
            washboard,
            u_range=[-2 * PI, 2 * PI],
            v_range=[-1, 1],
            resolution=(60, 8),
            fill_color=BLUE,
            fill_opacity=0.6,
            stroke_width=0.3,
        )
        self.play(Create(surface), run_time=2)

        # Analytic SDF sphere "marble" sitting in the potential minimum
        marble = SDFSurface.sphere(radius=0.25, color=YELLOW, opacity=0.9)
        phi_star = np.arcsin(Delta / lambdaG)
        V_star = -Delta * phi_star + lambdaG * (1 - np.cos(phi_star))
        marble.move_to(np.array([phi_star * 0.25, 0.0, V_star * 0.4 + 0.25]))

        lock_label = Text("LOCKED", font_size=22, color=GREEN).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(lock_label)
        self.play(FadeIn(marble, scale=2), FadeIn(lock_label))
        self.wait(0.5)

        # Camera orbit
        self.play(
            Rotate(VGroup(surface, marble), angle=PI / 2, axis=UP), run_time=3,
        )

        # Tilt washboard -> slip regime
        def washboard_slip(u, v):
            V = -1.5 * u + lambdaG * (1 - np.cos(u))
            return np.array([u * 0.25, v * 0.6, V * 0.4])

        surface_slip = Surface(
            washboard_slip,
            u_range=[-2 * PI, 2 * PI],
            v_range=[-1, 1],
            resolution=(60, 8),
            fill_color=RED,
            fill_opacity=0.6,
            stroke_width=0.3,
        )
        slip_label = Text("SLIP", font_size=22, color=RED).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(slip_label)
        self.play(FadeOut(lock_label))
        self.play(
            Transform(surface, surface_slip),
            marble.animate.shift(np.array([0.0, 0.0, -0.8])),
            FadeIn(slip_label),
            run_time=3,
        )

        sig = MathTex(
            r"r_b = |\Delta|/\pi", font_size=32, color=YELLOW,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(sig)
        self.play(Write(sig))
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects])


# ---- Scene 2 ---------------------------------------------------------------

class ParityLockBifurcation(ThreeDScene):
    """SDF torus morph: thin (slip) -> thick (locked) with exploded view.

    The torus tube-radius represents coupling strength lambdaG.
    MorphBetweenSDFs interpolates smoothly between the two shapes.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)

        title = MathTex(
            r"\text{Coupling: thin torus} \to \text{thick torus}",
            font_size=36,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #3 + #12 -- Parity Lock Bifurcation",
            font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # Thin torus (slip regime) — analytic parametric surface
        torus_mesh = SDFSurface.torus(R=1.0, r=0.15, color=RED, opacity=0.7)
        slip_text = Text("Weak coupling (slip)", font_size=20, color=RED)
        slip_text.to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(slip_text)
        self.play(FadeIn(torus_mesh), FadeIn(slip_text), run_time=2)
        self.play(Rotate(torus_mesh, angle=PI / 2, axis=UP), run_time=2)

        # Morph thin -> thick (parametric interpolation, no triangles)
        lock_text = Text("Strong coupling (locked)", font_size=20, color=GREEN)
        lock_text.to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(lock_text)
        self.play(FadeOut(slip_text))
        self.play(
            MorphBetweenSDFs(
                torus_mesh,
                torus_parametric(1.0, 0.15),
                torus_parametric(1.0, 0.5),
                keyframes=8,
            ),
            FadeIn(lock_text),
            run_time=4,
        )

        # Camera orbit of the locked torus
        self.play(Rotate(torus_mesh, angle=TAU, axis=UP), run_time=3)

        asym = MathTex(
            r"\frac{1}{\pi}", font_size=48, color=YELLOW,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(asym)
        self.play(Write(asym))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
