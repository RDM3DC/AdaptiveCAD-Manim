"""TopEquations #5 -- QWZ Chern-Insulator with AdaptiveCAD 3D.

    H(k) = sin(kx) sigma_x + sin(ky) sigma_y + (u + cos(kx) + cos(ky)) sigma_z

3D band structure with SDF-meshed Dirac-point markers.
Topology morph: sphere (trivial, C=0) -> torus (topological, C!=0).

Run:
    manim -pql examples/qwz_chern_insulator.py QWZBandStructure
    manim -pql examples/qwz_chern_insulator.py QWZTopologyMorph
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
    VGroup,
    Write,
)
from manim.mobject.three_d.three_dimensions import Surface

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface
from cadmanim.animations import MorphBetweenSDFs
from cadmanim.utils import sphere_parametric, torus_parametric


# ---- helpers ---------------------------------------------------------------

def qwz_energies(kx, ky, u):
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = u + np.cos(kx) + np.cos(ky)
    E = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return -E, E


def sdf_sphere(x, y, z, r=1.0):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2) - r


def sdf_torus(x, y, z, R=1.0, r=0.35):
    q = np.sqrt(x ** 2 + z ** 2) - R
    return np.sqrt(q ** 2 + y ** 2) - r


# ---- Scene 1 ---------------------------------------------------------------

class QWZBandStructure(ThreeDScene):
    """3D band structure surfaces with SDF spheres at Dirac contact points."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)

        title = MathTex(
            r"H(\mathbf{k}) = \mathbf{d}(\mathbf{k}) \cdot \boldsymbol{\sigma}",
            font_size=36,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #5 -- QWZ Chern Insulator", font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        u_val = -1.0  # topological C=+1

        def lower_band(u, v):
            E_lo, _ = qwz_energies(u, v, u_val)
            return np.array([u, v, E_lo * 0.6])

        def upper_band(u, v):
            _, E_hi = qwz_energies(u, v, u_val)
            return np.array([u, v, E_hi * 0.6])

        lower = Surface(
            lower_band, u_range=[-PI, PI], v_range=[-PI, PI],
            resolution=(35, 35), fill_color=BLUE, fill_opacity=0.6,
            stroke_width=0.2,
        )
        upper = Surface(
            upper_band, u_range=[-PI, PI], v_range=[-PI, PI],
            resolution=(35, 35), fill_color=RED, fill_opacity=0.4,
            stroke_width=0.2,
        )
        self.play(Create(lower), Create(upper), run_time=3)

        # Analytic SDF sphere at the Dirac point (Gamma)
        dirac = SDFSurface.sphere(radius=0.3, color=YELLOW, opacity=0.9)
        E_gap = abs(u_val + 2)
        dirac.move_to(np.array([0.0, 0.0, 0.0]))
        gap_label = Text("Dirac point", font_size=18, color=YELLOW).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(gap_label)
        self.play(FadeIn(dirac, scale=2), FadeIn(gap_label))

        # Camera orbit
        self.play(
            Rotate(VGroup(lower, upper, dirac), angle=PI, axis=UP), run_time=4,
        )

        phase_label = MathTex(
            r"C = +1", font_size=30, color=GREEN,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(phase_label)
        self.play(FadeIn(phase_label))
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects])


# ---- Scene 2 ---------------------------------------------------------------

class QWZTopologyMorph(ThreeDScene):
    """Visual metaphor: sphere (trivial, C=0) -> torus (topological, C!=0).

    Uses MorphBetweenSDFs to smoothly change the genus of the surface,
    representing the topological phase transition.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        title = MathTex(
            r"\text{Trivial} \;\to\; \text{Topological}", font_size=40,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "Genus change: sphere (C=0) -> torus (C=1)",
            font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # Analytic SDF sphere (trivial topology)
        shape = SDFSurface.sphere(radius=1.0, color=BLUE, opacity=0.7)
        trivial = MathTex(
            r"C = 0\;\text{(trivial)}", font_size=28, color=WHITE,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(trivial)
        self.play(FadeIn(shape), FadeIn(trivial), run_time=2)
        self.play(Rotate(shape, angle=PI / 3, axis=UP), run_time=1.5)

        # Morph sphere -> torus
        topo = MathTex(
            r"C = +1\;\text{(topological)}", font_size=28, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(topo)
        self.play(FadeOut(trivial))
        self.play(
            MorphBetweenSDFs(
                shape,
                sphere_parametric(1.0),
                torus_parametric(1.0, 0.35),
                keyframes=10,
            ),
            FadeIn(topo),
            run_time=5,
        )

        self.play(Rotate(shape, angle=TAU, axis=UP), run_time=4)

        chern_eq = MathTex(
            r"C = \frac{1}{2\pi}\oint \mathbf{F}\cdot d\mathbf{S}",
            font_size=30, color=YELLOW,
        ).to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(chern_eq)
        self.play(Write(chern_eq))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
