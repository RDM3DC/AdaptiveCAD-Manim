"""Seifert surface inflation from trefoil to full torus-knot lattice.

A Seifert surface (the spanning surface of a trefoil knot) inflates
outward, then the boundary knot generalises through a family of
(p,q) torus knots, building a lattice of knotted tubes.
μ=0.8, v=1.2 ramp control the inflation dynamics.

All geometry is parametric Surface — no triangles.

Run:
    manim -pql examples/seifert_inflation.py SeifertInflation
    manim -pql examples/seifert_inflation.py TorusKnotLattice
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
    BLUE_D,
    BLUE_E,
    RED_E,
    TEAL,
    GOLD,
    MAROON,
    PINK,
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
from manim.mobject.three_d.three_dimensions import Surface

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface

_TAU = 2.0 * np.pi
_PI = np.pi

MU = 0.8
V_RAMP = 1.2


# ---- Torus knot curve and tube factories -----------------------------------

def torus_knot_curve(p, q, R=1.5, r=0.6):
    """Parametric (p,q) torus knot curve on a torus of radii (R, r)."""

    def func(t):
        x = (R + r * np.cos(q * t)) * np.cos(p * t)
        y = (R + r * np.cos(q * t)) * np.sin(p * t)
        z = r * np.sin(q * t)
        return np.array([x, y, z])

    return func


def torus_knot_tangent(p, q, R=1.5, r=0.6):
    """Tangent vector of (p,q) torus knot."""

    def func(t):
        dx = -p * (R + r * np.cos(q * t)) * np.sin(p * t) \
             - r * q * np.sin(q * t) * np.cos(p * t)
        dy = p * (R + r * np.cos(q * t)) * np.cos(p * t) \
             - r * q * np.sin(q * t) * np.sin(p * t)
        dz = r * q * np.cos(q * t)
        T = np.array([dx, dy, dz])
        norm = np.linalg.norm(T)
        return T / (norm + 1e-12)

    return func


def torus_knot_tube(p, q, tube_r=0.07, R=1.5, r=0.6):
    """Tube surface around (p,q) torus knot. u=cross-section, v=spine."""
    curve_fn = torus_knot_curve(p, q, R, r)
    tang_fn = torus_knot_tangent(p, q, R, r)

    def func(u, v):
        pt = curve_fn(v)
        T = tang_fn(v)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(T, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])
        N = np.cross(T, ref)
        N = N / (np.linalg.norm(N) + 1e-12)
        B = np.cross(T, N)
        return pt + tube_r * (N * np.cos(u) + B * np.sin(u))

    return func


# ---- Seifert surface for the trefoil --------------------------------------

def seifert_surface(inflation=0.0, R=1.5, r=0.6):
    """Parametric Seifert surface for the (2,3) trefoil.

    The surface spans the trefoil knot boundary.  *inflation* controls
    how much the surface billows outward (0 = flat, 1 = fully inflated).

    u ∈ [0, TAU] : radial parameter across the surface
    v ∈ [0, TAU] : angular parameter around the knot
    """

    def func(u, v):
        # s interpolates from centre (s=0) to the knot boundary (s=1)
        s = u / _TAU

        # Knot boundary — trefoil (p=2, q=3)
        bx = (R + r * np.cos(3 * v)) * np.cos(2 * v)
        by = (R + r * np.cos(3 * v)) * np.sin(2 * v)
        bz = r * np.sin(3 * v)

        # Centre of the spanning disk
        cx, cy, cz = 0.0, 0.0, 0.0

        # Interpolate from centre to boundary
        x = cx + s * (bx - cx)
        y = cy + s * (by - cy)
        z = cz + s * (bz - cz)

        # Inflation: push outward along the normal (z-direction + radial)
        inflate_profile = inflation * np.sin(_PI * s) * 0.5
        z = z + inflate_profile * np.cos(v * 3)
        # Add radial billowing
        radial_push = inflate_profile * 0.3
        x = x + radial_push * np.cos(2 * v)
        y = y + radial_push * np.sin(2 * v)

        return np.array([x, y, z])

    return func


def _inflation_profile(t, mu=MU, v=V_RAMP):
    """Inflation ramp: t ∈ [0,1] → smooth inflation with μ, v dynamics."""
    return (1 - np.exp(-mu * t * 4)) * (1 + 0.15 * np.sin(v * t * 6))


# ---- Scene 1: Seifert Surface Inflation ------------------------------------

class SeifertInflation(ThreeDScene):
    """Inflate Seifert surface from flat trefoil span to billowed manifold."""

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-35 * DEGREES)

        title = Text("Seifert Surface Inflation — Trefoil",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"K_{2,3}:\;"
            r"\mathbf{r}(t) = \bigl((R + r\cos 3t)\cos 2t,\;"
            r"(R + r\cos 3t)\sin 2t,\; r\sin 3t\bigr)",
            font_size=18,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Trefoil knot boundary tube
        knot = SDFSurface(
            torus_knot_tube(2, 3, tube_r=0.06),
            u_range=[0, _TAU],
            v_range=[0, _TAU],
            resolution=(12, 64),
            color=RED,
            opacity=0.85,
        )
        self.play(FadeIn(knot), run_time=2)
        self.play(Rotate(knot, angle=PI / 4, axis=UP), run_time=1)

        # Flat Seifert surface
        seifert = SDFSurface(
            seifert_surface(inflation=0.0),
            u_range=[0, _TAU],
            v_range=[0, _TAU],
            resolution=(24, 48),
            color=BLUE_D,
            opacity=0.45,
        )
        self.play(FadeIn(seifert), run_time=2)

        # Inflate through stages
        inflate_times = [0.2, 0.5, 0.8, 1.0]
        colors = [BLUE_D, BLUE, TEAL, GREEN]

        for t_val, col in zip(inflate_times, colors):
            infl = float(_inflation_profile(t_val))
            target = SDFSurface(
                seifert_surface(inflation=infl),
                u_range=[0, _TAU],
                v_range=[0, _TAU],
                resolution=(24, 48),
                color=col,
                opacity=0.5,
            )
            label = MathTex(
                rf"\text{{inflate}} = {infl:.2f}",
                font_size=22,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(label)
            self.play(Transform(seifert, target), FadeIn(label), run_time=2)
            self.play(Rotate(VGroup(knot, seifert), PI / 3, axis=UP), run_time=1)
            self.remove(label)

        # Final full rotation
        self.play(Rotate(VGroup(knot, seifert), TAU, axis=UP), run_time=4)
        self.play(FadeOut(seifert), FadeOut(knot), FadeOut(eq), FadeOut(title))
        self.wait(0.5)


# ---- Scene 2: Torus Knot Lattice ------------------------------------------

class TorusKnotLattice(ThreeDScene):
    """Generalise from trefoil through (p,q) torus knot family lattice."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = Text("Torus Knot Lattice — (p,q) Family",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"K_{p,q}:\; \mu = 0.8,\; v = 1.2",
            font_size=22,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Knot family: (p, q) pairs with increasing complexity
        knot_family = [
            (2, 3, "Trefoil", RED, 0.07),
            (2, 5, "Solomon", ORANGE, 0.06),
            (3, 4, "(3,4)", GOLD, 0.055),
            (3, 5, "(3,5)", PURPLE, 0.05),
            (2, 7, "(2,7)", MAROON, 0.05),
        ]

        # Start with trefoil
        p0, q0, name0, col0, tr0 = knot_family[0]
        current_knot = SDFSurface(
            torus_knot_tube(p0, q0, tube_r=tr0),
            u_range=[0, _TAU],
            v_range=[0, _TAU],
            resolution=(12, 64),
            color=col0,
            opacity=0.8,
        )
        lbl_knot = MathTex(
            rf"K_{{{p0},{q0}}}\;\text{{{name0}}}",
            font_size=22,
        ).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(lbl_knot)
        self.play(FadeIn(current_knot), FadeIn(lbl_knot), run_time=2)
        self.play(Rotate(current_knot, PI / 3, axis=UP), run_time=1.5)

        # Morph through knot family
        for p, q, name, col, tr in knot_family[1:]:
            target = SDFSurface(
                torus_knot_tube(p, q, tube_r=tr),
                u_range=[0, _TAU],
                v_range=[0, _TAU],
                resolution=(12, 64),
                color=col,
                opacity=0.8,
            )
            new_lbl = MathTex(
                rf"K_{{{p},{q}}}\;\text{{{name}}}",
                font_size=22,
            ).next_to(eq, DOWN)
            self.remove(lbl_knot)
            self.add_fixed_in_frame_mobjects(new_lbl)
            self.play(
                Transform(current_knot, target),
                FadeIn(new_lbl),
                run_time=2.5,
            )
            self.play(Rotate(current_knot, PI / 3, axis=UP), run_time=1.5)
            lbl_knot = new_lbl

        # Build lattice: show multiple knots simultaneously
        self.remove(lbl_knot)
        lattice_label = Text("Knot Lattice", font_size=22).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(lattice_label)
        self.play(FadeOut(current_knot), FadeIn(lattice_label), run_time=1)

        # Arrange 4 knots in a 2×2 grid
        offsets = [
            np.array([-1.5, 1.0, 0.0]),
            np.array([1.5, 1.0, 0.0]),
            np.array([-1.5, -1.0, 0.0]),
            np.array([1.5, -1.0, 0.0]),
        ]
        lattice_knots = VGroup()
        for i, (p, q, name, col, tr) in enumerate(knot_family[:4]):
            k = SDFSurface(
                torus_knot_tube(p, q, tube_r=tr, R=0.7, r=0.3),
                u_range=[0, _TAU],
                v_range=[0, _TAU],
                resolution=(10, 48),
                color=col,
                opacity=0.75,
            )
            k.shift(offsets[i])
            lattice_knots.add(k)

        self.play(FadeIn(lattice_knots), run_time=2.5)
        self.play(Rotate(lattice_knots, TAU, axis=UP), run_time=5)
        self.play(
            FadeOut(lattice_knots), FadeOut(eq),
            FadeOut(title), FadeOut(lattice_label),
        )
        self.wait(0.5)
