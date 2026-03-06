"""Trefoil knot self-assembly from twisted filament with nodal defects.

A straight twisted filament progressively deforms into a trefoil knot
tube surface.  Nodal defects (self-crossing regions) are highlighted
as the topology locks in.  All geometry is parametric — no triangles.

Run:
    manim -pql examples/trefoil_knot.py TrefoilSelfAssembly
    manim -pql examples/trefoil_knot.py NodalDefects
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
    MAROON,
    Create,
    Dot3D,
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
from cadmanim.utils import sphere_parametric

_TAU = 2.0 * np.pi
_PI = np.pi


# ---- Parametric tube factories ---------------------------------------------

def straight_helix_tube(radius=0.12, helix_r=0.6, length=3.0, twists=3):
    """Straight helical filament along Z axis.  u = tube angle, v = spine."""

    def func(u, v):
        # v : [0, TAU] → vertical position along filament
        t = v / _TAU  # [0, 1]
        z = length * (t - 0.5)
        # Helical backbone
        spine_angle = twists * _TAU * t
        cx = helix_r * np.cos(spine_angle)
        cy = helix_r * np.sin(spine_angle)
        # Tube cross-section around spine
        nx = np.cos(spine_angle)  # local normal (radial)
        ny = np.sin(spine_angle)
        bx, by = -ny, nx  # binormal in XY
        x = cx + radius * (nx * np.cos(u) + 0 * np.sin(u))
        y = cy + radius * (ny * np.cos(u) + 0 * np.sin(u))
        z2 = z + radius * np.sin(u)
        return np.array([x, y, z2])

    return func


def trefoil_tube(radius=0.12, scale=1.2):
    """Tube surface around a trefoil knot curve.  u = tube angle, v = spine."""

    def trefoil_point(t):
        """Trefoil curve in R3."""
        x = scale * (np.sin(t) + 2 * np.sin(2 * t))
        y = scale * (np.cos(t) - 2 * np.cos(2 * t))
        z = scale * (-np.sin(3 * t))
        return np.array([x, y, z])

    def trefoil_tangent(t):
        """Derivative of trefoil curve."""
        dx = scale * (np.cos(t) + 4 * np.cos(2 * t))
        dy = scale * (-np.sin(t) + 4 * np.sin(2 * t))
        dz = scale * (-3 * np.cos(3 * t))
        T = np.array([dx, dy, dz])
        norm = np.linalg.norm(T)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return T / norm

    def func(u, v):
        t = v  # v ∈ [0, TAU]
        p = trefoil_point(t)
        T = trefoil_tangent(t)
        # Construct local frame via cross product with reference
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(T, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])
        N = np.cross(T, ref)
        N = N / (np.linalg.norm(N) + 1e-12)
        B = np.cross(T, N)
        return p + radius * (N * np.cos(u) + B * np.sin(u))

    return func


def blended_filament_to_trefoil(alpha, radius=0.12):
    """Interpolate from straight helix to trefoil tube at parameter alpha."""
    f_start = straight_helix_tube(radius=radius)
    f_end = trefoil_tube(radius=radius)

    def func(u, v):
        a = np.asarray(f_start(u, v), dtype=np.float64)
        b = np.asarray(f_end(u, v), dtype=np.float64)
        return (1 - alpha) * a + alpha * b

    return func


# ---- Nodal defect positions on trefoil ------------------------------------

def trefoil_crossing_params():
    """Return approximate v-parameter values where the trefoil self-crosses
    in the XY-projection (nodal defect sites)."""
    # The trefoil has 3 crossings roughly at these t values
    return [0.9, 2.5, 4.1]


# ---- Scene 1: Self-Assembly ------------------------------------------------

class TrefoilSelfAssembly(ThreeDScene):
    """Animate a twisted filament assembling into a trefoil knot."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        title = Text("Trefoil Knot Self-Assembly", font_size=32).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\mathbf{r}(t) = \bigl(\sin t + 2\sin 2t,\;"
            r"\cos t - 2\cos 2t,\;"
            r"-\!\sin 3t\bigr)",
            font_size=22,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Start with straight helix
        filament = SDFSurface(
            straight_helix_tube(),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(24, 64),
            color=BLUE_D,
            opacity=0.75,
        )
        self.play(FadeIn(filament), run_time=2)
        self.play(Rotate(filament, angle=PI / 3, axis=UP), run_time=1.5)

        # Morph through intermediate stages
        stages = [0.25, 0.5, 0.75, 1.0]
        colors = [BLUE, PURPLE, MAROON, RED]

        for alpha, col in zip(stages, colors):
            target = SDFSurface(
                blended_filament_to_trefoil(alpha),
                u_range=[0, TAU],
                v_range=[0, TAU],
                resolution=(24, 64),
                color=col,
                opacity=0.75,
            )
            label = MathTex(rf"\alpha = {alpha:.2f}", font_size=24).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(label)
            self.play(Transform(filament, target), FadeIn(label), run_time=2.5)
            self.play(Rotate(filament, angle=PI / 4, axis=UP), run_time=1)
            self.remove(label)

        # Final rotation
        self.play(Rotate(filament, angle=TAU, axis=UP), run_time=4)
        self.play(FadeOut(filament), FadeOut(eq), FadeOut(title))
        self.wait(0.5)


# ---- Scene 2: Nodal Defects -----------------------------------------------

class NodalDefects(ThreeDScene):
    """Highlight nodal defect sites on the formed trefoil knot."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = Text("Trefoil Knot — Nodal Defects", font_size=32).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\text{Crossing number} = 3 \quad (\text{minimal})",
            font_size=24,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Full trefoil knot
        knot = SDFSurface(
            trefoil_tube(radius=0.1, scale=1.0),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(20, 64),
            color=BLUE,
            opacity=0.6,
        )
        self.play(FadeIn(knot), run_time=2)

        # Mark nodal defects (crossing sites) with glowing spheres
        crossing_ts = trefoil_crossing_params()
        defects = VGroup()
        scale = 1.0
        for t_cross in crossing_ts:
            px = scale * (np.sin(t_cross) + 2 * np.sin(2 * t_cross))
            py = scale * (np.cos(t_cross) - 2 * np.cos(2 * t_cross))
            pz = scale * (-np.sin(3 * t_cross))
            dot = Dot3D(point=[px, py, pz], radius=0.12, color=YELLOW)
            defects.add(dot)

        for d in defects:
            self.play(FadeIn(d, scale=2.0), run_time=0.7)

        defect_label = MathTex(
            r"\text{Nodal defects at crossings}",
            font_size=22,
        ).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(defect_label)
        self.play(FadeIn(defect_label))

        self.play(Rotate(VGroup(knot, defects), angle=TAU, axis=UP), run_time=4)

        # Pulse defects
        for _ in range(2):
            self.play(
                *[d.animate.scale(1.5) for d in defects],
                run_time=0.4,
            )
            self.play(
                *[d.animate.scale(1 / 1.5) for d in defects],
                run_time=0.4,
            )

        self.play(
            FadeOut(knot),
            FadeOut(defects),
            FadeOut(eq),
            FadeOut(title),
            FadeOut(defect_label),
        )
        self.wait(0.5)
