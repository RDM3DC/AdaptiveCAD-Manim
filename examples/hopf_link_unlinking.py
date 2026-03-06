"""Hopf link unlinking via ARP filament untwist — α sweep on blue/red pairs.

Two linked torus-knot tubes (blue/red Hopf link) progressively unlink
as the coupling parameter α sweeps from 1 → 0.  The linking number
decreases through intermediate filament deformations.  μ=0.8, v=1.2
ramp control the twist decay dynamics.  All geometry is parametric —
no triangles.

Run:
    manim -pql examples/hopf_link_unlinking.py HopfUnlinkingSweep
    manim -pql examples/hopf_link_unlinking.py ARPFilamentDecay
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

# ARP ramp parameters
MU = 0.8
V_RAMP = 1.2


# ---- Parametric Hopf link tube factories -----------------------------------

def _hopf_component_a(alpha, tube_r=0.08, R=1.0):
    """First component of Hopf link — a circle in XY plane, lifted by alpha."""

    def func(u, v):
        # Backbone: circle in XY at z=0, tilted by alpha
        cx = R * np.cos(v)
        cy = R * np.sin(v)
        cz = alpha * 0.3 * np.sin(v)

        # Tangent along backbone
        tx = -R * np.sin(v)
        ty = R * np.cos(v)
        tz = alpha * 0.3 * np.cos(v)
        t_len = np.sqrt(tx**2 + ty**2 + tz**2) + 1e-12
        tx, ty, tz = tx / t_len, ty / t_len, tz / t_len

        # Normal and binormal
        ref = np.array([0.0, 0.0, 1.0])
        T = np.array([tx, ty, tz])
        if abs(np.dot(T, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])
        N = np.cross(T, ref)
        N = N / (np.linalg.norm(N) + 1e-12)
        B = np.cross(T, N)

        x = cx + tube_r * (N[0] * np.cos(u) + B[0] * np.sin(u))
        y = cy + tube_r * (N[1] * np.cos(u) + B[1] * np.sin(u))
        z = cz + tube_r * (N[2] * np.cos(u) + B[2] * np.sin(u))
        return np.array([x, y, z])

    return func


def _hopf_component_b(alpha, tube_r=0.08, R=1.0):
    """Second component — a circle threaded through the first when alpha=1."""

    def func(u, v):
        # When alpha=1: circle in XZ plane offset so it threads through A
        # When alpha=0: circle in XZ plane separated (unlinked)
        offset = alpha * R  # how far the centre is inside A's loop
        cx = offset * np.cos(v) * 0.3 + R * 0.5 * (1 - alpha)
        cy = R * np.cos(v)
        cz = R * np.sin(v) + alpha * 0.2

        # Tangent
        tx = -offset * np.sin(v) * 0.3
        ty = -R * np.sin(v)
        tz = R * np.cos(v)
        t_len = np.sqrt(tx**2 + ty**2 + tz**2) + 1e-12
        tx, ty, tz = tx / t_len, ty / t_len, tz / t_len

        ref = np.array([1.0, 0.0, 0.0])
        T = np.array([tx, ty, tz])
        if abs(np.dot(T, ref)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])
        N = np.cross(T, ref)
        N = N / (np.linalg.norm(N) + 1e-12)
        B = np.cross(T, N)

        x = cx + tube_r * (N[0] * np.cos(u) + B[0] * np.sin(u))
        y = cy + tube_r * (N[1] * np.cos(u) + B[1] * np.sin(u))
        z = cz + tube_r * (N[2] * np.cos(u) + B[2] * np.sin(u))
        return np.array([x, y, z])

    return func


def _arp_decay(alpha_0, t, mu=MU, v=V_RAMP):
    """ARP filament twist decay: α(t) = α₀ · exp(-μ t) · cos(v t)."""
    return alpha_0 * np.exp(-mu * t) * np.cos(v * t)


def _make_hopf_pair(alpha, res=(16, 48)):
    """Build blue + red Hopf link tube pair at given alpha."""
    comp_a = SDFSurface(
        _hopf_component_a(alpha),
        u_range=[0, _TAU],
        v_range=[0, _TAU],
        resolution=res,
        color=BLUE_D,
        opacity=0.75,
    )
    comp_b = SDFSurface(
        _hopf_component_b(alpha),
        u_range=[0, _TAU],
        v_range=[0, _TAU],
        resolution=res,
        color=RED_E,
        opacity=0.75,
    )
    return VGroup(comp_a, comp_b)


# ---- Scene 1: Hopf Unlinking α Sweep --------------------------------------

class HopfUnlinkingSweep(ThreeDScene):
    """Sweep α from 1→0 to unlink Hopf link with ARP twist decay."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)

        title = Text("Hopf Link Unlinking — ARP Filament Untwist",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\alpha(t) = \alpha_0 \, e^{-\mu t}\cos(v\,t)"
            r"\quad \mu{=}0.8,\; v{=}1.2",
            font_size=22,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Fully linked state
        link = _make_hopf_pair(1.0)
        self.play(FadeIn(link), run_time=2)
        self.play(Rotate(link, angle=PI / 4, axis=UP), run_time=1.5)

        # Sweep through α values using ARP decay
        time_steps = [0.3, 0.8, 1.4, 2.2, 3.5]
        for t_val in time_steps:
            alpha = float(np.clip(_arp_decay(1.0, t_val), 0.0, 1.0))
            target = _make_hopf_pair(alpha)
            label = MathTex(
                rf"\alpha = {alpha:.2f}\quad (t={t_val:.1f})",
                font_size=22,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(label)
            self.play(Transform(link, target), FadeIn(label), run_time=2)
            self.play(Rotate(link, angle=PI / 3, axis=UP), run_time=1)
            self.remove(label)

        # Final unlinked state
        target_final = _make_hopf_pair(0.0)
        lbl_final = MathTex(r"\alpha = 0 \;\text{(unlinked)}",
                            font_size=22).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(lbl_final)
        self.play(Transform(link, target_final), FadeIn(lbl_final), run_time=2.5)
        self.play(Rotate(link, angle=TAU, axis=UP), run_time=3)
        self.play(FadeOut(link), FadeOut(eq), FadeOut(title), FadeOut(lbl_final))
        self.wait(0.5)


# ---- Scene 2: ARP Filament Decay Dynamics ----------------------------------

class ARPFilamentDecay(ThreeDScene):
    """Show ARP decay on a single filament with μ=0.8, v=1.2 oscillation."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-35 * DEGREES)

        title = Text("ARP Filament Twist Decay", font_size=30).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\alpha(t) = e^{-0.8\,t}\cos(1.2\,t)",
            font_size=24,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Build initial twisted filament (full linking)
        pair = _make_hopf_pair(1.0, res=(14, 40))
        self.play(FadeIn(pair), run_time=1.5)

        # Fine-grained ARP sweep showing oscillatory decay
        t_vals = np.linspace(0, 5.0, 12)
        colors_a = [BLUE_D, BLUE, BLUE_E, BLUE_D, BLUE, TEAL,
                     TEAL, GREEN, GREEN, GREEN, BLUE, BLUE_D]
        colors_b = [RED_E, RED, ORANGE, RED_E, RED, GOLD,
                     GOLD, YELLOW, YELLOW, YELLOW, RED, RED_E]

        for i, t_val in enumerate(t_vals):
            raw_alpha = _arp_decay(1.0, float(t_val))
            alpha = float(np.clip(abs(raw_alpha), 0.0, 1.0))

            comp_a = SDFSurface(
                _hopf_component_a(alpha),
                u_range=[0, _TAU],
                v_range=[0, _TAU],
                resolution=(14, 40),
                color=colors_a[i],
                opacity=0.7,
            )
            comp_b = SDFSurface(
                _hopf_component_b(alpha),
                u_range=[0, _TAU],
                v_range=[0, _TAU],
                resolution=(14, 40),
                color=colors_b[i],
                opacity=0.7,
            )
            target = VGroup(comp_a, comp_b)

            lbl = MathTex(
                rf"t={t_val:.1f},\;\alpha={raw_alpha:.2f}",
                font_size=20,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(lbl)
            self.play(Transform(pair, target), FadeIn(lbl), run_time=1.5)
            self.remove(lbl)

        self.play(Rotate(pair, angle=PI, axis=UP), run_time=2)
        self.play(FadeOut(pair), FadeOut(eq), FadeOut(title))
        self.wait(0.5)
