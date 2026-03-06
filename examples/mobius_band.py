"""Möbius band evolution with boundary instabilities and untwist.

A Möbius strip is built as a parametric surface, then:
  1) boundary instabilities grow (wavy perturbation on edges),
  2) the half-twist parameter continuously unwinds to produce a
     regular (orientable) strip.

All geometry is parametric Surface — no triangles.

Run:
    manim -pql examples/mobius_band.py MobiusEvolution
    manim -pql examples/mobius_band.py MobiusUntwist
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
    TEAL,
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


# ---- Parametric Möbius factories -------------------------------------------

def mobius_parametric(half_twists=1, width=0.5, R=1.2, wave_amp=0.0, wave_freq=5):
    """Parametric Möbius band with controllable twist and boundary waves.

    Parameters
    ----------
    half_twists : float
        Number of half-twists (1 = Möbius, 0 = flat strip, 2 = full twist).
    width : float
        Half-width of the strip.
    R : float
        Major radius of the band loop.
    wave_amp : float
        Amplitude of boundary instability perturbation.
    wave_freq : int
        Frequency of boundary waves.
    """

    def func(u, v):
        # u ∈ [0, TAU] → angle around the loop
        # v ∈ [0, TAU] → mapped to [-width, width] across strip
        s = width * (2.0 * v / _TAU - 1.0)   # [-width, width]

        # Boundary instability: add wavy perturbation scaled by |s|
        if wave_amp > 0:
            edge_factor = abs(s) / width  # 0 at centre, 1 at edge
            perturbation = wave_amp * edge_factor ** 2 * np.sin(wave_freq * u)
            s_eff = s + perturbation
        else:
            s_eff = s

        twist_angle = half_twists * u / 2.0

        x = (R + s_eff * np.cos(twist_angle)) * np.cos(u)
        y = (R + s_eff * np.cos(twist_angle)) * np.sin(u)
        z = s_eff * np.sin(twist_angle)
        return np.array([x, y, z])

    return func


# ---- Scene 1: Evolution with boundary instabilities -----------------------

class MobiusEvolution(ThreeDScene):
    """Möbius band with growing boundary instabilities."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-35 * DEGREES)

        title = Text("Möbius Band — Boundary Instabilities", font_size=30).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\mathbf{r}(u,s) = "
            r"\bigl(R + s\cos\tfrac{u}{2}\bigr)"
            r"\bigl(\cos u,\, \sin u\bigr),\;"
            r"s\sin\tfrac{u}{2}",
            font_size=20,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Clean Möbius strip
        mobius = SDFSurface(
            mobius_parametric(half_twists=1, wave_amp=0.0),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(48, 16),
            color=BLUE_D,
            opacity=0.8,
        )
        self.play(FadeIn(mobius), run_time=2)
        self.play(Rotate(mobius, angle=PI / 2, axis=UP), run_time=2)

        # Grow boundary instabilities
        wave_amps = [0.05, 0.12, 0.22, 0.35]
        colors = [BLUE, TEAL, GREEN, ORANGE]

        for amp, col in zip(wave_amps, colors):
            target = SDFSurface(
                mobius_parametric(half_twists=1, wave_amp=amp, wave_freq=7),
                u_range=[0, TAU],
                v_range=[0, TAU],
                resolution=(48, 16),
                color=col,
                opacity=0.8,
            )
            amp_label = MathTex(
                rf"A_{{\mathrm{{wave}}}} = {amp:.2f}",
                font_size=22,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(amp_label)
            self.play(Transform(mobius, target), FadeIn(amp_label), run_time=2)
            self.play(Rotate(mobius, angle=PI / 3, axis=UP), run_time=1.5)
            self.remove(amp_label)

        # Return to clean
        clean = SDFSurface(
            mobius_parametric(half_twists=1, wave_amp=0.0),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(48, 16),
            color=BLUE_D,
            opacity=0.8,
        )
        self.play(Transform(mobius, clean), run_time=2)
        self.play(Rotate(mobius, angle=TAU / 2, axis=UP), run_time=2)
        self.play(FadeOut(mobius), FadeOut(eq), FadeOut(title))
        self.wait(0.5)


# ---- Scene 2: Untwist animation -------------------------------------------

class MobiusUntwist(ThreeDScene):
    """Continuously untwist a Möbius band from 1 half-twist → 0.

    Topological transition: non-orientable → orientable.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = Text("Möbius Untwist: Non-orientable → Orientable", font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"n_{\mathrm{twist}} : 1 \;\to\; 0",
            font_size=28,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Start: Möbius (1 half-twist)
        mobius = SDFSurface(
            mobius_parametric(half_twists=1.0),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(48, 16),
            color=PURPLE,
            opacity=0.8,
        )
        self.play(FadeIn(mobius), run_time=2)
        self.play(Rotate(mobius, angle=PI / 3, axis=UP), run_time=1.5)

        # Continuous untwist
        twist_values = [0.75, 0.5, 0.25, 0.0]
        colors = [PURPLE, BLUE, TEAL, GREEN]

        for tw, col in zip(twist_values, colors):
            target = SDFSurface(
                mobius_parametric(half_twists=tw),
                u_range=[0, TAU],
                v_range=[0, TAU],
                resolution=(48, 16),
                color=col,
                opacity=0.8,
            )
            tw_label = MathTex(
                rf"n_{{\mathrm{{twist}}}} = {tw:.2f}",
                font_size=22,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(tw_label)
            self.play(Transform(mobius, target), FadeIn(tw_label), run_time=2.5)
            self.play(Rotate(mobius, angle=PI / 4, axis=UP), run_time=1)
            self.remove(tw_label)

        orientable_label = MathTex(
            r"\text{Orientable strip } (n=0)",
            font_size=22,
        ).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(orientable_label)
        self.play(FadeIn(orientable_label))

        self.play(Rotate(mobius, angle=TAU, axis=UP), run_time=3)
        self.play(FadeOut(mobius), FadeOut(eq), FadeOut(title), FadeOut(orientable_label))
        self.wait(0.5)
