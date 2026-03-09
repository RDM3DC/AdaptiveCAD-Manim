"""Adiabatic Spin Precession — Cone on the Bloch Sphere.

A spin-½ particle in a slowly rotating magnetic field adiabatically
follows the field direction, tracing a cone on the Bloch sphere.
The accumulated geometric phase is shown as an unwrapping ribbon.

Left panel:  Bloch sphere with precession cone + state trajectory
Right panel: Phase ribbon unwrapping, colour-coded by accumulated angle

Acts
----
1. Title card
2. Draw Bloch sphere + magnetic field arrow
3. Animate precession: state dot traces cone circumference
4. Show accumulated phase ribbon on right
5. Camera orbit
6. Summary card

Run
---
    manim -pql examples/bloch_adiabatic.py AdiabaticSpin
    manim -qh  examples/bloch_adiabatic.py AdiabaticSpin
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Sphere,
    Arrow3D,
    Dot3D,
    Line3D,
    ParametricFunction,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    SurroundingRectangle,
    PI,
    TAU,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    YELLOW,
    RED,
    GREEN,
    BLUE,
    BLUE_D,
    BLUE_E,
    ORANGE,
    GOLD,
    TEAL,
    GREY_A,
    GREY_D,
    PURPLE,
    PINK,
    interpolate_color,
)

DEG = PI / 180
BLOCH_R = 1.5
LEFT_X = -3.0
RIGHT_X = 3.2
CONE_THETA = 40 * DEG  # half-angle of precession cone


def _cone_surface_func(R, theta, cx=0.0):
    """Cone from (cx, 0, R) down to colatitude theta on sphere centred at cx."""

    def func(u, v):
        # v ∈ [0,1]: apex (north pole) → base circle
        r = v * R * np.sin(theta)
        z = R - v * R * (1 - np.cos(theta))
        return np.array([cx + r * np.cos(u), r * np.sin(u), z])

    return func


def _phase_ribbon_func(R, height, cx=0.0, half_w=PI / 4):
    """Flat radial ribbon that spirals upward — represents accumulated phase."""
    z_base = -height / 2

    def func(u, v):
        angle = u + (2 * v - 1) * half_w
        z = z_base + height * u / TAU
        return np.array([cx + R * np.cos(angle),
                         R * np.sin(angle), z])

    return func


class AdiabaticSpin(ThreeDScene):
    """Adiabatic spin precession and geometric phase accumulation."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Adiabatic Spin Precession",
                    font_size=44, color=GOLD)
        sub = Text("Geometric phase from a rotating field",
                    font_size=22, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"\gamma = -\pi(1 - \cos\theta)",
            font_size=28, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Bloch sphere + field arrow ──────────────────────────
        self.set_camera_orientation(phi=65 * DEG, theta=-50 * DEG)

        sc = np.array([LEFT_X, 0, 0])
        hc = np.array([RIGHT_X, 0, 0])

        sphere = Sphere(
            radius=BLOCH_R, resolution=(40, 28),
            fill_opacity=0.06, stroke_width=0.3,
            stroke_color=BLUE_E,
        )
        sphere.set_color(BLUE_E)
        sphere.move_to(sc)

        al = BLOCH_R * 1.25
        z_ax = Arrow3D(sc - al * np.array([0, 0, 1]),
                        sc + al * np.array([0, 0, 1]),
                        color=BLUE, thickness=0.01)

        equator = ParametricFunction(
            lambda t: sc + BLOCH_R * np.array(
                [np.cos(t), np.sin(t), 0]),
            t_range=[0, TAU], color=GREY_A, stroke_width=0.7,
        )

        # Magnetic field arrow — tilted at CONE_THETA from z
        B_end = sc + BLOCH_R * 1.1 * np.array([
            np.sin(CONE_THETA), 0, np.cos(CONE_THETA),
        ])
        B_arrow = Arrow3D(sc, B_end, color=RED, thickness=0.02)

        self.play(
            Create(sphere), Create(equator), Create(z_ax),
            Create(B_arrow),
            run_time=1.5,
        )

        lbl_B = MathTex(r"\vec{B}(t)", font_size=18, color=RED)
        lbl_B.move_to(UP * 2.0 + LEFT * 1.8)
        lbl_z = MathTex(r"\hat{z}", font_size=16, color=BLUE)
        lbl_z.move_to(UP * 2.6 + LEFT * 3.2)
        self.add_fixed_in_frame_mobjects(lbl_B, lbl_z)
        self.play(FadeIn(lbl_B), FadeIn(lbl_z), run_time=0.4)

        # ─── Act 3  Precession cone + trajectory ────────────────────────
        # Cone surface (translucent)
        cone = Surface(
            _cone_surface_func(BLOCH_R, CONE_THETA, cx=LEFT_X),
            u_range=[0, TAU], v_range=[0, 1],
            resolution=(32, 8),
            fill_color=ORANGE, fill_opacity=0.2,
            stroke_width=0.2, stroke_color=ORANGE,
        )

        # Circular trajectory at colatitude
        N_SEG = 80
        t_vals = np.linspace(0, TAU, N_SEG + 1)
        rain = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]

        def _rc(frac):
            n = len(rain) - 1
            idx = frac * n
            lo = min(int(idx), n - 1)
            return interpolate_color(rain[lo], rain[lo + 1], idx - lo)

        traj = VGroup()
        for i in range(N_SEG):
            t0, t1 = t_vals[i], t_vals[i + 1]
            p0 = sc + BLOCH_R * np.array([
                np.sin(CONE_THETA) * np.cos(t0),
                np.sin(CONE_THETA) * np.sin(t0),
                np.cos(CONE_THETA),
            ])
            p1 = sc + BLOCH_R * np.array([
                np.sin(CONE_THETA) * np.cos(t1),
                np.sin(CONE_THETA) * np.sin(t1),
                np.cos(CONE_THETA),
            ])
            traj.add(Line3D(p0, p1, color=_rc(i / N_SEG),
                            thickness=0.016))

        state_dot = Dot3D(
            sc + BLOCH_R * np.array([
                np.sin(CONE_THETA), 0, np.cos(CONE_THETA)]),
            radius=0.07, color=RED,
        )

        self.play(FadeIn(cone), run_time=0.8)
        self.play(
            Create(traj, lag_ratio=0.02),
            FadeIn(state_dot),
            run_time=2.5,
        )

        # ─── Act 4  Phase ribbon on right ────────────────────────────────
        ribbon_h = 2.5
        ribbon_r = 0.55
        ribbon = Surface(
            _phase_ribbon_func(ribbon_r, ribbon_h,
                               cx=RIGHT_X, half_w=PI / 3),
            u_range=[0, TAU], v_range=[0, 1],
            resolution=(48, 6),
            fill_color=ORANGE, fill_opacity=0.8,
            stroke_width=0.3, stroke_color=ORANGE,
        )
        ribbon2 = Surface(
            _phase_ribbon_func(ribbon_r, ribbon_h,
                               cx=RIGHT_X, half_w=PI / 3),
            u_range=[PI, PI + TAU], v_range=[0, 1],
            resolution=(48, 6),
            fill_color=BLUE_D, fill_opacity=0.8,
            stroke_width=0.3, stroke_color=BLUE_D,
        )

        h_ax = Line3D(
            hc + np.array([0, 0, -ribbon_h * 0.6]),
            hc + np.array([0, 0,  ribbon_h * 0.6]),
            color=WHITE, thickness=0.003,
        )

        # Phase value
        gamma_val = -PI * (1 - np.cos(CONE_THETA))
        gamma_deg = np.degrees(gamma_val)

        phase_lbl = MathTex(
            r"\gamma = " + f"{gamma_deg:.1f}" + r"^\circ",
            font_size=22, color=YELLOW,
        )
        phase_lbl.to_edge(DOWN, buff=0.4)

        theta_lbl = MathTex(
            r"\theta = " + f"{np.degrees(CONE_THETA):.0f}"
            + r"^\circ",
            font_size=18, color=ORANGE,
        )
        theta_lbl.to_corner(UP + RIGHT, buff=0.4)

        for lbl in [phase_lbl, theta_lbl]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            Create(h_ax),
            Create(ribbon), Create(ribbon2),
            FadeIn(phase_lbl), FadeIn(theta_lbl),
            run_time=1.5,
        )
        self.wait(0.5)

        # ─── Act 5  Camera orbit ─────────────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(3.5)
        self.stop_ambient_camera_rotation()

        # ─── Act 6  Summary ──────────────────────────────────────────────
        all_3d = [sphere, equator, z_ax, B_arrow, cone, traj,
                  state_dot, h_ax, ribbon, ribbon2]
        frame_lbls = [lbl_B, lbl_z, phase_lbl, theta_lbl]

        self.play(
            *[FadeOut(m) for m in all_3d],
            *[FadeOut(lbl) for lbl in frame_lbls],
            run_time=0.8,
        )
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Adiabatic Spin", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\vec{B}(t)\;\text{rotates slowly} \Rightarrow"
                r"\;\text{spin follows adiabatically}",
                font_size=18,
            ),
            MathTex(
                r"\text{Precession cone at colatitude }\theta"
                r"\;\text{on Bloch sphere}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\gamma = -\pi(1-\cos\theta)"
                r"\;\;\text{(geometric / Berry phase)}",
                font_size=18, color=YELLOW,
            ),
            MathTex(
                r"\text{Phase ribbon unwraps the accumulated angle}",
                font_size=18, color=TEAL,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card, DOWN, buff=0.35)
        box = SurroundingRectangle(
            VGroup(card, bullets), color=GOLD, buff=0.25,
            corner_radius=0.1,
        )
        self.add_fixed_in_frame_mobjects(card, box, *bullets)
        self.play(Write(card), FadeIn(box), run_time=0.8)
        for b in bullets:
            self.play(FadeIn(b), run_time=0.5)
        self.wait(2)
        self.play(
            FadeOut(card), FadeOut(box),
            *[FadeOut(b) for b in bullets],
            run_time=1,
        )
