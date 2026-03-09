"""Berry Phase on the Bloch Sphere.

A qubit state traces a *closed* loop on S².  The geometric (Berry) phase
acquired equals minus half the solid angle Ω enclosed by the loop:

    γ_B = −Ω / 2

Acts
----
1. Title card
2. Draw Bloch sphere with axes
3. Trace a closed conical loop at colatitude θ₀
4. Shade the solid-angle cap, display γ_B in real-time
5. Camera orbit
6. Summary card

Run
---
    manim -pql examples/bloch_berry_phase.py BerryPhase
    manim -qh  examples/bloch_berry_phase.py BerryPhase
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Sphere,
    Arrow3D,
    Dot3D,
    ParametricFunction,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Transform,
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
    interpolate_color,
)

DEG = PI / 180
BLOCH_R = 1.8


def _sphere_cap_func(R, theta_max):
    """Parametric spherical cap from north pole to colatitude theta_max."""

    def func(u, v):
        theta = v * theta_max       # v ∈ [0,1] → [0, theta_max]
        return np.array([
            R * np.sin(theta) * np.cos(u),
            R * np.sin(theta) * np.sin(u),
            R * np.cos(theta),
        ])

    return func


class BerryPhase(ThreeDScene):
    """Berry phase from a closed loop on the Bloch sphere."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Berry Phase", font_size=48, color=GOLD)
        sub = Text("Geometric phase from cyclic evolution",
                    font_size=22, color=BLUE_D)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"\gamma_B = -\frac{\Omega}{2}",
            font_size=32, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Bloch sphere ─────────────────────────────────────────
        self.set_camera_orientation(phi=65 * DEG, theta=-50 * DEG)

        sphere = Sphere(
            radius=BLOCH_R, resolution=(40, 28),
            fill_opacity=0.06, stroke_width=0.3,
            stroke_color=BLUE_E,
        )
        sphere.set_color(BLUE_E)

        al = BLOCH_R * 1.25
        x_ax = Arrow3D(-al * RIGHT, al * RIGHT,
                        color=RED, thickness=0.01)
        y_ax = Arrow3D(-al * np.array([0, 1, 0]),
                        al * np.array([0, 1, 0]),
                        color=GREEN, thickness=0.01)
        z_ax = Arrow3D(-al * np.array([0, 0, 1]),
                        al * np.array([0, 0, 1]),
                        color=BLUE, thickness=0.01)

        equator = ParametricFunction(
            lambda t: BLOCH_R * np.array([np.cos(t), np.sin(t), 0]),
            t_range=[0, TAU], color=GREY_A, stroke_width=0.8,
        )

        lbl_0 = MathTex(r"|0\rangle", font_size=20, color=BLUE)
        lbl_0.move_to(UP * 2.6 + LEFT * 0.3)
        lbl_1 = MathTex(r"|1\rangle", font_size=20, color=BLUE)
        lbl_1.move_to(DOWN * 2.6 + LEFT * 0.3)
        for lbl in [lbl_0, lbl_1]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            Create(sphere), Create(equator),
            Create(x_ax), Create(y_ax), Create(z_ax),
            FadeIn(lbl_0), FadeIn(lbl_1),
            run_time=1.5,
        )

        # ─── Act 3  Trace loop at colatitude θ₀ ─────────────────────────
        theta0 = 55 * DEG  # colatitude of the loop

        loop = ParametricFunction(
            lambda t: BLOCH_R * np.array([
                np.sin(theta0) * np.cos(t),
                np.sin(theta0) * np.sin(t),
                np.cos(theta0),
            ]),
            t_range=[0, TAU],
            color=YELLOW,
            stroke_width=3,
        )
        dot_loop = Dot3D(
            BLOCH_R * np.array([np.sin(theta0), 0, np.cos(theta0)]),
            radius=0.07, color=RED,
        )

        lbl_loop = MathTex(
            r"\theta_0 = 55^\circ",
            font_size=18, color=YELLOW,
        )
        lbl_loop.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_loop)

        self.play(Create(loop), FadeIn(dot_loop),
                  FadeIn(lbl_loop), run_time=1.5)

        # ─── Act 4  Shade solid-angle cap, display γ_B ──────────────────
        cap = Surface(
            _sphere_cap_func(BLOCH_R * 1.002, theta0),
            u_range=[0, TAU],
            v_range=[0, 1],
            resolution=(32, 12),
            fill_color=ORANGE,
            fill_opacity=0.35,
            stroke_width=0,
        )

        solid_angle = 2 * PI * (1 - np.cos(theta0))
        berry = -solid_angle / 2
        berry_deg = np.degrees(berry)

        phase_lbl = MathTex(
            r"\Omega = 2\pi(1-\cos\theta_0) = "
            + f"{solid_angle:.2f}" + r"\;\mathrm{sr}",
            font_size=18, color=ORANGE,
        )
        phase_lbl.to_edge(DOWN, buff=0.5)
        berry_lbl = MathTex(
            r"\gamma_B = -\Omega/2 = "
            + f"{berry_deg:.1f}" + r"^\circ",
            font_size=20, color=YELLOW,
        )
        berry_lbl.next_to(phase_lbl, UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(phase_lbl, berry_lbl)

        self.play(
            FadeIn(cap),
            FadeIn(phase_lbl), FadeIn(berry_lbl),
            run_time=1.2,
        )
        self.wait(0.5)

        # Show a second loop at different angle for comparison
        theta1 = 80 * DEG
        loop2 = ParametricFunction(
            lambda t: BLOCH_R * np.array([
                np.sin(theta1) * np.cos(t),
                np.sin(theta1) * np.sin(t),
                np.cos(theta1),
            ]),
            t_range=[0, TAU], color=TEAL, stroke_width=2.5,
        )
        cap2 = Surface(
            _sphere_cap_func(BLOCH_R * 1.004, theta1),
            u_range=[0, TAU], v_range=[0, 1],
            resolution=(32, 12),
            fill_color=TEAL, fill_opacity=0.2,
            stroke_width=0,
        )
        solid2 = 2 * PI * (1 - np.cos(theta1))
        berry2_deg = np.degrees(-solid2 / 2)
        berry_lbl2 = MathTex(
            r"\theta_0=80^\circ:\;\gamma_B = "
            + f"{berry2_deg:.1f}" + r"^\circ",
            font_size=18, color=TEAL,
        )
        berry_lbl2.next_to(berry_lbl, UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(berry_lbl2)

        self.play(
            Create(loop2), FadeIn(cap2), FadeIn(berry_lbl2),
            run_time=1,
        )
        self.wait(0.5)

        # ─── Act 5  Camera orbit ─────────────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # ─── Act 6  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in [lbl_0, lbl_1, lbl_loop,
              phase_lbl, berry_lbl, berry_lbl2]],
            run_time=0.8,
        )

        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Berry Phase", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\text{Cyclic evolution on }S^2"
                r"\;\Rightarrow\;\text{geometric phase}",
                font_size=19,
            ),
            MathTex(
                r"\gamma_B = -\Omega/2"
                r"\;\;(\Omega = \text{solid angle enclosed})",
                font_size=19, color=ORANGE,
            ),
            MathTex(
                r"\text{Larger loop}\;\Rightarrow"
                r"\;\text{larger Berry phase}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Observable in interferometry, "
                r"adiabatic transport}",
                font_size=18, color=YELLOW,
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
