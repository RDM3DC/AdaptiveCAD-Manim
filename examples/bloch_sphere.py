"""Bloch Sphere — Quantum State Visualization.

A pure qubit state |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
maps to a point on the Bloch sphere S^2.

• Gates are rotations: Hadamard = 180 deg about (X+Z)/sqrt(2)
• X gate = 180 deg about X axis  (bit flip)
• Z gate = 180 deg about Z axis  (phase flip)
• Decoherence shrinks the Bloch vector toward the centre

Acts
----
1. Title card
2. Draw Bloch sphere with |0>, |1>, |+>, |-> labels
3. State vector + coordinates
4. X gate rotation
5. Hadamard gate rotation
6. Z gate rotation
7. Decoherence (shrink toward centre)
8. Summary card

Run
---
    manim -pql examples/bloch_sphere.py BlochSphere
    manim -qh  examples/bloch_sphere.py BlochSphere
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Sphere,
    Arrow3D,
    Line3D,
    Dot3D,
    ParametricFunction,
    VMobject,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Uncreate,
    Transform,
    Indicate,
    Rotate,
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
    RED_E,
    GREEN,
    BLUE,
    BLUE_D,
    BLUE_E,
    ORANGE,
    GOLD,
    TEAL,
    GREY,
    GREY_A,
    GREY_D,
    PURPLE,
    PINK,
    config,
    interpolate_color,
    np as mnp,
)

# ═══════════════════════════════════════════════════════════════════════════
# Bloch geometry helpers
# ═══════════════════════════════════════════════════════════════════════════
BLOCH_R = 1.8  # sphere display radius


def _bloch_xyz(theta, phi):
    """Bloch angles -> Cartesian on the display sphere."""
    return BLOCH_R * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _state_arrow(theta, phi, color=YELLOW):
    """Arrow from origin to state point on Bloch sphere."""
    end = _bloch_xyz(theta, phi)
    return Arrow3D(
        start=ORIGIN, end=end,
        color=color,
        thickness=0.025,
        height=0.2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class BlochSphere(ThreeDScene):
    """Qubit states on the Bloch sphere with gate rotations."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("The Bloch Sphere", font_size=48, color=GOLD)
        sub = Text("Qubit states as points on S^2",
                    font_size=24, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"|\psi\rangle = \cos\frac{\theta}{2}|0\rangle"
            r" + e^{i\phi}\sin\frac{\theta}{2}|1\rangle",
            font_size=28, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1.2)                              # 1
        self.play(FadeIn(sub), run_time=0.8)                             # 2
        self.play(Write(eq), run_time=1)                                 # 3
        self.wait(0.8)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))               # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Draw Bloch sphere
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(
            phi=70 * PI / 180, theta=-55 * PI / 180,
        )

        sphere = Sphere(
            radius=BLOCH_R,
            resolution=(32, 24),
            fill_opacity=0.08,
            stroke_width=0.4,
            stroke_color=GREY_D,
        )
        sphere.set_color(BLUE_E)

        # Axes
        x_ax = Arrow3D(
            start=-BLOCH_R * 1.3 * RIGHT,
            end=BLOCH_R * 1.3 * RIGHT,
            color=RED, thickness=0.012,
        )
        y_ax = Arrow3D(
            start=-BLOCH_R * 1.3 * UP,
            end=BLOCH_R * 1.3 * UP,
            color=GREEN, thickness=0.012,
        )
        z_ax = Arrow3D(
            start=-BLOCH_R * 1.3 * np.array([0, 0, 1]),
            end=BLOCH_R * 1.3 * np.array([0, 0, 1]),
            color=BLUE, thickness=0.012,
        )

        # Equator circle
        equator = ParametricFunction(
            lambda t: BLOCH_R * np.array([np.cos(t), np.sin(t), 0]),
            t_range=[0, TAU],
            color=GREY_A,
            stroke_width=1,
        )

        self.play(
            Create(sphere),
            Create(x_ax), Create(y_ax), Create(z_ax),
            Create(equator),
            run_time=1.5,
        )                                                                 # 5

        # Labels — |0⟩ at north, |1⟩ at south, |+⟩, |−⟩
        lbl_0 = MathTex(r"|0\rangle", font_size=22, color=BLUE)
        lbl_0.move_to(UP * 2.5 + LEFT * 0.3)
        lbl_1 = MathTex(r"|1\rangle", font_size=22, color=BLUE)
        lbl_1.move_to(DOWN * 2.5 + LEFT * 0.3)
        lbl_plus = MathTex(r"|+\rangle", font_size=20, color=RED)
        lbl_plus.move_to(RIGHT * 2.7)
        lbl_minus = MathTex(r"|-\rangle", font_size=20, color=RED)
        lbl_minus.move_to(LEFT * 2.7)

        for lbl in [lbl_0, lbl_1, lbl_plus, lbl_minus]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(lbl_0), FadeIn(lbl_1),
            FadeIn(lbl_plus), FadeIn(lbl_minus),
            run_time=0.6,
        )                                                                 # 6

        axis_labels = MathTex(
            r"X\;\; Y \;\; Z", font_size=16, color=GREY_A,
        )
        axis_labels.to_corner(DOWN + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(axis_labels)
        self.play(FadeIn(axis_labels), run_time=0.3)                     # 7

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — State vector |0⟩
        # ─────────────────────────────────────────────────────────────────
        theta_s, phi_s = 0, 0  # |0⟩ state
        state_vec = _state_arrow(theta_s, phi_s, YELLOW)
        tip_dot = Dot3D(
            _bloch_xyz(theta_s, phi_s), radius=0.06, color=YELLOW,
        )

        state_lbl = MathTex(
            r"|\psi\rangle = |0\rangle",
            font_size=22, color=YELLOW,
        )
        state_lbl.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(state_lbl)

        self.play(
            Create(state_vec), FadeIn(tip_dot),
            FadeIn(state_lbl),
            run_time=1,
        )                                                                 # 8

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — X gate: |0⟩ → |1⟩  (180° about X axis)
        # ─────────────────────────────────────────────────────────────────
        gate_lbl = MathTex(
            r"X\text{ gate: } 180^\circ \text{ about } \hat{x}",
            font_size=20, color=RED,
        )
        gate_lbl.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(gate_lbl)
        self.play(FadeIn(gate_lbl), run_time=0.4)                        # 9

        # Rotate state vector 180° about X
        theta_s, phi_s = PI, 0  # |1⟩
        new_vec = _state_arrow(theta_s, phi_s, RED)
        new_dot = Dot3D(
            _bloch_xyz(theta_s, phi_s), radius=0.06, color=RED,
        )
        new_state_lbl = MathTex(
            r"|\psi\rangle = |1\rangle",
            font_size=22, color=RED,
        )
        new_state_lbl.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(new_state_lbl)

        self.play(
            Transform(state_vec, new_vec),
            Transform(tip_dot, new_dot),
            Transform(state_lbl, new_state_lbl),
            run_time=1.2,
        )                                                                 # 10

        self.wait(0.5)
        self.play(FadeOut(gate_lbl), run_time=0.3)                       # 11

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Hadamard: |1⟩ → |−⟩  (180° about (X+Z)/√2)
        # ─────────────────────────────────────────────────────────────────
        gate_lbl2 = MathTex(
            r"H\text{ gate: } 180^\circ \text{ about } "
            r"(\hat{x}+\hat{z})/\sqrt{2}",
            font_size=20, color=TEAL,
        )
        gate_lbl2.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(gate_lbl2)
        self.play(FadeIn(gate_lbl2), run_time=0.4)                       # 12

        # H|1⟩ = |−⟩ = on the −X axis
        theta_s, phi_s = PI / 2, PI  # |−⟩
        new_vec2 = _state_arrow(theta_s, phi_s, TEAL)
        new_dot2 = Dot3D(
            _bloch_xyz(theta_s, phi_s), radius=0.06, color=TEAL,
        )
        new_state_lbl2 = MathTex(
            r"|\psi\rangle = |-\rangle",
            font_size=22, color=TEAL,
        )
        new_state_lbl2.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(new_state_lbl2)

        self.play(
            Transform(state_vec, new_vec2),
            Transform(tip_dot, new_dot2),
            Transform(state_lbl, new_state_lbl2),
            run_time=1.2,
        )                                                                 # 13

        self.wait(0.5)
        self.play(FadeOut(gate_lbl2), run_time=0.3)                      # 14

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Z gate: |−⟩ → |+⟩  (180° about Z axis)
        # ─────────────────────────────────────────────────────────────────
        gate_lbl3 = MathTex(
            r"Z\text{ gate: } 180^\circ \text{ about } \hat{z}",
            font_size=20, color=GREEN,
        )
        gate_lbl3.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(gate_lbl3)
        self.play(FadeIn(gate_lbl3), run_time=0.4)                       # 15

        # Z|−⟩ = −|+⟩ ≡ |+⟩ (global phase)
        theta_s, phi_s = PI / 2, 0  # |+⟩
        new_vec3 = _state_arrow(theta_s, phi_s, GREEN)
        new_dot3 = Dot3D(
            _bloch_xyz(theta_s, phi_s), radius=0.06, color=GREEN,
        )
        new_state_lbl3 = MathTex(
            r"|\psi\rangle = |+\rangle",
            font_size=22, color=GREEN,
        )
        new_state_lbl3.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(new_state_lbl3)

        self.play(
            Transform(state_vec, new_vec3),
            Transform(tip_dot, new_dot3),
            Transform(state_lbl, new_state_lbl3),
            run_time=1.2,
        )                                                                 # 16

        self.wait(0.5)
        self.play(FadeOut(gate_lbl3), run_time=0.3)                      # 17

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Decoherence: vector shrinks toward centre
        # ─────────────────────────────────────────────────────────────────
        deco_lbl = MathTex(
            r"\text{Decoherence: } |\vec{r}| \to 0",
            font_size=20, color=PURPLE,
        )
        deco_lbl.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(deco_lbl)
        self.play(FadeIn(deco_lbl), run_time=0.4)                        # 18

        # Shrink in steps
        for frac in [0.75, 0.50, 0.25, 0.08]:
            end_pt = frac * _bloch_xyz(theta_s, phi_s)
            shrunk_vec = Arrow3D(
                start=ORIGIN, end=end_pt,
                color=interpolate_color(GREEN, GREY, 1 - frac),
                thickness=0.025,
                height=0.15,
            )
            shrunk_dot = Dot3D(end_pt, radius=0.05,
                               color=interpolate_color(GREEN, GREY, 1 - frac))
            self.play(
                Transform(state_vec, shrunk_vec),
                Transform(tip_dot, shrunk_dot),
                run_time=0.6,
            )                                                     # 19-22

        deco_note = MathTex(
            r"\text{Maximally mixed state: } \rho = I/2",
            font_size=18, color=GREY_A,
        )
        deco_note.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(deco_note)
        self.play(
            Transform(state_lbl, deco_note),
            run_time=0.5,
        )                                                                 # 23

        self.wait(0.8)

        # ─────────────────────────────────────────────────────────────────
        # Act 8 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(state_lbl), FadeOut(deco_lbl),
            FadeOut(lbl_0), FadeOut(lbl_1),
            FadeOut(lbl_plus), FadeOut(lbl_minus),
            FadeOut(axis_labels),
            run_time=0.8,
        )                                                                 # 24

        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card_title = Text(
            "The Bloch Sphere", font_size=34, color=GOLD,
        )
        card_title.to_edge(UP, buff=0.5)
        self.add_fixed_in_frame_mobjects(card_title)
        self.play(Write(card_title))                                      # 25

        bullets = VGroup(
            MathTex(
                r"|\psi\rangle \leftrightarrow "
                r"(\theta,\phi) \in S^2"
                r"\;\;\text{(pure states on surface)}",
                font_size=18,
            ),
            MathTex(
                r"\text{Gates = rotations: } "
                r"X(\pi),\; H(\pi \text{ about } \frac{X+Z}{\sqrt{2}}),\; "
                r"Z(\pi)",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{Decoherence: Bloch vector shrinks } "
                r"|\vec{r}| \to 0 \;\;(\rho = I/2)",
                font_size=18, color=PURPLE,
            ),
            MathTex(
                r"|0\rangle = \text{north}, \;"
                r"|1\rangle = \text{south}, \;"
                r"|+\rangle = +\hat{x}, \;"
                r"|-\rangle = -\hat{x}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"SU(2) \to SO(3): "
                r"\text{qubit unitaries act as 3D rotations}",
                font_size=18, color=YELLOW,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, bullets),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.add_fixed_in_frame_mobjects(box, *bullets)
        self.play(FadeIn(box), run_time=0.5)                             # 26
        for b in bullets:
            self.play(FadeIn(b), run_time=0.6)                           # 27-31

        self.wait(2)
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(box),
            *[FadeOut(b) for b in bullets],
            run_time=1.5,
        )                                                                 # 32
