"""Topological Protection — Robustness of the Winding Number.

Side-by-side comparison: a clean w=2 helix versus a noisy / perturbed
one.  Despite the random bumps the winding number stays w=2 — the
topological invariant is immune to smooth perturbations.

Acts
----
1. Title card
2. Build two helices side by side: clean (left) and noisy (right)
3. Overlay winding labels — both show w=2
4. Camera orbit
5. Summary card

Run
---
    manim -pql examples/bloch_topo_protect.py TopologicalProtection
    manim -qh  examples/bloch_topo_protect.py TopologicalProtection
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Line3D,
    Dot3D,
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
    PURPLE,
    interpolate_color,
)

DEG = PI / 180
N_WINDS = 2
HELIX_R = 0.65
HELIX_H = 3.4
HALF_W = PI / 3
LEFT_X = -2.8
RIGHT_X = 2.8


def _clean_ribbon(R, n_winds, height, half_w, phase, cx, color):
    pitch = height / n_winds
    z_base = -height / 2

    def func(u, v):
        angle = u + phase + (2 * v - 1) * half_w
        z = z_base + pitch * u / TAU
        return np.array([cx + R * np.cos(angle),
                         R * np.sin(angle), z])

    return Surface(
        func,
        u_range=[0, n_winds * TAU], v_range=[0, 1],
        resolution=(64, 6),
        fill_color=color, fill_opacity=0.85,
        stroke_width=0.3, stroke_color=color,
    )


def _noisy_ribbon(R, n_winds, height, half_w, phase, cx, color, seed=42):
    """Ribbon with smooth radial + height perturbations."""
    rng = np.random.RandomState(seed)
    # Pre-compute noise coefficients for reproducibility
    n_modes = 8
    amp_r = rng.uniform(-0.12, 0.12, n_modes)
    freq_r = rng.uniform(1, 5, n_modes)
    phase_r = rng.uniform(0, TAU, n_modes)
    amp_z = rng.uniform(-0.08, 0.08, n_modes)
    freq_z = rng.uniform(1, 6, n_modes)
    phase_z = rng.uniform(0, TAU, n_modes)

    pitch = height / n_winds
    z_base = -height / 2

    def func(u, v):
        # noise on radius
        dr = sum(a * np.sin(f * u + p)
                 for a, f, p in zip(amp_r, freq_r, phase_r))
        dz = sum(a * np.sin(f * u + p)
                 for a, f, p in zip(amp_z, freq_z, phase_z))
        r = R + dr
        angle = u + phase + (2 * v - 1) * half_w
        z = z_base + pitch * u / TAU + dz
        return np.array([cx + r * np.cos(angle),
                         r * np.sin(angle), z])

    return Surface(
        func,
        u_range=[0, n_winds * TAU], v_range=[0, 1],
        resolution=(80, 6),
        fill_color=color, fill_opacity=0.85,
        stroke_width=0.3, stroke_color=color,
    )


class TopologicalProtection(ThreeDScene):
    """Show that smooth perturbations don't change the winding number."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Topological Protection", font_size=46, color=GOLD)
        sub = Text("Winding number resists smooth perturbations",
                    font_size=22, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub))

        # ─── Act 2  Two helices side by side ─────────────────────────────
        self.set_camera_orientation(phi=62 * DEG, theta=-48 * DEG)

        # Central axes
        ax_L = Line3D(
            np.array([LEFT_X, 0, -HELIX_H * 0.55]),
            np.array([LEFT_X, 0,  HELIX_H * 0.55]),
            color=WHITE, thickness=0.003,
        )
        ax_R = Line3D(
            np.array([RIGHT_X, 0, -HELIX_H * 0.55]),
            np.array([RIGHT_X, 0,  HELIX_H * 0.55]),
            color=WHITE, thickness=0.003,
        )

        # Clean ribbons (left)
        cA = _clean_ribbon(HELIX_R, N_WINDS, HELIX_H, HALF_W,
                           0, LEFT_X, ORANGE)
        cB = _clean_ribbon(HELIX_R, N_WINDS, HELIX_H, HALF_W,
                           PI, LEFT_X, BLUE_D)

        # Noisy ribbons (right)
        nA = _noisy_ribbon(HELIX_R, N_WINDS, HELIX_H, HALF_W,
                           0, RIGHT_X, ORANGE, seed=42)
        nB = _noisy_ribbon(HELIX_R, N_WINDS, HELIX_H, HALF_W,
                           PI, RIGHT_X, BLUE_D, seed=99)

        self.play(
            Create(ax_L), Create(ax_R),
            Create(cA), Create(cB),
            Create(nA), Create(nB),
            run_time=2,
        )

        # ─── Act 3  Winding labels ──────────────────────────────────────
        lbl_clean = Text("Clean", font_size=18, color=ORANGE)
        lbl_clean.move_to(LEFT * 2.8 + UP * 2.5)
        lbl_noisy = Text("Perturbed", font_size=18, color=TEAL)
        lbl_noisy.move_to(RIGHT * 2.8 + UP * 2.5)

        w_clean = MathTex(r"w = 2", font_size=26, color=YELLOW)
        w_clean.move_to(LEFT * 2.8 + DOWN * 2.5)
        w_noisy = MathTex(r"w = 2", font_size=26, color=YELLOW)
        w_noisy.move_to(RIGHT * 2.8 + DOWN * 2.5)

        eq_lbl = MathTex(r"=", font_size=36, color=GREEN)
        eq_lbl.move_to(DOWN * 2.5)

        for lbl in [lbl_clean, lbl_noisy, w_clean, w_noisy, eq_lbl]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(lbl_clean), FadeIn(lbl_noisy),
            FadeIn(w_clean), FadeIn(w_noisy), FadeIn(eq_lbl),
            run_time=0.8,
        )
        self.wait(0.5)

        # ─── Act 4  Camera orbit ─────────────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        # ─── Act 5  Summary ──────────────────────────────────────────────
        all_3d = [ax_L, ax_R, cA, cB, nA, nB]
        frame_lbls = [lbl_clean, lbl_noisy, w_clean, w_noisy, eq_lbl]

        self.play(
            *[FadeOut(m) for m in all_3d],
            *[FadeOut(lbl) for lbl in frame_lbls],
            run_time=0.8,
        )
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Topological Protection", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"w \in \mathbb{Z}\;\text{is a topological invariant}",
                font_size=19,
            ),
            MathTex(
                r"\text{Smooth perturbations} \;\not\!\!\Rightarrow"
                r"\; \Delta w",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Only singularities (gap closure) change } w",
                font_size=19, color=ORANGE,
            ),
            MathTex(
                r"\text{Foundation of topological quantum computing}",
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
