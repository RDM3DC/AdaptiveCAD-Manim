"""Winding Number Transition — Topological Phase Diagram.

Morphs a ribbon from winding number w=0 (flat ring) through w=1, w=2,
w=3, showing each additional twist appear as the topological invariant
changes.  A "phase dial" counter tracks the integer w in real time.

Acts
----
1. Title card
2. Build flat ring (w = 0) + winding label
3. Morph w=0 → w=1 (single helix twist appears)
4. Morph w=1 → w=2
5. Morph w=2 → w=3
6. Camera orbit on final w=3
7. Summary card

Run
---
    manim -pql examples/bloch_winding_transition.py WindingTransition
    manim -qh  examples/bloch_winding_transition.py WindingTransition
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Line3D,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    ReplacementTransform,
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
RIBBON_R = 0.9
RIBBON_HW = PI / 3.5  # half-width


def _ribbon_surface(n_winds, R=RIBBON_R, height=None, half_w=RIBBON_HW,
                     phase=0.0, color=ORANGE):
    """Create a ribbon Surface with n_winds twists.

    For n_winds=0 the ribbon is a flat annular ring.
    """
    if height is None:
        height = max(0.3, n_winds * 1.2)  # auto height
    if n_winds == 0:
        # flat ring
        def func(u, v):
            r = R + (2 * v - 1) * 0.25
            return np.array([r * np.cos(u), r * np.sin(u), 0.0])
        return Surface(
            func,
            u_range=[0, TAU], v_range=[0, 1],
            resolution=(48, 6),
            fill_color=color, fill_opacity=0.85,
            stroke_width=0.3, stroke_color=color,
        )
    pitch = height / n_winds
    z_base = -height / 2

    def func(u, v):
        angle = u + phase + (2 * v - 1) * half_w
        z = z_base + pitch * u / TAU
        return np.array([R * np.cos(angle), R * np.sin(angle), z])

    return Surface(
        func,
        u_range=[0, n_winds * TAU],
        v_range=[0, 1],
        resolution=(48 * max(n_winds, 1), 6),
        fill_color=color, fill_opacity=0.85,
        stroke_width=0.3, stroke_color=color,
    )


# Paired blue ribbon offset by π
def _pair(n_winds, **kw):
    a = _ribbon_surface(n_winds, color=ORANGE, phase=0.0, **kw)
    b = _ribbon_surface(n_winds, color=BLUE_D, phase=PI, **kw)
    return a, b


class WindingTransition(ThreeDScene):
    """Morph ribbons through winding numbers w = 0 → 1 → 2 → 3."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Winding Number Transition", font_size=44, color=GOLD)
        sub = Text("Topological phase diagram  w = 0, 1, 2, 3",
                    font_size=22, color=ORANGE)
        sub.next_to(ttl, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub))

        # ─── Act 2  Start at w = 0 ──────────────────────────────────────
        self.set_camera_orientation(phi=60 * DEG, theta=-45 * DEG)

        axis = Line3D(
            np.array([0, 0, -2.5]),
            np.array([0, 0,  2.5]),
            color=WHITE, thickness=0.003,
        )
        self.play(Create(axis), run_time=0.5)

        ra, rb = _pair(0)
        w_lbl = MathTex(r"w = 0", font_size=32, color=YELLOW)
        w_lbl.to_corner(UP + LEFT, buff=0.4)
        self.add_fixed_in_frame_mobjects(w_lbl)

        self.play(Create(ra), Create(rb), FadeIn(w_lbl), run_time=1.2)
        self.wait(0.5)

        # ─── Acts 3-5  Morph w → w+1 ────────────────────────────────────
        for w in [1, 2, 3]:
            new_a, new_b = _pair(w)
            new_lbl = MathTex(
                r"w = " + str(w), font_size=32, color=YELLOW,
            )
            new_lbl.to_corner(UP + LEFT, buff=0.4)
            self.add_fixed_in_frame_mobjects(new_lbl)

            desc = MathTex(
                r"\text{twist}" if w == 1
                else r"\text{twists}",
                font_size=18, color=GREY_A,
            )
            desc.to_edge(DOWN, buff=0.4)
            self.add_fixed_in_frame_mobjects(desc)

            self.play(
                ReplacementTransform(ra, new_a),
                ReplacementTransform(rb, new_b),
                ReplacementTransform(w_lbl, new_lbl),
                FadeIn(desc),
                run_time=1.8,
            )
            self.wait(0.4)
            self.play(FadeOut(desc), run_time=0.3)
            ra, rb, w_lbl = new_a, new_b, new_lbl

        # ─── Act 6  Camera orbit on w=3 ─────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # ─── Act 7  Summary ──────────────────────────────────────────────
        self.play(
            FadeOut(ra), FadeOut(rb), FadeOut(axis), FadeOut(w_lbl),
            run_time=0.8,
        )
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Winding Transition", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"w=0:\;\text{flat ring (trivial phase)}",
                font_size=19,
            ),
            MathTex(
                r"w=1,2,3:\;\text{each twist adds a topological charge}",
                font_size=19, color=ORANGE,
            ),
            MathTex(
                r"w\in\mathbb{Z}\;\text{cannot change by smooth deformation}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Transitions require gap closure or singularity}",
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
