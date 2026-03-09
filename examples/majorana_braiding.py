"""Majorana Zero-Mode Braiding on a Nanowire.

Two Majorana zero modes γ₁, γ₂ sit at the ends of a topological
nanowire.  Braiding them swaps their position and picks up a π-phase
(non-Abelian exchange statistics).

Panel layout
------------
  Top:   1-D nanowire with Majorana dots at both ends
  Below: Braid worldlines in (x, t) spacetime showing the exchange
  Right: Phase dial accumulating π per braid

Acts
----
1. Title card
2. Draw topological nanowire with γ₁, γ₂ dots
3. First braid — γ₁ ↔ γ₂ swap, worldlines cross, phase jumps to π
4. Second braid — γ₂ ↔ γ₁ swap, worldlines cross again, phase → 2π
5. Show fusion channel label
6. Summary card

Run
---
    manim -pql examples/majorana_braiding.py MajoranaBraiding
    manim -qh  examples/majorana_braiding.py MajoranaBraiding
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
    Transform,
    ReplacementTransform,
    MoveAlongPath,
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
    interpolate_color,
    config,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Geometry constants
# ═══════════════════════════════════════════════════════════════════════════
WIRE_LEN = 5.0           # nanowire display length
WIRE_Y   = 1.8           # y-position of the wire
BRAID_Y0 = 0.8           # top of braid diagram
BRAID_Y1 = -2.0          # bottom of braid diagram
GAP_X    = 1.6           # horizontal separation of γ modes
WIRE_X0  = -GAP_X / 2    # left end
WIRE_X1  =  GAP_X / 2    # right end


def _nanowire_ribbon(y, half_len, half_h=0.12, z=0.0):
    """Parametric flat ribbon representing superconducting nanowire."""

    def func(u, v):
        x = -half_len + u * 2 * half_len
        yy = y + (2 * v - 1) * half_h
        return np.array([x, yy, z])

    return Surface(
        func, u_range=[0, 1], v_range=[0, 1],
        resolution=(2, 2),
        fill_color=GREY_D, fill_opacity=0.8,
        stroke_width=0.5, stroke_color=GREY_A,
    )


def _braid_arc(x_start, x_end, y_top, y_bot, over=True):
    """Semicircular worldline arc from (x_start, y_top) to (x_end, y_bot).
    'over' controls which strand passes in front (visual z-order)."""
    cx = (x_start + x_end) / 2
    cy = (y_top + y_bot) / 2
    rx = abs(x_end - x_start) / 2
    ry = (y_top - y_bot) / 2
    sign = 1 if x_end > x_start else -1

    def func(t):
        return np.array([
            cx + sign * rx * np.cos(t),
            cy + ry * np.sin(t),
            0.0,
        ])

    return ParametricFunction(
        func, t_range=[PI, 0] if sign > 0 else [0, PI],
        stroke_width=3.5 if over else 2.5,
    )


class MajoranaBraiding(ThreeDScene):
    """Majorana zero-mode braiding with π-phase jumps."""

    def construct(self):
        self.camera.background_color = "#080818"
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # ─── Act 1  Title ────────────────────────────────────────────────
        ttl = Text("Majorana Zero-Mode Braiding",
                    font_size=44, color=GOLD)
        sub = Text("Non-Abelian exchange on a nanowire",
                    font_size=22, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"\gamma_1 \leftrightarrow \gamma_2"
            r"\;\Rightarrow\; e^{i\pi/2}\,"
            r"\text{(non-Abelian)}",
            font_size=26, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Nanowire + Majorana endpoints ───────────────────────
        wire = _nanowire_ribbon(WIRE_Y, WIRE_LEN / 2)

        # Superconducting gap shading
        gap_ribbon = _nanowire_ribbon(WIRE_Y, WIRE_LEN / 2 - 0.5,
                                       half_h=0.08)
        gap_ribbon.set_color(BLUE_E)
        gap_ribbon.set_opacity(0.35)

        g1_dot = Dot3D(np.array([WIRE_X0, WIRE_Y, 0.0]),
                       radius=0.12, color=ORANGE)
        g2_dot = Dot3D(np.array([WIRE_X1, WIRE_Y, 0.0]),
                       radius=0.12, color=TEAL)

        lbl_g1 = MathTex(r"\gamma_1", font_size=20, color=ORANGE)
        lbl_g1.move_to(np.array([WIRE_X0, WIRE_Y + 0.4, 0]))
        lbl_g2 = MathTex(r"\gamma_2", font_size=20, color=TEAL)
        lbl_g2.move_to(np.array([WIRE_X1, WIRE_Y + 0.4, 0]))

        wire_lbl = Text("Topological nanowire", font_size=14,
                         color=GREY_A)
        wire_lbl.move_to(np.array([0, WIRE_Y - 0.4, 0]))

        frame_lbls = [lbl_g1, lbl_g2, wire_lbl]
        for lbl in frame_lbls:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            Create(wire), FadeIn(gap_ribbon),
            FadeIn(g1_dot), FadeIn(g2_dot),
            FadeIn(lbl_g1), FadeIn(lbl_g2), FadeIn(wire_lbl),
            run_time=1.2,
        )

        # Braid diagram axes
        t_ax = Arrow3D(
            np.array([-2.5, BRAID_Y0 + 0.2, 0]),
            np.array([-2.5, BRAID_Y1 - 0.2, 0]),
            color=GREY_A, thickness=0.008,
        )
        t_lbl = MathTex(r"t", font_size=18, color=GREY_A)
        t_lbl.move_to(np.array([-2.8, (BRAID_Y0 + BRAID_Y1) / 2, 0]))
        self.add_fixed_in_frame_mobjects(t_lbl)
        self.play(Create(t_ax), FadeIn(t_lbl), run_time=0.5)

        # Vertical initial worldlines (before braid)
        wl1_init = Line3D(
            np.array([WIRE_X0, BRAID_Y0, 0]),
            np.array([WIRE_X0, BRAID_Y0 - 0.3, 0]),
            color=ORANGE, thickness=0.015,
        )
        wl2_init = Line3D(
            np.array([WIRE_X1, BRAID_Y0, 0]),
            np.array([WIRE_X1, BRAID_Y0 - 0.3, 0]),
            color=TEAL, thickness=0.015,
        )
        self.play(Create(wl1_init), Create(wl2_init), run_time=0.5)

        # Phase label on right
        phase_lbl = MathTex(r"\phi = 0", font_size=24, color=YELLOW)
        phase_lbl.move_to(RIGHT * 4.0 + UP * 0.5)
        self.add_fixed_in_frame_mobjects(phase_lbl)
        self.play(FadeIn(phase_lbl), run_time=0.3)

        # ─── Act 3  First braid ─────────────────────────────────────────
        mid_y = (BRAID_Y0 - 0.3 + (BRAID_Y0 + BRAID_Y1) / 2) / 2
        arc1_over = _braid_arc(WIRE_X0, WIRE_X1,
                                BRAID_Y0 - 0.3,
                                (BRAID_Y0 + BRAID_Y1) / 2 + 0.2,
                                over=True)
        arc1_over.set_color(ORANGE)
        arc1_under = _braid_arc(WIRE_X1, WIRE_X0,
                                 BRAID_Y0 - 0.3,
                                 (BRAID_Y0 + BRAID_Y1) / 2 + 0.2,
                                 over=False)
        arc1_under.set_color(TEAL)
        arc1_under.set_stroke(opacity=0.5)

        braid_lbl1 = MathTex(
            r"\gamma_1 \leftrightarrow \gamma_2",
            font_size=16, color=YELLOW,
        )
        braid_lbl1.move_to(RIGHT * 2.5 +
                           UP * ((BRAID_Y0 + BRAID_Y1) / 2 + 0.5))
        self.add_fixed_in_frame_mobjects(braid_lbl1)

        # Swap dots on nanowire simultaneously
        g1_target = np.array([WIRE_X1, WIRE_Y, 0.0])
        g2_target = np.array([WIRE_X0, WIRE_Y, 0.0])

        new_phase = MathTex(r"\phi = \pi", font_size=24, color=YELLOW)
        new_phase.move_to(RIGHT * 4.0 + UP * 0.5)
        self.add_fixed_in_frame_mobjects(new_phase)

        self.play(
            Create(arc1_over), Create(arc1_under),
            g1_dot.animate.move_to(g1_target),
            g2_dot.animate.move_to(g2_target),
            FadeIn(braid_lbl1),
            run_time=1.5,
        )
        self.play(
            ReplacementTransform(phase_lbl, new_phase),
            run_time=0.5,
        )
        phase_lbl = new_phase

        # short vertical segments after first braid
        mid_y2 = (BRAID_Y0 + BRAID_Y1) / 2 + 0.2
        wl1_mid = Line3D(
            np.array([WIRE_X1, mid_y2, 0]),
            np.array([WIRE_X1, mid_y2 - 0.3, 0]),
            color=ORANGE, thickness=0.015,
        )
        wl2_mid = Line3D(
            np.array([WIRE_X0, mid_y2, 0]),
            np.array([WIRE_X0, mid_y2 - 0.3, 0]),
            color=TEAL, thickness=0.015,
        )
        self.play(Create(wl1_mid), Create(wl2_mid), run_time=0.3)
        self.wait(0.3)

        # ─── Act 4  Second braid ────────────────────────────────────────
        arc2_over = _braid_arc(WIRE_X1, WIRE_X0,
                                mid_y2 - 0.3,
                                BRAID_Y1 + 0.15,
                                over=True)
        arc2_over.set_color(ORANGE)
        arc2_under = _braid_arc(WIRE_X0, WIRE_X1,
                                 mid_y2 - 0.3,
                                 BRAID_Y1 + 0.15,
                                 over=False)
        arc2_under.set_color(TEAL)
        arc2_under.set_stroke(opacity=0.5)

        # Swap back
        new_phase2 = MathTex(r"\phi = 2\pi", font_size=24, color=YELLOW)
        new_phase2.move_to(RIGHT * 4.0 + UP * 0.5)
        self.add_fixed_in_frame_mobjects(new_phase2)

        self.play(
            Create(arc2_over), Create(arc2_under),
            g1_dot.animate.move_to(np.array([WIRE_X0, WIRE_Y, 0.0])),
            g2_dot.animate.move_to(np.array([WIRE_X1, WIRE_Y, 0.0])),
            run_time=1.5,
        )
        self.play(
            ReplacementTransform(phase_lbl, new_phase2),
            run_time=0.5,
        )

        # Final vertical tails
        wl1_end = Line3D(
            np.array([WIRE_X0, BRAID_Y1 + 0.15, 0]),
            np.array([WIRE_X0, BRAID_Y1, 0]),
            color=ORANGE, thickness=0.015,
        )
        wl2_end = Line3D(
            np.array([WIRE_X1, BRAID_Y1 + 0.15, 0]),
            np.array([WIRE_X1, BRAID_Y1, 0]),
            color=TEAL, thickness=0.015,
        )
        self.play(Create(wl1_end), Create(wl2_end), run_time=0.3)

        # ─── Act 5  Fusion channel ──────────────────────────────────────
        fusion = MathTex(
            r"\gamma_1 \gamma_2 = e^{i\pi}\,"
            r"|\text{vacuum}\rangle",
            font_size=20, color=TEAL,
        )
        fusion.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(fusion)
        self.play(FadeIn(fusion), run_time=0.6)
        self.wait(1)

        # ─── Act 6  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in [
                lbl_g1, lbl_g2, wire_lbl, t_lbl, new_phase2,
                braid_lbl1, fusion]],
            run_time=0.8,
        )

        card = Text("Majorana Braiding", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\gamma_{1,2}\;\text{are their own antiparticles}",
                font_size=19,
            ),
            MathTex(
                r"\text{Exchange } \gamma_1 \leftrightarrow \gamma_2"
                r"\;\Rightarrow\;\pi\text{-phase jump}",
                font_size=19, color=ORANGE,
            ),
            MathTex(
                r"\text{Non-Abelian statistics: braid order matters}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Foundation for topological quantum gates}",
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
