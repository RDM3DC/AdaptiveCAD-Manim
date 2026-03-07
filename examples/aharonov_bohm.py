"""Aharonov-Bohm Effect — Phase Shift Without a Force.

Two electron wave packets split at a beam splitter, travel around a
solenoid (one left, one right), and recombine.  Although the magnetic
field B is zero outside the solenoid, the vector potential A is not,
and the enclosed flux produces a measurable phase shift:

    Delta phi = e Phi / hbar

The interference fringes shift as the flux is varied.

Uses the solver module ``solver.famous.aharonov_bohm`` for the
Phase-Lift winding calculation.

Acts
----
1. Title card
2. Setup — solenoid + two paths, introduce A-field
3. Wave-packet split and propagation around solenoid
4. Recombination — interference pattern with phase shift
5. Vary flux — fringes slide, solver verification
6. Key equation reveal and physical meaning
7. Summary card

Run
---
    manim -pql examples/aharonov_bohm.py AharonovBohm
    manim -qh  examples/aharonov_bohm.py AharonovBohm
"""

from __future__ import annotations

import sys, os, math
import numpy as np

# Ensure the repo root is on sys.path so `solver` can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solver.famous import aharonov_bohm as ab_solver

from manim import (
    Scene,
    Circle,
    Dot,
    DashedLine,
    Line,
    Arc,
    ArcBetweenPoints,
    VGroup,
    MathTex,
    Tex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Uncreate,
    Transform,
    Indicate,
    Flash,
    MoveAlongPath,
    SurroundingRectangle,
    Arrow,
    CurvedArrow,
    NumberLine,
    FunctionGraph,
    Axes,
    DecimalNumber,
    ValueTracker,
    always_redraw,
    DEGREES,
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
    GREY_B,
    PURPLE,
    config,
    interpolate_color,
)

# ═══════════════════════════════════════════════════════════════════════════
# Layout constants
# ═══════════════════════════════════════════════════════════════════════════
SOL_POS    = ORIGIN                    # solenoid centre
SOL_R      = 0.45                      # solenoid visual radius
PATH_R     = 1.8                       # path radius (semi-circles)
SRC_X      = -3.2                      # source position (left)
DET_X      =  3.2                      # detector position (right)
SPLIT_X    = SOL_POS[0] - PATH_R       # beam-splitter x
MERGE_X    = SOL_POS[0] + PATH_R       # merger x
FRINGE_Y   = -2.2                      # interference pattern y-centre
N_FRINGES  = 60                        # bars in fringe pattern

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _wave_dot(pos, color=BLUE):
    return Dot(pos, radius=0.12, color=color).set_opacity(0.85)


def _make_fringes(phase_shift: float, n: int = N_FRINGES):
    """Return a VGroup of thin rectangles simulating an interference pattern
    whose peak positions shift by ``phase_shift`` in [0, 2pi]."""
    from manim import Rectangle
    bars = VGroup()
    for i in range(n):
        x_pos = -3.0 + 6.0 * i / n
        # intensity ~ cos^2(k*x + phase_shift/2)
        intensity = math.cos(PI * i / 3 + phase_shift / 2) ** 2
        col = interpolate_color(BLUE_E, WHITE, intensity)
        bar = Rectangle(
            width=6.0 / n * 0.9,
            height=0.45,
            fill_color=col,
            fill_opacity=0.85,
            stroke_width=0,
        ).move_to(np.array([x_pos, FRINGE_Y, 0]))
        bars.add(bar)
    return bars


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class AharonovBohm(Scene):
    """Two electron beams split around a solenoid and interfere."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("The Aharonov-Bohm Effect", font_size=48, color=GOLD)
        sub = Text("Phase shift without a force",
                    font_size=24, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        desc = Text(
            "Electrons encircle a solenoid where B = 0\n"
            "yet acquire a measurable phase shift from A.",
            font_size=18, color=WHITE, line_spacing=1.3,
        )
        desc.next_to(sub, DOWN, buff=0.35)

        self.play(Write(ttl), run_time=1.5)                              # 1
        self.play(FadeIn(sub), run_time=0.8)                             # 2
        self.play(FadeIn(desc), run_time=0.8)                            # 3
        self.wait(1)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(desc))             # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Setup: solenoid + paths + A-field
        # ─────────────────────────────────────────────────────────────────
        # Solenoid (cross-section circle)
        sol_outer = Circle(
            radius=SOL_R, color=YELLOW, stroke_width=3,
        ).move_to(SOL_POS)
        sol_fill = Circle(
            radius=SOL_R, color=YELLOW, fill_opacity=0.15,
            stroke_width=0,
        ).move_to(SOL_POS)
        sol_lbl = MathTex(r"\Phi", font_size=24, color=YELLOW)
        sol_lbl.move_to(SOL_POS)
        b_lbl = MathTex(
            r"\vec{B} \neq 0 \text{ inside}",
            font_size=16, color=YELLOW,
        )
        b_lbl.next_to(sol_outer, DOWN, buff=0.15)

        self.play(
            Create(sol_outer), FadeIn(sol_fill), Write(sol_lbl),
            run_time=1,
        )                                                                 # 5
        self.play(FadeIn(b_lbl), run_time=0.5)                           # 6

        # B = 0 outside label
        b_out_lbl = MathTex(
            r"\vec{B} = 0 \text{ outside}",
            font_size=16, color=GREY_A,
        )
        b_out_lbl.next_to(sol_outer, UP + RIGHT, buff=0.3)
        self.play(FadeIn(b_out_lbl), run_time=0.5)                       # 7

        # A-field circulation arrows (tangential around solenoid)
        a_arrows = VGroup()
        for ang_deg in [30, 120, 210, 300]:
            ang = ang_deg * DEGREES
            r_a = SOL_R + 0.55
            start = SOL_POS + r_a * np.array([np.cos(ang), np.sin(ang), 0])
            end_ang = ang + 40 * DEGREES
            end = SOL_POS + r_a * np.array([np.cos(end_ang), np.sin(end_ang), 0])
            arr = CurvedArrow(
                start, end,
                color=TEAL, stroke_width=2,
                tip_length=0.12,
            )
            a_arrows.add(arr)
        a_lbl = MathTex(r"\vec{A}", font_size=20, color=TEAL)
        a_lbl.next_to(sol_outer, LEFT, buff=0.7)
        self.play(
            *[Create(a) for a in a_arrows],
            FadeIn(a_lbl),
            run_time=1,
        )                                                                 # 8

        # Two semi-circular paths
        left_pt = np.array([SPLIT_X, 0, 0])
        right_pt = np.array([MERGE_X, 0, 0])

        path_upper = ArcBetweenPoints(
            left_pt, right_pt, angle=-PI, color=BLUE, stroke_width=2,
        )
        path_lower = ArcBetweenPoints(
            left_pt, right_pt, angle=PI, color=RED, stroke_width=2,
        )

        path1_lbl = MathTex(r"\text{Path 1}", font_size=16, color=BLUE)
        path1_lbl.next_to(
            SOL_POS + np.array([0, PATH_R + 0.1, 0]), UP, buff=0.05,
        )
        path2_lbl = MathTex(r"\text{Path 2}", font_size=16, color=RED)
        path2_lbl.next_to(
            SOL_POS + np.array([0, -PATH_R - 0.1, 0]), DOWN, buff=0.05,
        )

        self.play(
            Create(path_upper), Create(path_lower),
            FadeIn(path1_lbl), FadeIn(path2_lbl),
            run_time=1.2,
        )                                                                 # 9

        # Source and detector
        src_dot = Dot(np.array([SRC_X, 0, 0]), radius=0.1, color=GREEN)
        src_lbl = MathTex(r"\text{source}", font_size=14, color=GREEN)
        src_lbl.next_to(src_dot, DOWN, buff=0.1)
        det_dot = Dot(np.array([DET_X, 0, 0]), radius=0.1, color=ORANGE)
        det_lbl = MathTex(r"\text{detector}", font_size=14, color=ORANGE)
        det_lbl.next_to(det_dot, DOWN, buff=0.1)

        # Lines connecting source/detector to split/merge
        src_line = Line(
            np.array([SRC_X, 0, 0]), left_pt,
            color=GREY_D, stroke_width=1.5,
        )
        det_line = Line(
            right_pt, np.array([DET_X, 0, 0]),
            color=GREY_D, stroke_width=1.5,
        )

        self.play(
            FadeIn(src_dot), FadeIn(src_lbl),
            FadeIn(det_dot), FadeIn(det_lbl),
            Create(src_line), Create(det_line),
            run_time=0.8,
        )                                                                 # 10

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Wave-packet propagation
        # ─────────────────────────────────────────────────────────────────
        wave1 = _wave_dot(np.array([SRC_X, 0, 0]), BLUE)
        wave2 = _wave_dot(np.array([SRC_X, 0, 0]), RED)

        self.play(FadeIn(wave1), FadeIn(wave2), run_time=0.4)            # 11

        # Move to split point
        self.play(
            wave1.animate.move_to(left_pt),
            wave2.animate.move_to(left_pt),
            run_time=0.6,
        )                                                                 # 12

        # Propagate along paths simultaneously
        self.play(
            MoveAlongPath(wave1, path_upper),
            MoveAlongPath(wave2, path_lower),
            run_time=2,
        )                                                                 # 13

        # Phase labels at merge point
        phase1_lbl = MathTex(
            r"\phi_1 = \int_1 \vec{A}\cdot d\vec{\ell}",
            font_size=16, color=BLUE,
        )
        phase1_lbl.next_to(right_pt, UP + RIGHT, buff=0.15)
        phase2_lbl = MathTex(
            r"\phi_2 = \int_2 \vec{A}\cdot d\vec{\ell}",
            font_size=16, color=RED,
        )
        phase2_lbl.next_to(right_pt, DOWN + RIGHT, buff=0.15)

        self.play(
            FadeIn(phase1_lbl), FadeIn(phase2_lbl),
            run_time=0.8,
        )                                                                 # 14

        delta_phi_eq = MathTex(
            r"\Delta\phi = \phi_1 - \phi_2"
            r" = \oint \vec{A}\cdot d\vec{\ell}"
            r" = \frac{e\Phi}{\hbar}",
            font_size=20, color=GOLD,
        )
        delta_phi_eq.to_edge(UP, buff=0.2)
        self.play(Write(delta_phi_eq), run_time=1.2)                     # 15

        self.play(Flash(right_pt, color=YELLOW), run_time=0.5)           # 16

        # Move to detector
        self.play(
            wave1.animate.move_to(np.array([DET_X, 0, 0])),
            wave2.animate.move_to(np.array([DET_X, 0, 0])),
            run_time=0.6,
        )                                                                 # 17

        self.play(FadeOut(wave1), FadeOut(wave2), run_time=0.3)          # 18

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Interference pattern
        # ─────────────────────────────────────────────────────────────────
        # Clear phase labels
        self.play(
            FadeOut(phase1_lbl), FadeOut(phase2_lbl),
            run_time=0.4,
        )                                                                 # 19

        fringe_title = MathTex(
            r"\text{Interference pattern at detector}",
            font_size=18, color=ORANGE,
        )
        fringe_title.move_to(np.array([0, FRINGE_Y + 0.55, 0]))
        self.play(FadeIn(fringe_title), run_time=0.4)                    # 20

        # Phase shift = 0 (no flux)
        fringes0 = _make_fringes(0)
        self.play(FadeIn(fringes0), run_time=0.8)                        # 21

        phi_label = MathTex(
            r"\Phi = 0 \;\;\Rightarrow\;\; \Delta\phi = 0",
            font_size=18, color=WHITE,
        )
        phi_label.move_to(np.array([0, FRINGE_Y - 0.55, 0]))
        self.play(FadeIn(phi_label), run_time=0.5)                       # 22

        self.wait(0.5)

        # Phase shift = pi  (half flux quantum)
        fringes_pi = _make_fringes(PI)
        phi_label_pi = MathTex(
            r"\Phi = \Phi_0/2 \;\;\Rightarrow\;\; \Delta\phi = \pi",
            font_size=18, color=YELLOW,
        )
        phi_label_pi.move_to(phi_label.get_center())

        self.play(
            Transform(fringes0, fringes_pi),
            Transform(phi_label, phi_label_pi),
            Indicate(sol_lbl, color=RED),
            run_time=1,
        )                                                                 # 23

        self.wait(0.5)

        # Phase shift = 2pi  (one flux quantum)
        fringes_2pi = _make_fringes(TAU)
        phi_label_2pi = MathTex(
            r"\Phi = \Phi_0 \;\;\Rightarrow\;\; \Delta\phi = 2\pi",
            font_size=18, color=TEAL,
        )
        phi_label_2pi.move_to(phi_label.get_center())

        self.play(
            Transform(fringes0, fringes_2pi),
            Transform(phi_label, phi_label_2pi),
            run_time=1,
        )                                                                 # 24

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Sweep flux continuously + solver verification
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(fringes0), FadeOut(phi_label),
            FadeOut(fringe_title),
            run_time=0.4,
        )                                                                 # 25

        sweep_title = Text(
            "Sweeping enclosed flux", font_size=20, color=GOLD,
        )
        sweep_title.move_to(np.array([0, FRINGE_Y + 0.55, 0]))
        self.play(FadeIn(sweep_title), run_time=0.4)                     # 26

        flux_tracker = ValueTracker(0)

        flux_readout = always_redraw(lambda: MathTex(
            r"\Phi / \Phi_0 = "
            + f"{flux_tracker.get_value():.2f}",
            font_size=20, color=YELLOW,
        ).move_to(np.array([-3.5, FRINGE_Y - 0.55, 0])))

        phase_readout = always_redraw(lambda: MathTex(
            r"\Delta\phi = "
            + f"{flux_tracker.get_value() * TAU:.2f}",
            font_size=20, color=TEAL,
        ).move_to(np.array([3.5, FRINGE_Y - 0.55, 0])))

        live_fringes = always_redraw(
            lambda: _make_fringes(flux_tracker.get_value() * TAU)
        )

        self.play(
            FadeIn(flux_readout),
            FadeIn(phase_readout),
            FadeIn(live_fringes),
            run_time=0.5,
        )                                                                 # 27

        # Sweep 0 → 2 flux quanta
        self.play(
            flux_tracker.animate.set_value(2.0),
            run_time=5,
            rate_func=lambda t: t,  # linear
        )                                                                 # 28

        self.wait(0.3)

        # ── Solver verification ──
        solver_result = ab_solver(flux_quanta=1.0)
        solver_box_text = VGroup(
            Text("Solver verification", font_size=16, color=GOLD),
            MathTex(
                r"\text{Phase-Lift winding: }"
                + f"w = {solver_result['winding_w']}",
                font_size=16, color=WHITE,
            ),
            MathTex(
                r"\theta_R = "
                + f"{solver_result['theta_R_final']:.4f}"
                + r"\;\; \text{expected } "
                + f"{solver_result['expected_phase']:.4f}",
                font_size=16, color=WHITE,
            ),
            MathTex(
                r"\text{pass: }" + str(solver_result["pass"]),
                font_size=16,
                color=GREEN if solver_result["pass"] else RED,
            ),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        solver_box_text.to_corner(DOWN + LEFT, buff=0.25)
        solver_box = SurroundingRectangle(
            solver_box_text, color=GOLD, buff=0.12, corner_radius=0.08,
        )
        self.play(FadeIn(solver_box), FadeIn(solver_box_text),
                  run_time=0.8)                                           # 29

        self.wait(1)

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Key equation reveal
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(live_fringes), FadeOut(flux_readout),
            FadeOut(phase_readout), FadeOut(sweep_title),
            FadeOut(solver_box), FadeOut(solver_box_text),
            run_time=0.6,
        )                                                                 # 30

        key_eq_title = Text(
            "The Key Insight", font_size=24, color=GOLD,
        )
        key_eq_title.move_to(np.array([0, FRINGE_Y + 0.6, 0]))
        self.play(FadeIn(key_eq_title), run_time=0.5)                    # 31

        insights = VGroup(
            MathTex(
                r"\text{Classically: no } \vec{B} "
                r"\;\Rightarrow\; \text{no force on electron}",
                font_size=18, color=WHITE,
            ),
            MathTex(
                r"\text{Quantum: } \vec{A} \neq 0 "
                r"\;\Rightarrow\; "
                r"\text{phase shift is real and measurable}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\Delta\phi = \frac{e}{\hbar}"
                r"\oint \vec{A}\cdot d\vec{\ell}"
                r" = \frac{e\Phi}{\hbar}",
                font_size=22, color=YELLOW,
            ),
            MathTex(
                r"\text{The vector potential } \vec{A} "
                r"\text{ is physically real, not just a math trick}",
                font_size=18, color=ORANGE,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        insights.move_to(np.array([0, FRINGE_Y - 0.15, 0]))

        for line in insights:
            self.play(FadeIn(line), run_time=0.7)                         # 32-35

        self.play(
            Indicate(delta_phi_eq, color=RED, scale_factor=1.1),
            run_time=0.8,
        )                                                                 # 36

        # Stokes' theorem link
        stokes = MathTex(
            r"\oint \vec{A}\cdot d\vec{\ell}"
            r" = \int_S (\nabla\times\vec{A})\cdot d\vec{S}"
            r" = \int_S \vec{B}\cdot d\vec{S} = \Phi",
            font_size=18, color=GREY_A,
        )
        stokes.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(stokes), run_time=0.8)                          # 37

        self.wait(1.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1,
        )                                                                 # 38

        card_title = Text(
            "Aharonov-Bohm Effect", font_size=34, color=GOLD,
        )
        card_title.to_edge(UP, buff=0.5)
        self.play(Write(card_title))                                      # 39

        bullets = VGroup(
            MathTex(
                r"\Delta\phi = \frac{e\Phi}{\hbar}"
                r"\;\;\text{(phase shift from enclosed flux)}",
                font_size=18,
            ),
            MathTex(
                r"\vec{B} = 0 \text{ outside solenoid, yet}"
                r"\;\vec{A} \neq 0 \text{ produces real effects}",
                font_size=18,
            ),
            MathTex(
                r"\text{Interference fringes shift with } \Phi "
                r"\text{ — confirmed experimentally}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{Topological effect: } \pi_1(U(1)) = \mathbb{Z}"
                r"\;\;\text{(winding number of gauge)}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\text{Solver Phase-Lift: winding } w = "
                + f"{solver_result['winding_w']}"
                + r"\text{, } \theta_R = "
                + f"{solver_result['theta_R_final']:.4f}",
                font_size=18, color=GREEN,
            ),
            MathTex(
                r"\text{The potential is real — "
                r"geometry + topology govern physics}",
                font_size=18, color=YELLOW,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, bullets),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.play(FadeIn(box), run_time=0.5)                             # 40
        for b in bullets:
            self.play(FadeIn(b), run_time=0.6)                           # 41-46

        self.wait(2)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )                                                                 # 47
