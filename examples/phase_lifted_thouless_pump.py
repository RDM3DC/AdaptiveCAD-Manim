"""Phase-Lifted Thouless Pump Memory Law — visual proof.

    Q_cycle = e ( C + Δθ_R / (2π_a) )

The transported charge per cycle contains the usual topological
contribution eC plus a branch-history correction from the lifted
phase increment Δθ_R.  The law proposes that adiabatic pumping is
quantized on the topological sector but measurably shifted by
history-resolved phase memory when the transport cycle is tracked
on a lifted cover.

Acts
----
  1. Title card with equation
  2. Build Bloch torus with pump cycle path
  3. Show standard Thouless pump (C=1, Δθ_R=0)
  4. Introduce lifted phase: show branch cuts vs lifted cover
  5. Animate pumped charge with memory correction bar
  6. Side-by-side: principal-branch vs lifted-memory cycle counts
  7. Summary card

Run
---
    manim -pql examples/phase_lifted_thouless_pump.py PhaseLiftedThoulessPump
    manim -qh  examples/phase_lifted_thouless_pump.py PhaseLiftedThoulessPump
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
    Uncreate,
    Transform,
    ReplacementTransform,
    SurroundingRectangle,
    Indicate,
    Rotate,
    LaggedStart,
    Rectangle,
    Line,
    Arrow,
    DecimalNumber,
    Axes,
    DashedLine,
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
    GREY,
    GREY_A,
    GREY_D,
    PURPLE,
    PINK,
    interpolate_color,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Geometry constants
# ═══════════════════════════════════════════════════════════════════════════
TORUS_R = 1.4
TORUS_r = 0.5
CX = -1.5
_RAIN = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]


def _torus_func(R, r, cx=0.0):
    def func(u, v):
        return np.array([
            cx + (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v),
        ])
    return func


def _torus_curve(R, r, n_wind, cx=0.0):
    def func(t):
        u = t
        v = n_wind * t
        return np.array([
            cx + (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v),
        ])
    return func


def _rainbow(frac):
    n = len(_RAIN) - 1
    idx = frac * n
    lo = min(int(idx), n - 1)
    return interpolate_color(_RAIN[lo], _RAIN[lo + 1], idx - lo)


class PhaseLiftedThoulessPump(ThreeDScene):
    """Phase-lifted Thouless pump with memory correction."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ═════════════════════════════════════════════════════════════
        # ACT 1 — Title + Equation
        # ═════════════════════════════════════════════════════════════
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Phase-Lifted Thouless Pump", font_size=44, color=GOLD)
        sub = Text("Memory Law", font_size=24, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.2)

        eq = MathTex(
            r"Q_{\mathrm{cycle}}=e\left(",
            r"C",
            r"+",
            r"\frac{\Delta\theta_R}{2\pi_a}",
            r"\right)",
            font_size=32, color=WHITE,
        )
        eq.next_to(sub, DOWN, buff=0.3)
        eq[1].set_color(BLUE)
        eq[3].set_color(PURPLE)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1.2)

        # Annotate terms
        topo_lbl = Text("Topological sector", font_size=14,
                         color=BLUE)
        topo_lbl.next_to(eq[1], DOWN, buff=0.15)
        mem_lbl = Text("Phase memory correction", font_size=14,
                        color=PURPLE)
        mem_lbl.next_to(eq[3], DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(topo_lbl, mem_lbl)
        self.play(
            Indicate(eq[1], color=BLUE),
            FadeIn(topo_lbl),
            run_time=0.6,
        )
        self.play(
            Indicate(eq[3], color=PURPLE),
            FadeIn(mem_lbl),
            run_time=0.6,
        )
        self.wait(0.8)
        self.play(
            FadeOut(ttl), FadeOut(sub), FadeOut(topo_lbl), FadeOut(mem_lbl),
        )
        # Keep equation — move to top
        eq.generate_target()
        eq.target.scale(0.8).to_edge(UP, buff=0.15)
        self.play(Transform(eq, eq.target), run_time=0.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 2 — Draw Bloch torus
        # ═════════════════════════════════════════════════════════════
        self.set_camera_orientation(phi=65 * DEG, theta=-50 * DEG)

        torus = Surface(
            _torus_func(TORUS_R, TORUS_r, cx=CX),
            u_range=[0, TAU], v_range=[0, TAU],
            resolution=(48, 24),
            fill_color=BLUE_E, fill_opacity=0.10,
            stroke_width=0.3, stroke_color=BLUE_D,
        )

        lbl_k = MathTex(r"k", font_size=18, color=RED)
        lbl_k.move_to(LEFT * 4.0 + DOWN * 0.3)
        lbl_t = MathTex(r"t", font_size=18, color=GREEN)
        lbl_t.move_to(LEFT * 0.5 + UP * 1.5)
        for lbl in [lbl_k, lbl_t]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(Create(torus), FadeIn(lbl_k), FadeIn(lbl_t),
                  run_time=1.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 3 — Standard pump: C=1, Δθ_R=0
        # ═════════════════════════════════════════════════════════════
        # w=1 winding curve (standard Thouless pump)
        N_SEG = 80
        cf1 = _torus_curve(TORUS_R, TORUS_r, 1, cx=CX)
        ts = np.linspace(0, TAU, N_SEG + 1)
        curve1 = VGroup()
        for i in range(N_SEG):
            seg = Line3D(cf1(ts[i]), cf1(ts[i + 1]),
                         color=_rainbow(i / N_SEG), thickness=0.016)
            curve1.add(seg)

        w_lbl = MathTex(r"C = 1", font_size=24, color=BLUE)
        w_lbl.move_to(RIGHT * 3.5 + UP * 1.5)
        self.add_fixed_in_frame_mobjects(w_lbl)

        self.play(Create(curve1), FadeIn(w_lbl), run_time=2)

        # Charge counter at right
        charge_box_pos = RIGHT * 3.5 + DOWN * 0.5
        q_lbl = MathTex(r"Q/e = ", font_size=22, color=YELLOW)
        q_lbl.move_to(charge_box_pos + LEFT * 0.5)

        q_val = DecimalNumber(1.0, num_decimal_places=3,
                              font_size=24, color=GOLD)
        q_val.next_to(q_lbl, RIGHT, buff=0.1)
        self.add_fixed_in_frame_mobjects(q_lbl, q_val)
        self.play(FadeIn(q_lbl), FadeIn(q_val), run_time=0.5)

        std_note = MathTex(
            r"\Delta\theta_R = 0 \implies Q = eC",
            font_size=18, color=GREY_A,
        )
        std_note.move_to(charge_box_pos + DOWN * 0.6)
        self.add_fixed_in_frame_mobjects(std_note)
        self.play(FadeIn(std_note), run_time=0.5)
        self.wait(1)

        # ═════════════════════════════════════════════════════════════
        # ACT 4 — Lifted phase: branch cut visualization
        # ═════════════════════════════════════════════════════════════
        self.play(FadeOut(curve1), FadeOut(w_lbl), FadeOut(std_note),
                  run_time=0.4)

        # Show a more complex winding with phase memory
        # w=1 but with accumulated phase shift (visualized as slight offset)
        def _lifted_curve(t):
            u = t
            v = t + 0.3 * np.sin(2 * t)  # lifted phase offset
            return np.array([
                CX + (TORUS_R + TORUS_r * np.cos(v)) * np.cos(u),
                (TORUS_R + TORUS_r * np.cos(v)) * np.sin(u),
                TORUS_r * np.sin(v),
            ])

        lifted_pts = np.linspace(0, TAU, N_SEG + 1)
        curve_lifted = VGroup()
        for i in range(N_SEG):
            seg = Line3D(
                _lifted_curve(lifted_pts[i]),
                _lifted_curve(lifted_pts[i + 1]),
                color=interpolate_color(PURPLE, PINK, i / N_SEG),
                thickness=0.018,
            )
            curve_lifted.add(seg)

        lifted_lbl = MathTex(
            r"\theta_R \neq \theta_{\mathrm{principal}}",
            font_size=20, color=PURPLE,
        )
        lifted_lbl.move_to(RIGHT * 3.2 + UP * 1.5)
        self.add_fixed_in_frame_mobjects(lifted_lbl)

        self.play(Create(curve_lifted), FadeIn(lifted_lbl), run_time=2)

        # Update charge to show memory correction
        delta_theta = 0.3 * 2  # net accumulated phase offset over cycle
        pi_a = np.pi
        memory_shift = delta_theta / (2 * pi_a)
        q_new = 1.0 + memory_shift

        # Animate charge changing
        self.play(
            q_val.animate.set_value(q_new),
            run_time=1.5,
        )

        mem_note = MathTex(
            r"\Delta\theta_R \neq 0 \implies Q = e\!\left(C + "
            r"\frac{\Delta\theta_R}{2\pi_a}\right)",
            font_size=18, color=PURPLE,
        )
        mem_note.move_to(charge_box_pos + DOWN * 0.6)
        self.add_fixed_in_frame_mobjects(mem_note)
        self.play(FadeIn(mem_note), run_time=0.5)
        self.wait(1)

        # ═════════════════════════════════════════════════════════════
        # ACT 5 — Charge bars: cycle-by-cycle comparison
        # ═════════════════════════════════════════════════════════════
        self.play(
            FadeOut(torus), FadeOut(curve_lifted), FadeOut(lifted_lbl),
            FadeOut(lbl_k), FadeOut(lbl_t), FadeOut(q_lbl), FadeOut(q_val),
            FadeOut(mem_note),
            run_time=0.5,
        )
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        bar_title = Text("Cycle-by-cycle pumped charge",
                         font_size=22, color=GOLD)
        bar_title.to_edge(UP, buff=0.6)
        self.add_fixed_in_frame_mobjects(bar_title)
        self.play(FadeOut(eq), Write(bar_title), run_time=0.5)

        n_cycles = 6
        np.random.seed(42)
        delta_thetas = np.random.normal(0, 0.15, n_cycles)
        delta_thetas[0] = 0  # first cycle pristine

        bar_width = 0.6
        gap = 0.15
        total_w = n_cycles * (bar_width + gap)
        start_x = -total_w / 2

        # Standard bars (C=1 for all)
        std_bars = VGroup()
        mem_bars = VGroup()
        for i in range(n_cycles):
            x = start_x + i * (bar_width + gap)
            # Standard bar (Q = e)
            std = Rectangle(
                width=bar_width, height=2.0,
                fill_color=BLUE, fill_opacity=0.5,
                stroke_color=BLUE, stroke_width=1.5,
            )
            std.move_to(np.array([x, -0.5, 0]))
            std_bars.add(std)

            # Memory bar (Q = e(1 + Δθ/2πa))
            mem_h = 2.0 * (1.0 + delta_thetas[i] / (2 * pi_a))
            mem = Rectangle(
                width=bar_width * 0.5, height=mem_h,
                fill_color=PURPLE, fill_opacity=0.7,
                stroke_color=PURPLE, stroke_width=1.5,
            )
            mem.move_to(np.array([x + bar_width * 0.25, -0.5 +
                                  (mem_h - 2.0) / 2, 0]))
            mem_bars.add(mem)

        # Reference line at Q=e
        ref_line = DashedLine(
            np.array([start_x - 0.3, 0.5, 0]),
            np.array([start_x + total_w, 0.5, 0]),
            color=GREY_A, stroke_width=1,
        )
        ref_lbl = MathTex(r"Q=e", font_size=16, color=GREY_A)
        ref_lbl.next_to(ref_line, RIGHT, buff=0.1)

        legend_std = VGroup(
            Rectangle(width=0.2, height=0.15, fill_color=BLUE,
                      fill_opacity=0.5, stroke_width=0),
            Text("Principal branch", font_size=12, color=BLUE),
        ).arrange(RIGHT, buff=0.08)
        legend_mem = VGroup(
            Rectangle(width=0.2, height=0.15, fill_color=PURPLE,
                      fill_opacity=0.7, stroke_width=0),
            Text("Lifted memory", font_size=12, color=PURPLE),
        ).arrange(RIGHT, buff=0.08)
        bar_legend = VGroup(legend_std, legend_mem).arrange(
            RIGHT, buff=0.4).to_edge(DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ref_lbl, *bar_legend)
        self.play(
            Create(ref_line), FadeIn(ref_lbl),
            LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in std_bars],
                        lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in mem_bars],
                        lag_ratio=0.1),
            FadeIn(bar_legend),
            run_time=1.5,
        )

        # Cycle labels
        for i in range(n_cycles):
            x = start_x + i * (bar_width + gap)
            c_lbl = Text(f"C{i + 1}", font_size=11, color=GREY_A)
            c_lbl.move_to(np.array([x, -1.8, 0]))
            self.add_fixed_in_frame_mobjects(c_lbl)
            self.add(c_lbl)

        self.wait(2)

        # ═════════════════════════════════════════════════════════════
        # ACT 6 — Summary card
        # ═════════════════════════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

        s_ttl = Text("Phase-Lifted Thouless Pump",
                      font_size=36, color=GOLD)
        s_eq = MathTex(
            r"Q_{\mathrm{cycle}}=e\left(C+"
            r"\frac{\Delta\theta_R}{2\pi_a}\right)",
            font_size=26, color=WHITE,
        )
        bullets = VGroup(
            Text("• Reduces to standard pump when Δθ_R = 0",
                 font_size=15, color=BLUE),
            Text("• Memory correction shifts quantized charge",
                 font_size=15, color=PURPLE),
            Text("• Falsifiable: compare principal vs lifted cycle counts",
                 font_size=15, color=TEAL),
            Text("• π_a sets effective phase period for normalisation",
                 font_size=15, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        card = VGroup(s_ttl, s_eq, bullets).arrange(DOWN, buff=0.25)
        box = SurroundingRectangle(card, color=BLUE_D, buff=0.3,
                                   corner_radius=0.1, stroke_width=1.5)

        self.add_fixed_in_frame_mobjects(box, s_ttl, s_eq, *bullets)
        self.play(FadeIn(box), Write(s_ttl), run_time=0.8)
        self.play(Write(s_eq), run_time=0.8)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.35)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
