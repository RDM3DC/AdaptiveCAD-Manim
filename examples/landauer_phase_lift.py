"""Landauer–Phase-Lift Conductance Law — visual proof.

    G = (2e²/h) Σ_n T_n cos²(θ_{R,n} / (2π_a))

Phase-memory extension of Landauer transport: each channel T_n is
modulated by a bounded lifted-phase factor.  Channels with unresolved
or slip-prone phase history are suppressed.

Acts
----
  1. Title card with equation
  2. Standard Landauer conductance — channel bar chart
  3. Introduce phase modulation — cos² factor per channel
  4. Animate channel-selective suppression
  5. Sweep π_a from stiff to soft: show conductance change
  6. Ablation comparison: full lifted vs principal-branch vs no-memory
  7. Summary card

Run
---
    manim -pql examples/landauer_phase_lift.py LandauerPhaseLift
    manim -qh  examples/landauer_phase_lift.py LandauerPhaseLift
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene,
    VGroup,
    Rectangle,
    Line,
    Arrow,
    Text,
    MathTex,
    DecimalNumber,
    RoundedRectangle,
    Square,
    FadeIn,
    FadeOut,
    Create,
    Write,
    Transform,
    ReplacementTransform,
    Indicate,
    LaggedStart,
    SurroundingRectangle,
    Axes,
    DashedLine,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    ORIGIN,
    BLUE,
    BLUE_D,
    BLUE_E,
    RED,
    RED_E,
    GREEN,
    GREEN_E,
    YELLOW,
    WHITE,
    GREY,
    GREY_A,
    GREY_D,
    ORANGE,
    PURPLE,
    GOLD,
    TEAL,
    PINK,
    config,
    interpolate_color,
    AnimationGroup,
    ParametricFunction,
    PI,
    TAU,
)

# ═══════════════════════════════════════════════════════════════════════════
# Physics setup
# ═══════════════════════════════════════════════════════════════════════════

N_CHANNELS = 8


def _make_channels(seed=42):
    """Generate transmission coefficients and lifted phases."""
    rng = np.random.RandomState(seed)
    T = np.sort(rng.uniform(0.3, 1.0, N_CHANNELS))[::-1]
    theta_R = rng.uniform(-2.0, 2.0, N_CHANNELS)
    # Make a couple of channels have large phase slip
    theta_R[5] = np.pi * 0.95
    theta_R[6] = -np.pi * 1.1
    return T, theta_R


def landauer_G_standard(T):
    """Standard Landauer: G = (2e²/h) Σ T_n."""
    return T.sum()  # In units of 2e²/h


def landauer_G_lifted(T, theta_R, pi_a):
    """Phase-lifted Landauer: G = (2e²/h) Σ T_n cos²(θ_{R,n}/(2π_a))."""
    mod = np.cos(theta_R / (2 * pi_a)) ** 2
    return (T * mod).sum()


def channel_modulation(theta_R, pi_a):
    return np.cos(theta_R / (2 * pi_a)) ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Colour
# ═══════════════════════════════════════════════════════════════════════════

def _channel_color(idx, n):
    """Rainbow-ish color per channel."""
    cols = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED, PURPLE, PINK]
    return cols[idx % len(cols)]


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class LandauerPhaseLift(Scene):
    """Landauer–Phase-Lift Conductance Law animation."""

    def construct(self):
        T, theta_R = _make_channels()
        pi_a = np.pi

        # ═════════════════════════════════════════════════════════════
        # ACT 1 — Title + Equation
        # ═════════════════════════════════════════════════════════════
        title = Text("Landauer–Phase-Lift Conductance",
                     font_size=40, color=GOLD)
        subtitle = Text("Memory-modulated mesoscopic transport",
                        font_size=20, color=TEAL)
        subtitle.next_to(title, DOWN, buff=0.15)

        eq = MathTex(
            r"G=",
            r"\frac{2e^2}{h}",
            r"\sum_n",
            r"T_n",
            r"\cos^2\!\left(\frac{\theta_{R,n}}{2\pi_a}\right)",
            font_size=28, color=WHITE,
        )
        eq.next_to(subtitle, DOWN, buff=0.35)
        eq[1].set_color(GREY_A)
        eq[3].set_color(BLUE)
        eq[4].set_color(PURPLE)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle), run_time=0.4)
        self.play(Write(eq), run_time=1.2)

        # Annotate
        std_lbl = Text("Landauer skeleton", font_size=13, color=BLUE)
        std_lbl.next_to(eq[3], UP, buff=0.12)
        mem_lbl = Text("Phase memory modulation", font_size=13, color=PURPLE)
        mem_lbl.next_to(eq[4], DOWN, buff=0.12)
        self.play(
            Indicate(eq[3], color=BLUE),
            FadeIn(std_lbl), run_time=0.5)
        self.play(
            Indicate(eq[4], color=PURPLE),
            FadeIn(mem_lbl), run_time=0.5)
        self.wait(0.8)

        self.play(
            FadeOut(title), FadeOut(subtitle),
            FadeOut(std_lbl), FadeOut(mem_lbl),
            run_time=0.4,
        )
        eq.generate_target()
        eq.target.scale(0.8).to_edge(UP, buff=0.15)
        self.play(Transform(eq, eq.target), run_time=0.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 2 — Standard Landauer bar chart
        # ═════════════════════════════════════════════════════════════
        act_lbl = Text("Standard Landauer channels",
                       font_size=18, color=GREY_A)
        act_lbl.next_to(eq, DOWN, buff=0.3)
        self.play(FadeIn(act_lbl), run_time=0.3)

        bar_w = 0.55
        gap = 0.12
        total = N_CHANNELS * (bar_w + gap)
        x0 = -total / 2
        max_h = 2.5
        y_base = -1.8

        std_bars = VGroup()
        bar_labels = VGroup()
        for i in range(N_CHANNELS):
            x = x0 + i * (bar_w + gap)
            h = max_h * T[i]
            bar = Rectangle(
                width=bar_w, height=h,
                fill_color=_channel_color(i, N_CHANNELS),
                fill_opacity=0.7,
                stroke_color=_channel_color(i, N_CHANNELS),
                stroke_width=1.5,
            )
            bar.move_to(np.array([x, y_base + h / 2, 0]))
            std_bars.add(bar)

            lbl = MathTex(f"T_{{{i + 1}}}", font_size=12,
                          color=GREY_A)
            lbl.move_to(np.array([x, y_base - 0.2, 0]))
            bar_labels.add(lbl)

        # G_std value
        G_std = landauer_G_standard(T)
        g_text = MathTex(
            r"G_{\mathrm{std}} = " + f"{G_std:.2f}" + r"\;\frac{2e^2}{h}",
            font_size=20, color=BLUE,
        )
        g_text.to_edge(LEFT, buff=0.3).shift(UP * 0.5)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in std_bars],
                        lag_ratio=0.08),
            LaggedStart(*[FadeIn(l) for l in bar_labels],
                        lag_ratio=0.08),
            run_time=1.5,
        )
        self.play(Write(g_text), run_time=0.6)
        self.wait(0.8)

        # ═════════════════════════════════════════════════════════════
        # ACT 3 — Phase modulation overlay
        # ═════════════════════════════════════════════════════════════
        new_lbl = Text("Phase-lifted modulation",
                       font_size=18, color=PURPLE)
        new_lbl.move_to(act_lbl.get_center())
        self.play(ReplacementTransform(act_lbl, new_lbl), run_time=0.3)

        mod = channel_modulation(theta_R, pi_a)
        mod_bars = VGroup()
        for i in range(N_CHANNELS):
            x = x0 + i * (bar_w + gap)
            h_new = max_h * T[i] * mod[i]
            bar = Rectangle(
                width=bar_w * 0.6, height=h_new,
                fill_color=PURPLE, fill_opacity=0.8,
                stroke_color=PURPLE, stroke_width=1.5,
            )
            bar.move_to(np.array([x, y_base + h_new / 2, 0]))
            mod_bars.add(bar)

        # Show θ_R values per channel
        theta_labels = VGroup()
        for i in range(N_CHANNELS):
            x = x0 + i * (bar_w + gap)
            val = theta_R[i]
            t_lbl = MathTex(
                f"{val:.1f}", font_size=10, color=PURPLE,
            )
            t_lbl.move_to(np.array([x, y_base - 0.4, 0]))
            theta_labels.add(t_lbl)

        theta_header = MathTex(r"\theta_{R,n}:", font_size=12,
                               color=PURPLE)
        theta_header.move_to(np.array([x0 - bar_w, y_base - 0.4, 0]))

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in mod_bars],
                        lag_ratio=0.08),
            FadeIn(theta_header),
            LaggedStart(*[FadeIn(l) for l in theta_labels],
                        lag_ratio=0.05),
            run_time=1.5,
        )

        # Highlight suppressed channels
        for i in [5, 6]:
            x = x0 + i * (bar_w + gap)
            suppress_lbl = Text("suppressed", font_size=10, color=RED)
            suppress_lbl.move_to(np.array([x, y_base + max_h * T[i] + 0.2,
                                           0]))
            self.play(
                Indicate(mod_bars[i], color=RED),
                FadeIn(suppress_lbl),
                run_time=0.4,
            )

        # G_lifted value
        G_lift = landauer_G_lifted(T, theta_R, pi_a)
        g_lift_text = MathTex(
            r"G_{\mathrm{lifted}} = " + f"{G_lift:.2f}" +
            r"\;\frac{2e^2}{h}",
            font_size=20, color=PURPLE,
        )
        g_lift_text.next_to(g_text, DOWN, buff=0.2)
        self.play(Write(g_lift_text), run_time=0.6)
        self.wait(1)

        # ═════════════════════════════════════════════════════════════
        # ACT 4 — cos² modulation curve
        # ═════════════════════════════════════════════════════════════
        self.play(
            FadeOut(std_bars), FadeOut(mod_bars), FadeOut(bar_labels),
            FadeOut(theta_labels), FadeOut(theta_header),
            FadeOut(g_text), FadeOut(g_lift_text), FadeOut(new_lbl),
            *[FadeOut(m) for m in self.mobjects
              if m not in [eq]],
            run_time=0.5,
        )

        # Plot cos²(θ/(2π_a)) for different π_a values
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.1, 0.25],
            x_length=6.0,
            y_length=2.8,
            axis_config={"color": GREY_A, "include_numbers": True,
                         "font_size": 14},
            tips=False,
        ).shift(DOWN * 0.5)

        ax_lbl_x = MathTex(r"\theta_{R,n}", font_size=16,
                           color=GREY_A).next_to(axes.x_axis, DOWN, buff=0.1)
        ax_lbl_y = MathTex(r"\cos^2", font_size=16,
                           color=GREY_A).next_to(axes.y_axis, LEFT,
                                                  buff=0.1)

        curve_title = Text("Modulation factor vs lifted phase",
                           font_size=16, color=GOLD)
        curve_title.next_to(eq, DOWN, buff=0.25)

        self.play(
            Create(axes), FadeIn(ax_lbl_x), FadeIn(ax_lbl_y),
            FadeIn(curve_title),
            run_time=0.8,
        )

        pi_a_values = [
            (np.pi * 2, "π_a = 2π (stiff)", BLUE),
            (np.pi, "π_a = π (standard)", TEAL),
            (np.pi / 2, "π_a = π/2 (soft)", ORANGE),
        ]

        curves_group = VGroup()
        for pa, label, col in pi_a_values:
            curve = ParametricFunction(
                lambda t, pa=pa: axes.c2p(
                    t, np.cos(t / (2 * pa)) ** 2),
                t_range=[-4, 4],
                color=col, stroke_width=2.5,
            )
            c_lbl = Text(label, font_size=12, color=col)
            c_lbl.next_to(curve, RIGHT, buff=0.1).shift(UP * 0.1)
            curves_group.add(VGroup(curve, c_lbl))
            self.play(Create(curve), FadeIn(c_lbl), run_time=1.0)

        # Mark channel positions on the standard curve
        for i in range(N_CHANNELS):
            x = theta_R[i]
            y = np.cos(x / (2 * pi_a)) ** 2
            dot = Square(side_length=0.08, fill_color=YELLOW,
                         fill_opacity=1, stroke_width=0)
            dot.move_to(axes.c2p(x, y))
            self.add(dot)

        self.wait(1.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 5 — Ablation comparison bar chart
        # ═════════════════════════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

        ab_title = Text("Ablation comparison", font_size=22, color=GOLD)
        ab_title.to_edge(UP, buff=0.3)
        self.play(Write(ab_title), run_time=0.5)

        # Three conditions
        conditions = [
            ("Full lifted\ncos²(θ_R/2π_a)",
             landauer_G_lifted(T, theta_R, pi_a), PURPLE),
            ("Principal\nbranch",
             landauer_G_lifted(T, np.mod(theta_R + np.pi, 2 * np.pi) - np.pi,
                               pi_a), TEAL),
            ("No memory\n(standard)",
             landauer_G_standard(T), BLUE),
        ]

        bars = VGroup()
        for i, (label, G_val, col) in enumerate(conditions):
            x = (i - 1) * 2.5
            h = G_val / landauer_G_standard(T) * 3.0
            bar = Rectangle(
                width=1.2, height=h,
                fill_color=col, fill_opacity=0.7,
                stroke_color=col, stroke_width=2,
            )
            bar.move_to(np.array([x, -1.5 + h / 2, 0]))
            bar_label = Text(label, font_size=12, color=col)
            bar_label.next_to(bar, DOWN, buff=0.15)
            val_label = MathTex(
                f"G = {G_val:.2f}", font_size=16, color=WHITE,
            )
            val_label.next_to(bar, UP, buff=0.1)
            bars.add(VGroup(bar, bar_label, val_label))

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in bars],
                        lag_ratio=0.2),
            run_time=2,
        )

        ablation_note = Text(
            "Lifted-phase model predicts selective channel suppression",
            font_size=14, color=GREY_A,
        )
        ablation_note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ablation_note), run_time=0.5)
        self.wait(1.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 6 — Summary card
        # ═════════════════════════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

        s_ttl = Text("Landauer–Phase-Lift Conductance",
                      font_size=34, color=GOLD)
        s_eq = MathTex(
            r"G=\frac{2e^2}{h}\sum_n T_n\cos^2\!\left("
            r"\frac{\theta_{R,n}}{2\pi_a}\right)",
            font_size=24, color=WHITE,
        )
        bullets = VGroup(
            Text("• Reduces to Landauer when cos² → 1",
                 font_size=15, color=BLUE),
            Text("• Channels with slip-prone θ_R are suppressed",
                 font_size=15, color=PURPLE),
            Text("• π_a tunes memory-suppression sensitivity",
                 font_size=15, color=ORANGE),
            Text("• Falsifiable: compare against no-memory and "
                 "principal-branch",
                 font_size=15, color=TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        card = VGroup(s_ttl, s_eq, bullets).arrange(DOWN, buff=0.25)
        box = SurroundingRectangle(card, color=BLUE_D, buff=0.3,
                                   corner_radius=0.1, stroke_width=1.5)

        self.play(FadeIn(box), Write(s_ttl), run_time=0.8)
        self.play(Write(s_eq), run_time=0.8)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.35)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
