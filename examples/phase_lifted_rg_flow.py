"""Phase-Lifted RG Memory Flow — visual proof.

    dg/d(ln μ) = β(g) − λ_M g sin²(θ_R / (2π_a))

Renormalisation-group flow with a bounded phase-memory correction.
The standard β-function drives scale evolution, while the lifted-phase
term penalises branch-inconsistent history and introduces a
memory-sensitive suppression of coupling flow.

Acts
----
  1. Title card with equation
  2. Standard RG flow: β(g) drives g(μ) to fixed points
  3. Introduce memory term: sin² suppression
  4. Side-by-side flow diagrams: memory-free vs memory-corrected
  5. Phase portrait with memory-shifted fixed points
  6. Ablation: λ_M sweep showing crossover shifts
  7. Summary card

Run
---
    manim -pql examples/phase_lifted_rg_flow.py PhaseLiftedRGFlow
    manim -qh  examples/phase_lifted_rg_flow.py PhaseLiftedRGFlow
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene,
    VGroup,
    Line,
    Arrow,
    Dot,
    Text,
    MathTex,
    DecimalNumber,
    Rectangle,
    RoundedRectangle,
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
    ParametricFunction,
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
    PI,
    TAU,
    config,
    interpolate_color,
    AnimationGroup,
    TracedPath,
    ValueTracker,
)


# ═══════════════════════════════════════════════════════════════════════════
# Physics: β-function and RG flow
# ═══════════════════════════════════════════════════════════════════════════

def beta_phi4(g):
    """One-loop φ⁴ β-function: β(g) = b₂ g² (asymptotic freedom style).

    We use a toy model: β(g) = g²  − 0.5g  giving fixed points at g=0 and g=0.5.
    """
    return g ** 2 - 0.5 * g


def memory_term(g, theta_R, lam_M, pi_a):
    """Phase-memory suppression: −λ_M g sin²(θ_R/(2π_a))."""
    return -lam_M * g * np.sin(theta_R / (2 * pi_a)) ** 2


def rg_flow_standard(g0, n_steps=500, dlnmu=0.02):
    """Integrate standard RG flow dg/dlnμ = β(g)."""
    g = np.zeros(n_steps)
    g[0] = g0
    for i in range(1, n_steps):
        dg = beta_phi4(g[i - 1]) * dlnmu
        g[i] = g[i - 1] + dg
        g[i] = max(0, min(g[i], 2.0))
    return g


def rg_flow_memory(g0, theta_R_0, lam_M=0.3, pi_a=np.pi,
                   n_steps=500, dlnmu=0.02):
    """Integrate memory-corrected RG flow."""
    g = np.zeros(n_steps)
    theta_R = np.zeros(n_steps)
    g[0] = g0
    theta_R[0] = theta_R_0
    for i in range(1, n_steps):
        b = beta_phi4(g[i - 1])
        mem = memory_term(g[i - 1], theta_R[i - 1], lam_M, pi_a)
        dg = (b + mem) * dlnmu
        g[i] = g[i - 1] + dg
        g[i] = max(0, min(g[i], 2.0))
        # Phase accumulates proportional to flow
        theta_R[i] = theta_R[i - 1] + 0.1 * g[i - 1] * dlnmu
    return g, theta_R


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class PhaseLiftedRGFlow(Scene):
    """Phase-Lifted RG Memory Flow animation."""

    def construct(self):
        # ═════════════════════════════════════════════════════════════
        # ACT 1 — Title + Equation
        # ═════════════════════════════════════════════════════════════
        title = Text("Phase-Lifted RG Memory Flow",
                     font_size=42, color=GOLD)
        subtitle = Text("Scale evolution with phase history",
                        font_size=20, color=TEAL)
        subtitle.next_to(title, DOWN, buff=0.15)

        eq = MathTex(
            r"\frac{d g}{d\ln \mu}=",
            r"\beta(g)",
            r"-",
            r"\lambda_M g\sin^2\!\left(\frac{\theta_R}{2\pi_a}\right)",
            font_size=30, color=WHITE,
        )
        eq.next_to(subtitle, DOWN, buff=0.35)
        eq[1].set_color(BLUE)
        eq[3].set_color(PURPLE)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle), run_time=0.4)
        self.play(Write(eq), run_time=1.2)

        # Annotate terms
        beta_lbl = Text("Standard RG flow", font_size=14, color=BLUE)
        beta_lbl.next_to(eq[1], UP, buff=0.12)
        mem_lbl = Text("Phase-memory suppression", font_size=14, color=PURPLE)
        mem_lbl.next_to(eq[3], DOWN, buff=0.12)

        self.play(Indicate(eq[1], color=BLUE), FadeIn(beta_lbl),
                  run_time=0.5)
        self.play(Indicate(eq[3], color=PURPLE), FadeIn(mem_lbl),
                  run_time=0.5)
        self.wait(0.8)

        self.play(
            FadeOut(title), FadeOut(subtitle),
            FadeOut(beta_lbl), FadeOut(mem_lbl),
            run_time=0.4,
        )
        eq.generate_target()
        eq.target.scale(0.8).to_edge(UP, buff=0.15)
        self.play(Transform(eq, eq.target), run_time=0.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 2 — Standard β(g) plot
        # ═════════════════════════════════════════════════════════════
        beta_title = Text("β-function: β(g) = g² − 0.5g",
                          font_size=16, color=GREY_A)
        beta_title.next_to(eq, DOWN, buff=0.25)
        self.play(FadeIn(beta_title), run_time=0.3)

        axes_beta = Axes(
            x_range=[0, 1.2, 0.25],
            y_range=[-0.15, 0.5, 0.1],
            x_length=5.5,
            y_length=2.5,
            axis_config={"color": GREY_A, "include_numbers": True,
                         "font_size": 12},
            tips=False,
        ).shift(DOWN * 0.8)

        x_lbl = MathTex(r"g", font_size=16, color=GREY_A).next_to(
            axes_beta.x_axis, DOWN, buff=0.1)
        y_lbl = MathTex(r"\beta(g)", font_size=16, color=GREY_A).next_to(
            axes_beta.y_axis, LEFT, buff=0.1)

        # β(g) curve
        beta_curve = ParametricFunction(
            lambda t: axes_beta.c2p(t, beta_phi4(t)),
            t_range=[0, 1.2],
            color=BLUE, stroke_width=3,
        )

        # Zero line
        zero_line = DashedLine(
            axes_beta.c2p(0, 0), axes_beta.c2p(1.2, 0),
            color=GREY_D, stroke_width=1,
        )

        # Fixed points
        fp_0 = Dot(axes_beta.c2p(0, 0), color=GREEN, radius=0.08)
        fp_05 = Dot(axes_beta.c2p(0.5, 0), color=RED, radius=0.08)
        fp0_lbl = MathTex(r"g^*=0", font_size=13, color=GREEN)
        fp0_lbl.next_to(fp_0, DOWN, buff=0.1)
        fp05_lbl = MathTex(r"g^*=0.5", font_size=13, color=RED)
        fp05_lbl.next_to(fp_05, DOWN, buff=0.1)

        self.play(
            Create(axes_beta), FadeIn(x_lbl), FadeIn(y_lbl),
            Create(zero_line),
            run_time=0.8,
        )
        self.play(Create(beta_curve), run_time=1.5)
        self.play(
            FadeIn(fp_0), FadeIn(fp_05),
            FadeIn(fp0_lbl), FadeIn(fp05_lbl),
            run_time=0.6,
        )

        # Flow arrows on g-axis
        arrow_positions = [0.1, 0.2, 0.35, 0.6, 0.8, 1.0]
        flow_arrows = VGroup()
        for gp in arrow_positions:
            bg = beta_phi4(gp)
            direction = RIGHT if bg > 0 else LEFT
            arr = Arrow(
                axes_beta.c2p(gp, -0.05),
                axes_beta.c2p(gp + 0.06 * np.sign(bg), -0.05),
                color=YELLOW, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.4,
            )
            flow_arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(a) for a in flow_arrows],
                        lag_ratio=0.1),
            run_time=1.0,
        )
        self.wait(1)

        # ═════════════════════════════════════════════════════════════
        # ACT 3 — Memory-corrected β effective
        # ═════════════════════════════════════════════════════════════
        self.play(Indicate(eq[3], color=PURPLE, scale_factor=1.3),
                  run_time=0.5)

        # Show effective flow for θ_R = π/2 (strong memory)
        theta_R_fixed = np.pi / 2
        lam_M = 0.3
        pi_a = np.pi

        def beta_effective(g):
            return beta_phi4(g) + memory_term(g, theta_R_fixed, lam_M, pi_a)

        mem_curve = ParametricFunction(
            lambda t: axes_beta.c2p(t, beta_effective(t)),
            t_range=[0, 1.2],
            color=PURPLE, stroke_width=3,
        )

        mem_curve_lbl = Text("β_eff (with memory)", font_size=12,
                              color=PURPLE)
        mem_curve_lbl.move_to(axes_beta.c2p(1.0, 0.35))
        std_curve_lbl = Text("β (standard)", font_size=12, color=BLUE)
        std_curve_lbl.move_to(axes_beta.c2p(0.9, 0.45))

        self.play(Create(mem_curve), FadeIn(mem_curve_lbl),
                  FadeIn(std_curve_lbl), run_time=1.5)

        # New fixed point for effective flow
        # Solve g² - 0.5g - lam_M * g * sin²(θ_R/(2π_a)) = 0
        # g(g - 0.5 - lam_M sin²(...)) = 0
        sin2_val = np.sin(theta_R_fixed / (2 * pi_a)) ** 2
        g_star_mem = 0.5 + lam_M * sin2_val  # shifted fixed point
        fp_mem = Dot(axes_beta.c2p(g_star_mem, 0), color=PURPLE,
                     radius=0.08)
        fp_mem_lbl = MathTex(
            r"g^*_{\mathrm{mem}}", font_size=13, color=PURPLE,
        )
        fp_mem_lbl.next_to(fp_mem, UP, buff=0.1)

        self.play(FadeIn(fp_mem), FadeIn(fp_mem_lbl), run_time=0.5)

        shift_arrow = Arrow(
            axes_beta.c2p(0.5, -0.1), axes_beta.c2p(g_star_mem, -0.1),
            color=ORANGE, buff=0.05, stroke_width=2,
        )
        shift_lbl = Text("Memory shift", font_size=11, color=ORANGE)
        shift_lbl.next_to(shift_arrow, DOWN, buff=0.05)
        self.play(Create(shift_arrow), FadeIn(shift_lbl), run_time=0.5)
        self.wait(1)

        # ═════════════════════════════════════════════════════════════
        # ACT 4 — Side-by-side RG flow trajectories
        # ═════════════════════════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not eq], run_time=0.5)

        flow_title = Text("RG flow trajectories: g(ln μ)",
                          font_size=18, color=GOLD)
        flow_title.next_to(eq, DOWN, buff=0.25)
        self.play(FadeIn(flow_title), run_time=0.3)

        axes_flow = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.2, 0.25],
            x_length=6.0,
            y_length=3.0,
            axis_config={"color": GREY_A, "include_numbers": True,
                         "font_size": 12},
            tips=False,
        ).shift(DOWN * 0.6)

        fx_lbl = MathTex(r"\ln\mu", font_size=16, color=GREY_A).next_to(
            axes_flow.x_axis, DOWN, buff=0.1)
        fy_lbl = MathTex(r"g", font_size=16, color=GREY_A).next_to(
            axes_flow.y_axis, LEFT, buff=0.1)

        self.play(Create(axes_flow), FadeIn(fx_lbl), FadeIn(fy_lbl),
                  run_time=0.8)

        # Multiple initial conditions
        g0_values = [0.1, 0.3, 0.7, 0.9]
        n_steps = 500
        dlnmu = 0.02
        lnmu = np.arange(n_steps) * dlnmu

        for g0 in g0_values:
            # Standard flow
            g_std = rg_flow_standard(g0, n_steps, dlnmu)
            pts_std = [axes_flow.c2p(lnmu[i], g_std[i])
                       for i in range(0, n_steps, 3)]
            curve_std = VGroup()
            for i in range(len(pts_std) - 1):
                seg = Line(pts_std[i], pts_std[i + 1],
                           stroke_width=2, color=BLUE)
                curve_std.add(seg)

            # Memory flow
            g_mem, _ = rg_flow_memory(g0, theta_R_0=1.0, lam_M=0.3,
                                      n_steps=n_steps, dlnmu=dlnmu)
            pts_mem = [axes_flow.c2p(lnmu[i], g_mem[i])
                       for i in range(0, n_steps, 3)]
            curve_mem = VGroup()
            for i in range(len(pts_mem) - 1):
                seg = Line(pts_mem[i], pts_mem[i + 1],
                           stroke_width=2, color=PURPLE)
                curve_mem.add(seg)

            self.play(Create(curve_std), Create(curve_mem), run_time=1.0)

        # Legend
        l_std = VGroup(
            Line(ORIGIN, RIGHT * 0.3, color=BLUE, stroke_width=3),
            Text("Standard β", font_size=12, color=BLUE),
        ).arrange(RIGHT, buff=0.1)
        l_mem = VGroup(
            Line(ORIGIN, RIGHT * 0.3, color=PURPLE, stroke_width=3),
            Text("Memory-corrected", font_size=12, color=PURPLE),
        ).arrange(RIGHT, buff=0.1)
        flow_legend = VGroup(l_std, l_mem).arrange(
            RIGHT, buff=0.4).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(flow_legend), run_time=0.4)

        # Fixed-point reference lines
        fp_line_std = DashedLine(
            axes_flow.c2p(0, 0.5), axes_flow.c2p(10, 0.5),
            color=BLUE, stroke_width=1,
        )
        fp_line_mem = DashedLine(
            axes_flow.c2p(0, g_star_mem), axes_flow.c2p(10, g_star_mem),
            color=PURPLE, stroke_width=1,
        )
        self.play(Create(fp_line_std), Create(fp_line_mem), run_time=0.5)
        self.wait(1.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 5 — λ_M sweep: crossover shift
        # ═════════════════════════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not eq], run_time=0.5)

        sweep_title = Text("λ_M sweep: memory strength → fixed-point shift",
                           font_size=16, color=GOLD)
        sweep_title.next_to(eq, DOWN, buff=0.25)
        self.play(FadeIn(sweep_title), run_time=0.3)

        axes_sweep = Axes(
            x_range=[0, 1.0, 0.2],
            y_range=[0.4, 1.0, 0.1],
            x_length=5.5,
            y_length=3.0,
            axis_config={"color": GREY_A, "include_numbers": True,
                         "font_size": 12},
            tips=False,
        ).shift(DOWN * 0.5)

        sx_lbl = MathTex(r"\lambda_M", font_size=16,
                         color=GREY_A).next_to(
            axes_sweep.x_axis, DOWN, buff=0.1)
        sy_lbl = MathTex(r"g^*", font_size=16,
                         color=GREY_A).next_to(
            axes_sweep.y_axis, LEFT, buff=0.1)

        self.play(Create(axes_sweep), FadeIn(sx_lbl), FadeIn(sy_lbl),
                  run_time=0.8)

        # For different θ_R values
        theta_vals = [
            (np.pi / 4, "θ_R = π/4", TEAL),
            (np.pi / 2, "θ_R = π/2", ORANGE),
            (3 * np.pi / 4, "θ_R = 3π/4", RED),
        ]

        for theta_v, label, col in theta_vals:
            sin2 = np.sin(theta_v / (2 * pi_a)) ** 2
            pts = []
            for lm in np.linspace(0, 1.0, 100):
                g_star = 0.5 + lm * sin2
                if g_star < 1.0:
                    pts.append(axes_sweep.c2p(lm, g_star))
            curve = VGroup()
            for i in range(len(pts) - 1):
                seg = Line(pts[i], pts[i + 1],
                           stroke_width=2.5, color=col)
                curve.add(seg)
            c_lbl = Text(label, font_size=11, color=col)
            if len(pts) > 0:
                c_lbl.next_to(pts[-1], RIGHT, buff=0.1)
            self.play(Create(curve), FadeIn(c_lbl), run_time=1.0)

        # Reference line at g* = 0.5 (standard)
        ref_line = DashedLine(
            axes_sweep.c2p(0, 0.5), axes_sweep.c2p(1.0, 0.5),
            color=GREY_D, stroke_width=1,
        )
        ref_lbl = MathTex(r"g^*_{\mathrm{std}} = 0.5", font_size=12,
                          color=GREY_D)
        ref_lbl.next_to(ref_line, LEFT, buff=0.1)
        self.play(Create(ref_line), FadeIn(ref_lbl), run_time=0.5)
        self.wait(1.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 6 — Summary card
        # ═════════════════════════════════════════════════════════════
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

        s_ttl = Text("Phase-Lifted RG Memory Flow",
                      font_size=34, color=GOLD)
        s_eq = MathTex(
            r"\frac{d g}{d\ln \mu}=\beta(g)"
            r"-\lambda_M g\sin^2\!\left("
            r"\frac{\theta_R}{2\pi_a}\right)",
            font_size=24, color=WHITE,
        )
        bullets = VGroup(
            Text("• Reduces to standard RG when λ_M = 0",
                 font_size=15, color=BLUE),
            Text("• Phase memory shifts fixed points and crossover scales",
                 font_size=15, color=PURPLE),
            Text("• sin² bounded correction preserves β-function skeleton",
                 font_size=15, color=ORANGE),
            Text("• Falsifiable: compare flow trajectories for "
                 "lifted vs no-memory",
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
