"""Wealth Monopole — Hedgehog cascade meets compounding returns.

A magnetic monopole radiates B = g/(4πr²) r̂ through every sphere that
encloses it.  The total flux Φ = g is quantised: eg = nℏ/2.  A "hedgehog"
field configuration has topological charge — the field cannot be combed
smooth on S².

Map the physics to finance:
  • Magnetic charge  g  →  initial deposit (source of all returns)
  • Flux quanta  n     →  compounding periods
  • Field lines        →  radiating returns growing with r → ∞
  • Hedgehog vectors   →  diversified portfolio directions
  • Dirac string       →  hidden risk singularity (veiled losses)
  • Live $ readout     →  balance radiating outward with each quantum

Acts
----
1.  Title — "The Wealth Monopole"
2.  Deposit the charge g — monopole appears, radial B-field lines grow
3.  Flux quantisation — concentric spheres show Φ = g through each
4.  Hedgehog cascade — vectors at every direction radiate, twist colour
    by topological winding number
5.  The Dirac string — hidden risk line, briefly revealed then veiled
6.  Compounding readout — field lines pulse outward in n quanta,
    live $ counter ticks up:  W(n) = g·(1 + r)^n
7.  Summary card

Run
---
    manim -pql examples/wealth_monopole.py WealthMonopole
    manim -qh  examples/wealth_monopole.py WealthMonopole
"""

from __future__ import annotations

import colorsys
import numpy as np
from manim import (
    Scene,
    Circle,
    Dot,
    Arrow,
    Line,
    VGroup,
    MathTex,
    Text,
    DecimalNumber,
    Write,
    FadeIn,
    FadeOut,
    Create,
    GrowFromCenter,
    GrowArrow,
    Transform,
    Indicate,
    Flash,
    SurroundingRectangle,
    AnimationGroup,
    Succession,
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
    GREEN_E,
    BLUE,
    BLUE_D,
    BLUE_E,
    ORANGE,
    GOLD,
    TEAL,
    GREY,
    GREY_A,
    PURPLE,
    config,
    interpolate_color,
)

# ═══════════════════════════════════════════════════════════════════════════
# Physical / financial constants
# ═══════════════════════════════════════════════════════════════════════════
G_CHARGE   = 10_000       # initial deposit  ($)
RATE       = 0.12         # annual return  (12 %)
N_QUANTA   = 20           # compounding periods (years)
N_DIRS     = 16           # hedgehog directions (field lines)
INNER_R    = 0.25         # monopole dot radius
MAX_R      = 3.0          # max field-line display radius


# ═══════════════════════════════════════════════════════════════════════════
# Colour helpers
# ═══════════════════════════════════════════════════════════════════════════

def _hue_col(frac, s=0.85, v=0.95):
    """HSV → hex, frac ∈ [0, 1]."""
    r, g, b = colorsys.hsv_to_rgb(frac % 1, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _wealth_color(frac):
    """Dark-blue → cyan → gold → white as wealth grows."""
    if frac < 0.33:
        return interpolate_color(BLUE_E, TEAL, frac / 0.33)
    elif frac < 0.66:
        return interpolate_color(TEAL, GOLD, (frac - 0.33) / 0.33)
    else:
        return interpolate_color(GOLD, WHITE, (frac - 0.66) / 0.34)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class WealthMonopole(Scene):
    """2-D projection of a hedgehog monopole with live wealth readout."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("The Wealth Monopole", font_size=48, color=GOLD)
        sub = MathTex(
            r"\mathbf{B} = \frac{g}{4\pi r^2}\,\hat{r}"
            r"\qquad \Phi = g \qquad eg = \tfrac{n\hbar}{2}",
            font_size=26, color=BLUE,
        )
        sub.next_to(ttl, DOWN, buff=0.3)
        tag = Text(
            "point-source returns radiating flux quanta\n"
            "through every sphere of opportunity",
            font_size=18, color=GREY_A, line_spacing=1.2,
        )
        tag.next_to(sub, DOWN, buff=0.35)

        self.play(Write(ttl), run_time=1.5)                             # 1
        self.play(FadeIn(sub), run_time=1)                               # 2
        self.play(FadeIn(tag), run_time=1)                               # 3
        self.wait(1.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(tag))              # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Deposit the charge g — monopole + field lines grow
        # ─────────────────────────────────────────────────────────────────
        center = ORIGIN

        # Monopole dot
        mono = Dot(center, radius=0.18, color=GOLD)
        g_lbl = MathTex(r"g", font_size=22, color=GOLD)
        g_lbl.next_to(mono, DOWN, buff=0.15)

        # Deposit label
        dep_lbl = Text(
            f"Deposit  g = ${G_CHARGE:,.0f}",
            font_size=22, color=GREEN,
        )
        dep_lbl.to_corner(UP + LEFT, buff=0.3)

        self.play(GrowFromCenter(mono), FadeIn(g_lbl), run_time=1)       # 5
        self.play(Flash(mono, color=GOLD, flash_radius=0.8), run_time=0.6)  # 6
        self.play(FadeIn(dep_lbl), run_time=0.6)                        # 7

        # Grow N_DIRS radial field lines (short initially)
        angles = [TAU * i / N_DIRS for i in range(N_DIRS)]
        lines_grp = VGroup()
        arrows_grp = VGroup()
        for i, ang in enumerate(angles):
            col = _hue_col(i / N_DIRS)
            d = np.array([np.cos(ang), np.sin(ang), 0])
            ln = Line(
                center + 0.22 * d,
                center + 1.0 * d,
                color=col,
                stroke_width=2.5,
            )
            lines_grp.add(ln)
            ar = Arrow(
                center + 0.8 * d,
                center + 1.0 * d,
                color=col,
                buff=0,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.35,
            )
            arrows_grp.add(ar)

        self.play(
            *[Create(ln) for ln in lines_grp],
            run_time=1.5,
        )                                                                # 8
        self.play(
            *[GrowArrow(ar) for ar in arrows_grp],
            run_time=0.8,
        )                                                                # 9

        # Equation
        b_eq = MathTex(
            r"\mathbf{B} = \frac{g}{4\pi r^2}\,\hat{r}",
            font_size=24, color=WHITE,
        )
        b_eq.to_corner(UP + RIGHT, buff=0.3)
        self.play(FadeIn(b_eq), run_time=0.5)                           # 10

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Flux quantisation through concentric spheres
        # ─────────────────────────────────────────────────────────────────
        flux_eq = MathTex(
            r"\oint_{S^2} \mathbf{B}\cdot d\mathbf{A} = g"
            r"\;\;\;\forall\;r",
            font_size=22, color=YELLOW,
        )
        flux_eq.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(flux_eq), run_time=0.5)                        # 11

        # Draw 3 concentric "sphere" circles
        spheres = VGroup()
        sphere_lbls = VGroup()
        for k, rad in enumerate([1.3, 2.0, 2.7]):
            c = Circle(radius=rad, color=BLUE_D, stroke_width=1.5,
                        stroke_opacity=0.5)
            spheres.add(c)
            lbl = MathTex(
                r"\Phi = g", font_size=14, color=BLUE_D,
            )
            lbl.move_to(np.array([rad * 0.7, rad * 0.7, 0]))
            sphere_lbls.add(lbl)

        for c, lbl in zip(spheres, sphere_lbls):
            self.play(Create(c), FadeIn(lbl), run_time=0.6)             # 12-14

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Hedgehog cascade — vectors at every direction
        # ─────────────────────────────────────────────────────────────────
        self.play(FadeOut(flux_eq), FadeOut(sphere_lbls), run_time=0.3)  # 15

        hedge_title = Text(
            "Hedgehog Field — topological charge",
            font_size=20, color=RED,
        )
        hedge_title.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(hedge_title), run_time=0.5)                     # 16

        # Add more intermediate arrows to show hedgehog density
        N_HEDGE = 32
        hedge_arrows = VGroup()
        for i in range(N_HEDGE):
            ang = TAU * i / N_HEDGE
            col = _hue_col(i / N_HEDGE)
            d = np.array([np.cos(ang), np.sin(ang), 0])
            r_base = 1.1 + 0.4 * np.sin(3 * ang)  # slight wobble
            ar = Arrow(
                center + r_base * d,
                center + (r_base + 0.45) * d,
                color=col,
                buff=0,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.3,
            )
            hedge_arrows.add(ar)

        self.play(
            *[GrowArrow(a) for a in hedge_arrows],
            run_time=1.5,
        )                                                                # 17

        # Winding number equation
        wind_eq = MathTex(
            r"Q = \frac{1}{4\pi}\oint \hat{n}\cdot"
            r"\left(\partial_\theta\hat{n}\times"
            r"\partial_\phi\hat{n}\right)d\Omega = 1",
            font_size=20, color=WHITE,
        )
        wind_eq.to_corner(DOWN + RIGHT, buff=0.3)
        self.play(FadeIn(wind_eq), run_time=0.8)                        # 18
        self.play(Indicate(wind_eq, color=GOLD), run_time=0.6)           # 19

        self.wait(0.5)
        self.play(
            FadeOut(hedge_title), FadeOut(hedge_arrows),
            FadeOut(wind_eq),
            run_time=0.6,
        )                                                                # 20

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Dirac string — hidden risk, briefly revealed
        # ─────────────────────────────────────────────────────────────────
        string_lbl = Text(
            "The Dirac String — hidden risk",
            font_size=20, color=RED,
        )
        string_lbl.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(string_lbl), run_time=0.5)                     # 21

        # String along −y (downward)
        string_line = Line(
            center + 0.2 * DOWN,
            center + 3.5 * DOWN,
            color=RED,
            stroke_width=5,
            stroke_opacity=0.9,
        )
        string_glow = Line(
            center + 0.2 * DOWN,
            center + 3.5 * DOWN,
            color=RED_E,
            stroke_width=12,
            stroke_opacity=0.25,
        )

        self.play(Create(string_glow), Create(string_line), run_time=1) # 22

        risk_lbl = MathTex(
            r"\text{A singular } \;\Rightarrow\;"
            r"\text{losses if you look directly}",
            font_size=18, color=RED,
        )
        risk_lbl.to_corner(DOWN + LEFT, buff=0.3)
        self.play(FadeIn(risk_lbl), run_time=0.6)                       # 23

        # Veil it — fade the string to near-invisible
        veil_lbl = Text(
            "Gauge transform veils the singularity",
            font_size=16, color=GREY_A,
        )
        veil_lbl.to_corner(UP + LEFT, buff=0.3)
        self.play(
            string_line.animate.set_opacity(0.08),
            string_glow.animate.set_opacity(0.03),
            FadeOut(risk_lbl),
            FadeIn(veil_lbl),
            run_time=1.5,
        )                                                                # 24

        self.play(
            FadeOut(string_lbl), FadeOut(veil_lbl),
            run_time=0.4,
        )                                                                # 25

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Compounding readout — field lines pulse, $ ticks up
        # ─────────────────────────────────────────────────────────────────
        comp_title = Text(
            "Flux Quanta  →  Compounding Periods",
            font_size=22, color=GOLD,
        )
        comp_title.to_edge(UP, buff=0.15)
        self.play(FadeOut(dep_lbl), FadeOut(b_eq), FadeIn(comp_title),
                  run_time=0.6)                                          # 26

        # Compounding equation
        comp_eq = MathTex(
            r"W(n) = g\,(1 + r)^n",
            font_size=28, color=GREEN,
        )
        comp_eq.next_to(comp_title, DOWN, buff=0.25)
        self.play(Write(comp_eq), run_time=1)                            # 27

        # Live $ readout
        balance = DecimalNumber(
            G_CHARGE,
            num_decimal_places=0,
            font_size=30,
            color=GREEN,
            include_sign=False,
        )
        dollar = Text("$", font_size=30, color=GREEN)
        bal_grp = VGroup(dollar, balance).arrange(RIGHT, buff=0.05)
        bal_grp.to_corner(DOWN + RIGHT, buff=0.35)
        self.play(FadeIn(bal_grp), run_time=0.5)                        # 28

        # Quantum / year label
        n_lbl = MathTex(r"n = 0", font_size=22, color=WHITE)
        n_lbl.to_corner(DOWN + LEFT, buff=0.35)
        self.play(FadeIn(n_lbl), run_time=0.3)                          # 29

        # Pulse field lines outward in quanta — extend lines + update $
        current_balance = float(G_CHARGE)

        for q in range(1, N_QUANTA + 1):
            current_balance *= (1 + RATE)
            frac = q / N_QUANTA

            # Extend field lines
            new_r = 1.0 + (MAX_R - 1.0) * frac
            new_lines = VGroup()
            new_arrows = VGroup()
            for i, ang in enumerate(angles):
                col = _wealth_color(frac)
                d = np.array([np.cos(ang), np.sin(ang), 0])
                ln = Line(
                    center + 0.22 * d,
                    center + new_r * d,
                    color=col,
                    stroke_width=2.5 + 1.5 * frac,
                )
                new_lines.add(ln)
                ar = Arrow(
                    center + (new_r - 0.2) * d,
                    center + new_r * d,
                    color=col,
                    buff=0,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.3,
                )
                new_arrows.add(ar)

            # Update n label
            new_n_lbl = MathTex(
                r"n = " + str(q), font_size=22, color=WHITE,
            )
            new_n_lbl.to_corner(DOWN + LEFT, buff=0.35)

            # Update balance
            new_bal = DecimalNumber(
                current_balance,
                num_decimal_places=0,
                font_size=30,
                color=_wealth_color(frac),
                include_sign=False,
            )
            new_dollar = Text("$", font_size=30, color=_wealth_color(frac))
            new_bal_grp = VGroup(new_dollar, new_bal).arrange(RIGHT, buff=0.05)
            new_bal_grp.to_corner(DOWN + RIGHT, buff=0.35)

            # Grow concentric sphere (every 5 quanta)
            extras = []
            if q % 5 == 0:
                sph_c = Circle(
                    radius=new_r,
                    color=_wealth_color(frac),
                    stroke_width=1.0,
                    stroke_opacity=0.3,
                )
                extras.append(Create(sph_c))

            self.play(
                Transform(lines_grp, new_lines),
                Transform(arrows_grp, new_arrows),
                Transform(n_lbl, new_n_lbl),
                Transform(bal_grp, new_bal_grp),
                *extras,
                run_time=0.30,
            )                                                    # 30-49

        # Final flash
        final_lbl = Text(
            f"${current_balance:,.0f}", font_size=36, color=GOLD,
        )
        final_lbl.next_to(comp_eq, DOWN, buff=0.3)
        self.play(
            FadeIn(final_lbl),
            Flash(mono, color=GOLD, flash_radius=1.5),
            run_time=1,
        )                                                                # 50

        years_lbl = Text(
            f"{N_QUANTA} flux quanta  =  {N_QUANTA} years  @  {RATE*100:.0f}%",
            font_size=18, color=GREY_A,
        )
        years_lbl.next_to(final_lbl, DOWN, buff=0.15)
        self.play(FadeIn(years_lbl), run_time=0.6)                      # 51

        # Dirac quantisation tie-in
        quant_eq = MathTex(
            r"eg = \frac{n\hbar}{2}"
            r"\;\;\Rightarrow\;\;\text{returns are quantised}",
            font_size=22, color=YELLOW,
        )
        quant_eq.to_edge(DOWN, buff=0.15)
        self.play(Write(quant_eq), run_time=1)                          # 52
        self.play(Indicate(quant_eq, color=GOLD), run_time=0.8)          # 53

        self.wait(1.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1,
        )                                                                # 54

        card_title = Text("The Wealth Monopole", font_size=34, color=GOLD)
        card_title.to_edge(UP, buff=0.4)
        self.play(Write(card_title))                                     # 55

        lines = VGroup(
            MathTex(
                r"\text{Magnetic charge } g = \$10{,}000"
                r"\;\to\;\text{initial deposit}",
                font_size=19,
            ),
            MathTex(
                r"\text{Flux quanta } n = 20"
                r"\;\to\;\text{compounding periods (years)}",
                font_size=19,
            ),
            MathTex(
                r"\text{Hedgehog } \hat{r} \text{ field}"
                r"\;\to\;\text{diversified returns in all directions}",
                font_size=19,
            ),
            MathTex(
                r"\text{Dirac string (singularity)}"
                r"\;\to\;\text{hidden risk, gauge-veiled}",
                font_size=19, color=RED,
            ),
            MathTex(
                r"eg = \tfrac{n\hbar}{2}"
                r"\;\to\;\text{one monopole quantises all charges}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"W(20) = \$10{,}000 \times 1.12^{20}"
                r"\;=\;\$" + f"{G_CHARGE * (1 + RATE)**N_QUANTA:,.0f}",
                font_size=19, color=GOLD,
            ),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        lines.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, lines),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        disclaimer = Text(
            "Not financial advice.  Magnetic monopoles have not been observed.",
            font_size=12, color=GREY,
        )
        disclaimer.to_edge(DOWN, buff=0.1)

        self.play(FadeIn(box), run_time=0.5)                            # 56
        for line in lines:
            self.play(FadeIn(line), run_time=0.7)                        # 57-62
        self.play(FadeIn(disclaimer), run_time=0.4)                      # 63

        self.wait(2)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(lines), FadeOut(box),
            FadeOut(disclaimer),
            run_time=1.5,
        )                                                                # 64
