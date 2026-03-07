"""Geodesic Deviation — Tidal Forces & Spaghettification.

A circular ball of test particles in radial free-fall toward a massive body.
The tidal (Riemann) tensor stretches the ball radially and squeezes it
laterally, turning a circle into an ever-thinning ellipse — spaghettification.

    ξ̈^μ = −R^μ_{νρσ} u^ν ξ^ρ u^σ     (geodesic deviation equation)

For a Schwarzschild black hole (radial fall along r):
  • Radial tidal acceleration   ∝  +2M/r³   (stretch)
  • Lateral tidal acceleration  ∝  −M/r³     (squeeze)

Acts
----
1. Title card
2. Far field — ball of particles at large r, introduce deviation equation
3. Infall — ball moves downward toward the black hole, tidal gradient
   begins to stretch radially and squeeze laterally
4. Close approach — extreme stretching, spaghettification,
   colour shift blue→red with increasing tidal force
5. Riemann tensor reveal — show the physical meaning
6. Summary card

Run
---
    manim -pql examples/geodesic_deviation.py GeodesicDeviation
    manim -qh  examples/geodesic_deviation.py GeodesicDeviation
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene,
    Circle,
    Dot,
    Ellipse,
    Line,
    DashedLine,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Transform,
    Indicate,
    Flash,
    SurroundingRectangle,
    AnimationGroup,
    Arrow,
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
    PURPLE,
    config,
    interpolate_color,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
N_PARTICLES = 24          # dots in the test-particle ring
N_STEPS     = 40          # animation steps during infall
R_START     = 3.0         # starting screen-y (top of frame)
R_END       = -2.8        # ending screen-y (near BH)
BALL_R0     = 0.55        # initial ball radius
M_BH        = 1.0         # mass parameter (controls tidal strength)


# ═══════════════════════════════════════════════════════════════════════════
# Tidal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _tidal_axes(r_screen: float):
    """Return (radial_stretch, lateral_squeeze) as multiplicative factors
    relative to the initial ball radius.

    Maps screen-y position to a pseudo-Schwarzschild tidal field:
      stretch  ∝  2M/r³    squeeze  ∝  M/r³
    with r mapped so that r_screen ∈ [R_START, R_END] corresponds to
    approaching the horizon.
    """
    # Map screen y → effective r  (larger y = farther from BH)
    t = (R_START - r_screen) / (R_START - R_END)       # 0 at start, 1 at end
    t = np.clip(t, 0, 1)

    # Gentle at first, extreme near the end
    tidal = 1.0 + 8.0 * t ** 2.5

    stretch = tidal          # radial  (vertical on screen)
    squeeze = 1.0 / tidal    # lateral (horizontal) — volume preserved

    return stretch, squeeze


def _tidal_color(t_frac: float):
    """Colour ramp: BLUE (far) → TEAL → YELLOW → RED (near)."""
    if t_frac < 0.33:
        return interpolate_color(BLUE, TEAL, t_frac / 0.33)
    elif t_frac < 0.66:
        return interpolate_color(TEAL, YELLOW, (t_frac - 0.33) / 0.33)
    else:
        return interpolate_color(YELLOW, RED, (t_frac - 0.66) / 0.34)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class GeodesicDeviation(Scene):
    """A ball of test particles spaghettifies during radial infall."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("Geodesic Deviation", font_size=50, color=GOLD)
        sub = Text("Tidal Forces & Spaghettification",
                    font_size=26, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        desc = Text(
            "A ball of test particles stretches radially\n"
            "and squeezes laterally as it falls toward\n"
            "a black hole — the Riemann tensor at work.",
            font_size=18, color=WHITE, line_spacing=1.3,
        )
        desc.next_to(sub, DOWN, buff=0.4)

        self.play(Write(ttl), run_time=1.5)                              # 1
        self.play(FadeIn(sub), run_time=1)                                # 2
        self.play(FadeIn(desc), run_time=1)                               # 3
        self.wait(1.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(desc))              # 4

        # ─────────────────────────────────────────────────────────────────
        # Background — black hole at bottom + coordinate reference
        # ─────────────────────────────────────────────────────────────────
        # BH representation
        bh_dot = Dot(
            np.array([0, R_END - 0.6, 0]),
            radius=0.35, color=WHITE,
        ).set_fill(opacity=0.0).set_stroke(WHITE, width=2)
        bh_lbl = MathTex(r"M", font_size=22, color=GREY_A)
        bh_lbl.next_to(bh_dot, DOWN, buff=0.1)

        # "r" axis
        r_arrow = Arrow(
            start=np.array([-5.5, R_END - 0.3, 0]),
            end=np.array([-5.5, R_START + 0.5, 0]),
            color=GREY_D,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05,
        )
        r_lbl = MathTex(r"r", font_size=20, color=GREY_D)
        r_lbl.next_to(r_arrow, UP, buff=0.08)

        self.play(
            FadeIn(bh_dot), FadeIn(bh_lbl),
            Create(r_arrow), FadeIn(r_lbl),
            run_time=1,
        )                                                                 # 5

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Ball of particles at large r
        # ─────────────────────────────────────────────────────────────────
        center = np.array([0, R_START, 0])

        # Create ring of dots
        dots = VGroup()
        for i in range(N_PARTICLES):
            ang = TAU * i / N_PARTICLES
            pos = center + BALL_R0 * np.array([np.cos(ang), np.sin(ang), 0])
            d = Dot(pos, radius=0.05, color=BLUE)
            dots.add(d)

        # Outline ellipse (starts as circle)
        outline = Ellipse(
            width=2 * BALL_R0, height=2 * BALL_R0,
            color=BLUE, stroke_width=1.5, stroke_opacity=0.4,
        ).move_to(center)

        self.play(
            *[FadeIn(d) for d in dots],
            Create(outline),
            run_time=1.5,
        )                                                                 # 6

        # Geodesic deviation equation
        dev_eq = MathTex(
            r"\ddot{\xi}^\mu = "
            r"-R^\mu{}_{\nu\rho\sigma}\,"
            r"u^\nu \xi^\rho u^\sigma",
            font_size=24, color=WHITE,
        )
        dev_eq.to_corner(UP + LEFT, buff=0.3)
        self.play(Write(dev_eq), run_time=1.2)                           # 7

        # Arrows showing tidal directions (conceptual)
        stretch_lbl = MathTex(
            r"\text{stretch } \uparrow\downarrow",
            font_size=18, color=YELLOW,
        )
        squeeze_lbl = MathTex(
            r"\text{squeeze } \leftarrow\rightarrow",
            font_size=18, color=TEAL,
        )
        stretch_lbl.to_corner(UP + RIGHT, buff=0.3)
        squeeze_lbl.next_to(stretch_lbl, DOWN, buff=0.12)
        self.play(FadeIn(stretch_lbl), FadeIn(squeeze_lbl),
                  run_time=0.6)                                           # 8

        # Tidal force magnitudes
        tidal_eq = MathTex(
            r"\text{radial: } +\frac{2M}{r^3}"
            r"\qquad"
            r"\text{lateral: } -\frac{M}{r^3}",
            font_size=20, color=WHITE,
        )
        tidal_eq.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(tidal_eq), run_time=0.6)                        # 9

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Infall: ball stretches + squeezes progressively
        # ─────────────────────────────────────────────────────────────────
        for step in range(1, N_STEPS + 1):
            frac = step / N_STEPS
            y_pos = R_START + (R_END - R_START) * frac
            cur_center = np.array([0, y_pos, 0])

            stretch, squeeze = _tidal_axes(y_pos)

            # Clamp to prevent absurd sizes
            a_x = BALL_R0 * squeeze   # lateral (width)
            a_y = BALL_R0 * stretch   # radial  (height)
            a_x = max(a_x, 0.015)
            a_y = min(a_y, 4.5)

            col = _tidal_color(frac)

            # Move dots to ellipse positions
            new_dots = VGroup()
            for i in range(N_PARTICLES):
                ang = TAU * i / N_PARTICLES
                pos = cur_center + np.array([
                    a_x * np.cos(ang),
                    a_y * np.sin(ang),
                    0,
                ])
                d = Dot(pos, radius=0.05, color=col)
                new_dots.add(d)

            # New outline ellipse
            new_outline = Ellipse(
                width=2 * a_x, height=2 * a_y,
                color=col, stroke_width=1.5, stroke_opacity=0.35,
            ).move_to(cur_center)

            self.play(
                Transform(dots, new_dots),
                Transform(outline, new_outline),
                run_time=0.12,
            )                                                     # 10-49

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Extreme spaghettification freeze-frame
        # ─────────────────────────────────────────────────────────────────
        spag_lbl = Text(
            "Spaghettification",
            font_size=28, color=RED,
        )
        spag_lbl.to_edge(RIGHT, buff=0.4)
        self.play(FadeIn(spag_lbl), run_time=0.6)                        # 50

        spag_note = MathTex(
            r"\Delta F \propto \frac{2M}{r^3}\,\delta r"
            r"\;\;\overset{r\to r_s}{\longrightarrow}\;\;\infty",
            font_size=22, color=RED,
        )
        spag_note.next_to(spag_lbl, DOWN, buff=0.2)
        self.play(Write(spag_note), run_time=1)                           # 51

        # Flash the spaghettified ball
        self.play(
            Indicate(dots, color=YELLOW, scale_factor=1.05),
            run_time=0.8,
        )                                                                 # 52

        self.wait(1)

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Riemann tensor reveal
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(spag_lbl), FadeOut(spag_note),
            FadeOut(tidal_eq), FadeOut(stretch_lbl), FadeOut(squeeze_lbl),
            run_time=0.6,
        )                                                                 # 53

        riemann_title = Text(
            "The Riemann Tensor", font_size=24, color=GOLD,
        )
        riemann_title.to_edge(UP, buff=0.15)
        self.play(FadeIn(riemann_title), run_time=0.5)                    # 54

        meaning_lines = VGroup(
            MathTex(
                r"R^\mu{}_{\nu\rho\sigma}"
                r"\;\text{ measures how nearby geodesics}",
                font_size=19, color=WHITE,
            ),
            MathTex(
                r"\text{accelerate relative to each other}",
                font_size=19, color=WHITE,
            ),
            MathTex(
                r"\text{Non-zero } R \;\Leftrightarrow\;"
                r"\text{spacetime is curved}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Tidal forces = physical observable of curvature}",
                font_size=19, color=YELLOW,
            ),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        meaning_lines.to_corner(DOWN + LEFT, buff=0.3)

        for line in meaning_lines:
            self.play(FadeIn(line), run_time=0.6)                         # 55-58

        self.play(Indicate(dev_eq, color=GOLD), run_time=0.8)             # 59

        # Schwarzschild specific
        schwarz_eq = MathTex(
            r"R^r{}_{t r t} = \frac{2M}{r^3}"
            r"\qquad "
            r"R^\theta{}_{t \theta t} = -\frac{M}{r^3}",
            font_size=20, color=ORANGE,
        )
        schwarz_eq.to_corner(DOWN + RIGHT, buff=0.3)
        self.play(FadeIn(schwarz_eq), run_time=0.8)                       # 60

        self.wait(1.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1,
        )                                                                 # 61

        card_title = Text(
            "Geodesic Deviation", font_size=34, color=GOLD,
        )
        card_title.to_edge(UP, buff=0.4)
        self.play(Write(card_title))                                      # 62

        lines = VGroup(
            MathTex(
                r"\ddot{\xi}^\mu = "
                r"-R^\mu{}_{\nu\rho\sigma}\,"
                r"u^\nu \xi^\rho u^\sigma"
                r"\;\;\text{(geodesic deviation equation)}",
                font_size=18,
            ),
            MathTex(
                r"\text{A ball of particles} \;\to\; "
                r"\text{ellipse: stretch radially, squeeze laterally}",
                font_size=18,
            ),
            MathTex(
                r"\text{Schwarzschild: radial } +2M/r^3, "
                r"\text{ lateral } -M/r^3",
                font_size=18,
            ),
            MathTex(
                r"r \to r_s: \;\text{tidal forces diverge} "
                r"\;\to\; \text{spaghettification}",
                font_size=18, color=RED,
            ),
            MathTex(
                r"\text{Riemann tensor } R^\mu{}_{\nu\rho\sigma}"
                r"\;\leftrightarrow\;\text{physical tidal forces}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{Flat space: } R = 0 "
                r"\;\Rightarrow\; \text{ball stays circular forever}",
                font_size=18, color=ORANGE,
            ),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        lines.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, lines),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.play(FadeIn(box), run_time=0.5)                             # 63
        for line in lines:
            self.play(FadeIn(line), run_time=0.7)                         # 64-69

        self.wait(2)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(lines), FadeOut(box),
            run_time=1.5,
        )                                                                 # 70
