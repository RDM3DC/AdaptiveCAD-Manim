"""CMB Blackbody — Visualising the most perfect blackbody in physics.

Scenes:
  1. CMBBlackbody        — Build B(ν,T) from scratch, overlay FIRAS data,
                           show residuals, spectral distortions, SZ effect
  2. AlternativeFailures — Side-by-side comparison: CMB vs tired-light,
                           local dust, steady-state predictions

Run:
    manim -pqh examples/cmb_blackbody.py CMBBlackbody
    manim -pqh examples/cmb_blackbody.py AlternativeFailures
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
from manim import (
    Scene, VGroup,
    Axes, FunctionGraph, DashedLine,
    Text, MathTex, DecimalNumber, Tex,
    Rectangle, RoundedRectangle, Line, Arrow, Dot,
    FadeIn, FadeOut, Create, Write, Transform,
    ReplacementTransform, AnimationGroup,
    Indicate, Flash,
    LEFT, RIGHT, UP, DOWN, ORIGIN, UL, UR, DL, DR,
    BLUE, RED, GREEN, YELLOW, WHITE, GREY, ORANGE, PURPLE,
    GOLD, TEAL, PINK, MAROON, GREY_A, GREY_B, GREY_C, GREY_D,
    rate_functions, config,
    color as mcolor,
)

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants (SI)
# ═══════════════════════════════════════════════════════════════════════════

H_PLANCK = 6.62607015e-34
K_B      = 1.380649e-23
C_SI     = 2.99792458e8
T_CMB    = 2.7255
M_E_C2   = 9.1093837e-31 * C_SI**2

# ═══════════════════════════════════════════════════════════════════════════
# Spectrum functions
# ═══════════════════════════════════════════════════════════════════════════

def planck_intensity(nu_ghz: float, T: float) -> float:
    """B(ν, T) in MJy/sr.  nu_ghz in GHz."""
    nu = nu_ghz * 1e9
    x = H_PLANCK * nu / (K_B * T)
    if x > 500:
        return 0.0
    B = (2 * H_PLANCK * nu**3 / C_SI**2) / (math.expm1(x))
    return B * 1e20  # W/m²/sr/Hz → MJy/sr

def planck_array(nu_arr, T):
    return np.array([planck_intensity(n, T) for n in nu_arr])

def mu_distorted(nu_ghz, T, mu):
    """Bose-Einstein spectrum with chemical potential μ."""
    nu = nu_ghz * 1e9
    x = H_PLANCK * nu / (K_B * T)
    if x > 500:
        return 0.0
    B = (2 * H_PLANCK * nu**3 / C_SI**2) / (math.exp(x + mu) - 1)
    return B * 1e20

def y_distorted_delta(nu_ghz, T, y_param):
    """Fractional SZ / y-distortion: ΔB at frequency ν."""
    nu = nu_ghz * 1e9
    x = H_PLANCK * nu / (K_B * T)
    if x > 30:
        return 0.0
    ex = math.exp(x)
    g_x = x * (ex + 1) / (ex - 1) - 4
    B0 = planck_intensity(nu_ghz, T)
    return B0 * y_param * g_x

def sz_spectral(nu_ghz, T):
    """Normalised SZ spectral function g(x)."""
    nu = nu_ghz * 1e9
    x = H_PLANCK * nu / (K_B * T)
    if x > 30:
        return x - 4
    ex = math.exp(x)
    return x * (ex + 1) / (ex - 1) - 4


# ═══════════════════════════════════════════════════════════════════════════
# FIRAS "data" — representative channel frequencies + perfect blackbody values
# The actual FIRAS residuals are < 50 ppm of peak; at plot scale they sit
# exactly on the Planck curve, which is the whole point.
# ═══════════════════════════════════════════════════════════════════════════

FIRAS_FREQS = np.array([
    68, 80, 95, 110, 125, 140, 155, 170, 185, 200,
    217, 235, 250, 270, 290, 310, 330, 350, 375,
    400, 425, 450, 480, 510, 540, 570, 600,
])
FIRAS_INTENSITIES = planck_array(FIRAS_FREQS, T_CMB)
# Add tiny Gaussian scatter ±0.3 MJy/sr to simulate error bars
# (actual FIRAS error bars are ~0.1 MJy/sr at peak frequencies)
np.random.seed(42)
FIRAS_SCATTER = np.random.normal(0, 0.3, len(FIRAS_FREQS))
FIRAS_DATA = FIRAS_INTENSITIES + FIRAS_SCATTER

# Wien peak
NU_PEAK = 2.821 * K_B * T_CMB / H_PLANCK / 1e9  # ~160 GHz
B_PEAK = planck_intensity(NU_PEAK, T_CMB)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 1: CMBBlackbody
# ═══════════════════════════════════════════════════════════════════════════

class CMBBlackbody(Scene):
    """The most perfect blackbody in physics — built from first principles."""

    def construct(self):
        # ── Act 0: Title Card ────────────────────────────────────────────
        title = Text("The Most Perfect Blackbody in Physics",
                      font_size=38, color=WHITE)
        subtitle = Text("COBE/FIRAS measurement of the CMB spectrum",
                         font_size=22, color=GREY_A)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Act 1: Build Planck curve ────────────────────────────────────
        axes = Axes(
            x_range=[0, 650, 100],
            y_range=[0, 420, 100],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "font_size": 20},
            x_axis_config={"numbers_to_include": [100, 200, 300, 400, 500, 600]},
            y_axis_config={"numbers_to_include": [100, 200, 300, 400]},
        ).shift(DOWN * 0.3 + LEFT * 0.3)

        x_label = axes.get_x_axis_label(
            MathTex(r"\nu \;\text{(GHz)}", font_size=24), direction=DOWN
        )
        y_label = axes.get_y_axis_label(
            MathTex(r"B(\nu)\;\text{(MJy/sr)}", font_size=24),
            direction=LEFT, buff=0.3,
        )

        eq_planck = MathTex(
            r"B(\nu, T) = \frac{2h\nu^3}{c^2}"
            r"\;\frac{1}{e^{h\nu/kT} - 1}",
            font_size=28,
        ).to_corner(UR, buff=0.4)

        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1.5)
        self.play(Write(eq_planck))

        # Planck curve
        planck_curve = axes.plot(
            lambda nu: planck_intensity(nu, T_CMB) if nu > 5 else 0,
            x_range=[5, 640, 1],
            color=YELLOW,
            stroke_width=3,
        )
        planck_label = MathTex(
            r"T = 2.7255\;\text{K}", font_size=22, color=YELLOW,
        ).next_to(eq_planck, DOWN, buff=0.3)

        self.play(Create(planck_curve), FadeIn(planck_label), run_time=3)

        # Wien peak marker
        peak_dot = Dot(
            axes.c2p(NU_PEAK, B_PEAK),
            color=RED, radius=0.06,
        )
        peak_text = MathTex(
            r"\nu_{\mathrm{peak}} = 160\;\text{GHz}", font_size=18, color=RED,
        ).next_to(peak_dot, UP + RIGHT, buff=0.15)
        self.play(FadeIn(peak_dot), Write(peak_text))
        self.wait(1)

        # ── Act 2: Overlay FIRAS data ────────────────────────────────────
        firas_title = Text("COBE/FIRAS Data (1996)", font_size=20, color=TEAL)
        firas_title.next_to(planck_label, DOWN, buff=0.3)

        # Data points
        firas_dots = VGroup()
        for freq, intensity in zip(FIRAS_FREQS, FIRAS_DATA):
            dot = Dot(
                axes.c2p(freq, max(0, intensity)),
                color=TEAL, radius=0.045,
            )
            firas_dots.add(dot)

        # Error bars (±5 MJy/sr visual — exaggerated 50× for visibility)
        error_bars = VGroup()
        for freq, intensity in zip(FIRAS_FREQS, FIRAS_DATA):
            err = 5  # visual error bar size
            top = axes.c2p(freq, min(420, intensity + err))
            bot = axes.c2p(freq, max(0, intensity - err))
            bar = Line(bot, top, stroke_width=1, color=TEAL)
            error_bars.add(bar)

        self.play(Write(firas_title))
        self.play(
            AnimationGroup(
                *[FadeIn(d, scale=2) for d in firas_dots],
                lag_ratio=0.03,
            ),
            FadeIn(error_bars),
            run_time=2,
        )
        self.wait(1)

        # ── Precision callout ────────────────────────────────────────────
        precision_box = RoundedRectangle(
            width=4.8, height=1.6, corner_radius=0.15,
            fill_color="#0a0a2e", fill_opacity=0.9,
            stroke_color=TEAL, stroke_width=1.5,
        ).to_corner(DL, buff=0.3)

        precision_text = VGroup(
            Text("Residual from perfect blackbody:", font_size=16, color=WHITE),
            MathTex(r"< 50 \;\text{parts per million}", font_size=22, color=TEAL),
            Text("No lab source achieves this precision", font_size=14, color=GREY_A),
        ).arrange(DOWN, buff=0.12).move_to(precision_box.get_center())

        self.play(FadeIn(precision_box), Write(precision_text), run_time=1.5)
        self.wait(2)

        # ── Act 3: What distortions would look like ──────────────────────
        self.play(FadeOut(precision_box), FadeOut(precision_text),
                  FadeOut(peak_dot), FadeOut(peak_text))

        # μ-distortion (exaggerated to μ=0.1 for visibility; real limit is 9×10⁻⁵)
        mu_curve = axes.plot(
            lambda nu: mu_distorted(nu, T_CMB, 0.1) if nu > 5 else 0,
            x_range=[5, 640, 1],
            color=RED,
            stroke_width=2,
        )
        mu_label = MathTex(
            r"\mu\text{-distortion}\;(\mu = 0.1)", font_size=18, color=RED,
        ).move_to(axes.c2p(450, 250))

        firas_limit = MathTex(
            r"\text{FIRAS limit: } |\mu| < 9\times 10^{-5}",
            font_size=16, color=ORANGE,
        ).next_to(mu_label, DOWN, buff=0.2)

        self.play(Create(mu_curve), Write(mu_label), run_time=2)
        self.play(Write(firas_limit))
        self.wait(2)

        # Explain: real distortion is 1000× smaller than plotted
        invisible_note = Text(
            "Actual limit is 1000× smaller — invisible at this scale",
            font_size=14, color=GREY_A,
        ).next_to(firas_limit, DOWN, buff=0.15)
        self.play(FadeIn(invisible_note))
        self.wait(1.5)

        self.play(FadeOut(mu_curve), FadeOut(mu_label),
                  FadeOut(firas_limit), FadeOut(invisible_note))

        # ── Act 4: SZ effect spectral shape ──────────────────────────────
        # Show the SZ decrement/increment pattern
        sz_title = Text(
            "Sunyaev-Zel'dovich Effect — CMB behind galaxy clusters",
            font_size=18, color=GOLD,
        ).to_corner(DL, buff=0.4)

        # Plot g(x) scaled for visibility
        sz_scale = 40  # visual scale factor
        sz_curve = axes.plot(
            lambda nu: 200 + sz_scale * sz_spectral(nu, T_CMB) if nu > 20 else 200,
            x_range=[20, 640, 1],
            color=GOLD,
            stroke_width=2.5,
        )

        # Baseline at 200 MJy/sr
        sz_baseline = DashedLine(
            axes.c2p(20, 200), axes.c2p(640, 200),
            stroke_width=1, color=GREY,
        )

        # Crossover at 217 GHz
        crossover_line = DashedLine(
            axes.c2p(217, 120), axes.c2p(217, 280),
            stroke_width=1, color=GREY,
        )
        co_label = MathTex(
            r"217\;\text{GHz}", font_size=14, color=GREY_A,
        ).next_to(crossover_line, DOWN, buff=0.1)

        # Decrement / increment labels
        dec_label = Text("decrement", font_size=14, color=BLUE).move_to(axes.c2p(120, 160))
        inc_label = Text("increment", font_size=14, color=RED).move_to(axes.c2p(400, 280))

        self.play(Write(sz_title), Create(sz_baseline), run_time=1)
        self.play(Create(sz_curve), run_time=2)
        self.play(Create(crossover_line), Write(co_label),
                  Write(dec_label), Write(inc_label))

        sz_evidence = Text(
            "Detected in 1600+ clusters → CMB is at cosmological distance",
            font_size=14, color=GOLD,
        ).next_to(sz_title, DOWN, buff=0.15)
        self.play(Write(sz_evidence))
        self.wait(2.5)

        # Clean up SZ
        self.play(
            *[FadeOut(m) for m in [
                sz_curve, sz_baseline, crossover_line, co_label,
                dec_label, inc_label, sz_title, sz_evidence,
            ]]
        )

        # ── Act 5: Temperature comparison ───────────────────────────────
        # Show what the CMB would look like at different temperatures
        temp_text = Text(
            "The universe at different temperatures",
            font_size=18, color=WHITE,
        ).to_corner(DL, buff=0.4)
        self.play(Write(temp_text))

        colors_t = [BLUE, PURPLE, RED]
        temps = [2.0, 2.725, 4.0]
        labels_t = [r"T=2.0\;\text{K}", r"T=2.725\;\text{K}", r"T=4.0\;\text{K}"]
        curves_t = []

        for T_val, col, lab in zip(temps, colors_t, labels_t):
            c = axes.plot(
                lambda nu, _T=T_val: planck_intensity(nu, _T) if nu > 5 else 0,
                x_range=[5, 640, 1],
                color=col,
                stroke_width=2 if T_val != T_CMB else 3,
            )
            curves_t.append(c)

        # Show all three simultaneously
        legend_items = VGroup()
        for col, lab in zip(colors_t, labels_t):
            item = VGroup(
                Dot(color=col, radius=0.04),
                MathTex(lab, font_size=16, color=col),
            ).arrange(RIGHT, buff=0.1)
            legend_items.add(item)
        legend_items.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend_items.next_to(eq_planck, DOWN, buff=0.5)

        self.play(
            *[Create(c) for c in curves_t],
            FadeIn(legend_items),
            run_time=2,
        )
        self.wait(2)

        # Remove temperature curves except the CMB one
        self.play(
            FadeOut(curves_t[0]), FadeOut(curves_t[2]),
            FadeOut(legend_items), FadeOut(temp_text),
        )

        # ── Act 6: Summary Card ──────────────────────────────────────────
        summary_box = RoundedRectangle(
            width=5, height=3, corner_radius=0.15,
            fill_color="#0a0a2e", fill_opacity=0.95,
            stroke_color=YELLOW, stroke_width=1.5,
        ).to_corner(DL, buff=0.25)

        summary = VGroup(
            Text("CMB Blackbody — Key Results", font_size=18, color=YELLOW),
            Text("", font_size=6),
            Text("T = 2.7255 ± 0.0006 K", font_size=14, color=WHITE),
            Text("Peak: 160 GHz  (1.06 mm)", font_size=14, color=WHITE),
            Text("Residual: < 50 parts per million", font_size=14, color=TEAL),
            Text("411 photons/cm³", font_size=14, color=WHITE),
            Text("1.6 billion photons per baryon", font_size=14, color=WHITE),
            Text("", font_size=6),
            Text("SZ effect in 1600+ clusters", font_size=14, color=GOLD),
            Text("Most perfect blackbody ever measured", font_size=14, color=ORANGE),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT).move_to(summary_box.get_center())

        self.play(FadeIn(summary_box), Write(summary), run_time=2)
        self.wait(3)

        # Final fadeout
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 2: Alternative Failures
# ═══════════════════════════════════════════════════════════════════════════

class AlternativeFailures(Scene):
    """Show why alternative CMB origins fail quantitative tests."""

    def construct(self):
        # ── Title ────────────────────────────────────────────────────────
        title = Text("Can anything else make this blackbody?",
                      font_size=34, color=WHITE)
        subtitle = Text("Three alternatives tested against the data",
                         font_size=20, color=GREY_A)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Axes for spectrum comparison ─────────────────────────────────
        axes = Axes(
            x_range=[0, 650, 100],
            y_range=[0, 420, 100],
            x_length=8.5,
            y_length=4.5,
            axis_config={"include_numbers": True, "font_size": 18},
            x_axis_config={"numbers_to_include": [100, 200, 300, 400, 500, 600]},
            y_axis_config={"numbers_to_include": [100, 200, 300, 400]},
        ).shift(DOWN * 0.5 + LEFT * 0.3)

        x_label = axes.get_x_axis_label(
            MathTex(r"\nu \;\text{(GHz)}", font_size=20), direction=DOWN,
        )
        y_label = axes.get_y_axis_label(
            MathTex(r"\text{MJy/sr}", font_size=20), direction=LEFT, buff=0.2,
        )

        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1)

        # CMB Planck curve (ground truth)
        cmb_curve = axes.plot(
            lambda nu: planck_intensity(nu, T_CMB) if nu > 5 else 0,
            x_range=[5, 640, 1],
            color=YELLOW, stroke_width=3,
        )
        cmb_label = Text("CMB (T = 2.725 K)", font_size=16, color=YELLOW)
        cmb_label.to_corner(UR, buff=0.4)

        self.play(Create(cmb_curve), Write(cmb_label), run_time=1.5)

        # ── Test 1: Tired Light ──────────────────────────────────────────
        test_title = Text(
            "Test 1: Tired Light — Thermalized Starlight",
            font_size=20, color=RED,
        ).to_corner(UL, buff=0.4)
        self.play(Write(test_title))

        # Tired light would produce a composite spectrum — sum of stellar
        # blackbodies at various redshifts. Approximate as a broad hump.
        # Key failure: can't produce single-T blackbody to 50 ppm.
        def tired_light_approx(nu):
            """Rough composite starlight: several temperatures blended."""
            b1 = planck_intensity(nu, 3.5) * 0.3
            b2 = planck_intensity(nu, 2.5) * 0.5
            b3 = planck_intensity(nu, 2.0) * 0.2
            return (b1 + b2 + b3) * 0.07  # × energy deficit factor

        tired_curve = axes.plot(
            lambda nu: tired_light_approx(nu) if nu > 5 else 0,
            x_range=[5, 640, 1],
            color=RED, stroke_width=2,
        )
        tired_label = Text("Tired light prediction", font_size=14, color=RED)
        tired_label.move_to(axes.c2p(500, 50))

        self.play(Create(tired_curve), Write(tired_label), run_time=2)

        fail_1 = VGroup(
            Text("FAILS:", font_size=16, color=RED),
            Text("• Wrong shape (composite, not single-T BB)", font_size=13),
            Text("• 14× too little energy", font_size=13),
            Text("• No SN time dilation", font_size=13),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        fail_1.to_corner(DL, buff=0.3)
        self.play(Write(fail_1), run_time=1.5)
        self.wait(2)

        self.play(FadeOut(tired_curve), FadeOut(tired_label),
                  FadeOut(fail_1), FadeOut(test_title))

        # ── Test 2: Local Dust ───────────────────────────────────────────
        test_title2 = Text(
            "Test 2: Local Dust — Galactic Thermal Emission",
            font_size=20, color=ORANGE,
        ).to_corner(UL, buff=0.4)
        self.play(Write(test_title2))

        # Dust emission follows modified blackbody: B(ν,T) × ν^β
        # with T ~ 20K for ISM dust — completely wrong shape
        def dust_emission(nu):
            """Modified blackbody at T=20K with emissivity index β=1.5."""
            T_dust = 20.0
            beta = 1.5
            B = planck_intensity(nu, T_dust) if nu > 5 else 0
            ref_nu = 100  # normalise at 100 GHz
            return B * (max(nu, 1) / ref_nu)**beta * 0.001  # scaled down

        dust_curve = axes.plot(
            lambda nu: min(dust_emission(nu), 400) if nu > 5 else 0,
            x_range=[5, 640, 1],
            color=ORANGE, stroke_width=2,
        )
        dust_label = Text("Galactic dust (T≈20K)", font_size=14, color=ORANGE)
        dust_label.move_to(axes.c2p(500, 300))

        self.play(Create(dust_curve), Write(dust_label), run_time=2)

        fail_2 = VGroup(
            Text("FAILS:", font_size=16, color=ORANGE),
            Text("• Wrong temperature (20K, not 2.725K)", font_size=13),
            Text("• Not isotropic (follows Galactic plane)", font_size=13),
            Text("• Galaxy optically thin at 100 GHz (τ < 0.01)", font_size=13),
            Text("• SZ effect proves CMB is behind clusters", font_size=13),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        fail_2.to_corner(DL, buff=0.3)
        self.play(Write(fail_2), run_time=1.5)
        self.wait(2)

        self.play(FadeOut(dust_curve), FadeOut(dust_label),
                  FadeOut(fail_2), FadeOut(test_title2))

        # ── Test 3: Show the real answer ─────────────────────────────────
        test_title3 = Text(
            "Result: Only an optically thick thermal source works",
            font_size=20, color=GREEN,
        ).to_corner(UL, buff=0.4)
        self.play(Write(test_title3))

        # Re-overlay FIRAS data
        firas_dots = VGroup()
        for freq, intensity in zip(FIRAS_FREQS, FIRAS_DATA):
            dot = Dot(
                axes.c2p(freq, max(0, intensity)),
                color=TEAL, radius=0.04,
            )
            firas_dots.add(dot)

        self.play(
            AnimationGroup(
                *[FadeIn(d, scale=2) for d in firas_dots],
                lag_ratio=0.02,
            ),
            run_time=1.5,
        )

        # The core conclusion
        conclusion_box = RoundedRectangle(
            width=6.5, height=2.8, corner_radius=0.15,
            fill_color="#0a0a2e", fill_opacity=0.95,
            stroke_color=GREEN, stroke_width=1.5,
        ).to_corner(DL, buff=0.2)

        conclusion = VGroup(
            Text("The Core Tension", font_size=18, color=GREEN),
            Text("", font_size=4),
            Text("A blackbody proves thermal equilibrium,", font_size=14, color=WHITE),
            Text("not origin. Any opaque 2.725K body works.", font_size=14, color=WHITE),
            Text("", font_size=4),
            Text("But NO alternative explains ALL of:", font_size=14, color=YELLOW),
            Text("  Blackbody + Isotropy + Acoustic peaks", font_size=13, color=GREY_A),
            Text("  + SZ effect + BBN match + BAO", font_size=13, color=GREY_A),
            Text("", font_size=4),
            Text("One model, 6 parameters, all data.", font_size=14, color=GOLD),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        conclusion.move_to(conclusion_box.get_center())

        self.play(FadeIn(conclusion_box), Write(conclusion), run_time=2)
        self.wait(4)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
