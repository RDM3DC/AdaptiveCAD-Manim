"""Quadrupole Radiation & Binary Inspiral — Post-Newtonian Chirp.

Quadrupole gravitational-wave emission drains orbital energy:

    dE/dt = −(32/5) (G⁴/c⁵) (M₁M₂)² (M₁+M₂) / a⁵

or equivalently in terms of the chirp mass ℳ and orbital angular
frequency ω:

    dE/dt ∝ (G ℳ/c³)^{5/3} ω^{10/3}

This drives the orbit to shrink (the "chirp"), the frequency to ramp
up, and ultimately to the final plunge.

The animation shows:
- Two masses M₁, M₂ orbiting in a tightening dashed spiral
- Blue-to-gold colour shift as the orbit contracts
- Live GW strain waveform h(t) below the orbit panel
- Frequency & energy-loss counters
- Summary card with the key post-Newtonian formulae

Acts
----
1. Title card with quadrupole formula
2. Build orbit panel (top) + waveform panel (bottom)
3. Animate inspiral: shrinking dashed orbit, accelerating masses,
   waveform growing in frequency/amplitude, blue→gold hue shift
4. Final plunge / merger flash
5. Summary card

Run
---
    manim -pql examples/quadrupole_inspiral.py QuadrupoleInspiral
    manim -qh  examples/quadrupole_inspiral.py QuadrupoleInspiral
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Dot,
    DashedVMobject,
    Circle,
    ParametricFunction,
    VMobject,
    VGroup,
    Axes,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Flash,
    SurroundingRectangle,
    ValueTracker,
    DecimalNumber,
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
    ORANGE,
    GOLD,
    TEAL,
    BLUE,
    BLUE_D,
    GREY_A,
    GREY_D,
    GREEN,
    interpolate_color,
    rate_functions,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Physics (geometric / scaled units)
# ═══════════════════════════════════════════════════════════════════════════
M1 = 1.4          # solar-mass units (neutron star)
M2 = 1.4
M_TOT = M1 + M2
ETA = M1 * M2 / M_TOT**2                    # symmetric mass ratio
M_CHIRP = M_TOT * ETA**(3 / 5)              # chirp mass

A_INIT = 2.8      # initial orbital separation (display units)
A_MIN = 0.35      # merger separation

# Post-Newtonian inspiral: a(t) shrinks as
#   a(t) = a_0 (1 − t/t_c)^{1/4}
# We parameterise by τ = t/t_c ∈ [0, 1).

ORBIT_PANEL_Y = 1.5     # centre of orbit panel
WAVE_PANEL_Y = -2.2      # centre of waveform panel
WAVE_PANEL_W = 6.0       # waveform width
WAVE_PANEL_H = 1.2       # waveform height

N_TRAIL_PTS = 300        # trail history for dashed orbit
CHIRP_DURATION = 10.0    # seconds of animation for the inspiral
TOTAL_GW_CYCLES = 18     # approximate total GW cycles


def _sep(tau):
    """Orbital separation a(τ) with τ ∈ [0, 1)."""
    return A_INIT * max(1 - tau, 1e-4) ** 0.25


def _omega(tau):
    """Orbital angular frequency ω ∝ a^{−3/2}."""
    a = _sep(tau)
    return (A_INIT / a) ** 1.5


def _gw_phase(tau):
    """Accumulated GW phase (twice orbital phase)."""
    # Integrated from the PN inspiral:
    # Φ_GW ∝ (1−τ)^{5/8}  →  total TOTAL_GW_CYCLES * 2π
    s = max(1 - tau, 1e-4)
    return TOTAL_GW_CYCLES * TAU * (1.0 - s ** (5 / 8))


def _gw_amplitude(tau):
    """GW strain amplitude ∝ ω^{2/3} ∝ a^{−1}."""
    a = _sep(tau)
    return 0.15 * A_INIT / a   # normalised so it starts small


def _orbit_color(tau):
    """Blue → Gold colour ramp as orbit tightens."""
    return interpolate_color(BLUE_D, GOLD, min(tau * 1.1, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class QuadrupoleInspiral(ThreeDScene):
    """Post-Newtonian binary inspiral with quadrupole radiation."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # ─── Act 1  Title ────────────────────────────────────────────────
        ttl = Text("Quadrupole Radiation  —  Binary Inspiral",
                    font_size=38, color=GOLD)
        sub = Text("Post-Newtonian chirp & orbital decay",
                    font_size=20, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"\frac{dE}{dt}",
            r"\propto",
            r"\left(\frac{G\,\mathcal{M}}{c^3}\right)^{\!5/3}",
            r"\omega^{10/3}",
            font_size=24, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1.2)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Build panels ────────────────────────────────────────
        # --- Orbit panel (upper half) ---
        orbit_ctr = np.array([0, ORBIT_PANEL_Y, 0])

        # Mass labels
        m1_lbl = MathTex(r"M_1", font_size=16, color=ORANGE)
        m2_lbl = MathTex(r"M_2", font_size=16, color=TEAL)
        mass_info = MathTex(
            r"M_1 = M_2 = 1.4\,M_\odot",
            font_size=14, color=GREY_A,
        )
        mass_info.move_to([3.2, ORBIT_PANEL_Y + 1.5, 0])
        chirp_info = MathTex(
            r"\mathcal{M} = " + f"{M_CHIRP:.2f}" + r"\,M_\odot",
            font_size=14, color=YELLOW,
        )
        chirp_info.next_to(mass_info, DOWN, buff=0.15)

        # Two mass dots
        dot1 = Dot(radius=0.10, color=ORANGE)
        dot2 = Dot(radius=0.10, color=TEAL)

        # Smooth orbit trail (cubic Bézier via set_points_smoothly)
        trail = VMobject(stroke_width=1.8, stroke_opacity=0.6)
        trail.set_points_smoothly([orbit_ctr, orbit_ctr + RIGHT * 0.01])

        # --- Waveform panel (lower half) ---
        wave_axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            x_length=WAVE_PANEL_W,
            y_length=WAVE_PANEL_H,
            axis_config={"stroke_width": 1, "color": GREY_D,
                         "include_ticks": False},
        )
        wave_axes.move_to([0, WAVE_PANEL_Y, 0])
        wave_lbl = Text("h(t)  strain", font_size=12, color=GREY_A)
        wave_lbl.next_to(wave_axes, UP, buff=0.1)
        wave_lbl.align_to(wave_axes, LEFT)

        # Waveform curve — ParametricFunction from closed-form h(τ)
        wave_curve = VMobject(stroke_width=1.8, color=GOLD)

        # Counters
        freq_counter = VGroup(
            MathTex(r"f_{\rm GW}=", font_size=14, color=YELLOW),
            DecimalNumber(0, num_decimal_places=1,
                          font_size=14, color=YELLOW),
            MathTex(r"\;\text{Hz}", font_size=14, color=YELLOW),
        ).arrange(RIGHT, buff=0.05)
        freq_counter.move_to([-2.5, ORBIT_PANEL_Y + 1.5, 0])

        sep_counter = VGroup(
            MathTex(r"a=", font_size=14, color=BLUE_D),
            DecimalNumber(A_INIT, num_decimal_places=2,
                          font_size=14, color=BLUE_D),
        ).arrange(RIGHT, buff=0.05)
        sep_counter.next_to(freq_counter, DOWN, buff=0.12)

        for m in [mass_info, chirp_info, wave_lbl, m1_lbl, m2_lbl,
                  freq_counter, sep_counter]:
            self.add_fixed_in_frame_mobjects(m)

        self.play(
            FadeIn(dot1), FadeIn(dot2),
            FadeIn(mass_info), FadeIn(chirp_info),
            Create(wave_axes), FadeIn(wave_lbl),
            FadeIn(freq_counter), FadeIn(sep_counter),
            run_time=1.2,
        )
        self.wait(0.3)

        # ─── Act 3  Animate inspiral ────────────────────────────────────
        tracker = ValueTracker(0)  # τ from 0 to 1

        # Storage for trail points
        trail_pts = []

        def _updater(mob, dt=None):
            tau = tracker.get_value()
            a = _sep(tau)
            phi_gw = _gw_phase(tau)
            phi_orb = phi_gw / 2  # orbital phase = half GW phase
            amp = _gw_amplitude(tau)
            col = _orbit_color(tau)

            # ------ Masses ------
            r1 = a * M2 / M_TOT
            r2 = a * M1 / M_TOT
            x1 = orbit_ctr[0] + r1 * np.cos(phi_orb)
            y1 = orbit_ctr[1] + r1 * np.sin(phi_orb)
            x2 = orbit_ctr[0] - r2 * np.cos(phi_orb)
            y2 = orbit_ctr[1] - r2 * np.sin(phi_orb)
            dot1.move_to([x1, y1, 0])
            dot2.move_to([x2, y2, 0])
            dot1.set_color(col)
            dot2.set_color(interpolate_color(TEAL, GOLD,
                                              min(tau * 1.1, 1.0)))

            # Labels follow dots
            m1_lbl.move_to([x1, y1 + 0.18, 0])
            m2_lbl.move_to([x2, y2 + 0.18, 0])

            # ------ Smooth orbit trail ------
            trail_pts.append(np.array([x1, y1, 0]))
            if len(trail_pts) > N_TRAIL_PTS:
                del trail_pts[:-N_TRAIL_PTS]
            if len(trail_pts) > 4:
                # Downsample for smooth Bézier interpolation
                step = max(1, len(trail_pts) // 80)
                smooth_pts = trail_pts[::step]
                if smooth_pts[-1] is not trail_pts[-1]:
                    smooth_pts.append(trail_pts[-1])
                trail.set_points_smoothly(smooth_pts)
                trail.set_stroke(color=col, opacity=0.45)

            # ------ Waveform (ParametricFunction) ------
            # Rebuild a smooth parametric waveform up to current τ
            tau_now = tau
            if tau_now > 0.01:
                def _wave_func(t, _tn=tau_now, _axes=wave_axes):
                    tt = t * _tn  # map [0,1] → [0, τ_now]
                    ph = _gw_phase(tt)
                    am = _gw_amplitude(tt)
                    val = am * np.cos(ph)
                    px = _axes.c2p(tt, 0)[0]
                    py = _axes.c2p(0, np.clip(val, -0.95, 0.95))[1]
                    return np.array([px, py, 0])
                new_wave = ParametricFunction(
                    _wave_func, t_range=[0, 1, 0.005],
                    color=col, stroke_width=1.6,
                )
                wave_curve.become(new_wave)

            # ------ Counters ------
            # f_GW = ω / π  (normalised display value)
            omega = _omega(tau)
            freq_counter[1].set_value(omega * 10)  # arbitrary scale
            sep_counter[1].set_value(a)

        # Add everything to scene
        self.add(trail, wave_curve)
        dot1.add_updater(_updater)

        self.play(
            tracker.animate.set_value(0.96),
            run_time=CHIRP_DURATION,
            rate_func=lambda t: t,
        )
        dot1.remove_updater(_updater)
        self.wait(0.2)

        # ─── Act 4  Merger flash ────────────────────────────────────────
        merger_dot = Dot(
            point=[orbit_ctr[0], orbit_ctr[1], 0],
            radius=0.15, color=WHITE,
        )
        self.play(FadeIn(merger_dot), run_time=0.15)
        self.play(
            Flash(merger_dot, color=GOLD, line_length=0.6,
                  num_lines=16, flash_radius=1.2,
                  run_time=0.8),
        )

        merge_lbl = Text("MERGER", font_size=24, color=GOLD)
        merge_lbl.move_to([0, ORBIT_PANEL_Y, 0])
        self.add_fixed_in_frame_mobjects(merge_lbl)
        self.play(FadeIn(merge_lbl), run_time=0.4)
        self.wait(0.8)

        # ─── Act 5  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in [
                mass_info, chirp_info, wave_lbl, m1_lbl, m2_lbl,
                freq_counter, sep_counter, merge_lbl,
            ]],
            run_time=0.8,
        )

        card = Text("Quadrupole Radiation", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\frac{dE}{dt}", r"\propto",
                r"\left(\frac{G\mathcal{M}}{c^3}\right)^{5/3}",
                r"\omega^{10/3}",
                font_size=19,
            ),
            MathTex(
                r"a(t)", r"=", r"a_0\,(1 - t/t_c)^{1/4}",
                font_size=19, color=BLUE_D,
            ),
            MathTex(
                r"f_{\rm GW}", r"\propto",
                r"(t_c - t)^{-3/8}",
                r"\;\;\text{(chirp)}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\mathcal{M}", r"=",
                r"(M_1 M_2)^{3/5}/(M_1+M_2)^{1/5}",
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
