"""Schwarzschild Radial Infall — Proper-Time Freeze at the Horizon.

A test particle falls radially into a Schwarzschild black hole.

Metric:  ds² = −(1 − rₛ/r) dt² + (1 − rₛ/r)⁻¹ dr² + r² dΩ²

Key physics:
  • Coordinate time t → ∞ as r → rₛ  (distant observer sees freeze)
  • Proper time τ stays finite through the horizon
  • Redshift glow:  1 + z = (1 − rₛ/r)^{−1/2}  → colour shifts red

Acts
----
1. Title card with Schwarzschild metric
2. Draw BH (dark disk) + horizon circle + radial coordinate axis
3. Particle falls inward; coordinate-time clock slows, redshift grows
4. Glow colour shifts blue → yellow → orange → deep red near horizon
5. Dual clock display: τ (proper) vs t (coordinate)
6. Summary card

Run
---
    manim -pql examples/schwarzschild_infall.py SchwarzschildInfall
    manim -qh  examples/schwarzschild_infall.py SchwarzschildInfall
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
    Circle,
    Annulus,
    ParametricFunction,
    VGroup,
    MathTex,
    Text,
    DecimalNumber,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Transform,
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
    BLACK,
    interpolate_color,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Physics
# ═══════════════════════════════════════════════════════════════════════════
RS = 1.0          # Schwarzschild radius (display units)
R_START = 5.0     # starting r/rₛ  (in units of rₛ)
N_STEPS = 50      # animation steps in the infall


def _redshift_color(r_over_rs):
    """Map r/rₛ → colour.  Far = blue-white, near horizon = deep red."""
    if r_over_rs <= 1.0:
        return RED_E
    frac = min(1.0, (r_over_rs - 1.0) / (R_START - 1.0))
    # frac=1 at start (far), frac=0 at horizon
    if frac > 0.66:
        return interpolate_color(BLUE, WHITE, (frac - 0.66) / 0.34)
    if frac > 0.33:
        return interpolate_color(YELLOW, BLUE, (frac - 0.33) / 0.33)
    if frac > 0.10:
        return interpolate_color(ORANGE, YELLOW, (frac - 0.10) / 0.23)
    return interpolate_color(RED_E, ORANGE, frac / 0.10)


def _proper_time_steps(r_start, rs, n):
    """Compute r(τ) for radial free-fall from rest at r_start.

    For a particle dropped from rest at r_start the relation is:
        (dr/dτ)² = rₛ/r − rₛ/r_start
    We integrate numerically.
    """
    r_vals = [r_start]
    tau_vals = [0.0]
    dr = (r_start - rs * 1.02) / n  # stop just outside horizon
    r = r_start
    tau = 0.0
    for _ in range(n):
        speed_sq = rs / r - rs / r_start
        if speed_sq <= 0:
            speed_sq = 1e-8
        dtau = dr / np.sqrt(speed_sq)
        tau += dtau
        r -= dr
        r_vals.append(r)
        tau_vals.append(tau)
    return np.array(r_vals), np.array(tau_vals)


def _coord_time_steps(r_vals, rs):
    """Approximate Schwarzschild coordinate time for each r step.

    dt/dr = −(1 − rₛ/r)⁻¹ (rₛ/r)^{−1/2}  (for free-fall from ∞ approx)
    We use the exact expression for E=1 (dropped from infinity):
        dt/dr = −√(rₛ/r) / (1 − rₛ/r)
    """
    t_vals = [0.0]
    t = 0.0
    for i in range(1, len(r_vals)):
        r = (r_vals[i] + r_vals[i - 1]) / 2
        dr = r_vals[i - 1] - r_vals[i]
        factor = 1 - rs / r
        if abs(factor) < 1e-6:
            factor = 1e-6
        dtdr = np.sqrt(rs / r) / factor
        t += dtdr * dr
        t_vals.append(t)
    return np.array(t_vals)


# Pre-compute trajectory
_r_vals, _tau_vals = _proper_time_steps(R_START * RS, RS, N_STEPS)
_t_vals = _coord_time_steps(_r_vals, RS)

# Display scale: map r to screen x (BH at origin)
DISPLAY_SCALE = 0.7  # screen units per rₛ


def _r_to_screen(r):
    return r * DISPLAY_SCALE


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class SchwarzschildInfall(ThreeDScene):
    """Radial infall into a Schwarzschild black hole."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # ─── Act 1  Title ────────────────────────────────────────────────
        ttl = Text("Schwarzschild Radial Infall",
                    font_size=44, color=GOLD)
        sub = Text("Proper-time freeze & gravitational redshift",
                    font_size=20, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"ds^2 = -\!\left(1 - \frac{r_s}{r}\right)dt^2"
            r" + \left(1 - \frac{r_s}{r}\right)^{\!-1}\!dr^2"
            r" + r^2\,d\Omega^2",
            font_size=22, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  BH + horizon + radial axis ──────────────────────────
        # Dark disk for the BH interior
        bh_radius = _r_to_screen(RS)

        bh_disk = Surface(
            lambda u, v: np.array([
                v * bh_radius * np.cos(u),
                v * bh_radius * np.sin(u),
                -0.01,
            ]),
            u_range=[0, TAU], v_range=[0, 1],
            resolution=(32, 4),
            fill_color=BLACK, fill_opacity=0.95,
            stroke_width=0,
        )

        horizon = ParametricFunction(
            lambda t: np.array([
                bh_radius * np.cos(t),
                bh_radius * np.sin(t), 0]),
            t_range=[0, TAU],
            color=RED, stroke_width=2.5,
        )

        # Radial axis going right
        r_axis = Arrow3D(
            ORIGIN, np.array([_r_to_screen(R_START + 0.5), 0, 0]),
            color=GREY_A, thickness=0.008,
        )

        # Tick marks at r/rₛ = 1, 2, 3, 4, 5
        ticks = VGroup()
        tick_lbls = []
        for rr in [1, 2, 3, 4, 5]:
            x = _r_to_screen(rr * RS)
            tick = Line3D(
                np.array([x, -0.08, 0]),
                np.array([x, 0.08, 0]),
                color=GREY_A, thickness=0.005,
            )
            ticks.add(tick)
            lbl = MathTex(str(rr), font_size=12, color=GREY_A)
            lbl.move_to(np.array([x, -0.25, 0]))
            tick_lbls.append(lbl)

        r_label = MathTex(r"r / r_s", font_size=14, color=GREY_A)
        r_label.move_to(np.array([_r_to_screen(R_START + 0.5), -0.4, 0]))
        horizon_lbl = Text("horizon", font_size=12, color=RED)
        horizon_lbl.move_to(np.array([0, bh_radius + 0.25, 0]))

        frame_lbls = [r_label, horizon_lbl] + tick_lbls
        for lbl in frame_lbls:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(bh_disk), Create(horizon),
            Create(r_axis), Create(ticks),
            *[FadeIn(lbl) for lbl in frame_lbls],
            run_time=1.2,
        )

        # ─── Act 3  Infall animation ────────────────────────────────────
        # Particle dot
        start_x = _r_to_screen(_r_vals[0])
        particle = Dot3D(
            np.array([start_x, 0, 0]),
            radius=0.10, color=BLUE,
        )
        glow = Dot3D(
            np.array([start_x, 0, 0]),
            radius=0.18, color=BLUE,
        )
        glow.set_opacity(0.3)

        self.play(FadeIn(particle), FadeIn(glow), run_time=0.5)

        # Clock displays
        tau_header = Text("proper τ", font_size=14, color=TEAL)
        tau_header.move_to(RIGHT * 4.5 + UP * 2.5)
        tau_num = DecimalNumber(0, num_decimal_places=2,
                                 font_size=20, color=TEAL)
        tau_num.next_to(tau_header, DOWN, buff=0.15)

        t_header = Text("coord t", font_size=14, color=ORANGE)
        t_header.move_to(RIGHT * 4.5 + UP * 1.3)
        t_num = DecimalNumber(0, num_decimal_places=1,
                               font_size=20, color=ORANGE)
        t_num.next_to(t_header, DOWN, buff=0.15)

        z_header = Text("redshift z", font_size=14, color=RED)
        z_header.move_to(RIGHT * 4.5 + UP * 0.1)
        z_num = DecimalNumber(0, num_decimal_places=2,
                               font_size=20, color=RED)
        z_num.next_to(z_header, DOWN, buff=0.15)

        for lbl in [tau_header, tau_num, t_header, t_num,
                     z_header, z_num]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(tau_header), FadeIn(tau_num),
            FadeIn(t_header), FadeIn(t_num),
            FadeIn(z_header), FadeIn(z_num),
            run_time=0.5,
        )

        # Animate infall step by step
        for i in range(1, N_STEPS + 1):
            r = _r_vals[i]
            x = _r_to_screen(r)
            r_over_rs = r / RS
            col = _redshift_color(r_over_rs)

            # Redshift
            factor = 1 - RS / r
            if factor < 1e-4:
                factor = 1e-4
            z_val = 1.0 / np.sqrt(factor) - 1.0

            new_particle = Dot3D(
                np.array([x, 0, 0]), radius=0.10, color=col,
            )
            # Glow grows as redshift increases
            glow_r = 0.18 + 0.15 * min(z_val / 5.0, 1.0)
            new_glow = Dot3D(
                np.array([x, 0, 0]),
                radius=glow_r, color=col,
            )
            new_glow.set_opacity(0.25)

            # Speed: slower near horizon (more frames per step)
            run = 0.08 + 0.22 * (1 - i / N_STEPS) ** 2

            new_tau = DecimalNumber(
                _tau_vals[i], num_decimal_places=2,
                font_size=20, color=TEAL,
            )
            new_tau.next_to(tau_header, DOWN, buff=0.15)
            new_t = DecimalNumber(
                min(_t_vals[i], 999.9), num_decimal_places=1,
                font_size=20, color=ORANGE,
            )
            new_t.next_to(t_header, DOWN, buff=0.15)
            new_z = DecimalNumber(
                min(z_val, 999.9), num_decimal_places=2,
                font_size=20, color=RED,
            )
            new_z.next_to(z_header, DOWN, buff=0.15)

            self.add_fixed_in_frame_mobjects(new_tau, new_t, new_z)

            self.play(
                Transform(particle, new_particle),
                Transform(glow, new_glow),
                Transform(tau_num, new_tau),
                Transform(t_num, new_t),
                Transform(z_num, new_z),
                run_time=run,
            )

        self.wait(0.8)

        # ─── Act 4  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in frame_lbls],
            *[FadeOut(lbl) for lbl in [tau_header, tau_num,
              t_header, t_num, z_header, z_num, horizon_lbl,
              r_label]],
            run_time=0.8,
        )

        card = Text("Schwarzschild Infall", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\tau\;\text{(proper time) finite through horizon}",
                font_size=19,
            ),
            MathTex(
                r"t\;\text{(coordinate time)} \to \infty"
                r"\;\text{as } r \to r_s",
                font_size=19, color=ORANGE,
            ),
            MathTex(
                r"1+z = (1 - r_s/r)^{-1/2}"
                r"\;\to\;\infty\;\text{(infinite redshift)}",
                font_size=18, color=RED,
            ),
            MathTex(
                r"\text{Distant observer: particle ``freezes'' "
                r"at horizon}",
                font_size=18, color=TEAL,
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
