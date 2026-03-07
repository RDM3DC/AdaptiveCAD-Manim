"""Kerr Metric Time Dilation — frame-dragging, ergosphere, and the wealth bubble.

A rotating Kerr black hole drags spacetime, tilting light cones and
dilating proper time.  We visualise:

  Act 1 — Schwarzschild: non-rotating BH, gravitational time dilation
          γ_grav = 1/√(1 - r_s/r), light cones tilt toward singularity
  Act 2 — Spin up: Kerr parameter a/M ramps from 0 → 0.998,
          ergosphere forms, frame-dragging velocity shown
  Act 3 — Proper time map: colour-coded dilation field around the hole
  Act 4 — Tipler cylinder sketch: finite rotating shell, CTC region
  Act 5 — The Wealth Bubble: park your bank account at r = 1.01 r+,
          watch cosmic time fly while your proper time crawls —
          compound interest piles up at γ ≈ 50×

The physics is real (Boyer-Lindquist coordinates, exact Kerr metric).
The financial advice is not.

Run:
    manim -pql examples/kerr_time_dilation.py KerrTimeDilation
    manim -qh  examples/kerr_time_dilation.py KerrTimeDilation
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_D,
    BLUE_E,
    DEGREES,
    DOWN,
    GREEN,
    GREEN_E,
    GREY,
    GREY_A,
    LEFT,
    ORANGE,
    PI,
    PURPLE,
    RED,
    RED_E,
    RIGHT,
    TAU,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    GOLD,
    MAROON,
    Annulus,
    Arc,
    Arrow,
    Circle,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    MathTex,
    NumberLine,
    Scene,
    Text,
    Tex,
    Transform,
    ReplacementTransform,
    VGroup,
    Write,
    interpolate_color,
    rate_functions,
    Polygon,
    Rectangle,
    always_redraw,
    ValueTracker,
    Indicate,
)

# ═══════════════════════════════════════════════════════════════════════════
# Kerr metric physics (Boyer-Lindquist, G = c = 1 units)
# ═══════════════════════════════════════════════════════════════════════════

def _r_plus(M: float, a: float) -> float:
    """Outer horizon radius r+ = M + √(M² - a²)."""
    return M + np.sqrt(max(M**2 - a**2, 0))


def _r_ergo(M: float, a: float, theta: float = np.pi / 2) -> float:
    """Ergosphere radius at angle θ: r_ergo = M + √(M² - a²cos²θ)."""
    return M + np.sqrt(max(M**2 - (a * np.cos(theta))**2, 0))


def _delta(r: float, M: float, a: float) -> float:
    """Δ = r² - 2Mr + a²."""
    return r**2 - 2 * M * r + a**2


def _sigma(r: float, theta: float, a: float) -> float:
    """Σ = r² + a²cos²θ."""
    return r**2 + (a * np.cos(theta))**2


def _grav_dilation(r: float, M: float, a: float = 0.0,
                    theta: float = np.pi / 2) -> float:
    """Gravitational time dilation factor dτ/dt for a ZAMO observer.

    For Schwarzschild (a=0): dτ/dt = √(1 - r_s/r) where r_s = 2M.
    For Kerr (equatorial): uses the full metric component.
    Returns γ = dt/dτ (> 1 means time runs slower).
    """
    if a == 0:
        rs = 2 * M
        if r <= rs:
            return float('inf')
        return 1.0 / np.sqrt(1.0 - rs / r)

    sig = _sigma(r, theta, a)
    delt = _delta(r, M, a)
    rp = _r_plus(M, a)
    if r <= rp * 1.001:
        return float('inf')

    # g_tt component in Boyer-Lindquist
    g_tt = -(1 - 2 * M * r / sig)
    # For ZAMO (zero angular momentum observer):
    # dτ/dt = √(-g_tt - g_tφ²/g_φφ)
    A = (r**2 + a**2)**2 - a**2 * delt * np.sin(theta)**2
    omega_drag = 2 * M * a * r / A  # frame-dragging angular velocity
    g_phiphi = A * np.sin(theta)**2 / sig
    g_tphi = -2 * M * a * r * np.sin(theta)**2 / sig

    alpha2 = delt * sig / A  # lapse² for ZAMO
    if alpha2 <= 0:
        return float('inf')
    return 1.0 / np.sqrt(alpha2)


def _frame_drag_omega(r: float, M: float, a: float,
                       theta: float = np.pi / 2) -> float:
    """Frame-dragging angular velocity ω = -g_tφ/g_φφ."""
    sig = _sigma(r, theta, a)
    A = (r**2 + a**2)**2 - a**2 * _delta(r, M, a) * np.sin(theta)**2
    if A == 0:
        return 0
    return 2 * M * a * r / A


# ═══════════════════════════════════════════════════════════════════════════
# Visual builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_bh_circle(r_h: float, scale: float, color=WHITE):
    """Filled circle for the event horizon."""
    c = Circle(radius=r_h * scale, color=color, fill_opacity=0.9,
               stroke_width=1)
    return c


def _build_ergosphere(r_ergo_eq: float, r_h: float, scale: float,
                       color=PURPLE):
    """Ergosphere annulus."""
    return Annulus(
        inner_radius=r_h * scale,
        outer_radius=r_ergo_eq * scale,
        color=color, fill_opacity=0.2, stroke_width=1.5,
        stroke_color=color,
    )


def _build_light_cone_2d(x: float, y: float, tilt: float = 0.0,
                          size: float = 0.3, color=YELLOW, opacity=0.7):
    """A small 2D light cone (triangle pair) at position (x, y).
    tilt: angle in radians the cone tilts toward the BH (0 = vertical).
    """
    half = size / 2
    # Future cone (upward triangle)
    tip_future = np.array([x + half * np.sin(tilt),
                           y + size * np.cos(tilt), 0])
    bl = np.array([x - half * np.cos(tilt) + half * np.sin(tilt) * 0.3,
                   y + half * np.sin(tilt) * 0.3, 0])
    br = np.array([x + half * np.cos(tilt) + half * np.sin(tilt) * 0.3,
                   y + half * np.sin(tilt) * 0.3, 0])
    base = np.array([x, y, 0])

    # Simplified: two lines from base diverging upward, tilted
    angle_open = 0.4  # half-opening angle
    len_line = size

    dir_left = np.array([
        np.sin(tilt - angle_open),
        np.cos(tilt - angle_open), 0
    ])
    dir_right = np.array([
        np.sin(tilt + angle_open),
        np.cos(tilt + angle_open), 0
    ])

    future_l = Line(base, base + dir_left * len_line,
                     stroke_width=2, color=color).set_opacity(opacity)
    future_r = Line(base, base + dir_right * len_line,
                     stroke_width=2, color=color).set_opacity(opacity)

    # Past cone (downward)
    past_l = Line(base, base - dir_left * len_line * 0.6,
                   stroke_width=1.5, color=color).set_opacity(opacity * 0.5)
    past_r = Line(base, base - dir_right * len_line * 0.6,
                   stroke_width=1.5, color=color).set_opacity(opacity * 0.5)

    return VGroup(future_l, future_r, past_l, past_r)


def _build_dilation_field(M: float, a: float, scale: float,
                           r_min: float = None, r_max: float = 8.0,
                           n_rings: int = 12, n_angles: int = 24):
    """Colour-coded dilation map: dots coloured by γ factor."""
    rp = _r_plus(M, a)
    if r_min is None:
        r_min = rp * 1.05

    dots = VGroup()
    r_vals = np.linspace(r_min, r_max, n_rings)

    for r in r_vals:
        for k in range(n_angles):
            angle = TAU * k / n_angles
            x = r * scale * np.cos(angle)
            y = r * scale * np.sin(angle)
            g = _grav_dilation(r, M, a)
            g = min(g, 50)
            # Map γ: 1 → blue, 5 → green, 20 → red, 50 → white
            if g < 2:
                col = interpolate_color(BLUE, GREEN, (g - 1))
            elif g < 10:
                col = interpolate_color(GREEN, RED, (g - 2) / 8)
            else:
                col = interpolate_color(RED, WHITE, min((g - 10) / 40, 1))
            d = Dot([x, y, 0], radius=0.04, color=col).set_opacity(0.7)
            dots.add(d)

    return dots


def _build_frame_drag_arrows(M: float, a: float, scale: float,
                               r_vals=None, n_angles: int = 8):
    """Arrows showing frame-dragging direction and magnitude."""
    if r_vals is None:
        rp = _r_plus(M, a)
        r_vals = [rp * 1.3, rp * 2, rp * 3, rp * 5]

    arrows = VGroup()
    for r in r_vals:
        omega = _frame_drag_omega(r, M, a)
        arrow_len = min(omega * 80, 0.6)  # scale for visibility
        if arrow_len < 0.05:
            continue
        for k in range(n_angles):
            angle = TAU * k / n_angles
            x = r * scale * np.cos(angle)
            y = r * scale * np.sin(angle)
            # Tangential direction (perpendicular to radial, in drag dir)
            dx = -arrow_len * np.sin(angle)
            dy = arrow_len * np.cos(angle)
            arr = Arrow(
                [x, y, 0], [x + dx, y + dy, 0],
                buff=0, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.3,
                color=TEAL,
            ).set_opacity(0.6)
            arrows.add(arr)

    return arrows


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class KerrTimeDilation(Scene):
    """Kerr metric time dilation with frame-dragging and the wealth bubble."""

    def construct(self):
        M = 1.0       # mass (geometric units)
        SCALE = 0.45  # r → screen coords scale factor

        # ── Act 1: Schwarzschild ───────────────────────────────────────
        title = Text("Gravitational Time Dilation",
                      font_size=30).to_edge(UP)
        self.play(Write(title))

        schwarz_eq = MathTex(
            r"\gamma_{\text{grav}} = \frac{1}{\sqrt{1 - r_s/r}}"
            r",\quad r_s = 2GM/c^2",
            font_size=20,
        ).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(schwarz_eq))

        # Black hole
        r_s = 2 * M
        bh = _build_bh_circle(r_s, SCALE, color=GREY)
        bh_label = MathTex(r"r_s", font_size=18, color=GREY_A
                            ).next_to(bh, DOWN, buff=0.1)
        self.play(GrowFromCenter(bh), FadeIn(bh_label), run_time=1)

        # Light cones at various radii
        cone_radii = [3, 4.5, 6, 8]
        cones_group = VGroup()
        gamma_labels = VGroup()

        for r in cone_radii:
            g = _grav_dilation(r, M, a=0)
            # Tilt: more tilt closer to BH
            tilt = 0.4 * (r_s / r) ** 1.5
            for angle_k in range(6):
                ang = TAU * angle_k / 6
                x = r * SCALE * np.cos(ang)
                y = r * SCALE * np.sin(ang)
                cone = _build_light_cone_2d(x, y, tilt=tilt * np.cos(ang),
                                             size=0.25, opacity=0.6)
                cones_group.add(cone)

            # γ label at one position
            lx = r * SCALE * 1.05
            gl = MathTex(rf"\gamma={g:.1f}", font_size=14,
                          color=YELLOW).move_to([lx, 0.3, 0])
            gamma_labels.add(gl)

        self.play(FadeIn(cones_group), run_time=2)
        self.play(FadeIn(gamma_labels), run_time=1)
        self.wait(0.5)

        # Clean act 1
        self.play(FadeOut(cones_group), FadeOut(gamma_labels),
                  FadeOut(bh_label), run_time=0.8)

        # ── Act 2: Spin up → Kerr ─────────────────────────────────────
        kerr_title = Text("Spinning Up — Kerr Black Hole",
                           font_size=24, color=PURPLE).to_edge(DOWN)
        self.play(
            ReplacementTransform(
                schwarz_eq,
                MathTex(
                    r"\text{Kerr: } ds^2 = -\left(1-\frac{2Mr}{\Sigma}"
                    r"\right)dt^2 + \cdots",
                    font_size=18,
                ).next_to(title, DOWN, buff=0.15),
            ),
            FadeIn(kerr_title),
            run_time=1,
        )
        schwarz_eq = self.mobjects[-3]  # updated ref

        spins = [0.0, 0.3, 0.6, 0.9, 0.95, 0.998]
        spin_colors = [GREY, BLUE_D, TEAL, GREEN_E, ORANGE, RED]

        ergo = None
        spin_lbl = None
        drag_arrows = None

        for a, col in zip(spins, spin_colors):
            rp = _r_plus(M, a)
            re_eq = _r_ergo(M, a)

            new_bh = _build_bh_circle(rp, SCALE, color=col)

            new_spin_lbl = MathTex(
                rf"a/M = {a:.3f},\; r_+ = {rp:.3f}M",
                font_size=18,
            ).move_to([3.5, -2.5, 0])

            anims = [Transform(bh, new_bh)]

            if a > 0:
                new_ergo = _build_ergosphere(re_eq, rp, SCALE, color=PURPLE)
                new_drag = _build_frame_drag_arrows(M, a, SCALE)

                if ergo is not None:
                    anims.extend([
                        Transform(ergo, new_ergo),
                        ReplacementTransform(drag_arrows, new_drag),
                    ])
                else:
                    anims.extend([FadeIn(new_ergo), FadeIn(new_drag)])
                    ergo = new_ergo
                    drag_arrows = new_drag

                if ergo is not new_ergo and ergo is not None:
                    pass  # Transform handles it

            if spin_lbl is not None:
                anims.append(ReplacementTransform(spin_lbl, new_spin_lbl))
            else:
                anims.append(FadeIn(new_spin_lbl))

            self.play(*anims, run_time=1.5)
            spin_lbl = new_spin_lbl
            self.wait(0.3)

        self.play(FadeOut(kerr_title), run_time=0.5)

        # ── Act 3: Dilation field ──────────────────────────────────────
        dil_title = Text("Time Dilation Field",
                          font_size=24, color=ORANGE).to_edge(DOWN)
        self.play(FadeIn(dil_title), run_time=0.5)

        a_final = 0.998
        dil_dots = _build_dilation_field(M, a_final, SCALE)
        legend = VGroup(
            Dot(color=BLUE, radius=0.05),
            MathTex(r"\gamma \approx 1", font_size=14),
            Dot(color=GREEN, radius=0.05),
            MathTex(r"\gamma \approx 5", font_size=14),
            Dot(color=RED, radius=0.05),
            MathTex(r"\gamma \approx 20", font_size=14),
            Dot(color=WHITE, radius=0.05),
            MathTex(r"\gamma \geq 50", font_size=14),
        ).arrange(RIGHT, buff=0.12).move_to([-3.5, -3, 0]).scale(0.8)

        self.play(FadeIn(dil_dots), FadeIn(legend), run_time=2)
        self.wait(1)
        self.play(FadeOut(dil_dots), FadeOut(legend),
                  FadeOut(dil_title), run_time=0.8)

        # ── Act 4: Tipler cylinder sketch ──────────────────────────────
        tip_title = Text("Tipler Cylinder — Rotating Shell",
                          font_size=24, color=TEAL).to_edge(DOWN)

        tip_eq = MathTex(
            r"\text{Dense rotating cylinder: } \omega, \rho, L "
            r"\;\Rightarrow\; \text{frame-dragging} \;\Rightarrow\;"
            r"\text{closed timelike curves?}",
            font_size=16,
        ).move_to([0, -2, 0])

        # Simple cylinder representation: nested rectangles
        cyl_outer = Rectangle(width=1.5, height=3.0, color=TEAL,
                               stroke_width=2).set_opacity(0.3)
        cyl_inner = Rectangle(width=0.8, height=3.0, color=TEAL,
                               stroke_width=1.5).set_opacity(0.15)
        cyl_label = MathTex(r"\omega", font_size=20, color=TEAL
                             ).next_to(cyl_outer, RIGHT, buff=0.15)
        # Rotation arrows
        rot_arcs = VGroup()
        for yy in [-1, 0, 1]:
            arc = Arc(radius=0.25, start_angle=0, angle=1.5 * PI,
                       color=TEAL, stroke_width=1.5).move_to([0, yy, 0])
            rot_arcs.add(arc)

        cyl_group = VGroup(cyl_outer, cyl_inner, cyl_label, rot_arcs)
        cyl_group.move_to([0, 0.5, 0])

        # Temporarily hide BH stuff
        if ergo is not None:
            self.play(
                FadeOut(bh), FadeOut(ergo), FadeOut(drag_arrows),
                FadeOut(spin_lbl),
                FadeIn(tip_title),
                run_time=0.8,
            )
        else:
            self.play(FadeOut(bh), FadeOut(spin_lbl),
                      FadeIn(tip_title), run_time=0.8)

        self.play(Create(cyl_group), FadeIn(tip_eq), run_time=1.5)
        self.wait(1)

        ctc_note = Text(
            "If ρ·ω² > threshold → light cones tip over → CTCs form",
            font_size=16, color=YELLOW,
        ).move_to([0, -2.8, 0])
        self.play(FadeIn(ctc_note), run_time=1)
        self.wait(1)

        self.play(FadeOut(cyl_group), FadeOut(tip_eq),
                  FadeOut(ctc_note), FadeOut(tip_title), run_time=0.8)

        # ── Act 5: The Wealth Bubble ───────────────────────────────────
        wealth_title = Text("The Wealth Bubble",
                             font_size=28, color=GOLD).to_edge(UP)
        self.play(
            ReplacementTransform(title, wealth_title),
            run_time=0.8,
        )

        # Remove old subtitle
        for m in self.mobjects[:]:
            if isinstance(m, MathTex) and m is not wealth_title:
                self.remove(m)

        premise = Text(
            "Park your bank account at r = 1.01 r₊ of a near-extremal Kerr hole",
            font_size=18, color=WHITE,
        ).next_to(wealth_title, DOWN, buff=0.2)
        self.play(FadeIn(premise), run_time=1)

        # Show the BH again
        a_wb = 0.998
        rp_wb = _r_plus(M, a_wb)
        r_park = rp_wb * 1.01
        gamma_park = _grav_dilation(r_park, M, a_wb)

        bh_wb = _build_bh_circle(rp_wb, SCALE, color=RED)
        ergo_wb = _build_ergosphere(_r_ergo(M, a_wb), rp_wb, SCALE)
        self.play(GrowFromCenter(bh_wb), FadeIn(ergo_wb), run_time=1)

        # Mark parking orbit
        park_circle = Circle(
            radius=r_park * SCALE, color=GOLD,
            stroke_width=2,
        ).set_fill(opacity=0)
        park_dot = Dot([r_park * SCALE, 0, 0], radius=0.08, color=GOLD)
        park_lbl = MathTex(
            rf"r = {r_park:.3f}M,\;\gamma = {gamma_park:.0f}",
            font_size=16, color=GOLD,
        ).next_to(park_dot, UP + RIGHT, buff=0.1)

        self.play(Create(park_circle), FadeIn(park_dot),
                  FadeIn(park_lbl), run_time=1.5)

        # Compound interest animation
        rate_text = MathTex(
            r"\text{Interest rate: } 5\%\text{/yr (cosmic time)}",
            font_size=18,
        ).move_to([-3.5, 1.5, 0])
        self.play(FadeIn(rate_text), run_time=0.5)

        # Account balance readout
        initial_balance = -50000  # starts negative!
        interest_rate = 0.05

        balance_tracker = ValueTracker(0)  # cosmic years elapsed

        balance_label = always_redraw(lambda: MathTex(
            r"\text{Cosmic time: }"
            + f"{balance_tracker.get_value():.0f}"
            + r"\text{ yr}",
            font_size=18,
        ).move_to([-3.5, 0.5, 0]))

        proper_label = always_redraw(lambda: MathTex(
            r"\text{Your time: }"
            + f"{balance_tracker.get_value() / gamma_park:.1f}"
            + r"\text{ yr}",
            font_size=18, color=TEAL,
        ).move_to([-3.5, 0, 0]))

        current_balance = always_redraw(lambda: MathTex(
            r"\$"
            + f"{initial_balance * (1 + interest_rate) ** balance_tracker.get_value():,.0f}",
            font_size=24,
            color=GREEN if initial_balance * (1 + interest_rate)
            ** balance_tracker.get_value() > 0 else RED,
        ).move_to([-3.5, -1, 0]))

        self.play(FadeIn(balance_label), FadeIn(proper_label),
                  FadeIn(current_balance), run_time=0.5)

        # Animate time passing
        # At γ≈50, 1 year of your time = 50 cosmic years
        # $-50,000 at 5%: breaks even around ln(2)/ln(1.05) ≈ 14.2 years
        # But that's 14.2 cosmic years → 0.28 years of your time!
        self.play(
            balance_tracker.animate.set_value(5),
            run_time=1.5,
            rate_func=rate_functions.linear,
        )
        self.play(
            balance_tracker.animate.set_value(15),
            run_time=2,
            rate_func=rate_functions.linear,
        )

        # Balance just went positive!
        cross_note = Text("Balance crosses zero!",
                           font_size=16, color=GREEN).move_to([-3.5, -1.8, 0])
        self.play(FadeIn(cross_note), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(cross_note), run_time=0.3)

        # Now let it really cook
        self.play(
            balance_tracker.animate.set_value(50),
            run_time=3,
            rate_func=rate_functions.linear,
        )
        self.play(
            balance_tracker.animate.set_value(100),
            run_time=2.5,
            rate_func=rate_functions.ease_in_quad,
        )

        # Final readout
        final_cosmic = 100
        final_proper = final_cosmic / gamma_park
        final_balance = initial_balance * (1 + interest_rate) ** final_cosmic

        punchline = VGroup(
            Text(f"100 cosmic years = {final_proper:.1f} years for you",
                 font_size=18, color=TEAL),
            Text(f"Balance: ${final_balance:,.0f}",
                 font_size=22, color=GREEN),
            Text("Financial advice from a black hole. Not investment advice.",
                 font_size=12, color=GREY),
        ).arrange(DOWN, buff=0.15).move_to([0, -2.5, 0])

        self.play(FadeIn(punchline), run_time=1.5)
        self.wait(2)

        # Fade all
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(0.5)
