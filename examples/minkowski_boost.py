"""Minkowski spacetime grid under Lorentz boost — hyperbolic shearing.

A (ct, x) coordinate grid is drawn in the lab frame, then smoothly
Lorentz-boosted through increasing β.  The grid lines hyperbolically
shear, simultaneity surfaces tilt, and the light cone stays invariant
at 45°.  World-lines of stationary and moving observers show proper
time dilation and length contraction.

Five acts:
  1. Lab-frame grid with light cone
  2. Smooth β sweep (0 → 0.9): grid shears, colour encodes γ
  3. Simultaneity breakdown — tilting "now" slices
  4. Worldline comparison — stationary vs boosted clocks
  5. Ultra-relativistic limit (β → 0.99) — grid collapses onto cone

Run:
    manim -pql examples/minkowski_boost.py MinkowskiBoost
    manim -qh  examples/minkowski_boost.py MinkowskiBoost
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_D,
    DEGREES,
    DOWN,
    GREEN,
    GREEN_E,
    GREY,
    LEFT,
    ORANGE,
    PI,
    RED,
    RED_E,
    RIGHT,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    GOLD,
    MAROON,
    Arrow,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    NumberPlane,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
    interpolate_color,
    ParametricFunction,
    ReplacementTransform,
    Uncreate,
    rate_functions,
)


# ═══════════════════════════════════════════════════════════════════════════
# Physics
# ═══════════════════════════════════════════════════════════════════════════

def _gamma(beta: float) -> float:
    return 1.0 / np.sqrt(1.0 - beta ** 2)


def _boost_point(ct: float, x: float, beta: float):
    """Apply Lorentz boost Λ to (ct, x) → (ct', x')."""
    g = _gamma(beta)
    ct_p = g * (ct - beta * x)
    x_p = g * (x - beta * ct)
    return ct_p, x_p


# ═══════════════════════════════════════════════════════════════════════════
# Grid builders
# ═══════════════════════════════════════════════════════════════════════════

_GRID_RANGE = 4.0   # ±4 in ct and x
_N_LINES = 9        # lines at -4, -3, ..., 3, 4


def _build_grid_lines(beta: float, color_x=BLUE, color_ct=RED,
                      opacity=0.5, stroke=1.5):
    """Build boosted constant-x and constant-ct lines as VGroups."""
    ct_lines = VGroup()   # lines of constant x' (vertical in lab → tilted)
    x_lines = VGroup()    # lines of constant ct' (horizontal → tilted)

    vals = np.linspace(-_GRID_RANGE, _GRID_RANGE, _N_LINES)
    t_span = np.linspace(-_GRID_RANGE * 1.5, _GRID_RANGE * 1.5, 80)

    for v in vals:
        # constant x' = v  →  parametrise by ct'
        pts_x = []
        for t in t_span:
            ct, x = _boost_point(t, v, -beta)  # inverse boost
            if abs(ct) <= _GRID_RANGE * 1.2 and abs(x) <= _GRID_RANGE * 1.2:
                pts_x.append([x, ct, 0])
        if len(pts_x) >= 2:
            line = VGroup()
            for i in range(len(pts_x) - 1):
                seg = Line(pts_x[i], pts_x[i + 1],
                           stroke_width=stroke, color=color_x)
                seg.set_opacity(opacity)
                line.add(seg)
            ct_lines.add(line)

        # constant ct' = v  →  parametrise by x'
        pts_ct = []
        for s in t_span:
            ct, x = _boost_point(v, s, -beta)
            if abs(ct) <= _GRID_RANGE * 1.2 and abs(x) <= _GRID_RANGE * 1.2:
                pts_ct.append([x, ct, 0])
        if len(pts_ct) >= 2:
            line = VGroup()
            for i in range(len(pts_ct) - 1):
                seg = Line(pts_ct[i], pts_ct[i + 1],
                           stroke_width=stroke, color=color_ct)
                seg.set_opacity(opacity)
                line.add(seg)
            x_lines.add(line)

    return VGroup(ct_lines, x_lines)


def _build_light_cone(extent=_GRID_RANGE, color=YELLOW, stroke=2.5):
    """45° light cone from origin."""
    cone = VGroup()
    for sx in [1, -1]:
        for st in [1, -1]:
            cone.add(Line(
                [0, 0, 0], [sx * extent, st * extent, 0],
                stroke_width=stroke, color=color,
            ))
    return cone


def _build_axes(extent=_GRID_RANGE):
    """ct (vertical) and x (horizontal) arrows with labels."""
    x_ax = Arrow([-extent - 0.3, 0, 0], [extent + 0.5, 0, 0],
                 buff=0, stroke_width=2, color=WHITE)
    ct_ax = Arrow([0, -extent - 0.3, 0], [0, extent + 0.5, 0],
                  buff=0, stroke_width=2, color=WHITE)
    x_lbl = MathTex("x", font_size=28).next_to(x_ax, RIGHT, buff=0.1)
    ct_lbl = MathTex("ct", font_size=28).next_to(ct_ax, UP, buff=0.1)
    return VGroup(x_ax, ct_ax, x_lbl, ct_lbl)


def _build_simultaneity_line(beta: float, ct_prime: float = 0.0,
                              color=GREEN, stroke=2.5):
    """A line of constant ct' = ct_prime in the lab frame.
    ct' = γ(ct - βx) = ct_prime  →  ct = ct_prime/γ + βx."""
    g = _gamma(beta) if beta != 0 else 1.0
    ext = _GRID_RANGE * 1.2

    x_vals = np.linspace(-ext, ext, 2)
    pts = []
    for x in x_vals:
        ct = ct_prime / g + beta * x
        pts.append([x, ct, 0])
    return Line(pts[0], pts[1], stroke_width=stroke, color=color)


def _build_worldline(beta_obs: float, color=TEAL, stroke=2.5):
    """Worldline of an observer moving at beta_obs.
    x = beta_obs * ct  →  straight line through origin."""
    ext = _GRID_RANGE
    ct_max = ext
    x_max = beta_obs * ct_max
    pts = [[-x_max, -ct_max, 0], [x_max, ct_max, 0]]
    return Line(pts[0], pts[1], stroke_width=stroke, color=color)


def _build_tick_marks(beta: float, n_ticks: int = 6, color=TEAL,
                       radius=0.06):
    """Proper-time ticks on a worldline at β."""
    g = _gamma(beta)
    ticks = VGroup()
    for k in range(-n_ticks, n_ticks + 1):
        tau = k * 0.6  # proper time spacing
        ct = g * tau
        x = g * beta * tau
        if abs(ct) <= _GRID_RANGE and abs(x) <= _GRID_RANGE:
            d = Dot([x, ct, 0], radius=radius, color=color)
            ticks.add(d)
    return ticks


def _build_hyperbola(s_squared: float, color=ORANGE, stroke=1.5, n_pts=200):
    """Invariant hyperbola x² - (ct)² = s² (spacelike) or (ct)² - x² = s² (timelike)."""
    curves = VGroup()
    ext = _GRID_RANGE * 1.1

    if s_squared > 0:
        # Spacelike: x = ±√(s² + ct²)
        s = np.sqrt(s_squared)
        ct_vals = np.linspace(-ext, ext, n_pts)
        for sign in [1, -1]:
            pts = []
            for ct in ct_vals:
                x = sign * np.sqrt(s_squared + ct ** 2)
                if abs(x) <= ext:
                    pts.append([x, ct, 0])
            if len(pts) >= 2:
                segs = VGroup()
                for i in range(len(pts) - 1):
                    segs.add(Line(pts[i], pts[i+1],
                                  stroke_width=stroke, color=color))
                curves.add(segs)
    else:
        # Timelike: ct = ±√(|s²| + x²)
        s2 = abs(s_squared)
        x_vals = np.linspace(-ext, ext, n_pts)
        for sign in [1, -1]:
            pts = []
            for x in x_vals:
                ct = sign * np.sqrt(s2 + x ** 2)
                if abs(ct) <= ext:
                    pts.append([x, ct, 0])
            if len(pts) >= 2:
                segs = VGroup()
                for i in range(len(pts) - 1):
                    segs.add(Line(pts[i], pts[i+1],
                                  stroke_width=stroke, color=color))
                curves.add(segs)

    return curves


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class MinkowskiBoost(Scene):
    """Minkowski spacetime grid under progressive Lorentz boost."""

    def construct(self):
        # ── Act 1: Lab frame ────────────────────────────────────────────
        title = Text("Minkowski Spacetime — Lorentz Boost",
                      font_size=30).to_edge(UP)
        self.play(Write(title))

        eq = MathTex(
            r"\Lambda = \begin{pmatrix} \gamma & -\beta\gamma \\"
            r" -\beta\gamma & \gamma \end{pmatrix}",
            font_size=22,
        ).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(eq))

        axes = _build_axes()
        light_cone = _build_light_cone()
        light_cone.set_opacity(0.4)

        grid_0 = _build_grid_lines(0.0, color_x=BLUE_D, color_ct=RED_E)

        self.play(Create(axes), run_time=1)
        self.play(FadeIn(grid_0), run_time=1.5)

        lc_lbl = MathTex(r"x = \pm ct", font_size=18, color=YELLOW
                          ).move_to([2.8, 3.3, 0])
        self.play(Create(light_cone), FadeIn(lc_lbl), run_time=1)
        self.wait(0.5)

        # ── Act 2: Smooth β sweep ──────────────────────────────────────
        betas = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        beta_colors = [BLUE, TEAL, GREEN, ORANGE, RED, MAROON]

        current_grid = grid_0
        lbl = None

        for beta, col in zip(betas, beta_colors):
            g = _gamma(beta)
            new_grid = _build_grid_lines(
                beta,
                color_x=interpolate_color(BLUE_D, col, 0.6),
                color_ct=interpolate_color(RED_E, col, 0.6),
                opacity=0.45,
            )

            new_lbl = MathTex(
                rf"\beta = {beta:.2f},\;\gamma = {g:.2f}",
                font_size=22,
            ).next_to(eq, DOWN, buff=0.15)

            anims = [ReplacementTransform(current_grid, new_grid)]
            if lbl is not None:
                anims.append(ReplacementTransform(lbl, new_lbl))
            else:
                anims.append(FadeIn(new_lbl))

            self.play(*anims, run_time=1.8)
            current_grid = new_grid
            lbl = new_lbl
            self.wait(0.3)

        self.wait(0.5)

        # ── Act 3: Simultaneity breakdown ──────────────────────────────
        sim_title = Text("Simultaneity is frame-dependent",
                          font_size=22, color=GREEN).to_edge(DOWN)
        self.play(FadeIn(sim_title), run_time=0.5)

        # Show "now" lines at ct'=0 for different β
        sim_betas = [0.0, 0.3, 0.6, 0.9]
        sim_lines = VGroup()
        sim_labels = VGroup()

        for i, sb in enumerate(sim_betas):
            sl = _build_simultaneity_line(
                sb, ct_prime=0.0,
                color=interpolate_color(GREEN, YELLOW, i / 3),
                stroke=2.0,
            )
            sl_lbl = MathTex(
                rf"\beta={sb:.1f}", font_size=14,
                color=interpolate_color(GREEN, YELLOW, i / 3),
            ).move_to(sl.get_end() + np.array([0.4, 0.2, 0]))
            sim_lines.add(sl)
            sim_labels.add(sl_lbl)

        self.play(
            *[Create(sl) for sl in sim_lines],
            *[FadeIn(sl) for sl in sim_labels],
            run_time=2,
        )
        self.wait(1)
        self.play(FadeOut(sim_lines), FadeOut(sim_labels),
                  FadeOut(sim_title), run_time=0.8)

        # ── Act 4: Worldlines + proper time ────────────────────────────
        wl_title = Text("Worldlines & proper time",
                         font_size=22, color=TEAL).to_edge(DOWN)
        self.play(FadeIn(wl_title), run_time=0.5)

        # Stationary observer worldline (β=0 → vertical)
        wl_stat = _build_worldline(0.0, color=WHITE, stroke=2.5)
        ticks_stat = _build_tick_marks(0.0, n_ticks=5, color=WHITE,
                                        radius=0.05)
        wl_stat_lbl = MathTex(r"\beta=0", font_size=16, color=WHITE
                               ).move_to([0.35, 3.2, 0])

        # Moving observer (β=0.6)
        wl_mov = _build_worldline(0.6, color=TEAL, stroke=2.5)
        ticks_mov = _build_tick_marks(0.6, n_ticks=5, color=TEAL,
                                       radius=0.05)
        wl_mov_lbl = MathTex(r"\beta=0.6", font_size=16, color=TEAL
                              ).move_to([2.5, 3.5, 0])

        # Fast observer (β=0.9)
        wl_fast = _build_worldline(0.9, color=GOLD, stroke=2.5)
        ticks_fast = _build_tick_marks(0.9, n_ticks=5, color=GOLD,
                                        radius=0.05)
        wl_fast_lbl = MathTex(r"\beta=0.9", font_size=16, color=GOLD
                               ).move_to([3.5, 3.0, 0])

        # Proper time formula
        tau_eq = MathTex(
            r"d\tau = dt\sqrt{1-\beta^2} = dt/\gamma",
            font_size=20,
        ).next_to(lbl, DOWN, buff=0.2)

        self.play(
            Create(wl_stat), FadeIn(ticks_stat), FadeIn(wl_stat_lbl),
            run_time=1,
        )
        self.play(
            Create(wl_mov), FadeIn(ticks_mov), FadeIn(wl_mov_lbl),
            FadeIn(tau_eq),
            run_time=1.2,
        )
        self.play(
            Create(wl_fast), FadeIn(ticks_fast), FadeIn(wl_fast_lbl),
            run_time=1,
        )
        self.wait(1)

        # Show invariant hyperbolas
        hyp_t = _build_hyperbola(-1.0, color=ORANGE, stroke=1.2)
        hyp_s = _build_hyperbola(1.0, color=PURPLE if False else ORANGE,
                                  stroke=1.2)
        hyp_t.set_opacity(0.4)
        hyp_s.set_opacity(0.4)
        hyp_lbl = MathTex(
            r"x^2 - (ct)^2 = \text{const}", font_size=16, color=ORANGE,
        ).move_to([3.0, 1.0, 0])

        self.play(FadeIn(hyp_t), FadeIn(hyp_s), FadeIn(hyp_lbl),
                  run_time=1.5)
        self.wait(1)

        # Clean up worldlines
        wl_group = VGroup(wl_stat, wl_mov, wl_fast,
                          ticks_stat, ticks_mov, ticks_fast,
                          wl_stat_lbl, wl_mov_lbl, wl_fast_lbl,
                          tau_eq, hyp_t, hyp_s, hyp_lbl, wl_title)
        self.play(FadeOut(wl_group), run_time=0.8)

        # ── Act 5: Ultra-relativistic collapse ─────────────────────────
        ur_title = Text("Ultra-relativistic limit",
                         font_size=22, color=RED).to_edge(DOWN)
        self.play(FadeIn(ur_title), run_time=0.5)

        ultra_betas = [0.92, 0.95, 0.97, 0.99]
        for beta in ultra_betas:
            g = _gamma(beta)
            new_grid = _build_grid_lines(
                beta,
                color_x=interpolate_color(RED, YELLOW, 0.3),
                color_ct=interpolate_color(RED, YELLOW, 0.3),
                opacity=0.35,
            )
            new_lbl = MathTex(
                rf"\beta = {beta:.2f},\;\gamma = {g:.1f}",
                font_size=22,
            ).next_to(eq, DOWN, buff=0.15)

            self.play(
                ReplacementTransform(current_grid, new_grid),
                ReplacementTransform(lbl, new_lbl),
                run_time=1.5,
            )
            current_grid = new_grid
            lbl = new_lbl
            self.wait(0.2)

        # Final note
        final = MathTex(
            r"\beta \to 1:\;"
            r"\text{grid collapses onto the light cone}",
            font_size=20, color=YELLOW,
        ).next_to(lbl, DOWN, buff=0.2)
        self.play(FadeIn(final), run_time=1)
        self.wait(1.5)

        # Fade out
        self.play(
            FadeOut(current_grid), FadeOut(light_cone), FadeOut(axes),
            FadeOut(eq), FadeOut(title), FadeOut(lbl), FadeOut(lc_lbl),
            FadeOut(ur_title), FadeOut(final),
        )
        self.wait(0.5)
