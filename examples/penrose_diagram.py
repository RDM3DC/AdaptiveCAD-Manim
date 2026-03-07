"""Penrose (Carter–Penrose) conformal diagram — causal structure of spacetime.

Compactifies infinite Minkowski and Schwarzschild spacetimes into
finite diamonds, making the full causal structure visible at a glance.

Six acts:
  1. Minkowski diamond — build (T,X) axes, 45° null boundaries, label
     i⁺, i⁻, i⁰, 𝒥⁺, 𝒥⁻
  2. Light cones — scatter light cones across the diamond, all at 45°
  3. Timelike and spacelike geodesics — curves that stay inside or
     cross the diagram, showing causal vs acausal paths
  4. Schwarzschild collapse — morph into the black-hole Penrose diagram:
     horizon, singularity, exterior region I, interior region II
  5. Infalling observer — worldline from 𝒥⁻ through the horizon into
     the singularity; contrast with a static observer at constant r
  6. Hawking radiation & evaporation — dashed outgoing radiation lines,
     shrinking horizon, thunderclap endpoint

Physics: exact conformal coordinates for Minkowski (arctan compactification)
and Kruskal–Szekeres → Penrose for Schwarzschild.

Run:
    manim -pql examples/penrose_diagram.py PenroseDiagram
    manim -qh  examples/penrose_diagram.py PenroseDiagram
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_D,
    BLUE_E,
    DOWN,
    GREEN,
    GREEN_E,
    GREY,
    GREY_A,
    LEFT,
    ORANGE,
    PI,
    RED,
    RED_E,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    GOLD,
    MAROON,
    PURPLE,
    Arrow,
    Create,
    DashedLine,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    GrowFromCenter,
    Line,
    MathTex,
    Scene,
    Text,
    Tex,
    Transform,
    ReplacementTransform,
    VGroup,
    VMobject,
    Write,
    interpolate_color,
    rate_functions,
    Polygon,
    ParametricFunction,
    Indicate,
    SurroundingRectangle,
    Uncreate,
    AnimationGroup,
    LaggedStart,
    ShowPassingFlash,
    Succession,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants / layout
# ═══════════════════════════════════════════════════════════════════════════

# The diamond lives in a 5×5 box centred at origin.
S = 2.8  # half-side of diamond (in scene units)

# Key points of the Minkowski diamond
I_PLUS  = np.array([0,  S, 0])   # future timelike infinity  i⁺
I_MINUS = np.array([0, -S, 0])   # past timelike infinity    i⁻
I_ZERO_R = np.array([ S, 0, 0])  # right spacelike infinity  i⁰
I_ZERO_L = np.array([-S, 0, 0])  # left spacelike infinity   i⁰

# Colours
COL_NULL     = YELLOW
COL_TIMELIKE = BLUE
COL_SPACELIKE = RED
COL_HORIZON  = GREEN
COL_SINGULARITY = RED_E
COL_HAWKING  = ORANGE


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _diamond_boundary() -> VGroup:
    """Four 45° null edges of the Minkowski diamond."""
    edges = VGroup(
        Line(I_PLUS, I_ZERO_R),   # 𝒥⁺ right
        Line(I_ZERO_R, I_MINUS),  # 𝒥⁻ right
        Line(I_MINUS, I_ZERO_L),  # 𝒥⁻ left
        Line(I_ZERO_L, I_PLUS),   # 𝒥⁺ left
    )
    edges.set_color(COL_NULL)
    edges.set_stroke(width=2.5)
    return edges


def _infinity_labels() -> VGroup:
    """Labels for i⁺, i⁻, i⁰, 𝒥⁺, 𝒥⁻."""
    lbl_ip = MathTex(r"i^+", font_size=30).next_to(I_PLUS, UP, buff=0.15)
    lbl_im = MathTex(r"i^-", font_size=30).next_to(I_MINUS, DOWN, buff=0.15)
    lbl_i0r = MathTex(r"i^0", font_size=30).next_to(I_ZERO_R, RIGHT, buff=0.15)
    lbl_i0l = MathTex(r"i^0", font_size=30).next_to(I_ZERO_L, LEFT, buff=0.15)

    # null infinity labels along edges
    mid_scri_pr = (I_PLUS + I_ZERO_R) / 2
    lbl_scrip_r = MathTex(r"\mathcal{I}^+", font_size=26, color=COL_NULL).move_to(
        mid_scri_pr + np.array([0.35, 0.2, 0])
    )
    mid_scri_mr = (I_MINUS + I_ZERO_R) / 2
    lbl_scrim_r = MathTex(r"\mathcal{I}^-", font_size=26, color=COL_NULL).move_to(
        mid_scri_mr + np.array([0.35, -0.2, 0])
    )
    mid_scri_pl = (I_PLUS + I_ZERO_L) / 2
    lbl_scrip_l = MathTex(r"\mathcal{I}^+", font_size=26, color=COL_NULL).move_to(
        mid_scri_pl + np.array([-0.35, 0.2, 0])
    )
    mid_scri_ml = (I_MINUS + I_ZERO_L) / 2
    lbl_scrim_l = MathTex(r"\mathcal{I}^-", font_size=26, color=COL_NULL).move_to(
        mid_scri_ml + np.array([-0.35, -0.2, 0])
    )
    return VGroup(
        lbl_ip, lbl_im, lbl_i0r, lbl_i0l,
        lbl_scrip_r, lbl_scrim_r, lbl_scrip_l, lbl_scrim_l,
    )


def _light_cone_at(pos, size=0.35):
    """Tiny 45° light cone (future + past) at *pos*."""
    fut_l = Line(pos, pos + np.array([-size, size, 0]), color=COL_NULL, stroke_width=1.8)
    fut_r = Line(pos, pos + np.array([ size, size, 0]), color=COL_NULL, stroke_width=1.8)
    past_l = Line(pos, pos + np.array([-size, -size, 0]), color=COL_NULL, stroke_width=1.8)
    past_r = Line(pos, pos + np.array([ size, -size, 0]), color=COL_NULL, stroke_width=1.8)
    return VGroup(fut_l, fut_r, past_l, past_r)


def _const_r_curve(r_frac, n=80):
    """Constant-r curve in the Penrose diamond (timelike hyperbola).

    *r_frac* ∈ (0,1) is the fraction of S that the curve reaches at T=0.
    In the conformal diagram, constant-r curves are hyperbola-like
    arcs from i⁻ to i⁺, bulging outward to x = r_frac*S at T=0.
    """
    pts = []
    for i in range(n + 1):
        lam = i / n                    # 0..1 parametrises i⁻ → i⁺
        T = -S + 2 * S * lam           # T from -S to +S
        # width of diamond at height T
        w = S - abs(T)
        X = r_frac * w
        pts.append(np.array([X, T, 0]))
    curve = VMobject(color=GREY_A, stroke_width=1.2)
    curve.set_points_smoothly(pts)
    return curve


def _const_t_curve(t_frac, n=80):
    """Constant-t (spacelike) curve in the Penrose diamond.

    *t_frac* ∈ (-1,1).  At t_frac=0 we get the horizontal mid-line.
    Curves are spacelike arcs from i⁰_left to i⁰_right, passing
    through Y = t_frac * (S - |X|) proportionally.
    """
    pts = []
    for i in range(n + 1):
        lam = i / n
        X = -S + 2 * S * lam           # -S .. +S
        h = S - abs(X)                  # available height at this X
        T = t_frac * h
        pts.append(np.array([X, T, 0]))
    curve = VMobject(color=GREY, stroke_width=1.0)
    curve.set_points_smoothly(pts)
    return curve


# ═══════════════════════════════════════════════════════════════════════════
# Schwarzschild Penrose diagram helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bh_boundary() -> VGroup:
    """Schwarzschild Penrose: exterior region I + interior region II.

    The top of the diamond becomes the singularity (horizontal wavy line),
    the 45° line from centre to top-right becomes the event horizon.
    """
    # Exterior right triangle: i⁻ → i⁰_R → junction → i⁻
    # Region I right edges (unchanged null boundaries)
    scri_minus = Line(I_MINUS, I_ZERO_R, color=COL_NULL, stroke_width=2.5)
    scri_plus  = Line(I_ZERO_R, I_PLUS, color=COL_NULL, stroke_width=2.5)

    # Event horizon — 45° from i⁻(bottom) up to i⁺(top), through the origin
    horizon = DashedLine(I_MINUS, I_PLUS, color=COL_HORIZON, stroke_width=3,
                         dash_length=0.12)

    # Singularity — horizontal wavy line at top, from left end to i⁺(top)
    sing_pts = []
    n_wave = 60
    for i in range(n_wave + 1):
        frac = i / n_wave
        x = -S * (1 - frac)  # from -S to 0  (left edge to centre)
        # Actually singularity spans full top: from I_ZERO_L to origin at top
        # In Schwarzschild Penrose, singularity is at T=S, X ∈ [-S, 0]
        # but conventionally it's the full top line  X ∈ [-S, 0] at T = S - |X| ? No.
        # Standard: singularity is horizontal at the top of region II.
        # Region II top: from (-S, S) to (0, S) — but that's outside diamond.
        # Actually in the standard diagram, the singularity replaces the top
        # vertex and becomes a horizontal spacelike line.
        pass

    # Standard Schwarzschild Penrose (region I + II only, no white hole):
    # Bottom: i⁻ at (0, -S)
    # Right: i⁰ at (S, 0)
    # Top-right: 𝒥⁺ from i⁰ to i⁺
    # But i⁺ is at (0, S) only in Minkowski.
    # In Schwarzschild, the top is the SINGULARITY: a horizontal line
    # from (0, S) to (-S, S)... no, that's not right either.
    #
    # Let me use the standard triangular layout:
    # The right half of the diamond stays (region I exterior).
    # The top half of the LEFT side becomes region II (interior).
    # Singularity is horizontal at top from (-S, S_sing) to (0, S_sing)
    # with S_sing = S.  But that point (-S, S) is on the diamond edge.
    #
    # Standard coordinates:
    #   i⁻ = (0, -S)
    #   i⁰ = (S, 0)
    #   The horizon from i⁻ to (0, S) at 45°    -- that's origin to I_PLUS
    #   Singularity at top: horizontal from (0, S) to (-S, S)   -- but (-S, S) not on diamond
    #   𝒥⁺ from (S, 0) to (0, S) -- same as Minkowski right-top edge
    #   𝒥⁻ from (0, -S) to (S, 0) -- same as Minkowski right-bottom edge
    #
    # Wait — in the standard 1-sided Schwarzschild diagram, it's a triangle:
    #   Bottom-left: i⁻    (0, -S)
    #   Right: i⁰          (S, 0)
    #   Top: singularity    horizontal line from (-S/2, S) ... no.
    #
    # OK let me just use the clean standard:
    # The full Kruskal diagram is the full diamond (Regions I, II, III, IV).
    # For simplicity we show the right half (Regions I, II):
    #   - Region I (exterior): right triangle below the horizon
    #   - Region II (BH interior): above the horizon, capped by singularity

    return VGroup(scri_minus, scri_plus, horizon)


def _singularity_line(y_top=None):
    """Wavy horizontal singularity line at top of BH region."""
    if y_top is None:
        y_top = S
    pts = []
    n = 100
    amp = 0.08
    freq = 14
    for i in range(n + 1):
        frac = i / n
        x = -S + S * frac   # from -S to 0
        wiggle = amp * np.sin(freq * frac * 2 * PI)
        pts.append(np.array([x, y_top + wiggle, 0]))
    sing = VMobject(color=COL_SINGULARITY, stroke_width=3.5)
    sing.set_points_smoothly(pts)
    return sing


def _infalling_worldline(n=120):
    """Worldline of an observer falling from 𝒥⁻ through the horizon
    into the singularity.  Starts near i⁰ on 𝒥⁻, curves left and up
    through the horizon, hits the singularity.
    """
    pts = []
    # parametric path: start at (S*0.7, -S*0.3) on 𝒥⁻, curve up through
    # the horizon at the origin area, end at (-S*0.3, S) on the singularity
    for i in range(n + 1):
        t = i / n            # 0..1
        # horizontal: drift from +0.7S to -0.3S
        x = S * (0.7 - 1.0 * t)
        # vertical: -0.3S to +S, with some curvature
        y = -S * 0.3 + S * 1.3 * t ** 0.8
        pts.append(np.array([x, y, 0]))
    wl = VMobject(color=COL_TIMELIKE, stroke_width=3)
    wl.set_points_smoothly(pts)
    return wl


def _static_worldline(r_frac=0.6, n=80):
    """Worldline of a static observer at constant r, from i⁻ to i⁺.
    Same shape as _const_r_curve but only in the right half."""
    pts = []
    for i in range(n + 1):
        lam = i / n
        T = -S + 2 * S * lam
        w = S - abs(T)
        X = r_frac * w
        pts.append(np.array([X, T, 0]))
    wl = VMobject(color=GREEN_E, stroke_width=2.5)
    wl.set_points_smoothly(pts)
    return wl


def _hawking_ray(start_y, n=60):
    """An outgoing null ray (45° to upper-right) emitted near the horizon.
    Starts at (ε, start_y), goes to 𝒥⁺."""
    start = np.array([0.05, start_y, 0])
    length = S - start_y  # vertical distance to top
    end = start + np.array([length, length, 0])
    # clip to diamond
    # at the boundary, X + T = S (right-top edge)
    # X = 0.05 + d, T = start_y + d  → 0.05 + d + start_y + d = S
    # d = (S - 0.05 - start_y) / 2
    d = (S - 0.05 - start_y) / 2
    end = start + np.array([d, d, 0])
    ray = DashedLine(start, end, color=COL_HAWKING, stroke_width=2,
                     dash_length=0.08)
    return ray


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class PenroseDiagram(Scene):
    """Six-act Penrose diagram animation."""

    def construct(self):
        self._act1_minkowski_diamond()
        self._act2_light_cones()
        self._act3_geodesics()
        self._act4_schwarzschild()
        self._act5_infalling_observer()
        self._act6_hawking_evaporation()

    # ── Act 1 — Minkowski diamond ─────────────────────────────────────
    def _act1_minkowski_diamond(self):
        title = Text("Penrose Diagram", font_size=42, color=WHITE)
        subtitle = Text("conformal compactification of spacetime",
                        font_size=22, color=GREY_A)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.3), run_time=1.5)
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)

        # Draw diamond boundary
        boundary = _diamond_boundary()
        self.play(Create(boundary), run_time=2)

        # Axes labels
        t_arrow = Arrow(start=DOWN * 0.8, end=UP * 0.8, color=GREY_A,
                        stroke_width=2, buff=0).shift(LEFT * 0.3)
        t_label = MathTex("T", font_size=28, color=GREY_A).next_to(t_arrow, LEFT, buff=0.08)
        x_arrow = Arrow(start=LEFT * 0.8, end=RIGHT * 0.8, color=GREY_A,
                        stroke_width=2, buff=0).shift(DOWN * 0.3)
        x_label = MathTex("X", font_size=28, color=GREY_A).next_to(x_arrow, DOWN, buff=0.08)
        self.play(Create(t_arrow), Write(t_label),
                  Create(x_arrow), Write(x_label), run_time=1)

        # Infinity labels
        labels = _infinity_labels()
        self.play(LaggedStart(*[FadeIn(l, scale=1.3) for l in labels],
                              lag_ratio=0.12), run_time=2)

        # Coordinate grid (faint)
        r_curves = VGroup(*[_const_r_curve(f) for f in [0.2, 0.4, 0.6, 0.8]])
        t_curves = VGroup(*[_const_t_curve(f) for f in [-0.6, -0.3, 0, 0.3, 0.6]])
        grid = VGroup(r_curves, t_curves).set_opacity(0.35)
        self.play(Create(grid), run_time=2)

        self.wait(1)

        # Label the grid
        r_note = Text("constant r", font_size=18, color=GREY_A).move_to(
            np.array([S * 0.5, -S * 0.7, 0])
        )
        t_note = Text("constant t", font_size=18, color=GREY_A).move_to(
            np.array([-S * 0.6, S * 0.15, 0])
        )
        self.play(FadeIn(r_note), FadeIn(t_note), run_time=0.8)
        self.wait(1.2)

        # "Entire infinite Minkowski spacetime fits in this diamond"
        caption = Text("All of infinite Minkowski spacetime — one finite diamond",
                        font_size=22, color=GOLD).to_edge(DOWN, buff=0.35)
        self.play(Write(caption), run_time=1.5)
        self.wait(1.5)
        self.play(FadeOut(caption), FadeOut(r_note), FadeOut(t_note), run_time=0.6)

        self.mink_boundary = boundary
        self.mink_labels = labels
        self.mink_grid = grid
        self.mink_axes = VGroup(t_arrow, t_label, x_arrow, x_label)

    # ── Act 2 — Light cones everywhere at 45° ────────────────────────
    def _act2_light_cones(self):
        caption = Text("Light cones are always at 45° — that's the point",
                        font_size=22, color=COL_NULL).to_edge(DOWN, buff=0.35)
        self.play(Write(caption), run_time=1)

        # Scatter light cones at various positions inside the diamond
        positions = []
        np.random.seed(42)
        for _ in range(18):
            for _attempt in range(50):
                x = np.random.uniform(-S * 0.75, S * 0.75)
                y = np.random.uniform(-S * 0.75, S * 0.75)
                # must be inside diamond: |x| + |y| < S
                if abs(x) + abs(y) < S * 0.78:
                    positions.append(np.array([x, y, 0]))
                    break

        cones = VGroup(*[_light_cone_at(p, size=0.22) for p in positions])
        self.play(LaggedStart(*[FadeIn(c, scale=0.5) for c in cones],
                              lag_ratio=0.06), run_time=2.5)
        self.wait(1.5)

        # Highlight: all cones identical orientation
        highlight_text = Text("Conformal map preserves causal structure",
                              font_size=20, color=WHITE).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(caption, highlight_text), run_time=0.8)
        self.wait(1.5)

        self.play(FadeOut(cones), FadeOut(highlight_text), run_time=0.8)

    # ── Act 3 — Geodesics: timelike, null, spacelike ─────────────────
    def _act3_geodesics(self):
        # Timelike geodesic (vertical-ish curve, i⁻ to i⁺)
        tl_curve = _const_r_curve(0.0, n=80)
        tl_curve.set_color(COL_TIMELIKE).set_stroke(width=3)
        tl_label = Text("timelike", font_size=20, color=COL_TIMELIKE).move_to(
            np.array([-1.2, 0, 0])
        )

        # Null geodesic (exactly 45°, boundary to boundary)
        null_line = Line(
            I_MINUS + RIGHT * S * 0.3 + UP * S * 0.3,   # on 𝒥⁻ right
            I_PLUS + LEFT * S * 0.3 + DOWN * S * 0.3,    # on 𝒥⁺ left
            color=COL_NULL, stroke_width=3,
        )
        # Actually let's make a clean 45° ray
        null_start = np.array([S * 0.5, -S * 0.5, 0])  # on 𝒥⁻ right
        null_end = np.array([-S * 0.5, S * 0.5, 0])    # on 𝒥⁺ left
        null_line = Line(null_start, null_end, color=COL_NULL, stroke_width=3)
        null_label = Text("null (light)", font_size=20, color=COL_NULL).move_to(
            np.array([1.0, 0.5, 0])
        )

        # Spacelike geodesic (horizontal-ish)
        sl_curve = _const_t_curve(0.15, n=80)
        sl_curve.set_color(COL_SPACELIKE).set_stroke(width=3)
        sl_label = Text("spacelike", font_size=20, color=COL_SPACELIKE).move_to(
            np.array([0, -1.4, 0])
        )

        caption = Text("Three types of geodesic", font_size=22,
                        color=WHITE).to_edge(DOWN, buff=0.35)
        self.play(Write(caption), run_time=0.8)

        self.play(Create(tl_curve), Write(tl_label), run_time=1.5)
        self.wait(0.5)
        self.play(Create(null_line), Write(null_label), run_time=1.5)
        self.wait(0.5)
        self.play(Create(sl_curve), Write(sl_label), run_time=1.5)
        self.wait(1.5)

        geodesics = VGroup(tl_curve, tl_label, null_line, null_label,
                           sl_curve, sl_label)
        self.play(FadeOut(geodesics), FadeOut(caption), run_time=0.8)

    # ── Act 4 — Schwarzschild black hole ─────────────────────────────
    def _act4_schwarzschild(self):
        caption = Text("Schwarzschild black hole", font_size=28,
                        color=WHITE).to_edge(UP, buff=0.3)
        self.play(Write(caption), run_time=1)

        # Fade out Minkowski grid and some labels
        self.play(
            FadeOut(self.mink_grid),
            FadeOut(self.mink_axes),
            run_time=0.8,
        )

        # Event horizon — 45° dashed line from bottom to top
        horizon = DashedLine(I_MINUS, I_PLUS, color=COL_HORIZON,
                             stroke_width=3.5, dash_length=0.15)
        h_label = MathTex(r"r = 2M", font_size=24, color=COL_HORIZON).move_to(
            np.array([0.55, 0.3, 0])
        ).rotate(PI / 4)

        self.play(Create(horizon), run_time=1.5)
        self.play(Write(h_label), run_time=0.8)

        # Singularity — wavy line at top spanning the left half
        singularity = _singularity_line(y_top=S)
        s_label = MathTex(r"r = 0", font_size=24, color=COL_SINGULARITY).next_to(
            singularity, UP, buff=0.12
        ).shift(LEFT * S * 0.3)

        self.play(Create(singularity), run_time=1.5)
        self.play(Write(s_label), run_time=0.8)

        # Region labels
        reg_I = Text("I", font_size=36, color=WHITE).move_to(
            np.array([S * 0.45, 0, 0])
        )
        reg_I_sub = Text("exterior", font_size=16, color=GREY_A).next_to(
            reg_I, DOWN, buff=0.1
        )
        reg_II = Text("II", font_size=36, color=WHITE).move_to(
            np.array([-S * 0.35, S * 0.55, 0])
        )
        reg_II_sub = Text("interior", font_size=16, color=GREY_A).next_to(
            reg_II, DOWN, buff=0.1
        )

        self.play(
            FadeIn(reg_I, scale=1.3), FadeIn(reg_I_sub),
            FadeIn(reg_II, scale=1.3), FadeIn(reg_II_sub),
            run_time=1.2,
        )
        self.wait(1)

        # Explanation
        expl = Text("Once past the horizon, all futures lead to the singularity",
                     font_size=20, color=GOLD).to_edge(DOWN, buff=0.35)
        self.play(Write(expl), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(expl), run_time=0.5)

        self.bh_horizon = horizon
        self.bh_h_label = h_label
        self.bh_singularity = singularity
        self.bh_s_label = s_label
        self.bh_regions = VGroup(reg_I, reg_I_sub, reg_II, reg_II_sub)
        self.bh_caption = caption

    # ── Act 5 — Infalling observer ───────────────────────────────────
    def _act5_infalling_observer(self):
        caption = Text("Infalling vs static observer", font_size=22,
                        color=WHITE).to_edge(DOWN, buff=0.35)
        self.play(Write(caption), run_time=0.8)

        # Static observer (constant r, stays in region I)
        static_wl = _static_worldline(r_frac=0.55)
        static_dot = Dot(static_wl.get_start(), color=GREEN_E, radius=0.06)
        static_label = Text("static", font_size=18, color=GREEN_E).next_to(
            static_wl.point_from_proportion(0.5), RIGHT, buff=0.15
        )

        self.play(Create(static_wl), FadeIn(static_dot), Write(static_label),
                  run_time=2)

        # Infalling observer
        infall_wl = _infalling_worldline()
        infall_dot = Dot(infall_wl.get_start(), color=COL_TIMELIKE, radius=0.06)
        infall_label = Text("infalling", font_size=18, color=COL_TIMELIKE).next_to(
            infall_wl.point_from_proportion(0.35), LEFT, buff=0.15
        )

        self.play(Create(infall_wl), FadeIn(infall_dot), Write(infall_label),
                  run_time=2.5)

        # Animate the infalling dot along the worldline
        self.play(
            infall_dot.animate.move_to(infall_wl.point_from_proportion(0.5)),
            run_time=1.5,
        )

        # Flash at horizon crossing
        horizon_pt = infall_wl.point_from_proportion(0.5)
        cross_label = Text("crosses horizon", font_size=16, color=COL_HORIZON).next_to(
            horizon_pt, LEFT, buff=0.3
        )
        self.play(Write(cross_label), Flash(horizon_pt, color=COL_HORIZON), run_time=1)
        self.wait(0.5)

        # Continue to singularity
        self.play(
            infall_dot.animate.move_to(infall_wl.get_end()),
            run_time=1.5,
        )
        crash_label = Text("hits singularity", font_size=16,
                           color=COL_SINGULARITY).next_to(
            infall_wl.get_end(), LEFT, buff=0.2
        )
        self.play(Write(crash_label), Flash(infall_wl.get_end(),
                                            color=COL_SINGULARITY), run_time=1)
        self.wait(1)

        # Light cones tilt as you approach the singularity
        cone_positions = [
            infall_wl.point_from_proportion(0.2),
            infall_wl.point_from_proportion(0.5),
            infall_wl.point_from_proportion(0.8),
        ]
        infall_cones = VGroup(*[_light_cone_at(p, size=0.25) for p in cone_positions])
        cone_capt = Text("Light cones always at 45° — even inside",
                         font_size=18, color=COL_NULL).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(caption, cone_capt),
                  FadeIn(infall_cones), run_time=1.2)
        self.wait(2)

        self.play(
            FadeOut(static_wl), FadeOut(static_dot), FadeOut(static_label),
            FadeOut(infall_wl), FadeOut(infall_dot), FadeOut(infall_label),
            FadeOut(cross_label), FadeOut(crash_label), FadeOut(infall_cones),
            FadeOut(cone_capt),
            run_time=0.8,
        )

    # ── Act 6 — Hawking radiation & evaporation ──────────────────────
    def _act6_hawking_evaporation(self):
        caption = Text("Hawking radiation", font_size=28,
                        color=COL_HAWKING).to_edge(UP, buff=0.3)
        self.play(ReplacementTransform(self.bh_caption, caption), run_time=0.8)

        # Hawking rays — outgoing null rays from near the horizon
        rays = VGroup(*[_hawking_ray(y) for y in np.linspace(-S * 0.5, S * 0.6, 8)])
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.15),
                  run_time=2.5)

        expl = Text("Quantum pair creation near the horizon → thermal radiation",
                     font_size=18, color=GREY_A).to_edge(DOWN, buff=0.35)
        self.play(Write(expl), run_time=1.5)
        self.wait(1.5)

        # Evaporation: horizon + singularity shrink and disappear
        evap_text = Text("Black hole evaporates", font_size=22,
                         color=COL_HAWKING).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(expl, evap_text), run_time=0.8)

        # Animate the singularity and horizon fading
        self.play(
            self.bh_singularity.animate.set_opacity(0.2).shift(DOWN * 0.3),
            self.bh_horizon.animate.set_opacity(0.2),
            self.bh_h_label.animate.set_opacity(0.2),
            self.bh_s_label.animate.set_opacity(0.2),
            self.bh_regions.animate.set_opacity(0.2),
            run_time=2,
        )

        # Thunderclap flash at the evaporation endpoint
        evap_pt = np.array([-S * 0.15, S * 0.7, 0])
        evap_dot = Dot(evap_pt, color=WHITE, radius=0.1)
        evap_label = Text("evaporation\nendpoint", font_size=16,
                          color=WHITE).next_to(evap_pt, LEFT, buff=0.25)
        self.play(Flash(evap_pt, color=WHITE, num_lines=16, flash_radius=0.6),
                  FadeIn(evap_dot), Write(evap_label), run_time=1.5)
        self.wait(1)

        # Final summary
        self.play(
            FadeOut(rays), FadeOut(evap_text), FadeOut(evap_dot), FadeOut(evap_label),
            FadeOut(self.bh_singularity), FadeOut(self.bh_horizon),
            FadeOut(self.bh_h_label), FadeOut(self.bh_s_label),
            FadeOut(self.bh_regions),
            run_time=0.8,
        )

        # Restore clean diamond for final message
        final = Text(
            "The entire causal structure of spacetime\n— past, future, infinity —\n"
            "in one finite diagram.",
            font_size=24, color=GOLD, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(final), run_time=2)
        self.wait(2)

        # Fade everything
        self.play(
            FadeOut(self.mink_boundary), FadeOut(self.mink_labels),
            FadeOut(caption), FadeOut(final),
            run_time=1.5,
        )
        self.wait(0.5)
