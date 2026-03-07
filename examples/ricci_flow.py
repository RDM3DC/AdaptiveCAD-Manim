"""Ricci Flow — A Bumpy Surface Smooths Toward Constant Curvature.

Under the Ricci flow  dg/dt = -2 Ric, regions of high positive curvature
shrink while negatively-curved regions expand.  A bumpy 2-sphere evolves
toward a perfect round sphere — the essence of Perelman's proof of the
Poincare conjecture.

We animate a 2D cross-section: a closed curve whose local curvature
drives contraction/expansion, smoothing toward a circle.

Acts
----
1. Title card
2. Initial bumpy curve + curvature heat-map
3. Ricci flow evolution (curve smooths step-by-step)
4. Curvature histogram flattens
5. Perelman & Poincare reveal
6. Summary card

Run
---
    manim -pql examples/ricci_flow.py RicciFlow
    manim -qh  examples/ricci_flow.py RicciFlow
"""

from __future__ import annotations

import numpy as np
import math
from manim import (
    Scene,
    VMobject,
    VGroup,
    Dot,
    Line,
    Rectangle,
    Circle,
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
    Arrow,
    Axes,
    BarChart,
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
# Curve-shortening / 2-D Ricci-flow helpers
# ═══════════════════════════════════════════════════════════════════════════
N_PTS = 200           # points on the closed curve
N_STEPS = 30          # flow steps to animate
DT = 0.004            # time-step per flow iteration
SCALE = 2.0           # display scale


def _initial_curve():
    """Bumpy closed curve: circle + Fourier bumps."""
    t = np.linspace(0, TAU, N_PTS, endpoint=False)
    r = 1.0 + 0.30 * np.sin(3 * t) + 0.20 * np.cos(5 * t) + 0.15 * np.sin(7 * t)
    x = r * np.cos(t) * SCALE
    y = r * np.sin(t) * SCALE
    return np.column_stack([x, y])


def _curvature(pts):
    """Discrete unsigned curvature at each point of a closed polygon."""
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        l1 = np.linalg.norm(v1) + 1e-12
        l2 = np.linalg.norm(v2) + 1e-12
        kappa[i] = 2.0 * cross / (l1 * l2 * (l1 + l2))
    return kappa


def _flow_step(pts, dt=DT):
    """One step of curve-shortening flow (2D analogue of Ricci flow)."""
    n = len(pts)
    new = pts.copy()
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        # Laplacian ~ (p0 + p2 - 2*p1) is the mean-curvature normal
        lap = (p0 + p2 - 2 * p1)
        new[i] = p1 + dt * lap / (np.linalg.norm(lap) + 1e-12) * np.linalg.norm(lap)
    return new


def _pts_to_vmob(pts, color=BLUE, width=2.5):
    """Convert Nx2 array into a closed VMobject."""
    mob = VMobject(color=color, stroke_width=width)
    pts3 = np.column_stack([pts, np.zeros(len(pts))])
    # Close the curve
    pts3 = np.vstack([pts3, pts3[0:1]])
    mob.set_points_as_corners(pts3)
    return mob


def _curvature_color(k, k_max):
    """Map curvature to colour: blue (low) → yellow (mid) → red (high)."""
    frac = min(abs(k) / (k_max + 1e-12), 1.0)
    if frac < 0.5:
        return interpolate_color(BLUE, YELLOW, frac * 2)
    return interpolate_color(YELLOW, RED, (frac - 0.5) * 2)


def _colored_curve(pts, kappa, k_max, width=3.0):
    """Build a VGroup of short line segments coloured by curvature."""
    segs = VGroup()
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        p1 = np.array([pts[i][0], pts[i][1], 0])
        p2 = np.array([pts[j][0], pts[j][1], 0])
        col = _curvature_color(kappa[i], k_max)
        seg = Line(p1, p2, color=col, stroke_width=width)
        segs.add(seg)
    return segs


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class RicciFlow(Scene):
    """A bumpy closed curve smooths under curve-shortening flow."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("Ricci Flow", font_size=50, color=GOLD)
        sub = Text("Smoothing geometry toward constant curvature",
                    font_size=24, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"\frac{\partial g}{\partial t} = -2\,\text{Ric}",
            font_size=36, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.35)

        self.play(Write(ttl), run_time=1.2)                              # 1
        self.play(FadeIn(sub), run_time=0.8)                             # 2
        self.play(Write(eq), run_time=1)                                 # 3
        self.wait(1)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))               # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Initial bumpy curve with curvature heat-map
        # ─────────────────────────────────────────────────────────────────
        pts = _initial_curve()
        kappa = _curvature(pts)
        k_max = np.max(np.abs(kappa))

        curve_mob = _colored_curve(pts, kappa, k_max)
        curve_mob.shift(LEFT * 1.5)

        flow_eq = MathTex(
            r"\frac{\partial g}{\partial t} = -2\,\text{Ric}",
            font_size=22, color=GOLD,
        )
        flow_eq.to_corner(UP + LEFT, buff=0.25)

        step_lbl = Text("Step 0", font_size=18, color=GREY_A)
        step_lbl.to_corner(UP + RIGHT, buff=0.25)

        legend_title = Text("Curvature", font_size=14, color=WHITE)
        legend_title.to_edge(RIGHT, buff=0.3).shift(DOWN * 0.5)
        legend_hi = Text("high", font_size=12, color=RED)
        legend_lo = Text("low", font_size=12, color=BLUE)
        legend_hi.next_to(legend_title, DOWN, buff=0.1)
        legend_lo.next_to(legend_hi, DOWN, buff=0.08)

        self.play(
            Create(curve_mob),
            Write(flow_eq),
            FadeIn(step_lbl),
            FadeIn(legend_title), FadeIn(legend_hi), FadeIn(legend_lo),
            run_time=1.5,
        )                                                                 # 5

        note = MathTex(
            r"\text{High curvature regions contract faster}",
            font_size=16, color=YELLOW,
        )
        note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note), run_time=0.5)                            # 6

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Ricci flow evolution
        # ─────────────────────────────────────────────────────────────────
        for step in range(1, N_STEPS + 1):
            # Multiple sub-steps per visual frame for smoother flow
            for _ in range(8):
                pts = _flow_step(pts)

            kappa = _curvature(pts)
            k_max_new = max(np.max(np.abs(kappa)), 0.01)

            new_curve = _colored_curve(pts, kappa, k_max_new)
            new_curve.shift(LEFT * 1.5)

            new_step = Text(f"Step {step}", font_size=18, color=GREY_A)
            new_step.to_corner(UP + RIGHT, buff=0.25)

            self.play(
                Transform(curve_mob, new_curve),
                Transform(step_lbl, new_step),
                run_time=0.15,
            )                                                     # 7-36

        self.play(FadeOut(note), run_time=0.3)                           # 37

        # Final state label
        final_lbl = Text(
            "Constant curvature (round circle)",
            font_size=18, color=GREEN,
        )
        final_lbl.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(final_lbl), run_time=0.5)                       # 38

        self.play(
            Indicate(curve_mob, color=GOLD, scale_factor=1.03),
            run_time=0.8,
        )                                                                 # 39

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Perelman & Poincare
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(curve_mob), FadeOut(step_lbl), FadeOut(final_lbl),
            FadeOut(legend_title), FadeOut(legend_hi), FadeOut(legend_lo),
            run_time=0.5,
        )                                                                 # 40

        poincare_title = Text(
            "From Ricci Flow to the Poincare Conjecture",
            font_size=24, color=GOLD,
        )
        poincare_title.to_edge(UP, buff=0.6)
        self.play(FadeIn(poincare_title), run_time=0.6)                  # 41

        steps_text = VGroup(
            MathTex(
                r"\text{1. Start with any closed simply-connected 3-manifold}",
                font_size=18, color=WHITE,
            ),
            MathTex(
                r"\text{2. Run Ricci flow: } "
                r"\partial_t g = -2\,\text{Ric}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{3. Surgeries handle singularities (neck pinches)}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\text{4. Flow converges to constant positive curvature}",
                font_size=18, color=GREEN,
            ),
            MathTex(
                r"\text{5. Constant positive curvature } \Rightarrow S^3"
                r"\text{ (round 3-sphere)}",
                font_size=18, color=YELLOW,
            ),
            MathTex(
                r"\therefore\; \text{Every simply-connected closed 3-manifold}"
                r" \cong S^3",
                font_size=20, color=GOLD,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        steps_text.next_to(poincare_title, DOWN, buff=0.3)

        for line in steps_text:
            self.play(FadeIn(line), run_time=0.7)                         # 42-47

        note2 = Text(
            "Perelman (2003) — declined Fields Medal",
            font_size=16, color=GREY_A,
        )
        note2.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note2), run_time=0.5)                           # 48

        self.wait(1.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1,
        )                                                                 # 49

        card_title = Text("Ricci Flow", font_size=34, color=GOLD)
        card_title.to_edge(UP, buff=0.5)
        self.play(Write(card_title))                                      # 50

        bullets = VGroup(
            MathTex(
                r"\partial_t g_{ij} = -2\,R_{ij}"
                r"\;\;\text{(Ricci flow equation)}",
                font_size=18,
            ),
            MathTex(
                r"\text{High curvature contracts, low curvature expands}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{2D: bumpy curve } \to \text{ round circle}",
                font_size=18,
            ),
            MathTex(
                r"\text{3D: any simply-connected manifold } \to S^3",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\text{Hamilton (1982) introduced; "
                r"Perelman (2003) completed}",
                font_size=18, color=YELLOW,
            ),
            MathTex(
                r"\text{Proves the Poincare conjecture: "
                r"topology from geometry}",
                font_size=18, color=GREEN,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, bullets),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.play(FadeIn(box), run_time=0.5)                             # 51
        for b in bullets:
            self.play(FadeIn(b), run_time=0.6)                           # 52-57

        self.wait(2)
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )                                                                 # 58
