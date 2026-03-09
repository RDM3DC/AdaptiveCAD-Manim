"""Thouless Charge Pump — Winding on the Bloch Torus.

In a Thouless pump a 1-D insulator has its Hamiltonian cycled
adiabatically through parameter space.  The Berry phase accumulated
over one Brillouin zone equals the Chern number, pumping exactly
one charge per cycle.

Geometry
--------
  Left:  Bloch torus (k, t) ∈ T² with a closed path winding once
         around the poloidal direction (one pump cycle)
  Right: Charge bar showing pumped charge Q = w = integer

Acts
----
1. Title card
2. Draw Bloch torus T²
3. Trace closed winding paths on torus: w=0 (contractible), w=1, w=2
4. Show pumped charge counter for each winding
5. Camera orbit
6. Summary card

Run
---
    manim -pql examples/thouless_pump.py ThoulessPump
    manim -qh  examples/thouless_pump.py ThoulessPump
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Line3D,
    Dot3D,
    ParametricFunction,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Uncreate,
    ReplacementTransform,
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
    interpolate_color,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Geometry constants
# ═══════════════════════════════════════════════════════════════════════════
TORUS_R = 1.6     # major radius
TORUS_r = 0.55    # minor radius
CX = -1.5         # torus x-centre
_RAIN = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]


def _torus_func(R, r, cx=0.0):
    """Standard torus parametric.  u = toroidal, v = poloidal."""

    def func(u, v):
        return np.array([
            cx + (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v),
        ])

    return func


def _torus_curve(R, r, n_wind, cx=0.0, n_pts=200):
    """Curve on the torus winding n_wind times in the poloidal direction
    while going once around toroidally."""

    def func(t):
        u = t
        v = n_wind * t
        return np.array([
            cx + (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v),
        ])

    return func


def _rainbow(frac):
    n = len(_RAIN) - 1
    idx = frac * n
    lo = min(int(idx), n - 1)
    return interpolate_color(_RAIN[lo], _RAIN[lo + 1], idx - lo)


class ThoulessPump(ThreeDScene):
    """Thouless charge pump — winding paths on the Bloch torus."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Thouless Charge Pump", font_size=46, color=GOLD)
        sub = Text("Winding on the Bloch torus T²",
                    font_size=22, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"Q = \frac{1}{2\pi}\oint_{\mathrm{BZ}}"
            r"\mathcal{A}(k)\,dk = w \in \mathbb{Z}",
            font_size=26, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Draw Bloch torus ─────────────────────────────────────
        self.set_camera_orientation(phi=65 * DEG, theta=-50 * DEG)

        torus = Surface(
            _torus_func(TORUS_R, TORUS_r, cx=CX),
            u_range=[0, TAU], v_range=[0, TAU],
            resolution=(48, 24),
            fill_color=BLUE_E, fill_opacity=0.10,
            stroke_width=0.3, stroke_color=BLUE_D,
        )

        # axis labels
        lbl_k = MathTex(r"k", font_size=18, color=RED)
        lbl_k.move_to(LEFT * 4.0 + DOWN * 0.3)
        lbl_t = MathTex(r"t", font_size=18, color=GREEN)
        lbl_t.move_to(LEFT * 0.5 + UP * 1.5)
        for lbl in [lbl_k, lbl_t]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(Create(torus), FadeIn(lbl_k), FadeIn(lbl_t),
                  run_time=1.5)

        # ─── Act 3  Winding paths ────────────────────────────────────────
        charge_x = 3.5
        all_curves = []
        frame_lbl_list = [lbl_k, lbl_t]

        for w in [0, 1, 2]:
            if w == 0:
                curve = ParametricFunction(
                    _torus_curve(TORUS_R, TORUS_r, 0, cx=CX),
                    t_range=[0, TAU],
                    color=GREY_A, stroke_width=3,
                )
            else:
                # Rainbow-coloured segments
                N_SEG = 80
                cf = _torus_curve(TORUS_R, TORUS_r, w, cx=CX)
                ts = np.linspace(0, TAU, N_SEG + 1)
                curve = VGroup()
                for i in range(N_SEG):
                    seg = Line3D(
                        cf(ts[i]), cf(ts[i + 1]),
                        color=_rainbow(i / N_SEG),
                        thickness=0.016,
                    )
                    curve.add(seg)

            # Winding / charge label
            w_lbl = MathTex(
                r"w = " + str(w),
                font_size=24, color=YELLOW,
            )
            w_lbl.move_to(RIGHT * charge_x + UP * 1.5)

            q_lbl = MathTex(
                r"Q = " + str(w) + r"e",
                font_size=22,
                color=GREEN if w > 0 else GREY_A,
            )
            q_lbl.move_to(RIGHT * charge_x + UP * 0.8)

            desc = Text(
                "contractible" if w == 0
                else f"{w} poloidal winding{'s' if w > 1 else ''}",
                font_size=14, color=GREY_A,
            )
            desc.move_to(RIGHT * charge_x + UP * 0.2)

            for lbl in [w_lbl, q_lbl, desc]:
                self.add_fixed_in_frame_mobjects(lbl)

            if w == 0:
                self.play(Create(curve), FadeIn(w_lbl),
                          FadeIn(q_lbl), FadeIn(desc),
                          run_time=1.5)
            else:
                self.play(
                    Create(curve, lag_ratio=0.02 if isinstance(curve, VGroup) else 0),
                    FadeIn(w_lbl), FadeIn(q_lbl), FadeIn(desc),
                    run_time=2,
                )

            self.wait(0.6)

            # keep track for cleanup
            all_curves.append(curve)
            frame_lbl_list.extend([w_lbl, q_lbl, desc])

            if w < 2:
                self.play(
                    FadeOut(curve), FadeOut(w_lbl),
                    FadeOut(q_lbl), FadeOut(desc),
                    run_time=0.5,
                )
                # pop the ones we just faded
                all_curves.pop()
                for _ in range(3):
                    frame_lbl_list.pop()

        # ─── Act 4  Camera orbit ─────────────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.14)
        self.wait(3.5)
        self.stop_ambient_camera_rotation()

        # ─── Act 5  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in frame_lbl_list],
            run_time=0.8,
        )
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Thouless Pump", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"H(k, t)\;\text{cycled adiabatically on }T^2",
                font_size=19,
            ),
            MathTex(
                r"w = 0:\;\text{contractible loop — no charge pumped}",
                font_size=19, color=GREY_A,
            ),
            MathTex(
                r"w = 1,2,\ldots:\;\text{poloidal winding — }"
                r"Q = we\;\text{pumped per cycle}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\text{Chern number } C_1 = w"
                r"\;\;\text{(topologically quantised)}",
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
