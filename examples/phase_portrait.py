"""Phase Portrait — Hamiltonian Flow of a Simple Pendulum.

The simple pendulum has Hamiltonian
    H(theta, p) = p^2/(2 m l^2) - m g l cos(theta)

Its phase portrait features:
• Elliptic fixed point at (0, 0) — stable equilibrium
• Hyperbolic fixed point at (pi, 0) — unstable equilibrium
• Separatrices dividing libration from rotation
• Liouville's theorem: phase-space area is preserved

Acts
----
1. Title card
2. Draw phase-space axes + potential energy
3. Flow field (vector arrows)
4. Libration orbits (small oscillations)
5. Separatrix (homoclinic orbit)
6. Rotation orbits (going over the top)
7. Liouville area preservation demo
8. Summary card

Run
---
    manim -pql examples/phase_portrait.py PhasePortrait
    manim -qh  examples/phase_portrait.py PhasePortrait
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
    Arrow,
    Rectangle,
    Circle,
    Polygon,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Uncreate,
    Transform,
    Indicate,
    Flash,
    SurroundingRectangle,
    Axes,
    StreamLines,
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
# Pendulum Hamiltonian
# ═══════════════════════════════════════════════════════════════════════════
# Normalised: H = p^2/2 - cos(theta)
# Hamilton's equations:  dtheta/dt = p,  dp/dt = -sin(theta)


def _hamiltonian(theta, p):
    """H = p^2/2 - cos(theta)."""
    return 0.5 * p ** 2 - np.cos(theta)


def _flow_field(pos):
    """Phase-space velocity at (theta, p)."""
    theta, p = pos[0], pos[1]
    return np.array([p, -np.sin(theta), 0])


def _integrate_orbit(theta0, p0, dt=0.02, n_steps=500):
    """RK4 integration of pendulum orbit."""
    pts = [(theta0, p0)]
    th, p = theta0, p0
    for _ in range(n_steps):
        k1t, k1p = p, -np.sin(th)
        k2t, k2p = p + 0.5 * dt * k1p, -np.sin(th + 0.5 * dt * k1t)
        k3t, k3p = p + 0.5 * dt * k2p, -np.sin(th + 0.5 * dt * k2t)
        k4t, k4p = p + dt * k3p, -np.sin(th + dt * k3t)
        th += dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        p += dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        pts.append((th, p))
    return np.array(pts)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class PhasePortrait(Scene):
    """Phase portrait of the simple pendulum."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("Phase Portrait", font_size=48, color=GOLD)
        sub = Text("Hamiltonian flow of a simple pendulum",
                    font_size=24, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        ham = MathTex(
            r"H(\theta, p) = \frac{p^2}{2} - \cos\theta",
            font_size=30, color=YELLOW,
        )
        ham.next_to(sub, DOWN, buff=0.3)

        self.play(Write(ttl), run_time=1.2)                              # 1
        self.play(FadeIn(sub), run_time=0.8)                             # 2
        self.play(Write(ham), run_time=1)                                # 3
        self.wait(0.8)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(ham))              # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Axes
        # ─────────────────────────────────────────────────────────────────
        axes = Axes(
            x_range=[-PI - 0.5, PI + 0.5, PI / 2],
            y_range=[-3.5, 3.5, 1],
            x_length=10,
            y_length=6,
            axis_config={"color": GREY_D, "stroke_width": 1.5},
            tips=False,
        )

        x_lbl = MathTex(r"\theta", font_size=22, color=WHITE)
        x_lbl.next_to(axes.x_axis, RIGHT, buff=0.1)
        y_lbl = MathTex(r"p", font_size=22, color=WHITE)
        y_lbl.next_to(axes.y_axis, UP, buff=0.1)

        # Tick labels for -π, -π/2, π/2, π
        ticks = VGroup(
            MathTex(r"-\pi", font_size=14).next_to(
                axes.c2p(-PI, 0), DOWN, buff=0.1),
            MathTex(r"\pi", font_size=14).next_to(
                axes.c2p(PI, 0), DOWN, buff=0.1),
        )

        ham_eq = MathTex(
            r"H = \frac{p^2}{2} - \cos\theta",
            font_size=18, color=GOLD,
        )
        ham_eq.to_corner(UP + LEFT, buff=0.25)

        self.play(
            Create(axes), FadeIn(x_lbl), FadeIn(y_lbl),
            FadeIn(ticks), FadeIn(ham_eq),
            run_time=1.2,
        )                                                                 # 5

        # Fixed points
        fp_stable = Dot(axes.c2p(0, 0), radius=0.08, color=GREEN)
        fp_unstable_l = Dot(axes.c2p(-PI, 0), radius=0.08, color=RED)
        fp_unstable_r = Dot(axes.c2p(PI, 0), radius=0.08, color=RED)

        fp_lbl_s = MathTex(r"\text{stable}", font_size=12, color=GREEN)
        fp_lbl_s.next_to(fp_stable, DOWN + RIGHT, buff=0.08)
        fp_lbl_u = MathTex(r"\text{unstable}", font_size=12, color=RED)
        fp_lbl_u.next_to(fp_unstable_r, UP + RIGHT, buff=0.08)

        self.play(
            FadeIn(fp_stable), FadeIn(fp_unstable_l), FadeIn(fp_unstable_r),
            FadeIn(fp_lbl_s), FadeIn(fp_lbl_u),
            run_time=0.6,
        )                                                                 # 6

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Flow field arrows
        # ─────────────────────────────────────────────────────────────────
        field_arrows = VGroup()
        for th_val in np.linspace(-PI, PI, 13):
            for p_val in np.linspace(-3, 3, 9):
                v = _flow_field(np.array([th_val, p_val, 0]))
                spd = np.linalg.norm(v[:2])
                if spd < 0.1:
                    continue
                scale = min(spd * 0.12, 0.35)
                start = axes.c2p(th_val, p_val)
                end = axes.c2p(
                    th_val + v[0] * scale / spd * 0.6,
                    p_val + v[1] * scale / spd * 0.6,
                )
                arr = Arrow(
                    start=start, end=end,
                    color=GREY_A,
                    stroke_width=1.2,
                    max_tip_length_to_length_ratio=0.3,
                    buff=0,
                )
                field_arrows.add(arr)

        self.play(
            *[Create(a) for a in field_arrows],
            run_time=1.5,
        )                                                                 # 7

        flow_note = MathTex(
            r"\dot{\theta} = p, \quad \dot{p} = -\sin\theta",
            font_size=18, color=TEAL,
        )
        flow_note.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(flow_note), run_time=0.4)                       # 8

        self.wait(0.5)
        self.play(FadeOut(field_arrows), run_time=0.4)                   # 9

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Libration orbits (small oscillations)
        # ─────────────────────────────────────────────────────────────────
        lib_colors = [BLUE, TEAL, GREEN]
        lib_energies = [0.3, 0.7, 0.95]  # H values < 1 (separatrix at H=1)

        lib_label = MathTex(
            r"\text{Libration orbits (} H < 1 \text{)}",
            font_size=18, color=BLUE,
        )
        lib_label.to_corner(UP + RIGHT, buff=0.25)
        self.play(FadeIn(lib_label), run_time=0.3)                       # 10

        lib_curves = VGroup()
        for idx, E in enumerate(lib_energies):
            # For libration: H = p^2/2 - cos(theta) = E
            # p = ±sqrt(2(E + cos(theta)))
            # theta range: |theta| < arccos(-E)
            th_max = np.arccos(max(-E, -1.0))
            th_arr = np.linspace(-th_max + 0.01, th_max - 0.01, 200)
            p_upper = np.sqrt(np.maximum(2 * (E + np.cos(th_arr)), 0))
            p_lower = -p_upper

            # Upper branch
            pts_u = [axes.c2p(th_arr[i], p_upper[i])
                      for i in range(len(th_arr))]
            # Lower branch (reversed for closed loop)
            pts_l = [axes.c2p(th_arr[i], p_lower[i])
                      for i in range(len(th_arr) - 1, -1, -1)]

            all_pts = pts_u + pts_l + [pts_u[0]]
            curve = VMobject(color=lib_colors[idx], stroke_width=2.5)
            curve.set_points_as_corners(
                [np.array(p) for p in all_pts]
            )
            lib_curves.add(curve)
            self.play(Create(curve), run_time=0.8)                # 11-13

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Separatrix
        # ─────────────────────────────────────────────────────────────────
        sep_label = MathTex(
            r"\text{Separatrix } (H = 1)",
            font_size=18, color=YELLOW,
        )
        sep_label.next_to(lib_label, DOWN, buff=0.15)
        self.play(FadeIn(sep_label), run_time=0.3)                       # 14

        # Separatrix: H = 1 → p = ±2cos(theta/2)
        th_sep = np.linspace(-PI + 0.05, PI - 0.05, 300)
        p_sep_u = 2 * np.cos(th_sep / 2)
        p_sep_l = -2 * np.cos(th_sep / 2)

        sep_upper = VMobject(color=YELLOW, stroke_width=3)
        sep_upper.set_points_as_corners(
            [np.array(axes.c2p(th_sep[i], p_sep_u[i]))
             for i in range(len(th_sep))]
        )
        sep_lower = VMobject(color=YELLOW, stroke_width=3)
        sep_lower.set_points_as_corners(
            [np.array(axes.c2p(th_sep[i], p_sep_l[i]))
             for i in range(len(th_sep))]
        )

        self.play(Create(sep_upper), Create(sep_lower), run_time=1.2)   # 15

        self.play(
            Indicate(fp_unstable_l, color=YELLOW),
            Indicate(fp_unstable_r, color=YELLOW),
            run_time=0.6,
        )                                                                 # 16

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Rotation orbits (going over the top)
        # ─────────────────────────────────────────────────────────────────
        rot_label = MathTex(
            r"\text{Rotation orbits (} H > 1 \text{)}",
            font_size=18, color=ORANGE,
        )
        rot_label.next_to(sep_label, DOWN, buff=0.15)
        self.play(FadeIn(rot_label), run_time=0.3)                       # 17

        rot_energies = [1.5, 2.5]
        rot_curves = VGroup()
        for idx, E in enumerate(rot_energies):
            th_r = np.linspace(-PI, PI, 300)
            p_r = np.sqrt(np.maximum(2 * (E + np.cos(th_r)), 0))
            col = [ORANGE, RED_E][idx]

            # Upper rotation curve
            curve_u = VMobject(color=col, stroke_width=2)
            curve_u.set_points_as_corners(
                [np.array(axes.c2p(th_r[i], p_r[i]))
                 for i in range(len(th_r))]
            )
            # Lower rotation curve
            curve_l = VMobject(color=col, stroke_width=2)
            curve_l.set_points_as_corners(
                [np.array(axes.c2p(th_r[i], -p_r[i]))
                 for i in range(len(th_r))]
            )
            rot_curves.add(curve_u, curve_l)
            self.play(Create(curve_u), Create(curve_l),
                      run_time=0.8)                               # 18-19

        self.wait(0.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Liouville area preservation
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(flow_note),
            run_time=0.3,
        )                                                                 # 20

        liouville = MathTex(
            r"\text{Liouville: } "
            r"\nabla \cdot (\dot{\theta}, \dot{p}) = 0"
            r"\;\;\Rightarrow\;\;\text{area preserved}",
            font_size=18, color=GREEN,
        )
        liouville.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(liouville), run_time=0.6)                       # 21

        # Animate a small square of initial conditions evolving
        # Show two rectangles — "before" and "after" — same area, deformed
        sq_cx, sq_cy = 0.5, 0.8
        sq_w, sq_h = 0.4, 0.4
        corners_init = [
            (sq_cx - sq_w / 2, sq_cy - sq_h / 2),
            (sq_cx + sq_w / 2, sq_cy - sq_h / 2),
            (sq_cx + sq_w / 2, sq_cy + sq_h / 2),
            (sq_cx - sq_w / 2, sq_cy + sq_h / 2),
        ]

        patch_init = Polygon(
            *[axes.c2p(c[0], c[1]) for c in corners_init],
            color=GREEN, fill_opacity=0.3, stroke_width=2,
        )

        self.play(Create(patch_init), run_time=0.5)                      # 22

        # Evolve each corner forward
        T_evolve = 1.5
        n_sub = int(T_evolve / 0.01)
        corners_final = []
        for c in corners_init:
            orbit = _integrate_orbit(c[0], c[1], dt=0.01, n_steps=n_sub)
            corners_final.append((orbit[-1, 0], orbit[-1, 1]))

        patch_final = Polygon(
            *[axes.c2p(c[0], c[1]) for c in corners_final],
            color=YELLOW, fill_opacity=0.3, stroke_width=2,
        )

        self.play(Transform(patch_init, patch_final), run_time=1.5)      # 23

        area_note = MathTex(
            r"\text{Shape changes, area stays the same!}",
            font_size=16, color=YELLOW,
        )
        area_note.next_to(liouville, UP, buff=0.1)
        self.play(FadeIn(area_note), run_time=0.4)                       # 24

        self.wait(1)

        # ─────────────────────────────────────────────────────────────────
        # Act 8 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1,
        )                                                                 # 25

        card_title = Text(
            "Phase Portrait — Pendulum", font_size=34, color=GOLD,
        )
        card_title.to_edge(UP, buff=0.5)
        self.play(Write(card_title))                                      # 26

        bullets = VGroup(
            MathTex(
                r"H = p^2/2 - \cos\theta"
                r"\;\;\text{(simple pendulum)}",
                font_size=18,
            ),
            MathTex(
                r"\text{Elliptic fixed point at } (0,0)"
                r"\text{ — stable equilibrium}",
                font_size=18, color=GREEN,
            ),
            MathTex(
                r"\text{Hyperbolic fixed point at } (\pm\pi,0)"
                r"\text{ — unstable (saddle)}",
                font_size=18, color=RED,
            ),
            MathTex(
                r"\text{Separatrix } H=1"
                r"\text{ divides libration from rotation}",
                font_size=18, color=YELLOW,
            ),
            MathTex(
                r"\text{Liouville: } "
                r"\text{Hamiltonian flow preserves phase-space area}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{Symplectic structure } \omega = d\theta \wedge dp"
                r"\text{ is the hidden geometry}",
                font_size=18, color=ORANGE,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, bullets),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.play(FadeIn(box), run_time=0.5)                             # 27
        for b in bullets:
            self.play(FadeIn(b), run_time=0.6)                           # 28-33

        self.wait(2)
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )                                                                 # 34
