"""Hopf Fibration — S³ → S²

Every point on the 2-sphere maps to a circle (fibre) in 3-space
via stereographic projection.  All fibres are pairwise linked.
Fibres over a latitude circle on S² form a torus in ℝ³ — different
latitudes give nested tori (the Villarceau decomposition).

    h(z₁, z₂) = (2 Re(z̄₁z₂),  2 Im(z̄₁z₂),  |z₁|² − |z₂|²)  ∈ S²

Acts
----
1. Title card
2. Single fibre — circle over the north pole of S²
3. Six equatorial fibres — linked circles, rainbow-coloured
4. Clifford torus — all equatorial fibres as a surface
5. Nested tori — three latitudes → three nested tori
6. Camera orbit — full beauty reveal
7. Summary card — π₃(S²) = ℤ, fibre bundles

Run
---
    manim -pql examples/hopf_fibration.py HopfFibration
    manim -qh  examples/hopf_fibration.py HopfFibration
"""

from __future__ import annotations

import colorsys
import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    ParametricFunction,
    Circle,
    Dot,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Indicate,
    SurroundingRectangle,
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
    GREEN,
    BLUE,
    BLUE_D,
    BLUE_E,
    ORANGE,
    GOLD,
    TEAL,
    GREY,
    config,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
SCALE = 0.70  # global scale for stereographic projection


# ═══════════════════════════════════════════════════════════════════════════
# Colour helpers
# ═══════════════════════════════════════════════════════════════════════════

def _hue(h, s=0.88, v=0.95):
    """HSV → hex colour string.  h ∈ [0, 1]."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ═══════════════════════════════════════════════════════════════════════════
# Hopf fibre  (stereographic S³ → ℝ³)
# ═══════════════════════════════════════════════════════════════════════════

def _fibre_func(theta_b: float, phi_b: float):
    """Return  t ↦ [x,y,z]  for the Hopf fibre over (θ_b, φ_b) ∈ S².

    Parameters
    ----------
    theta_b : colatitude on S²  (0 = north pole,  π = south pole)
    phi_b   : azimuth on S²

    Stereographic projection from the south pole of S³,  i.e.
        from  (0, 0, 0, −1).
    """
    ct = np.cos(theta_b / 2.0)
    st = np.sin(theta_b / 2.0)

    def f(t):
        denom = 1.0 + st * np.sin(t + phi_b)
        d = max(denom, 0.02)
        return SCALE * np.array([
            ct * np.cos(t) / d,
            ct * np.sin(t) / d,
            st * np.cos(t + phi_b) / d,
        ])

    return f


def _torus_func(theta_b: float):
    """Parametric surface  (u, v) ↦ [x,y,z]  for the torus of fibres
    over the latitude circle θ = θ_b on S².

    u = φ_b ∈ [0, 2π)   (which base point)
    v = α  ∈ [0, 2π)    (position along fibre)
    """
    ct = np.cos(theta_b / 2.0)
    st = np.sin(theta_b / 2.0)

    def f(u, v):
        denom = 1.0 + st * np.sin(v + u)
        d = max(denom, 0.02)
        return SCALE * np.array([
            ct * np.cos(v) / d,
            ct * np.sin(v) / d,
            st * np.cos(v + u) / d,
        ])

    return f


def _make_fibre(theta_b, phi_b, color, sw=2.5):
    """Create a ParametricFunction for one Hopf fibre."""
    return ParametricFunction(
        _fibre_func(theta_b, phi_b),
        t_range=[0, TAU, TAU / 200],
        color=color,
        stroke_width=sw,
    )


def _make_torus(theta_b, color, opacity=0.18, res=32):
    """Create a Surface for the Hopf torus at colatitude θ_b."""
    return Surface(
        _torus_func(theta_b),
        u_range=[0, TAU],
        v_range=[0, TAU],
        resolution=(res, res),
        fill_color=color,
        fill_opacity=opacity,
        stroke_color=color,
        stroke_width=0.3,
        stroke_opacity=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Base-sphere inset helpers  (2D, fixed-in-frame)
# ═══════════════════════════════════════════════════════════════════════════

def _base_dot(theta_b, phi_b, center, radius, color):
    """Small dot on the 2D base-sphere inset (azimuthal equidistant)."""
    r = radius * (theta_b / PI)
    pos = center + np.array([r * np.cos(phi_b), r * np.sin(phi_b), 0])
    return Dot(pos, color=color, radius=0.04)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class HopfFibration(ThreeDScene):
    """Hopf fibration: linked circles, Clifford torus, nested tori."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("Hopf Fibration", font_size=56, color=GOLD)
        sub = MathTex(r"h : S^3 \;\longrightarrow\; S^2",
                      font_size=32, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        desc = Text(
            "Every point on the 2-sphere\n"
            "is a circle in 3-space.\n"
            "All circles are linked.",
            font_size=20, color=WHITE, line_spacing=1.3,
        )
        desc.next_to(sub, DOWN, buff=0.4)

        self.add_fixed_in_frame_mobjects(ttl, sub, desc)
        self.play(Write(ttl), run_time=1.5)                          # 1
        self.play(FadeIn(sub), run_time=1)                            # 2
        self.play(FadeIn(desc), run_time=1)                           # 3
        self.wait(1.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(desc))          # 4

        # ─────────────────────────────────────────────────────────────────
        # Camera + base-sphere inset
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(
            phi=70 * DEGREES, theta=-40 * DEGREES, zoom=0.80,
        )

        # Small S² diagram (2D, top-right corner)
        base_circ = Circle(radius=0.45, color=WHITE, stroke_width=1.5)
        base_circ.to_corner(UP + RIGHT, buff=0.3)
        bc = base_circ.get_center()
        base_lbl = MathTex(r"S^2", font_size=18, color=WHITE)
        base_lbl.next_to(base_circ, DOWN, buff=0.08)
        self.add_fixed_in_frame_mobjects(base_circ, base_lbl)
        self.play(Create(base_circ), FadeIn(base_lbl), run_time=0.8) # 5

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — North-pole fibre (a single circle in the xy plane)
        # ─────────────────────────────────────────────────────────────────
        np_fibre = _make_fibre(0, 0, WHITE, sw=3)
        np_dot = _base_dot(0, 0, bc, 0.45, WHITE)
        self.add_fixed_in_frame_mobjects(np_dot)

        eq_fibre = MathTex(
            r"\theta_b = 0 \;\;\text{(north pole)}",
            font_size=20, color=WHITE,
        )
        eq_fibre.to_corner(DOWN + LEFT, buff=0.25)
        self.add_fixed_in_frame_mobjects(eq_fibre)

        self.play(Create(np_fibre), FadeIn(np_dot), run_time=1.5)    # 6
        self.play(FadeIn(eq_fibre), run_time=0.6)                    # 7

        # Equation
        hopf_eq = MathTex(
            r"h(z_1, z_2) = \frac{z_1}{z_2}",
            font_size=24, color=BLUE,
        )
        hopf_eq.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(hopf_eq)
        self.play(FadeIn(hopf_eq), run_time=0.6)                     # 8

        self.move_camera(theta=-30 * DEGREES, run_time=1.5)          # 9

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Equatorial fibres (linked, rainbow)
        # ─────────────────────────────────────────────────────────────────
        self.play(FadeOut(eq_fibre), run_time=0.4)                    # 10

        N_EQ = 6
        eq_fibres = VGroup()
        eq_dots = VGroup()
        for i in range(N_EQ):
            phi = TAU * i / N_EQ
            col = _hue(i / N_EQ)
            fb = _make_fibre(PI / 2, phi, col)
            eq_fibres.add(fb)
            d = _base_dot(PI / 2, phi, bc, 0.45, col)
            eq_dots.add(d)

        self.add_fixed_in_frame_mobjects(*eq_dots)

        # Build fibres one at a time
        for i in range(N_EQ):
            self.play(
                Create(eq_fibres[i]),
                FadeIn(eq_dots[i]),
                run_time=0.7,
            )                                                         # 11-16

        linked_lbl = MathTex(
            r"\text{All fibres are pairwise linked}",
            font_size=20, color=YELLOW,
        )
        linked_lbl.to_edge(DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(linked_lbl)
        self.play(FadeIn(linked_lbl), run_time=0.6)                  # 17

        # Orbit to show linking
        self.move_camera(
            phi=55 * DEGREES, theta=-15 * DEGREES, run_time=2.5,
        )                                                             # 18

        self.play(FadeOut(linked_lbl), run_time=0.4)                  # 19

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Clifford torus  (equatorial fibres as a surface)
        # ─────────────────────────────────────────────────────────────────
        torus_lbl = MathTex(
            r"\text{Equatorial latitude} \;\to\; \text{Clifford torus}",
            font_size=20, color=GOLD,
        )
        torus_lbl.to_edge(DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(torus_lbl)
        self.play(FadeIn(torus_lbl), run_time=0.5)                   # 20

        clifford = _make_torus(PI / 2, GOLD, opacity=0.20, res=40)
        self.play(
            FadeOut(eq_fibres),
            Create(clifford),
            run_time=2,
        )                                                             # 21

        self.move_camera(
            phi=65 * DEGREES, theta=-50 * DEGREES, run_time=2,
        )                                                             # 22

        # Redraw a few fibres ON the torus for reference
        ref_fibres = VGroup()
        for i in range(4):
            phi = TAU * i / 4
            col = _hue(i / 4, v=1.0)
            fb = _make_fibre(PI / 2, phi, col, sw=2)
            ref_fibres.add(fb)
        self.play(Create(ref_fibres), run_time=1)                    # 23

        self.play(FadeOut(torus_lbl), run_time=0.3)                  # 24

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Nested tori (three latitudes)
        # ─────────────────────────────────────────────────────────────────
        nested_lbl = MathTex(
            r"\text{Different latitudes} \;\to\; \text{nested tori}",
            font_size=20, color=TEAL,
        )
        nested_lbl.to_edge(DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(nested_lbl)
        self.play(FadeIn(nested_lbl), run_time=0.5)                  # 25

        # Inner torus (near north pole — thin ring)
        inner = _make_torus(PI / 5, BLUE, opacity=0.15, res=32)
        # Outer torus (closer to south pole — fatter)
        outer = _make_torus(3 * PI / 5, ORANGE, opacity=0.15, res=32)

        # Base-sphere dots for the latitude circles
        inner_ring = Circle(
            radius=0.45 * (PI / 5) / PI,
            color=BLUE, stroke_width=1.5,
        ).move_to(bc)
        outer_ring = Circle(
            radius=0.45 * (3 * PI / 5) / PI,
            color=ORANGE, stroke_width=1.5,
        ).move_to(bc)
        eq_ring = Circle(
            radius=0.45 * 0.5,
            color=GOLD, stroke_width=1.5,
        ).move_to(bc)

        self.add_fixed_in_frame_mobjects(inner_ring, outer_ring, eq_ring)

        self.play(
            Create(inner),
            FadeIn(inner_ring),
            run_time=1.5,
        )                                                             # 26
        self.play(
            Create(outer),
            FadeIn(outer_ring),
            FadeIn(eq_ring),
            run_time=1.5,
        )                                                             # 27

        self.play(FadeOut(nested_lbl), run_time=0.3)                  # 28

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Camera orbit  (full beauty shot)
        # ─────────────────────────────────────────────────────────────────
        beauty_lbl = MathTex(
            r"\pi_3(S^2) = \mathbb{Z}",
            font_size=28, color=WHITE,
        )
        beauty_lbl.to_corner(DOWN + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(beauty_lbl)
        self.play(FadeIn(beauty_lbl), run_time=0.6)                  # 29

        self.move_camera(
            phi=80 * DEGREES, theta=10 * DEGREES, run_time=3,
        )                                                             # 30
        self.move_camera(
            phi=50 * DEGREES, theta=-70 * DEGREES, run_time=3,
        )                                                             # 31
        self.move_camera(
            phi=65 * DEGREES, theta=-40 * DEGREES, run_time=2,
        )                                                             # 32

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(np_fibre), FadeOut(clifford), FadeOut(inner),
            FadeOut(outer), FadeOut(ref_fibres),
            FadeOut(beauty_lbl), FadeOut(hopf_eq),
            FadeOut(np_dot), FadeOut(eq_dots),
            FadeOut(base_circ), FadeOut(base_lbl),
            FadeOut(inner_ring), FadeOut(outer_ring), FadeOut(eq_ring),
            run_time=1,
        )                                                             # 33

        card_title = Text("Hopf Fibration", font_size=36, color=GOLD)
        card_title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(card_title)
        self.play(Write(card_title))                                  # 34

        lines = VGroup(
            MathTex(
                r"h : S^3 \to S^2, \quad "
                r"h^{-1}(\text{point}) = S^1 \;\text{(circle)}",
                font_size=20,
            ),
            MathTex(
                r"\text{Fibres over a latitude} \;\to\; "
                r"\text{torus in } \mathbb{R}^3",
                font_size=20,
            ),
            MathTex(
                r"\text{Every two fibres are linked once} "
                r"\;\Rightarrow\; \text{linking number } = 1",
                font_size=20,
            ),
            MathTex(
                r"\pi_3(S^2) = \mathbb{Z}: \; "
                r"\text{the Hopf invariant is } \pm 1",
                font_size=20, color=TEAL,
            ),
            MathTex(
                r"\text{Berry phase, magnetic monopoles, "
                r"Bloch sphere — all Hopf}",
                font_size=20, color=ORANGE,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        lines.next_to(card_title, DOWN, buff=0.4)

        box = SurroundingRectangle(
            VGroup(card_title, lines),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.add_fixed_in_frame_mobjects(lines, box)
        self.play(FadeIn(box), run_time=0.5)                         # 35
        for line in lines:
            self.play(FadeIn(line), run_time=0.8)                     # 36-40

        self.wait(2)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(lines), FadeOut(box),
            run_time=1.5,
        )                                                             # 41
