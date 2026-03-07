"""Holonomy on a sphere — parallel transport around geodesic triangles.

A tangent vector is parallel-transported around a closed geodesic path
on the unit sphere.  Upon return it has rotated by an angle equal to
the enclosed solid angle — Gauss–Bonnet in action.

    holonomy angle  Ω  =  ∮ ω  =  ∫∫ K dA  =  Area(△)

Acts
----
1. Title card
2. Translucent sphere with latitude / longitude grid
3. Geodesic triangle  N → A → B → N  and initial tangent vector
4. Parallel transport around all three legs  (vector slides, arrow
   updates via Transform; ghost arrows mark waypoints)
5. Holonomy reveal — rotation = π/2 = area of the spherical triangle
6. Gauss–Bonnet / Berry-phase summary

Run
---
    manim -pql examples/holonomy_sphere.py HolonomySphere
    manim -qh  examples/holonomy_sphere.py HolonomySphere
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    ParametricFunction,
    Arrow3D,
    Dot3D,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Transform,
    GrowFromCenter,
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
    RED_E,
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
RS = 2.0        # sphere display radius
AL = 0.55       # tangent-vector arrow length
SPL = 6         # transport sub-steps per leg (Transforms per leg)

# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sph(theta: float, phi: float) -> np.ndarray:
    """Spherical  (co-latitude θ, azimuth φ) → Cartesian, radius RS."""
    return RS * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _pt(V0, A, B, frac):
    """Parallel-transport tangent vector *V0* from *A* toward *B* on
    the sphere of radius RS.

    Parameters
    ----------
    V0   : tangent vector at A  (perpendicular to radial direction)
    A, B : points ON the sphere
    frac : progress ∈ [0, 1] along the great-circle arc

    Returns
    -------
    (position_on_sphere, transported_vector)
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    An = A / np.linalg.norm(A)
    Bn = B / np.linalg.norm(B)
    omega = np.arccos(np.clip(np.dot(An, Bn), -1.0, 1.0))
    if omega < 1e-12:
        return A.copy(), V0.copy()
    T = (Bn - np.dot(An, Bn) * An) / np.sin(omega)   # unit tangent at A
    N = np.cross(An, T)                                # binormal
    a = float(np.dot(V0, T))
    b = float(np.dot(V0, N))
    t = frac * omega
    pos = RS * (np.cos(t) * An + np.sin(t) * T)
    tang = -np.sin(t) * An + np.cos(t) * T
    return pos, a * tang + b * N


def _arr(pos, vec, color=YELLOW, length=AL):
    """Create a small Arrow3D for a tangent vector at *pos*."""
    d = np.asarray(vec, float)
    n = np.linalg.norm(d)
    d = d / max(n, 1e-15)
    return Arrow3D(
        start=pos,
        end=pos + length * d,
        color=color,
        thickness=0.015,
        height=0.12,
        base_radius=0.05,
    )


def _gc(A, B, **kw):
    """ParametricFunction along the great-circle arc A → B, t ∈ [0,1]."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    om = np.arccos(np.clip(
        np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)),
        -1.0, 1.0,
    ))
    s = np.sin(om)
    if s < 1e-12:
        return ParametricFunction(lambda t: A.copy(), t_range=[0, 1], **kw)
    def f(t):
        return np.sin((1 - t) * om) / s * A + np.sin(t * om) / s * B
    return ParametricFunction(f, t_range=[0, 1], **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class HolonomySphere(ThreeDScene):
    """Parallel transport of a tangent vector around a spherical triangle."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("Holonomy", font_size=56, color=GOLD)
        sub = Text("Geometry Remembers the Path", font_size=26, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)
        desc = Text(
            "A vector parallel-transported around a\n"
            "closed loop on a curved surface returns rotated.",
            font_size=20,
            color=WHITE,
        )
        desc.next_to(sub, DOWN, buff=0.4)

        self.add_fixed_in_frame_mobjects(ttl, sub, desc)
        self.play(Write(ttl), run_time=1.5)                         # 1
        self.play(FadeIn(sub), run_time=1)                           # 2
        self.play(FadeIn(desc), run_time=1)                          # 3
        self.wait(1.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(desc))         # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Build sphere with coordinate grid
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(
            phi=60 * DEGREES, theta=-45 * DEGREES, zoom=0.85,
        )

        sphere = Surface(
            lambda u, v: np.array([
                RS * np.sin(u) * np.cos(v),
                RS * np.sin(u) * np.sin(v),
                RS * np.cos(u),
            ]),
            u_range=[0, PI],
            v_range=[0, TAU],
            resolution=(32, 32),
            fill_opacity=0.10,
            stroke_width=0.3,
            stroke_color=BLUE_E,
            fill_color=BLUE,
        )

        lats = VGroup(*[
            ParametricFunction(
                lambda t, th=th: np.array([
                    RS * np.sin(th) * np.cos(t),
                    RS * np.sin(th) * np.sin(t),
                    RS * np.cos(th),
                ]),
                t_range=[0, TAU],
                color=BLUE_D,
                stroke_width=0.8,
                stroke_opacity=0.35,
            )
            for th in np.linspace(PI / 6, 5 * PI / 6, 5)
        ])

        lons = VGroup(*[
            ParametricFunction(
                lambda t, ph=ph: np.array([
                    RS * np.sin(t) * np.cos(ph),
                    RS * np.sin(t) * np.sin(ph),
                    RS * np.cos(t),
                ]),
                t_range=[0, PI],
                color=BLUE_D,
                stroke_width=0.8,
                stroke_opacity=0.35,
            )
            for ph in np.linspace(0, TAU, 12, endpoint=False)
        ])

        s2lbl = MathTex(r"S^2", font_size=28, color=BLUE)
        s2lbl.to_corner(UP + LEFT)
        self.add_fixed_in_frame_mobjects(s2lbl)

        self.play(Create(sphere), run_time=2)                        # 5
        self.play(Create(lats), Create(lons), run_time=1.5)          # 6
        self.play(FadeIn(s2lbl))                                     # 7
        self.move_camera(theta=-35 * DEGREES, run_time=2)            # 8

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Geodesic triangle + initial tangent vector
        # ─────────────────────────────────────────────────────────────────
        # Triangle vertices
        PN = _sph(0, 0)               # North pole  (0, 0, RS)
        PA = _sph(PI / 2, 0)          # Equator φ=0 (RS, 0, 0)
        PB = _sph(PI / 2, PI / 2)    # Equator φ=π/2 (0, RS, 0)

        # Geodesic arcs (great-circle segments)
        arc_NA = _gc(PN, PA, color=GOLD, stroke_width=3)
        arc_AB = _gc(PA, PB, color=GOLD, stroke_width=3)
        arc_BN = _gc(PB, PN, color=GOLD, stroke_width=3)
        arcs = [arc_NA, arc_AB, arc_BN]

        # Vertex dots
        dot_N = Dot3D(PN, color=WHITE, radius=0.06)
        dot_A = Dot3D(PA, color=WHITE, radius=0.06)
        dot_B = Dot3D(PB, color=WHITE, radius=0.06)

        # Vertex labels (fixed orientation — always face the camera)
        lbl_N = MathTex("N", font_size=22, color=WHITE)
        lbl_N.move_to(PN * 1.18)
        lbl_A = MathTex("A", font_size=22, color=WHITE)
        lbl_A.move_to(PA * 1.18)
        lbl_B = MathTex("B", font_size=22, color=WHITE)
        lbl_B.move_to(PB * 1.18)
        self.add_fixed_orientation_mobjects(lbl_N, lbl_A, lbl_B)

        self.play(Create(arc_NA), run_time=1)                        # 9
        self.play(Create(arc_AB), run_time=1)                        # 10
        self.play(Create(arc_BN), run_time=1)                        # 11
        self.play(
            FadeIn(dot_N), FadeIn(dot_A), FadeIn(dot_B),
            FadeIn(lbl_N), FadeIn(lbl_A), FadeIn(lbl_B),
            run_time=1,
        )                                                            # 12

        # Area label
        tri_lbl = MathTex(
            r"\text{Area} = \frac{\pi}{2}", font_size=22, color=GOLD,
        )
        tri_lbl.to_corner(DOWN + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(tri_lbl)
        self.play(FadeIn(tri_lbl))                                   # 13

        # Initial tangent vector at N → points along +x (toward A)
        V0 = np.array([1.0, 0.0, 0.0])
        arrow = _arr(PN, V0)
        self.play(GrowFromCenter(arrow), run_time=1)                 # 14

        v0_lbl = Text("initial vector", font_size=16, color=YELLOW)
        v0_lbl.to_edge(DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(v0_lbl)
        self.play(FadeIn(v0_lbl))                                    # 15

        # Equation: parallel transport preserves dot product
        pt_eq = MathTex(
            r"\nabla_{\dot\gamma}\, \mathbf{v} = 0",
            font_size=24, color=WHITE,
        )
        pt_eq.to_corner(UP + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(pt_eq)
        self.play(FadeIn(pt_eq))                                     # 16

        self.move_camera(phi=55 * DEGREES, theta=-40 * DEGREES,
                         run_time=1.5)                               # 17

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Parallel transport around the triangle (3 legs)
        # ─────────────────────────────────────────────────────────────────

        # Precompute leg start vectors
        # Leg 0: N→A  v0=(1,0,0)  → v1=(0,0,-1)
        # Leg 1: A→B  v1=(0,0,-1) → v2=(0,0,-1)
        # Leg 2: B→N  v2=(0,0,-1) → v3=(0,1,0)
        _, V1 = _pt(V0, PN, PA, 1.0)
        _, V2 = _pt(V1, PA, PB, 1.0)
        _, V3 = _pt(V2, PB, PN, 1.0)

        vertices = [PN, PA, PB]
        V_starts = [V0, V1, V2]
        leg_names = [
            r"\text{Leg 1: } N \to A",
            r"\text{Leg 2: } A \to B",
            r"\text{Leg 3: } B \to N",
        ]
        leg_colors = [RED_E, GREEN, TEAL]

        ghosts = VGroup()

        for leg in range(3):
            P_start = vertices[leg]
            P_end = vertices[(leg + 1) % 3]
            V_init = V_starts[leg]

            # Leg label
            leg_lbl = MathTex(leg_names[leg], font_size=20,
                              color=leg_colors[leg])
            leg_lbl.to_edge(DOWN, buff=0.5)
            self.add_fixed_in_frame_mobjects(leg_lbl)

            # Highlight current arc
            highlight = _gc(P_start, P_end,
                            color=leg_colors[leg], stroke_width=5)

            self.play(
                FadeOut(v0_lbl) if leg == 0 else FadeIn(leg_lbl),
                FadeIn(highlight) if leg > 0 else FadeIn(leg_lbl),
                run_time=0.6,
            )                                                        # 18,25,32

            if leg == 0:
                self.play(FadeIn(highlight), run_time=0.4)           # 19

            # Transport steps
            for step in range(1, SPL + 1):
                frac = step / SPL
                pos, vec = _pt(V_init, P_start, P_end, frac)
                target = _arr(pos, vec)
                self.play(
                    Transform(arrow, target),
                    run_time=0.35,
                )                                    # 20-25 / 26-31 / 33-38

            # Ghost arrow at leg endpoint
            pos_end, vec_end = _pt(V_init, P_start, P_end, 1.0)
            ghost = _arr(pos_end, vec_end, color=YELLOW, length=AL)
            ghost.set_opacity(0.25)
            ghosts.add(ghost)
            self.play(FadeIn(ghost), run_time=0.3)                   # 26,32,39

            # Cleanup
            self.play(FadeOut(leg_lbl), FadeOut(highlight),
                      run_time=0.4)                                  # 27,33,40

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Holonomy reveal
        # ─────────────────────────────────────────────────────────────────
        reveal_lbl = Text("Holonomy Angle", font_size=24, color=RED)
        reveal_lbl.to_edge(UP + RIGHT, buff=0.2)
        self.add_fixed_in_frame_mobjects(reveal_lbl)
        self.play(FadeIn(reveal_lbl))                                # 41

        # Brighter ghost of initial vector
        ghost_v0 = _arr(PN, V0, color=ORANGE, length=AL)
        ghost_v0.set_opacity(0.6)
        self.play(FadeIn(ghost_v0), run_time=0.8)                   # 42

        # Move arrow to final position to make it visible
        final_arrow = _arr(PN, V3, color=YELLOW, length=AL)
        self.play(Transform(arrow, final_arrow), run_time=0.6)      # 43

        # Angle arc from V0 tip to V3 tip in tangent plane at N
        angle_arc = ParametricFunction(
            lambda t: np.array([
                PN[0] + AL * np.cos(t),
                PN[1] + AL * np.sin(t),
                PN[2],
            ]),
            t_range=[0, PI / 2],
            color=RED,
            stroke_width=4,
        )
        self.play(Create(angle_arc), run_time=1.2)                   # 44

        # Angle value
        omega_lbl = MathTex(
            r"\Omega = \frac{\pi}{2}",
            font_size=28, color=RED,
        )
        omega_lbl.to_corner(DOWN + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(omega_lbl)
        self.play(Write(omega_lbl))                                  # 45

        # Equals area
        eq_area = MathTex(
            r"= \text{Area}(\triangle) \;\text{on unit sphere}",
            font_size=22, color=GOLD,
        )
        eq_area.next_to(omega_lbl, DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(eq_area)
        self.play(FadeIn(eq_area))                                   # 46

        # Gauss-Bonnet
        gb_eq = MathTex(
            r"\Omega = \oint_{\partial\triangle} \omega "
            r"= \iint_{\triangle} K \, dA",
            font_size=26, color=WHITE,
        )
        gb_eq.to_edge(DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(gb_eq)
        self.play(Write(gb_eq), run_time=1.5)                       # 47
        self.play(Indicate(gb_eq, color=GOLD), run_time=1)           # 48

        # Camera orbit for drama
        self.move_camera(
            phi=50 * DEGREES, theta=-20 * DEGREES, run_time=3,
        )                                                            # 49

        self.wait(1)

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Summary
        # ─────────────────────────────────────────────────────────────────
        # Fade transport artifacts
        self.play(
            FadeOut(arrow), FadeOut(ghosts), FadeOut(ghost_v0),
            FadeOut(angle_arc), FadeOut(reveal_lbl),
            FadeOut(omega_lbl), FadeOut(eq_area), FadeOut(gb_eq),
            FadeOut(pt_eq), FadeOut(tri_lbl), FadeOut(v0_lbl),
            run_time=1,
        )                                                            # 50

        # Summary card
        card_title = Text("Holonomy", font_size=36, color=GOLD)
        card_title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(card_title)
        self.play(Write(card_title))                                 # 51

        lines = VGroup(
            MathTex(
                r"\text{Curvature } K \neq 0 \;\Rightarrow\; "
                r"\text{parallel transport remembers the path}",
                font_size=20,
            ),
            MathTex(
                r"\text{Holonomy angle } \Omega "
                r"= \iint K\,dA = \alpha + \beta + \gamma - \pi",
                font_size=20,
            ),
            MathTex(
                r"\text{Flat space } (K=0): \;\Omega = 0 "
                r"\;\text{— vector returns unchanged}",
                font_size=20,
            ),
            MathTex(
                r"\text{Quantum analogue: Berry phase } "
                r"\gamma_B = \oint \mathcal{A} \cdot dk",
                font_size=20, color=TEAL,
            ),
            MathTex(
                r"\text{Gravity: Riemann tensor } "
                r"R^{\mu}{}_{\nu\rho\sigma} \text{ measures holonomy}",
                font_size=20, color=ORANGE,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        lines.next_to(card_title, DOWN, buff=0.4)

        box = SurroundingRectangle(
            VGroup(card_title, lines),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.add_fixed_in_frame_mobjects(lines, box)
        self.play(FadeIn(box), run_time=0.5)                        # 52
        for line in lines:
            self.play(FadeIn(line), run_time=0.8)                    # 53-57

        self.wait(2)

        # Final fade
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(lines), FadeOut(box),
            FadeOut(s2lbl),
            run_time=1.5,
        )                                                            # 58
