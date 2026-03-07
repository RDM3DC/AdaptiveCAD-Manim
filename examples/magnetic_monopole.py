"""Magnetic Monopole — Dirac string, gauge patches & charge quantisation.

A magnetic monopole radiates B-field lines radially just like an electric
charge radiates E-field.  But ∇·B = 0 means you need a singular "Dirac
string" in any single vector-potential patch.  The fix: cover S² with two
patches (N and S), each string-free in its own hemisphere, joined by a
U(1) transition function on the equator.  Consistency of the wave function
around the equator forces  eg = nℏ/2  (Dirac quantisation).

Acts
----
1. Title card
2. Radial B-field lines from a point monopole  (3D)
3. Dirac string — a singular line along −z carrying return flux
4. Two gauge patches  U_N and U_S — each covers a hemisphere
5. Transition function  g_{NS} = e^{inφ}  on the equator
6. Dirac quantisation  eg = nℏ/2
7. Summary card

Run
---
    manim -pql examples/magnetic_monopole.py MagneticMonopole
    manim -qh  examples/magnetic_monopole.py MagneticMonopole
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    ParametricFunction,
    Arrow3D,
    Line3D,
    Dot3D,
    Sphere,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    GrowFromCenter,
    Indicate,
    SurroundingRectangle,
    Flash,
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
    GREEN_E,
    BLUE,
    BLUE_D,
    BLUE_E,
    ORANGE,
    GOLD,
    TEAL,
    GREY,
    PURPLE,
    config,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
RS = 2.0    # display sphere radius
FL = 2.8    # field-line length (beyond sphere)


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sph(theta, phi, r=RS):
    """Spherical → Cartesian."""
    return r * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _radial_arrow(theta, phi, r_in=0.15, r_out=FL, color=YELLOW):
    """Arrow3D pointing radially outward at (θ, φ)."""
    start = _sph(theta, phi, r_in)
    end = _sph(theta, phi, r_out)
    return Arrow3D(
        start=start, end=end,
        color=color,
        thickness=0.012,
        height=0.10,
        base_radius=0.04,
    )


def _field_line(theta, phi, color=YELLOW, sw=1.8):
    """ParametricFunction for a straight radial field line."""
    d = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    return ParametricFunction(
        lambda t: t * d,
        t_range=[0.12, FL],
        color=color,
        stroke_width=sw,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class MagneticMonopole(ThreeDScene):
    """Dirac magnetic monopole: field lines, string, gauge patches."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title
        # ─────────────────────────────────────────────────────────────────
        ttl = Text("Magnetic Monopole", font_size=52, color=GOLD)
        sub = MathTex(
            r"\nabla \cdot \mathbf{B} = g\,\delta^3(\mathbf{r})",
            font_size=30, color=BLUE,
        )
        sub.next_to(ttl, DOWN, buff=0.3)
        desc = Text(
            "A point source of magnetic flux.\n"
            "Gauge theory says it can exist —\n"
            "and forces electric charge to be quantised.",
            font_size=19, color=WHITE, line_spacing=1.3,
        )
        desc.next_to(sub, DOWN, buff=0.4)

        self.add_fixed_in_frame_mobjects(ttl, sub, desc)
        self.play(Write(ttl), run_time=1.5)                            # 1
        self.play(FadeIn(sub), run_time=1)                              # 2
        self.play(FadeIn(desc), run_time=1)                             # 3
        self.wait(1.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(desc))            # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Radial B-field lines + translucent sphere
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(
            phi=65 * DEGREES, theta=-40 * DEGREES, zoom=0.78,
        )

        # Translucent sphere
        sphere = Surface(
            lambda u, v: RS * np.array([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u),
            ]),
            u_range=[0, PI],
            v_range=[0, TAU],
            resolution=(28, 28),
            fill_opacity=0.06,
            stroke_width=0.3,
            stroke_color=BLUE_E,
            fill_color=BLUE,
        )

        # Monopole dot at origin
        mono_dot = Dot3D(ORIGIN, color=RED, radius=0.12)

        # Field-line directions — roughly uniform on S²
        fl_dirs = []
        for th in [0.4, 1.0, PI / 2, PI - 1.0, PI - 0.4]:
            n_phi = max(3, int(5 * np.sin(th)))
            for i in range(n_phi):
                fl_dirs.append((th, TAU * i / n_phi))

        field_lines = VGroup()
        arrows = VGroup()
        for idx, (th, ph) in enumerate(fl_dirs):
            fl = _field_line(th, ph, YELLOW, sw=1.8)
            field_lines.add(fl)
            if idx % 3 == 0:  # arrows on every 3rd line only
                ar = _radial_arrow(th, ph, r_in=FL * 0.75, r_out=FL, color=YELLOW)
                arrows.add(ar)

        # Flux label
        flux_lbl = MathTex(
            r"\oint_{S^2} \mathbf{B}\cdot d\mathbf{A} = g",
            font_size=24, color=YELLOW,
        )
        flux_lbl.to_corner(DOWN + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(flux_lbl)

        b_lbl = MathTex(
            r"\mathbf{B} = \frac{g}{4\pi r^2}\,\hat{r}",
            font_size=24, color=WHITE,
        )
        b_lbl.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(b_lbl)

        self.play(FadeIn(sphere), GrowFromCenter(mono_dot),
                  run_time=1.5)                                         # 5
        self.play(
            *[Create(fl) for fl in field_lines],
            run_time=2,
        )                                                               # 6
        self.play(
            *[GrowFromCenter(a) for a in arrows],
            run_time=1,
        )                                                               # 7
        self.play(FadeIn(flux_lbl), FadeIn(b_lbl), run_time=0.8)       # 8

        self.move_camera(
            phi=55 * DEGREES, theta=-25 * DEGREES, run_time=2,
        )                                                               # 9

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Dirac string
        # ─────────────────────────────────────────────────────────────────
        self.play(FadeOut(flux_lbl), run_time=0.3)                      # 10

        string_lbl = MathTex(
            r"\text{Dirac string } \;\mathcal{S}",
            font_size=22, color=RED,
        )
        string_lbl.to_edge(DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(string_lbl)

        # String along −z axis
        string_line = ParametricFunction(
            lambda t: np.array([0, 0, -t]),
            t_range=[0.15, 4.0],
            color=RED,
            stroke_width=5,
            stroke_opacity=0.9,
        )

        # "Singular!" label
        sing_lbl = MathTex(
            r"\mathbf{A}\;\text{singular here}",
            font_size=18, color=RED_E,
        )
        sing_lbl.move_to(np.array([0, 0, -3.5]))
        self.add_fixed_orientation_mobjects(sing_lbl)

        self.play(Create(string_line), run_time=1.5)                    # 11
        self.play(FadeIn(string_lbl), FadeIn(sing_lbl), run_time=0.8)   # 12

        # Explain: ∇×A = B everywhere except on string
        curl_lbl = MathTex(
            r"\nabla \times \mathbf{A} = \mathbf{B}"
            r"\;\;\text{(except on } \mathcal{S}\text{)}",
            font_size=20, color=WHITE,
        )
        curl_lbl.to_corner(DOWN + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(curl_lbl)
        self.play(FadeIn(curl_lbl), run_time=0.6)                      # 13

        self.move_camera(
            phi=75 * DEGREES, theta=-50 * DEGREES, run_time=2,
        )                                                               # 14

        self.wait(1)

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Two gauge patches  U_N  and  U_S
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(string_line), FadeOut(string_lbl),
            FadeOut(sing_lbl), FadeOut(curl_lbl),
            FadeOut(field_lines), FadeOut(arrows), FadeOut(b_lbl),
            run_time=0.8,
        )                                                               # 15

        patch_title = MathTex(
            r"\text{Fix: two patches on } S^2",
            font_size=22, color=GOLD,
        )
        patch_title.to_edge(UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(patch_title)
        self.play(FadeIn(patch_title), run_time=0.6)                    # 16

        # Northern hemisphere patch (blue)
        north_patch = Surface(
            lambda u, v: (RS + 0.03) * np.array([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u),
            ]),
            u_range=[0, PI / 2 + 0.25],
            v_range=[0, TAU],
            resolution=(16, 24),
            fill_opacity=0.25,
            fill_color=BLUE,
            stroke_width=0.5,
            stroke_color=BLUE_D,
        )

        # Southern hemisphere patch (green)
        south_patch = Surface(
            lambda u, v: (RS + 0.03) * np.array([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u),
            ]),
            u_range=[PI / 2 - 0.25, PI],
            v_range=[0, TAU],
            resolution=(16, 24),
            fill_opacity=0.25,
            fill_color=GREEN,
            stroke_width=0.5,
            stroke_color=GREEN_E,
        )

        un_lbl = MathTex(r"U_N", font_size=24, color=BLUE)
        un_lbl.move_to(_sph(0.3, 0.5, RS + 0.6))
        self.add_fixed_orientation_mobjects(un_lbl)

        us_lbl = MathTex(r"U_S", font_size=24, color=GREEN)
        us_lbl.move_to(_sph(PI - 0.3, 0.5, RS + 0.6))
        self.add_fixed_orientation_mobjects(us_lbl)

        self.play(Create(north_patch), FadeIn(un_lbl), run_time=1.5)    # 17
        self.play(Create(south_patch), FadeIn(us_lbl), run_time=1.5)    # 18

        # Overlap band (equator region)
        overlap = Surface(
            lambda u, v: (RS + 0.06) * np.array([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u),
            ]),
            u_range=[PI / 2 - 0.25, PI / 2 + 0.25],
            v_range=[0, TAU],
            resolution=(6, 24),
            fill_opacity=0.30,
            fill_color=YELLOW,
            stroke_width=1.0,
            stroke_color=YELLOW,
        )

        ov_lbl = MathTex(
            r"U_N \cap U_S \simeq S^1",
            font_size=20, color=YELLOW,
        )
        ov_lbl.to_edge(DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(ov_lbl)

        self.play(Create(overlap), FadeIn(ov_lbl), run_time=1.2)        # 19

        # Gauge potential labels
        an_eq = MathTex(
            r"A_N = \frac{g(1 - \cos\theta)}{r\sin\theta}\,"
            r"\hat{\phi}",
            font_size=18, color=BLUE,
        )
        an_eq.to_corner(DOWN + LEFT, buff=0.25)
        as_eq = MathTex(
            r"A_S = \frac{-g(1 + \cos\theta)}{r\sin\theta}\,"
            r"\hat{\phi}",
            font_size=18, color=GREEN,
        )
        as_eq.next_to(an_eq, DOWN, buff=0.15)

        self.add_fixed_in_frame_mobjects(an_eq, as_eq)
        self.play(FadeIn(an_eq), run_time=0.6)                          # 20
        self.play(FadeIn(as_eq), run_time=0.6)                          # 21

        # Note: A_N has string at south pole, A_S at north pole
        note = MathTex(
            r"A_N \;\text{good on } U_N,\;\;"
            r"A_S \;\text{good on } U_S",
            font_size=18, color=WHITE,
        )
        note.to_corner(UP + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note), run_time=0.6)                           # 22

        self.move_camera(
            phi=60 * DEGREES, theta=-35 * DEGREES, run_time=2,
        )                                                               # 23

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Transition function on the equator
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(an_eq), FadeOut(as_eq), FadeOut(note),
            FadeOut(patch_title), FadeOut(ov_lbl),
            run_time=0.6,
        )                                                               # 24

        trans_title = MathTex(
            r"\text{Transition function on equator}",
            font_size=22, color=GOLD,
        )
        trans_title.to_edge(UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(trans_title)
        self.play(FadeIn(trans_title), run_time=0.5)                    # 25

        # g_{NS} = e^{inφ}
        trans_eq = MathTex(
            r"g_{NS}(\phi) = e^{in\phi}",
            font_size=28, color=YELLOW,
        )
        trans_eq.to_corner(DOWN + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(trans_eq)
        self.play(Write(trans_eq), run_time=1)                          # 26

        # Show winding dots around equator
        N_DOTS = 12
        eq_dots = VGroup()
        eq_arrows_t = VGroup()
        for i in range(N_DOTS):
            phi = TAU * i / N_DOTS
            pos = _sph(PI / 2, phi, RS + 0.10)
            d = Dot3D(pos, color=YELLOW, radius=0.05)
            eq_dots.add(d)
            # Tangent arrow showing phase = nφ  (n=1)
            tangent = np.array([
                -np.sin(phi),
                np.cos(phi),
                0,
            ])
            # Rotate tangent by angle φ to show the winding
            c, s = np.cos(phi), np.sin(phi)
            rotated = np.array([
                c * tangent[0] - s * tangent[1],
                s * tangent[0] + c * tangent[1],
                tangent[2],
            ])
            ar = Arrow3D(
                start=pos,
                end=pos + 0.35 * rotated,
                color=ORANGE,
                thickness=0.010,
                height=0.08,
                base_radius=0.03,
            )
            eq_arrows_t.add(ar)

        self.play(
            *[FadeIn(d) for d in eq_dots],
            run_time=0.8,
        )                                                               # 27
        self.play(
            *[GrowFromCenter(a) for a in eq_arrows_t],
            run_time=1,
        )                                                               # 28

        # The key point: single-valuedness
        sv_eq = MathTex(
            r"g_{NS}(\phi + 2\pi) = g_{NS}(\phi)"
            r"\;\;\Rightarrow\;\; n \in \mathbb{Z}",
            font_size=22, color=WHITE,
        )
        sv_eq.to_edge(DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(sv_eq)
        self.play(Write(sv_eq), run_time=1.2)                          # 29
        self.play(Indicate(sv_eq, color=GOLD), run_time=0.8)            # 30

        self.move_camera(
            phi=50 * DEGREES, theta=-20 * DEGREES, run_time=2,
        )                                                               # 31

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Dirac quantisation
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(trans_title), FadeOut(sv_eq), FadeOut(trans_eq),
            FadeOut(eq_dots), FadeOut(eq_arrows_t),
            run_time=0.6,
        )                                                               # 32

        quant_title = MathTex(
            r"\text{Dirac Quantisation}",
            font_size=26, color=GOLD,
        )
        quant_title.to_edge(UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(quant_title)
        self.play(FadeIn(quant_title), run_time=0.5)                    # 33

        # Big equation
        dirac_eq = MathTex(
            r"eg = \frac{n\hbar}{2}",
            font_size=40, color=YELLOW,
        )
        dirac_eq.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(dirac_eq)
        self.play(Write(dirac_eq), run_time=1.5)                       # 34
        self.play(
            Flash(dirac_eq, color=GOLD, flash_radius=1.2),
            run_time=1,
        )                                                               # 35

        # Implication
        impl = MathTex(
            r"\text{If even one monopole exists, all electric}",
            font_size=20, color=WHITE,
        )
        impl2 = MathTex(
            r"\text{charges must be integer multiples of } "
            r"e_{\min} = \frac{\hbar}{2g}",
            font_size=20, color=TEAL,
        )
        impl.to_edge(DOWN, buff=0.45)
        impl2.next_to(impl, DOWN, buff=0.12)
        self.add_fixed_in_frame_mobjects(impl, impl2)
        self.play(FadeIn(impl), run_time=0.8)                          # 36
        self.play(FadeIn(impl2), run_time=0.8)                         # 37
        self.play(Indicate(impl2, color=GOLD), run_time=0.8)            # 38

        self.wait(1.5)

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.play(
            FadeOut(quant_title), FadeOut(dirac_eq),
            FadeOut(impl), FadeOut(impl2),
            FadeOut(sphere), FadeOut(mono_dot),
            FadeOut(north_patch), FadeOut(south_patch), FadeOut(overlap),
            FadeOut(un_lbl), FadeOut(us_lbl),
            run_time=1,
        )                                                               # 39

        card_title = Text("Magnetic Monopole", font_size=36, color=GOLD)
        card_title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(card_title)
        self.play(Write(card_title))                                    # 40

        lines = VGroup(
            MathTex(
                r"\mathbf{B} = \frac{g}{4\pi r^2}\hat{r}"
                r"\;\;\text{— radial, like an electric charge}",
                font_size=19,
            ),
            MathTex(
                r"\text{Single } A_\mu \text{ needs a Dirac string"
                r" (singularity along a half-line)}",
                font_size=19,
            ),
            MathTex(
                r"\text{Two patches } U_N, U_S \text{ on } S^2"
                r"\text{ — each string-free in its hemisphere}",
                font_size=19,
            ),
            MathTex(
                r"\text{Transition } g_{NS} = e^{in\phi}: "
                r"\;\pi_1(U(1)) = \mathbb{Z}"
                r"\;\;\to\;\; n\in\mathbb{Z}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"eg = \frac{n\hbar}{2}"
                r"\;\;\text{— one monopole quantises all charges}",
                font_size=19, color=ORANGE,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        lines.next_to(card_title, DOWN, buff=0.4)

        box = SurroundingRectangle(
            VGroup(card_title, lines),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.add_fixed_in_frame_mobjects(lines, box)
        self.play(FadeIn(box), run_time=0.5)                           # 41
        for line in lines:
            self.play(FadeIn(line), run_time=0.8)                       # 42-46

        self.wait(2)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(lines), FadeOut(box),
            run_time=1.5,
        )                                                               # 47
