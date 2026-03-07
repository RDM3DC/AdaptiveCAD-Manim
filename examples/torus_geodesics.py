"""Geodesics on a Torus — Rational vs Irrational Winding.

A geodesic on a flat torus wraps with slope p/q.
• Rational slope → closed orbit after q windings
• Irrational slope → orbit never closes, fills the torus densely

We draw the torus in 3D (ThreeDScene) and trace geodesics with
different winding ratios, showing the beautiful spiralling paths.

Acts
----
1. Title card
2. Draw the torus
3. Rational geodesic p/q = 1/1 (meridian-longitude diagonal)
4. Rational geodesic p/q = 2/3
5. Rational geodesic p/q = 3/5
6. Irrational geodesic (golden ratio) — dense fill
7. Summary card

Run
---
    manim -pql examples/torus_geodesics.py TorusGeodesics
    manim -qh  examples/torus_geodesics.py TorusGeodesics
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    ParametricFunction,
    VMobject,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    Uncreate,
    Transform,
    Indicate,
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
    config,
)

# ═══════════════════════════════════════════════════════════════════════════
# Torus parameters
# ═══════════════════════════════════════════════════════════════════════════
R_MAJOR = 2.0       # distance from torus centre to tube centre
R_MINOR = 0.8       # tube radius


def _torus_pt(u, v):
    """Torus parametrisation: u = longitude [0,2pi], v = meridian [0,2pi]."""
    x = (R_MAJOR + R_MINOR * np.cos(v)) * np.cos(u)
    y = (R_MAJOR + R_MINOR * np.cos(v)) * np.sin(u)
    z = R_MINOR * np.sin(v)
    return np.array([x, y, z])


def _geodesic_curve(p, q, n_pts=600, windings=None):
    """Geodesic on the torus with winding ratio p/q.

    p = number of times around the meridian per q times around longitude.
    For irrational slope, pass a float for p/q via windings parameter.
    """
    if windings is not None:
        slope = windings
        total_u = TAU * 8  # many loops for dense fill
    else:
        slope = p / q
        total_u = TAU * q  # exactly q full longitude trips

    t = np.linspace(0, total_u, n_pts)
    pts = np.array([_torus_pt(u, slope * u) for u in t])
    return pts


def _make_geodesic_mob(pts, color=YELLOW, width=2.5):
    """Convert Nx3 array to a ParametricFunction-like VMobject."""
    mob = VMobject(color=color, stroke_width=width)
    mob.set_points_as_corners(pts)
    return mob


def _hue_hex(h):
    """HSV hue [0,1] → hex colour string."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.9, 1.0)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class TorusGeodesics(ThreeDScene):
    """Geodesics on a torus: closed rational vs dense irrational."""

    def construct(self):
        # ─────────────────────────────────────────────────────────────────
        # Act 1 — Title (fixed frame for text)
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Geodesics on a Torus", font_size=48, color=GOLD)
        sub = Text("Rational = closed, Irrational = dense",
                    font_size=24, color=BLUE)
        sub.next_to(ttl, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub)
        self.play(Write(ttl), run_time=1.2)                              # 1
        self.play(FadeIn(sub), run_time=0.8)                             # 2
        self.wait(0.8)
        self.play(FadeOut(ttl), FadeOut(sub))                            # 3

        # ─────────────────────────────────────────────────────────────────
        # Act 2 — Draw the torus
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(phi=65 * PI / 180, theta=-50 * PI / 180)

        torus = Surface(
            lambda u, v: _torus_pt(u, v),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(48, 24),
            fill_opacity=0.15,
            stroke_width=0.5,
            stroke_color=GREY_D,
            checkerboard_colors=[BLUE_E, BLUE_D],
        )

        self.play(Create(torus), run_time=1.5)                           # 4

        # ─────────────────────────────────────────────────────────────────
        # Act 3 — Rational 1/1
        # ─────────────────────────────────────────────────────────────────
        lbl_11 = MathTex(r"p/q = 1/1", font_size=28, color=YELLOW)
        lbl_11.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_11)

        pts_11 = _geodesic_curve(1, 1)
        geo_11 = _make_geodesic_mob(pts_11, YELLOW, 3)

        self.play(FadeIn(lbl_11), run_time=0.3)                          # 5
        self.play(Create(geo_11), run_time=2)                            # 6

        note_11 = MathTex(
            r"\text{Closes after 1 longitude loop}",
            font_size=18, color=YELLOW,
        )
        note_11.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(note_11)
        self.play(FadeIn(note_11), run_time=0.4)                         # 7

        self.wait(0.5)
        self.play(FadeOut(geo_11), FadeOut(note_11), FadeOut(lbl_11),
                  run_time=0.5)                                           # 8

        # ─────────────────────────────────────────────────────────────────
        # Act 4 — Rational 2/3
        # ─────────────────────────────────────────────────────────────────
        lbl_23 = MathTex(r"p/q = 2/3", font_size=28, color=TEAL)
        lbl_23.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_23)

        pts_23 = _geodesic_curve(2, 3, n_pts=800)
        geo_23 = _make_geodesic_mob(pts_23, TEAL, 2.5)

        self.play(FadeIn(lbl_23), run_time=0.3)                          # 9
        self.play(Create(geo_23), run_time=2.5)                          # 10

        note_23 = MathTex(
            r"\text{Closes after 3 longitude loops}",
            font_size=18, color=TEAL,
        )
        note_23.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(note_23)
        self.play(FadeIn(note_23), run_time=0.4)                         # 11

        self.wait(0.5)
        self.play(FadeOut(geo_23), FadeOut(note_23), FadeOut(lbl_23),
                  run_time=0.5)                                           # 12

        # ─────────────────────────────────────────────────────────────────
        # Act 5 — Rational 3/5
        # ─────────────────────────────────────────────────────────────────
        lbl_35 = MathTex(r"p/q = 3/5", font_size=28, color=ORANGE)
        lbl_35.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_35)

        pts_35 = _geodesic_curve(3, 5, n_pts=1000)
        geo_35 = _make_geodesic_mob(pts_35, ORANGE, 2)

        self.play(FadeIn(lbl_35), run_time=0.3)                          # 13
        self.play(Create(geo_35), run_time=3)                            # 14

        note_35 = MathTex(
            r"\text{Closes after 5 longitude loops}",
            font_size=18, color=ORANGE,
        )
        note_35.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(note_35)
        self.play(FadeIn(note_35), run_time=0.4)                         # 15

        self.wait(0.5)
        self.play(FadeOut(geo_35), FadeOut(note_35), FadeOut(lbl_35),
                  run_time=0.5)                                           # 16

        # ─────────────────────────────────────────────────────────────────
        # Act 6 — Irrational (golden ratio) — dense fill
        # ─────────────────────────────────────────────────────────────────
        golden = (1 + np.sqrt(5)) / 2

        lbl_gold = MathTex(
            r"p/q = \varphi = \frac{1+\sqrt{5}}{2}",
            font_size=28, color=GOLD,
        )
        lbl_gold.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_gold)

        pts_gold = _geodesic_curve(0, 0, n_pts=3000, windings=golden)
        geo_gold = _make_geodesic_mob(pts_gold, GOLD, 1.2)

        self.play(FadeIn(lbl_gold), run_time=0.3)                        # 17
        self.play(Create(geo_gold), run_time=5)                          # 18

        note_gold = MathTex(
            r"\text{Never closes --- fills the torus densely!}",
            font_size=18, color=GOLD,
        )
        note_gold.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(note_gold)
        self.play(FadeIn(note_gold), run_time=0.5)                       # 19

        # Camera orbit for beauty
        self.move_camera(theta=-120 * PI / 180, run_time=3)              # 20

        self.wait(0.5)

        self.play(
            FadeOut(geo_gold), FadeOut(note_gold), FadeOut(lbl_gold),
            FadeOut(torus),
            run_time=0.8,
        )                                                                 # 21

        # ─────────────────────────────────────────────────────────────────
        # Act 7 — Summary
        # ─────────────────────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card_title = Text(
            "Geodesics on a Torus", font_size=34, color=GOLD,
        )
        card_title.to_edge(UP, buff=0.5)
        self.add_fixed_in_frame_mobjects(card_title)
        self.play(Write(card_title))                                      # 22

        bullets = VGroup(
            MathTex(
                r"\text{Flat torus: geodesics are straight lines on } "
                r"[0,2\pi)^2",
                font_size=18,
            ),
            MathTex(
                r"\text{Slope } p/q \text{ rational } "
                r"\Rightarrow \text{ closed after } q \text{ loops}",
                font_size=18, color=TEAL,
            ),
            MathTex(
                r"\text{Slope irrational } "
                r"\Rightarrow \text{ orbit is dense (never closes)}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\varphi = (1+\sqrt{5})/2 "
                r"\text{ --- most irrational, slowest to fill}",
                font_size=18, color=GOLD,
            ),
            MathTex(
                r"\text{KAM theory: irrational tori survive perturbation}",
                font_size=18, color=YELLOW,
            ),
            MathTex(
                r"\text{Topology meets dynamics: winding number } \in "
                r"\mathbb{Q} \text{ vs } \mathbb{R}\setminus\mathbb{Q}",
                font_size=18, color=GREEN,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card_title, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card_title, bullets),
            color=GOLD, buff=0.25, corner_radius=0.1,
        )

        self.add_fixed_in_frame_mobjects(box, *bullets)
        self.play(FadeIn(box), run_time=0.5)                             # 23
        for b in bullets:
            self.play(FadeIn(b), run_time=0.6)                           # 24-29

        self.wait(2)
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(card_title), FadeOut(box),
            *[FadeOut(b) for b in bullets],
            run_time=1.5,
        )                                                                 # 30
