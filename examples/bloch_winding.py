"""Bloch Sphere with Topological Winding — Phase Coherence Visualization.

Dual-panel 3-D scene inspired by topological quantum computing diagrams:

  Left:  Translucent Bloch sphere with rainbow state-evolution arc and
         glowing state markers
  Right: Unwrapped helical ribbon (orange / blue) revealing the integer
         winding number w; phase disk at base shows the π_a field

The qubit state |ψ(t)⟩ traces a path on S² while the Riemann
phase θ_R unwraps into a helix.  Each full twist of the ribbon
adds one unit of winding number.

Acts
----
1. Title card
2. Build Bloch sphere wireframe + helix scaffold
3. Rainbow trajectory on sphere; helical ribbons grow alongside
4. Phase disk at helix base, winding-number labels
5. Slow camera orbit for depth appreciation
6. Summary card

Run
---
    manim -pql examples/bloch_winding.py BlochWinding
    manim -qh  examples/bloch_winding.py BlochWinding
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Sphere,
    Arrow3D,
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

# ═══════════════════════════════════════════════════════════════════════════
# Layout & geometry constants
# ═══════════════════════════════════════════════════════════════════════════
BLOCH_R  = 1.5               # Bloch-sphere display radius
HELIX_R  = 0.65              # helix cylinder radius
HELIX_H  = 3.6               # total helix height
N_WINDS  = 2                 # number of full windings
HALF_W   = PI / 3            # angular half-width of each ribbon strip
LEFT_X   = -3.0              # sphere x-centre (screen-left)
RIGHT_X  = 3.2               # helix  x-centre (screen-right)
DEG      = PI / 180          # degree → radian factor
ARC_SPAN = 5 * PI / 3        # trajectory arc extent (300°)

# Rainbow palette for the state trajectory
_RAIN = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]


# ═══════════════════════════════════════════════════════════════════════════
# Parametric-geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def _traj_point(t: float, center: np.ndarray) -> np.ndarray:
    """Point on a great circle tilted 50° from the equator."""
    tilt = 50 * DEG
    return center + BLOCH_R * np.array([
        np.cos(t),
        np.sin(t) * np.cos(tilt),
        np.sin(t) * np.sin(tilt),
    ])


def _rainbow_color(frac: float):
    """Map fraction ∈ [0, 1] → rainbow colour."""
    n = len(_RAIN) - 1
    idx = frac * n
    lo = min(int(idx), n - 1)
    return interpolate_color(_RAIN[lo], _RAIN[lo + 1], idx - lo)


def _helix_ribbon_func(R, n_winds, height, half_width,
                        phase=0.0, cx=0.0):
    """Parametric helical ribbon on a cylinder.

    u ∈ [0, n_winds·2π]  — along helix
    v ∈ [0, 1]           — across ribbon width
    """
    pitch = height / n_winds
    z_base = -height / 2

    def func(u, v):
        angle = u + phase + (2 * v - 1) * half_width
        z = z_base + pitch * u / TAU
        return np.array([
            cx + R * np.cos(angle),
            R * np.sin(angle),
            z,
        ])

    return func


def _annulus_func(r_in, r_out, cx=0.0, z_level=0.0):
    """Half-annulus for the phase disk.  Use u_range [0,π] or [π,2π]."""

    def func(u, v):
        r = r_in + v * (r_out - r_in)
        return np.array([cx + r * np.cos(u), r * np.sin(u), z_level])

    return func


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class BlochWinding(ThreeDScene):
    """Bloch sphere + helical winding-number visualisation."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Bloch Sphere & Topological Winding",
                    font_size=42, color=GOLD)
        sub = Text("Phase coherence via winding number",
                    font_size=22, color=BLUE_D)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"w = \frac{1}{2\pi}\oint d\theta_R"
            r"\;\in\;\mathbb{Z}",
            font_size=28, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.6)
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Build scaffolding ────────────────────────────────────
        self.set_camera_orientation(phi=65 * DEG, theta=-50 * DEG)

        sc = np.array([LEFT_X, 0.0, 0.0])     # sphere centre
        hc = np.array([RIGHT_X, 0.0, 0.0])    # helix  centre

        # -- Bloch sphere --
        sphere = Sphere(
            radius=BLOCH_R, resolution=(48, 32),
            fill_opacity=0.06, stroke_width=0.3,
            stroke_color=BLUE_E,
        )
        sphere.set_color(BLUE_E)
        sphere.move_to(sc)

        al = BLOCH_R * 1.3  # axis half-length
        x_ax = Arrow3D(sc - al * RIGHT, sc + al * RIGHT,
                        color=RED, thickness=0.01)
        y_ax = Arrow3D(sc - al * np.array([0, 1, 0]),
                        sc + al * np.array([0, 1, 0]),
                        color=GREEN, thickness=0.01)
        z_ax = Arrow3D(sc - al * np.array([0, 0, 1]),
                        sc + al * np.array([0, 0, 1]),
                        color=BLUE, thickness=0.01)

        equator = ParametricFunction(
            lambda t: sc + BLOCH_R * np.array(
                [np.cos(t), np.sin(t), 0]),
            t_range=[0, TAU], color=GREY_A, stroke_width=0.8,
        )

        # -- Helix central axis --
        h_ax = Line3D(
            hc + np.array([0, 0, -HELIX_H * 0.55]),
            hc + np.array([0, 0,  HELIX_H * 0.55]),
            color=WHITE, thickness=0.003,
        )

        self.play(
            Create(sphere), Create(equator),
            Create(x_ax), Create(y_ax), Create(z_ax),
            Create(h_ax),
            run_time=1.5,
        )

        # -- Fixed-in-frame labels --
        lbl_psi = MathTex(r"|\psi_0\rangle", font_size=20, color=BLUE)
        lbl_psi.move_to(UP * 2.8 + LEFT * 3.0)

        lbl_x = Text("X", font_size=16, color=RED)
        lbl_x.move_to(LEFT * 5.0)

        lbl_y = Text("Y", font_size=16, color=GREEN)
        lbl_y.move_to(LEFT * 1.5 + DOWN * 0.8)

        lbl_rea = MathTex(r"\mathrm{Re}(a)", font_size=14, color=GREEN)
        lbl_rea.next_to(lbl_y, DOWN, buff=0.15)

        lbl_htop = MathTex(r"|\psi_0\rangle", font_size=18, color=YELLOW)
        lbl_htop.move_to(UP * 2.8 + RIGHT * 3.2)

        lbl_w0 = MathTex(r"0", font_size=22, color=WHITE)
        lbl_w0.move_to(RIGHT * 3.8 + DOWN * 0.2)

        lbl_w1 = MathTex(r"1", font_size=22, color=WHITE)
        lbl_w1.move_to(RIGHT * 3.8 + UP * 1.3)

        lbl_wind = Text("winding", font_size=13, color=GREY_A)
        lbl_wind.move_to(RIGHT * 3.2 + DOWN * 2.5)

        frame_lbls = [lbl_psi, lbl_x, lbl_y, lbl_rea,
                       lbl_htop, lbl_w0, lbl_w1, lbl_wind]
        for lbl in frame_lbls:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(lbl_psi), FadeIn(lbl_x), FadeIn(lbl_y),
            run_time=0.5,
        )

        # ─── Act 3  Rainbow trajectory + helical ribbons ─────────────────
        N_SEG = 60
        t_vals = np.linspace(0, ARC_SPAN, N_SEG + 1)

        traj_lines = VGroup()
        for i in range(N_SEG):
            seg = Line3D(
                _traj_point(t_vals[i], sc),
                _traj_point(t_vals[i + 1], sc),
                color=_rainbow_color(i / N_SEG),
                thickness=0.018,
            )
            traj_lines.add(seg)

        dot_start = Dot3D(_traj_point(0, sc),
                          radius=0.07, color=RED)
        dot_end   = Dot3D(_traj_point(ARC_SPAN, sc),
                          radius=0.07, color=RED)

        # Helical ribbons — orange and blue offset by π
        ribbon_a = Surface(
            _helix_ribbon_func(HELIX_R, N_WINDS, HELIX_H, HALF_W,
                               phase=0.0, cx=RIGHT_X),
            u_range=[0, N_WINDS * TAU],
            v_range=[0, 1],
            resolution=(64, 6),
            fill_color=ORANGE, fill_opacity=0.85,
            stroke_width=0.3, stroke_color=ORANGE,
        )
        ribbon_b = Surface(
            _helix_ribbon_func(HELIX_R, N_WINDS, HELIX_H, HALF_W,
                               phase=PI, cx=RIGHT_X),
            u_range=[0, N_WINDS * TAU],
            v_range=[0, 1],
            resolution=(64, 6),
            fill_color=BLUE_D, fill_opacity=0.85,
            stroke_width=0.3, stroke_color=BLUE_D,
        )

        self.play(
            FadeIn(dot_start),
            Create(traj_lines, lag_ratio=0.03),
            Create(ribbon_a), Create(ribbon_b),
            run_time=3,
        )
        self.play(FadeIn(dot_end), run_time=0.4)

        # ─── Act 4  Phase disk + winding labels ─────────────────────────
        disk_z = -HELIX_H / 2 - 0.1
        disk_a = Surface(
            _annulus_func(HELIX_R * 0.3, HELIX_R * 1.3,
                          cx=RIGHT_X, z_level=disk_z),
            u_range=[0, PI], v_range=[0, 1],
            resolution=(24, 4),
            fill_color=ORANGE, fill_opacity=0.45,
            stroke_width=0.2, stroke_color=ORANGE,
        )
        disk_b = Surface(
            _annulus_func(HELIX_R * 0.3, HELIX_R * 1.3,
                          cx=RIGHT_X, z_level=disk_z),
            u_range=[PI, TAU], v_range=[0, 1],
            resolution=(24, 4),
            fill_color=BLUE_D, fill_opacity=0.45,
            stroke_width=0.2, stroke_color=BLUE_D,
        )

        peak_dot = Dot3D(
            hc + np.array([0, 0, HELIX_H / 2]),
            radius=0.08, color=RED,
        )

        self.play(
            FadeIn(disk_a), FadeIn(disk_b), FadeIn(peak_dot),
            FadeIn(lbl_htop), FadeIn(lbl_w0), FadeIn(lbl_w1),
            FadeIn(lbl_wind), FadeIn(lbl_rea),
            run_time=0.8,
        )
        self.wait(0.5)

        # ─── Act 5  Camera orbit ─────────────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(0.3)

        # ─── Act 6  Summary ──────────────────────────────────────────────
        all_3d = [sphere, equator, x_ax, y_ax, z_ax,
                  h_ax, traj_lines, dot_start, dot_end,
                  ribbon_a, ribbon_b, disk_a, disk_b, peak_dot]

        self.play(
            *[FadeOut(m) for m in all_3d],
            *[FadeOut(lbl) for lbl in frame_lbls],
            run_time=0.8,
        )

        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Topological Winding", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"|\psi\rangle \;\text{traces a path on }S^2",
                font_size=20,
            ),
            MathTex(
                r"\theta_R \;\text{unwraps into helical ribbon}",
                font_size=20, color=ORANGE,
            ),
            MathTex(
                r"w = \tfrac{1}{2\pi}\oint d\theta_R"
                r"\;\in\;\mathbb{Z}"
                r"\;\;\text{(integer winding)}",
                font_size=20, color=YELLOW,
            ),
            MathTex(
                r"\text{Topological protection: }w"
                r"\text{ robust against smooth perturbations}",
                font_size=18, color=TEAL,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card, DOWN, buff=0.35)

        box = SurroundingRectangle(
            VGroup(card, bullets),
            color=GOLD, buff=0.25, corner_radius=0.1,
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
