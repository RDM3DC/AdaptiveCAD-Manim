"""Einstein Rings — Null Geodesic Bending Around a Massive Lens.

Light rays (null geodesics) in Schwarzschild spacetime bend by:

    Δφ = 4GM / (b c²)    (weak-field deflection angle)

where b is the impact parameter.  When source, lens, and observer
are aligned, the image is a perfect ring — the Einstein ring with
angular radius:

    θ_E = √(4GM D_{LS} / (c² D_L D_S))

This animation traces multiple null rays with different impact
parameters bending around a central mass, showing ring formation.

Acts
----
1. Title card with deflection equation
2. Draw central lens mass + source position
3. Trace null geodesics with different b — rays bend toward lens
4. Show ring formation from converging rays
5. Full Einstein ring glow
6. Summary card

Run
---
    manim -pql examples/einstein_rings.py EinsteinRings
    manim -qh  examples/einstein_rings.py EinsteinRings
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
    Sphere,
    Dot3D,
    Line3D,
    Arrow3D,
    Circle,
    ParametricFunction,
    VMobject,
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
    BLACK,
    interpolate_color,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Physics — weak-field ray tracing in 2D
# ═══════════════════════════════════════════════════════════════════════════
RS_DISPLAY = 0.3       # Schwarzschild radius display size
M_PARAM = 0.4          # GM/c² in display units (controls bending)
N_RAYS = 12            # number of rays around the ring
RAY_STEPS = 200        # integration steps per ray


def _trace_ray(b, M, x_start=-4.5, x_end=4.5, n_steps=RAY_STEPS):
    """Trace a null ray with impact parameter b past a mass M at origin.

    Uses weak-field deflection: the ray travels along x, and we
    accumulate the transverse deflection from the gravitational
    potential.  For visual appeal we use the exact integral form.

    Returns (N, 2) array of (x, y) points.
    """
    # The ray starts at (x_start, b) heading in +x direction
    # Deflection angle: α = 4M / b  (total, split symmetrically)
    # We model it as a smooth bending using:
    #   dy/dx = -2M·x / (x² + b²)^{3/2} · b   (linearised geodesic eq)
    xs = np.linspace(x_start, x_end, n_steps)
    dx = xs[1] - xs[0]
    y = b
    vy = 0.0  # transverse velocity component
    pts = []
    for x in xs:
        pts.append([x, y])
        r2 = x * x + y * y
        r = np.sqrt(r2)
        if r < RS_DISPLAY * 0.8:
            break  # absorbed
        # gravitational deflection (perpendicular acceleration)
        # d²y/dx² ≈ -2M·y / r³
        ay = -2 * M * y / (r2 * r)
        vy += ay * dx
        y += vy * dx
    return np.array(pts)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class EinsteinRings(ThreeDScene):
    """Null geodesics bending into Einstein rings."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # ─── Act 1  Title ────────────────────────────────────────────────
        ttl = Text("Einstein Rings", font_size=48, color=GOLD)
        sub = Text("Gravitational lensing of null geodesics",
                    font_size=22, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"\Delta\phi = \frac{4GM}{bc^2}"
            r"\qquad"
            r"\theta_E = \sqrt{\frac{4GM\,D_{LS}}{c^2\,D_L\,D_S}}",
            font_size=22, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Lens mass + labels ───────────────────────────────────
        # Central mass
        lens_dot = Dot3D(ORIGIN, radius=0.15, color=ORANGE)
        lens_glow = Dot3D(ORIGIN, radius=0.35, color=ORANGE)
        lens_glow.set_opacity(0.15)

        # Schwarzschild radius ring
        rs_ring = ParametricFunction(
            lambda t: np.array([
                RS_DISPLAY * np.cos(t),
                RS_DISPLAY * np.sin(t), 0]),
            t_range=[0, TAU],
            color=RED, stroke_width=1, stroke_opacity=0.5,
        )

        lbl_M = MathTex(r"M", font_size=22, color=ORANGE)
        lbl_M.move_to(DOWN * 0.5)
        lbl_src = Text("source", font_size=12, color=GREY_A)
        lbl_src.move_to(LEFT * 4.8 + DOWN * 0.3)
        lbl_obs = Text("observer", font_size=12, color=GREY_A)
        lbl_obs.move_to(RIGHT * 4.8 + DOWN * 0.3)

        for lbl in [lbl_M, lbl_src, lbl_obs]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(lens_dot), FadeIn(lens_glow), Create(rs_ring),
            FadeIn(lbl_M), FadeIn(lbl_src), FadeIn(lbl_obs),
            run_time=0.8,
        )

        # ─── Act 3  Trace null rays ─────────────────────────────────────
        # Impact parameters: symmetric above and below
        b_vals = np.linspace(0.5, 2.0, N_RAYS // 2)
        ray_colors_top = [interpolate_color(YELLOW, WHITE, i / (N_RAYS // 2 - 1))
                          for i in range(N_RAYS // 2)]

        all_rays = VGroup()
        for idx, b in enumerate(b_vals):
            for sign in [1, -1]:
                pts_2d = _trace_ray(sign * b, M_PARAM)
                # Build lightweight VMobject curve
                pts_3d = np.column_stack(
                    [pts_2d, np.zeros(len(pts_2d))]
                )
                ray = VMobject(
                    color=ray_colors_top[idx],
                    stroke_width=1.5,
                )
                ray.set_points_smoothly(pts_3d)
                all_rays.add(ray)

        self.play(
            Create(all_rays, lag_ratio=0.02),
            run_time=3,
        )
        self.wait(0.5)

        # ─── Act 4  Deflection label ────────────────────────────────────
        defl_lbl = MathTex(
            r"\hat{\alpha} = \frac{4GM}{bc^2}",
            font_size=20, color=YELLOW,
        )
        defl_lbl.to_corner(UP + LEFT, buff=0.4)
        self.add_fixed_in_frame_mobjects(defl_lbl)
        self.play(FadeIn(defl_lbl), run_time=0.5)

        # ─── Act 5  Einstein ring glow ──────────────────────────────────
        # The ring forms at the convergence radius
        # For our setup, b ≈ 1.0 gives roughly the Einstein radius
        ring_r = 1.2  # approximate convergence point

        # Build the ring on the observer side (right)
        obs_x = 3.5
        ring = ParametricFunction(
            lambda t: np.array([
                obs_x + ring_r * 0.3 * np.cos(t),
                ring_r * 0.3 * np.sin(t), 0]),
            t_range=[0, TAU],
            color=YELLOW, stroke_width=4,
        )
        ring_glow = ParametricFunction(
            lambda t: np.array([
                obs_x + ring_r * 0.3 * np.cos(t),
                ring_r * 0.3 * np.sin(t), 0]),
            t_range=[0, TAU],
            color=YELLOW, stroke_width=12, stroke_opacity=0.2,
        )

        ring_lbl = MathTex(
            r"\theta_E", font_size=18, color=YELLOW,
        )
        ring_lbl.move_to(np.array([obs_x, ring_r * 0.3 + 0.3, 0]))
        self.add_fixed_in_frame_mobjects(ring_lbl)

        self.play(
            Create(ring), Create(ring_glow),
            FadeIn(ring_lbl),
            run_time=1.2,
        )
        self.wait(0.8)

        # Add full face-on Einstein ring view inset
        full_ring = ParametricFunction(
            lambda t: np.array([
                3.5 + 0.8 * np.cos(t),
                2.0 + 0.8 * np.sin(t), 0]),
            t_range=[0, TAU],
            color=GOLD, stroke_width=5,
        )
        full_ring_glow = ParametricFunction(
            lambda t: np.array([
                3.5 + 0.8 * np.cos(t),
                2.0 + 0.8 * np.sin(t), 0]),
            t_range=[0, TAU],
            color=GOLD, stroke_width=15, stroke_opacity=0.15,
        )
        fr_lbl = Text("face-on view", font_size=11, color=GREY_A)
        fr_lbl.move_to(np.array([3.5, 2.0, 0]))
        self.add_fixed_in_frame_mobjects(fr_lbl)

        self.play(
            Create(full_ring), Create(full_ring_glow),
            FadeIn(fr_lbl),
            run_time=1,
        )
        self.wait(1)

        # ─── Act 6  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in [lbl_M, lbl_src, lbl_obs,
              defl_lbl, ring_lbl, fr_lbl]],
            run_time=0.8,
        )

        card = Text("Einstein Rings", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\text{Null geodesics bend: }"
                r"\hat{\alpha}=4GM/(bc^2)",
                font_size=19,
            ),
            MathTex(
                r"\text{Perfect alignment} \;\Rightarrow\;"
                r"\text{Einstein ring at }\theta_E",
                font_size=19, color=GOLD,
            ),
            MathTex(
                r"\text{Impact parameter }b\;\text{controls "
                r"deflection strength}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Strong lensing: multiple images, arcs, "
                r"and rings}",
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
