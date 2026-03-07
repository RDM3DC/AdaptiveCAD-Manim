"""Schwarzschild Embedding — Flamm's paraboloid with geodesics & time dilation.

The spatial geometry around a non-rotating black hole, embedded as
Flamm's paraboloid z = 2√(r_s(r − r_s)), plus:
  • geodesic orbits (stable circular, plunging, deflected)
  • test particles (E8-style nodes) colour-coded by proper-time dilation
    dτ/dt = √(1 − r_s/r)
  • progressive zoom toward the horizon showing curvature steepening

Six acts:
  1. Build the Flamm paraboloid as an SDFSurface, slow orbit camera
  2. Draw Schwarzschild coordinate grid on the surface (r, φ)
  3. Geodesics — circular orbit at r = 3r_s (ISCO), plunging orbit,
     and a deflected flyby
  4. Test particles — scatter nodes on surface, colour = dτ/dt
     BLUE (far, γ≈1) → RED (near horizon, γ→∞)
  5. Zoom toward the horizon — curvature steepens, particles redshift
  6. Comparison with flat space — morph paraboloid back to plane

Run:
    manim -pql examples/flamm_paraboloid.py FlammParaboloid
    manim -qh  examples/flamm_paraboloid.py FlammParaboloid
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_D,
    BLUE_E,
    DEGREES,
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
    TAU,
    UP,
    WHITE,
    YELLOW,
    GOLD,
    MAROON,
    PURPLE,
    TEAL,
    Create,
    Dot3D,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Text,
    Tex,
    ThreeDScene,
    Transform,
    ReplacementTransform,
    VGroup,
    VMobject,
    Write,
    interpolate_color,
    rate_functions,
    Rotate,
    Uncreate,
    config,
    ParametricFunction,
)
from manim.mobject.three_d.three_dimensions import Surface

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface


# ═══════════════════════════════════════════════════════════════════════════
# Physics
# ═══════════════════════════════════════════════════════════════════════════

R_S = 1.0  # Schwarzschild radius (geometric units, G = c = 1)

# Radial range: from just outside horizon to far field
R_MIN = 1.05 * R_S   # inner edge (just outside r_s)
R_MAX = 8.0 * R_S     # outer edge
R_ISCO = 3.0 * R_S    # innermost stable circular orbit (Schwarzschild)


def flamm_z(r: float) -> float:
    """Flamm paraboloid: z = 2√(r_s(r − r_s)) for r ≥ r_s."""
    arg = R_S * (r - R_S)
    if arg < 0:
        return 0.0
    return 2.0 * np.sqrt(arg)


def dilation_factor(r: float) -> float:
    """Proper-time dilation: dτ/dt = √(1 − r_s/r)."""
    if r <= R_S:
        return 0.0
    return np.sqrt(1.0 - R_S / r)


def _dilation_color(r: float):
    """Map dilation factor to colour: BLUE (far) → YELLOW (mid) → RED (near)."""
    d = dilation_factor(r)
    if d > 0.85:
        return interpolate_color(BLUE, TEAL, (1.0 - d) / 0.15)
    elif d > 0.5:
        return interpolate_color(TEAL, YELLOW, (0.85 - d) / 0.35)
    elif d > 0.2:
        return interpolate_color(YELLOW, ORANGE, (0.5 - d) / 0.3)
    else:
        return interpolate_color(ORANGE, RED, (0.2 - d) / 0.2)


# ═══════════════════════════════════════════════════════════════════════════
# Parametric surfaces
# ═══════════════════════════════════════════════════════════════════════════

SCALE = 0.55  # scene-unit scale factor so it fits on screen


def _flamm_parametric(r_min=R_MIN, r_max=R_MAX):
    """Parametric Flamm paraboloid.  u = φ ∈ [0, TAU], v maps to r."""

    def func(u, v):
        # v ∈ [0, TAU] → r ∈ [r_min, r_max]
        r = r_min + (r_max - r_min) * v / TAU
        z = flamm_z(r)
        x = r * np.cos(u)
        y = r * np.sin(u)
        return np.array([x * SCALE, y * SCALE, z * SCALE])

    return func


def _flat_parametric(r_min=R_MIN, r_max=R_MAX):
    """Flat disk (z=0) for comparison — same domain as Flamm."""

    def func(u, v):
        r = r_min + (r_max - r_min) * v / TAU
        x = r * np.cos(u)
        y = r * np.sin(u)
        return np.array([x * SCALE, y * SCALE, 0.0])

    return func


# ═══════════════════════════════════════════════════════════════════════════
# Geodesic helpers
# ═══════════════════════════════════════════════════════════════════════════

def _circular_orbit_3d(r_orb, n=200):
    """Circular orbit at constant r on the Flamm surface."""
    pts = []
    for i in range(n + 1):
        phi = TAU * i / n
        z = flamm_z(r_orb)
        pts.append(np.array([r_orb * np.cos(phi) * SCALE,
                             r_orb * np.sin(phi) * SCALE,
                             z * SCALE]))
    curve = VMobject(color=GREEN, stroke_width=3)
    curve.set_points_smoothly(pts)
    return curve


def _plunging_orbit_3d(r_start=6.0, phi_range=2.5, n=200):
    """Radially plunging orbit that spirals in toward the horizon."""
    pts = []
    for i in range(n + 1):
        t = i / n
        phi = phi_range * t
        # r decreases from r_start toward r_s with angular momentum
        r = r_start - (r_start - R_MIN) * t ** 1.3
        z = flamm_z(r)
        pts.append(np.array([r * np.cos(phi) * SCALE,
                             r * np.sin(phi) * SCALE,
                             z * SCALE]))
    curve = VMobject(color=RED, stroke_width=3)
    curve.set_points_smoothly(pts)
    return curve


def _deflected_orbit_3d(r_min_approach=2.0, n=200):
    """Hyperbolic flyby — deflected by the black hole, escapes to infinity."""
    pts = []
    phi_range = PI * 1.3
    for i in range(n + 1):
        t = i / n
        phi = -phi_range / 2 + phi_range * t
        # r follows a rough hyperbolic shape
        r = r_min_approach + (R_MAX - r_min_approach) * (2 * t - 1) ** 2
        r = max(r, R_MIN)
        z = flamm_z(r)
        pts.append(np.array([r * np.cos(phi) * SCALE,
                             r * np.sin(phi) * SCALE,
                             z * SCALE]))
    curve = VMobject(color=YELLOW, stroke_width=3)
    curve.set_points_smoothly(pts)
    return curve


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate grid on the surface
# ═══════════════════════════════════════════════════════════════════════════

def _radial_grid_lines(n_r=6, n_phi=12, pts_per_line=80):
    """Build constant-r circles and constant-φ radial lines on the surface."""
    lines = VGroup()

    # Constant-r circles
    r_values = np.linspace(R_MIN + 0.3, R_MAX - 0.3, n_r)
    for r in r_values:
        pts = []
        for i in range(pts_per_line + 1):
            phi = TAU * i / pts_per_line
            z = flamm_z(r)
            pts.append(np.array([r * np.cos(phi) * SCALE,
                                 r * np.sin(phi) * SCALE,
                                 z * SCALE]))
        ring = VMobject(color=GREY, stroke_width=1.0, stroke_opacity=0.5)
        ring.set_points_smoothly(pts)
        lines.add(ring)

    # Constant-φ radial lines
    phi_values = np.linspace(0, TAU, n_phi, endpoint=False)
    for phi in phi_values:
        pts = []
        for i in range(pts_per_line + 1):
            r = R_MIN + (R_MAX - R_MIN) * i / pts_per_line
            z = flamm_z(r)
            pts.append(np.array([r * np.cos(phi) * SCALE,
                                 r * np.sin(phi) * SCALE,
                                 z * SCALE]))
        radial = VMobject(color=GREY, stroke_width=0.8, stroke_opacity=0.4)
        radial.set_points_smoothly(pts)
        lines.add(radial)

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Test particles
# ═══════════════════════════════════════════════════════════════════════════

def _test_particles(n=24):
    """Scatter test particles on the Flamm surface, coloured by dτ/dt."""
    particles = VGroup()
    np.random.seed(7)
    for _ in range(n):
        r = np.random.uniform(R_MIN + 0.1, R_MAX - 0.5)
        phi = np.random.uniform(0, TAU)
        z = flamm_z(r)
        pos = np.array([r * np.cos(phi) * SCALE,
                        r * np.sin(phi) * SCALE,
                        z * SCALE])
        col = _dilation_color(r)
        dot = Dot3D(pos, color=col, radius=0.06)
        particles.add(dot)
    return particles


def _test_particles_labels():
    """Colour bar legend for dilation factor."""
    # small coloured dots + labels, fixed in frame
    legend = VGroup()
    samples = [
        (1.0, "far field", BLUE),
        (0.7, "moderate", TEAL),
        (0.4, "strong", YELLOW),
        (0.1, "extreme", RED),
    ]
    for i, (_, label, col) in enumerate(samples):
        from manim import Dot, Square
        swatch = Square(side_length=0.18, color=col,
                        fill_color=col, fill_opacity=1,
                        stroke_width=0.5).shift(RIGHT * 0 + DOWN * i * 0.35)
        txt = Text(label, font_size=16, color=col).next_to(swatch, RIGHT, buff=0.1)
        legend.add(VGroup(swatch, txt))
    legend.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    return legend


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class FlammParaboloid(ThreeDScene):
    """Six-act Flamm paraboloid animation."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES,
                                    zoom=0.85)
        self._act1_build_paraboloid()
        self._act2_coordinate_grid()
        self._act3_geodesics()
        self._act4_test_particles()
        self._act5_zoom_horizon()
        self._act6_comparison()

    # ── Act 1 — Build ────────────────────────────────────────────────
    def _act1_build_paraboloid(self):
        title = Text("Schwarzschild Embedding", font_size=38, color=WHITE)
        subtitle = Text("Flamm's paraboloid", font_size=24, color=GREY_A)
        subtitle.next_to(title, DOWN, buff=0.15)
        grp = VGroup(title, subtitle)
        self.add_fixed_in_frame_mobjects(grp)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2), run_time=1.5)
        self.wait(0.6)
        self.play(FadeOut(grp), run_time=0.5)

        # Equation
        eq = MathTex(
            r"z = 2\sqrt{r_s\,(r - r_s)}",
            font_size=30,
        ).to_corner(UP + RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(Write(eq), run_time=1)

        # Build the paraboloid
        self.paraboloid = SDFSurface(
            _flamm_parametric(),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(64, 48),
            color=BLUE_D,
            opacity=0.65,
        )
        self.play(FadeIn(self.paraboloid), run_time=2.5)

        # Slow orbit
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # Horizon indicator — ring at r = r_s
        horizon_pts = []
        for i in range(201):
            phi = TAU * i / 200
            z = flamm_z(R_S * 1.01)
            horizon_pts.append(np.array([R_S * 1.01 * np.cos(phi) * SCALE,
                                         R_S * 1.01 * np.sin(phi) * SCALE,
                                         z * SCALE]))
        self.horizon_ring = VMobject(color=RED, stroke_width=3)
        self.horizon_ring.set_points_smoothly(horizon_pts)

        h_label = MathTex(r"r = r_s", font_size=22, color=RED).to_corner(
            DOWN + RIGHT, buff=0.3
        )
        self.add_fixed_in_frame_mobjects(h_label)
        self.play(Create(self.horizon_ring), Write(h_label), run_time=1.5)
        self.wait(1)

        self.eq = eq
        self.h_label = h_label

    # ── Act 2 — Coordinate grid ──────────────────────────────────────
    def _act2_coordinate_grid(self):
        grid_label = Text("Schwarzschild coords (r, φ)", font_size=20,
                          color=GREY_A).to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(grid_label)

        self.grid = _radial_grid_lines()
        self.play(Create(self.grid), Write(grid_label), run_time=2.5)

        # Note how the grid squares stretch near the horizon
        note = Text("Grid cells stretch → spatial curvature", font_size=18,
                     color=GOLD).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(Write(note), run_time=1)

        self.begin_ambient_camera_rotation(rate=0.06)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(note), FadeOut(grid_label), run_time=0.5)

    # ── Act 3 — Geodesics ────────────────────────────────────────────
    def _act3_geodesics(self):
        geo_title = Text("Geodesics on curved spacetime", font_size=22,
                         color=WHITE).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(geo_title)
        self.play(Write(geo_title), run_time=0.8)

        # ISCO circular orbit
        isco = _circular_orbit_3d(R_ISCO)
        isco_lbl = Text("ISCO (r = 3rₛ)", font_size=18, color=GREEN).to_corner(
            DOWN + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(isco_lbl)
        self.play(Create(isco), Write(isco_lbl), run_time=2)

        # Plunging orbit
        plunge = _plunging_orbit_3d()
        plunge_lbl = Text("plunging spiral", font_size=18, color=RED).next_to(
            isco_lbl, UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(plunge_lbl)
        self.play(Create(plunge), Write(plunge_lbl), run_time=2)

        # Deflected flyby
        flyby = _deflected_orbit_3d()
        flyby_lbl = Text("deflected flyby", font_size=18, color=YELLOW).next_to(
            plunge_lbl, UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(flyby_lbl)
        self.play(Create(flyby), Write(flyby_lbl), run_time=2)

        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.geodesics = VGroup(isco, plunge, flyby)
        self.geo_labels = VGroup(isco_lbl, plunge_lbl, flyby_lbl)
        self.play(FadeOut(geo_title), run_time=0.5)

    # ── Act 4 — Test particles coloured by dτ/dt ────────────────────
    def _act4_test_particles(self):
        # Fade geodesics
        self.play(FadeOut(self.geodesics), FadeOut(self.geo_labels), run_time=0.6)

        dilation_eq = MathTex(
            r"\frac{d\tau}{dt} = \sqrt{1 - \frac{r_s}{r}}",
            font_size=28,
        ).to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(dilation_eq)
        self.play(Write(dilation_eq), run_time=1)

        particles = _test_particles(n=30)
        self.play(FadeIn(particles), run_time=2)

        # Legend
        legend = _test_particles_labels()
        legend.to_corner(DOWN + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(legend)
        self.play(FadeIn(legend), run_time=1)

        cap = Text("Colour = proper-time dilation", font_size=18,
                    color=GOLD).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(Write(cap), run_time=0.8)

        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(particles), FadeOut(legend), FadeOut(cap),
                  FadeOut(dilation_eq), run_time=0.8)
        self.particles = particles

    # ── Act 5 — Zoom toward horizon ──────────────────────────────────
    def _act5_zoom_horizon(self):
        zoom_cap = Text("Approaching the horizon — curvature diverges",
                        font_size=20, color=RED).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(zoom_cap)
        self.play(Write(zoom_cap), run_time=0.8)

        # Camera zoom in + tilt
        self.move_camera(
            phi=40 * DEGREES, theta=-30 * DEGREES,
            zoom=1.8,
            run_time=4,
        )
        self.wait(2)

        # Proper time label at horizon
        horizon_note = MathTex(
            r"\frac{d\tau}{dt}\to 0 \;\;\text{as}\;\; r\to r_s",
            font_size=26, color=RED,
        ).to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(horizon_note)
        self.play(Write(horizon_note), run_time=1)
        self.wait(2)

        # Zoom back out
        self.move_camera(
            phi=60 * DEGREES, theta=-45 * DEGREES,
            zoom=0.85,
            run_time=3,
        )
        self.play(FadeOut(zoom_cap), FadeOut(horizon_note), run_time=0.5)

    # ── Act 6 — Compare with flat space ──────────────────────────────
    def _act6_comparison(self):
        # Morph paraboloid to flat disk
        compare_cap = Text("Curved spacetime → flat space (r_s → 0)",
                           font_size=20, color=GOLD).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(compare_cap)
        self.play(Write(compare_cap), run_time=0.8)

        flat_disk = SDFSurface(
            _flat_parametric(),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(64, 48),
            color=BLUE,
            opacity=0.5,
        )

        # Also flatten the grid
        flat_grid = _radial_grid_lines()
        # Override z to zero for flat grid
        for mob in flat_grid:
            pts = mob.get_points()
            pts[:, 2] = 0  # flatten
            mob.set_points(pts)

        self.play(
            Transform(self.paraboloid, flat_disk),
            Transform(self.grid, flat_grid),
            self.horizon_ring.animate.set_opacity(0),
            run_time=4,
            rate_func=rate_functions.ease_in_out_cubic,
        )

        flat_note = MathTex(
            r"r_s \to 0 \;\Rightarrow\; \text{Minkowski}",
            font_size=26,
        ).to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(flat_note)
        self.play(Write(flat_note), run_time=1)

        self.begin_ambient_camera_rotation(rate=0.06)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # Morph back to curved
        curved_again = SDFSurface(
            _flamm_parametric(),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(64, 48),
            color=BLUE_D,
            opacity=0.65,
        )
        curved_grid = _radial_grid_lines()

        self.play(
            Transform(self.paraboloid, curved_again),
            Transform(self.grid, curved_grid),
            self.horizon_ring.animate.set_opacity(1),
            run_time=4,
            rate_func=rate_functions.ease_in_out_cubic,
        )
        self.wait(1)

        # Finale
        self.play(
            FadeOut(self.paraboloid), FadeOut(self.grid),
            FadeOut(self.horizon_ring), FadeOut(self.eq),
            FadeOut(self.h_label), FadeOut(compare_cap),
            FadeOut(flat_note),
            run_time=1.5,
        )

        final = Text(
            "Mass curves space.\nSpace tells matter how to move.",
            font_size=28, color=GOLD, line_spacing=1.4,
        )
        self.add_fixed_in_frame_mobjects(final)
        self.play(Write(final), run_time=2)
        self.wait(2)
        self.play(FadeOut(final), run_time=1)
