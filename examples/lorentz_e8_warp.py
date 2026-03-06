"""E8 root bloom through relativistic Lorentz boost — spacetime warp.

The 248-node E8 root system is projected to 3D, then each frame
applies a Lorentz boost parametrised by α, stretching the starburst
along the light-cone axis.  Dendrite edges morph into trefoil-knot
filaments as α ramps through spacetime twists.  Slow-motion warp
effect via time-dilation scaling.

All geometry is parametric — no triangles.

Run:
    manim -pql examples/lorentz_e8_warp.py LorentzBloom
    manim -pql examples/lorentz_e8_warp.py LightConeDendrites
"""

from __future__ import annotations

import sys
import os
import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    PI,
    PURPLE,
    RED,
    RIGHT,
    TAU,
    UP,
    WHITE,
    YELLOW,
    BLUE_D,
    BLUE_E,
    RED_E,
    TEAL,
    GOLD,
    MAROON,
    PINK,
    GREEN_E,
    GREY,
    Line3D,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    MathTex,
    Rotate,
    Text,
    ThreeDScene,
    Transform,
    VGroup,
    Write,
    interpolate_color,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface

_TAU = 2.0 * np.pi
_PI = np.pi

MU = 0.8
V_RAMP = 1.2


# ---- E8 roots (reused from e8_root_system) --------------------------------

def _generate_e8_roots():
    """Generate the 240 roots of E8 in 8D."""
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    r = np.zeros(8)
                    r[i] = si
                    r[j] = sj
                    roots.append(r)
    for bits in range(256):
        signs = np.array([(bits >> k) & 1 for k in range(8)], dtype=np.float64)
        signs = 1.0 - 2.0 * signs
        if np.sum(signs < 0) % 2 == 0:
            roots.append(signs * 0.5)
    return np.array(roots)


def _project_8d_to_3d(roots_8d):
    """Coxeter-inspired projection from 8D to 3D."""
    phi = (1 + np.sqrt(5)) / 2
    proj = np.array([
        [1, phi, 0, -1 / phi, 1, 0, -1, phi],
        [phi, 0, 1, 1, -1 / phi, -1, phi, 0],
        [0, 1, phi, 0, 1, phi, 0, -1 / phi],
    ]) / np.sqrt(8)
    pts = (proj @ roots_8d.T).T
    mx = np.max(np.linalg.norm(pts, axis=1))
    pts *= 2.8 / (mx + 1e-12)
    return pts


def _subsample_roots(pts, n=80):
    """Farthest-point sampling for visual clarity."""
    if len(pts) <= n:
        return pts, np.arange(len(pts))
    sel = [0]
    md = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(n - 1):
        f = np.argmax(md)
        sel.append(f)
        nd = np.linalg.norm(pts - pts[f], axis=1)
        md = np.minimum(md, nd)
        md[sel] = -1
    idx = np.array(sel)
    return pts[idx], idx


def _nearest_edges(pts, k=4):
    """k-nearest neighbour edges."""
    n = len(pts)
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.linalg.norm(pts - pts[i], axis=1)
        d[i, i] = np.inf
    kth = []
    for i in range(n):
        sd = np.sort(d[i])
        kth.append(sd[min(k - 1, len(sd) - 1)])
    cutoff = float(np.median(kth)) * 1.1
    edges = set()
    for i in range(n):
        for j in np.where(d[i] < cutoff)[0][:k]:
            edges.add((min(i, j), max(i, j)))
    return list(edges)


# ---- Lorentz boost ---------------------------------------------------------

def _lorentz_boost_3d(pts, beta, axis=None):
    """Apply a Lorentz boost to 3D spatial points.

    Treats z as the "spatial boost direction" and computes the
    contracted/dilated coordinates as if embedding in Minkowski
    spacetime with ct=0 slice.  The boost stretches points along
    the axis by the Lorentz factor γ.

    β = v/c, clamped to (-1, 1).
    """
    beta = float(np.clip(beta, -0.99, 0.99))
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)

    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # Project each point onto and perpendicular to boost axis
    proj = np.outer(pts @ axis, axis)        # (N,3)
    perp = pts - proj                         # (N,3)

    # Lorentz contraction along boost axis → but for visual drama
    # we show the γ dilation (length contraction is 1/γ, but we want
    # the starburst stretch, so we apply γ directly)
    boosted = perp + proj * gamma

    return boosted


def _time_dilation_factor(beta):
    """Lorentz γ factor for slow-mo warp."""
    beta = float(np.clip(abs(beta), 0, 0.99))
    return 1.0 / np.sqrt(1.0 - beta ** 2)


# ---- Trefoil dendrite tube -------------------------------------------------

def _trefoil_dendrite_tube(p1, p2, alpha, tube_r=0.012):
    """Parametric tube that morphs from straight line to trefoil-warped
    dendrite as alpha increases.  Connects points p1 → p2."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    length = np.linalg.norm(p2 - p1)

    def func(u, v):
        t = v / _TAU  # [0, 1]
        # Spine: straight line from p1 to p2
        spine = p1 + t * (p2 - p1)

        # Trefoil perturbation perpendicular to the edge
        edge_dir = (p2 - p1) / (length + 1e-12)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(edge_dir, ref)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        n1 = np.cross(edge_dir, ref)
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = np.cross(edge_dir, n1)

        # Trefoil wiggle amplitude grows with alpha
        amp = alpha * length * 0.15
        trefoil_x = amp * np.sin(3 * _TAU * t) * np.sin(_TAU * t)
        trefoil_y = amp * np.sin(3 * _TAU * t) * np.cos(_TAU * t)
        spine = spine + trefoil_x * n1 + trefoil_y * n2

        # Tube cross-section
        T = edge_dir + alpha * 3 * _TAU * amp * np.cos(3 * _TAU * t) * n1
        T = T / (np.linalg.norm(T) + 1e-12)
        N = np.cross(T, ref)
        if np.linalg.norm(N) < 1e-8:
            N = np.cross(T, np.array([1.0, 0.0, 0.0]))
        N = N / (np.linalg.norm(N) + 1e-12)
        B = np.cross(T, N)

        return spine + tube_r * (N * np.cos(u) + B * np.sin(u))

    return func


# ---- Build mobjects --------------------------------------------------------

def _build_boosted_nodes(pts, color=BLUE_D, radius=0.05):
    """VGroup of SDFSurface spheres at given positions."""
    nodes = VGroup()
    for pt in pts:
        s = SDFSurface.sphere(
            radius=radius, color=color, opacity=0.85,
            resolution=(8, 8),
        )
        s.move_to(pt)
        nodes.add(s)
    return nodes


def _build_edges(pts, edges, color=GOLD, thickness=0.008):
    """VGroup of Line3D between pairs."""
    lines = VGroup()
    for i, j in edges:
        lines.add(Line3D(
            start=pts[i], end=pts[j],
            thickness=thickness, color=color,
        ))
    return lines


def _build_dendrite_tubes(pts, edges, alpha, color=PURPLE, tube_r=0.012):
    """VGroup of trefoil-warped dendrite tubes."""
    tubes = VGroup()
    for i, j in edges:
        t = SDFSurface(
            _trefoil_dendrite_tube(pts[i], pts[j], alpha, tube_r),
            u_range=[0, _TAU],
            v_range=[0, _TAU],
            resolution=(8, 24),
            color=color,
            opacity=0.7,
        )
        tubes.add(t)
    return tubes


# ---- Light-cone wireframe --------------------------------------------------

def _build_light_cone(height=3.0, radius=3.0, n_lines=12, color=YELLOW):
    """Wireframe light cone along z-axis."""
    cone = VGroup()
    for k in range(n_lines):
        angle = _TAU * k / n_lines
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # Future cone
        cone.add(Line3D(
            start=[0, 0, 0], end=[x, y, height],
            thickness=0.004, color=color,
        ))
        # Past cone
        cone.add(Line3D(
            start=[0, 0, 0], end=[x, y, -height],
            thickness=0.004, color=color,
        ))
    return cone


# ---- Scene 1: Lorentz Bloom ------------------------------------------------

class LorentzBloom(ThreeDScene):
    """E8 root bloom under relativistic Lorentz boost — α dials β."""

    def construct(self):
        self.set_camera_orientation(phi=72 * DEGREES, theta=-35 * DEGREES)

        title = Text("E8 Lorentz Bloom — Spacetime Warp",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"x'_\parallel = \gamma\, x_\parallel,\quad"
            r"\gamma = (1 - \beta^2)^{-1/2}",
            font_size=20,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Generate E8 roots and project
        roots_8d = _generate_e8_roots()
        pts_base = _project_8d_to_3d(roots_8d)
        pts_sub, _ = _subsample_roots(pts_base, n=50)
        edges = _nearest_edges(pts_sub, k=3)

        # Light cone backdrop
        cone = _build_light_cone(height=3.5, radius=3.5, n_lines=10,
                                  color=YELLOW)
        cone_faded = cone.copy().set_opacity(0.15)
        self.play(FadeIn(cone_faded), run_time=1.5)

        # Initial unboosted state (β=0)
        nodes = _build_boosted_nodes(pts_sub, color=BLUE_D, radius=0.05)
        web = _build_edges(pts_sub, edges, color=GOLD, thickness=0.008)
        self.play(FadeIn(nodes), FadeIn(web), run_time=2)
        self.play(Rotate(VGroup(nodes, web, cone_faded),
                         PI / 4, axis=UP), run_time=1.5)

        # Sweep β through increasing boosts
        betas = [0.3, 0.55, 0.75, 0.88, 0.95]
        boost_colors = [BLUE, TEAL, GREEN, ORANGE, RED]

        for beta, col in zip(betas, boost_colors):
            gamma = _time_dilation_factor(beta)
            boosted_pts = _lorentz_boost_3d(pts_sub, beta)

            new_nodes = _build_boosted_nodes(boosted_pts, color=col,
                                              radius=0.05)
            new_web = _build_edges(boosted_pts, edges, color=col,
                                    thickness=0.006)

            lbl = MathTex(
                rf"\beta = {beta:.2f},\;\gamma = {gamma:.2f}",
                font_size=20,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(lbl)

            # Slow-mo: run_time scales with γ (capped)
            rt = min(2.5, 1.5 * gamma / 3.0 + 1.0)
            self.play(
                Transform(nodes, new_nodes),
                Transform(web, new_web),
                FadeIn(lbl),
                run_time=rt,
            )
            self.play(
                Rotate(VGroup(nodes, web, cone_faded),
                       PI / 3, axis=UP),
                run_time=1.5,
            )
            self.remove(lbl)

        # Final ultra-relativistic rotation
        self.play(
            Rotate(VGroup(nodes, web, cone_faded), TAU, axis=UP),
            run_time=4,
        )
        self.play(
            FadeOut(nodes), FadeOut(web),
            FadeOut(cone_faded), FadeOut(eq), FadeOut(title),
        )
        self.wait(0.5)


# ---- Scene 2: Light-Cone Dendrites ----------------------------------------

class LightConeDendrites(ThreeDScene):
    """Trefoil-knot dendrites stretch across the light cone as α ramps."""

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-40 * DEGREES)

        title = Text("Light-Cone Dendrites — Trefoil Warp",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\alpha\;\text{twist} \times \gamma\;\text{stretch}"
            r"\quad \mu{=}0.8,\; v{=}1.2",
            font_size=20,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # E8 roots — smaller subset for dendrite tubes
        roots_8d = _generate_e8_roots()
        pts_base = _project_8d_to_3d(roots_8d)
        pts_sub, _ = _subsample_roots(pts_base, n=30)
        edges = _nearest_edges(pts_sub, k=3)

        # Light cone
        cone = _build_light_cone(height=3.5, radius=3.5, n_lines=10,
                                  color=YELLOW)
        cone.set_opacity(0.12)
        self.play(FadeIn(cone), run_time=1)

        # Start with straight edges at β=0
        beta_0 = 0.0
        pts_0 = _lorentz_boost_3d(pts_sub, beta_0)
        nodes = _build_boosted_nodes(pts_0, color=BLUE_E, radius=0.04)
        web = _build_edges(pts_0, edges, color=GREY, thickness=0.006)
        self.play(FadeIn(nodes), FadeIn(web), run_time=1.5)

        # Progressive α twist + β boost
        stages = [
            (0.15, 0.3, TEAL),
            (0.35, 0.55, GREEN),
            (0.6, 0.75, ORANGE),
            (0.85, 0.88, RED_E),
            (1.0, 0.95, MAROON),
        ]

        for alpha, beta, col in stages:
            gamma = _time_dilation_factor(beta)
            boosted = _lorentz_boost_3d(pts_sub, beta)

            # Build trefoil dendrite tubes (limited count for perf)
            top_edges = edges[:min(len(edges), 25)]
            dendrites = _build_dendrite_tubes(
                boosted, top_edges, alpha, color=col, tube_r=0.015,
            )
            new_nodes = _build_boosted_nodes(boosted, color=col, radius=0.04)

            lbl = MathTex(
                rf"\alpha={alpha:.2f},\;\beta={beta:.2f},\;"
                rf"\gamma={gamma:.1f}",
                font_size=18,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(lbl)

            self.play(FadeOut(web), run_time=0.3)
            self.play(
                Transform(nodes, new_nodes),
                FadeIn(dendrites),
                FadeIn(lbl),
                run_time=2,
            )
            web = dendrites
            self.play(
                Rotate(VGroup(nodes, web, cone), PI / 3, axis=UP),
                run_time=1.5,
            )
            self.remove(lbl)

        # Final slow-mo warp rotation
        final_lbl = MathTex(
            r"\text{full warp: }\gamma \approx 3.2",
            font_size=20,
        ).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(final_lbl)
        self.play(FadeIn(final_lbl), run_time=0.5)
        self.play(
            Rotate(VGroup(nodes, web, cone), TAU, axis=UP),
            run_time=5,
        )
        self.play(
            FadeOut(nodes), FadeOut(web), FadeOut(cone),
            FadeOut(eq), FadeOut(title), FadeOut(final_lbl),
        )
        self.wait(0.5)
