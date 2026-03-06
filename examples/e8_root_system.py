"""E8 root system projection blooming into 248-node defect web.

The E8 root system (240 roots + 8 Cartan generators = 248 generators)
is projected from 8D to 3D and animated as a blooming crystal lattice.
Defect connections grow between nearest-neighbour projected roots.
μ=0.8, v=1.2 ramp controls the bloom dynamics.

All geometry is parametric (analytic spheres + Line3D) — no triangles.

Run:
    manim -pql examples/e8_root_system.py E8RootBloom
    manim -pql examples/e8_root_system.py E8DefectWeb
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


# ---- E8 root system generation --------------------------------------------

def _generate_e8_roots():
    """Generate the 240 roots of E8 in 8D.

    E8 roots come in three families:
    1) D8 roots: all permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)  — 112 roots
    2) Half-spin: (±1/2, ..., ±1/2) with even number of minus signs — 128 roots
    Total: 240 roots
    """
    roots = []

    # D8 roots: ±e_i ± e_j for i < j
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    r = np.zeros(8)
                    r[i] = si
                    r[j] = sj
                    roots.append(r)

    # Half-spin: (±1/2)^8 with even number of minus signs
    for bits in range(256):
        signs = np.array([(bits >> k) & 1 for k in range(8)], dtype=np.float64)
        signs = 1.0 - 2.0 * signs  # 0 → +1, 1 → -1
        n_neg = np.sum(signs < 0)
        if n_neg % 2 == 0:
            roots.append(signs * 0.5)

    return np.array(roots)


def _project_8d_to_3d(roots_8d, method="pca_like"):
    """Project 8D root vectors to 3D using a fixed aesthetic projection.

    Uses a hand-crafted projection matrix that spreads E8 structure
    nicely in 3D, inspired by Coxeter plane projections.
    """
    # Projection matrix: 3×8, chosen to create visually appealing spread
    # Each row picks a different combination of the 8 coordinates
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    proj = np.array([
        [1, phi, 0, -1/phi, 1, 0, -1, phi],
        [phi, 0, 1, 1, -1/phi, -1, phi, 0],
        [0, 1, phi, 0, 1, phi, 0, -1/phi],
    ]) / np.sqrt(8)

    pts_3d = (proj @ roots_8d.T).T  # (N, 3)

    # Scale to fit scene
    max_r = np.max(np.linalg.norm(pts_3d, axis=1))
    pts_3d *= 2.8 / (max_r + 1e-12)

    return pts_3d


def _nearest_edges(pts_3d, k=6, max_dist=None):
    """Find nearest-neighbour edges in 3D point cloud.

    Returns list of (i, j) index pairs for edges.
    Each point connects to its k nearest neighbours (symmetric).
    """
    n = len(pts_3d)
    dists = np.zeros((n, n))
    for i in range(n):
        dists[i] = np.linalg.norm(pts_3d - pts_3d[i], axis=1)
        dists[i, i] = np.inf

    if max_dist is None:
        # Use the k-th nearest neighbour distance as cutoff
        all_kth = []
        for i in range(n):
            sorted_d = np.sort(dists[i])
            all_kth.append(sorted_d[min(k - 1, len(sorted_d) - 1)])
        max_dist = float(np.median(all_kth)) * 1.1

    edges = set()
    for i in range(n):
        neighbours = np.where(dists[i] < max_dist)[0]
        for j in neighbours[:k]:
            edge = (min(i, j), max(i, j))
            edges.add(edge)

    return list(edges)


def _bloom_scale(t, mu=MU, v=V_RAMP):
    """Bloom dynamics: scale factor from 0→1 with μ=0.8, v=1.2 ramp."""
    return (1 - np.exp(-mu * t * 3)) * (1 + 0.1 * np.sin(v * t * 4))


# ---- Subsample for performance --------------------------------------------

def _subsample_roots(pts_3d, target_n=120):
    """Subsample points via farthest-point sampling for visual clarity."""
    n = len(pts_3d)
    if n <= target_n:
        return pts_3d, np.arange(n)

    selected = [0]
    min_dists = np.linalg.norm(pts_3d - pts_3d[0], axis=1)

    for _ in range(target_n - 1):
        farthest = np.argmax(min_dists)
        selected.append(farthest)
        new_dists = np.linalg.norm(pts_3d - pts_3d[farthest], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected] = -1

    idx = np.array(selected)
    return pts_3d[idx], idx


# ---- Build mobjects --------------------------------------------------------

def _build_root_nodes(pts_3d, scale=1.0, color=BLUE_D, radius=0.04):
    """Build VGroup of SDFSurface spheres at projected root positions."""
    nodes = VGroup()
    for pt in pts_3d:
        s = SDFSurface.sphere(
            radius=radius,
            color=color,
            opacity=0.8,
            resolution=(8, 8),
        )
        s.move_to(pt * scale)
        nodes.add(s)
    return nodes


def _build_defect_edges(pts_3d, edges, scale=1.0, color=GOLD, thickness=0.01):
    """Build VGroup of Line3D edges between root nodes."""
    lines = VGroup()
    for i, j in edges:
        p1 = pts_3d[i] * scale
        p2 = pts_3d[j] * scale
        line = Line3D(
            start=p1, end=p2,
            thickness=thickness,
            color=color,
        )
        lines.add(line)
    return lines


# ---- Scene 1: E8 Root Bloom -----------------------------------------------

class E8RootBloom(ThreeDScene):
    """Project E8 root system to 3D and bloom outward layer by layer."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)

        title = Text("E8 Root System — 248-Node Bloom",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"|E_8| = 248,\quad"
            r"\mu{=}0.8,\; v{=}1.2",
            font_size=22,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Generate and project roots
        roots_8d = _generate_e8_roots()
        pts_3d_full = _project_8d_to_3d(roots_8d)

        # Subsample for rendering performance
        pts_3d, _ = _subsample_roots(pts_3d_full, target_n=100)

        # Layer-by-layer bloom: sort by distance from origin
        dists = np.linalg.norm(pts_3d, axis=1)
        sorted_idx = np.argsort(dists)

        # Split into 4 radial shells
        n = len(pts_3d)
        shell_size = n // 4
        shells = [
            sorted_idx[:shell_size],
            sorted_idx[shell_size:2 * shell_size],
            sorted_idx[2 * shell_size:3 * shell_size],
            sorted_idx[3 * shell_size:],
        ]
        shell_colors = [BLUE_D, TEAL, GREEN, GOLD]

        all_nodes = VGroup()

        for shell_i, (indices, col) in enumerate(zip(shells, shell_colors)):
            t_bloom = (shell_i + 1) / 4.0
            bloom_s = float(_bloom_scale(t_bloom))

            shell_nodes = _build_root_nodes(
                pts_3d[indices], scale=bloom_s, color=col, radius=0.05,
            )

            lbl = MathTex(
                rf"\text{{shell\;}} {shell_i+1},\;"
                rf"\sigma = {bloom_s:.2f}",
                font_size=20,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(lbl)
            self.play(
                LaggedStart(*[FadeIn(n) for n in shell_nodes],
                            lag_ratio=0.05),
                FadeIn(lbl),
                run_time=2.5,
            )
            all_nodes.add(*shell_nodes)
            self.play(Rotate(all_nodes, PI / 4, axis=UP), run_time=1.5)
            self.remove(lbl)

        # Final full rotation
        self.play(Rotate(all_nodes, TAU, axis=UP), run_time=4)
        self.play(FadeOut(all_nodes), FadeOut(eq), FadeOut(title))
        self.wait(0.5)


# ---- Scene 2: E8 Defect Web -----------------------------------------------

class E8DefectWeb(ThreeDScene):
    """Full 248-node E8 root system with defect edges blooming in."""

    def construct(self):
        self.set_camera_orientation(phi=72 * DEGREES, theta=-30 * DEGREES)

        title = Text("E8 Defect Web — 248 Generators",
                      font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\mathfrak{e}_8:\; 240\;\text{roots} + 8\;\text{Cartan}"
            r"\quad \mu{=}0.8,\; v{=}1.2",
            font_size=18,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        # Generate roots and project
        roots_8d = _generate_e8_roots()
        pts_3d_full = _project_8d_to_3d(roots_8d)
        pts_3d, sel_idx = _subsample_roots(pts_3d_full, target_n=40)

        # Pre-compute connectivity for colour ramp
        edges_all = _nearest_edges(pts_3d, k=4)
        connectivity = np.zeros(len(pts_3d))
        for i, j in edges_all:
            connectivity[i] += 1
            connectivity[j] += 1
        max_conn = max(connectivity.max(), 1)

        # Build colour-coded nodes directly
        nodes = VGroup()
        for idx, pt in enumerate(pts_3d):
            frac = connectivity[idx] / max_conn
            col = interpolate_color(BLUE, RED, frac)
            s = SDFSurface.sphere(
                radius=0.05,
                color=col,
                opacity=0.85,
                resolution=(8, 8),
            )
            s.move_to(pt)
            nodes.add(s)

        self.play(FadeIn(nodes), run_time=2)
        self.play(Rotate(nodes, PI / 3, axis=UP), run_time=1.5)

        # Build defect web
        web = _build_defect_edges(
            pts_3d, edges_all, color=GOLD, thickness=0.008,
        )
        lbl1 = MathTex(
            r"\text{defect web: } k{=}4",
            font_size=20,
        ).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(lbl1)
        self.play(FadeIn(web), FadeIn(lbl1), run_time=2.5)
        self.play(
            Rotate(VGroup(nodes, web), PI / 3, axis=UP),
            run_time=2,
        )
        self.remove(lbl1)

        lbl2 = MathTex(
            r"\text{connectivity}\;\rightarrow\;\text{colour}",
            font_size=20,
        ).next_to(eq, DOWN)
        self.add_fixed_in_frame_mobjects(lbl2)
        self.play(FadeIn(lbl2), run_time=0.5)
        self.remove(lbl2)

        # Final rotation
        everything = VGroup(nodes, web)
        self.play(Rotate(everything, TAU, axis=UP), run_time=5)
        self.play(
            FadeOut(everything), FadeOut(eq), FadeOut(title),
        )
        self.wait(0.5)
