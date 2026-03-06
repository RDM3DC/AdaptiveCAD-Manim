"""Hierarchical branching lattice from bicrystal seed in strong coupling.

A single seed crystal (sphere) branches level by level into a 3D
lattice tree.  Each branch node is an analytic SDF sphere; connecting
arms are parametric cylinders.  The coupling parameter controls branch
angle and arm thickness.  All geometry is triangle-free.

Run:
    manim -pql examples/branching_lattice.py BicrystalGrowth
    manim -pql examples/branching_lattice.py StrongCouplingLattice
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
    RED,
    RIGHT,
    TAU,
    UP,
    WHITE,
    YELLOW,
    BLUE_D,
    BLUE_E,
    TEAL,
    GOLD,
    MAROON,
    Create,
    Dot3D,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Line3D,
    MathTex,
    Rotate,
    Text,
    ThreeDScene,
    Transform,
    VGroup,
    Write,
    interpolate_color,
    rate_functions,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface
from cadmanim.animations import AnimateExplodedView


# ---- Lattice tree generation -----------------------------------------------

def _rotation_matrix_axis(axis, angle):
    """Rodrigues rotation matrix for rotation about *axis* by *angle*."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def generate_branch_tree(
    levels=3,
    branch_factor=3,
    arm_length=1.0,
    branch_angle=0.6,
    length_decay=0.6,
):
    """Generate hierarchical branching node positions and edges.

    Returns
    -------
    nodes : list of np.ndarray
        3D positions indexed by level then by branch.
    edges : list of (parent_idx, child_idx)
        Pairs of global node indices.
    level_indices : list of list of int
        Global node indices grouped by level.
    """
    nodes = [np.array([0.0, 0.0, 0.0])]
    edges = []
    level_indices = [[0]]
    parent_directions = [np.array([0.0, 1.0, 0.0])]  # initial growth direction

    current_parents = [0]
    current_dirs = [np.array([0.0, 1.0, 0.0])]

    for lv in range(levels):
        length = arm_length * (length_decay ** lv)
        next_parents = []
        next_dirs = []
        level_nodes = []

        for p_idx, p_dir in zip(current_parents, current_dirs):
            p_pos = nodes[p_idx]
            # Choose a perpendicular reference axis for branching
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(p_dir, ref)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0])
            perp = np.cross(p_dir, ref)
            perp = perp / (np.linalg.norm(perp) + 1e-12)

            for b in range(branch_factor):
                phi = TAU * b / branch_factor
                # Rotate the perpendicular around p_dir
                rot_around_dir = _rotation_matrix_axis(p_dir, phi)
                branch_perp = rot_around_dir @ perp
                # Tilt away from parent direction
                rot_tilt = _rotation_matrix_axis(branch_perp, branch_angle)
                child_dir = rot_tilt @ p_dir
                child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-12)

                child_pos = p_pos + child_dir * length
                child_idx = len(nodes)
                nodes.append(child_pos)
                edges.append((p_idx, child_idx))
                level_nodes.append(child_idx)
                next_parents.append(child_idx)
                next_dirs.append(child_dir)

        level_indices.append(level_nodes)
        current_parents = next_parents
        current_dirs = next_dirs

    return nodes, edges, level_indices


# ---- Build Manim VGroup from tree -----------------------------------------

def build_lattice_mobjects(
    nodes,
    edges,
    level_indices,
    node_radius=0.1,
    arm_thickness=0.03,
):
    """Create VGroups of SDFSurface spheres (nodes) and Line3D (arms) by level."""
    level_groups = []

    for lv, indices in enumerate(level_indices):
        grp = VGroup()
        col = interpolate_color(BLUE, ORANGE, lv / max(len(level_indices) - 1, 1))
        for idx in indices:
            pos = nodes[idx]
            # Smaller nodes at deeper levels
            r = node_radius * (0.7 ** lv)
            sphere = SDFSurface.sphere(radius=r, color=col, opacity=0.85, resolution=(12, 12))
            sphere.move_to(pos)
            grp.add(sphere)
        level_groups.append(grp)

    arm_group = VGroup()
    for p_idx, c_idx in edges:
        p, c = nodes[p_idx], nodes[c_idx]
        arm = Line3D(start=p, end=c, thickness=arm_thickness, color=WHITE)
        arm_group.add(arm)

    return level_groups, arm_group


# ---- Scene 1: Growth animation --------------------------------------------

class BicrystalGrowth(ThreeDScene):
    """Animate hierarchical branching from a bicrystal seed."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)

        title = Text("Hierarchical Branching from Bicrystal Seed", font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\mathcal{L}_{n+1} = \mathcal{B}_k \cdot \mathcal{L}_n "
            r"\quad (k=" r"3,\; \ell \to 0.6\,\ell)",
            font_size=24,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        nodes, edges, level_indices = generate_branch_tree(
            levels=3, branch_factor=3, arm_length=1.2,
            branch_angle=0.55, length_decay=0.6,
        )
        level_groups, arm_group = build_lattice_mobjects(
            nodes, edges, level_indices,
            node_radius=0.12, arm_thickness=0.02,
        )

        # Grow level by level
        # Level 0: seed
        seed = level_groups[0]
        self.play(FadeIn(seed, scale=3.0), run_time=1.5)

        # Build edge sub-groups by level
        edge_by_level = []
        offset = 0
        for lv in range(1, len(level_indices)):
            n_edges = len(level_indices[lv])
            edge_sub = VGroup(*arm_group[offset:offset + n_edges])
            edge_by_level.append(edge_sub)
            offset += n_edges

        for lv in range(1, len(level_indices)):
            lv_label = MathTex(
                rf"\text{{Level }} {lv}",
                font_size=22,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(lv_label)

            # Arms first, then nodes
            arms = edge_by_level[lv - 1]
            node_grp = level_groups[lv]

            self.play(
                LaggedStart(*[Create(a) for a in arms], lag_ratio=0.1),
                FadeIn(lv_label),
                run_time=2,
            )
            self.play(
                LaggedStart(*[GrowFromCenter(n) for n in node_grp], lag_ratio=0.1),
                run_time=1.5,
            )
            self.remove(lv_label)

        # Rotate full lattice
        full = VGroup(*level_groups, arm_group)
        self.play(Rotate(full, angle=TAU, axis=UP), run_time=4)
        self.play(FadeOut(full), FadeOut(eq), FadeOut(title))
        self.wait(0.5)


# ---- Scene 2: Strong coupling parameter sweep -----------------------------

class StrongCouplingLattice(ThreeDScene):
    """Sweep the coupling (branch angle) from weak to strong."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        title = Text("Branching Lattice — Strong Coupling Sweep", font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        eq = MathTex(
            r"\theta_{\mathrm{branch}} \;\uparrow\; "
            r"\Rightarrow \;\text{wider lattice}",
            font_size=24,
        ).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(FadeIn(eq))

        coupling_angles = [0.2, 0.45, 0.7, 1.0, 1.3]
        colors = [BLUE_E, BLUE, TEAL, GREEN, GOLD]

        # Build first lattice
        nodes, edges, level_indices = generate_branch_tree(
            levels=3, branch_factor=3, arm_length=1.0,
            branch_angle=coupling_angles[0], length_decay=0.55,
        )
        level_groups, arm_group = build_lattice_mobjects(
            nodes, edges, level_indices, node_radius=0.08,
        )
        lattice = VGroup(*level_groups, arm_group)
        self.play(FadeIn(lattice), run_time=2)
        self.play(Rotate(lattice, angle=PI / 3, axis=UP), run_time=1.5)

        for angle, col in zip(coupling_angles[1:], colors[1:]):
            n2, e2, li2 = generate_branch_tree(
                levels=3, branch_factor=3, arm_length=1.0,
                branch_angle=angle, length_decay=0.55,
            )
            lg2, ag2 = build_lattice_mobjects(n2, e2, li2, node_radius=0.08)
            new_lattice = VGroup(*lg2, ag2)

            angle_label = MathTex(
                rf"\theta = {angle:.1f}\;\mathrm{{rad}}",
                font_size=22,
            ).next_to(eq, DOWN)
            self.add_fixed_in_frame_mobjects(angle_label)
            self.play(
                Transform(lattice, new_lattice),
                FadeIn(angle_label),
                run_time=2.5,
            )
            self.play(Rotate(lattice, angle=PI / 3, axis=UP), run_time=1.5)
            self.remove(angle_label)

        # Final exploded view
        self.play(Rotate(lattice, angle=PI, axis=UP), run_time=2)
        self.play(FadeOut(lattice), FadeOut(eq), FadeOut(title))
        self.wait(0.5)
