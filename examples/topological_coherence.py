"""TopEquations #7 -- Topological Coherence with AdaptiveCAD 3D.

    Psi = (1/N_p) * sum_p cos(Theta_p / pi_a)

SDF contour stack morph from gyroid (chaotic, Psi~0) to sphere (ordered, Psi=1).
3D lattice of analytic SDF spheres colored by holonomy — all triangle-free.

Run:
    manim -pql examples/topological_coherence.py CoherenceOrderParameter
    manim -pql examples/topological_coherence.py HolonomyLattice
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
    Create,
    FadeIn,
    FadeOut,
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
from cadmanim.mobjects import SDFSurface, SDFContourStack
from cadmanim.animations import AnimateExplodedView


# ---- SDF primitives --------------------------------------------------------

def sdf_sphere(x, y, z, r=1.0):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2) - r


def sdf_gyroid(x, y, z, scale=3.0, thickness=0.3):
    sx, sy, sz = x * scale, y * scale, z * scale
    return (
        np.abs(
            np.sin(sx) * np.cos(sy)
            + np.sin(sy) * np.cos(sz)
            + np.sin(sz) * np.cos(sx)
        )
        - thickness
    )


# ---- Scene 1 ---------------------------------------------------------------

class CoherenceOrderParameter(ThreeDScene):
    """SDF morph: gyroid (disordered, Psi~0) -> sphere (ordered, Psi=1).

    The geometry transition represents the coherence order parameter
    evolving from chaotic to fully locked.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = MathTex(
            r"\Psi = \frac{1}{N_p}\sum_p \cos\!\left(\frac{\Theta_p}{\pi_a}\right)",
            font_size=38,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #7 -- Topological Coherence",
            font_size=18, color=YELLOW,
        ).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title), FadeIn(ref))

        # Start with gyroid contour stack (disordered state)
        shape = SDFContourStack(
            sdf_gyroid, bounds=(-2, 2), n_slices=28,
            color=RED, opacity=0.6,
        )
        chaos = MathTex(
            r"\Psi \approx 0\;\text{(chaotic)}", font_size=24, color=RED,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(chaos)
        self.play(FadeIn(shape), FadeIn(chaos), run_time=2)
        self.play(Rotate(shape, angle=PI / 3, axis=UP), run_time=2)

        # Morph gyroid -> sphere (ordered) via contour-stack Transform
        order = MathTex(
            r"\Psi \to 1\;\text{(locked)}", font_size=24, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(order)
        self.play(FadeOut(chaos))

        sphere_target = SDFContourStack(
            sdf_sphere, bounds=(-2, 2), n_slices=28,
            color=GREEN, opacity=0.6,
        )
        self.play(
            Transform(shape, sphere_target),
            FadeIn(order),
            run_time=5,
        )

        self.play(Rotate(shape, angle=TAU, axis=UP), run_time=4)
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---- Scene 2 ---------------------------------------------------------------

class HolonomyLattice(ThreeDScene):
    """3D lattice of SDF spheres representing plaquette holonomies.

    Each sphere is colored by cos(Theta_p / pi_a): red (anti-commensurate)
    to green (commensurate). Animates from disorder to order, then
    shows exploded view.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-35 * DEGREES)

        title = MathTex(
            r"\Psi = \frac{1}{N_p}\sum_p \cos\!\left(\frac{\Theta_p}{\pi_a}\right)",
            font_size=34,
        ).to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Build a 3x3x3 lattice of analytic SDF spheres
        Nx, Ny, Nz = 3, 3, 3
        spacing = 1.2
        pi_a = np.pi

        np.random.seed(7)
        base_phases = np.random.uniform(0, 2 * np.pi, (Nx, Ny, Nz))

        # Create one reference sphere, then clone
        ref_sphere = SDFSurface.sphere(
            radius=0.35, color=RED, opacity=0.8,
            resolution=(16, 16),
        )

        spheres = VGroup()
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    s = ref_sphere.copy()
                    pos = np.array([
                        (i - 1) * spacing,
                        (j - 1) * spacing,
                        (k - 1) * spacing,
                    ])
                    s.move_to(pos)
                    spheres.add(s)

        # Color by holonomy (initial: disordered)
        flat_phases = base_phases.flatten()
        for idx, sphere in enumerate(spheres):
            val = np.cos(flat_phases[idx] / pi_a)
            t_col = (val + 1) / 2
            col = interpolate_color(RED, GREEN, t_col)
            sphere.set_fill(col, opacity=0.8)

        self.play(FadeIn(spheres), run_time=2)

        psi_val = np.mean(np.cos(flat_phases / pi_a))
        psi_text = MathTex(
            r"\Psi = " + f"{psi_val:.2f}", font_size=28, color=ORANGE,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(psi_text)
        self.play(FadeIn(psi_text))

        # Animate ordering: phases lock to multiples of pi_a
        for coupling in np.linspace(0, 5, 30):
            pull = 1 - np.exp(-0.8 * coupling)
            nearest = np.round(base_phases / pi_a) * pi_a
            phases = base_phases + pull * (nearest - base_phases)
            flat = phases.flatten()
            for idx, sphere in enumerate(spheres):
                val = np.cos(flat[idx] / pi_a)
                t_col = (val + 1) / 2
                col = interpolate_color(RED, GREEN, t_col)
                sphere.set_fill(col, opacity=0.8)
            self.wait(1 / 15)

        # Update psi readout
        self.remove(psi_text)
        psi_final = MathTex(
            r"\Psi \approx 1.00", font_size=28, color=GREEN,
        ).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(psi_final)
        self.play(FadeIn(psi_final))

        # Camera orbit
        self.play(Rotate(spheres, angle=PI, axis=UP), run_time=3)

        # Exploded view
        self.play(AnimateExplodedView(spheres, scale_factor=2.5), run_time=3)

        insight = Text(
            "All green = locked lattice", font_size=18, color=WHITE,
        ).to_edge(DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(insight)
        self.play(FadeIn(insight))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
