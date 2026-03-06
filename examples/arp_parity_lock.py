"""TopEquations #6 + #9 -- ARP Reinforce/Decay & Parity Lock with AdaptiveCAD 3D.

    G_dot = alpha * A(phi, G) - mu * (G - G0)
    A     = G * |sin(phi)|

SDF meshes: sphere (phase particle) + torus (oscillator).
Assembly animation and exploded view.

Run:
    manim -pql examples/arp_parity_lock.py ARPReinforcementDecay
    manim -pql examples/arp_parity_lock.py ParityLockMechanism
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
    VGroup,
    Write,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadmanim.mobjects import SDFSurface
from cadmanim.animations import (
    AnimateAssembly,
    AnimateExplodedView,
    MorphBetweenSDFs,
)
from cadmanim.utils import sphere_parametric, torus_parametric


# ---- SDF primitives (kept for documentation) -------------------------------

def sdf_sphere(x, y, z, r=1.0):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2) - r


def sdf_torus(x, y, z, R=1.0, r=0.35):
    q = np.sqrt(x ** 2 + z ** 2) - R
    return np.sqrt(q ** 2 + y ** 2) - r


# ---- Scene 1 ---------------------------------------------------------------

class ARPReinforcementDecay(ThreeDScene):
    """SDF sphere morphing small -> large as coupling G grows.

    The sphere radius represents G: small sphere = weak coupling,
    large sphere = strong coupling after reinforcement.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        eq = MathTex(
            r"\dot{G} = \alpha A(\phi, G) - \mu(G - G_0)", font_size=40,
        ).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #6 -- ARP Reinforce/Decay", font_size=18, color=YELLOW,
        ).next_to(eq, DOWN, buff=0.12)
        self.add_fixed_in_frame_mobjects(eq, ref)
        self.play(Write(eq), FadeIn(ref))

        # Start with small sphere (weak G) — analytic parametric surface
        coupling = SDFSurface.sphere(radius=0.8, color=RED, opacity=0.7)
        weak = Text("G weak", font_size=20, color=RED).to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(weak)
        self.play(FadeIn(coupling), FadeIn(weak), run_time=2)
        self.play(Rotate(coupling, angle=PI / 3, axis=UP), run_time=1.5)

        # Morph to larger sphere (G grows) — parametric interpolation
        strong = Text("G strong -- reinforced!", font_size=20, color=GREEN)
        strong.to_corner(DOWN + LEFT)
        self.add_fixed_in_frame_mobjects(strong)
        self.play(FadeOut(weak))
        self.play(
            MorphBetweenSDFs(
                coupling,
                sphere_parametric(0.8),
                sphere_parametric(2.0),
                keyframes=8,
            ),
            FadeIn(strong),
            run_time=4,
        )

        self.play(Rotate(coupling, angle=TAU / 2, axis=UP), run_time=3)

        # Rotate to show the reinforced shape
        self.play(Rotate(coupling, angle=TAU, axis=UP), run_time=3)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---- Scene 2 ---------------------------------------------------------------

class ParityLockMechanism(ThreeDScene):
    """SDF torus (oscillator) + SDF sphere (phase) assembly.

    AnimateAssembly snaps them together, camera orbits the result,
    then AnimateExplodedView separates the components.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)

        title = VGroup(
            MathTex(r"\dot{\phi} = \Delta - \lambda G \sin\phi", font_size=32),
            MathTex(r"\dot{G} = \alpha G|\sin\phi| - \mu(G-G_0)", font_size=32),
        ).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.3)
        ref = Text(
            "TopEquations #3 + #6 + #9 -- Parity Lock",
            font_size=16, color=YELLOW,
        ).next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(title, ref)
        self.play(Write(title[0]), Write(title[1]), FadeIn(ref))

        # Analytic SDF torus -- the oscillator / coupling
        oscillator = SDFSurface.torus(R=1.0, r=0.35, color=BLUE, opacity=0.6)

        # Analytic SDF sphere -- the phase particle
        phase_ball = SDFSurface.sphere(radius=0.5, color=YELLOW, opacity=0.85)

        assembly = VGroup(oscillator, phase_ball)

        # Assembly animation (parts fly in from spread positions)
        self.play(AnimateAssembly(assembly, spread=4.0), run_time=3)

        osc_lab = Text("Oscillator (torus)", font_size=16, color=BLUE)
        osc_lab.to_corner(DOWN + LEFT)
        phase_lab = Text("Phase (sphere)", font_size=16, color=YELLOW)
        phase_lab.to_corner(DOWN + RIGHT)
        self.add_fixed_in_frame_mobjects(osc_lab, phase_lab)
        self.play(FadeIn(osc_lab), FadeIn(phase_lab))

        # Camera orbit
        self.play(Rotate(assembly, angle=TAU, axis=UP), run_time=5)

        # Exploded view
        self.play(AnimateExplodedView(assembly, scale_factor=3.0), run_time=3)

        insight = MathTex(
            r"\sin\phi \to 0 \;\Rightarrow\; A \to 0 \;\Rightarrow\; G \to G_0",
            font_size=28, color=ORANGE,
        ).to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(insight)
        self.play(Write(insight))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
