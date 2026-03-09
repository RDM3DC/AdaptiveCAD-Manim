"""Kitaev Chain Phase Transition — Trivial to Topological.

The Kitaev chain Hamiltonian:
    H = -μ Σ c†c  -  t Σ (c†c + h.c.)  +  Δ Σ (cc + h.c.)

At μ = 0, |Δ| = t the chain is in the topological phase with
unpaired Majorana modes at the ends.  As μ/t crosses 2, it flips
to trivial.

This animation shows:
  Left:  Spin chain in trivial phase (paired dimers, all gapped)
  Right: Topological phase (inter-site pairing, edge Majoranas)
  Centre: Transition — spins "unwind" into helical modes

Acts
----
1. Title card
2. Draw trivial chain: intra-site pairing (arcs within each site)
3. Animate transition: arcs morph to inter-site pairing
4. Highlight unpaired Majorana edge modes
5. Show helical spin texture via ribbon
6. Summary card

Run
---
    manim -pql examples/kitaev_chain.py KitaevChain
    manim -qh  examples/kitaev_chain.py KitaevChain
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Surface,
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
    Uncreate,
    Transform,
    ReplacementTransform,
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

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Chain layout
# ═══════════════════════════════════════════════════════════════════════════
N_SITES = 6
SITE_SPACING = 1.0
CHAIN_Y = 0.0
CHAIN_X0 = -(N_SITES - 1) * SITE_SPACING / 2
PAIR_ARC_H = 0.5      # height of pairing arcs
HELIX_R = 0.35
HELIX_H_PER_SITE = 0.8


def _site_x(i):
    return CHAIN_X0 + i * SITE_SPACING


def _pairing_arc(x_start, x_end, y_base, arc_h,
                  color=TEAL, above=True):
    """Semicircular arc connecting two x-positions."""
    cx = (x_start + x_end) / 2
    rx = abs(x_end - x_start) / 2
    sign = 1 if above else -1

    def func(t):
        return np.array([
            cx + rx * np.cos(t),
            y_base + sign * arc_h * np.sin(t),
            0.0,
        ])

    return ParametricFunction(
        func, t_range=[0, PI],
        color=color, stroke_width=2.5,
    )


def _majorana_dot(x, y, color, label_text, label_dir=UP):
    """Create a Majorana dot + label."""
    dot = Dot3D(np.array([x, y, 0]), radius=0.10, color=color)
    return dot


def _helical_ribbon(x_start, n_sites, spacing, R, height_per_site,
                     color, phase=0.0, half_w=PI / 3):
    """Helical ribbon along the chain axis (x-direction)."""
    total_len = (n_sites - 1) * spacing
    x0 = x_start

    def func(u, v):
        x = x0 + total_len * u / TAU
        angle = u + phase + (2 * v - 1) * half_w
        return np.array([
            x,
            R * np.cos(angle),
            R * np.sin(angle),
        ])

    return Surface(
        func,
        u_range=[0, TAU],
        v_range=[0, 1],
        resolution=(48, 6),
        fill_color=color, fill_opacity=0.7,
        stroke_width=0.3, stroke_color=color,
    )


class KitaevChain(ThreeDScene):
    """Kitaev chain trivial → topological phase transition."""

    def construct(self):
        self.camera.background_color = "#080818"

        # ─── Act 1  Title ────────────────────────────────────────────────
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        ttl = Text("Kitaev Chain Phase Transition",
                    font_size=44, color=GOLD)
        sub = Text("Trivial → Topological",
                    font_size=24, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"H = -\mu\sum_i c_i^\dagger c_i"
            r" - t\sum_i (c_i^\dagger c_{i+1} + \text{h.c.})"
            r" + \Delta\sum_i (c_i c_{i+1} + \text{h.c.})",
            font_size=18, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Trivial chain — intra-site pairing ──────────────────
        # Backbone
        chain_line = Line3D(
            np.array([_site_x(0) - 0.3, CHAIN_Y, 0]),
            np.array([_site_x(N_SITES - 1) + 0.3, CHAIN_Y, 0]),
            color=GREY_D, thickness=0.008,
        )

        # Site dots — each site has two Majorana modes γ_A, γ_B
        site_dots_A = VGroup()
        site_dots_B = VGroup()
        for i in range(N_SITES):
            x = _site_x(i)
            dA = Dot3D(np.array([x - 0.15, CHAIN_Y, 0]),
                       radius=0.08, color=ORANGE)
            dB = Dot3D(np.array([x + 0.15, CHAIN_Y, 0]),
                       radius=0.08, color=BLUE_D)
            site_dots_A.add(dA)
            site_dots_B.add(dB)

        # Intra-site pairing arcs (trivial phase)
        trivial_arcs = VGroup()
        for i in range(N_SITES):
            x = _site_x(i)
            arc = _pairing_arc(x - 0.15, x + 0.15, CHAIN_Y,
                               PAIR_ARC_H, color=GREY_A, above=True)
            trivial_arcs.add(arc)

        phase_lbl = Text("TRIVIAL", font_size=22, color=GREY_A)
        phase_lbl.move_to(UP * 2.0)
        mu_lbl = MathTex(r"\mu / t > 2", font_size=20, color=GREY_A)
        mu_lbl.move_to(UP * 1.5)
        desc1 = Text("intra-site pairing (gapped)", font_size=14,
                      color=GREY_A)
        desc1.move_to(DOWN * 1.5)

        for lbl in [phase_lbl, mu_lbl, desc1]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            Create(chain_line),
            *[FadeIn(d) for d in site_dots_A],
            *[FadeIn(d) for d in site_dots_B],
            run_time=0.8,
        )
        self.play(
            Create(trivial_arcs),
            FadeIn(phase_lbl), FadeIn(mu_lbl), FadeIn(desc1),
            run_time=1,
        )
        self.wait(0.8)

        # ─── Act 3  Transition: arcs morph to inter-site pairing ────────
        # Inter-site pairing arcs (topological phase)
        topo_arcs = VGroup()
        for i in range(N_SITES - 1):
            x0 = _site_x(i) + 0.15      # γ_B of site i
            x1 = _site_x(i + 1) - 0.15  # γ_A of site i+1
            arc = _pairing_arc(x0, x1, CHAIN_Y, PAIR_ARC_H,
                               color=TEAL, above=True)
            topo_arcs.add(arc)

        new_phase_lbl = Text("TOPOLOGICAL", font_size=22, color=TEAL)
        new_phase_lbl.move_to(UP * 2.0)
        new_mu_lbl = MathTex(r"\mu / t < 2", font_size=20, color=TEAL)
        new_mu_lbl.move_to(UP * 1.5)
        desc2 = Text("inter-site pairing (edge modes!)", font_size=14,
                      color=TEAL)
        desc2.move_to(DOWN * 1.5)

        for lbl in [new_phase_lbl, new_mu_lbl, desc2]:
            self.add_fixed_in_frame_mobjects(lbl)

        # Need to pad trivial_arcs or topo_arcs so Transform works
        # Trivial has N_SITES arcs, topo has N_SITES-1 arcs
        # Add a dummy invisible arc to topo_arcs to match count
        dummy = _pairing_arc(
            _site_x(N_SITES - 1) - 0.15,
            _site_x(N_SITES - 1) + 0.15,
            CHAIN_Y, PAIR_ARC_H * 0.01, color=TEAL,
        )
        dummy.set_opacity(0)
        topo_arcs.add(dummy)

        self.play(
            ReplacementTransform(trivial_arcs, topo_arcs),
            ReplacementTransform(phase_lbl, new_phase_lbl),
            ReplacementTransform(mu_lbl, new_mu_lbl),
            ReplacementTransform(desc1, desc2),
            run_time=2,
        )
        self.wait(0.5)

        # ─── Act 4  Highlight unpaired Majorana edge modes ──────────────
        # Left edge: γ_A of site 0 is unpaired
        # Right edge: γ_B of site N-1 is unpaired
        edge_L = Dot3D(
            np.array([_site_x(0) - 0.15, CHAIN_Y, 0]),
            radius=0.14, color=RED,
        )
        edge_R = Dot3D(
            np.array([_site_x(N_SITES - 1) + 0.15, CHAIN_Y, 0]),
            radius=0.14, color=RED,
        )

        edge_lbl_L = MathTex(r"\gamma_L", font_size=18, color=RED)
        edge_lbl_L.move_to(np.array([_site_x(0) - 0.15,
                                      CHAIN_Y - 0.6, 0]))
        edge_lbl_R = MathTex(r"\gamma_R", font_size=18, color=RED)
        edge_lbl_R.move_to(np.array([_site_x(N_SITES - 1) + 0.15,
                                      CHAIN_Y - 0.6, 0]))

        edge_note = MathTex(
            r"\text{Unpaired Majorana edge modes}",
            font_size=18, color=RED,
        )
        edge_note.to_edge(DOWN, buff=0.3)

        for lbl in [edge_lbl_L, edge_lbl_R, edge_note]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(edge_L), FadeIn(edge_R),
            FadeIn(edge_lbl_L), FadeIn(edge_lbl_R),
            FadeIn(edge_note),
            run_time=0.8,
        )
        self.play(
            Indicate(edge_L, scale_factor=1.5, color=YELLOW),
            Indicate(edge_R, scale_factor=1.5, color=YELLOW),
            run_time=0.8,
        )
        self.wait(0.5)

        # ─── Act 5  Helical spin texture ribbon ─────────────────────────
        self.play(
            FadeOut(topo_arcs), FadeOut(edge_note),
            FadeOut(new_mu_lbl), FadeOut(desc2),
            run_time=0.5,
        )
        self.set_camera_orientation(phi=55 * DEG, theta=-50 * DEG)

        ribbon_a = _helical_ribbon(
            _site_x(0), N_SITES, SITE_SPACING,
            HELIX_R, HELIX_H_PER_SITE,
            ORANGE, phase=0.0,
        )
        ribbon_b = _helical_ribbon(
            _site_x(0), N_SITES, SITE_SPACING,
            HELIX_R, HELIX_H_PER_SITE,
            BLUE_D, phase=PI,
        )

        helix_lbl = MathTex(
            r"\text{Helical mode texture}",
            font_size=18, color=ORANGE,
        )
        helix_lbl.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(helix_lbl)

        self.play(
            Create(ribbon_a), Create(ribbon_b),
            FadeIn(helix_lbl),
            run_time=1.5,
        )

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # ─── Act 6  Summary ──────────────────────────────────────────────
        all_3d = [chain_line, site_dots_A, site_dots_B,
                  edge_L, edge_R, ribbon_a, ribbon_b]
        all_frame = [new_phase_lbl, edge_lbl_L, edge_lbl_R, helix_lbl]

        self.play(
            *[FadeOut(m) for m in all_3d],
            *[FadeOut(lbl) for lbl in all_frame],
            run_time=0.8,
        )
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        card = Text("Kitaev Chain", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"\mu/t > 2:\;\text{trivial — intra-site pairing, gapped}",
                font_size=19,
            ),
            MathTex(
                r"\mu/t < 2:\;\text{topological — inter-site pairing}",
                font_size=19, color=TEAL,
            ),
            MathTex(
                r"\text{Unpaired } \gamma_L, \gamma_R"
                r"\;\text{at chain ends (zero-energy modes)}",
                font_size=19, color=RED,
            ),
            MathTex(
                r"\text{Helical texture: spins unwind into "
                r"topological ribbon}",
                font_size=18, color=ORANGE,
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
