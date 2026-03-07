"""Topological Self-Healing — EGATL QWZ Block Lattice Animation.

Visualises the adaptive topological self-healing protocol:
  1. A 6×6 QWZ lattice with 2-band block-admittance structure
  2. Current flowing from source (top-left) to sink (top-right)
  3. Damage event: top-edge bonds are severed at t=10
  4. Self-healing: the EGATL adaptive law reroutes currents through
     boundary states, recovering topological transport

The animation shows:
  - Lattice nodes coloured by wavefunction amplitude
  - Edges coloured and thickened by current flow
  - Damaged bonds flash red and shrink
  - Live gauges: entropy S, boundary current fraction, transfer efficiency
  - Phase: pre-damage → damage flash → healing → steady state

Run:  manim -pqh examples/topological_self_healing.py TopologicalSelfHealing
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Circle, Line, Arrow, Dot,
    Text, MathTex, DecimalNumber,
    Rectangle, RoundedRectangle, Square,
    FadeIn, FadeOut, Create, Write, Transform,
    ReplacementTransform, AnimationGroup,
    Flash, Indicate, ShowPassingFlash, Circumscribe,
    LEFT, RIGHT, UP, DOWN, ORIGIN, UL, UR, DL, DR,
    BLUE, RED, GREEN, YELLOW, WHITE, GREY, ORANGE, PURPLE,
    GOLD, TEAL, PINK, MAROON,
    rate_functions, config,
)

# Add simulator path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNLOADS = os.path.expanduser("~/Documents/Downloads")
if os.path.isdir(DOWNLOADS):
    sys.path.insert(0, DOWNLOADS)

from hafc_sim2_qwz_block_complete import (
    build_qwz_block_benchmark,
    run_qwz_block_recovery_protocol,
    time_series_effective_transfer,
    time_series_boundary_fraction_block,
    time_series_top_edge_fraction,
    slip_density,
    top_edge_damage_bonds,
    EGATLParams, EntropyParams, RulerParams,
)


# ═══════════════════════════════════════════════════════════════════════════
# Colour utilities
# ═══════════════════════════════════════════════════════════════════════════

def lerp_color_hex(t: float, c0: str, c1: str) -> str:
    """Linearly interpolate between two hex colours."""
    t = max(0.0, min(1.0, t))

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    r0, g0, b0 = hex_to_rgb(c0)
    r1, g1, b1 = hex_to_rgb(c1)
    r = int(r0 + t * (r1 - r0))
    g = int(g0 + t * (g1 - g0))
    b = int(b0 + t * (b1 - b0))
    return f"#{r:02x}{g:02x}{b:02x}"


def current_to_color(I_norm: float, I_max: float) -> str:
    """Map current magnitude to a blue→cyan→yellow→red colour ramp."""
    if I_max < 1e-12:
        return "#1a1a2e"
    t = min(1.0, I_norm / I_max)
    if t < 0.33:
        return lerp_color_hex(t / 0.33, "#1a1a2e", "#00b4d8")
    elif t < 0.66:
        return lerp_color_hex((t - 0.33) / 0.33, "#00b4d8", "#f9c74f")
    else:
        return lerp_color_hex((t - 0.66) / 0.34, "#f9c74f", "#e63946")


def g_to_color(g_abs: float, g_max: float) -> str:
    """Map edge weight |g| to a dark→bright green ramp."""
    if g_max < 1e-12:
        return "#0d1b0d"
    t = min(1.0, g_abs / g_max)
    return lerp_color_hex(t, "#0d1b0d", "#2ecc71")


def phi_to_color(phi_abs: float, phi_max: float) -> str:
    """Map wavefunction amplitude to a dark→blue→white ramp."""
    if phi_max < 1e-12:
        return "#16213e"
    t = min(1.0, phi_abs / phi_max)
    if t < 0.5:
        return lerp_color_hex(t / 0.5, "#16213e", "#0077b6")
    else:
        return lerp_color_hex((t - 0.5) / 0.5, "#0077b6", "#caf0f8")


# ═══════════════════════════════════════════════════════════════════════════
# Data pre-computation
# ═══════════════════════════════════════════════════════════════════════════

def precompute_simulation():
    """Run the QWZ block simulation and extract animation data."""
    bench, out = run_qwz_block_recovery_protocol(
        nx=6, ny=6, T=24.0, dt=0.1, seed=0,
        damage_time=10.0, mass=-0.25,
        phase_mode="lifted", adaptive_pi=True,
    )

    transfer = time_series_effective_transfer(
        out["phi"], bench.source_cell, bench.sink_cell
    )
    boundary = time_series_boundary_fraction_block(
        out["I_norm"], bench.bonds
    )
    topedge = time_series_top_edge_fraction(out["I_norm"], bench)
    damage_idx = set(top_edge_damage_bonds(bench))

    return bench, out, transfer, boundary, topedge, damage_idx


# ═══════════════════════════════════════════════════════════════════════════
# The Scene
# ═══════════════════════════════════════════════════════════════════════════

class TopologicalSelfHealing(Scene):
    """Animate the EGATL topological self-healing protocol on a QWZ lattice."""

    def construct(self):
        # ── Pre-compute simulation ───────────────────────────────────────
        bench, out, transfer, boundary, topedge, damage_idx = precompute_simulation()

        t_arr = out["t"]
        g_hist = out["g"]
        phi_hist = out["phi"]
        I_norm_hist = out["I_norm"]
        S_hist = out["S"]
        pi_hist = out["pi_a"]
        K = len(t_arr)

        # Key frame indices
        damage_k = int(10.0 / 0.1)  # k=100
        pre_k = damage_k - 1
        post_flash_k = damage_k + 5
        heal_mid_k = damage_k + 40
        final_k = K - 1

        # ── Layout constants ─────────────────────────────────────────────
        CELL_SPACING = 0.85
        LATTICE_OFFSET = np.array([-0.5, 0.3, 0])
        NODE_RADIUS = 0.15
        EDGE_WIDTH_MIN = 1.5
        EDGE_WIDTH_MAX = 8.0

        # ── Build lattice mobjects ───────────────────────────────────────
        # Nodes
        node_dots = {}
        node_positions = {}
        for c in range(bench.n_cells):
            gx, gy = bench.cell_xy[c]
            pos = np.array([
                gx * CELL_SPACING - (bench.nx - 1) * CELL_SPACING / 2,
                gy * CELL_SPACING - (bench.ny - 1) * CELL_SPACING / 2,
                0
            ]) + LATTICE_OFFSET
            node_positions[c] = pos
            dot = Circle(radius=NODE_RADIUS, fill_opacity=0.9, stroke_width=1.5)
            dot.move_to(pos)
            dot.set_fill(color="#16213e")
            dot.set_stroke(color=WHITE, width=0.8)
            node_dots[c] = dot

        # Mark source and sink
        src_label = Text("S", font_size=18, color=YELLOW).move_to(
            node_positions[bench.source_cell] + UP * 0.32
        )
        snk_label = Text("D", font_size=18, color=ORANGE).move_to(
            node_positions[bench.sink_cell] + UP * 0.32
        )

        # Edges
        edge_lines = {}
        for e_idx, bond in enumerate(bench.bonds):
            p1 = node_positions[bond.u]
            p2 = node_positions[bond.v]
            line = Line(p1, p2, stroke_width=2.5, color="#1a1a2e")
            edge_lines[e_idx] = line

        # Assemble lattice group
        lattice_edges = VGroup(*[edge_lines[i] for i in range(len(bench.bonds))])
        lattice_nodes = VGroup(*[node_dots[c] for c in range(bench.n_cells)])

        # ── Title and HUD ────────────────────────────────────────────────
        title = Text(
            "Topological Self-Healing", font_size=36, color=WHITE
        ).to_edge(UP, buff=0.3)

        subtitle = Text(
            "EGATL Adaptive Dynamics on QWZ Block Lattice",
            font_size=20, color=GREY
        ).next_to(title, DOWN, buff=0.1)

        # Info panel (right side)
        panel_bg = RoundedRectangle(
            width=3.4, height=5.8, corner_radius=0.15,
            fill_color="#0d1117", fill_opacity=0.85,
            stroke_color=BLUE, stroke_width=1.5,
        ).to_edge(RIGHT, buff=0.3).shift(DOWN * 0.2)

        # Gauge labels
        gauge_y_start = panel_bg.get_top()[1] - 0.4
        gauge_x = panel_bg.get_center()[0]
        gauge_font = 18
        gauge_val_font = 24

        lbl_time = Text("Time", font_size=gauge_font, color=GREY).move_to(
            [gauge_x, gauge_y_start, 0]
        )
        val_time = DecimalNumber(0, num_decimal_places=1, font_size=gauge_val_font,
                                  color=WHITE).next_to(lbl_time, DOWN, buff=0.08)

        lbl_phase = Text("Phase", font_size=gauge_font, color=GREY).next_to(
            val_time, DOWN, buff=0.25
        )
        val_phase = Text("PRE-DAMAGE", font_size=16, color=GREEN).next_to(
            lbl_phase, DOWN, buff=0.08
        )

        lbl_S = Text("Entropy S", font_size=gauge_font, color=GREY).next_to(
            val_phase, DOWN, buff=0.25
        )
        val_S = DecimalNumber(0, num_decimal_places=2, font_size=gauge_val_font,
                               color=TEAL).next_to(lbl_S, DOWN, buff=0.08)

        lbl_bnd = Text("Boundary %", font_size=gauge_font, color=GREY).next_to(
            val_S, DOWN, buff=0.25
        )
        val_bnd = DecimalNumber(0, num_decimal_places=1, font_size=gauge_val_font,
                                 color=BLUE).next_to(lbl_bnd, DOWN, buff=0.08)

        lbl_trn = Text("Transfer", font_size=gauge_font, color=GREY).next_to(
            val_bnd, DOWN, buff=0.25
        )
        val_trn = DecimalNumber(0, num_decimal_places=3, font_size=gauge_val_font,
                                 color=GOLD).next_to(lbl_trn, DOWN, buff=0.08)

        lbl_top = Text("Top Edge %", font_size=gauge_font, color=GREY).next_to(
            val_trn, DOWN, buff=0.25
        )
        val_top = DecimalNumber(0, num_decimal_places=1, font_size=gauge_val_font,
                                 color=PURPLE).next_to(lbl_top, DOWN, buff=0.08)

        hud = VGroup(
            panel_bg,
            lbl_time, val_time,
            lbl_phase, val_phase,
            lbl_S, val_S,
            lbl_bnd, val_bnd,
            lbl_trn, val_trn,
            lbl_top, val_top,
        )

        # ── Legend (bottom) ──────────────────────────────────────────────
        legend_items = VGroup()
        for label, color in [
            ("Low current", "#1a1a2e"),
            ("Medium current", "#00b4d8"),
            ("High current", "#f9c74f"),
            ("Peak current", "#e63946"),
        ]:
            swatch = Square(side_length=0.2, fill_color=color, fill_opacity=1.0,
                           stroke_width=0).shift(LEFT * 0.2)
            txt = Text(label, font_size=14, color=GREY).next_to(swatch, RIGHT, buff=0.1)
            item = VGroup(swatch, txt)
            legend_items.add(item)
        legend_items.arrange(RIGHT, buff=0.4)
        legend_items.to_edge(DOWN, buff=0.25).shift(LEFT * 0.5)

        # ── Helper: update lattice for frame k ───────────────────────────
        def update_lattice(k: int):
            g_k = g_hist[k]
            phi_k = phi_hist[k]
            I_k = I_norm_hist[k]

            I_max = max(I_k.max(), 1e-12)
            g_abs = np.abs(g_k)
            g_max_val = max(g_abs.max(), 1e-12)

            # Update edges
            for e_idx in range(len(bench.bonds)):
                line = edge_lines[e_idx]
                current = I_k[e_idx]
                g_val = g_abs[e_idx]

                # Width: proportional to current
                width = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * min(1.0, current / I_max)
                line.set_stroke(width=width)

                # Colour: by current magnitude
                col = current_to_color(current, I_max)
                line.set_color(col)

                # If bond is nearly dead (damaged), dim it
                if g_val < 0.01:
                    line.set_stroke(width=1.0)
                    line.set_color("#1a0000")
                    line.set_opacity(0.3)
                else:
                    line.set_opacity(1.0)

            # Update nodes — colour by wavefunction amplitude
            phi_amp = np.zeros(bench.n_cells)
            for c in range(bench.n_cells):
                phi_amp[c] = np.linalg.norm(phi_k[2*c:2*c+2])
            phi_max = max(phi_amp.max(), 1e-12)

            for c in range(bench.n_cells):
                col = phi_to_color(phi_amp[c], phi_max)
                node_dots[c].set_fill(color=col)

        # ── SCENE CONSTRUCTION ───────────────────────────────────────────

        # 1. Intro
        self.play(Write(title), FadeIn(subtitle), run_time=1.5)
        self.wait(0.5)

        # 2. Build lattice
        update_lattice(0)
        self.play(
            FadeIn(lattice_edges, lag_ratio=0.02),
            FadeIn(lattice_nodes, lag_ratio=0.02),
            run_time=2.0,
        )
        self.play(FadeIn(src_label), FadeIn(snk_label), run_time=0.5)
        self.play(FadeIn(hud), FadeIn(legend_items), run_time=1.0)

        # 3. Pre-damage evolution (t=0 to t=9.5)
        # Animate several snapshots
        pre_frames = list(range(0, damage_k, 10))  # every 10 steps = 1s sim time
        for k in pre_frames:
            update_lattice(k)
            val_time.set_value(t_arr[k])
            val_S.set_value(S_hist[k])
            val_bnd.set_value(boundary[k] * 100)
            val_trn.set_value(transfer[k])
            val_top.set_value(topedge[k] * 100)
            self.wait(0.15)

        self.wait(0.5)

        # 4. DAMAGE EVENT — dramatic
        # Flash the phase label
        damage_label = Text("⚡ DAMAGE ⚡", font_size=28, color=RED).move_to(
            val_phase.get_center()
        )
        self.play(
            ReplacementTransform(val_phase, damage_label),
            run_time=0.3,
        )
        val_phase = damage_label

        # Flash damaged bonds red
        damaged_lines = [edge_lines[i] for i in damage_idx]
        for line in damaged_lines:
            line.set_color(RED)
            line.set_stroke(width=6)
        self.wait(0.1)

        # Screen flash
        flash_rect = Rectangle(
            width=config.frame_width, height=config.frame_height,
            fill_color=RED, fill_opacity=0.3, stroke_width=0,
        )
        self.play(FadeIn(flash_rect, run_time=0.15))
        self.play(FadeOut(flash_rect, run_time=0.3))

        # Show damage frame
        update_lattice(damage_k)
        val_time.set_value(t_arr[damage_k])
        val_S.set_value(S_hist[damage_k])
        val_bnd.set_value(boundary[damage_k] * 100)
        val_trn.set_value(transfer[damage_k])
        val_top.set_value(topedge[damage_k] * 100)

        # Flash individual damaged bonds
        for line in damaged_lines:
            self.play(Flash(line, color=RED, flash_radius=0.3, run_time=0.15))

        self.wait(0.5)

        # 5. HEALING PHASE (t=10 to t=24)
        healing_label = Text("HEALING...", font_size=16, color=YELLOW).move_to(
            val_phase.get_center()
        )
        self.play(
            ReplacementTransform(val_phase, healing_label),
            run_time=0.3,
        )
        val_phase = healing_label

        # Animate healing — more frames for drama
        heal_frames = list(range(damage_k + 1, final_k + 1, 5))  # every 5 steps
        for k in heal_frames:
            update_lattice(k)
            val_time.set_value(t_arr[k])
            val_S.set_value(S_hist[k])
            val_bnd.set_value(boundary[k] * 100)
            val_trn.set_value(transfer[k])
            val_top.set_value(topedge[k] * 100)
            self.wait(0.12)

        # 6. Final state
        recovered_label = Text("RECOVERED ✓", font_size=16, color=GREEN).move_to(
            val_phase.get_center()
        )
        self.play(
            ReplacementTransform(val_phase, recovered_label),
            run_time=0.5,
        )
        val_phase = recovered_label

        update_lattice(final_k)
        val_time.set_value(t_arr[final_k])
        val_S.set_value(S_hist[final_k])
        val_bnd.set_value(boundary[final_k] * 100)
        val_trn.set_value(transfer[final_k])
        val_top.set_value(topedge[final_k] * 100)

        self.wait(1.0)

        # 7. Summary card
        summary_bg = RoundedRectangle(
            width=8, height=3.5, corner_radius=0.2,
            fill_color="#0d1117", fill_opacity=0.95,
            stroke_color=GREEN, stroke_width=2,
        )
        summary_lines = VGroup(
            Text("Topological Self-Healing Complete", font_size=28, color=GREEN),
            Text(f"Boundary current: {boundary[final_k]*100:.1f}%   "
                 f"(pre-damage: {boundary[pre_k]*100:.1f}%)",
                 font_size=18, color=WHITE),
            Text(f"Transfer efficiency recovered to: {transfer[final_k]:.3f}",
                 font_size=18, color=GOLD),
            Text(f"Top-edge fraction: {topedge[final_k]*100:.1f}%",
                 font_size=18, color=PURPLE),
            Text("Current rerouted through boundary states — topology protects transport",
                 font_size=16, color=TEAL),
        ).arrange(DOWN, buff=0.15)
        summary = VGroup(summary_bg, summary_lines)

        self.play(FadeIn(summary), run_time=1.5)
        self.wait(3.0)
        self.play(FadeOut(summary), run_time=0.5)
        self.wait(1.0)
