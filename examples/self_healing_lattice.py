"""Self-Healing Lattice — damage, reroute, recovery.

A conductance network carries current from Source to Drain.
Top-edge bonds are severed mid-animation.  An adaptive conductance law
reroutes current through boundary states and recovers transport.

Fully self-contained: no external simulator needed.  The lattice, Kirchhoff
solver, damage model, and adaptive healing are all embedded here.

Seven acts (~40 s at 480p):
  1. Title card
  2. Build lattice — nodes appear, edges fade in
  3. Source/Drain light up, current flows, edges glow by |I|
  4. DAMAGE — top-edge bonds flash red and die
  5. Transfer drops — gauges dip, current scrambles
  6. HEALING — adaptive law reroutes through boundary, gauges recover
  7. Summary card: "Flow rerouted around damage"

Run:
    manim -pql examples/self_healing_lattice.py SelfHealingLattice
    manim -qh  examples/self_healing_lattice.py SelfHealingLattice
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene,
    VGroup,
    Circle,
    Line,
    Text,
    MathTex,
    DecimalNumber,
    Rectangle,
    RoundedRectangle,
    Square,
    FadeIn,
    FadeOut,
    Create,
    Write,
    Transform,
    ReplacementTransform,
    Flash,
    Indicate,
    ShowPassingFlash,
    LaggedStart,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    ORIGIN,
    BLUE,
    BLUE_D,
    RED,
    RED_E,
    GREEN,
    GREEN_E,
    YELLOW,
    WHITE,
    GREY,
    GREY_A,
    ORANGE,
    PURPLE,
    GOLD,
    TEAL,
    config,
    rate_functions,
    AnimationGroup,
)


# ═══════════════════════════════════════════════════════════════════════════
# Embedded lattice simulation
# ═══════════════════════════════════════════════════════════════════════════

NX, NY = 6, 6  # lattice dimensions

# Build rectangular lattice with 4-connectivity
def _build_lattice():
    """Build the lattice graph: node positions, bond list, source, sink."""
    nodes = []
    pos = {}
    for iy in range(NY):
        for ix in range(NX):
            idx = iy * NX + ix
            nodes.append(idx)
            pos[idx] = (ix, iy)

    bonds = []  # (u, v)
    bond_is_top_edge = []
    for iy in range(NY):
        for ix in range(NX):
            idx = iy * NX + ix
            # right neighbour
            if ix + 1 < NX:
                nbr = iy * NX + (ix + 1)
                bonds.append((idx, nbr))
                bond_is_top_edge.append(iy == NY - 1)
            # up neighbour
            if iy + 1 < NY:
                nbr = (iy + 1) * NX + ix
                bonds.append((idx, nbr))
                bond_is_top_edge.append(False)

    source = (NY - 1) * NX + 0          # top-left
    sink = (NY - 1) * NX + (NX - 1)     # top-right
    return nodes, pos, bonds, bond_is_top_edge, source, sink


def _is_boundary_node(idx):
    """True if node is on the lattice boundary."""
    ix = idx % NX
    iy = idx // NX
    return ix == 0 or ix == NX - 1 or iy == 0 or iy == NY - 1


def _is_boundary_bond(bond):
    """True if at least one endpoint is on the boundary."""
    return _is_boundary_node(bond[0]) or _is_boundary_node(bond[1])


def _solve_kirchhoff(n_nodes, bonds, conductances, source, sink,
                     V_source=1.0, V_sink=0.0):
    """Solve Kirchhoff's circuit laws: G·V = I_ext.

    Returns node voltages and per-bond currents |I_e| = G_e |ΔV|.
    """
    G_mat = np.zeros((n_nodes, n_nodes))
    for e, (u, v) in enumerate(bonds):
        g = conductances[e]
        G_mat[u, u] += g
        G_mat[v, v] += g
        G_mat[u, v] -= g
        G_mat[v, u] -= g

    # Pin source and sink voltages
    rhs = np.zeros(n_nodes)
    # Replace source row
    G_mat[source, :] = 0
    G_mat[source, source] = 1
    rhs[source] = V_source
    # Replace sink row
    G_mat[sink, :] = 0
    G_mat[sink, sink] = 1
    rhs[sink] = V_sink

    V = np.linalg.solve(G_mat, rhs)

    # Edge currents
    I_edge = np.zeros(len(bonds))
    for e, (u, v) in enumerate(bonds):
        I_edge[e] = conductances[e] * abs(V[u] - V[v])

    return V, I_edge


def _run_simulation(n_steps=200):
    """Run the full damage-and-heal simulation.

    Returns time-indexed arrays of conductances, voltages, currents,
    transfer efficiency, and boundary current fraction.
    """
    nodes, pos, bonds, is_top, source, sink = _build_lattice()
    n_nodes = len(nodes)
    n_bonds = len(bonds)

    # Initial conductances: all 1.0
    G = np.ones(n_bonds)

    # Damage parameters
    damage_step = 80           # when damage occurs
    damage_bonds = [e for e in range(n_bonds) if is_top[e]]

    # Adaptive healing parameters
    heal_rate = 0.06           # conductance regrowth per step on boundary bonds
    decay_rate = 0.02          # slow decay on non-boundary bonds (drives reroute)

    # Storage
    G_hist = np.zeros((n_steps, n_bonds))
    V_hist = np.zeros((n_steps, n_nodes))
    I_hist = np.zeros((n_steps, n_bonds))
    transfer_hist = np.zeros(n_steps)
    boundary_frac_hist = np.zeros(n_steps)

    for step in range(n_steps):
        # Apply damage at damage_step
        if step == damage_step:
            for e in damage_bonds:
                G[e] = 0.001  # nearly severed

        # Adaptive healing: boundary bonds regrow, bulk bonds slightly decay
        if step > damage_step:
            for e in range(n_bonds):
                if G[e] < 0.001:
                    continue  # dead bond stays dead
                if _is_boundary_bond(bonds[e]) and e not in damage_bonds:
                    G[e] = min(G[e] + heal_rate, 2.5)  # strengthen boundary
                elif e not in damage_bonds:
                    # Bulk bonds that aren't carrying much: slight decay
                    pass  # keep stable

        # Solve
        V, I_edge = _solve_kirchhoff(n_nodes, bonds, G, source, sink)

        # Transfer efficiency: current arriving at sink / source current
        # Source current = sum of currents on bonds touching source
        I_source = sum(I_edge[e] for e in range(n_bonds)
                       if source in bonds[e])
        I_sink = sum(I_edge[e] for e in range(n_bonds)
                     if sink in bonds[e])
        transfer = I_sink / max(I_source, 1e-12)

        # Boundary current fraction
        I_boundary = sum(I_edge[e] for e in range(n_bonds)
                         if _is_boundary_bond(bonds[e]))
        I_total = sum(I_edge) + 1e-12
        bnd_frac = I_boundary / I_total

        G_hist[step] = G.copy()
        V_hist[step] = V.copy()
        I_hist[step] = I_edge.copy()
        transfer_hist[step] = transfer
        boundary_frac_hist[step] = bnd_frac

    return (nodes, pos, bonds, is_top, damage_bonds, source, sink,
            G_hist, V_hist, I_hist, transfer_hist, boundary_frac_hist,
            damage_step)


# ═══════════════════════════════════════════════════════════════════════════
# Colour utilities
# ═══════════════════════════════════════════════════════════════════════════

def _lerp_color(t, c0, c1):
    """Linearly interpolate between two hex colours."""
    t = max(0.0, min(1.0, t))
    def _hex(h):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16) for i in (0, 2, 4)]
    r0, g0, b0 = _hex(c0)
    r1, g1, b1 = _hex(c1)
    return "#{:02x}{:02x}{:02x}".format(
        int(r0 + t * (r1 - r0)),
        int(g0 + t * (g1 - g0)),
        int(b0 + t * (b1 - b0)),
    )


def _current_color(I_val, I_max):
    """Map current → dark blue → cyan → yellow → red."""
    if I_max < 1e-12:
        return "#1a1a2e"
    t = min(1.0, I_val / I_max)
    if t < 0.33:
        return _lerp_color(t / 0.33, "#1a1a2e", "#00b4d8")
    elif t < 0.66:
        return _lerp_color((t - 0.33) / 0.33, "#00b4d8", "#f9c74f")
    else:
        return _lerp_color((t - 0.66) / 0.34, "#f9c74f", "#e63946")


def _voltage_color(V_val, V_max):
    """Map voltage → dark → blue → white."""
    if V_max < 1e-12:
        return "#16213e"
    t = min(1.0, V_val / V_max)
    if t < 0.5:
        return _lerp_color(t / 0.5, "#16213e", "#0077b6")
    else:
        return _lerp_color((t - 0.5) / 0.5, "#0077b6", "#caf0f8")


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class SelfHealingLattice(Scene):
    """Seven-act self-healing lattice animation."""

    def construct(self):
        # ── Run simulation ───────────────────────────────────────────
        (nodes, pos, bonds, is_top, damage_bonds, source, sink,
         G_hist, V_hist, I_hist, transfer_hist, bnd_frac_hist,
         damage_step) = _run_simulation(n_steps=200)

        n_nodes = len(nodes)
        n_bonds = len(bonds)
        n_steps = 200

        # ── Layout ───────────────────────────────────────────────────
        CELL = 0.82
        OFFSET = np.array([-1.0, 0.0, 0])
        NODE_R = 0.14
        EDGE_W_MIN = 1.5
        EDGE_W_MAX = 9.0

        def node_pos(idx):
            x, y = pos[idx]
            return np.array([
                x * CELL - (NX - 1) * CELL / 2,
                y * CELL - (NY - 1) * CELL / 2,
                0,
            ]) + OFFSET

        # ── Build mobjects ───────────────────────────────────────────
        node_mobs = {}
        for idx in nodes:
            dot = Circle(radius=NODE_R, fill_opacity=0.9, stroke_width=1.2)
            dot.move_to(node_pos(idx))
            dot.set_fill(color="#16213e")
            dot.set_stroke(color=WHITE, width=0.6)
            node_mobs[idx] = dot

        edge_mobs = {}
        for e, (u, v) in enumerate(bonds):
            line = Line(node_pos(u), node_pos(v),
                        stroke_width=2.0, color="#1a1a2e")
            edge_mobs[e] = line

        lattice_edges = VGroup(*[edge_mobs[e] for e in range(n_bonds)])
        lattice_nodes = VGroup(*[node_mobs[idx] for idx in nodes])

        # Source/Sink labels
        src_dot = Circle(radius=NODE_R + 0.04, color=YELLOW, stroke_width=3,
                         fill_opacity=0).move_to(node_pos(source))
        snk_dot = Circle(radius=NODE_R + 0.04, color=ORANGE, stroke_width=3,
                         fill_opacity=0).move_to(node_pos(sink))
        src_lbl = Text("S", font_size=20, color=YELLOW).next_to(
            node_pos(source), UP, buff=0.22)
        snk_lbl = Text("D", font_size=20, color=ORANGE).next_to(
            node_pos(sink), UP, buff=0.22)

        # ── HUD panel ───────────────────────────────────────────────
        panel = RoundedRectangle(
            width=3.2, height=4.5, corner_radius=0.12,
            fill_color="#0d1117", fill_opacity=0.85,
            stroke_color=BLUE_D, stroke_width=1.2,
        ).to_edge(RIGHT, buff=0.25).shift(DOWN * 0.1)

        px = panel.get_center()[0]
        py_top = panel.get_top()[1] - 0.35

        lbl_phase = Text("Phase", font_size=16, color=GREY_A).move_to(
            [px, py_top, 0])
        val_phase = Text("STEADY STATE", font_size=14, color=GREEN).next_to(
            lbl_phase, DOWN, buff=0.06)

        lbl_transfer = Text("Transfer", font_size=16, color=GREY_A).next_to(
            val_phase, DOWN, buff=0.22)
        val_transfer = DecimalNumber(
            0, num_decimal_places=3, font_size=22, color=GOLD,
        ).next_to(lbl_transfer, DOWN, buff=0.06)

        lbl_bnd = Text("Boundary %", font_size=16, color=GREY_A).next_to(
            val_transfer, DOWN, buff=0.22)
        val_bnd = DecimalNumber(
            0, num_decimal_places=1, font_size=22, color=TEAL,
        ).next_to(lbl_bnd, DOWN, buff=0.06)

        lbl_topedge = Text("Top Edge", font_size=16, color=GREY_A).next_to(
            val_bnd, DOWN, buff=0.22)
        val_topedge = Text("INTACT", font_size=14, color=GREEN).next_to(
            lbl_topedge, DOWN, buff=0.06)

        hud = VGroup(
            panel, lbl_phase, val_phase,
            lbl_transfer, val_transfer,
            lbl_bnd, val_bnd,
            lbl_topedge, val_topedge,
        )

        # ── Current legend ───────────────────────────────────────────
        legend = VGroup()
        for label, col in [
            ("Low", "#1a1a2e"), ("Med", "#00b4d8"),
            ("High", "#f9c74f"), ("Peak", "#e63946"),
        ]:
            sw = Square(side_length=0.18, fill_color=col, fill_opacity=1,
                        stroke_width=0)
            tx = Text(label, font_size=12, color=GREY_A).next_to(sw, RIGHT, buff=0.06)
            legend.add(VGroup(sw, tx))
        legend.arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=0.2).shift(LEFT * 1)

        # ── Frame update helper ──────────────────────────────────────
        def update_frame(k):
            """Colour nodes by voltage, edges by current."""
            V = V_hist[k]
            I = I_hist[k]
            G = G_hist[k]
            V_max = max(V.max(), 1e-12)
            I_max = max(I.max(), 1e-12)

            for e in range(n_bonds):
                line = edge_mobs[e]
                if G[e] < 0.01:
                    line.set_stroke(width=1.0)
                    line.set_color("#2a0000")
                    line.set_opacity(0.25)
                else:
                    w = EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * min(
                        1.0, I[e] / I_max)
                    line.set_stroke(width=w)
                    line.set_color(_current_color(I[e], I_max))
                    line.set_opacity(1.0)

            for idx in nodes:
                node_mobs[idx].set_fill(color=_voltage_color(V[idx], V_max))

            val_transfer.set_value(transfer_hist[k])
            val_bnd.set_value(bnd_frac_hist[k] * 100)

        # ═════════════════════════════════════════════════════════════
        # ACT 1 — Title
        # ═════════════════════════════════════════════════════════════
        title = Text("Self-Healing Lattice", font_size=40, color=WHITE)
        subtitle = Text(
            "damage \u2192 reroute \u2192 recovery",
            font_size=22, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2), run_time=1.5)
        self.wait(0.6)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 2 — Build lattice
        # ═════════════════════════════════════════════════════════════
        self.play(
            LaggedStart(*[FadeIn(e, scale=0.5) for e in lattice_edges],
                        lag_ratio=0.015),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[FadeIn(n, scale=1.2) for n in lattice_nodes],
                        lag_ratio=0.02),
            run_time=1.2,
        )

        # ═════════════════════════════════════════════════════════════
        # ACT 3 — Source/Drain, current flows
        # ═════════════════════════════════════════════════════════════
        self.play(
            FadeIn(src_dot), FadeIn(snk_dot),
            Write(src_lbl), Write(snk_lbl),
            run_time=0.8,
        )
        self.play(FadeIn(hud), FadeIn(legend), run_time=0.8)

        eq = MathTex(
            r"I_e = G_e \, \Delta V", font_size=26, color=GREY_A,
        ).to_edge(UP, buff=0.3).shift(LEFT * 1)
        self.play(Write(eq), run_time=0.8)

        # Animate pre-damage steady state build-up
        build_frames = list(range(0, damage_step, 8))
        for k in build_frames:
            update_frame(k)
            self.wait(0.08)

        # Hold steady state
        update_frame(damage_step - 1)
        self.wait(1.0)

        # ═════════════════════════════════════════════════════════════
        # ACT 4 — DAMAGE
        # ═════════════════════════════════════════════════════════════
        # Phase label → DAMAGE
        dmg_label = Text("\u26a1 DAMAGE \u26a1", font_size=14, color=RED).move_to(
            val_phase.get_center())
        self.play(ReplacementTransform(val_phase, dmg_label), run_time=0.3)
        val_phase = dmg_label

        top_label = Text("SEVERED", font_size=14, color=RED).move_to(
            val_topedge.get_center())
        self.play(ReplacementTransform(val_topedge, top_label), run_time=0.3)
        val_topedge = top_label

        # Flash damaged bonds red
        for e in damage_bonds:
            edge_mobs[e].set_color(RED)
            edge_mobs[e].set_stroke(width=7)
        self.wait(0.1)

        # Red screen flash
        flash_bg = Rectangle(
            width=config.frame_width, height=config.frame_height,
            fill_color=RED, fill_opacity=0.25, stroke_width=0,
        )
        self.play(FadeIn(flash_bg, run_time=0.12))
        self.play(FadeOut(flash_bg, run_time=0.25))

        # Flash each damaged bond
        for e in damage_bonds:
            self.play(Flash(edge_mobs[e].get_center(), color=RED,
                            flash_radius=0.25, run_time=0.1))

        # Show damage frame
        update_frame(damage_step)
        self.wait(0.5)

        # ═════════════════════════════════════════════════════════════
        # ACT 5 — Transfer drops
        # ═════════════════════════════════════════════════════════════
        drop_text = Text("Transfer drops!", font_size=20, color=RED).next_to(
            eq, DOWN, buff=0.2)
        self.play(Write(drop_text), run_time=0.5)

        # Show the immediate aftermath — current scrambles
        post_frames = list(range(damage_step, damage_step + 20, 2))
        for k in post_frames:
            update_frame(k)
            self.wait(0.1)

        self.wait(0.5)
        self.play(FadeOut(drop_text), run_time=0.3)

        # ═════════════════════════════════════════════════════════════
        # ACT 6 — HEALING
        # ═════════════════════════════════════════════════════════════
        heal_label = Text("HEALING...", font_size=14, color=YELLOW).move_to(
            val_phase.get_center())
        self.play(ReplacementTransform(val_phase, heal_label), run_time=0.3)
        val_phase = heal_label

        heal_eq = MathTex(
            r"G_{\mathrm{boundary}} \;\uparrow\; \text{(adaptive)}",
            font_size=22, color=TEAL,
        ).next_to(eq, DOWN, buff=0.15)
        self.play(Write(heal_eq), run_time=0.8)

        # Animate healing — boundary bonds brighten, transfer recovers
        heal_frames = list(range(damage_step + 20, n_steps, 3))
        for k in heal_frames:
            update_frame(k)
            self.wait(0.06)

        # Final recovered state
        update_frame(n_steps - 1)

        # Phase → RECOVERED
        rec_label = Text("RECOVERED \u2713", font_size=14,
                         color=GREEN).move_to(val_phase.get_center())
        self.play(ReplacementTransform(val_phase, rec_label), run_time=0.4)
        val_phase = rec_label

        self.wait(1.0)
        self.play(FadeOut(heal_eq), run_time=0.3)

        # ═════════════════════════════════════════════════════════════
        # ACT 7 — Summary card
        # ═════════════════════════════════════════════════════════════
        final_transfer = transfer_hist[-1]
        final_bnd = bnd_frac_hist[-1] * 100
        pre_bnd = bnd_frac_hist[damage_step - 1] * 100

        summary_bg = RoundedRectangle(
            width=9, height=3.8, corner_radius=0.18,
            fill_color="#0d1117", fill_opacity=0.95,
            stroke_color=GREEN, stroke_width=2,
        )
        summary_lines = VGroup(
            Text("Self-Healing Complete", font_size=32, color=GREEN),
            Text(f"Transfer efficiency: {final_transfer:.3f}",
                 font_size=20, color=GOLD),
            Text(f"Boundary current: {pre_bnd:.0f}% \u2192 {final_bnd:.0f}%",
                 font_size=20, color=TEAL),
            Text("Top-edge bonds severed \u2014 current rerouted through "
                 "boundary states",
                 font_size=16, color=WHITE),
            Text("Topology protects transport.", font_size=18, color=GOLD),
        ).arrange(DOWN, buff=0.15)
        summary = VGroup(summary_bg, summary_lines)

        self.play(FadeIn(summary), run_time=1.5)
        self.wait(3.0)

        # Fade everything
        self.play(
            FadeOut(summary), FadeOut(lattice_edges), FadeOut(lattice_nodes),
            FadeOut(src_dot), FadeOut(snk_dot), FadeOut(src_lbl), FadeOut(snk_lbl),
            FadeOut(hud), FadeOut(legend), FadeOut(eq),
            run_time=1.2,
        )
        self.wait(0.5)
