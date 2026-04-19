"""HAFC — Active π_f Graph Simulator with Canonical #1 Law.

Implements the **Surprise-Weighted Frustration-Healed BZ Conductance Law**
(canonical #1 in the TopEquations registry) on a simple resistive graph.

Canonical equation
------------------

    dG̃_ij/dt = α_G(S) * (1 + κ_Ψ Ψ) / (1 + ξ M_ij^(U)) * |I_ij| * e^{iθ_R,ij}
             - μ_G(S) * (1 - ζ χ_ij^edge) * G̃_ij

Supporting state equations
--------------------------

    τ_M dM_ij^(U)/dt = U_ij * Σ_ij^slip − M_ij^(U)
    Σ_ij^slip = sin²(θ_R,ij / 2π_a)

    χ_ij^edge = Σ_{p∋(i,j)} b_p |ρ_p| / (Σ_{p∋(i,j)} |ρ_p| + ε)

Run
---
    python hafc_sim_pif_active.py               # single graph, full dashboard
    python hafc_sim_pif_active.py --ablation    # three-way ablation comparison
    python hafc_sim_pif_active.py --no-plot     # headless / benchmark only
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise an array to [0, 1]; returns zeros if all-constant."""
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span < eps:
        return np.zeros_like(v, dtype=float)
    return (v.astype(float) - lo) / span


def wrap_to_pi(angle: float) -> float:
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2 * math.pi
    return a


# ------------------------------------------------------------------ #
#  Data-classes                                                        #
# ------------------------------------------------------------------ #

@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]

    def incidence(self) -> np.ndarray:
        m = len(self.edges)
        B = np.zeros((m, self.n), dtype=float)
        for e, (u, v) in enumerate(self.edges):
            B[e, u] = 1.0
            B[e, v] = -1.0
        return B


@dataclass
class ARPParams:
    alpha_G: float = 1.0
    mu_G: float = 0.4
    G_min: float = 1e-3
    G_budget: Optional[float] = None


@dataclass
class PiAParams:
    pi0: float = math.pi
    alpha_pi: float = 0.6
    mu_pi: float = 0.25
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


@dataclass
class PiFParams:
    """Controls for the active π_f edge-coupling layer.

    enabled         – activates the π_f transport-side gain.
    edge_coupling   – weight of η_edge boost on the ARP numerator.

    # Canonical #1 parameters (Surprise-Weighted Frustration-Healed law)
    kappa_psi       – global loop-coherence numerator gain.
    xi_memory       – denominator weight for surprise-weighted memory M^(U).
    tau_memory      – EMA time constant for M^(U).
    zeta_edge       – passive decay-attenuation via χ_edge.
    slip_floor      – minimum slip signal (numerical guard).
    memory_clip     – upper bound on M^(U) (prevents runaway in constant-surprise regime).
    surprise_blend  – convex blend weight for U_edge (loop-mismatch vs η-gradient).
    """
    enabled: bool = True
    edge_coupling: float = 0.50

    # Surprise-weighted #1 controls
    kappa_psi: float = 0.75
    xi_memory: float = 1.20
    tau_memory: float = 1.50
    zeta_edge: float = 0.45
    slip_floor: float = 1e-9
    memory_clip: float = 10.0
    surprise_blend: float = 0.50


@dataclass
class PhaseLiftState:
    theta_prev: float
    theta_R: float
    theta_R0: float
    w: int
    b: int  # +/-1

    @staticmethod
    def init_from_theta(theta0: float) -> "PhaseLiftState":
        return PhaseLiftState(
            theta_prev=theta0, theta_R=theta0, theta_R0=theta0, w=0, b=1
        )


# ------------------------------------------------------------------ #
#  Phase-lift                                                          #
# ------------------------------------------------------------------ #

def phase_lift_step(
    st: PhaseLiftState, theta: float, pi_a: float,
) -> Tuple[PhaseLiftState, float, float, int, int, int]:
    r = wrap_to_pi(theta - st.theta_prev)
    r_clip = float(np.clip(r, -pi_a, +pi_a))
    theta_R = st.theta_R + r_clip
    w = int(np.round((theta_R - st.theta_R0) / (2 * math.pi)))
    b = 1 if (w % 2 == 0) else -1
    flip = 1 if (b != st.b) else 0
    st2 = PhaseLiftState(theta_prev=theta, theta_R=theta_R,
                         theta_R0=st.theta_R0, w=w, b=b)
    return st2, r, r_clip, w, b, flip


# ------------------------------------------------------------------ #
#  Resistive flow                                                      #
# ------------------------------------------------------------------ #

def solve_resistive_flows(
    graph: Graph, G: np.ndarray, source: int, sink: int, I_in: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = graph.n
    B = graph.incidence()
    m = B.shape[0]
    L = B.T @ (G.reshape(m, 1) * B)
    b_vec = np.zeros(n)
    b_vec[source] = I_in
    b_vec[sink] = -I_in
    mask = np.ones(n, dtype=bool)
    mask[sink] = False
    phi = np.zeros(n)
    phi[mask] = np.linalg.solve(L[mask][:, mask], b_vec[mask])
    I = np.array([G[e] * (phi[u] - phi[v])
                  for e, (u, v) in enumerate(graph.edges)])
    return phi, I


# ------------------------------------------------------------------ #
#  Loop-signature helpers                                              #
# ------------------------------------------------------------------ #

def build_cycle_basis(graph: Graph) -> np.ndarray:
    """Return a cycle-incidence matrix C of shape (n_loops, m).

    Uses a spanning-tree approach: edges not in the spanning tree each
    define one fundamental cycle.  Returns float array with ±1/0 entries.
    """
    n, m = graph.n, len(graph.edges)
    edge_index = {e: i for i, e in enumerate(graph.edges)}
    edge_index.update({(v, u): i for i, (u, v) in enumerate(graph.edges)})

    # BFS spanning tree
    parent = [-1] * n
    parent_edge = [-1] * n
    parent_dir = [0] * n
    visited = [False] * n
    tree_edges = set()
    queue = [0]
    visited[0] = True
    while queue:
        u = queue.pop(0)
        for i, (a, b) in enumerate(graph.edges):
            if a == u and not visited[b]:
                visited[b] = True; parent[b] = u; parent_edge[b] = i
                parent_dir[b] = +1; tree_edges.add(i); queue.append(b)
            elif b == u and not visited[a]:
                visited[a] = True; parent[a] = u; parent_edge[a] = i
                parent_dir[a] = -1; tree_edges.add(i); queue.append(a)

    # For each non-tree edge build its fundamental cycle
    cycles = []
    for chord_idx in range(m):
        if chord_idx in tree_edges:
            continue
        u0, v0 = graph.edges[chord_idx]
        # Find path v0 -> u0 in spanning tree
        row = np.zeros(m, dtype=float)
        row[chord_idx] = +1.0  # chord traversed u0->v0
        # trace v0 back to root
        path_v = []
        node = v0
        while parent[node] != -1:
            path_v.append((parent_edge[node], parent_dir[node], node, parent[node]))
            node = parent[node]
        root = node
        # trace u0 back to root
        path_u = []
        node = u0
        while parent[node] != -1:
            path_u.append((parent_edge[node], parent_dir[node], node, parent[node]))
            node = parent[node]
        # find LCA
        anc_v = {n for _, _, n, _ in path_v}
        anc_v.add(root)
        lca = u0
        for _, _, nd, _ in [(None, None, u0, None)] + path_u:
            if nd in anc_v:
                lca = nd; break
        # edges on path v0->lca (forward in cycle = towards lca)
        for ei, d, _frm, _to in path_v:
            u_e, v_e = graph.edges[ei]
            # direction in cycle: we travel from v0 toward lca
            if _frm == v_e:   # tree edge points u_e->v_e, we go v_e->u_e => -1
                row[ei] += -d  # d was parent_dir[_frm]
            else:
                row[ei] += +d
        # edges on path u0->lca (backward in cycle)
        for ei, d, _frm, _to in path_u:
            u_e, v_e = graph.edges[ei]
            if _frm == u_e:
                row[ei] += +d
            else:
                row[ei] += -d
        cycles.append(row)

    if not cycles:
        return np.zeros((0, m), dtype=float)
    return np.array(cycles, dtype=float)


def loop_signatures(
    C: np.ndarray, G: np.ndarray, theta_R_edge: np.ndarray, pi_a: float,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mismatch, sig) for each loop.

    mismatch[i] = |sum_j C[i,j] * theta_R_edge[j]| / (|C[i,j]| sum + eps)
    sig[i]      = cos(pi * mismatch[i])  in [-1, 1]
    """
    if C.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    weighted = C @ theta_R_edge
    norm = np.abs(C) @ np.ones(C.shape[1]) + eps
    mismatch = np.abs(weighted) / norm
    sig = np.cos(math.pi * mismatch)
    return mismatch, sig


# ------------------------------------------------------------------ #
#  Canonical #1 helper functions                                       #
# ------------------------------------------------------------------ #

def edge_loop_coherence(
    C_abs: np.ndarray, mismatch: np.ndarray, eps: float = 1e-12,
) -> np.ndarray:
    """Edge-local loop coherence proxy in [0, 1].

    An edge sees high coherence when the loops touching it have low mismatch.
    """
    if C_abs.size == 0:
        return np.zeros(0, dtype=float)
    edge_counts = np.maximum(eps, C_abs.sum(axis=0))
    local_mismatch = (C_abs.T @ mismatch) / edge_counts
    scale = float(np.max(local_mismatch))
    if scale <= eps:
        return np.ones_like(local_mismatch, dtype=float)
    return np.clip(1.0 - local_mismatch / scale, 0.0, 1.0)


def edge_boundary_gate(graph: Graph, C_abs: np.ndarray) -> np.ndarray:
    """Edge-local boundary enrichment gate χ_edge in [0, 1].

    Low-degree nodes are a boundary proxy on graph modes; flux-participation
    blends with the degree score at 50/50 when loops exist.
    """
    m = len(graph.edges)
    deg = np.zeros(graph.n, dtype=float)
    for u, v in graph.edges:
        deg[u] += 1.0
        deg[v] += 1.0

    gate = np.zeros(m, dtype=float)
    for ei, (u, v) in enumerate(graph.edges):
        score_u = 1.0 / max(1.0, deg[u])
        score_v = 1.0 / max(1.0, deg[v])
        gate[ei] = max(score_u, score_v)

    if C_abs.size:
        loop_participation = np.clip(C_abs.sum(axis=0), 0.0, None)
        if float(loop_participation.max()) > 0:
            loop_participation = loop_participation / float(loop_participation.max())
            gate = 0.5 * gate + 0.5 * loop_participation

    gate = gate / max(1e-12, float(gate.max()))
    return np.clip(gate, 0.0, 1.0)


def surprise_signal(
    edge_fb: np.ndarray, eta_edge: np.ndarray, blend: float = 0.5,
) -> np.ndarray:
    """Bounded edge surprise proxy U_ij in [0, 1].

    Convex blend of loop-mismatch feedback and local η-gradient magnitude.
    """
    edge_part = np.abs(_normalize(edge_fb))
    eta_part = np.abs(_normalize(eta_edge))
    out = blend * edge_part + (1.0 - blend) * eta_part
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------ #
#  Main simulation: simulate_graph_active_pif                          #
# ------------------------------------------------------------------ #

def simulate_graph_active_pif(
    graph: Graph,
    source: int,
    sink: int,
    T: float = 60.0,
    dt: float = 0.05,
    seed: int = 7,
    arp: ARPParams = ARPParams(),
    pia: PiAParams = PiAParams(),
    pif: PiFParams = PiFParams(),
    omega: float = 0.65,
    noise_sigma: float = 0.10,
    kappa_flip: float = 0.35,
    damage_time: Optional[float] = None,
    damage_edges: Optional[List[int]] = None,
    damage_factor: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Run the graph under the canonical #1 (Surprise-Weighted Frustration-Healed) law.

    Parameters
    ----------
    graph           : Graph topology.
    source / sink   : Boundary nodes for current injection.
    T / dt          : Total time and step.
    seed            : RNG seed.
    arp             : Base ARP parameters (alpha_G, mu_G, G_min, G_budget).
    pia             : Adaptive pi_a parameters.
    pif             : PiFParams — enables surprise + edge-gate controls.
    omega           : Oscillation frequency for the scalar phase signal.
    noise_sigma     : Phase noise amplitude.
    kappa_flip      : Entropy coupling coefficient.
    damage_time     : If set, zero out damage_edges conductances at this time.
    damage_edges    : List of edge indices to damage (defaults to all).
    damage_factor   : Fractional residual conductance after damage.
    """
    rng = np.random.default_rng(seed)
    m = len(graph.edges)
    K = int(np.ceil(T / dt)) + 1
    t_arr = np.linspace(0.0, T, K)

    # Cycle basis for loop-signature computation
    C = build_cycle_basis(graph)
    C_abs = np.abs(C)
    n_loops = C.shape[0]

    # State
    G = np.ones(m, dtype=float)
    eta_edge = np.zeros(m, dtype=float)
    pi_a = pia.pi0
    st = PhaseLiftState.init_from_theta(0.0)

    # Canonical #1 state
    M_u = np.zeros(m, dtype=float)
    chi_edge = edge_boundary_gate(graph, C_abs)

    # Histories
    G_hist         = np.zeros((K, m))
    I_hist         = np.zeros((K, m))
    phi_hist       = np.zeros((K, graph.n))
    pi_hist        = np.zeros(K)
    theta_hist     = np.zeros(K)
    thetaR_hist    = np.zeros(K)
    w_hist         = np.zeros(K, dtype=int)
    b_hist         = np.zeros(K, dtype=int)
    flip_hist      = np.zeros(K, dtype=int)
    S_hist         = np.zeros(K)
    eta_hist       = np.zeros((K, m))
    M_u_hist       = np.zeros((K, m))
    chi_edge_hist  = np.zeros((K, m))
    U_hist         = np.zeros((K, m))
    Psi_hist       = np.zeros(K)
    sig_hist       = np.zeros((K, max(n_loops, 1)))
    mismatch_hist  = np.zeros((K, max(n_loops, 1)))
    edge_fb_hist   = np.zeros((K, m))

    damaged = False

    for k in range(K):
        tk = t_arr[k]

        # Damage event
        if (damage_time is not None) and (not damaged) and (tk >= damage_time):
            idxs = damage_edges if damage_edges is not None else list(range(m))
            G[idxs] *= damage_factor
            damaged = True

        # Flow solve
        phi, I = solve_resistive_flows(graph, G, source, sink)

        # Scalar phase signal
        activity = float(np.mean(np.abs(I)))
        z = (1.0 + 0.25 * activity) * np.exp(1j * omega * tk) + \
            noise_sigma * (rng.normal() + 1j * rng.normal())
        theta_scalar = float(np.angle(z))
        st, _r, _rclip, w, b, flip = phase_lift_step(st, theta_scalar, pi_a)
        S = activity + kappa_flip * flip

        # Per-edge θ_R proxy (uniform in graph mode: all edges share the global lift)
        theta_R_edge = np.full(m, st.theta_R, dtype=float)

        # Loop signatures
        mismatch = np.zeros(0)
        sig = np.zeros(0)
        if n_loops:
            mismatch, sig = loop_signatures(C, G, theta_R_edge, pi_a)

        # Active π_f edge update (η-field, edge feedback)
        abs_mismatch = np.abs(mismatch) if n_loops else np.zeros(0, dtype=float)
        if n_loops:
            # edge_fb: feedback signal from loop mismatches to their edges
            edge_fb = C_abs.T @ mismatch
            edge_fb = edge_fb / max(1e-12, float(np.max(np.abs(edge_fb))))
        else:
            edge_fb = np.zeros(m, dtype=float)

        if pif.enabled:
            # η updates: edge feedback drives η toward reducing mismatch
            d_eta = pif.edge_coupling * edge_fb - 0.5 * eta_edge
            eta_edge = eta_edge + dt * d_eta

        # ─────────────────────────────────────────────────────────────
        #  Canonical #1: surprise-weighted frustration-healed law
        # ─────────────────────────────────────────────────────────────
        Psi_edge = (edge_loop_coherence(C_abs, abs_mismatch)
                    if n_loops else np.ones(m, dtype=float))
        Psi = float(np.mean(Psi_edge)) if Psi_edge.size else 1.0

        U_edge = surprise_signal(edge_fb, eta_edge, blend=pif.surprise_blend)

        # Surprise-weighted slip memory state: τ_M dM^(U)/dt = U * Σ^slip − M^(U)
        theta_R_scalar = st.theta_R
        slip_scalar = math.sin(theta_R_scalar / max(2.0 * pi_a, 1e-9)) ** 2
        slip_vec = np.full(m, max(pif.slip_floor, slip_scalar), dtype=float)

        dM = (U_edge * slip_vec - M_u) / max(pif.tau_memory, 1e-9)
        M_u = np.clip(M_u + dt * dM, 0.0, pif.memory_clip)

        # Reinforcement (numerator / denominator from canonical #1)
        reinforce_num = 1.0 + pif.kappa_psi * Psi
        reinforce_den = 1.0 + pif.xi_memory * M_u
        reinforce = arp.alpha_G * reinforce_num / reinforce_den

        # Transport-side gain from active π_f (retain as in predecessor law)
        boost = 1.0 + (pif.edge_coupling * eta_edge if pif.enabled else 0.0)
        boost = np.clip(boost, 0.2, 3.0)

        # Passive edge protection via χ_edge
        decay_gate = np.clip(1.0 - pif.zeta_edge * chi_edge, 0.0, 1.0)

        I_eff = I + 1j * (eta_edge * np.abs(I))
        dG = reinforce * np.abs(I_eff) * boost - arp.mu_G * decay_gate * G
        G = np.maximum(arp.G_min, G + dt * dG)

        if arp.G_budget is not None:
            total = float(np.sum(G))
            if total > arp.G_budget > 0:
                G *= arp.G_budget / total

        # Adaptive pi_a
        dpi = pia.alpha_pi * S - pia.mu_pi * (pi_a - pia.pi0)
        pi_a = float(np.clip(pi_a + dt * dpi, pia.pi_min, pia.pi_max))

        # Store
        G_hist[k] = G; I_hist[k] = I; phi_hist[k] = phi
        pi_hist[k] = pi_a; theta_hist[k] = theta_scalar
        thetaR_hist[k] = st.theta_R; w_hist[k] = w
        b_hist[k] = b; flip_hist[k] = flip; S_hist[k] = S
        eta_hist[k] = eta_edge; edge_fb_hist[k] = edge_fb
        M_u_hist[k] = M_u
        chi_edge_hist[k] = chi_edge
        U_hist[k] = U_edge
        Psi_hist[k] = Psi
        if n_loops:
            sig_hist[k, :n_loops] = sig
            mismatch_hist[k, :n_loops] = mismatch

    rb = np.zeros(K)
    if K > 1:
        rb[1:] = np.cumsum(flip_hist[1:]) / np.arange(1, K)

    return dict(
        t=t_arr,
        G=G_hist, I=I_hist, phi=phi_hist,
        pi_a=pi_hist, theta=theta_hist, theta_R=thetaR_hist,
        w=w_hist, b=b_hist, flip=flip_hist, S=S_hist, r_b=rb,
        eta=eta_hist, edge_fb=edge_fb_hist,
        sig=sig_hist, mismatch=mismatch_hist,
        edges=np.array(graph.edges, dtype=int),
        # Canonical #1 diagnostics
        M_u=M_u_hist,
        chi_edge=chi_edge_hist,
        U_edge=U_hist,
        Psi=Psi_hist,
    )


# ------------------------------------------------------------------ #
#  Summarise                                                           #
# ------------------------------------------------------------------ #

def summarize_active_pif(
    out: Dict[str, np.ndarray],
    damage_time: Optional[float] = None,
    tail: int = 20,
) -> Dict[str, float]:
    """Compute scalar summary metrics from a simulation output dict."""
    t = out["t"]
    K = len(t)
    dt = float(t[1] - t[0]) if K > 1 else 1.0

    dmg_k = 0
    if damage_time is not None:
        dmg_k = int(damage_time / dt)

    pre_k = max(0, dmg_k - 1)
    post_k = min(K - 1, dmg_k + tail)

    G = out["G"]
    I = out["I"]
    M_u = out["M_u"]
    chi_edge = out["chi_edge"]
    U_edge = out["U_edge"]
    Psi = out["Psi"]

    mean_G_pre   = float(np.mean(G[pre_k]))
    mean_G_post  = float(np.mean(G[post_k]))
    mean_G_final = float(np.mean(G[-1]))

    mismatch = out["mismatch"]
    mismatch_pre  = float(np.mean(mismatch[pre_k]))  if mismatch.shape[1] > 0 else 0.0
    mismatch_tail = float(np.mean(mismatch[-tail:])) if mismatch.shape[1] > 0 else 0.0

    recovery_ratio = (mean_G_final / mean_G_pre) if mean_G_pre > 1e-9 else float("nan")

    return dict(
        mean_G_pre=mean_G_pre,
        mean_G_post=mean_G_post,
        mean_G_final=mean_G_final,
        recovery_ratio=recovery_ratio,
        mismatch_pre=mismatch_pre,
        mismatch_tail=mismatch_tail,
        mean_memory_final=float(np.mean(M_u[-1])),
        mean_edge_gate_final=float(np.mean(chi_edge[-1])),
        mean_surprise_final=float(np.mean(U_edge[-1])),
        mean_Psi_final=float(np.mean(Psi[-tail:])),
        final_pi_a=float(out["pi_a"][-1]),
        final_S=float(out["S"][-1]),
    )


# ------------------------------------------------------------------ #
#  Graphs                                                              #
# ------------------------------------------------------------------ #

def default_toy_graph() -> Tuple[Graph, int, int]:
    edges = [(0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5), (1, 3), (2, 4)]
    return Graph(n=6, edges=edges), 0, 5


def ladder_graph(n: int = 8) -> Tuple[Graph, int, int]:
    """Two parallel paths of length n with rungs."""
    top = list(range(n))
    bot = list(range(n, 2 * n))
    edges = ([(top[i], top[i + 1]) for i in range(n - 1)] +
             [(bot[i], bot[i + 1]) for i in range(n - 1)] +
             [(top[i], bot[i]) for i in range(n)])
    return Graph(n=2 * n, edges=edges), 0, n - 1


# ------------------------------------------------------------------ #
#  Dashboard                                                           #
# ------------------------------------------------------------------ #

def dashboard_active_pif(
    out: Dict[str, np.ndarray],
    title: str = "HAFC — Active π_f (Canonical #1)",
    damage_time: Optional[float] = None,
    save_path: str = "hafc_pif_active_dashboard.png",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = out["t"]
    G = out["G"]
    edges = out["edges"]
    m = G.shape[1]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    dmg_kw = dict(color="red", ls="--", alpha=0.5, lw=1.2)

    def vline(ax):
        if damage_time is not None:
            ax.axvline(damage_time, **dmg_kw, label="damage")

    # Row 0 — Conductances
    ax = axes[0, 0]
    for e in range(min(m, 12)):
        ax.plot(t, G[:, e], lw=0.9, alpha=0.7,
                label=f"{edges[e,0]}-{edges[e,1]}")
    vline(ax)
    ax.set_ylabel("G"); ax.set_title("Conductances")
    if m <= 12:
        ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # Row 0 — Adaptive pi_a
    ax = axes[0, 1]
    ax.plot(t, out["pi_a"], color="purple", lw=1.3)
    vline(ax)
    ax.set_ylabel("π_a"); ax.set_title("Adaptive π_a"); ax.grid(alpha=0.3)

    # Row 1 — Loop signatures (mean over loops)
    ax = axes[1, 0]
    sig = out["sig"]
    if sig.shape[1] > 0:
        ax.plot(t, sig.mean(axis=1), color="royalblue", lw=1.3, label="mean sig")
        ax.plot(t, out["mismatch"].mean(axis=1), color="crimson", lw=1.1,
                label="mean mismatch")
        ax.legend(fontsize=8)
    vline(ax)
    ax.set_ylabel("value"); ax.set_title("Loop Signatures & Mismatch"); ax.grid(alpha=0.3)

    # Row 1 — Edge feedback & η
    ax = axes[1, 1]
    ax.plot(t, out["edge_fb"].mean(axis=1), color="teal", lw=1.2, label="mean edge_fb")
    ax.plot(t, out["eta"].mean(axis=1), color="darkorange", lw=1.2, label="mean η")
    ax.legend(fontsize=8)
    vline(ax)
    ax.set_ylabel("value"); ax.set_title("Edge Feedback & η"); ax.grid(alpha=0.3)

    # Row 2 — Flip rate & entropy
    ax = axes[2, 0]
    ax.plot(t, out["r_b"], color="orange", lw=1.2)
    vline(ax)
    ax.set_ylabel("rate"); ax.set_title("Flip Rate r_b"); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, out["S"], color="sienna", lw=1.2, label="S")
    ax2 = ax.twinx()
    ax2.plot(t, np.abs(out["I"]).mean(axis=1), color="steelblue",
             lw=1.0, alpha=0.7, label="|I| mean")
    ax.set_ylabel("S"); ax2.set_ylabel("|I| mean")
    ax.set_title("Entropy & Mean Current"); ax.grid(alpha=0.3)

    # Row 3 — Canonical #1 diagnostics
    ax = axes[3, 0]
    ax.plot(t, out["M_u"].mean(axis=1), color="firebrick", lw=1.3, label="mean M^(U)")
    ax.plot(t, out["U_edge"].mean(axis=1), color="goldenrod", lw=1.2, label="mean U")
    ax.plot(t, out["Psi"], color="mediumslateblue", lw=1.3, label="Ψ")
    ax.legend(fontsize=8)
    vline(ax)
    ax.set_ylabel("value"); ax.set_title("Canonical #1: Memory / Surprise / Ψ")
    ax.grid(alpha=0.3)

    ax = axes[3, 1]
    ax.plot(t, out["chi_edge"].mean(axis=1), color="forestgreen", lw=1.3,
            label="mean χ_edge")
    ax.legend(fontsize=8)
    vline(ax)
    ax.set_ylabel("χ_edge"); ax.set_title("Passive Edge Gate χ_edge"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Dashboard saved: {save_path}")


# ------------------------------------------------------------------ #
#  Three-way ablation benchmark                                        #
# ------------------------------------------------------------------ #

def run_ablation(
    graph: Graph, source: int, sink: int,
    T: float = 60.0, dt: float = 0.05,
    damage_time: float = 20.0,
    seed: int = 7,
    tail: int = 20,
) -> None:
    """Run the three-way ablation comparing:
      (A) Legacy active π_f (no surprise memory, no edge gate).
      (B) Canonical #1 with ζ=0 (surprise memory only, no passive edge protection).
      (C) Canonical #1 full (surprise memory + edge gate).
    """
    damage_edges = list(range(len(graph.edges) // 2))

    configs = [
        ("(A) Legacy π_f (no canonical #1)",
         PiFParams(enabled=True, edge_coupling=0.50,
                   kappa_psi=0.0, xi_memory=0.0, tau_memory=1.5,
                   zeta_edge=0.0, surprise_blend=0.5)),
        ("(B) Canonical #1, ζ=0 (surprise only)",
         PiFParams(enabled=True, edge_coupling=0.50,
                   kappa_psi=0.75, xi_memory=1.20, tau_memory=1.5,
                   zeta_edge=0.0, surprise_blend=0.5)),
        ("(C) Canonical #1 full (surprise + edge gate)",
         PiFParams(enabled=True, edge_coupling=0.50,
                   kappa_psi=0.75, xi_memory=1.20, tau_memory=1.5,
                   zeta_edge=0.45, surprise_blend=0.5)),
    ]

    print("\n" + "=" * 72)
    print("  THREE-WAY ABLATION — Canonical #1 (Surprise-Weighted Frustration-Healed)")
    print("=" * 72)
    hdr = f"  {'Variant':<42s} {'Recov':>7s} {'Ψ_fin':>7s} {'M_fin':>7s} {'U_fin':>7s} {'χ_fin':>7s}"
    print(hdr)
    print("  " + "-" * 70)

    for label, pif_params in configs:
        out = simulate_graph_active_pif(
            graph, source, sink,
            T=T, dt=dt, seed=seed,
            pif=pif_params,
            damage_time=damage_time,
            damage_edges=damage_edges,
        )
        s = summarize_active_pif(out, damage_time=damage_time, tail=tail)
        print(
            f"  {label:<42s} "
            f"{s['recovery_ratio']:7.3f} "
            f"{s['mean_Psi_final']:7.3f} "
            f"{s['mean_memory_final']:7.3f} "
            f"{s['mean_surprise_final']:7.3f} "
            f"{s['mean_edge_gate_final']:7.3f}"
        )


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="HAFC Active π_f — Canonical #1 Law simulator"
    )
    ap.add_argument("--graph", choices=["toy", "ladder"], default="toy")
    ap.add_argument("--ladder-n", type=int, default=8)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--damage-time", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="Three-way ablation benchmark")
    ap.add_argument("--kappa-psi", type=float, default=0.75)
    ap.add_argument("--xi-memory", type=float, default=1.20)
    ap.add_argument("--tau-memory", type=float, default=1.50)
    ap.add_argument("--zeta-edge", type=float, default=0.45)
    args = ap.parse_args()

    if args.graph == "ladder":
        graph, source, sink = ladder_graph(args.ladder_n)
    else:
        graph, source, sink = default_toy_graph()

    pif = PiFParams(
        enabled=True,
        kappa_psi=args.kappa_psi,
        xi_memory=args.xi_memory,
        tau_memory=args.tau_memory,
        zeta_edge=args.zeta_edge,
    )

    print(f"[hafc_pif_active] graph={args.graph}  T={args.T}  "
          f"damage_t={args.damage_time}  seed={args.seed}")
    print(f"  κ_Ψ={pif.kappa_psi}  ξ={pif.xi_memory}  "
          f"τ_M={pif.tau_memory}  ζ={pif.zeta_edge}")

    out = simulate_graph_active_pif(
        graph, source, sink,
        T=args.T, dt=args.dt, seed=args.seed,
        pif=pif,
        damage_time=args.damage_time,
    )

    s = summarize_active_pif(out, damage_time=args.damage_time)

    print("\n  SUMMARY")
    print(f"  {'recovery_ratio':<26s}: {s['recovery_ratio']:.4f}")
    print(f"  {'mismatch_tail':<26s}: {s['mismatch_tail']:.4f}")
    print(f"  {'mean_memory_final':<26s}: {s['mean_memory_final']:.4f}")
    print(f"  {'mean_edge_gate_final':<26s}: {s['mean_edge_gate_final']:.4f}")
    print(f"  {'mean_surprise_final':<26s}: {s['mean_surprise_final']:.4f}")
    print(f"  {'mean_Psi_final':<26s}: {s['mean_Psi_final']:.4f}")
    print(f"  {'final_pi_a':<26s}: {s['final_pi_a']:.4f}")

    if args.ablation:
        run_ablation(graph, source, sink, T=args.T, dt=args.dt,
                     damage_time=args.damage_time, seed=args.seed)

    if not args.no_plot:
        try:
            dashboard_active_pif(out, damage_time=args.damage_time)
        except ImportError:
            print("  (matplotlib not available; skipping dashboard)")


if __name__ == "__main__":
    main()
