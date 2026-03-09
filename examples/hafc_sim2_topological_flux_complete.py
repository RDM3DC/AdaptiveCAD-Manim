
"""
hafc_sim2_ssh_complete.py

Complete EGATL / Phase-Lift simulator with:
- warm-startable simulation state
- lifted vs principal-branch phase modes
- adaptive or fixed pi_a
- timed interventions (damage / recovery experiments)
- SSH chain benchmark builder
- recovery metrics for self-healing edge transport
- quick plotting helpers and an ablation runner

Core model:
  d/dt G_e = alpha_G(S) |I_e| exp(i theta_R,e) - mu_G(S) G_e
  dS/dt    = sum_e |I_e|^2 / T_e * Re(1/G_e) + kappa * sum_e |Delta w_e| - gamma (S - S_eq)
  theta_R  = lifted branch state with adaptive clipping by pi_a
  d pi_a/dt = alpha_pi * S - mu_pi * (pi_a - pi0)

This is still a research toy, but it is now organized for:
  pretrain -> defect injection -> recovery -> ablation comparisons
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import gmres


# -----------------------------
# Utilities
# -----------------------------
def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Vectorized wrap to (-pi, pi]."""
    a = (x + math.pi) % (2 * math.pi) - math.pi
    a = np.where(a <= -math.pi, a + 2 * math.pi, a)
    return a


def logistic(x: float) -> float:
    """Stable-ish logistic."""
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# -----------------------------
# Graph / linear solve
# -----------------------------
@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]


@dataclass
class EdgeMeta:
    bond_type: str          # "intra" or "inter"
    cell: int
    is_boundary: bool


@dataclass
class BenchmarkGraph:
    graph: Graph
    source: int
    sink: int
    edge_meta: List[EdgeMeta]
    edge_index_by_name: Dict[Tuple[int, int], int]


def build_nodal_matrix_sparse(n: int, edges: List[Tuple[int, int]], Y: np.ndarray) -> csr_matrix:
    """Sparse nodal admittance matrix for undirected edges with admittance Y_e."""
    m = len(edges)
    rows = np.empty(4 * m, dtype=int)
    cols = np.empty(4 * m, dtype=int)
    data = np.empty(4 * m, dtype=Y.dtype)

    for i, (u, v) in enumerate(edges):
        k = 4 * i
        rows[k + 0], cols[k + 0], data[k + 0] = u, u, Y[i]
        rows[k + 1], cols[k + 1], data[k + 1] = v, v, Y[i]
        rows[k + 2], cols[k + 2], data[k + 2] = u, v, -Y[i]
        rows[k + 3], cols[k + 3], data[k + 3] = v, u, -Y[i]

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def grounded_gmres(
    M: csr_matrix,
    b: np.ndarray,
    ground: int,
    x0: Optional[np.ndarray],
    rtol: float,
    maxiter: int,
) -> Tuple[np.ndarray, int]:
    """Solve Mx=b with x[ground]=0 using GMRES on reduced system."""
    n = b.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[ground] = False

    Mr = M[mask][:, mask]
    br = b[mask]
    x0r = None if x0 is None else x0[mask].copy()

    xr, info = gmres(Mr, br, x0=x0r, rtol=rtol, atol=0.0, maxiter=maxiter, restart=50)
    x = np.zeros(n, dtype=complex)
    x[mask] = xr
    x[ground] = 0.0 + 0.0j
    return x, int(info)


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class EGATLParams:
    # alpha gate parameters: alpha(S)=alpha0/[1+exp((S-Sc)/dS)]
    alpha0: float = 1.0
    S_c: float = 1.0
    dS: float = 0.35

    # mu gate: mu(S)=mu0*(S/S0)
    mu0: float = 0.55
    S0: float = 1.0

    # complex G clamps
    G_min: float = 1e-3
    G_max: float = 50.0
    G_imag_max: float = 50.0

    # optional budget on sum(Re(G))
    budget_re: Optional[float] = 10.0

    # phase-coupled suppression
    lambda_s: float = 0.0  # set >0 to enable


@dataclass
class EntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25       # relaxation to S_eq
    kappa_slip: float = 0.15  # slip entropy weight
    Tij: float = 1.0          # constant T_ij for toy runs


@dataclass
class RulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


@dataclass
class SimState:
    Gc: np.ndarray
    S: float
    pi_a: float
    theta_R: np.ndarray
    theta_prev: np.ndarray
    w_prev: np.ndarray
    phi_prev: np.ndarray
    b_prev: int = 1
    flip_count: int = 0


# -----------------------------
# Core gates
# -----------------------------
def alpha_G_of_S(S: float, p: EGATLParams) -> float:
    # alpha0 / [1+exp((S-Sc)/dS)] = alpha0 * sigmoid(-(S-Sc)/dS)
    return p.alpha0 * logistic(-(S - p.S_c) / max(1e-12, p.dS))


def mu_G_of_S(S: float, p: EGATLParams) -> float:
    return p.mu0 * (S / max(1e-12, p.S0))


# -----------------------------
# State helpers
# -----------------------------
def make_initial_state(
    graph: Graph,
    G0: Optional[np.ndarray] = None,
    S0: float = 0.5,
    pi0: float = math.pi,
) -> SimState:
    m = len(graph.edges)
    if G0 is None:
        G0 = np.ones(m, dtype=complex)
    G0 = np.array(G0, dtype=complex).copy()
    if G0.shape != (m,):
        raise ValueError(f"G0 must have shape ({m},), got {G0.shape}")

    return SimState(
        Gc=G0,
        S=float(S0),
        pi_a=float(pi0),
        theta_R=np.zeros(m, dtype=float),
        theta_prev=np.zeros(m, dtype=float),
        w_prev=np.zeros(m, dtype=int),
        phi_prev=np.zeros(graph.n, dtype=complex),
        b_prev=1,
        flip_count=0,
    )


def clone_state(state0: SimState) -> SimState:
    return SimState(
        Gc=np.array(state0.Gc, dtype=complex).copy(),
        S=float(state0.S),
        pi_a=float(state0.pi_a),
        theta_R=np.array(state0.theta_R, dtype=float).copy(),
        theta_prev=np.array(state0.theta_prev, dtype=float).copy(),
        w_prev=np.array(state0.w_prev, dtype=int).copy(),
        phi_prev=np.array(state0.phi_prev, dtype=complex).copy(),
        b_prev=int(state0.b_prev),
        flip_count=int(state0.flip_count),
    )


# -----------------------------
# Interventions
# -----------------------------
def apply_interventions(
    t_now: float,
    state: SimState,
    interventions: Optional[List[dict]],
) -> None:
    if not interventions:
        return

    for ev in interventions:
        if ev.get("done", False):
            continue
        if t_now < float(ev["time"]):
            continue

        kind = ev["type"]

        if kind == "scale_edges":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            factor = complex(ev.get("factor", 0.25))
            state.Gc[idx] *= factor

        elif kind == "set_edges":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            value = complex(ev.get("value", 0.0 + 0.0j))
            state.Gc[idx] = value

        elif kind == "kick_phase":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            delta = float(ev.get("delta", math.pi))
            state.theta_R[idx] += delta

        elif kind == "reset_entropy":
            state.S = float(ev.get("value", state.S))

        elif kind == "set_pi_a":
            state.pi_a = float(ev.get("value", state.pi_a))

        elif kind == "add_noise_to_edges":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            sigma = float(ev.get("sigma", 0.1))
            noise = sigma * (np.random.normal(size=idx.size) + 1j * np.random.normal(size=idx.size))
            state.Gc[idx] += noise

        else:
            raise ValueError(f"Unknown intervention type: {kind}")

        ev["done"] = True


# -----------------------------
# Simulation
# -----------------------------
def simulate(
    graph: Graph,
    source: int,
    sink: int,
    T: float = 40.0,
    dt: float = 0.05,
    seed: int = 0,
    eg: EGATLParams = EGATLParams(),
    ent: EntropyParams = EntropyParams(),
    ruler: RulerParams = RulerParams(),
    rtol: float = 1e-10,
    maxiter: int = 2500,
    state0: Optional[SimState] = None,
    phase_mode: Literal["lifted", "principal"] = "lifted",
    adaptive_pi: bool = True,
    interventions: Optional[List[dict]] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns:
      t, Gc(t), I(t), phi(t), theta_R_e(t), theta_e(t), w_e(t), dW_e(t),
      S(t), pi_a(t), r_b(t), solve_info(t), flip(t), final_state
    """
    rng = np.random.default_rng(seed)
    m = len(graph.edges)
    K = int(np.ceil(T / dt)) + 1
    t = np.linspace(0.0, T, K)

    state = make_initial_state(graph, S0=ent.S_init, pi0=ruler.pi_init) if state0 is None else clone_state(state0)

    if state.Gc.shape != (m,):
        raise ValueError(f"state.Gc must have shape ({m},), got {state.Gc.shape}")
    if state.theta_R.shape != (m,):
        raise ValueError(f"state.theta_R must have shape ({m},), got {state.theta_R.shape}")
    if state.theta_prev.shape != (m,):
        raise ValueError(f"state.theta_prev must have shape ({m},), got {state.theta_prev.shape}")
    if state.w_prev.shape != (m,):
        raise ValueError(f"state.w_prev must have shape ({m},), got {state.w_prev.shape}")
    if state.phi_prev.shape != (graph.n,):
        raise ValueError(f"state.phi_prev must have shape ({graph.n},), got {state.phi_prev.shape}")

    # make an independent copy so the caller can reuse their event list
    local_interventions = None if interventions is None else copy.deepcopy(interventions)

    # Logs
    G_hist = np.zeros((K, m), dtype=complex)
    I_hist = np.zeros((K, m), dtype=complex)
    phi_hist = np.zeros((K, graph.n), dtype=complex)
    thetaR_hist = np.zeros((K, m), dtype=float)
    theta_hist = np.zeros((K, m), dtype=float)
    w_hist = np.zeros((K, m), dtype=int)
    dW_hist = np.zeros((K, m), dtype=float)
    S_hist = np.zeros(K, dtype=float)
    pi_hist = np.zeros(K, dtype=float)
    flip_hist = np.zeros(K, dtype=int)
    rb_hist = np.zeros(K, dtype=float)
    info_hist = np.zeros(K, dtype=int)

    for k in range(K):
        t_now = float(t[k])

        # 0) Timed interventions happen before the solve for this step
        apply_interventions(t_now, state, local_interventions)

        # 1) Solve complex nodal system
        bvec = np.zeros(graph.n, dtype=complex)
        bvec[source] = 1.0 + 0.0j
        bvec[sink] = -1.0 + 0.0j

        M = build_nodal_matrix_sparse(graph.n, graph.edges, state.Gc)
        phi, info = grounded_gmres(
            M, bvec, ground=sink, x0=state.phi_prev, rtol=rtol, maxiter=maxiter
        )
        state.phi_prev = phi
        info_hist[k] = info

        # 2) Edge currents and raw phases
        I = np.zeros(m, dtype=complex)
        theta = np.zeros(m, dtype=float)

        for e, (u, v) in enumerate(graph.edges):
            dV = phi[u] - phi[v]
            I[e] = state.Gc[e] * dV
            theta[e] = float(np.angle(I[e] + 1e-18))

        # 3) Phase update
        r = wrap_to_pi(theta - state.theta_prev)
        if phase_mode == "lifted":
            r_clip = np.clip(r, -state.pi_a, +state.pi_a)
            state.theta_R = state.theta_R + r_clip
        elif phase_mode == "principal":
            state.theta_R = theta.copy()
            r_clip = r
        else:
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        state.theta_prev = theta.copy()

        w = np.round(state.theta_R / (2.0 * math.pi)).astype(int)
        dW = np.abs(w - state.w_prev).astype(float)
        state.w_prev = w.copy()

        # toy global parity diagnostic, kept for backward compatibility
        b_edges = np.where((w % 2) == 0, 1, -1).astype(int)
        b = 1 if int(np.sum(b_edges)) >= 0 else -1
        flip = 1 if b != state.b_prev else 0
        state.b_prev = b
        state.flip_count += flip

        # 4) Entropy update
        Re_invG = np.real(1.0 / (state.Gc + 1e-18))
        term1 = float(
            np.sum((np.abs(I) ** 2) / max(1e-12, ent.Tij) * np.maximum(0.0, Re_invG))
        )
        term2 = float(ent.kappa_slip * np.sum(dW))
        term3 = float(-ent.gamma * (state.S - ent.S_eq))

        dS = term1 + term2 + term3
        state.S = float(max(0.0, state.S + dt * dS))

        # 5) Adaptive ruler update
        if adaptive_pi:
            dpi = ruler.alpha_pi * state.S - ruler.mu_pi * (state.pi_a - ruler.pi0)
            state.pi_a = float(np.clip(state.pi_a + dt * dpi, ruler.pi_min, ruler.pi_max))

        # 6) Conductance update + optional suppression
        aS = alpha_G_of_S(state.S, eg)
        mS = mu_G_of_S(state.S, eg)
        dGc = (aS * np.abs(I) * np.exp(1j * state.theta_R)) - (mS * state.Gc)

        if eg.lambda_s > 0:
            sup = np.sin(state.theta_R / (2.0 * state.pi_a + 1e-18)) ** 2
            dGc = dGc - eg.lambda_s * (sup * state.Gc)

        # small noise to break perfect symmetry
        dGc = dGc + 1e-6 * (rng.normal(size=m) + 1j * rng.normal(size=m))
        state.Gc = state.Gc + dt * dGc

        # 7) clamps
        Re = np.clip(state.Gc.real, eg.G_min, eg.G_max)
        Im = np.clip(state.Gc.imag, -eg.G_imag_max, eg.G_imag_max)
        state.Gc = Re + 1j * Im

        if eg.budget_re is not None:
            sRe = float(np.sum(state.Gc.real))
            if sRe > eg.budget_re and sRe > 0:
                scale = eg.budget_re / sRe
                state.Gc = (state.Gc.real * scale) + 1j * (state.Gc.imag * scale)

        # logs
        G_hist[k, :] = state.Gc
        I_hist[k, :] = I
        phi_hist[k, :] = phi
        thetaR_hist[k, :] = state.theta_R
        theta_hist[k, :] = theta
        w_hist[k, :] = w
        dW_hist[k, :] = dW
        S_hist[k] = state.S
        pi_hist[k] = state.pi_a
        flip_hist[k] = flip
        rb_hist[k] = state.flip_count / float(k + 1)

    return {
        "t": t,
        "Gc": G_hist,
        "I": I_hist,
        "phi": phi_hist,
        "theta_R_e": thetaR_hist,
        "theta_e": theta_hist,
        "w_e": w_hist,
        "dW_e": dW_hist,
        "S": S_hist,
        "pi_a": pi_hist,
        "flip": flip_hist,
        "r_b": rb_hist,
        "solve_info": info_hist,
        "edges": np.array(graph.edges, dtype=int),
        "final_state": state,
    }


# -----------------------------
# SSH benchmark
# -----------------------------
def ssh_chain_graph(n_cells: int) -> BenchmarkGraph:
    """
    Open-boundary SSH chain with 2*n_cells sites:
      A0-B0-A1-B1-...-A_{N-1}-B_{N-1}
    Edges alternate:
      intra: A_c -- B_c
      inter: B_c -- A_{c+1}
    """
    if n_cells < 2:
        raise ValueError("n_cells must be at least 2")

    edges: List[Tuple[int, int]] = []
    meta: List[EdgeMeta] = []
    edge_index_by_name: Dict[Tuple[int, int], int] = {}

    def add_edge(u: int, v: int, bond_type: str, cell: int, is_boundary: bool = False) -> None:
        idx = len(edges)
        edges.append((u, v))
        meta.append(EdgeMeta(bond_type=bond_type, cell=cell, is_boundary=is_boundary))
        edge_index_by_name[(u, v)] = idx

    n = 2 * n_cells
    for c in range(n_cells):
        u, v = 2 * c, 2 * c + 1
        add_edge(u, v, "intra", c, is_boundary=(c == 0 or c == n_cells - 1))
        if c < n_cells - 1:
            add_edge(v, 2 * (c + 1), "inter", c, is_boundary=(c == 0 or c == n_cells - 2))

    return BenchmarkGraph(
        graph=Graph(n=n, edges=edges),
        source=0,
        sink=n - 1,
        edge_meta=meta,
        edge_index_by_name=edge_index_by_name,
    )


def ssh_initial_G(
    edge_meta: List[EdgeMeta],
    g_intra: complex = 0.6 + 0.0j,
    g_inter: complex = 1.4 + 0.0j,
) -> np.ndarray:
    out = np.zeros(len(edge_meta), dtype=complex)
    for i, em in enumerate(edge_meta):
        out[i] = g_intra if em.bond_type == "intra" else g_inter
    return out


# -----------------------------
# Metrics
# -----------------------------
def effective_admittance(phi: np.ndarray, source: int, sink: int, eps: float = 1e-12) -> float:
    dv = phi[source] - phi[sink]
    return 1.0 / max(eps, abs(dv))


def time_series_effective_admittance(phi_hist: np.ndarray, source: int, sink: int) -> np.ndarray:
    out = np.zeros(phi_hist.shape[0], dtype=float)
    for k in range(phi_hist.shape[0]):
        out[k] = effective_admittance(phi_hist[k], source, sink)
    return out


def boundary_current_fraction(I: np.ndarray, edge_meta: List[EdgeMeta]) -> float:
    num = 0.0
    den = float(np.sum(np.abs(I))) + 1e-12
    for val, em in zip(I, edge_meta):
        if em.is_boundary:
            num += abs(val)
    return num / den


def time_series_boundary_fraction(I_hist: np.ndarray, edge_meta: List[EdgeMeta]) -> np.ndarray:
    out = np.zeros(I_hist.shape[0], dtype=float)
    for k in range(I_hist.shape[0]):
        out[k] = boundary_current_fraction(I_hist[k], edge_meta)
    return out


def bond_dimerization(Gc: np.ndarray, edge_meta: List[EdgeMeta]) -> float:
    intra = [Gc[i].real for i, em in enumerate(edge_meta) if em.bond_type == "intra"]
    inter = [Gc[i].real for i, em in enumerate(edge_meta) if em.bond_type == "inter"]
    return float(np.mean(inter) - np.mean(intra))


def time_series_bond_dimerization(G_hist: np.ndarray, edge_meta: List[EdgeMeta]) -> np.ndarray:
    out = np.zeros(G_hist.shape[0], dtype=float)
    for k in range(G_hist.shape[0]):
        out[k] = bond_dimerization(G_hist[k], edge_meta)
    return out


def slip_density(dW_hist: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(dW_hist), axis=1)


def summarize_recovery(
    out: Dict[str, np.ndarray],
    bench: BenchmarkGraph,
    damage_time: float,
    settle_window: float = 5.0,
) -> Dict[str, float]:
    t = out["t"]
    Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
    Bfrac = time_series_boundary_fraction(out["I"], bench.edge_meta)
    Dimer = time_series_bond_dimerization(out["Gc"], bench.edge_meta)

    pre_mask = (t >= max(0.0, damage_time - settle_window)) & (t < damage_time)
    post_mask = t >= max(damage_time + settle_window, damage_time)

    if not np.any(pre_mask):
        pre_mask = t < damage_time
    if not np.any(post_mask):
        post_mask = t >= damage_time

    pre_Y = float(np.mean(Yeff[pre_mask]))
    post_Y = float(np.mean(Yeff[post_mask]))
    pre_B = float(np.mean(Bfrac[pre_mask]))
    post_B = float(np.mean(Bfrac[post_mask]))
    pre_D = float(np.mean(Dimer[pre_mask]))
    post_D = float(np.mean(Dimer[post_mask]))

    return {
        "Yeff_pre": pre_Y,
        "Yeff_post": post_Y,
        "Yeff_recovery_ratio": post_Y / max(1e-12, pre_Y),
        "boundary_pre": pre_B,
        "boundary_post": post_B,
        "dimer_pre": pre_D,
        "dimer_post": post_D,
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fail_steps": float(np.sum(out["solve_info"] != 0)),
    }


# -----------------------------
# Protocols
# -----------------------------
def run_ssh_recovery_protocol(
    n_cells: int = 20,
    T: float = 60.0,
    dt: float = 0.05,
    seed: int = 0,
    damage_time: float = 20.0,
    damage_edge_idx: Optional[List[int]] = None,
    damage_factor: float = 0.15,
    phase_mode: Literal["lifted", "principal"] = "lifted",
    adaptive_pi: bool = True,
    eg: Optional[EGATLParams] = None,
    ent: Optional[EntropyParams] = None,
    ruler: Optional[RulerParams] = None,
) -> Tuple[BenchmarkGraph, Dict[str, np.ndarray]]:
    bench = ssh_chain_graph(n_cells)

    if eg is None:
        eg = EGATLParams(
            alpha0=1.0, S_c=1.0, dS=0.35,
            mu0=0.55, S0=1.0,
            budget_re=12.0, lambda_s=0.10,
        )
    if ent is None:
        ent = EntropyParams(S_init=0.5, S_eq=0.5, gamma=0.25, kappa_slip=0.15, Tij=1.0)
    if ruler is None:
        ruler = RulerParams(pi0=math.pi, pi_init=math.pi, alpha_pi=0.25, mu_pi=0.20)

    G0 = ssh_initial_G(bench.edge_meta, g_intra=0.6 + 0.0j, g_inter=1.4 + 0.0j)
    state0 = make_initial_state(bench.graph, G0=G0, S0=ent.S_init, pi0=ruler.pi_init)

    if damage_edge_idx is None:
        # damage left boundary and the first intercell bond by default
        damage_edge_idx = [0, 1, 2] if len(bench.graph.edges) >= 3 else [0]

    interventions = [
        {"time": float(damage_time), "type": "scale_edges", "edge_idx": damage_edge_idx, "factor": damage_factor},
    ]

    out = simulate(
        bench.graph,
        bench.source,
        bench.sink,
        T=T,
        dt=dt,
        seed=seed,
        eg=eg,
        ent=ent,
        ruler=ruler,
        state0=state0,
        phase_mode=phase_mode,
        adaptive_pi=adaptive_pi,
        interventions=interventions,
    )
    return bench, out


def compare_ablations(
    n_cells: int = 20,
    T: float = 60.0,
    dt: float = 0.05,
    seed: int = 0,
    damage_time: float = 20.0,
) -> Dict[str, Tuple[BenchmarkGraph, Dict[str, np.ndarray], Dict[str, float]]]:
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }

    results = {}
    for name, cfg in configs.items():
        bench, out = run_ssh_recovery_protocol(
            n_cells=n_cells,
            T=T,
            dt=dt,
            seed=seed,
            damage_time=damage_time,
            phase_mode=cfg["phase_mode"],
            adaptive_pi=cfg["adaptive_pi"],
        )
        summary = summarize_recovery(out, bench, damage_time=damage_time)
        results[name] = (bench, out, summary)

    return results




# -----------------------------
# 2D topological flux benchmark
# -----------------------------
def qwz_lattice_graph(nx: int, ny: int) -> BenchmarkGraph:
    """
    2D square lattice with open boundaries.

    Note:
      With the current scalar nodal-admittance model this is best interpreted as a
      synthetic-flux / Harper-Hofstadter-style benchmark rather than a literal two-band
      QWZ Hamiltonian. The function name is kept for continuity with the planned 2D
      topological benchmark path.
    """
    if nx < 3 or ny < 3:
        raise ValueError("Lattice must be at least 3x3")

    edges: List[Tuple[int, int]] = []
    meta: List[EdgeMeta] = []
    edge_index_by_name: Dict[Tuple[int, int], int] = {}

    def add_edge(u: int, v: int, bond_type: str, is_boundary: bool = False) -> None:
        idx = len(edges)
        edges.append((u, v))
        meta.append(EdgeMeta(bond_type=bond_type, cell=0, is_boundary=is_boundary))
        edge_index_by_name[(u, v)] = idx
        edge_index_by_name[(v, u)] = idx

    for y in range(ny):
        for x in range(nx):
            u = x + y * nx

            # x-direction bonds
            if x < nx - 1:
                v = (x + 1) + y * nx
                is_boundary = (y == 0 or y == ny - 1)
                add_edge(u, v, f"x_bond_y{y}", is_boundary=is_boundary)

            # y-direction bonds
            if y < ny - 1:
                v = x + (y + 1) * nx
                is_boundary = (x == 0 or x == nx - 1)
                add_edge(u, v, "y_bond", is_boundary=is_boundary)

    # Drive along the top boundary: top-left -> top-right
    source = 0 + (ny - 1) * nx
    sink = (nx - 1) + (ny - 1) * nx

    return BenchmarkGraph(
        graph=Graph(n=nx * ny, edges=edges),
        source=source,
        sink=sink,
        edge_meta=meta,
        edge_index_by_name=edge_index_by_name,
    )


def qwz_initial_G(
    bench: BenchmarkGraph,
    nx: int,
    g_mag: float = 1.0,
    flux_alpha: float = 0.25,
) -> np.ndarray:
    """
    Initialize a Landau-gauge synthetic flux:
      G_x(x, y) = g_mag * exp(i * 2*pi*alpha*y)
      G_y(x, y) = g_mag
    """
    out = np.zeros(len(bench.graph.edges), dtype=complex)
    for i, (u, v) in enumerate(bench.graph.edges):
        uy, ux = divmod(u, nx)
        vy, vx = divmod(v, nx)

        if uy == vy:  # x-bond
            phase = 2.0 * math.pi * flux_alpha * uy
            out[i] = g_mag * np.exp(1j * phase)
        else:         # y-bond
            out[i] = g_mag + 0.0j
    return out


def current_fraction_for_bond_prefix(I: np.ndarray, edge_meta: List[EdgeMeta], prefix: str) -> float:
    num = 0.0
    den = float(np.sum(np.abs(I))) + 1e-12
    for val, em in zip(I, edge_meta):
        if em.bond_type.startswith(prefix):
            num += abs(val)
    return num / den


def time_series_current_fraction_for_bond_prefix(
    I_hist: np.ndarray,
    edge_meta: List[EdgeMeta],
    prefix: str,
) -> np.ndarray:
    out = np.zeros(I_hist.shape[0], dtype=float)
    for k in range(I_hist.shape[0]):
        out[k] = current_fraction_for_bond_prefix(I_hist[k], edge_meta, prefix)
    return out


def run_qwz_recovery_protocol(
    nx: int = 8,
    ny: int = 8,
    T: float = 80.0,
    dt: float = 0.05,
    seed: int = 0,
    damage_time: float = 30.0,
    damage_factor: float = 1e-6,
    flux_alpha: float = 0.25,
    g_mag: float = 1.0,
    phase_mode: Literal["lifted", "principal"] = "lifted",
    adaptive_pi: bool = True,
    eg: Optional[EGATLParams] = None,
    ent: Optional[EntropyParams] = None,
    ruler: Optional[RulerParams] = None,
) -> Tuple[BenchmarkGraph, Dict[str, np.ndarray]]:
    bench = qwz_lattice_graph(nx, ny)

    if eg is None:
        eg = EGATLParams(
            alpha0=1.5, S_c=1.0, dS=0.35,
            mu0=0.55, S0=1.0,
            budget_re=25.0, lambda_s=0.15,
        )
    if ent is None:
        ent = EntropyParams(S_init=0.5, S_eq=0.5, gamma=0.25, kappa_slip=0.2, Tij=1.0)
    if ruler is None:
        ruler = RulerParams(pi0=math.pi, pi_init=math.pi, alpha_pi=0.25, mu_pi=0.20)

    G0 = qwz_initial_G(bench, nx=nx, g_mag=g_mag, flux_alpha=flux_alpha)
    state0 = make_initial_state(bench.graph, G0=G0, S0=ent.S_init, pi0=ruler.pi_init)

    # Damage the middle of the driven top edge by severing all incident bonds at that node.
    target_x = nx // 2
    target_y = ny - 1
    target_node = target_x + target_y * nx
    damage_idx: List[int] = []

    for i, (u, v) in enumerate(bench.graph.edges):
        if u == target_node or v == target_node:
            damage_idx.append(i)

    interventions = [
        {"time": float(damage_time), "type": "scale_edges", "edge_idx": damage_idx, "factor": damage_factor},
        {"time": float(damage_time), "type": "reset_entropy", "value": 3.0},
    ]

    out = simulate(
        bench.graph,
        bench.source,
        bench.sink,
        T=T,
        dt=dt,
        seed=seed,
        eg=eg,
        ent=ent,
        ruler=ruler,
        state0=state0,
        phase_mode=phase_mode,
        adaptive_pi=adaptive_pi,
        interventions=interventions,
    )
    return bench, out


def summarize_qwz_recovery(
    out: Dict[str, np.ndarray],
    bench: BenchmarkGraph,
    ny: int,
    damage_time: float,
    settle_window: float = 5.0,
) -> Dict[str, float]:
    t = out["t"]
    Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
    perimeter = time_series_boundary_fraction(out["I"], bench.edge_meta)
    top_edge = time_series_current_fraction_for_bond_prefix(out["I"], bench.edge_meta, f"x_bond_y{ny - 1}")
    slips = slip_density(out["dW_e"])

    pre_mask = (t >= max(0.0, damage_time - settle_window)) & (t < damage_time)
    post_mask = t >= max(damage_time + settle_window, damage_time)

    if not np.any(pre_mask):
        pre_mask = t < damage_time
    if not np.any(post_mask):
        post_mask = t >= damage_time

    pre_Y = float(np.mean(Yeff[pre_mask]))
    post_Y = float(np.mean(Yeff[post_mask]))
    pre_P = float(np.mean(perimeter[pre_mask]))
    post_P = float(np.mean(perimeter[post_mask]))
    pre_T = float(np.mean(top_edge[pre_mask]))
    post_T = float(np.mean(top_edge[post_mask]))
    post_slip = float(np.mean(slips[post_mask]))

    return {
        "Yeff_pre": pre_Y,
        "Yeff_post": post_Y,
        "Yeff_recovery_ratio": post_Y / max(1e-12, pre_Y),
        "perimeter_pre": pre_P,
        "perimeter_post": post_P,
        "top_edge_pre": pre_T,
        "top_edge_post": post_T,
        "post_slip_density": post_slip,
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fail_steps": float(np.sum(out["solve_info"] != 0)),
    }


def compare_ablations_qwz(
    nx: int = 8,
    ny: int = 8,
    T: float = 80.0,
    dt: float = 0.05,
    seed: int = 0,
    damage_time: float = 30.0,
    flux_alpha: float = 0.25,
) -> Dict[str, Tuple[BenchmarkGraph, Dict[str, np.ndarray], Dict[str, float]]]:
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }

    results = {}
    for name, cfg in configs.items():
        bench, out = run_qwz_recovery_protocol(
            nx=nx,
            ny=ny,
            T=T,
            dt=dt,
            seed=seed,
            damage_time=damage_time,
            flux_alpha=flux_alpha,
            phase_mode=cfg["phase_mode"],
            adaptive_pi=cfg["adaptive_pi"],
        )
        summary = summarize_qwz_recovery(out, bench, ny=ny, damage_time=damage_time)
        results[name] = (bench, out, summary)

    return results


def plot_ablation_overlay_qwz(
    results: Dict[str, Tuple[BenchmarkGraph, Dict[str, np.ndarray], Dict[str, float]]],
    ny: int,
    damage_time: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)

    for name, (bench, out, summary) in results.items():
        t = out["t"]
        Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
        perimeter = time_series_boundary_fraction(out["I"], bench.edge_meta)
        top_edge = time_series_current_fraction_for_bond_prefix(out["I"], bench.edge_meta, f"x_bond_y{ny - 1}")
        slips = slip_density(out["dW_e"])

        axs[0].plot(t, Yeff, lw=2, label=name)
        axs[1].plot(t, perimeter, lw=2, label=name)
        axs[2].plot(t, top_edge, lw=2, label=name)
        axs[3].plot(t, slips, lw=2, label=name)

    axs[0].set_title("Effective admittance")
    axs[1].set_title("Perimeter current fraction")
    axs[2].set_title(f"Top-edge current fraction (y={ny - 1})")
    axs[3].set_title("Slip density")

    for ax in axs:
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.3)
        if damage_time is not None:
            ax.axvline(damage_time, color="red", ls="--", alpha=0.5)
        ax.legend()

    plt.show()


def quick_dashboard_qwz(
    out: Dict[str, np.ndarray],
    bench: BenchmarkGraph,
    ny: int,
    title: str = "2D topological flux recovery",
    damage_time: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    t = out["t"]
    Gmag = np.abs(out["Gc"])
    S = out["S"]
    pi_a = out["pi_a"]
    rb = out["r_b"]
    Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
    perimeter = time_series_boundary_fraction(out["I"], bench.edge_meta)
    top_edge = time_series_current_fraction_for_bond_prefix(out["I"], bench.edge_meta, f"x_bond_y{ny - 1}")
    slips = slip_density(out["dW_e"])

    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    if Gmag.shape[1] <= 40:
        for i, (u, v) in enumerate(out["edges"]):
            axs[0, 0].plot(t, Gmag[:, i], lw=1.0, label=f"{u}-{v}")
        if Gmag.shape[1] <= 16:
            axs[0, 0].legend(fontsize=7, ncol=2)
    else:
        axs[0, 0].plot(t, np.mean(Gmag, axis=1), lw=2, label="mean |G|")
        axs[0, 0].plot(t, np.max(Gmag, axis=1), lw=2, label="max |G|")
        axs[0, 0].legend()

    axs[0, 0].set_title("|G_e|(t)")
    axs[0, 0].set_xlabel("time")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t, S, lw=2, label="S(t)")
    axs[0, 1].set_title("Entropy proxy S(t)")
    axs[0, 1].set_xlabel("time")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(t, pi_a, lw=2, label="pi_a")
    axs[1, 0].axhline(math.pi, color="gray", ls="--", alpha=0.7, label="pi0")
    axs[1, 0].set_title("Adaptive ruler pi_a(t)")
    axs[1, 0].set_xlabel("time")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(t, rb, lw=2, label="r_b")
    axs[1, 1].step(t, out["flip"], where="post", lw=1.2, alpha=0.7, label="flip")
    axs[1, 1].set_title("Parity / flip diagnostic")
    axs[1, 1].set_xlabel("time")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    axs[2, 0].plot(t, Yeff, lw=2, label="Y_eff")
    axs[2, 0].plot(t, perimeter, lw=2, label="perimeter fraction")
    axs[2, 0].plot(t, top_edge, lw=2, label="top-edge fraction")
    axs[2, 0].set_title("2D recovery metrics")
    axs[2, 0].set_xlabel("time")
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(t, slips, lw=2, label="mean |Δw|")
    axs[2, 1].set_title("Slip density")
    axs[2, 1].set_xlabel("time")
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.3)

    if damage_time is not None:
        for ax in axs.ravel():
            ax.axvline(damage_time, color="red", ls="--", alpha=0.5)

    plt.show()

# -----------------------------
# Visualization
# -----------------------------
def quick_dashboard(
    out: Dict[str, np.ndarray],
    title: str = "EGATL / Phase-Lift run",
    bench: Optional[BenchmarkGraph] = None,
    damage_time: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    t = out["t"]
    Gmag = np.abs(out["Gc"])
    S = out["S"]
    pi_a = out["pi_a"]
    rb = out["r_b"]

    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    for i, (u, v) in enumerate(out["edges"]):
        axs[0, 0].plot(t, Gmag[:, i], lw=1.0, label=f"{u}-{v}")
    axs[0, 0].set_title("|G_e|(t)")
    axs[0, 0].set_xlabel("time")
    axs[0, 0].set_ylabel("|G|")
    if Gmag.shape[1] <= 12:
        axs[0, 0].legend(fontsize=8, ncol=2)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t, S, lw=2, label="S(t)")
    axs[0, 1].set_title("Entropy proxy S(t)")
    axs[0, 1].set_xlabel("time")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(t, pi_a, lw=2, label="pi_a")
    axs[1, 0].axhline(math.pi, color="gray", ls="--", alpha=0.7, label="pi0")
    axs[1, 0].set_title("Adaptive ruler pi_a(t)")
    axs[1, 0].set_xlabel("time")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(t, rb, lw=2, label="r_b")
    axs[1, 1].step(t, out["flip"], where="post", lw=1.2, alpha=0.7, label="flip")
    axs[1, 1].set_title("Parity / flip diagnostic")
    axs[1, 1].set_xlabel("time")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    axs[2, 0].plot(t, slip_density(out["dW_e"]), lw=2, label="mean |Δw|")
    axs[2, 0].set_title("Slip density")
    axs[2, 0].set_xlabel("time")
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)

    if bench is not None:
        Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
        Bfrac = time_series_boundary_fraction(out["I"], bench.edge_meta)
        Dimer = time_series_bond_dimerization(out["Gc"], bench.edge_meta)

        axs[2, 1].plot(t, Yeff, lw=2, label="Y_eff")
        axs[2, 1].plot(t, Bfrac, lw=2, label="boundary current frac")
        axs[2, 1].plot(t, Dimer, lw=2, label="bond dimerization")
        axs[2, 1].set_title("SSH recovery metrics")
        axs[2, 1].set_xlabel("time")
        axs[2, 1].legend()
        axs[2, 1].grid(True, alpha=0.3)
    else:
        axs[2, 1].axis("off")

    if damage_time is not None:
        for ax in axs.ravel():
            ax.axvline(damage_time, color="red", ls="--", alpha=0.5)

    plt.show()


def plot_ablation_overlay(
    results: Dict[str, Tuple[BenchmarkGraph, Dict[str, np.ndarray], Dict[str, float]]],
    damage_time: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)

    for name, (bench, out, summary) in results.items():
        t = out["t"]
        Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
        Bfrac = time_series_boundary_fraction(out["I"], bench.edge_meta)
        Dimer = time_series_bond_dimerization(out["Gc"], bench.edge_meta)

        axs[0].plot(t, Yeff, lw=2, label=name)
        axs[1].plot(t, Bfrac, lw=2, label=name)
        axs[2].plot(t, Dimer, lw=2, label=name)

    axs[0].set_title("Effective admittance")
    axs[1].set_title("Boundary current fraction")
    axs[2].set_title("Bond dimerization")

    for ax in axs:
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.3)
        if damage_time is not None:
            ax.axvline(damage_time, color="red", ls="--", alpha=0.5)
        ax.legend()

    plt.show()


# -----------------------------
# Demo / CLI-like main
# -----------------------------
def print_top_edges(out: Dict[str, np.ndarray], k: int = 8) -> None:
    Gf = out["Gc"][-1]
    edges = out["edges"]
    idx = np.argsort(-np.abs(Gf))

    print("\nTop edges by final |G|:")
    for j in idx[: min(k, len(idx))]:
        u, v = edges[j]
        print(f"  {u}-{v}:  G={Gf[j].real:+.4f}{Gf[j].imag:+.4f}j   |G|={abs(Gf[j]):.4f}")


def main() -> None:
    damage_time = 20.0

    bench, out = run_ssh_recovery_protocol(
        n_cells=20,
        T=60.0,
        dt=0.05,
        seed=0,
        damage_time=damage_time,
        phase_mode="lifted",
        adaptive_pi=True,
    )
    summary = summarize_recovery(out, bench, damage_time=damage_time)

    print("Single-run summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")

    print_top_edges(out)
    quick_dashboard(
        out,
        title="SSH recovery benchmark — lifted phase + adaptive pi_a",
        bench=bench,
        damage_time=damage_time,
    )

    results = compare_ablations(
        n_cells=20,
        T=60.0,
        dt=0.05,
        seed=0,
        damage_time=damage_time,
    )

    print("\nAblation summaries:")
    for name, (_, _, summ) in results.items():
        print(f"\n{name}")
        for k, v in summ.items():
            print(f"  {k}: {v:.6f}")

    plot_ablation_overlay(results, damage_time=damage_time)


if __name__ == "__main__":
    main()
