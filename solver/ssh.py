"""1D scalar SSH chain benchmark for the EGATL simulator.

Su-Schrieffer-Heeger chain with scalar complex admittances G_e
evolved by the EGATL law:

    d/dt G_e = alpha_G(S)|I_e| exp(i theta_R,e) - mu_G(S) G_e

Topological signature: bond dimerisation (|G_inter| - |G_intra|)
and boundary-localised transport after damage.

Promoted from examples/hafc_sim2_ssh_complete.py into the solver framework.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import gmres


# ---------------------------------------------------------------------------
# Utilities (local; avoid coupling to egatl internals)
# ---------------------------------------------------------------------------

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    a = (x + math.pi) % (2 * math.pi) - math.pi
    return np.where(a <= -math.pi, a + 2 * math.pi, a)


def _logistic(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

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
class SSHBenchmark:
    graph: Graph
    source: int
    sink: int
    edge_meta: List[EdgeMeta]
    edge_index_by_name: Dict[Tuple[int, int], int]


# ---------------------------------------------------------------------------
# Sparse nodal solve
# ---------------------------------------------------------------------------

def _build_nodal_matrix(n: int, edges: List[Tuple[int, int]], Y: np.ndarray) -> csr_matrix:
    m = len(edges)
    rows = np.empty(4 * m, dtype=int)
    cols = np.empty(4 * m, dtype=int)
    data = np.empty(4 * m, dtype=Y.dtype)
    for i, (u, v) in enumerate(edges):
        k = 4 * i
        rows[k], cols[k], data[k] = u, u, Y[i]
        rows[k + 1], cols[k + 1], data[k + 1] = v, v, Y[i]
        rows[k + 2], cols[k + 2], data[k + 2] = u, v, -Y[i]
        rows[k + 3], cols[k + 3], data[k + 3] = v, u, -Y[i]
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _grounded_gmres(M, b, ground, x0, rtol=1e-10, maxiter=2500):
    n = b.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[ground] = False
    Mr = M[mask][:, mask]
    br = b[mask]
    x0r = None if x0 is None else x0[mask].copy()
    xr, info = gmres(Mr, br, x0=x0r, rtol=rtol, atol=0.0, maxiter=maxiter, restart=50)
    x = np.zeros(n, dtype=complex)
    x[mask] = xr
    return x, int(info)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class SSHParams:
    alpha0: float = 1.0
    S_c: float = 1.0
    dS: float = 0.35
    mu0: float = 0.55
    S0: float = 1.0
    G_min: float = 1e-3
    G_max: float = 50.0
    G_imag_max: float = 50.0
    budget_re: Optional[float] = 12.0
    lambda_s: float = 0.10


@dataclass
class SSHEntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25
    kappa_slip: float = 0.15
    Tij: float = 1.0


@dataclass
class SSHRulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


@dataclass
class SSHState:
    Gc: np.ndarray
    S: float
    pi_a: float
    theta_R: np.ndarray
    theta_prev: np.ndarray
    w_prev: np.ndarray
    phi_prev: np.ndarray
    b_prev: int = 1
    flip_count: int = 0


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _alpha_G(S, p):
    return p.alpha0 * _logistic(-(S - p.S_c) / max(1e-12, p.dS))


def _mu_G(S, p):
    return p.mu0 * (S / max(1e-12, p.S0))


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def make_initial_state(graph, G0=None, S0=0.5, pi0=math.pi):
    m = len(graph.edges)
    if G0 is None:
        G0 = np.ones(m, dtype=complex)
    G0 = np.array(G0, dtype=complex).copy()
    return SSHState(
        Gc=G0, S=float(S0), pi_a=float(pi0),
        theta_R=np.zeros(m), theta_prev=np.zeros(m),
        w_prev=np.zeros(m, dtype=int),
        phi_prev=np.zeros(graph.n, dtype=complex),
    )


def clone_state(s):
    return SSHState(
        Gc=s.Gc.copy(), S=s.S, pi_a=s.pi_a,
        theta_R=s.theta_R.copy(), theta_prev=s.theta_prev.copy(),
        w_prev=s.w_prev.copy(), phi_prev=s.phi_prev.copy(),
        b_prev=s.b_prev, flip_count=s.flip_count,
    )


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

def _apply_interventions(t_now, state, interventions):
    if not interventions:
        return
    for ev in interventions:
        if ev.get("done"):
            continue
        if t_now < float(ev["time"]):
            continue
        kind = ev["type"]
        if kind == "scale_edges":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            state.Gc[idx] *= complex(ev.get("factor", 0.25))
        elif kind == "set_edges":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            state.Gc[idx] = complex(ev.get("value", 0.0))
        elif kind == "kick_phase":
            idx = np.asarray(ev["edge_idx"], dtype=int)
            state.theta_R[idx] += float(ev.get("delta", math.pi))
        elif kind == "reset_entropy":
            state.S = float(ev.get("value", state.S))
        elif kind == "set_pi_a":
            state.pi_a = float(ev.get("value", state.pi_a))
        ev["done"] = True


# ---------------------------------------------------------------------------
# SSH chain builder
# ---------------------------------------------------------------------------

def ssh_chain_graph(n_cells: int) -> SSHBenchmark:
    """Open-boundary SSH chain: A0-B0-A1-B1-...-A_{N-1}-B_{N-1}."""
    if n_cells < 2:
        raise ValueError("n_cells must be at least 2")
    edges: List[Tuple[int, int]] = []
    meta: List[EdgeMeta] = []
    eidx: Dict[Tuple[int, int], int] = {}

    def _add(u, v, btype, cell, bnd=False):
        i = len(edges)
        edges.append((u, v))
        meta.append(EdgeMeta(bond_type=btype, cell=cell, is_boundary=bnd))
        eidx[(u, v)] = i

    n = 2 * n_cells
    for c in range(n_cells):
        u, v = 2 * c, 2 * c + 1
        _add(u, v, "intra", c, is_boundary=(c == 0 or c == n_cells - 1))
        if c < n_cells - 1:
            _add(v, 2 * (c + 1), "inter", c, is_boundary=(c == 0 or c == n_cells - 2))

    return SSHBenchmark(
        graph=Graph(n=n, edges=edges),
        source=0, sink=n - 1,
        edge_meta=meta, edge_index_by_name=eidx,
    )


def ssh_initial_G(edge_meta, g_intra=0.6 + 0j, g_inter=1.4 + 0j):
    out = np.zeros(len(edge_meta), dtype=complex)
    for i, em in enumerate(edge_meta):
        out[i] = g_intra if em.bond_type == "intra" else g_inter
    return out


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def simulate(
    graph: Graph, source: int, sink: int,
    T=40.0, dt=0.05, seed=0,
    eg: Optional[SSHParams] = None,
    ent: Optional[SSHEntropyParams] = None,
    ruler: Optional[SSHRulerParams] = None,
    state0: Optional[SSHState] = None,
    phase_mode: Literal["lifted", "principal"] = "lifted",
    adaptive_pi: bool = True,
    interventions: Optional[List[dict]] = None,
) -> Dict[str, np.ndarray]:
    if eg is None:
        eg = SSHParams()
    if ent is None:
        ent = SSHEntropyParams()
    if ruler is None:
        ruler = SSHRulerParams()

    rng = np.random.default_rng(seed)
    m = len(graph.edges)
    K = int(np.ceil(T / dt)) + 1
    t = np.linspace(0.0, T, K)

    state = make_initial_state(graph, S0=ent.S_init, pi0=ruler.pi_init) if state0 is None else clone_state(state0)
    local_iv = None if interventions is None else copy.deepcopy(interventions)

    G_h = np.zeros((K, m), dtype=complex)
    I_h = np.zeros((K, m), dtype=complex)
    phi_h = np.zeros((K, graph.n), dtype=complex)
    tR_h = np.zeros((K, m))
    th_h = np.zeros((K, m))
    w_h = np.zeros((K, m), dtype=int)
    dW_h = np.zeros((K, m))
    S_h = np.zeros(K)
    pi_h = np.zeros(K)
    flip_h = np.zeros(K, dtype=int)
    rb_h = np.zeros(K)
    info_h = np.zeros(K, dtype=int)

    for k in range(K):
        t_now = float(t[k])
        _apply_interventions(t_now, state, local_iv)

        bvec = np.zeros(graph.n, dtype=complex)
        bvec[source] = 1.0
        bvec[sink] = -1.0
        M = _build_nodal_matrix(graph.n, graph.edges, state.Gc)
        phi, info = _grounded_gmres(M, bvec, sink, state.phi_prev)
        state.phi_prev = phi
        info_h[k] = info

        I = np.zeros(m, dtype=complex)
        theta = np.zeros(m)
        for e, (u, v) in enumerate(graph.edges):
            I[e] = state.Gc[e] * (phi[u] - phi[v])
            theta[e] = float(np.angle(I[e] + 1e-18))

        r = _wrap_to_pi(theta - state.theta_prev)
        if phase_mode == "lifted":
            r_clip = np.clip(r, -state.pi_a, state.pi_a)
            state.theta_R += r_clip
        else:
            state.theta_R = theta.copy()
        state.theta_prev = theta.copy()

        w = np.round(state.theta_R / (2 * math.pi)).astype(int)
        dW = np.abs(w - state.w_prev).astype(float)
        state.w_prev = w.copy()

        b_edges = np.where((w % 2) == 0, 1, -1)
        b = 1 if int(np.sum(b_edges)) >= 0 else -1
        flip = int(b != state.b_prev)
        state.b_prev = b
        state.flip_count += flip

        Re_inv = np.real(1.0 / (state.Gc + 1e-18))
        t1 = float(np.sum(np.abs(I) ** 2 / max(1e-12, ent.Tij) * np.maximum(0.0, Re_inv)))
        t2 = float(ent.kappa_slip * np.sum(dW))
        t3 = float(-ent.gamma * (state.S - ent.S_eq))
        state.S = max(0.0, state.S + dt * (t1 + t2 + t3))

        if adaptive_pi:
            dpi = ruler.alpha_pi * state.S - ruler.mu_pi * (state.pi_a - ruler.pi0)
            state.pi_a = float(np.clip(state.pi_a + dt * dpi, ruler.pi_min, ruler.pi_max))

        aS = _alpha_G(state.S, eg)
        mS = _mu_G(state.S, eg)
        dGc = aS * np.abs(I) * np.exp(1j * state.theta_R) - mS * state.Gc
        if eg.lambda_s > 0:
            sup = np.sin(state.theta_R / (2 * state.pi_a + 1e-18)) ** 2
            dGc -= eg.lambda_s * sup * state.Gc
        dGc += 1e-6 * (rng.normal(size=m) + 1j * rng.normal(size=m))
        state.Gc += dt * dGc

        Re = np.clip(state.Gc.real, eg.G_min, eg.G_max)
        Im = np.clip(state.Gc.imag, -eg.G_imag_max, eg.G_imag_max)
        state.Gc = Re + 1j * Im
        if eg.budget_re is not None:
            sRe = float(np.sum(state.Gc.real))
            if sRe > eg.budget_re > 0:
                state.Gc *= eg.budget_re / sRe

        G_h[k] = state.Gc
        I_h[k] = I
        phi_h[k] = phi
        tR_h[k] = state.theta_R
        th_h[k] = theta
        w_h[k] = w
        dW_h[k] = dW
        S_h[k] = state.S
        pi_h[k] = state.pi_a
        flip_h[k] = flip
        rb_h[k] = state.flip_count / (k + 1)

    return {
        "t": t, "Gc": G_h, "I": I_h, "phi": phi_h,
        "theta_R_e": tR_h, "theta_e": th_h, "w_e": w_h, "dW_e": dW_h,
        "S": S_h, "pi_a": pi_h, "flip": flip_h, "r_b": rb_h,
        "solve_info": info_h, "edges": np.array(graph.edges, dtype=int),
        "final_state": state,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def effective_admittance(phi, source, sink, eps=1e-12):
    return 1.0 / max(eps, abs(phi[source] - phi[sink]))


def time_series_effective_admittance(phi_h, source, sink):
    return np.array([effective_admittance(phi_h[k], source, sink) for k in range(phi_h.shape[0])])


def boundary_current_fraction(I, edge_meta):
    den = float(np.sum(np.abs(I))) + 1e-12
    return sum(abs(v) for v, em in zip(I, edge_meta) if em.is_boundary) / den


def bond_dimerization(Gc, edge_meta):
    intra = [Gc[i].real for i, em in enumerate(edge_meta) if em.bond_type == "intra"]
    inter = [Gc[i].real for i, em in enumerate(edge_meta) if em.bond_type == "inter"]
    return float(np.mean(inter) - np.mean(intra))


def slip_density(dW_hist):
    return np.mean(np.abs(dW_hist), axis=1)


def summarize_recovery(out, bench, damage_time, settle_window=5.0):
    t = out["t"]
    Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
    Bfrac = np.array([boundary_current_fraction(out["I"][k], bench.edge_meta)
                       for k in range(len(t))])
    Dimer = np.array([bond_dimerization(out["Gc"][k], bench.edge_meta)
                       for k in range(len(t))])

    pre = (t >= max(0, damage_time - settle_window)) & (t < damage_time)
    post = t >= damage_time + settle_window
    if not np.any(pre):
        pre = t < damage_time
    if not np.any(post):
        post = t >= damage_time

    def _a(arr, mask):
        return float(np.mean(arr[mask])) if np.any(mask) else 0.0

    pre_Y, post_Y = _a(Yeff, pre), _a(Yeff, post)
    return {
        "Yeff_pre": pre_Y, "Yeff_post": post_Y,
        "Yeff_recovery_ratio": post_Y / max(1e-12, pre_Y),
        "boundary_pre": _a(Bfrac, pre), "boundary_post": _a(Bfrac, post),
        "dimer_pre": _a(Dimer, pre), "dimer_post": _a(Dimer, post),
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fails": int(np.sum(out["solve_info"] != 0)),
    }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def run_recovery_protocol(
    n_cells=20, T=60.0, dt=0.05, seed=0,
    damage_time=20.0, damage_edge_idx=None, damage_factor=0.15,
    phase_mode="lifted", adaptive_pi=True,
    eg=None, ent=None, ruler=None,
):
    bench = ssh_chain_graph(n_cells)
    if eg is None:
        eg = SSHParams(alpha0=1.0, S_c=1.0, dS=0.35, mu0=0.55, S0=1.0,
                       budget_re=12.0, lambda_s=0.10)
    if ent is None:
        ent = SSHEntropyParams()
    if ruler is None:
        ruler = SSHRulerParams()

    G0 = ssh_initial_G(bench.edge_meta)
    state0 = make_initial_state(bench.graph, G0=G0, S0=ent.S_init, pi0=ruler.pi_init)

    if damage_edge_idx is None:
        damage_edge_idx = [0, 1, 2] if len(bench.graph.edges) >= 3 else [0]

    interventions = [
        {"time": float(damage_time), "type": "scale_edges",
         "edge_idx": damage_edge_idx, "factor": damage_factor},
    ]
    out = simulate(
        bench.graph, bench.source, bench.sink,
        T=T, dt=dt, seed=seed, eg=eg, ent=ent, ruler=ruler,
        state0=state0, phase_mode=phase_mode,
        adaptive_pi=adaptive_pi, interventions=interventions,
    )
    return bench, out


def compare_ablations(
    n_cells=20, T=60.0, dt=0.05, seed=0, damage_time=20.0,
):
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }
    results = {}
    for name, cfg in configs.items():
        bench, out = run_recovery_protocol(
            n_cells=n_cells, T=T, dt=dt, seed=seed,
            damage_time=damage_time, **cfg,
        )
        summ = summarize_recovery(out, bench, damage_time)
        results[name] = (bench, out, summ)
    return results
