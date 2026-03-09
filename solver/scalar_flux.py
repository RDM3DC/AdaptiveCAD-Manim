"""2D scalar synthetic-flux lattice benchmark for the EGATL simulator.

Harper-Hofstadter-style square lattice with complex scalar admittances:
  G_x(x,y) = g_mag * exp(i 2pi alpha y)   (Landau gauge)
  G_y(x,y) = g_mag

evolved by the same EGATL law as the SSH module.

Also includes a Hofstadter butterfly parameter sweep:
  vary flux_alpha continuously and track how the EGATL law navigates
  the fractal gap structure.

Promoted from examples/hafc_sim2_topological_flux_complete.py
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
# Utilities
# ---------------------------------------------------------------------------

def _wrap_to_pi(x):
    a = (x + math.pi) % (2 * math.pi) - math.pi
    return np.where(a <= -math.pi, a + 2 * math.pi, a)


def _logistic(x):
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Graph primitives (same as ssh.py — intentionally independent)
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]


@dataclass
class EdgeMeta:
    bond_type: str
    cell: int
    is_boundary: bool


@dataclass
class FluxBenchmark:
    graph: Graph
    source: int
    sink: int
    edge_meta: List[EdgeMeta]
    edge_index_by_name: Dict[Tuple[int, int], int]
    nx: int
    ny: int


# ---------------------------------------------------------------------------
# Sparse nodal solve
# ---------------------------------------------------------------------------

def _build_nodal_matrix(n, edges, Y):
    m = len(edges)
    rows = np.empty(4 * m, dtype=int)
    cols = np.empty(4 * m, dtype=int)
    data = np.empty(4 * m, dtype=Y.dtype)
    for i, (u, v) in enumerate(edges):
        k = 4 * i
        rows[k], cols[k], data[k] = u, u, Y[i]
        rows[k+1], cols[k+1], data[k+1] = v, v, Y[i]
        rows[k+2], cols[k+2], data[k+2] = u, v, -Y[i]
        rows[k+3], cols[k+3], data[k+3] = v, u, -Y[i]
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
class FluxParams:
    alpha0: float = 1.5
    S_c: float = 1.0
    dS: float = 0.35
    mu0: float = 0.55
    S0: float = 1.0
    G_min: float = 1e-3
    G_max: float = 50.0
    G_imag_max: float = 50.0
    budget_re: Optional[float] = 25.0
    lambda_s: float = 0.15


@dataclass
class FluxEntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25
    kappa_slip: float = 0.20
    Tij: float = 1.0


@dataclass
class FluxRulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


@dataclass
class FluxState:
    Gc: np.ndarray
    S: float
    pi_a: float
    theta_R: np.ndarray
    theta_prev: np.ndarray
    w_prev: np.ndarray
    phi_prev: np.ndarray
    b_prev: int = 1
    flip_count: int = 0


def _alpha_G(S, p):
    return p.alpha0 * _logistic(-(S - p.S_c) / max(1e-12, p.dS))


def _mu_G(S, p):
    return p.mu0 * (S / max(1e-12, p.S0))


def make_initial_state(graph, G0=None, S0=0.5, pi0=math.pi):
    m = len(graph.edges)
    if G0 is None:
        G0 = np.ones(m, dtype=complex)
    return FluxState(
        Gc=np.array(G0, dtype=complex).copy(), S=float(S0), pi_a=float(pi0),
        theta_R=np.zeros(m), theta_prev=np.zeros(m),
        w_prev=np.zeros(m, dtype=int),
        phi_prev=np.zeros(graph.n, dtype=complex),
    )


def clone_state(s):
    return FluxState(
        Gc=s.Gc.copy(), S=s.S, pi_a=s.pi_a,
        theta_R=s.theta_R.copy(), theta_prev=s.theta_prev.copy(),
        w_prev=s.w_prev.copy(), phi_prev=s.phi_prev.copy(),
        b_prev=s.b_prev, flip_count=s.flip_count,
    )


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
# Lattice builder
# ---------------------------------------------------------------------------

def flux_lattice_graph(nx: int, ny: int) -> FluxBenchmark:
    """2D square lattice with open boundaries, scalar admittances."""
    if nx < 3 or ny < 3:
        raise ValueError("Lattice must be at least 3x3")

    edges: List[Tuple[int, int]] = []
    meta: List[EdgeMeta] = []
    eidx: Dict[Tuple[int, int], int] = {}

    def _add(u, v, btype, bnd=False):
        i = len(edges)
        edges.append((u, v))
        meta.append(EdgeMeta(bond_type=btype, cell=0, is_boundary=bnd))
        eidx[(u, v)] = i
        eidx[(v, u)] = i

    for y in range(ny):
        for x in range(nx):
            u = x + y * nx
            if x < nx - 1:
                v = (x + 1) + y * nx
                _add(u, v, f"x_bond_y{y}", bnd=(y == 0 or y == ny - 1))
            if y < ny - 1:
                v = x + (y + 1) * nx
                _add(u, v, "y_bond", bnd=(x == 0 or x == nx - 1))

    source = (ny - 1) * nx          # top-left
    sink = (nx - 1) + (ny - 1) * nx  # top-right

    return FluxBenchmark(
        graph=Graph(n=nx * ny, edges=edges),
        source=source, sink=sink,
        edge_meta=meta, edge_index_by_name=eidx,
        nx=nx, ny=ny,
    )


def flux_initial_G(bench, g_mag=1.0, flux_alpha=0.25):
    """Landau-gauge synthetic flux initialisation."""
    nx = bench.nx
    out = np.zeros(len(bench.graph.edges), dtype=complex)
    for i, (u, v) in enumerate(bench.graph.edges):
        uy, ux = divmod(u, nx)
        vy, _ = divmod(v, nx)
        if uy == vy:  # x-bond
            out[i] = g_mag * np.exp(1j * 2 * math.pi * flux_alpha * uy)
        else:
            out[i] = g_mag + 0j
    return out


# ---------------------------------------------------------------------------
# Simulator (same EGATL law, scalar version)
# ---------------------------------------------------------------------------

def simulate(
    graph, source, sink,
    T=80.0, dt=0.05, seed=0,
    eg=None, ent=None, ruler=None,
    state0=None,
    phase_mode="lifted", adaptive_pi=True,
    interventions=None,
):
    if eg is None:
        eg = FluxParams()
    if ent is None:
        ent = FluxEntropyParams()
    if ruler is None:
        ruler = FluxRulerParams()

    rng = np.random.default_rng(seed)
    m = len(graph.edges)
    K = int(np.ceil(T / dt)) + 1
    t = np.linspace(0, T, K)

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
        t1 = float(np.sum(np.abs(I)**2 / max(1e-12, ent.Tij) * np.maximum(0.0, Re_inv)))
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


def top_edge_current_fraction(I, edge_meta, ny):
    den = float(np.sum(np.abs(I))) + 1e-12
    prefix = f"x_bond_y{ny - 1}"
    return sum(abs(v) for v, em in zip(I, edge_meta) if em.bond_type.startswith(prefix)) / den


def slip_density(dW_hist):
    return np.mean(np.abs(dW_hist), axis=1)


def summarize_recovery(out, bench, damage_time, settle_window=5.0):
    t = out["t"]
    ny = bench.ny
    Yeff = time_series_effective_admittance(out["phi"], bench.source, bench.sink)
    perim = np.array([boundary_current_fraction(out["I"][k], bench.edge_meta)
                       for k in range(len(t))])
    top = np.array([top_edge_current_fraction(out["I"][k], bench.edge_meta, ny)
                     for k in range(len(t))])
    slips = slip_density(out["dW_e"])

    pre = (t >= max(0, damage_time - settle_window)) & (t < damage_time)
    post = t >= damage_time + settle_window
    if not np.any(pre):
        pre = t < damage_time
    if not np.any(post):
        post = t >= damage_time

    def _a(arr, m):
        return float(np.mean(arr[m])) if np.any(m) else 0.0

    pre_Y, post_Y = _a(Yeff, pre), _a(Yeff, post)
    return {
        "Yeff_pre": pre_Y, "Yeff_post": post_Y,
        "Yeff_recovery_ratio": post_Y / max(1e-12, pre_Y),
        "perimeter_pre": _a(perim, pre), "perimeter_post": _a(perim, post),
        "top_edge_pre": _a(top, pre), "top_edge_post": _a(top, post),
        "post_slip_density": _a(slips, post),
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fails": int(np.sum(out["solve_info"] != 0)),
    }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def run_recovery_protocol(
    nx=8, ny=8, T=80.0, dt=0.05, seed=0,
    damage_time=30.0, damage_factor=1e-6,
    flux_alpha=0.25, g_mag=1.0,
    phase_mode="lifted", adaptive_pi=True,
    eg=None, ent=None, ruler=None,
):
    bench = flux_lattice_graph(nx, ny)
    if eg is None:
        eg = FluxParams()
    if ent is None:
        ent = FluxEntropyParams()
    if ruler is None:
        ruler = FluxRulerParams()

    G0 = flux_initial_G(bench, g_mag=g_mag, flux_alpha=flux_alpha)
    state0 = make_initial_state(bench.graph, G0=G0, S0=ent.S_init, pi0=ruler.pi_init)

    # Damage middle of top edge
    target_node = nx // 2 + (ny - 1) * nx
    damage_idx = [i for i, (u, v) in enumerate(bench.graph.edges)
                  if u == target_node or v == target_node]

    interventions = [
        {"time": float(damage_time), "type": "scale_edges",
         "edge_idx": damage_idx, "factor": damage_factor},
        {"time": float(damage_time), "type": "reset_entropy", "value": 3.0},
    ]
    out = simulate(
        bench.graph, bench.source, bench.sink,
        T=T, dt=dt, seed=seed, eg=eg, ent=ent, ruler=ruler,
        state0=state0, phase_mode=phase_mode,
        adaptive_pi=adaptive_pi, interventions=interventions,
    )
    return bench, out


def compare_ablations(
    nx=8, ny=8, T=80.0, dt=0.05, seed=0,
    damage_time=30.0, flux_alpha=0.25,
):
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }
    results = {}
    for name, cfg in configs.items():
        bench, out = run_recovery_protocol(
            nx=nx, ny=ny, T=T, dt=dt, seed=seed,
            damage_time=damage_time, flux_alpha=flux_alpha, **cfg,
        )
        summ = summarize_recovery(out, bench, damage_time)
        results[name] = (bench, out, summ)
    return results


# ---------------------------------------------------------------------------
# Hofstadter butterfly sweep
# ---------------------------------------------------------------------------

def hofstadter_sweep(
    nx=6, ny=6, T=30.0, dt=0.1,
    n_alpha=21, seed=0,
    phase_mode="lifted", adaptive_pi=True,
):
    """Sweep flux_alpha from 0 to 0.5 and record final-state metrics.

    Returns a dict with arrays indexed by sweep step:
      alpha_vals, Yeff_final, S_final, pi_a_final, slip_final
    """
    alphas = np.linspace(0.0, 0.5, n_alpha)
    Yeff = np.zeros(n_alpha)
    S_out = np.zeros(n_alpha)
    pi_out = np.zeros(n_alpha)
    slip_out = np.zeros(n_alpha)

    for j, alpha in enumerate(alphas):
        bench = flux_lattice_graph(nx, ny)
        G0 = flux_initial_G(bench, flux_alpha=alpha)
        state0 = make_initial_state(bench.graph, G0=G0)
        out = simulate(
            bench.graph, bench.source, bench.sink,
            T=T, dt=dt, seed=seed,
            state0=state0, phase_mode=phase_mode,
            adaptive_pi=adaptive_pi,
        )
        Yeff[j] = effective_admittance(out["phi"][-1], bench.source, bench.sink)
        S_out[j] = float(out["S"][-1])
        pi_out[j] = float(out["pi_a"][-1])
        slip_out[j] = float(np.mean(slip_density(out["dW_e"])[-max(1, len(out["t"])//5):]))

    return {
        "alpha_vals": alphas,
        "Yeff_final": Yeff,
        "S_final": S_out,
        "pi_a_final": pi_out,
        "slip_final": slip_out,
    }
