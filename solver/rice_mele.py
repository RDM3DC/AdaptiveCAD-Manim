"""Rice–Mele chain benchmark for the EGATL simulator.

SSH with broken inversion symmetry via staggered onsite potentials:

    H = sum_n (t + (-1)^n delta) c†_n c_{n+1} + h.c.
      + sum_n (-1)^n Delta/2 c†_n c_n

where delta is the dimerisation and Delta is the sublattice potential.

Unlike SSH, the Zak phase sweeps *continuously* between 0 and pi as
(delta, Delta) trace a loop around the origin.  This makes it the
canonical test-bed for the adaptive phase ruler pi_a tracking smooth
topological crossovers (not just sharp transitions).

Uses scalar complex admittances G_e evolved by the EGATL law.
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
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class RiceMeleEdgeMeta:
    bond_type: str          # "intra" or "inter"
    sublattice_u: str       # "A" or "B"
    sublattice_v: str
    is_boundary: bool


@dataclass
class RiceMeleBenchmark:
    n_nodes: int
    edges: List[Tuple[int, int]]
    edge_meta: List[RiceMeleEdgeMeta]
    source: int
    sink: int
    n_cells: int
    sublattice: np.ndarray   # 0=A, 1=B per node
    delta: float              # dimerisation
    stagger: float            # sublattice potential Delta


@dataclass
class RiceMeleParams:
    alpha0: float = 1.0
    S_c: float = 1.0
    dS: float = 0.30
    mu0: float = 0.50
    S0: float = 1.0
    G_min: float = 1e-3
    G_max: float = 50.0
    G_imag_max: float = 50.0
    budget_re: Optional[float] = 12.0
    lambda_s: float = 0.10


@dataclass
class RiceMeleEntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.20
    kappa_slip: float = 0.15
    Tij: float = 1.0


@dataclass
class RiceMeleRulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.30
    mu_pi: float = 0.20
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


@dataclass
class RiceMeleState:
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


def _grounded_gmres(M, b, ground, x0, rtol=1e-10, maxiter=2000):
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
# Lattice builder
# ---------------------------------------------------------------------------

def rice_mele_chain(
    n_cells: int = 20,
    t_base: float = 1.0,
    delta: float = 0.3,
    stagger: float = 0.4,
) -> Tuple[RiceMeleBenchmark, np.ndarray]:
    """Build a 1D Rice–Mele chain.

    Node layout:  A0 -- B0 -- A1 -- B1 -- ...
                  |-intra-|  |--inter--|  |-intra-|

    Intra-cell hop = t_base + delta   (strong)
    Inter-cell hop = t_base - delta   (weak)

    Sublattice onsite potential ±stagger/2 is encoded as a
    small imaginary shift instead of modifying diagonal, so
    it nudges EGATL phases without breaking the current-network
    structure.

    Returns (benchmark, G0).
    """
    n = 2 * n_cells
    edges = []
    meta = []
    sublattice = np.zeros(n, dtype=int)
    for c in range(n_cells):
        sublattice[2*c] = 0      # A
        sublattice[2*c + 1] = 1  # B

    # Intra-cell bonds
    for c in range(n_cells):
        u, v = 2*c, 2*c + 1
        bnd = (c == 0 or c == n_cells - 1)
        edges.append((u, v))
        meta.append(RiceMeleEdgeMeta("intra", "A", "B", bnd))

    # Inter-cell bonds
    for c in range(n_cells - 1):
        u, v = 2*c + 1, 2*(c + 1)
        bnd = False
        edges.append((u, v))
        meta.append(RiceMeleEdgeMeta("inter", "B", "A", bnd))

    # Initial G
    m = len(edges)
    G0 = np.zeros(m, dtype=complex)
    for i in range(m):
        em = meta[i]
        if em.bond_type == "intra":
            G0[i] = t_base + delta
        else:
            G0[i] = t_base - delta
        # Staggered potential as small imaginary component
        G0[i] += 1j * stagger * 0.1 * (1 if em.sublattice_u == "A" else -1)

    bench = RiceMeleBenchmark(
        n_nodes=n, edges=edges, edge_meta=meta,
        source=0, sink=n - 1,
        n_cells=n_cells, sublattice=sublattice,
        delta=delta, stagger=stagger,
    )
    return bench, G0


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def make_initial_state(n_nodes, G0=None, m=None, S0=0.5, pi0=math.pi):
    if G0 is None:
        G0 = np.ones(m, dtype=complex)
    return RiceMeleState(
        Gc=np.array(G0, dtype=complex).copy(), S=float(S0), pi_a=float(pi0),
        theta_R=np.zeros(len(G0)), theta_prev=np.zeros(len(G0)),
        w_prev=np.zeros(len(G0), dtype=int),
        phi_prev=np.zeros(n_nodes, dtype=complex),
    )


def clone_state(s):
    return RiceMeleState(
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
        elif kind == "reset_entropy":
            state.S = float(ev.get("value", state.S))
        elif kind == "set_pi_a":
            state.pi_a = float(ev.get("value", state.pi_a))
        ev["done"] = True


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def simulate(
    bench, source, sink,
    T=60.0, dt=0.05, seed=0,
    eg=None, ent=None, ruler=None,
    state0=None,
    phase_mode="lifted", adaptive_pi=True,
    interventions=None,
):
    if eg is None:
        eg = RiceMeleParams()
    if ent is None:
        ent = RiceMeleEntropyParams()
    if ruler is None:
        ruler = RiceMeleRulerParams()

    rng = np.random.default_rng(seed)
    m = len(bench.edges)
    n = bench.n_nodes
    K = int(np.ceil(T / dt)) + 1
    t = np.linspace(0, T, K)

    state = make_initial_state(n, m=m, S0=ent.S_init, pi0=ruler.pi_init) if state0 is None else clone_state(state0)
    local_iv = None if interventions is None else copy.deepcopy(interventions)

    G_h = np.zeros((K, m), dtype=complex)
    I_h = np.zeros((K, m), dtype=complex)
    phi_h = np.zeros((K, n), dtype=complex)
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

        bvec = np.zeros(n, dtype=complex)
        bvec[source] = 1.0
        bvec[sink] = -1.0
        M_mat = _build_nodal_matrix(n, bench.edges, state.Gc)
        phi, info = _grounded_gmres(M_mat, bvec, sink, state.phi_prev)
        state.phi_prev = phi
        info_h[k] = info

        I = np.zeros(m, dtype=complex)
        theta = np.zeros(m)
        for e, (u, v) in enumerate(bench.edges):
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
        t2_ = float(ent.kappa_slip * np.sum(dW))
        t3 = float(-ent.gamma * (state.S - ent.S_eq))
        state.S = max(0.0, state.S + dt * (t1 + t2_ + t3))

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
        "solve_info": info_h, "edges": np.array(bench.edges, dtype=int),
        "final_state": state,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def effective_admittance(phi, source, sink, eps=1e-12):
    return 1.0 / max(eps, abs(phi[source] - phi[sink]))


def bond_dimerization(Gc, edge_meta):
    intra = [abs(g) for g, em in zip(Gc, edge_meta) if em.bond_type == "intra"]
    inter = [abs(g) for g, em in zip(Gc, edge_meta) if em.bond_type == "inter"]
    if not intra or not inter:
        return 0.0
    mi = np.mean(intra)
    me = np.mean(inter)
    return float((mi - me) / (mi + me + 1e-12))


def boundary_current_fraction(I, edge_meta):
    den = float(np.sum(np.abs(I))) + 1e-12
    return sum(abs(v) for v, em in zip(I, edge_meta) if em.is_boundary) / den


def phase_imbalance(Gc, edge_meta):
    """Mean phase angle difference between intra and inter bonds.

    For Rice–Mele, this tracks how the EGATL-adapted phases deviate
    from the initial staggered potential — a proxy for Zak phase motion.
    """
    intra_phase = [np.angle(g) for g, em in zip(Gc, edge_meta) if em.bond_type == "intra"]
    inter_phase = [np.angle(g) for g, em in zip(Gc, edge_meta) if em.bond_type == "inter"]
    if not intra_phase or not inter_phase:
        return 0.0
    return float(abs(np.mean(intra_phase) - np.mean(inter_phase)))


def slip_density(dW_hist):
    return np.mean(np.abs(dW_hist), axis=1)


def summarize_recovery(out, bench, damage_time, settle_window=5.0):
    t = out["t"]
    K = len(t)
    Yeff = np.array([effective_admittance(out["phi"][k], bench.source, bench.sink)
                      for k in range(K)])
    Dim = np.array([bond_dimerization(out["Gc"][k], bench.edge_meta) for k in range(K)])
    PhImb = np.array([phase_imbalance(out["Gc"][k], bench.edge_meta) for k in range(K)])

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
        "dimerization_pre": _a(Dim, pre), "dimerization_post": _a(Dim, post),
        "phase_imbalance_pre": _a(PhImb, pre), "phase_imbalance_post": _a(PhImb, post),
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fails": int(np.sum(out["solve_info"] != 0)),
    }


# ---------------------------------------------------------------------------
# Zak-phase sweep
# ---------------------------------------------------------------------------

def zak_phase_sweep(
    n_cells=20, T=40.0, dt=0.05, seed=0,
    t_base=1.0, n_angles=24,
    radius=0.3,
    eg=None, ent=None, ruler=None,
):
    """Sweep (delta, Delta) on a circle of given radius in parameter space.

    Returns arrays of angle, Yeff_final, S_final, pi_a_final,
    dimerization, phase_imbalance.

    At angle=0 the system is topological (delta=radius, Delta=0).
    At angle=pi/2 it is trivial (delta=0, Delta=radius).
    """
    angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    Yeff_arr = np.zeros(n_angles)
    S_arr = np.zeros(n_angles)
    pi_arr = np.zeros(n_angles)
    dim_arr = np.zeros(n_angles)
    phi_arr = np.zeros(n_angles)

    for ia, ang in enumerate(angles):
        delta_val = radius * math.cos(ang)
        stagger_val = radius * math.sin(ang)
        bench, G0 = rice_mele_chain(n_cells, t_base, delta_val, stagger_val)
        out = simulate(
            bench, bench.source, bench.sink,
            T=T, dt=dt, seed=seed,
            eg=eg, ent=ent, ruler=ruler,
        )
        Yeff_arr[ia] = effective_admittance(
            out["phi"][-1], bench.source, bench.sink)
        S_arr[ia] = out["S"][-1]
        pi_arr[ia] = out["pi_a"][-1]
        dim_arr[ia] = bond_dimerization(out["Gc"][-1], bench.edge_meta)
        phi_arr[ia] = phase_imbalance(out["Gc"][-1], bench.edge_meta)

    return {
        "angles": angles,
        "Yeff_final": Yeff_arr,
        "S_final": S_arr,
        "pi_a_final": pi_arr,
        "dimerization": dim_arr,
        "phase_imbalance": phi_arr,
    }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def run_recovery_protocol(
    n_cells=20, t_base=1.0, delta=0.3, stagger=0.4,
    T=60.0, dt=0.05, seed=0,
    damage_time=20.0, damage_factor=0.05,
    phase_mode="lifted", adaptive_pi=True,
    eg=None, ent=None, ruler=None,
):
    bench, G0 = rice_mele_chain(n_cells, t_base, delta, stagger)
    if eg is None:
        eg = RiceMeleParams()
    if ent is None:
        ent = RiceMeleEntropyParams()
    if ruler is None:
        ruler = RiceMeleRulerParams()

    state0 = make_initial_state(bench.n_nodes, G0=G0, S0=ent.S_init, pi0=ruler.pi_init)

    # Damage central intra-cell bonds
    mid = n_cells // 2
    damage_idx = []
    for i, (u, v) in enumerate(bench.edges):
        cell_u = u // 2
        if abs(cell_u - mid) <= 2 and bench.edge_meta[i].bond_type == "intra":
            damage_idx.append(i)

    interventions = [
        {"time": float(damage_time), "type": "scale_edges",
         "edge_idx": damage_idx, "factor": damage_factor},
        {"time": float(damage_time), "type": "reset_entropy", "value": 2.5},
    ]
    out = simulate(
        bench, bench.source, bench.sink,
        T=T, dt=dt, seed=seed, eg=eg, ent=ent, ruler=ruler,
        state0=state0, phase_mode=phase_mode,
        adaptive_pi=adaptive_pi, interventions=interventions,
    )
    return bench, out


def compare_ablations(
    n_cells=20, T=60.0, dt=0.05, seed=0,
    damage_time=20.0, delta=0.3, stagger=0.4,
):
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }
    results = {}
    for name, cfg in configs.items():
        bench, out = run_recovery_protocol(
            n_cells=n_cells, delta=delta, stagger=stagger,
            T=T, dt=dt, seed=seed,
            damage_time=damage_time, **cfg,
        )
        summ = summarize_recovery(out, bench, damage_time)
        results[name] = (bench, out, summ)
    return results
