"""Haldane honeycomb lattice benchmark for the EGATL simulator.

The original Chern insulator:  honeycomb lattice with real nearest-neighbour
hops t1 and complex next-nearest-neighbour hops t2 * exp(±i phi).

    H = t1 sum_<ij> c†_i c_j
      + t2 sum_<<ij>> exp(i nu_ij phi) c†_i c_j
      + M sum_i xi_i c†_i c_i

where nu_ij = ±1 for clockwise/counter-clockwise NNN and xi_i = ±1 for A/B
sublattice (staggered mass).

Chern number C = ±1 when |M/t2| < 3√3 sin(phi).

Uses scalar complex admittances G_e on each bond (NN + NNN), evolved by
the EGATL law.  The complex Haldane phase on NNN bonds is encoded in the
initial G_e values, then the EGATL dynamics adapt magnitudes + phases
autonomously.
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
class HaldaneEdgeMeta:
    bond_type: str          # "nn" or "nnn"
    sublattice_u: str       # "A" or "B"
    sublattice_v: str
    is_boundary: bool


@dataclass
class HaldaneBenchmark:
    n_nodes: int
    edges: List[Tuple[int, int]]
    edge_meta: List[HaldaneEdgeMeta]
    source: int
    sink: int
    nx: int
    ny: int
    sublattice: np.ndarray  # 0=A, 1=B per node


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
class HaldaneEGATLParams:
    alpha0: float = 1.5
    S_c: float = 1.0
    dS: float = 0.35
    mu0: float = 0.55
    S0: float = 1.0
    G_min: float = 1e-3
    G_max: float = 50.0
    G_imag_max: float = 50.0
    budget_re: Optional[float] = 40.0
    lambda_s: float = 0.12


@dataclass
class HaldaneEntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25
    kappa_slip: float = 0.18
    Tij: float = 1.0


@dataclass
class HaldaneRulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


@dataclass
class HaldaneState:
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


def make_initial_state(n_nodes, G0=None, m=None, S0=0.5, pi0=math.pi):
    if G0 is None:
        G0 = np.ones(m, dtype=complex)
    return HaldaneState(
        Gc=np.array(G0, dtype=complex).copy(), S=float(S0), pi_a=float(pi0),
        theta_R=np.zeros(len(G0)), theta_prev=np.zeros(len(G0)),
        w_prev=np.zeros(len(G0), dtype=int),
        phi_prev=np.zeros(n_nodes, dtype=complex),
    )


def clone_state(s):
    return HaldaneState(
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
# Honeycomb lattice builder
# ---------------------------------------------------------------------------

def _node_id(x, y, sub, nx):
    """Node index for site (x, y, sublattice).  sub=0 → A, sub=1 → B."""
    return (x + y * nx) * 2 + sub


def haldane_lattice(
    nx: int = 6,
    ny: int = 6,
    t1: float = 1.0,
    t2: float = 0.3,
    haldane_phi: float = math.pi / 2,
    staggered_mass: float = 0.0,
) -> Tuple[HaldaneBenchmark, np.ndarray]:
    """Build a honeycomb lattice with NN + NNN bonds.

    Returns (benchmark, G0) where G0 encodes the Haldane phases.

    Layout: rectangular unit cells at (x, y) each with an A and B site.
    NN bonds: A_xy -- B_xy, A_xy -- B_{x-1,y}, A_xy -- B_{x,y-1}
    NNN bonds: A↔A and B↔B within the honeycomb.
    """
    if nx < 3 or ny < 3:
        raise ValueError("Need at least 3x3 unit cells.")

    n_nodes = 2 * nx * ny
    edges: List[Tuple[int, int]] = []
    meta: List[HaldaneEdgeMeta] = []
    sublattice = np.zeros(n_nodes, dtype=int)
    added = set()

    for y in range(ny):
        for x in range(nx):
            sublattice[_node_id(x, y, 0, nx)] = 0  # A
            sublattice[_node_id(x, y, 1, nx)] = 1  # B

    def _add(u, v, btype, sub_u, sub_v, bnd=False):
        key = (min(u, v), max(u, v))
        if key in added:
            return
        added.add(key)
        edges.append((u, v))
        meta.append(HaldaneEdgeMeta(bond_type=btype, sublattice_u=sub_u,
                                     sublattice_v=sub_v, is_boundary=bnd))

    # NN bonds
    for y in range(ny):
        for x in range(nx):
            a = _node_id(x, y, 0, nx)
            b = _node_id(x, y, 1, nx)
            bnd = (x == 0 or x == nx - 1 or y == 0 or y == ny - 1)
            _add(a, b, "nn", "A", "B", bnd)
            if x > 0:
                b2 = _node_id(x - 1, y, 1, nx)
                _add(a, b2, "nn", "A", "B", bnd or x - 1 == 0)
            if y > 0:
                b3 = _node_id(x, y - 1, 1, nx)
                _add(a, b3, "nn", "A", "B", bnd or y - 1 == 0)

    # NNN bonds (A-A and B-B, 6 directions per sublattice)
    nnn_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for y in range(ny):
        for x in range(nx):
            for sub_idx, sub_name in [(0, "A"), (1, "B")]:
                u = _node_id(x, y, sub_idx, nx)
                for dx, dy in nnn_offsets:
                    nx2, ny2 = x + dx, y + dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny:
                        v = _node_id(nx2, ny2, sub_idx, nx)
                        bnd = (x == 0 or x == nx - 1 or y == 0 or y == ny - 1
                               or nx2 == 0 or nx2 == nx - 1 or ny2 == 0 or ny2 == ny - 1)
                        _add(u, v, "nnn", sub_name, sub_name, bnd)

    # Initial G: real t1 for NN, complex t2*exp(±i*phi) for NNN
    G0 = np.zeros(len(edges), dtype=complex)
    for i, (u, v) in enumerate(edges):
        em = meta[i]
        if em.bond_type == "nn":
            G0[i] = t1 + staggered_mass * (1.0 if em.sublattice_u == "A" else -1.0) * 0.1
        else:
            # NNN orientation sign: +phi for A→A clockwise, −phi for B→B
            # Simplified: A-A gets +phi, B-B gets -phi
            sign = 1.0 if em.sublattice_u == "A" else -1.0
            G0[i] = t2 * np.exp(1j * sign * haldane_phi)

    # Source/sink: A-site bottom-left → B-site top-right
    source = _node_id(0, 0, 0, nx)
    sink = _node_id(nx - 1, ny - 1, 1, nx)

    bench = HaldaneBenchmark(
        n_nodes=n_nodes, edges=edges, edge_meta=meta,
        source=source, sink=sink,
        nx=nx, ny=ny, sublattice=sublattice,
    )
    return bench, G0


# ---------------------------------------------------------------------------
# Chern number (analytical for Haldane model)
# ---------------------------------------------------------------------------

def haldane_chern_number(t1, t2, haldane_phi, M_mass, nk=31):
    """Discretised Chern number for the lower band of the Haldane Hamiltonian.

    H(k) = d(k) . sigma with:
      d_x = t1 (1 + cos(k.a1) + cos(k.a2))
      d_y = t1 (sin(k.a1) + sin(k.a2))
      d_z = M - 2 t2 sin(phi) (sin(k.a1) - sin(k.a2) - sin(k.a1-k.a2))
    """
    # Honeycomb reciprocal lattice
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, math.sqrt(3) / 2])

    ks = np.linspace(-math.pi, math.pi, nk, endpoint=False)
    u = np.zeros((nk, nk, 2), dtype=complex)

    for ix, k1 in enumerate(ks):
        for iy, k2 in enumerate(ks):
            ka1 = k1
            ka2 = k2

            dx = t1 * (1 + math.cos(ka1) + math.cos(ka2))
            dy = t1 * (math.sin(ka1) + math.sin(ka2))
            dz = (M_mass - 2 * t2 * math.sin(haldane_phi)
                  * (math.sin(ka1) - math.sin(ka2) - math.sin(ka1 - ka2)))

            H = np.array([
                [dz, dx - 1j * dy],
                [dx + 1j * dy, -dz],
            ], dtype=complex)

            vals, vecs = np.linalg.eigh(H)
            v = vecs[:, np.argmin(vals)]
            nrm = np.linalg.norm(v)
            u[ix, iy] = v / nrm if nrm > 1e-15 else np.array([1, 0], dtype=complex)

    total = 0.0
    for ix in range(nk):
        for iy in range(nk):
            ix1 = (ix + 1) % nk
            iy1 = (iy + 1) % nk
            U1 = np.vdot(u[ix, iy], u[ix1, iy])
            U2 = np.vdot(u[ix1, iy], u[ix1, iy1])
            U3 = np.vdot(u[ix, iy1], u[ix1, iy1])
            U4 = np.vdot(u[ix, iy], u[ix, iy1])
            for Uk in [U1, U2, U3, U4]:
                pass
            U1 /= max(1e-15, abs(U1))
            U2 /= max(1e-15, abs(U2))
            U3 /= max(1e-15, abs(U3))
            U4 /= max(1e-15, abs(U4))
            total += np.log(U1 * U2 / (U3 * U4)).imag

    return float(total / (2 * math.pi))


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
        eg = HaldaneEGATLParams()
    if ent is None:
        ent = HaldaneEntropyParams()
    if ruler is None:
        ruler = HaldaneRulerParams()

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


def boundary_current_fraction(I, edge_meta):
    den = float(np.sum(np.abs(I))) + 1e-12
    return sum(abs(v) for v, em in zip(I, edge_meta) if em.is_boundary) / den


def nn_vs_nnn_ratio(I, edge_meta):
    """Ratio of NNN current to total — indicates chiral edge transport."""
    den = float(np.sum(np.abs(I))) + 1e-12
    nnn = sum(abs(v) for v, em in zip(I, edge_meta) if em.bond_type == "nnn")
    return nnn / den


def slip_density(dW_hist):
    return np.mean(np.abs(dW_hist), axis=1)


def summarize_recovery(out, bench, damage_time, settle_window=5.0):
    t = out["t"]
    K = len(t)
    Yeff = np.array([effective_admittance(out["phi"][k], bench.source, bench.sink)
                      for k in range(K)])
    Bfrac = np.array([boundary_current_fraction(out["I"][k], bench.edge_meta)
                       for k in range(K)])
    NNNr = np.array([nn_vs_nnn_ratio(out["I"][k], bench.edge_meta)
                      for k in range(K)])

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
        "boundary_pre": _a(Bfrac, pre), "boundary_post": _a(Bfrac, post),
        "nnn_ratio_pre": _a(NNNr, pre), "nnn_ratio_post": _a(NNNr, post),
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fails": int(np.sum(out["solve_info"] != 0)),
    }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def run_recovery_protocol(
    nx=6, ny=6, t1=1.0, t2=0.3, haldane_phi=math.pi/2,
    staggered_mass=0.0,
    T=60.0, dt=0.05, seed=0,
    damage_time=20.0, damage_factor=0.05,
    phase_mode="lifted", adaptive_pi=True,
    eg=None, ent=None, ruler=None,
):
    bench, G0 = haldane_lattice(nx, ny, t1, t2, haldane_phi, staggered_mass)
    if eg is None:
        eg = HaldaneEGATLParams()
    if ent is None:
        ent = HaldaneEntropyParams()
    if ruler is None:
        ruler = HaldaneRulerParams()

    state0 = make_initial_state(bench.n_nodes, G0=G0, S0=ent.S_init, pi0=ruler.pi_init)

    # Damage NNN bonds near the centre
    cx, cy = nx // 2, ny // 2
    target_nodes = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tx, ty = cx + dx, cy + dy
            if 0 <= tx < nx and 0 <= ty < ny:
                target_nodes.add(_node_id(tx, ty, 0, nx))
                target_nodes.add(_node_id(tx, ty, 1, nx))

    damage_idx = [i for i, (u, v) in enumerate(bench.edges)
                  if u in target_nodes or v in target_nodes]

    interventions = [
        {"time": float(damage_time), "type": "scale_edges",
         "edge_idx": damage_idx, "factor": damage_factor},
        {"time": float(damage_time), "type": "reset_entropy", "value": 3.0},
    ]
    out = simulate(
        bench, bench.source, bench.sink,
        T=T, dt=dt, seed=seed, eg=eg, ent=ent, ruler=ruler,
        state0=state0, phase_mode=phase_mode,
        adaptive_pi=adaptive_pi, interventions=interventions,
    )
    return bench, out


def compare_ablations(
    nx=6, ny=6, T=60.0, dt=0.05, seed=0,
    damage_time=20.0, t2=0.3, haldane_phi=math.pi/2,
):
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }
    results = {}
    for name, cfg in configs.items():
        bench, out = run_recovery_protocol(
            nx=nx, ny=ny, t2=t2, haldane_phi=haldane_phi,
            T=T, dt=dt, seed=seed,
            damage_time=damage_time, **cfg,
        )
        summ = summarize_recovery(out, bench, damage_time)
        results[name] = (bench, out, summ)
    return results
