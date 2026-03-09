"""Kitaev chain benchmark for the EGATL simulator.

1D p-wave superconductor with Majorana edge modes.
Uses 2x2 Nambu (particle-hole) blocks on a 1D chain:

    H_BdG = sum_i [ -mu c†_i c_i
                     + t (c†_i c_{i+1} + h.c.)
                     + Delta (c_i c_{i+1} + h.c.) ]

In the Nambu basis psi_i = (c_i, c†_i)^T the bond block is:

    T_hop = [[ -t,   Delta ],
             [ -Delta*, t*  ]]

and onsite:

    H_onsite = [[ -mu/2 + eta,   0        ],
                [  0,            mu/2 + eta ]]

where eta > 0 is regularisation damping.

Topological phase:  |Delta| > 0 and |mu| < 2|t|
  => Majorana zero modes at chain ends
  => edge-localised transport should survive damage + EGATL recovery

The EGATL adaptive law evolves a scalar complex multiplier g_e on each
bond, exactly as in the QWZ block model.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import gmres, spsolve


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
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class KitaevEGATLParams:
    alpha0: float = 1.5
    S_c: float = 1.0
    dS: float = 0.35
    mu0: float = 0.55
    S0: float = 1.0
    lambda_s: float = 0.10
    g_min: float = 1e-6
    g_max: float = 5.0
    g_imag_max: float = 5.0
    budget_re: Optional[float] = 30.0


@dataclass
class KitaevEntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25
    kappa_slip: float = 0.15
    Tij: float = 1.0


@dataclass
class KitaevRulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.2
    pi_max: float = 8.0 * math.pi


def _alpha_G(S, p):
    return p.alpha0 * _logistic((S - p.S_c) / max(1e-12, p.dS))


def _mu_G(S, p):
    return p.mu0 * (1.0 + S / max(1e-12, p.S0))


# ---------------------------------------------------------------------------
# Block bond / lattice
# ---------------------------------------------------------------------------

@dataclass
class BlockBond:
    u: int
    v: int
    B: np.ndarray
    is_boundary: bool
    label: str = ""


@dataclass
class KitaevChain:
    n_sites: int
    bonds: List[BlockBond]
    source_site: int
    sink_site: int
    source_vec: np.ndarray
    sink_vec: np.ndarray
    mu_chem: float
    delta: complex
    t_hop: float
    eta: float
    onsite_block: np.ndarray  # will differ per site in general but here uniform


def build_kitaev_chain(
    n_sites: int = 20,
    mu_chem: float = 0.5,
    t_hop: float = 1.0,
    delta: complex = 1.0 + 0j,
    eta: float = 0.35,
) -> KitaevChain:
    """Build an open-boundary Kitaev chain with n_sites.

    Topological when |mu_chem| < 2|t_hop| and |delta| > 0.
    """
    if n_sites < 3:
        raise ValueError("Need at least 3 sites.")

    # Nambu bond block: T = [[-t, Delta], [-Delta*, t*]]
    T_bond = np.array([
        [-t_hop, delta],
        [-np.conj(delta), np.conj(t_hop)],
    ], dtype=complex)

    bonds: List[BlockBond] = []
    for i in range(n_sites - 1):
        bnd = (i == 0) or (i == n_sites - 2)
        bonds.append(BlockBond(
            u=i, v=i + 1, B=T_bond.copy(),
            is_boundary=bnd, label=f"bond_{i}",
        ))

    # Onsite: [[-mu/2 + eta, 0], [0, mu/2 + eta]]
    onsite = np.array([
        [-mu_chem / 2 + eta, 0],
        [0, mu_chem / 2 + eta],
    ], dtype=complex)

    # Inject equal superposition on particle + hole
    sv = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)

    return KitaevChain(
        n_sites=n_sites, bonds=bonds,
        source_site=0, sink_site=n_sites - 1,
        source_vec=sv.copy(), sink_vec=sv.copy(),
        mu_chem=mu_chem, delta=delta, t_hop=t_hop, eta=eta,
        onsite_block=onsite,
    )


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

@dataclass
class KitaevState:
    g: np.ndarray
    S: float
    pi_a: float
    theta_R: np.ndarray
    theta_prev: np.ndarray
    w_prev: np.ndarray
    phi_prev: np.ndarray
    b_prev: int = 1
    flip_count: int = 0


def make_initial_state(chain, g0=None, S0=0.5, pi0=math.pi):
    m = len(chain.bonds)
    nd = 2 * chain.n_sites
    if g0 is None:
        g0 = np.ones(m, dtype=complex)
    return KitaevState(
        g=np.array(g0, dtype=complex).copy(),
        S=float(S0), pi_a=float(pi0),
        theta_R=np.zeros(m), theta_prev=np.zeros(m),
        w_prev=np.zeros(m, dtype=int),
        phi_prev=np.zeros(nd, dtype=complex),
    )


def clone_state(s):
    return KitaevState(
        g=s.g.copy(), S=s.S, pi_a=s.pi_a,
        theta_R=s.theta_R.copy(), theta_prev=s.theta_prev.copy(),
        w_prev=s.w_prev.copy(), phi_prev=s.phi_prev.copy(),
        b_prev=s.b_prev, flip_count=s.flip_count,
    )


# ---------------------------------------------------------------------------
# Matrix assembly & solve
# ---------------------------------------------------------------------------

def _assemble_block_matrix(chain, g):
    n = chain.n_sites
    nd = 2 * n
    rows, cols, data = [], [], []

    def _add_block(ic, jc, block):
        bi, bj = 2 * ic, 2 * jc
        for a in range(2):
            for b in range(2):
                rows.append(bi + a)
                cols.append(bj + b)
                data.append(block[a, b])

    for c in range(n):
        _add_block(c, c, chain.onsite_block)

    for e, bond in enumerate(chain.bonds):
        ge = g[e]
        B = bond.B
        _add_block(bond.u, bond.u, ge * B)
        _add_block(bond.u, bond.v, -ge * B)
        _add_block(bond.v, bond.u, -np.conj(ge) * B.conj().T)
        _add_block(bond.v, bond.v, np.conj(ge) * B.conj().T)

    return coo_matrix(
        (np.array(data, dtype=complex), (rows, cols)), shape=(nd, nd)
    ).tocsr()


def _grounded_solve(M, b, grounds, x0):
    n = b.shape[0]
    mask = np.ones(n, dtype=bool)
    for g in grounds:
        mask[g] = False
    Mr = M[mask][:, mask]
    br = b[mask]
    xr = None
    info = 1
    try:
        xr = spsolve(Mr, br)
        if np.all(np.isfinite(xr)):
            info = 0
    except Exception:
        pass
    if xr is None or info != 0:
        x0r = None if x0 is None else x0[mask]
        xr, info = gmres(Mr, br, x0=x0r, rtol=1e-10, atol=0.0,
                         maxiter=2500, restart=50)
    x = np.zeros(n, dtype=complex)
    x[mask] = xr
    return x, int(info)


def _cell_slice(c):
    return slice(2 * c, 2 * c + 2)


def _bond_activity(phi, bond, ge):
    pu = phi[_cell_slice(bond.u)]
    pv = phi[_cell_slice(bond.v)]
    dphi = pu - pv
    I_vec = ge * (bond.B @ dphi)
    J = np.vdot(dphi, I_vec)
    return I_vec, J


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
        if kind == "scale_bonds":
            idx = np.asarray(ev["bond_idx"], dtype=int)
            state.g[idx] *= complex(ev.get("factor", 0.25))
        elif kind == "set_bonds":
            idx = np.asarray(ev["bond_idx"], dtype=int)
            state.g[idx] = complex(ev.get("value", 0.0))
        elif kind == "kick_phase":
            idx = np.asarray(ev["bond_idx"], dtype=int)
            state.theta_R[idx] += float(ev.get("delta", math.pi))
        elif kind == "reset_entropy":
            state.S = float(ev.get("value", state.S))
        elif kind == "set_pi_a":
            state.pi_a = float(ev.get("value", state.pi_a))
        ev["done"] = True


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def simulate(
    chain: KitaevChain,
    T=40.0, dt=0.05, seed=0,
    eg=None, ent=None, ruler=None,
    state0=None,
    phase_mode="lifted", adaptive_pi=True,
    interventions=None,
):
    if eg is None:
        eg = KitaevEGATLParams()
    if ent is None:
        ent = KitaevEntropyParams()
    if ruler is None:
        ruler = KitaevRulerParams()

    rng = np.random.default_rng(seed)
    m = len(chain.bonds)
    nd = 2 * chain.n_sites
    K = int(np.ceil(T / dt)) + 1
    t = np.linspace(0, T, K)

    state = make_initial_state(chain, S0=ent.S_init, pi0=ruler.pi_init) if state0 is None else clone_state(state0)
    local_iv = None if interventions is None else copy.deepcopy(interventions)

    g_h = np.zeros((K, m), dtype=complex)
    phi_h = np.zeros((K, nd), dtype=complex)
    J_h = np.zeros((K, m), dtype=complex)
    In_h = np.zeros((K, m))
    tR_h = np.zeros((K, m))
    th_h = np.zeros((K, m))
    w_h = np.zeros((K, m), dtype=int)
    dW_h = np.zeros((K, m))
    S_h = np.zeros(K)
    pi_h = np.zeros(K)
    flip_h = np.zeros(K, dtype=int)
    rb_h = np.zeros(K)
    info_h = np.zeros(K, dtype=int)

    sink_dof = 2 * chain.sink_site

    for k in range(K):
        t_now = float(t[k])
        _apply_interventions(t_now, state, local_iv)

        bvec = np.zeros(nd, dtype=complex)
        bvec[_cell_slice(chain.source_site)] += chain.source_vec
        bvec[_cell_slice(chain.sink_site)] -= chain.sink_vec

        M = _assemble_block_matrix(chain, state.g)
        phi, info = _grounded_solve(M, bvec, [sink_dof, sink_dof + 1], state.phi_prev)
        state.phi_prev = phi
        info_h[k] = info

        J = np.zeros(m, dtype=complex)
        I_norm = np.zeros(m)
        theta = np.zeros(m)
        for e, bond in enumerate(chain.bonds):
            I_vec, J_sc = _bond_activity(phi, bond, state.g[e])
            J[e] = J_sc
            I_norm[e] = float(np.linalg.norm(I_vec))
            theta[e] = float(np.angle(J_sc + 1e-18))

        if phase_mode == "lifted":
            r = _wrap_to_pi(theta - state.theta_prev)
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

        Re_inv = np.real(1.0 / (state.g + 1e-18))
        t1 = float(np.sum(I_norm**2 / max(1e-12, ent.Tij) * np.maximum(0.0, Re_inv)))
        t2 = float(ent.kappa_slip * np.sum(dW))
        t3 = float(-ent.gamma * (state.S - ent.S_eq))
        state.S = max(0.0, state.S + dt * (t1 + t2 + t3))

        if adaptive_pi:
            dpi = ruler.alpha_pi * state.S - ruler.mu_pi * (state.pi_a - ruler.pi0)
            state.pi_a = float(np.clip(state.pi_a + dt * dpi, ruler.pi_min, ruler.pi_max))

        aS = _alpha_G(state.S, eg)
        mS = _mu_G(state.S, eg)
        dg = aS * I_norm * np.exp(1j * state.theta_R) - mS * state.g
        if eg.lambda_s > 0:
            sup = np.sin(state.theta_R / (2 * state.pi_a + 1e-18)) ** 2
            dg -= eg.lambda_s * sup * state.g
        dg += 1e-6 * (rng.normal(size=m) + 1j * rng.normal(size=m))
        state.g += dt * dg

        Re = np.clip(state.g.real, eg.g_min, eg.g_max)
        Im = np.clip(state.g.imag, -eg.g_imag_max, eg.g_imag_max)
        state.g = Re + 1j * Im
        if eg.budget_re is not None:
            sRe = float(np.sum(state.g.real))
            if sRe > eg.budget_re > 0:
                state.g *= eg.budget_re / sRe

        g_h[k] = state.g
        phi_h[k] = phi
        J_h[k] = J
        In_h[k] = I_norm
        tR_h[k] = state.theta_R
        th_h[k] = theta
        w_h[k] = w
        dW_h[k] = dW
        S_h[k] = state.S
        pi_h[k] = state.pi_a
        flip_h[k] = flip
        rb_h[k] = state.flip_count / (k + 1)

    return {
        "t": t, "g": g_h, "phi": phi_h, "J": J_h,
        "I_norm": In_h, "theta_R_e": tR_h, "theta_e": th_h,
        "w_e": w_h, "dW_e": dW_h, "S": S_h, "pi_a": pi_h,
        "flip": flip_h, "r_b": rb_h, "solve_info": info_h,
        "final_state": state,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def effective_transfer(phi, src, snk):
    dv = phi[_cell_slice(src)] - phi[_cell_slice(snk)]
    return 1.0 / max(1e-12, float(np.linalg.norm(dv)))


def boundary_current_fraction(I_norm, bonds):
    den = float(np.sum(np.abs(I_norm))) + 1e-12
    return sum(abs(v) for v, b in zip(I_norm, bonds) if b.is_boundary) / den


def edge_ratio(I_norm, bonds):
    """Ratio of edge bond current to total — tests Majorana localisation."""
    if len(bonds) < 2:
        return 0.0
    den = float(np.sum(np.abs(I_norm))) + 1e-12
    edge_current = abs(I_norm[0]) + abs(I_norm[-1])
    return edge_current / den


def slip_density(dW_hist):
    return np.mean(np.abs(dW_hist), axis=1)


def kitaev_topological_gap(chain):
    """Analytical BdG gap for infinite chain (quick check).
    
    Gap closes at |mu| = 2|t| (topological transition).
    """
    return abs(abs(chain.delta) - abs(chain.mu_chem / 2 - chain.t_hop))


def summarize_recovery(out, chain, damage_time, settle_window=5.0):
    t = out["t"]
    K = len(t)
    Yeff = np.array([effective_transfer(out["phi"][k], chain.source_site, chain.sink_site)
                      for k in range(K)])
    Bfrac = np.array([boundary_current_fraction(out["I_norm"][k], chain.bonds)
                       for k in range(K)])
    Eratio = np.array([edge_ratio(out["I_norm"][k], chain.bonds)
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
        "transfer_pre": pre_Y, "transfer_post": post_Y,
        "transfer_recovery": post_Y / max(1e-12, pre_Y),
        "boundary_pre": _a(Bfrac, pre), "boundary_post": _a(Bfrac, post),
        "edge_ratio_pre": _a(Eratio, pre), "edge_ratio_post": _a(Eratio, post),
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fails": int(np.sum(out["solve_info"] != 0)),
    }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def run_recovery_protocol(
    n_sites=20, mu_chem=0.5, t_hop=1.0, delta=1.0+0j,
    T=40.0, dt=0.05, seed=0,
    damage_time=15.0, damage_factor=1e-5,
    phase_mode="lifted", adaptive_pi=True,
    eg=None, ent=None, ruler=None,
):
    chain = build_kitaev_chain(n_sites, mu_chem=mu_chem, t_hop=t_hop, delta=delta)
    if eg is None:
        eg = KitaevEGATLParams()
    if ent is None:
        ent = KitaevEntropyParams()
    if ruler is None:
        ruler = KitaevRulerParams()

    state0 = make_initial_state(chain, S0=ent.S_init, pi0=ruler.pi_init)

    # Damage the middle bonds
    mid = len(chain.bonds) // 2
    damage_idx = list(range(max(0, mid - 1), min(len(chain.bonds), mid + 2)))

    interventions = [
        {"time": float(damage_time), "type": "scale_bonds",
         "bond_idx": damage_idx, "factor": damage_factor},
        {"time": float(damage_time), "type": "reset_entropy", "value": 3.0},
    ]
    out = simulate(
        chain, T=T, dt=dt, seed=seed,
        eg=eg, ent=ent, ruler=ruler,
        state0=state0, phase_mode=phase_mode,
        adaptive_pi=adaptive_pi, interventions=interventions,
    )
    return chain, out


def compare_ablations(
    n_sites=20, T=40.0, dt=0.05, seed=0,
    damage_time=15.0, mu_chem=0.5,
):
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }
    results = {}
    for name, cfg in configs.items():
        ch, out = run_recovery_protocol(
            n_sites=n_sites, mu_chem=mu_chem,
            T=T, dt=dt, seed=seed,
            damage_time=damage_time, **cfg,
        )
        summ = summarize_recovery(out, ch, damage_time)
        results[name] = (ch, out, summ)
    return results
