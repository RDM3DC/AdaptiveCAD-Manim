"""EGATL + QWZ block-admittance simulator for topological lattice dynamics.

Two-band block-Hamiltonian with adaptive scalar edge multipliers evolved
by the EGATL law:

    d/dt g_e = α_G(S)·|J_e|·exp(iθ_R,e) − μ_G(S)·g_e − λ_s·suppression

Includes:
- QWZ-inspired 2×2 bond blocks (σ_x, σ_y, σ_z Pauli structure)
- Entropy-driven α/μ gating with adaptive phase ruler π_a
- Lifted / principal phase tracking with winding numbers
- Fukui-Hatsugai-Suzuki discretised Chern number (bulk topological proxy)
- Damage / recovery protocols with ablation comparison
- Metrics: effective transfer, boundary current, top-edge fraction, slip density

Adapted from hafc_sim2_qwz_block_complete.py into the ARP solver framework.
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

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    a = (x + math.pi) % (2 * math.pi) - math.pi
    return np.where(a <= -math.pi, a + 2 * math.pi, a)


def _logistic(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _sigma_x():
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _sigma_y():
    return np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)


def _sigma_z():
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _sigma_0():
    return np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class EGATLParams:
    """Adaptive edge-coupling parameters."""
    alpha0: float = 1.5
    S_c: float = 1.0
    dS: float = 0.35
    mu0: float = 0.55
    S0: float = 1.0
    lambda_s: float = 0.10
    g_min: float = 1e-6
    g_max: float = 5.0
    g_imag_max: float = 5.0
    budget_re: Optional[float] = 60.0


@dataclass
class EntropyParams:
    """Entropy dynamics."""
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25
    kappa_slip: float = 0.15
    Tij: float = 1.0


@dataclass
class RulerParams:
    """Adaptive phase ruler π_a."""
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.2
    pi_max: float = 8.0 * math.pi


def alpha_G(S: float, p: EGATLParams) -> float:
    """Entropy-gated reinforcement coupling."""
    return p.alpha0 * _logistic((S - p.S_c) / max(1e-12, p.dS))


def mu_G(S: float, p: EGATLParams) -> float:
    """Entropy-gated decay rate."""
    return p.mu0 * (1.0 + S / max(1e-12, p.S0))


# ---------------------------------------------------------------------------
# Block QWZ lattice
# ---------------------------------------------------------------------------

@dataclass
class BlockBond:
    u: int
    v: int
    B: np.ndarray
    direction: str
    is_boundary: bool
    label: str = ""


@dataclass
class QWZLattice:
    """QWZ block lattice benchmark."""
    nx: int
    ny: int
    n_cells: int
    bonds: List[BlockBond]
    source_cell: int
    sink_cell: int
    source_vec: np.ndarray
    sink_vec: np.ndarray
    mass: float
    eta: float
    onsite_block: np.ndarray
    cell_xy: Dict[int, Tuple[int, int]] = field(default_factory=dict)


def _cell_id(x: int, y: int, nx: int) -> int:
    return x + y * nx


def standard_qwz_bond_blocks(tx: float = 1.0, ty: float = 1.0):
    """QWZ nearest-neighbour bond matrices T_x, T_y."""
    Tx = 0.5 * tx * (_sigma_z() - 1j * _sigma_x())
    Ty = 0.5 * ty * (_sigma_z() - 1j * _sigma_y())
    return Tx, Ty


def build_qwz_lattice(
    nx: int = 8,
    ny: int = 8,
    mass: float = -0.25,
    eta: float = 0.35,
    tx: float = 1.0,
    ty: float = 1.0,
    source_orbital: Literal["plus", "A"] = "plus",
) -> QWZLattice:
    """Build an open-boundary QWZ-style block lattice.

    Parameters
    ----------
    nx, ny : int
        Lattice dimensions (must be ≥ 3).
    mass : float
        Topological mass channel (m·σ_z).
    eta : float
        Regularisation damping (η·I₂).
    tx, ty : float
        Hopping strengths in x, y directions.
    source_orbital : str
        'A' → inject on first orbital; 'plus' → equal superposition.
    """
    if nx < 3 or ny < 3:
        raise ValueError("Need at least a 3×3 lattice.")

    Tx, Ty = standard_qwz_bond_blocks(tx, ty)
    bonds: List[BlockBond] = []
    cell_xy: Dict[int, Tuple[int, int]] = {}

    for y in range(ny):
        for x in range(nx):
            c = _cell_id(x, y, nx)
            cell_xy[c] = (x, y)
            if x < nx - 1:
                v = _cell_id(x + 1, y, nx)
                bnd = y == 0 or y == ny - 1 or x == 0 or x + 1 == nx - 1
                bonds.append(BlockBond(
                    u=c, v=v, B=Tx.copy(), direction="x",
                    is_boundary=bnd, label=f"x({x},{y})",
                ))
            if y < ny - 1:
                v = _cell_id(x, y + 1, nx)
                bnd = x == 0 or x == nx - 1 or y == 0 or y + 1 == ny - 1
                bonds.append(BlockBond(
                    u=c, v=v, B=Ty.copy(), direction="y",
                    is_boundary=bnd, label=f"y({x},{y})",
                ))

    src_cell = _cell_id(0, ny - 1, nx)
    snk_cell = _cell_id(nx - 1, ny - 1, nx)
    if source_orbital == "A":
        sv = np.array([1.0, 0.0], dtype=complex)
    else:
        sv = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)

    return QWZLattice(
        nx=nx, ny=ny, n_cells=nx * ny, bonds=bonds,
        source_cell=src_cell, sink_cell=snk_cell,
        source_vec=sv.copy(), sink_vec=sv.copy(),
        mass=mass, eta=eta,
        onsite_block=eta * _sigma_0() + mass * _sigma_z(),
        cell_xy=cell_xy,
    )


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

@dataclass
class EGATLState:
    g: np.ndarray          # complex edge multipliers
    S: float               # entropy
    pi_a: float            # adaptive ruler
    theta_R: np.ndarray    # lifted phase
    theta_prev: np.ndarray
    w_prev: np.ndarray     # winding numbers
    phi_prev: np.ndarray   # last solution vector
    b_prev: int = 1
    flip_count: int = 0


def make_initial_state(
    lattice: QWZLattice,
    g0: Optional[np.ndarray] = None,
    S0: float = 0.5,
    pi0: float = math.pi,
) -> EGATLState:
    m = len(lattice.bonds)
    nd = 2 * lattice.n_cells
    if g0 is None:
        g0 = np.ones(m, dtype=complex)
    return EGATLState(
        g=np.array(g0, dtype=complex).copy(),
        S=float(S0),
        pi_a=float(pi0),
        theta_R=np.zeros(m),
        theta_prev=np.zeros(m),
        w_prev=np.zeros(m, dtype=int),
        phi_prev=np.zeros(nd, dtype=complex),
    )


def clone_state(s: EGATLState) -> EGATLState:
    return EGATLState(
        g=s.g.copy(), S=s.S, pi_a=s.pi_a,
        theta_R=s.theta_R.copy(), theta_prev=s.theta_prev.copy(),
        w_prev=s.w_prev.copy(), phi_prev=s.phi_prev.copy(),
        b_prev=s.b_prev, flip_count=s.flip_count,
    )


# ---------------------------------------------------------------------------
# Matrix assembly & solve
# ---------------------------------------------------------------------------

def _assemble_block_matrix(lattice: QWZLattice, g: np.ndarray) -> csr_matrix:
    n = lattice.n_cells
    nd = 2 * n
    rows, cols, data = [], [], []

    def add_block(ic, jc, block):
        bi, bj = 2 * ic, 2 * jc
        for a in range(2):
            for b in range(2):
                rows.append(bi + a)
                cols.append(bj + b)
                data.append(block[a, b])

    for c in range(n):
        add_block(c, c, lattice.onsite_block)

    for e, bond in enumerate(lattice.bonds):
        ge = g[e]
        B = bond.B
        add_block(bond.u, bond.u, ge * B)
        add_block(bond.u, bond.v, -ge * B)
        add_block(bond.v, bond.u, -np.conj(ge) * B.conj().T)
        add_block(bond.v, bond.v, np.conj(ge) * B.conj().T)

    return coo_matrix(
        (np.array(data, dtype=complex), (rows, cols)), shape=(nd, nd)
    ).tocsr()


def _grounded_solve(M, b, grounds, x0, rtol=1e-10, maxiter=2500):
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
        xr, info = gmres(Mr, br, x0=x0r, rtol=rtol, atol=0.0,
                         maxiter=maxiter, restart=50)

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
    lattice: QWZLattice,
    T: float = 80.0,
    dt: float = 0.05,
    seed: int = 0,
    eg: Optional[EGATLParams] = None,
    ent: Optional[EntropyParams] = None,
    ruler: Optional[RulerParams] = None,
    state0: Optional[EGATLState] = None,
    phase_mode: Literal["lifted", "principal"] = "lifted",
    adaptive_pi: bool = True,
    interventions: Optional[List[dict]] = None,
) -> Dict[str, np.ndarray]:
    """Run the EGATL block-admittance simulation.

    Returns dict with time-series arrays:
        t, g, phi, J, I_norm, theta_R_e, theta_e, w_e, dW_e,
        S, pi_a, r_b, flip, solve_info, final_state
    """
    if eg is None:
        eg = EGATLParams()
    if ent is None:
        ent = EntropyParams()
    if ruler is None:
        ruler = RulerParams()

    rng = np.random.default_rng(seed)
    m = len(lattice.bonds)
    nd = 2 * lattice.n_cells
    K = int(np.ceil(T / dt)) + 1
    t = np.linspace(0, T, K)

    if state0 is None:
        state = make_initial_state(lattice, S0=ent.S_init, pi0=ruler.pi_init)
    else:
        state = clone_state(state0)

    # History arrays
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

    sink_dof = 2 * lattice.sink_cell

    for k in range(K):
        t_now = float(t[k])
        _apply_interventions(t_now, state, interventions)

        # Build RHS
        bvec = np.zeros(nd, dtype=complex)
        bvec[_cell_slice(lattice.source_cell)] += lattice.source_vec
        bvec[_cell_slice(lattice.sink_cell)] -= lattice.sink_vec

        M = _assemble_block_matrix(lattice, state.g)
        phi, info = _grounded_solve(
            M, bvec, [sink_dof, sink_dof + 1], state.phi_prev
        )
        state.phi_prev = phi
        info_h[k] = info

        # Edge activities
        J = np.zeros(m, dtype=complex)
        I_norm = np.zeros(m)
        theta = np.zeros(m)
        for e, bond in enumerate(lattice.bonds):
            I_vec, J_sc = _bond_activity(phi, bond, state.g[e])
            J[e] = J_sc
            I_norm[e] = float(np.linalg.norm(I_vec))
            theta[e] = float(np.angle(J_sc + 1e-18))

        # Phase tracking
        if phase_mode == "lifted":
            r = _wrap_to_pi(theta - state.theta_prev)
            r_clip = np.clip(r, -state.pi_a, state.pi_a)
            state.theta_R += r_clip
        else:
            state.theta_R = theta.copy()
            r_clip = _wrap_to_pi(theta - state.theta_prev)
        state.theta_prev = theta.copy()

        w = np.round(state.theta_R / (2 * math.pi)).astype(int)
        dW = np.abs(w - state.w_prev).astype(float)
        state.w_prev = w.copy()

        b_edges = np.where((w % 2) == 0, 1, -1)
        b = 1 if int(np.sum(b_edges)) >= 0 else -1
        flip = int(b != state.b_prev)
        state.b_prev = b
        state.flip_count += flip

        # Entropy dynamics
        Re_inv = np.real(1.0 / (state.g + 1e-18))
        term1 = float(np.sum(
            I_norm ** 2 / max(1e-12, ent.Tij) * np.maximum(0.0, Re_inv)
        ))
        term2 = float(ent.kappa_slip * np.sum(dW))
        term3 = float(-ent.gamma * (state.S - ent.S_eq))
        state.S = max(0.0, state.S + dt * (term1 + term2 + term3))

        # Adaptive ruler
        if adaptive_pi:
            dpi = ruler.alpha_pi * state.S - ruler.mu_pi * (state.pi_a - ruler.pi0)
            state.pi_a = float(np.clip(
                state.pi_a + dt * dpi, ruler.pi_min, ruler.pi_max
            ))

        # EGATL edge update
        aS = alpha_G(state.S, eg)
        mS = mu_G(state.S, eg)
        dg = aS * I_norm * np.exp(1j * state.theta_R) - mS * state.g

        if eg.lambda_s > 0:
            sup = np.sin(state.theta_R / (2 * state.pi_a + 1e-18)) ** 2
            dg -= eg.lambda_s * sup * state.g

        dg += 1e-6 * (rng.normal(size=m) + 1j * rng.normal(size=m))
        state.g += dt * dg

        # Clamps
        Re = np.clip(state.g.real, eg.g_min, eg.g_max)
        Im = np.clip(state.g.imag, -eg.g_imag_max, eg.g_imag_max)
        state.g = Re + 1j * Im
        if eg.budget_re is not None:
            sRe = float(np.sum(state.g.real))
            if sRe > eg.budget_re > 0:
                state.g *= eg.budget_re / sRe

        # Store
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

def effective_transfer(phi, src_cell, snk_cell):
    dv = phi[_cell_slice(src_cell)] - phi[_cell_slice(snk_cell)]
    return 1.0 / max(1e-12, float(np.linalg.norm(dv)))


def boundary_current_fraction(I_norm, bonds):
    den = float(np.sum(np.abs(I_norm))) + 1e-12
    num = sum(abs(v) for v, b in zip(I_norm, bonds) if b.is_boundary)
    return num / den


def top_edge_fraction(I_norm, lattice):
    den = float(np.sum(np.abs(I_norm))) + 1e-12
    num = 0.0
    for v, bond in zip(I_norm, lattice.bonds):
        _, uy = lattice.cell_xy[bond.u]
        _, vy = lattice.cell_xy[bond.v]
        if uy == lattice.ny - 1 and vy == lattice.ny - 1:
            num += abs(v)
    return num / den


def slip_density(dW_hist):
    return np.mean(np.abs(dW_hist), axis=1)


def direction_strength(g, bonds, direction):
    vals = [abs(g[i]) for i, b in enumerate(bonds) if b.direction == direction]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Chern number (Fukui-Hatsugai-Suzuki)
# ---------------------------------------------------------------------------

def qwz_hamiltonian(kx, ky, tx, ty, mass):
    """Bloch Hamiltonian H(k) for the QWZ model."""
    dx = tx * math.sin(kx)
    dy = ty * math.sin(ky)
    dz = mass + tx * math.cos(kx) + ty * math.cos(ky)
    return dx * _sigma_x() + dy * _sigma_y() + dz * _sigma_z()


def chern_number(tx, ty, mass, nk=31):
    """Discretised Chern number for the lower band of QWZ H(k)."""
    ks = np.linspace(-math.pi, math.pi, nk, endpoint=False)
    u = np.zeros((nk, nk, 2), dtype=complex)

    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            vals, vecs = np.linalg.eigh(qwz_hamiltonian(kx, ky, tx, ty, mass))
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


def proxy_chern_series(g_hist, lattice, nk=25):
    """Time-series of the translationally-averaged proxy Chern number."""
    out = np.zeros(g_hist.shape[0])
    for k in range(g_hist.shape[0]):
        tx = direction_strength(g_hist[k], lattice.bonds, "x")
        ty = direction_strength(g_hist[k], lattice.bonds, "y")
        out[k] = chern_number(tx, ty, lattice.mass, nk)
    return out


# ---------------------------------------------------------------------------
# Damage / recovery protocols
# ---------------------------------------------------------------------------

def top_edge_damage_bonds(lattice):
    """Find bond indices touching the top edge near the centre."""
    xc = lattice.nx // 2
    targets = {
        _cell_id(max(0, xc - 1), lattice.ny - 1, lattice.nx),
        _cell_id(min(lattice.nx - 1, xc), lattice.ny - 1, lattice.nx),
        _cell_id(min(lattice.nx - 1, xc + 1), lattice.ny - 1, lattice.nx),
    }
    idx = []
    for i, bond in enumerate(lattice.bonds):
        if bond.u in targets or bond.v in targets:
            _, uy = lattice.cell_xy[bond.u]
            _, vy = lattice.cell_xy[bond.v]
            if uy == lattice.ny - 1 or vy == lattice.ny - 1:
                idx.append(i)
    return sorted(set(idx))


def run_recovery_protocol(
    nx=6, ny=6, T=24.0, dt=0.1, seed=0,
    damage_time=10.0, damage_factor=1e-5,
    mass=-1.0, eta=0.35, tx=1.0, ty=1.0,
    phase_mode="lifted", adaptive_pi=True,
    eg=None, ent=None, ruler=None,
):
    """Run a damage-then-recover protocol and return (lattice, output)."""
    lattice = build_qwz_lattice(nx, ny, mass, eta, tx, ty)

    if eg is None:
        eg = EGATLParams(
            alpha0=1.5, S_c=1.0, dS=0.35, mu0=0.55, S0=1.0,
            lambda_s=0.12, g_min=1e-6, g_max=6.0,
            g_imag_max=6.0, budget_re=75.0,
        )
    if ent is None:
        ent = EntropyParams(S_init=0.5, S_eq=0.5, gamma=0.25,
                            kappa_slip=0.18, Tij=1.0)
    if ruler is None:
        ruler = RulerParams()

    g0 = np.ones(len(lattice.bonds), dtype=complex)
    for i, b in enumerate(lattice.bonds):
        g0[i] = tx if b.direction == "x" else ty
    state0 = make_initial_state(lattice, g0, ent.S_init, ruler.pi_init)

    dmg_idx = top_edge_damage_bonds(lattice)
    interventions = [
        {"time": damage_time, "type": "scale_bonds",
         "bond_idx": dmg_idx, "factor": damage_factor},
        {"time": damage_time, "type": "reset_entropy", "value": 3.0},
    ]

    out = simulate(
        lattice, T, dt, seed, eg, ent, ruler, state0,
        phase_mode, adaptive_pi, interventions,
    )
    return lattice, out


def summarize_recovery(out, lattice, damage_time, settle_window=6.0, nk=25):
    """Post-damage recovery summary metrics."""
    t = out["t"]
    K = len(t)

    Yeff = np.array([
        effective_transfer(out["phi"][k], lattice.source_cell, lattice.sink_cell)
        for k in range(K)
    ])
    Bfrac = np.array([
        boundary_current_fraction(out["I_norm"][k], lattice.bonds)
        for k in range(K)
    ])
    Tfrac = np.array([
        top_edge_fraction(out["I_norm"][k], lattice) for k in range(K)
    ])
    Ch = proxy_chern_series(out["g"], lattice, nk)

    pre = (t >= max(0, damage_time - settle_window)) & (t < damage_time)
    post = t >= damage_time + settle_window
    if not np.any(pre):
        pre = t < damage_time
    if not np.any(post):
        post = t >= damage_time

    def _avg(arr, mask):
        return float(np.mean(arr[mask])) if np.any(mask) else 0.0

    return {
        "transfer_pre": _avg(Yeff, pre),
        "transfer_post": _avg(Yeff, post),
        "transfer_recovery": _avg(Yeff, post) / max(1e-12, _avg(Yeff, pre)),
        "boundary_pre": _avg(Bfrac, pre),
        "boundary_post": _avg(Bfrac, post),
        "top_edge_pre": _avg(Tfrac, pre),
        "top_edge_post": _avg(Tfrac, post),
        "chern_pre": _avg(Ch, pre),
        "chern_post": _avg(Ch, post),
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "final_r_b": float(out["r_b"][-1]),
        "gmres_fails": int(np.sum(out["solve_info"] != 0)),
    }


def compare_ablations(
    nx=6, ny=6, T=24.0, dt=0.1, seed=0,
    damage_time=10.0, mass=-0.25,
):
    """Run lifted/principal × fixed/adaptive ablation comparison."""
    configs = {
        "principal_fixed_pi": dict(phase_mode="principal", adaptive_pi=False),
        "lifted_fixed_pi": dict(phase_mode="lifted", adaptive_pi=False),
        "lifted_adaptive_pi": dict(phase_mode="lifted", adaptive_pi=True),
    }
    results = {}
    for name, cfg in configs.items():
        lat, out = run_recovery_protocol(
            nx=nx, ny=ny, T=T, dt=dt, seed=seed,
            damage_time=damage_time, mass=mass, **cfg,
        )
        summ = summarize_recovery(out, lat, damage_time)
        results[name] = (lat, out, summ)
    return results
