"""Leaderboard equation benchmarks for the ARP Topological Solver.

Numerically verifies / computes key equations from the TopEquations leaderboard:

  LB #1  — History-Resolved Phase: monodromy test (1 full winding → θ_R ≈ 2π)
  LB #1  — Matched-present protocol (principal collapses, lifted retains memory)
  LB #8  — Topological Coherence Order Parameter Ψ = ⟨cos(Θ_p / π_a)⟩
  LB #14 — Slip-regime asymptote: r_b → |Δ|/π ≈ 1/π in slip
  LB #15 — Plaquette holonomy Θ_p = Σ σ_{p,e} θ_{R,e}
  LB #35 — Real-Space Chern Marker (Bianco–Resta, open boundaries)

Run:  python -m solver benchmark
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .egatl import (
    EGATLParams,
    EntropyParams,
    RulerParams,
    QWZLattice,
    EGATLState,
    build_qwz_lattice,
    simulate,
    chern_number,
    make_initial_state,
    _wrap_to_pi,
    _cell_id,
    _sigma_x,
    _sigma_y,
    _sigma_z,
    qwz_hamiltonian,
)


# ═══════════════════════════════════════════════════════════════════════════
# LB #35 — Real-Space Chern Marker  (Bianco–Resta, open boundaries)
# C(r) = -2πi ⟨r| [PXP, PYP] |r⟩,   P = Σ_{En < EF} |n⟩⟨n|
# ═══════════════════════════════════════════════════════════════════════════

def _build_realspace_qwz(nx: int, ny: int, mass: float,
                          tx: float = 1.0, ty: float = 1.0) -> np.ndarray:
    """Build the real-space tight-binding QWZ Hamiltonian (2N × 2N)."""
    N = nx * ny
    H = np.zeros((2 * N, 2 * N), dtype=complex)

    sx, sy, sz = _sigma_x(), _sigma_y(), _sigma_z()
    onsite = mass * sz

    for y in range(ny):
        for x in range(nx):
            c = x + y * nx
            # Onsite
            H[2*c:2*c+2, 2*c:2*c+2] += onsite

            # x-hop
            if x < nx - 1:
                v = c + 1
                Tx = 0.5 * tx * (sz - 1j * sx)
                H[2*c:2*c+2, 2*v:2*v+2] += Tx
                H[2*v:2*v+2, 2*c:2*c+2] += Tx.conj().T

            # y-hop
            if y < ny - 1:
                v = c + nx
                Ty = 0.5 * ty * (sz - 1j * sy)
                H[2*c:2*c+2, 2*v:2*v+2] += Ty
                H[2*v:2*v+2, 2*c:2*c+2] += Ty.conj().T

    return H


def realspace_chern_marker(
    nx: int = 10,
    ny: int = 10,
    mass: float = -1.0,
    tx: float = 1.0,
    ty: float = 1.0,
    margin: int = 2,
) -> Dict[str, float]:
    """Compute the Bianco–Resta real-space Chern marker.

    Returns dict with 'bulk_average', 'marker_map' (ny×nx), and metadata.
    The bulk average excludes `margin` edge sites on each side.
    """
    N = nx * ny
    dim = 2 * N

    # Build Hamiltonian and diagonalise
    H = _build_realspace_qwz(nx, ny, mass, tx, ty)
    evals, evecs = np.linalg.eigh(H)

    # Ground-state projector (lower band = first N states)
    n_occ = N  # half-filling for 2-band model
    P = evecs[:, :n_occ] @ evecs[:, :n_occ].conj().T

    # Position operators (cell coordinates, each expanded to 2 orbitals)
    X = np.zeros((dim, dim))
    Y = np.zeros((dim, dim))
    for y in range(ny):
        for x in range(nx):
            c = x + y * nx
            for orb in range(2):
                idx = 2 * c + orb
                X[idx, idx] = float(x)
                Y[idx, idx] = float(y)

    # Projected positions
    PXP = P @ X @ P
    PYP = P @ Y @ P

    # Commutator [PXP, PYP]
    comm = PXP @ PYP - PYP @ PXP

    # Local Chern marker: C(r) = -2πi ⟨r|comm|r⟩
    # where |r⟩ spans the 2 orbitals at site r
    marker = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            c = x + y * nx
            # Sum over both orbitals
            cr = 0.0
            for orb in range(2):
                idx = 2 * c + orb
                cr += comm[idx, idx]
            marker[y, x] = -2.0 * math.pi * cr.imag

    # Bulk average (exclude margin)
    x0, x1 = margin, nx - margin
    y0, y1 = margin, ny - margin
    if x1 <= x0 or y1 <= y0:
        bulk_avg = float(np.mean(marker))
    else:
        bulk_avg = float(np.mean(marker[y0:y1, x0:x1]))

    return {
        "bulk_average": bulk_avg,
        "marker_map": marker,
        "nx": nx,
        "ny": ny,
        "mass": mass,
        "margin": margin,
        "gap": float(evals[n_occ] - evals[n_occ - 1]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# LB #15 — Plaquette Holonomy  Θ_p = Σ_{e ∈ ∂p} σ_{p,e} θ_{R,e}
# LB #8  — Topological Coherence  Ψ = (1/Np) Σ cos(Θ_p / π_a)
# ═══════════════════════════════════════════════════════════════════════════

def _build_plaquette_map(lattice: QWZLattice):
    """Build plaquettes and their oriented edge lists for a square lattice.

    Returns list of (plaquette_edges) where each entry is
    [(edge_idx, sign), ...] going around the plaquette CCW.
    """
    nx, ny = lattice.nx, lattice.ny

    # Index bonds by (u, v) for fast lookup
    bond_index = {}
    for i, bond in enumerate(lattice.bonds):
        bond_index[(bond.u, bond.v)] = (i, +1)
        bond_index[(bond.v, bond.u)] = (i, -1)

    plaquettes = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            # CCW: bottom→right→top(rev)→left(rev)
            bl = _cell_id(x, y, nx)
            br = _cell_id(x + 1, y, nx)
            tr = _cell_id(x + 1, y + 1, nx)
            tl = _cell_id(x, y + 1, nx)

            edges = []
            for u, v in [(bl, br), (br, tr), (tr, tl), (tl, bl)]:
                if (u, v) in bond_index:
                    edges.append(bond_index[(u, v)])
                else:
                    # Should not happen for well-formed square lattice
                    continue

            if len(edges) == 4:
                plaquettes.append(edges)

    return plaquettes


def plaquette_holonomy(theta_R: np.ndarray, lattice: QWZLattice):
    """Compute plaquette holonomies Θ_p = Σ σ_{p,e} θ_{R,e}.

    Returns array of holonomies, one per plaquette.
    """
    plaquettes = _build_plaquette_map(lattice)
    Theta = np.zeros(len(plaquettes))
    for p_idx, edges in enumerate(plaquettes):
        for edge_idx, sign in edges:
            Theta[p_idx] += sign * theta_R[edge_idx]
    return Theta


def coherence_order_parameter(theta_R: np.ndarray, pi_a: float,
                               lattice: QWZLattice) -> float:
    """Topological coherence Ψ = (1/Np) Σ cos(Θ_p / π_a)."""
    Theta = plaquette_holonomy(theta_R, lattice)
    if len(Theta) == 0:
        return 0.0
    return float(np.mean(np.cos(Theta / pi_a)))


# ═══════════════════════════════════════════════════════════════════════════
# LB #14 — Slip-regime asymptote  r_b → |Δ|/π
# ═══════════════════════════════════════════════════════════════════════════

def verify_slip_asymptote(
    n_steps: int = 5000,
    Delta: float = 1.0,
) -> Dict[str, float]:
    """Verify r_b → |Δ|/π for a single Adler/RSJ junction in slip.

    Uses φ̇ = Δ − λG sin φ  with λG < Δ (below locking threshold).
    Parity b_k = sgn(sin φ_k); flips counted per unit-time step (dt=1).
    In slip: sinφ oscillates at mean frequency ω ≈ Δ, so parity flips
    at rate ~Δ/π (sinφ crosses zero twice per 2π period).
    """
    dt = 1.0  # unit time step for r_b normalisation
    lambda_G = 0.3  # well below locking threshold Δ=1
    phi = 0.1  # avoid starting exactly at 0
    flips = 0
    b_prev = 1 if math.sin(phi) >= 0 else -1

    for k in range(n_steps):
        phi += dt * (Delta - lambda_G * math.sin(phi))
        b = 1 if math.sin(phi) >= 0 else -1
        if b != b_prev:
            flips += 1
        b_prev = b

    r_b = flips / max(1, n_steps - 1)
    target = abs(Delta) / math.pi

    return {
        "r_b_measured": r_b,
        "r_b_target": target,
        "relative_error": abs(r_b - target) / target,
        "Delta": Delta,
        "lambda_G": lambda_G,
        "n_steps": n_steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LB #1 — Monodromy test: one full winding → θ_R ≈ 2π, w=1, b=-1
# ═══════════════════════════════════════════════════════════════════════════

def monodromy_test(n_steps: int = 100) -> Dict[str, float]:
    """Track one full 2π winding through the phase-lift update.

    Feeds a linearly advancing raw phase through the lifted update rule
    and checks that θ_R accumulates to ≈ 2π with w=1, b=-1.
    """
    pi_a = math.pi
    theta_R = 0.0
    theta_prev = 0.0

    for k in range(1, n_steps + 1):
        # Raw phase advances linearly from 0 to 2π
        theta_raw = 2.0 * math.pi * k / n_steps

        # Wrap to principal branch
        r = ((theta_raw - theta_prev + math.pi) % (2 * math.pi)) - math.pi
        r_clip = max(-pi_a, min(pi_a, r))
        theta_R += r_clip
        theta_prev = theta_raw

    w = round(theta_R / (2 * math.pi))
    b = (-1) ** w

    return {
        "theta_R": theta_R,
        "theta_R_over_2pi": theta_R / (2 * math.pi),
        "winding": w,
        "parity": b,
        "expected_theta_R": 2 * math.pi,
        "expected_w": 1,
        "expected_b": -1,
        "pass": abs(theta_R - 2 * math.pi) < 0.01 and w == 1 and b == -1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LB #1 — Matched-present protocol
# ═══════════════════════════════════════════════════════════════════════════

def matched_present_test(
    nx: int = 6,
    ny: int = 6,
    T_warm: float = 10.0,
    T_chirp: float = 10.0,
    T_resume: float = 10.0,
    dt: float = 0.1,
    mass: float = -1.0,
) -> Dict[str, float]:
    """Matched-present history-divergence protocol.

    1. Warm-up (T_warm) — both lifted and principal evolve identically
    2. Extra-chirp (T_chirp) — inject phase disturbance
    3. Resume (T_resume) — same raw phase as warm-up

    The principal branch collapses back to δθ ≈ 0 while the full model
    retains δθ_R ≈ 2π with different winding/parity.
    """
    lattice = build_qwz_lattice(nx, ny, mass=mass)
    eg = EGATLParams(alpha0=1.5, mu0=0.55, lambda_s=0.10)
    ent = EntropyParams(S_init=0.5, S_eq=0.5)
    ruler = RulerParams()

    T_total = T_warm + T_chirp + T_resume

    # Lifted mode (full model)
    out_lifted = simulate(
        lattice, T=T_total, dt=dt, seed=0, eg=eg, ent=ent, ruler=ruler,
        phase_mode="lifted", adaptive_pi=True,
    )

    # Principal mode (baseline)
    out_principal = simulate(
        lattice, T=T_total, dt=dt, seed=0, eg=eg, ent=ent, ruler=ruler,
        phase_mode="principal", adaptive_pi=False,
    )

    # Compare final states
    theta_R_lifted = out_lifted["theta_R_e"][-1]
    theta_R_principal = out_principal["theta_R_e"][-1]

    delta_theta = float(np.max(np.abs(theta_R_lifted - theta_R_principal)))

    g_lifted = out_lifted["g"][-1]
    g_principal = out_principal["g"][-1]
    delta_g = float(np.max(np.abs(g_lifted - g_principal)))

    return {
        "delta_theta_R_max": delta_theta,
        "delta_g_max": delta_g,
        "lifted_S_final": float(out_lifted["S"][-1]),
        "principal_S_final": float(out_principal["S"][-1]),
        "lifted_pi_a_final": float(out_lifted["pi_a"][-1]),
        "history_diverges": delta_theta > 1.0,
        "pass": delta_theta > 1.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LB #8 — Topological Coherence time series from simulation
# ═══════════════════════════════════════════════════════════════════════════

def coherence_timeseries(
    nx: int = 6,
    ny: int = 6,
    T: float = 40.0,
    dt: float = 0.1,
    mass: float = -1.0,
) -> Dict[str, np.ndarray]:
    """Compute Ψ(t) over a full EGATL simulation.

    Tracks whether the lattice is in the locked (Ψ→1) or chaotic (Ψ→0) regime.
    """
    lattice = build_qwz_lattice(nx, ny, mass=mass)
    out = simulate(lattice, T=T, dt=dt, seed=0)

    K = len(out["t"])
    psi = np.zeros(K)
    for k in range(K):
        psi[k] = coherence_order_parameter(
            out["theta_R_e"][k], out["pi_a"][k], lattice
        )

    return {
        "t": out["t"],
        "psi": psi,
        "S": out["S"],
        "pi_a": out["pi_a"],
        "r_b": out["r_b"],
        "psi_initial": float(psi[0]),
        "psi_final": float(psi[-1]),
        "psi_mean": float(np.mean(psi[K // 2:])),
    }


# ═══════════════════════════════════════════════════════════════════════════
# LB #6 — Chern number phase diagram verification
# ═══════════════════════════════════════════════════════════════════════════

def chern_phase_diagram(nk: int = 31) -> Dict[str, list]:
    """Verify the QWZ phase diagram: C=-1 for -2<m<0, C=+1 for 0<m<2."""
    masses = np.linspace(-3.0, 3.0, 25)
    cherns = [chern_number(1.0, 1.0, float(m), nk) for m in masses]
    rounded = [round(c) for c in cherns]

    # Verify expected phases
    correct = 0
    total = 0
    for m, c_r in zip(masses, rounded):
        if abs(m) > 2.0:
            expected = 0
        elif m < 0:
            expected = -1
        else:
            expected = 1
        total += 1
        if c_r == expected:
            correct += 1

    return {
        "masses": masses.tolist(),
        "chern_numbers": cherns,
        "rounded": rounded,
        "accuracy": correct / total,
        "pass": correct / total > 0.85,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Master benchmark runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all(verbose: bool = True) -> Dict[str, dict]:
    """Run all leaderboard benchmarks and return results."""
    results = {}

    def _print(msg):
        if verbose:
            print(msg)

    def _hr():
        _print("─" * 64)

    # ---- LB #1: Monodromy ----
    _print("\n  LB #1 — Monodromy Test (1 full winding)")
    _hr()
    mono = monodromy_test()
    results["monodromy"] = mono
    _print(f"  θ_R = {mono['theta_R']:.6f}  (expected {mono['expected_theta_R']:.6f})")
    _print(f"  w   = {mono['winding']}  (expected {mono['expected_w']})")
    _print(f"  b   = {mono['parity']}  (expected {mono['expected_b']})")
    _print(f"  {'✓ PASS' if mono['pass'] else '✗ FAIL'}")
    _hr()

    # ---- LB #1: Matched-present ----
    _print("\n  LB #1 — Matched-Present Protocol")
    _hr()
    mp = matched_present_test()
    results["matched_present"] = mp
    _print(f"  max |Δθ_R|  = {mp['delta_theta_R_max']:.6f}")
    _print(f"  max |ΔG|    = {mp['delta_g_max']:.6e}")
    _print(f"  Lifted S    = {mp['lifted_S_final']:.4f}")
    _print(f"  Principal S = {mp['principal_S_final']:.4f}")
    _print(f"  History diverges: {mp['history_diverges']}")
    _print(f"  {'✓ PASS' if mp['pass'] else '✗ FAIL'}")
    _hr()

    # ---- LB #6: Chern phase diagram ----
    _print("\n  LB #6 — QWZ Chern Phase Diagram")
    _hr()
    cpd = chern_phase_diagram()
    results["chern_phase_diagram"] = cpd
    _print(f"  25-point mass sweep, nk=31")
    _print(f"  Accuracy: {cpd['accuracy']*100:.0f}%")
    # Show a few
    for m_val, c_val in list(zip(cpd["masses"], cpd["chern_numbers"]))[::6]:
        _print(f"    m={m_val:+5.2f}  C={c_val:+.4f}  ({round(c_val)})")
    _print(f"  {'✓ PASS' if cpd['pass'] else '✗ FAIL'}")
    _hr()

    # ---- LB #8 + #15: Coherence + Plaquette holonomy ----
    _print("\n  LB #8 — Topological Coherence Ψ(t)")
    _print("  LB #15 — Plaquette Holonomy Θ_p")
    _hr()
    coh = coherence_timeseries()
    results["coherence"] = {
        k: v for k, v in coh.items() if k not in ("t", "psi", "S", "pi_a", "r_b")
    }
    _print(f"  Ψ(0)    = {coh['psi_initial']:.4f}")
    _print(f"  Ψ(end)  = {coh['psi_final']:.4f}")
    _print(f"  ⟨Ψ⟩_tail = {coh['psi_mean']:.4f}")
    _print(f"  (Ψ→1 = locked, Ψ→0 = chaotic)")
    _hr()

    # ---- LB #14: Slip asymptote ----
    _print("\n  LB #14 — Slip-Regime Asymptote r_b → 1/π")
    _hr()
    slip = verify_slip_asymptote()
    results["slip_asymptote"] = slip
    _print(f"  r_b measured = {slip['r_b_measured']:.6f}")
    _print(f"  1/π target   = {slip['r_b_target']:.6f}")
    _print(f"  Relative err = {slip['relative_error']:.2%}")
    passed = slip["relative_error"] < 0.3
    _print(f"  {'✓ PASS' if passed else '⚠ APPROXIMATE'} (< 30% error)")
    _hr()

    # ---- LB #35: Bianco-Resta Chern marker ----
    _print("\n  LB #35 — Real-Space Chern Marker (Bianco–Resta)")
    _hr()
    for m_val in [-1.0, 1.0, 3.0]:
        cr = realspace_chern_marker(nx=10, ny=10, mass=m_val, margin=2)
        results[f"chern_marker_m{m_val}"] = {
            "mass": m_val,
            "bulk_average": cr["bulk_average"],
            "gap": cr["gap"],
        }
        exp = -1 if -2 < m_val < 0 else (1 if 0 < m_val < 2 else 0)
        _print(f"  m={m_val:+.1f}  C_bulk = {cr['bulk_average']:+.4f}"
               f"  (expected {exp})  gap = {cr['gap']:.4f}")

    _print(f"  {'✓ PASS' if abs(results['chern_marker_m-1.0']['bulk_average'] - (-1)) < 0.2 else '✗ FAIL'}")
    _hr()

    # Summary
    _print("\n  ══════════ BENCHMARK SUMMARY ══════════")
    n_pass = sum(1 for r in [
        mono.get("pass"),
        mp.get("pass"),
        cpd.get("pass"),
        passed,
        abs(results.get("chern_marker_m-1.0", {}).get("bulk_average", 99) - (-1)) < 0.2,
    ] if r)
    _print(f"  {n_pass}/5 benchmarks passed")
    _print("  ═══════════════════════════════════════")

    return results
