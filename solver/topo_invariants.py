"""Topological invariants for knot/link curves and defect webs.

Computes writhe, linking number, crossing signs, an Alexander-polynomial
approximation, a Kauffman bracket (Jones polynomial seed), and an
ARP-coupled knot energy that uses the μ/v decay dynamics.

Everything works on discrete polyline curves embedded in R³.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Sequence

_TAU = 2.0 * np.pi


# ---- Curve helpers ---------------------------------------------------------

def _tangent_vectors(curve: np.ndarray) -> np.ndarray:
    """Finite-difference tangent vectors along a closed discrete curve."""
    n = len(curve)
    T = np.empty_like(curve)
    for i in range(n):
        T[i] = curve[(i + 1) % n] - curve[i]
    lengths = np.linalg.norm(T, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-15)
    return T / lengths


def _segments(curve: np.ndarray):
    """Yield (start, end) segment pairs for a closed curve."""
    n = len(curve)
    for i in range(n):
        yield curve[i], curve[(i + 1) % n]


# ---- Crossing detection ---------------------------------------------------

def _seg_seg_crossing(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    tol: float = 1e-10,
) -> Optional[Tuple[float, float, int]]:
    """Detect projected crossing between two 3D segments.

    Projects to XY and checks for intersection.
    Returns (t_a, t_b, sign) where sign ∈ {+1, -1} based on
    which strand passes over (higher z at crossing).
    """
    # 2D intersection in XY
    d1 = a2[:2] - a1[:2]
    d2 = b2[:2] - b1[:2]
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < tol:
        return None

    dp = b1[:2] - a1[:2]
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / det
    s = (dp[0] * d1[1] - dp[1] * d1[0]) / det

    if not (0 < t < 1 and 0 < s < 1):
        return None

    # z-values at the crossing
    za = a1[2] + t * (a2[2] - a1[2])
    zb = b1[2] + s * (b2[2] - b1[2])
    sign = 1 if za > zb else -1

    # Crossing sign from orientation (right-hand rule on tangent cross)
    cross_z = d1[0] * d2[1] - d1[1] * d2[0]
    sign *= 1 if cross_z > 0 else -1

    return t, s, sign


# ---- Public API: single-curve invariants ----------------------------------

def crossing_sign(
    curve: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Find all self-crossings of a closed curve in XY projection.

    Returns list of (segment_i, segment_j, sign) where sign is ±1.
    """
    n = len(curve)
    crossings = []
    for i in range(n):
        a1, a2 = curve[i], curve[(i + 1) % n]
        for j in range(i + 2, n):
            if j == (i - 1) % n or j == (i + 1) % n:
                continue
            b1, b2 = curve[j], curve[(j + 1) % n]
            result = _seg_seg_crossing(a1, a2, b1, b2)
            if result is not None:
                crossings.append((i, j, result[2]))
    return crossings


def writhe(curve: np.ndarray) -> float:
    """Compute the writhe of a closed space curve.

    Uses the Gauss integral discretised over segment pairs:

        Wr = (1/4π) Σ_i Σ_j Ω(i,j)

    where Ω is the signed solid angle subtended by segments i,j.
    """
    n = len(curve)
    total = 0.0
    for i in range(n):
        a1, a2 = curve[i], curve[(i + 1) % n]
        da = a2 - a1
        for j in range(i + 2, n):
            if j == (i - 1) % n:
                continue
            b1, b2 = curve[j], curve[(j + 1) % n]
            db = b2 - b1

            # Gauss linking integral kernel
            r = a1 - b1
            rn = np.linalg.norm(r)
            if rn < 1e-12:
                continue

            cross = np.cross(da, db)
            total += np.dot(cross, r) / (rn ** 3)

    return total / (4.0 * np.pi)


def linking_number(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
) -> float:
    """Gauss linking number between two closed space curves.

        Lk = (1/4π) Σ_i∈A Σ_j∈B (da × db) · r / |r|³
    """
    na, nb = len(curve_a), len(curve_b)
    total = 0.0
    for i in range(na):
        a1, a2 = curve_a[i], curve_a[(i + 1) % na]
        da = a2 - a1
        for j in range(nb):
            b1, b2 = curve_b[j], curve_b[(j + 1) % nb]
            db = b2 - b1

            r = a1 - b1
            rn = np.linalg.norm(r)
            if rn < 1e-12:
                continue

            cross = np.cross(da, db)
            total += np.dot(cross, r) / (rn ** 3)

    return total / (4.0 * np.pi)


# ---- Alexander polynomial (combinatorial approximation) -------------------

def alexander_polynomial(
    curve: np.ndarray,
    var_values: Optional[Sequence[complex]] = None,
) -> np.ndarray:
    """Approximate Alexander polynomial Δ(t) from crossing data.

    Uses the Seifert matrix approach: constructs the matrix from
    crossing signs, computes det(V - t·V^T) at sample points.

    Parameters
    ----------
    curve : (N,3) closed polyline
    var_values : points at which to evaluate Δ(t).
        Defaults to [exp(2πi·k/8) for k=0..7] (unit circle samples).

    Returns
    -------
    values : complex ndarray of Δ(t) evaluated at var_values.
    """
    crossings = crossing_sign(curve)
    nc = len(crossings)
    if nc == 0:
        if var_values is None:
            var_values = np.exp(1j * np.linspace(0, _TAU, 8, endpoint=False))
        return np.ones(len(var_values), dtype=complex)

    # Build Seifert matrix from crossing data
    # Approximate: each crossing contributes ±1 to matrix entries
    n_gen = nc  # one generator per crossing (simplified)
    V = np.zeros((n_gen, n_gen))
    for k, (si, sj, sign) in enumerate(crossings):
        V[k, k] = sign * 0.5
        if k + 1 < n_gen:
            V[k, k + 1] = sign * 0.5
            V[k + 1, k] = 0

    if var_values is None:
        var_values = np.exp(1j * np.linspace(0, _TAU, 8, endpoint=False))
    var_values = np.asarray(var_values, dtype=complex)

    results = np.empty(len(var_values), dtype=complex)
    for idx, t in enumerate(var_values):
        M = V - t * V.T
        results[idx] = np.linalg.det(M) if n_gen > 0 else 1.0

    return results


# ---- Jones bracket (Kauffman bracket state sum) ----------------------------

def jones_bracket(curve: np.ndarray, A: complex = None) -> complex:
    """Kauffman bracket <K> at parameter A.

    Uses crossing signs to compute the state-sum:
        <K> = Σ_states A^(σ(state)) · (-A² - A⁻²)^(loops(state)-1)

    For a knot with c crossings, this is O(2^c) — fine for small c.
    Default A solves the Jones polynomial variable: A = exp(iπ/4).
    """
    if A is None:
        A = np.exp(1j * np.pi / 4)

    crossings = crossing_sign(curve)
    nc = len(crossings)
    if nc == 0:
        return complex(1.0)

    # Cap at 16 crossings for feasibility
    if nc > 16:
        crossings = crossings[:16]
        nc = 16

    loop_factor = -A ** 2 - A ** (-2)
    bracket = complex(0.0)

    for state in range(1 << nc):
        sigma = 0
        # Count A-smoothings (bit=0) vs B-smoothings (bit=1)
        for k in range(nc):
            sign = crossings[k][2]
            if (state >> k) & 1:
                sigma -= sign
            else:
                sigma += sign

        # Estimate loop count from Euler characteristic
        # Each smoothing produces some number of loops
        n_a = bin(state).count("0") - (32 - nc)  # A-smoothings
        n_b = bin(state).count("1")  # B-smoothings
        # Simplified: loops ≈ |crossings| - |connections| + 1
        n_loops = max(1, abs(n_a - n_b) + 1)

        bracket += A ** sigma * loop_factor ** (n_loops - 1)

    return bracket


# ---- ARP knot energy -------------------------------------------------------

def arp_knot_energy(
    curve: np.ndarray,
    mu: float = 0.8,
    v: float = 1.2,
    alpha_0: float = 1.0,
    n_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ARP-coupled knot energy over time.

    The energy functional is:
        E(t) = E_bend(t) + E_repulsion(t)
    modulated by the ARP decay:
        α(t) = α₀ · exp(-μt) · cos(v·t)

    Parameters
    ----------
    curve : (N,3) closed polyline
    mu, v : ARP decay parameters
    alpha_0 : initial coupling strength
    n_steps : number of time samples

    Returns
    -------
    times : (n_steps,) time values
    alphas : (n_steps,) ARP coupling α(t)
    energies : (n_steps,) total energy E(t)
    """
    n = len(curve)
    times = np.linspace(0, 5, n_steps)

    # Bending energy: sum of (1 - cos θ) at each vertex
    T = _tangent_vectors(curve)
    bend_base = 0.0
    for i in range(n):
        cos_th = np.clip(np.dot(T[i], T[(i + 1) % n]), -1, 1)
        bend_base += 1.0 - cos_th

    # Repulsion energy: Coulomb-like ΣΣ 1/|r_ij|
    repulsion_base = 0.0
    for i in range(n):
        for j in range(i + 2, n):
            if j == (i - 1) % n:
                continue
            r = np.linalg.norm(curve[i] - curve[j])
            if r > 1e-8:
                repulsion_base += 1.0 / r

    alphas = alpha_0 * np.exp(-mu * times) * np.cos(v * times)
    energies = np.abs(alphas) * bend_base + (1.0 - np.abs(alphas)) * repulsion_base

    return times, alphas, energies


# ---- Curve generators for testing ------------------------------------------

def trefoil_curve(n_pts: int = 200, scale: float = 1.2) -> np.ndarray:
    """Generate a discrete trefoil knot curve."""
    t = np.linspace(0, _TAU, n_pts, endpoint=False)
    x = scale * (np.sin(t) + 2 * np.sin(2 * t))
    y = scale * (np.cos(t) - 2 * np.cos(2 * t))
    z = scale * (-np.sin(3 * t))
    return np.column_stack([x, y, z])


def hopf_link_curves(
    alpha: float = 1.0,
    n_pts: int = 100,
    R: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two linked circles (Hopf link) at coupling α.

    α=1 → fully linked, α=0 → separated.
    """
    t = np.linspace(0, _TAU, n_pts, endpoint=False)
    # Component A: circle in XY
    a_x = R * np.cos(t)
    a_y = R * np.sin(t)
    a_z = alpha * 0.3 * np.sin(t)
    curve_a = np.column_stack([a_x, a_y, a_z])

    # Component B: threaded when α=1
    offset = alpha * R
    b_x = offset * np.cos(t) * 0.3 + R * 0.5 * (1 - alpha)
    b_y = R * np.cos(t)
    b_z = R * np.sin(t) + alpha * 0.2
    curve_b = np.column_stack([b_x, b_y, b_z])

    return curve_a, curve_b


def figure_eight_curve(n_pts: int = 200, scale: float = 1.0) -> np.ndarray:
    """Generate a discrete figure-eight knot (4₁) curve."""
    t = np.linspace(0, _TAU, n_pts, endpoint=False)
    x = scale * (2 + np.cos(2 * t)) * np.cos(3 * t)
    y = scale * (2 + np.cos(2 * t)) * np.sin(3 * t)
    z = scale * np.sin(4 * t)
    return np.column_stack([x, y, z])


def torus_knot_curve(
    p: int = 2,
    q: int = 3,
    n_pts: int = 200,
    R: float = 2.0,
    r: float = 0.8,
) -> np.ndarray:
    """Generate a (p,q) torus knot on a torus with radii R, r."""
    t = np.linspace(0, _TAU, n_pts, endpoint=False)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return np.column_stack([x, y, z])
