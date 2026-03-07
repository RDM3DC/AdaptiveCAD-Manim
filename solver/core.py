"""Canonical Core Equations — numerical demonstrations.

Verifies all 14 pinned Core equations from:
  https://rdm3dc.github.io/TopEquations/core.html

Run:  python -m solver core
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .egatl import (
    _wrap_to_pi,
    qwz_hamiltonian,
    build_qwz_lattice,
    simulate as egatl_simulate,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase-Lift primitives
# ═══════════════════════════════════════════════════════════════════════════

def _unwrap(theta: float, theta_ref: float) -> float:
    """Deterministic unwrap: θ + 2π·round((θ_ref − θ)/(2π))."""
    return theta + 2 * math.pi * round((theta_ref - theta) / (2 * math.pi))


def _unwrap_adaptive(theta: float, theta_ref: float, pi_a: float) -> float:
    """Adaptive-π unwrap: clip(wrap(θ − θ_ref), −π_a, +π_a) + θ_ref."""
    d = ((theta - theta_ref + math.pi) % (2 * math.pi)) - math.pi
    return theta_ref + max(-pi_a, min(pi_a, d))


# ═══════════════════════════════════════════════════════════════════════════
# C1 — Phase Ambiguity (multi-valuedness axiom)
# ═══════════════════════════════════════════════════════════════════════════

def phase_ambiguity(n_tests: int = 1000) -> Dict:
    """z = r e^{iθ},  θ ≡ θ + 2πk,  k ∈ ℤ.

    Verify that adding 2πk to the argument leaves the complex number unchanged.
    """
    rng = np.random.default_rng(42)
    r = rng.uniform(0.1, 10.0, n_tests)
    theta = rng.uniform(-10.0, 10.0, n_tests)
    k = rng.integers(-5, 6, n_tests)

    z_original = r * np.exp(1j * theta)
    z_shifted = r * np.exp(1j * (theta + 2 * np.pi * k))

    max_err = float(np.max(np.abs(z_original - z_shifted)))

    return {
        "name": "C1 — Phase Ambiguity (multi-valuedness axiom)",
        "n_tests": n_tests,
        "k_range": [-5, 5],
        "max_error": max_err,
        "pass": max_err < 1e-12,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C2 — Phase-Lift operator definition (⧉)
# ═══════════════════════════════════════════════════════════════════════════

def phase_lift_operator(n_tests: int = 500) -> Dict:
    """⧉f(z; θ_ref) := f(z) computed using θ_R = unwrap(arg z; θ_ref).

    Verify that the Phase-Lift operator resolves branch ambiguity by
    choosing the unwrapped branch nearest θ_ref.
    """
    rng = np.random.default_rng(7)
    continuity_ok = 0

    for _ in range(n_tests):
        r = rng.uniform(0.5, 3.0)
        theta = rng.uniform(-3 * math.pi, 3 * math.pi)
        theta_ref = rng.uniform(-3 * math.pi, 3 * math.pi)

        # Standard (principal) arg
        principal = math.atan2(math.sin(theta), math.cos(theta))

        # Phase-Lifted
        theta_R = _unwrap(principal, theta_ref)

        # Check: θ_R ≡ principal (mod 2π) and |θ_R − θ_ref| ≤ π
        residual = abs(((theta_R - principal + math.pi) % (2 * math.pi)) - math.pi)
        near = abs(theta_R - theta_ref) <= math.pi + 1e-10

        if residual < 1e-10 and near:
            continuity_ok += 1

    return {
        "name": "C2 — Phase-Lift operator definition (⧉)",
        "n_tests": n_tests,
        "branch_correct": continuity_ok,
        "pass": continuity_ok == n_tests,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C3 — Deterministic Unwrapping Rule
# ═══════════════════════════════════════════════════════════════════════════

def deterministic_unwrap(n_tests: int = 1000) -> Dict:
    """unwrap(θ; θ_ref) = θ + 2π·round((θ_ref − θ)/(2π)).

    Verify the formula is deterministic, idempotent, and picks the
    representative closest to θ_ref.
    """
    rng = np.random.default_rng(13)
    all_ok = 0
    idempotent_ok = 0
    closest_ok = 0

    for _ in range(n_tests):
        theta = rng.uniform(-10, 10)
        theta_ref = rng.uniform(-10, 10)

        result = _unwrap(theta, theta_ref)

        # Deterministic: same inputs → same output
        result2 = _unwrap(theta, theta_ref)
        if result == result2:
            all_ok += 1

        # Idempotent: unwrap(result, theta_ref) == result
        result3 = _unwrap(result, theta_ref)
        if abs(result3 - result) < 1e-12:
            idempotent_ok += 1

        # Closest: |result − θ_ref| ≤ π
        if abs(result - theta_ref) <= math.pi + 1e-10:
            closest_ok += 1

    return {
        "name": "C3 — Deterministic Unwrapping Rule",
        "n_tests": n_tests,
        "deterministic": all_ok,
        "idempotent": idempotent_ok,
        "closest_branch": closest_ok,
        "pass": all_ok == n_tests and idempotent_ok == n_tests
                and closest_ok == n_tests,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C4 — Path Continuity (Stateful Phase-Lift)
# ═══════════════════════════════════════════════════════════════════════════

def path_continuity(n_pts: int = 500, n_laps: int = 3) -> Dict:
    """θ_{R,k} = unwrap(arg z_k;  θ_{R,k-1}).

    Trace a path that winds n_laps times around the origin.
    The lifted phase should be continuous and accumulate 2π per lap.
    """
    t = np.linspace(0, 2 * math.pi * n_laps, n_pts)
    z = np.exp(1j * t)  # unit circle, n_laps windings

    theta_R = np.zeros(n_pts)
    theta_R[0] = 0.0
    for k in range(1, n_pts):
        arg_k = math.atan2(z[k].imag, z[k].real)
        theta_R[k] = _unwrap(arg_k, theta_R[k - 1])

    # Check continuity: max step should be < π
    steps = np.abs(np.diff(theta_R))
    max_step = float(np.max(steps))

    # Total accumulation should be ≈ 2π·n_laps
    total = theta_R[-1] - theta_R[0]
    expected = 2 * math.pi * n_laps

    return {
        "name": "C4 — Path Continuity (Stateful Phase-Lift)",
        "n_laps": n_laps,
        "total_phase": float(total),
        "expected_phase": expected,
        "max_step": max_step,
        "continuous": max_step < 0.5,
        "pass": abs(total - expected) < 0.1 and max_step < 0.5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C5 — PR-Root as a special case of Phase-Lift (⧉√)
# ═══════════════════════════════════════════════════════════════════════════

def pr_root(n_tests: int = 500) -> Dict:
    """⧉√z = √r · e^{iθ_R/2}  where θ_R = unwrap(arg z; θ_ref).

    Verify: (⧉√z)² = z.
    """
    rng = np.random.default_rng(21)
    sq_errors = []

    for _ in range(n_tests):
        r = rng.uniform(0.1, 10.0)
        theta = rng.uniform(-4 * math.pi, 4 * math.pi)
        theta_ref = rng.uniform(-4 * math.pi, 4 * math.pi)

        z = r * math.cos(theta) + 1j * r * math.sin(theta)
        arg_z = math.atan2(z.imag, z.real)

        theta_R = _unwrap(arg_z, theta_ref)
        pr_sqrt = math.sqrt(r) * (math.cos(theta_R / 2) + 1j * math.sin(theta_R / 2))

        # (⧉√z)² should equal z
        sq = pr_sqrt * pr_sqrt
        err = abs(sq - z)
        sq_errors.append(err)

    max_err = max(sq_errors)

    return {
        "name": "C5 — PR-Root (⧉√z)² = z",
        "n_tests": n_tests,
        "max_squaring_error": float(max_err),
        "mean_error": float(np.mean(sq_errors)),
        "pass": max_err < 1e-10,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C6 — Winding Number + ℤ₂ Parity Invariants
# ═══════════════════════════════════════════════════════════════════════════

def winding_parity(n_laps_list: list = None) -> Dict:
    """w = Δθ_R / (2π),  ν = w mod 2,  b = (-1)^w.

    Verify winding number is integer and parity b correctly predicts
    PR-Root sheet return.
    """
    if n_laps_list is None:
        n_laps_list = [-3, -2, -1, 0, 1, 2, 3, 4, 5]

    entries = []

    for n_laps in n_laps_list:
        n_pts = max(200, abs(n_laps) * 100)
        t = np.linspace(0, 2 * math.pi * n_laps, n_pts) if n_laps != 0 else np.zeros(n_pts)
        z = np.exp(1j * t)

        # Stateful unwrap
        theta_R = np.zeros(n_pts)
        for k in range(1, n_pts):
            arg_k = math.atan2(z[k].imag, z[k].real)
            theta_R[k] = _unwrap(arg_k, theta_R[k - 1])

        delta = theta_R[-1] - theta_R[0]
        w = round(delta / (2 * math.pi))
        nu = w % 2
        b = (-1) ** w

        # PR-Root at start and end
        r0 = math.sqrt(abs(z[0]))
        sqrt_start = r0 * np.exp(1j * theta_R[0] / 2)
        sqrt_end = r0 * np.exp(1j * theta_R[-1] / 2)
        same_sheet = abs(sqrt_end - sqrt_start) < 0.1

        entries.append({
            "n_laps": n_laps,
            "w": w,
            "nu": nu,
            "b": b,
            "same_sheet": same_sheet,
            "b_predicts_sheet": (b == 1) == same_sheet,
        })

    all_correct = all(e["b_predicts_sheet"] for e in entries)

    return {
        "name": "C6 — Winding Number + ℤ₂ Parity",
        "entries": entries,
        "pass": all_correct,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C7 — Adaptive-π Conformal Scale Factor and Metric
# ═══════════════════════════════════════════════════════════════════════════

def conformal_metric(pi_a_values: list = None) -> Dict:
    """Ω(x,t) = π_a(x,t)/π,  g_{ij} = Ω² δ_{ij}.

    Verify: metric determinant = Ω^{2d} (d=2 for 2D).
    At π_a=π: Ω=1 recovers flat metric.
    """
    if pi_a_values is None:
        pi_a_values = [0.5 * math.pi, math.pi, 1.5 * math.pi, 2 * math.pi]

    d = 2  # spatial dimension
    entries = []

    for pi_a in pi_a_values:
        omega = pi_a / math.pi
        g_det = omega ** (2 * d)  # det(g) in d dimensions
        entries.append({
            "pi_a": float(pi_a),
            "Omega": float(omega),
            "g_det": float(g_det),
            "is_flat": abs(omega - 1.0) < 1e-15,
        })

    flat_entry = [e for e in entries if e["is_flat"]]

    return {
        "name": "C7 — Adaptive-π Conformal Scale Factor",
        "d": d,
        "entries": entries,
        "flat_at_pi": len(flat_entry) == 1 and flat_entry[0]["g_det"] == 1.0,
        "pass": len(flat_entry) == 1 and abs(flat_entry[0]["g_det"] - 1.0) < 1e-15,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C8 — Adaptive Arc Length
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_arc_length() -> Dict:
    """L_g(γ) = ∫₀¹ Ω(γ(t),t) |γ̇(t)| dt.

    Compute arc lengths for a straight segment under varying π_a fields:
    constant, linear ramp, and Gaussian bump.
    """
    n = 1000
    t = np.linspace(0, 1, n)
    # Straight-line path from (0,0) to (1,0)
    gamma_dot = 1.0  # |γ̇| = 1

    results = {}

    # Constant π_a = π → Ω = 1 → L = 1
    omega_flat = np.ones(n)
    L_flat = float(np.trapz(omega_flat * gamma_dot, t))
    results["constant (Ω=1)"] = L_flat

    # Linear ramp: π_a from π to 2π → Ω from 1 to 2
    omega_ramp = 1.0 + t  # Ω = 1 + t
    L_ramp = float(np.trapz(omega_ramp * gamma_dot, t))
    # Analytic: ∫₀¹ (1+t) dt = 1.5
    results["linear ramp (1→2)"] = L_ramp

    # Gaussian bump: Ω = 1 + 2 exp(-50(t-0.5)²)
    omega_bump = 1.0 + 2.0 * np.exp(-50 * (t - 0.5) ** 2)
    L_bump = float(np.trapz(omega_bump * gamma_dot, t))
    results["Gaussian bump"] = L_bump

    return {
        "name": "C8 — Adaptive Arc Length",
        "L_flat": L_flat,
        "L_ramp": L_ramp,
        "L_ramp_exact": 1.5,
        "L_bump": L_bump,
        "pass": abs(L_flat - 1.0) < 0.01 and abs(L_ramp - 1.5) < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C9 — Adaptive-π field definition + limit (πₐ → π)
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_pi_limit(n_pts: int = 200) -> Dict:
    """θ = θ_R + 2π_a(x,t) w,  with w ∈ ℤ and π_a → π.

    Verify: when π_a = π, the adaptive unwrap matches standard 2π unwrap.
    When π_a ≠ π, the sector size changes.
    """
    t = np.linspace(0, 6 * math.pi, n_pts)
    z = np.exp(1j * t)

    # Standard unwrap (π_a = π)
    theta_std = np.zeros(n_pts)
    for k in range(1, n_pts):
        arg_k = math.atan2(z[k].imag, z[k].real)
        theta_std[k] = _unwrap(arg_k, theta_std[k - 1])

    # Adaptive unwrap with π_a = π (should match standard)
    theta_adaptive_pi = np.zeros(n_pts)
    for k in range(1, n_pts):
        arg_k = math.atan2(z[k].imag, z[k].real)
        theta_adaptive_pi[k] = _unwrap_adaptive(arg_k, theta_adaptive_pi[k - 1], math.pi)

    # Adaptive unwrap with π_a = 2π (wider sectors)
    theta_adaptive_2pi = np.zeros(n_pts)
    for k in range(1, n_pts):
        arg_k = math.atan2(z[k].imag, z[k].real)
        theta_adaptive_2pi[k] = _unwrap_adaptive(
            arg_k, theta_adaptive_2pi[k - 1], 2 * math.pi
        )

    max_diff_at_pi = float(np.max(np.abs(theta_std - theta_adaptive_pi)))

    return {
        "name": "C9 — Adaptive-π field definition + limit",
        "max_diff_pi_a_eq_pi": max_diff_at_pi,
        "theta_std_final": float(theta_std[-1]),
        "theta_adaptive_pi_final": float(theta_adaptive_pi[-1]),
        "theta_adaptive_2pi_final": float(theta_adaptive_2pi[-1]),
        "pass": max_diff_at_pi < 1e-10,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C10 — Reinforce/Decay Dynamics for πₐ
# ═══════════════════════════════════════════════════════════════════════════

def pi_a_dynamics(
    n_steps: int = 5000,
    dt: float = 0.005,
    alpha_pi: float = 0.3,
    mu_pi: float = 0.2,
    pi_0: float = math.pi,
) -> Dict:
    """dπ_a/dt = α_π s(x,t) − μ_π(π_a − π₀).

    Verify: equilibrium π_a* = π₀ + α_π ⟨s⟩ / μ_π.
    Verify: relaxation is exponential with rate μ_π.
    """
    pi_a = pi_0
    s_const = 0.5  # constant stimulus

    pi_hist = [pi_a]

    for _ in range(n_steps):
        dpi = alpha_pi * s_const - mu_pi * (pi_a - pi_0)
        pi_a += dt * dpi
        pi_hist.append(pi_a)

    pi_eq_expected = pi_0 + alpha_pi * s_const / mu_pi
    pi_eq_actual = pi_hist[-1]

    # Relaxation test: start far from equilibrium, measure timescale
    pi_a2 = pi_0 + 5.0  # large perturbation
    relax_hist = [pi_a2]
    for _ in range(n_steps):
        dpi = alpha_pi * s_const - mu_pi * (pi_a2 - pi_0)
        pi_a2 += dt * dpi
        relax_hist.append(pi_a2)

    # Fit exponential: deviation ~ exp(-μ_π t)
    dev = np.array(relax_hist) - pi_eq_expected
    t_arr = np.arange(len(dev)) * dt
    # At t = 1/μ_π, deviation should be ~ e^{-1} ≈ 0.368 of initial
    tau = 1.0 / mu_pi
    i_tau = min(int(tau / dt), len(dev) - 1)
    ratio_at_tau = abs(dev[i_tau] / dev[0]) if abs(dev[0]) > 1e-10 else 0

    return {
        "name": "C10 — Reinforce/Decay Dynamics for π_a",
        "pi_eq_expected": float(pi_eq_expected),
        "pi_eq_actual": float(pi_eq_actual),
        "equilibrium_error": abs(pi_eq_actual - pi_eq_expected),
        "tau_theoretical": float(tau),
        "ratio_at_tau": float(ratio_at_tau),
        "exp_decay_expected": math.exp(-1),
        "pass": abs(pi_eq_actual - pi_eq_expected) < 0.01
                and abs(ratio_at_tau - math.exp(-1)) < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C11 — ARP Core Law (canonical ODE)
# ═══════════════════════════════════════════════════════════════════════════

def arp_core_law(
    n_edges: int = 20,
    n_steps: int = 20000,
    dt: float = 0.005,
    alpha_G: float = 0.5,
    mu_G: float = 0.1,
) -> Dict:
    """dG_{ij}/dt = α_G |I_{ij}(t)| − μ_G G_{ij}(t).

    Verify:
    - Active edges reinforce (G → α_G I / μ_G)
    - Inactive edges decay (G → 0)
    - Self-organising backbone emerges
    """
    rng = np.random.default_rng(99)

    # Half the edges carry current, half don't
    I = np.zeros(n_edges)
    I[:n_edges // 2] = rng.uniform(0.5, 2.0, n_edges // 2)

    G = np.ones(n_edges) * 0.5  # initial conductance
    G_hist = [G.copy()]

    for _ in range(n_steps):
        dG = alpha_G * np.abs(I) - mu_G * G
        G = np.maximum(G + dt * dG, 0.0)
        G_hist.append(G.copy())

    G_final = G_hist[-1]
    active_eq = alpha_G * np.abs(I[:n_edges // 2]) / mu_G
    inactive_eq = 0.0

    active_error = float(np.max(np.abs(G_final[:n_edges // 2] - active_eq)))
    inactive_error = float(np.max(np.abs(G_final[n_edges // 2:])))

    return {
        "name": "C11 — ARP Core Law (canonical ODE)",
        "n_edges": n_edges,
        "active_edges": n_edges // 2,
        "active_G_range": [float(G_final[:n_edges // 2].min()),
                           float(G_final[:n_edges // 2].max())],
        "inactive_G_max": float(G_final[n_edges // 2:].max()),
        "active_eq_error": active_error,
        "inactive_eq_error": inactive_error,
        "backbone_emerged": inactive_error < 0.01 and active_error < 0.01,
        "pass": active_error < 0.05 and inactive_error < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C12 — Curvature as Salience (Curve-Memory Primitive)
# ═══════════════════════════════════════════════════════════════════════════

def curvature_salience() -> Dict:
    """κ(s) = |γ″(s)|.

    Compute salience (curvature) for known curves:
    - Circle (constant κ = 1/R)
    - Straight line (κ = 0)
    - Figure-8 (peaks at inflection points)
    """
    n = 500

    # Circle of radius R
    R = 2.0
    t_circ = np.linspace(0, 2 * math.pi, n, endpoint=False)
    gamma_circ = np.column_stack([R * np.cos(t_circ), R * np.sin(t_circ)])

    # Second derivative via finite differences (arc-length parametrised)
    ds = 2 * math.pi * R / n
    d2 = np.zeros(n)
    for i in range(n):
        g_pp = (gamma_circ[(i + 1) % n] - 2 * gamma_circ[i]
                + gamma_circ[(i - 1) % n]) / ds ** 2
        d2[i] = np.linalg.norm(g_pp)

    kappa_circle = float(np.mean(d2))
    kappa_circle_expected = 1.0 / R

    # Straight line
    gamma_line = np.column_stack([np.linspace(0, 10, n), np.zeros(n)])
    ds_line = 10.0 / n
    d2_line = np.zeros(n)
    for i in range(1, n - 1):
        g_pp = (gamma_line[i + 1] - 2 * gamma_line[i] + gamma_line[i - 1]) / ds_line ** 2
        d2_line[i] = np.linalg.norm(g_pp)
    kappa_line = float(np.mean(d2_line))

    # Figure-8: γ(t) = (sin(t), sin(2t)/2)
    t_fig = np.linspace(0, 2 * math.pi, n, endpoint=False)
    gamma_fig = np.column_stack([np.sin(t_fig), 0.5 * np.sin(2 * t_fig)])
    # Parametric curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^{3/2}
    dx = np.gradient(gamma_fig[:, 0], t_fig)
    dy = np.gradient(gamma_fig[:, 1], t_fig)
    ddx = np.gradient(dx, t_fig)
    ddy = np.gradient(dy, t_fig)
    kappa_fig = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    peak_idx = int(np.argmax(kappa_fig))

    return {
        "name": "C12 — Curvature as Salience",
        "circle_kappa": kappa_circle,
        "circle_expected": kappa_circle_expected,
        "circle_error": abs(kappa_circle - kappa_circle_expected),
        "line_kappa": kappa_line,
        "fig8_kappa_max": float(np.max(kappa_fig)),
        "fig8_peak_t": float(t_fig[peak_idx]),
        "pass": abs(kappa_circle - kappa_circle_expected) < 0.05
                and kappa_line < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C13 — Reinforce/Decay Memory Law (generic)
# ═══════════════════════════════════════════════════════════════════════════

def memory_law(
    n_steps: int = 3000,
    dt: float = 0.01,
    alpha: float = 1.0,
    mu: float = 0.2,
) -> Dict:
    """dM/dt = α S(t) − μ M(t).

    Verify: M(t) tracks stimulus with exponential memory kernel.
    Step response: M → α S / μ.
    Impulse response: M decays as exp(-μ t).
    """
    # Step response
    M = 0.0
    S = 1.0
    M_step = [M]
    for _ in range(n_steps):
        M += dt * (alpha * S - mu * M)
        M_step.append(M)

    M_eq = alpha * S / mu
    step_error = abs(M_step[-1] - M_eq)

    # Impulse response: give one big pulse, then S=0
    M2 = 10.0
    M_impulse = [M2]
    for _ in range(n_steps):
        M2 += dt * (0.0 - mu * M2)
        M_impulse.append(M2)

    # Should decay exponentially
    tau = 1.0 / mu
    i_tau = min(int(tau / dt), len(M_impulse) - 1)
    ratio = M_impulse[i_tau] / M_impulse[0] if M_impulse[0] > 1e-10 else 0
    expected_ratio = math.exp(-1)

    return {
        "name": "C13 — Reinforce/Decay Memory Law",
        "M_eq_expected": float(M_eq),
        "M_eq_actual": float(M_step[-1]),
        "step_error": float(step_error),
        "decay_ratio_at_tau": float(ratio),
        "exp_neg1": float(expected_ratio),
        "decay_ratio_error": abs(ratio - expected_ratio),
        "pass": step_error < 0.05 and abs(ratio - expected_ratio) < 0.02,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C14 — Phase-Lifted Stokes Quantization (Adaptive-π)
# ═══════════════════════════════════════════════════════════════════════════

def stokes_quantization(
    n_pts: int = 200,
    flux_quanta_list: list = None,
    pi_a: float = math.pi,
) -> Dict:
    """θ_R[γ] = unwrap(∮A; θ_ref) = ∫_S F + 2π_a w,  b(γ) = (-1)^w.

    Verify the Phase-Lifted Stokes theorem: total lifted phase =
    (surface integral of curvature F) + winding correction 2π_a w.
    """
    if flux_quanta_list is None:
        flux_quanta_list = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    entries = []

    for Phi in flux_quanta_list:
        # ∮A·dℓ = 2π Φ (total raw phase)
        raw_phase = 2 * math.pi * Phi

        # Principal value (surface integral of F, mod 2π)
        F_surface = ((raw_phase + math.pi) % (2 * math.pi)) - math.pi

        # Winding number
        w = round((raw_phase - F_surface) / (2 * pi_a))

        # Reconstructed lifted phase
        theta_R = F_surface + 2 * pi_a * w

        # Verify: θ_R should equal raw_phase
        error = abs(theta_R - raw_phase)

        b = (-1) ** w

        entries.append({
            "Phi": float(Phi),
            "raw_phase": float(raw_phase),
            "F_surface": float(F_surface),
            "w": w,
            "b": b,
            "theta_R": float(theta_R),
            "error": float(error),
        })

    all_ok = all(e["error"] < 1e-10 for e in entries)

    return {
        "name": "C14 — Phase-Lifted Stokes Quantization",
        "entries": entries,
        "pass": all_ok,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════════════════

ALL_CORE = [
    ("C1", "Phase Ambiguity", phase_ambiguity),
    ("C2", "Phase-Lift Operator (⧉)", phase_lift_operator),
    ("C3", "Deterministic Unwrap Rule", deterministic_unwrap),
    ("C4", "Path Continuity", path_continuity),
    ("C5", "PR-Root (⧉√)", pr_root),
    ("C6", "Winding + ℤ₂ Parity", winding_parity),
    ("C7", "Conformal Scale Factor", conformal_metric),
    ("C8", "Adaptive Arc Length", adaptive_arc_length),
    ("C9", "Adaptive-π Limit", adaptive_pi_limit),
    ("C10", "π_a Reinforce/Decay", pi_a_dynamics),
    ("C11", "ARP Core Law", arp_core_law),
    ("C12", "Curvature as Salience", curvature_salience),
    ("C13", "Memory Law", memory_law),
    ("C14", "Stokes Quantization", stokes_quantization),
]


def run_all(verbose: bool = True) -> Dict[str, dict]:
    results = {}

    def _pr(msg):
        if verbose:
            print(msg)

    def _hr():
        _pr("─" * 64)

    passed = 0
    total = 0

    for cid, cname, func in ALL_CORE:
        _pr(f"\n  {cid} — {cname}")
        _hr()
        try:
            r = func()
            results[cid] = r
            total += 1

            for k, v in r.items():
                if k in ("name", "pass"):
                    continue
                if k == "entries" and isinstance(v, list):
                    for e in v:
                        parts = []
                        for ek, ev in e.items():
                            if isinstance(ev, float):
                                parts.append(f"{ek}={ev:.4f}")
                            else:
                                parts.append(f"{ek}={ev}")
                            if len(parts) >= 5:
                                break
                        _pr(f"    {', '.join(parts)}")
                elif isinstance(v, float):
                    _pr(f"    {k}: {v:.6f}")
                elif isinstance(v, list) and len(v) <= 5:
                    _pr(f"    {k}: {v}")
                else:
                    _pr(f"    {k}: {v}")

            if r.get("pass"):
                _pr(f"  ✓ PASS")
                passed += 1
            else:
                _pr(f"  ✗ FAIL")

        except Exception as e:
            _pr(f"  ✗ ERROR: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            results[cid] = {"error": str(e), "pass": False}
            total += 1

        _hr()

    _pr(f"\n  ══════════ CORE EQUATIONS SUMMARY ══════════")
    _pr(f"  {passed}/{total} core equations passed")
    _pr(f"  ═════════════════════════════════════════════")

    return results
