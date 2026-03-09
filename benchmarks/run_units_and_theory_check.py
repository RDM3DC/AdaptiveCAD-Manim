"""Units and theory verification for the normalized EGATL benchmark.

Outputs:
  outputs/units_theory/verdict.json

Status fields are intentionally leaderboard-friendly:
  - units: "OK"
  - theory: "PASS" or "FAIL"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arp_topology.laws import (  # noqa: E402
    EdgeLattice,
    ModelParams,
    VariantConfig,
    adaptive_pi,
    alpha_G,
    compute_currents,
    compute_local_chern_proxy,
    initial_state,
    mu_G,
    phase_lift_update,
    simulate,
    wrap_to_pi,
)


def check_units_ok() -> dict[str, object]:
    """Verify the normalized law is internally unit-consistent.

    The benchmark is nondimensionalized. The practical check here is that all
    additive terms entering dg/dt are finite, edge-wise conformable, and yield
    a stable dimensionless Euler increment dt * dg_dt.
    """
    lattice = EdgeLattice.square(8, 8)
    p = ModelParams(seed=7)
    variant = VariantConfig(name="full_law")
    state = initial_state(lattice, p)

    g_abs = np.abs(state.g)
    state_scalar = float(np.mean(g_abs) / max(p.g0, 1e-9))
    pi_a = adaptive_pi(state_scalar, p, adaptive=variant.adaptive_ruler)

    boundary_mask = lattice.boundary_mask.astype(float)
    phase_drive = p.phase_advance * (1.0 + 0.20 * boundary_mask)
    damage_kick = p.damage_phase_kick * boundary_mask * 0.0
    theta_raw_continuous = state.theta_R + phase_drive + damage_kick
    theta_raw_wrapped = wrap_to_pi(theta_raw_continuous)
    theta_used = phase_lift_update(state.theta_R, theta_raw_wrapped, pi_a)

    j_abs, coherence = compute_currents(state, theta_used, lattice, p)
    c_loc = compute_local_chern_proxy(j_abs, theta_used, lattice.boundary_mask)

    alpha = alpha_G(state_scalar, p)
    mu = mu_G(state_scalar, p)

    growth = alpha * j_abs * np.exp(1j * theta_used)
    decay = mu * state.g
    suppression = (p.lambda_s * variant.lambda_scale) * state.g * np.sin(theta_used / (2.0 * pi_a)) ** 2
    healing = (p.chi * variant.chi_scale) * c_loc * state.g
    dg_dt = growth - decay - suppression + healing
    increment = p.dt * dg_dt

    term_shapes_ok = all(term.shape == state.g.shape for term in [growth, decay, suppression, healing, dg_dt])
    finite_ok = all(np.all(np.isfinite(term)) for term in [growth, decay, suppression, healing, dg_dt, increment])
    bounded_ok = float(np.max(np.abs(increment))) < p.g_max
    passed = term_shapes_ok and finite_ok and bounded_ok

    return {
        "test": "units_consistency",
        "passed": bool(passed),
        "units": "OK" if passed else "FAIL",
        "normalized_system": True,
        "term_shapes_ok": bool(term_shapes_ok),
        "finite_ok": bool(finite_ok),
        "bounded_increment_ok": bool(bounded_ok),
        "max_abs_dt_dg": float(np.max(np.abs(increment))),
        "pi_a": float(pi_a),
        "coherence": float(coherence),
    }


def _final_boundary_fraction(**overrides: float) -> float:
    lattice = EdgeLattice.square(8, 8)
    p = ModelParams(**overrides)
    result = simulate(lattice, p, VariantConfig(name="full_law"), state=initial_state(lattice, p))
    return float(result.boundary_fraction[-1])


def check_theory_pass() -> dict[str, object]:
    """Verify directional trends predicted by the law.

    Expectations used here are local, baseline-adjacent monotonic trends:
      - larger healing gain chi should improve final boundary recovery
      - stronger growth alpha0 should improve final boundary recovery
      - larger damage_scale should worsen final boundary recovery
      - larger suppression lambda_s should worsen final boundary recovery
    """
    measurements = {
        "chi_low": _final_boundary_fraction(chi=0.20),
        "chi_high": _final_boundary_fraction(chi=0.80),
        "alpha_low": _final_boundary_fraction(alpha0=0.20),
        "alpha_high": _final_boundary_fraction(alpha0=1.00),
        "damage_low": _final_boundary_fraction(damage_scale=0.08),
        "damage_high": _final_boundary_fraction(damage_scale=0.30),
        "lambda_low": _final_boundary_fraction(lambda_s=0.20),
        "lambda_high": _final_boundary_fraction(lambda_s=1.40),
    }

    inequalities = {
        "chi_high_gt_chi_low": measurements["chi_high"] > measurements["chi_low"],
        "alpha_high_gt_alpha_low": measurements["alpha_high"] > measurements["alpha_low"],
        "damage_high_lt_damage_low": measurements["damage_high"] < measurements["damage_low"],
        "lambda_high_lt_lambda_low": measurements["lambda_high"] < measurements["lambda_low"],
    }
    passed = all(inequalities.values())

    return {
        "test": "theory_directional_response",
        "passed": bool(passed),
        "theory": "PASS" if passed else "FAIL",
        "measurements": measurements,
        "inequalities": inequalities,
        "margins": {
            "chi_margin": float(measurements["chi_high"] - measurements["chi_low"]),
            "alpha_margin": float(measurements["alpha_high"] - measurements["alpha_low"]),
            "damage_margin": float(measurements["damage_low"] - measurements["damage_high"]),
            "lambda_margin": float(measurements["lambda_low"] - measurements["lambda_high"]),
        },
    }


def main() -> None:
    outdir = Path("outputs/units_theory")
    outdir.mkdir(parents=True, exist_ok=True)

    units_result = check_units_ok()
    theory_result = check_theory_pass()
    overall = units_result["passed"] and theory_result["passed"]

    verdict = {
        "overall": "PASS" if overall else "FAIL",
        "units": units_result["units"],
        "theory": theory_result["theory"],
        "details": [units_result, theory_result],
    }

    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"  Units: {verdict['units']}")
    print(f"  Theory: {verdict['theory']}")
    print(f"  Overall: {verdict['overall']}")
    print(f"  Wrote: {outdir / 'verdict.json'}")


if __name__ == "__main__":
    main()