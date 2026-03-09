"""Onset-map parameter sweeps.

Sweep key ModelParams (alpha0, lambda_s, chi, damage_scale) over 2-D grids
and record whether self-healing recovery occurs (boundary_fraction recovers
above 90 % of its pre-damage reference within the simulation window).

Outputs:
  outputs/onset_map/onset_alpha0_lambda_s.csv
  outputs/onset_map/onset_alpha0_lambda_s.png
  outputs/onset_map/onset_chi_damage_scale.csv
  outputs/onset_map/onset_chi_damage_scale.png
  outputs/onset_map/summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arp_topology.laws import (
    EdgeLattice,
    ModelParams,
    VariantConfig,
    initial_state,
    simulate,
)

OUTDIR = Path("outputs/onset_map")


def _recovery_metric(lattice, base_params, overrides, variant):
    """Run a simulation with overrides and return (boundary_fraction_final, recovered)."""
    kw = {f.name: getattr(base_params, f.name) for f in base_params.__dataclass_fields__.values()}
    kw.update(overrides)
    params = ModelParams(**kw)

    result = simulate(lattice, params, variant, state=initial_state(lattice, params))

    # Pre-damage reference: mean boundary_fraction before damage_step
    damage_idx = params.damage_step
    if damage_idx > 5:
        pre_ref = float(np.mean(result.boundary_fraction[max(0, damage_idx - 20):damage_idx]))
    else:
        pre_ref = float(result.boundary_fraction[0])

    final_bf = float(result.boundary_fraction[-1])
    # Recovery = final boundary_fraction ≥ 90 % of pre-damage reference
    recovered = final_bf >= 0.9 * pre_ref if pre_ref > 1e-6 else False
    return final_bf, recovered


def sweep_2d(lattice, base_params, variant, name_x, vals_x, name_y, vals_y):
    """Sweep two parameters and return (grid_bf, grid_rec) arrays."""
    nx, ny = len(vals_x), len(vals_y)
    grid_bf  = np.zeros((ny, nx))
    grid_rec = np.zeros((ny, nx), dtype=bool)

    total = nx * ny
    done = 0
    for iy, vy in enumerate(vals_y):
        for ix, vx in enumerate(vals_x):
            bf, rec = _recovery_metric(lattice, base_params, {name_x: vx, name_y: vy}, variant)
            grid_bf[iy, ix] = bf
            grid_rec[iy, ix] = rec
            done += 1
            if done % max(1, total // 10) == 0:
                print(f"    {done}/{total}", flush=True)

    return grid_bf, grid_rec


def save_csv(path, name_x, vals_x, name_y, vals_y, grid_bf, grid_rec):
    """Write a CSV with one row per grid cell."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([name_x, name_y, "boundary_fraction", "recovered"])
        for iy, vy in enumerate(vals_y):
            for ix, vx in enumerate(vals_x):
                w.writerow([f"{vx:.4f}", f"{vy:.4f}", f"{grid_bf[iy, ix]:.6f}",
                            "1" if grid_rec[iy, ix] else "0"])


def save_heatmap(path, name_x, vals_x, name_y, vals_y, grid_bf, grid_rec, title):
    """Save a heatmap PNG (boundary_fraction) with recovery contour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # boundary fraction heatmap
    ax = axes[0]
    im = ax.imshow(grid_bf, origin="lower", aspect="auto",
                   extent=[vals_x[0], vals_x[-1], vals_y[0], vals_y[-1]],
                   cmap="viridis")
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(f"{title}\nboundary_fraction (final)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # recovery map
    ax = axes[1]
    ax.imshow(grid_rec.astype(float), origin="lower", aspect="auto",
              extent=[vals_x[0], vals_x[-1], vals_y[0], vals_y[-1]],
              cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(f"{title}\nrecovered (green=yes)")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lattice = EdgeLattice.square(8, 8)
    base = ModelParams(seed=7)
    variant = VariantConfig(name="full_law")

    summary = {}

    # --- Sweep 1: alpha0 vs lambda_s ---
    print("  Sweep 1: alpha0 vs lambda_s")
    a0_vals = np.linspace(0.1, 2.0, 8)
    ls_vals = np.linspace(0.1, 2.0, 8)
    bf1, rec1 = sweep_2d(lattice, base, variant, "alpha0", a0_vals, "lambda_s", ls_vals)
    save_csv(OUTDIR / "onset_alpha0_lambda_s.csv", "alpha0", a0_vals, "lambda_s", ls_vals, bf1, rec1)
    save_heatmap(OUTDIR / "onset_alpha0_lambda_s.png", "alpha0", a0_vals, "lambda_s", ls_vals,
                 bf1, rec1, "alpha0 vs lambda_s")
    summary["alpha0_vs_lambda_s"] = {
        "recovery_fraction": float(np.mean(rec1)),
        "bf_mean": float(np.mean(bf1)),
        "bf_range": [float(np.min(bf1)), float(np.max(bf1))],
    }

    # --- Sweep 2: chi vs damage_scale ---
    print("  Sweep 2: chi vs damage_scale")
    chi_vals = np.linspace(0.05, 1.5, 8)
    ds_vals  = np.linspace(0.02, 0.5, 8)
    bf2, rec2 = sweep_2d(lattice, base, variant, "chi", chi_vals, "damage_scale", ds_vals)
    save_csv(OUTDIR / "onset_chi_damage_scale.csv", "chi", chi_vals, "damage_scale", ds_vals, bf2, rec2)
    save_heatmap(OUTDIR / "onset_chi_damage_scale.png", "chi", chi_vals, "damage_scale", ds_vals,
                 bf2, rec2, "chi vs damage_scale")
    summary["chi_vs_damage_scale"] = {
        "recovery_fraction": float(np.mean(rec2)),
        "bf_mean": float(np.mean(bf2)),
        "bf_range": [float(np.min(bf2)), float(np.max(bf2))],
    }

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  Onset map artifacts written to: {OUTDIR}")
    for k, v in summary.items():
        print(f"    {k}: recovery_fraction={v['recovery_fraction']:.2f}, "
              f"bf_range=[{v['bf_range'][0]:.3f}, {v['bf_range'][1]:.3f}]")


if __name__ == "__main__":
    main()
