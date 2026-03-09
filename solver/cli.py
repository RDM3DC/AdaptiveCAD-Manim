"""ARP Topological Solver — interactive REPL / CLI.

Usage:
    python -m solver.cli                     # launch interactive REPL
    python -m solver.cli seeds               # show E8 seed graph stats
    python -m solver.cli knot trefoil        # compute trefoil invariants
    python -m solver.cli knot figure8        # compute figure-eight invariants
    python -m solver.cli knot torus 3 5      # compute (3,5) torus knot
    python -m solver.cli hopf 0.8            # Hopf link at α=0.8
    python -m solver.cli energy              # run fractal energy minimiser
    python -m solver.cli energy --depth 3    # 540-gen polyhedron
    python -m solver.cli pack               # run lattice packing
    python -m solver.cli pack --n 64 --mode fcc   # FCC-seeded 64 spheres
    python -m solver.cli egatl               # EGATL simulation (6×6 QWZ)
    python -m solver.cli chern               # Chern number sweep
    python -m solver.cli ablation            # EGATL ablation comparison
    python -m solver.cli benchmark           # Run leaderboard equation benchmarks
    python -m solver.cli famous              # Run 14 famous equations (Phase-Lift)
    python -m solver.cli core                # Run 14 canonical core equations
    python -m solver.cli cosmo               # Cosmological calculator (MoM-z14 etc.)
    python -m solver.cli cmb                 # CMB evidence analysis
"""

from __future__ import annotations

import sys
import argparse
import numpy as np

from .e8_seeds import E8Seeds
from .topo_invariants import (
    writhe,
    linking_number,
    crossing_sign,
    alexander_polynomial,
    jones_bracket,
    arp_knot_energy,
    trefoil_curve,
    hopf_link_curves,
    figure_eight_curve,
    torus_knot_curve,
)
from .fractal_energy import (
    FractalEnergyMinimiser,
    polyhedron_540,
    branching_energy,
)
from .lattice_packing import (
    LatticePacker,
    sphere_packing_density,
    voronoi_energy,
)
from .egatl import (
    EGATLParams,
    EntropyParams,
    RulerParams,
    build_qwz_lattice,
    simulate as egatl_simulate,
    chern_number as egatl_chern,
    summarize_recovery,
    run_recovery_protocol,
    compare_ablations,
    effective_transfer,
    boundary_current_fraction,
    top_edge_fraction,
)
from .benchmarks import run_all as run_benchmarks
from .famous import run_all as run_famous
from .core import run_all as run_core
from .cosmo import run_all as run_cosmo
from .cmb import run_all as run_cmb

from . import ssh as ssh_mod
from . import scalar_flux as flux_mod
from . import kitaev as kitaev_mod
from . import haldane as haldane_mod
from . import rice_mele as rice_mele_mod


def _hr():
    print("─" * 60)


# ---- Commands --------------------------------------------------------------

def cmd_seeds(args):
    """Show E8 defect web seed statistics."""
    n = getattr(args, "n", 120)
    k = getattr(args, "k", 6)
    seeds = E8Seeds(target_n=n, k_neighbours=k)
    print(f"\n  E8 Defect Web Seeds")
    _hr()
    print(f"  240 roots in 8D → projected to 3D")
    print(f"  Subsampled: {len(seeds.points)} nodes")
    print(f"  Edges (k={k} NN):  {len(seeds.edges)}")
    print(f"  Degree sequence:   min={seeds.degree_sequence.min()} "
          f"max={seeds.degree_sequence.max()} "
          f"mean={seeds.degree_sequence.mean():.1f}")
    el = seeds.edge_lengths()
    print(f"  Edge lengths:      min={el.min():.4f} "
          f"max={el.max():.4f} mean={el.mean():.4f}")
    shells = seeds.radial_shells(4)
    for i, sh in enumerate(shells):
        print(f"  Shell {i}: {len(sh)} nodes")
    cycles = seeds.cycles(max_length=4)
    print(f"  Short cycles (≤4): {len(cycles)}")
    _hr()


def cmd_knot(args):
    """Compute knot invariants."""
    knot_type = getattr(args, "knot_type", "trefoil")

    if knot_type == "trefoil":
        curve = trefoil_curve(300)
        name = "Trefoil (3₁)"
    elif knot_type == "figure8":
        curve = figure_eight_curve(300)
        name = "Figure-Eight (4₁)"
    elif knot_type == "torus":
        p = getattr(args, "p", 2)
        q = getattr(args, "q", 3)
        curve = torus_knot_curve(p, q, 300)
        name = f"Torus ({p},{q})"
    else:
        print(f"  Unknown knot type: {knot_type}")
        return

    print(f"\n  Knot Invariants: {name}")
    _hr()

    # Crossings
    xings = crossing_sign(curve)
    n_plus = sum(1 for _, _, s in xings if s > 0)
    n_minus = sum(1 for _, _, s in xings if s < 0)
    print(f"  Crossings:     {len(xings)}  (+{n_plus} / -{n_minus})")

    # Writhe
    wr = writhe(curve)
    print(f"  Writhe:        {wr:.4f}")

    # Alexander polynomial (sampled on unit circle)
    t_vals = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    alex = alexander_polynomial(curve, t_vals)
    print(f"  Δ(t) samples:  {np.abs(alex[:4]).round(4)}")

    # Jones bracket
    jb = jones_bracket(curve)
    print(f"  ⟨K⟩ bracket:   {jb:.4f}")

    # ARP knot energy
    mu = getattr(args, "mu", 0.8)
    v = getattr(args, "v", 1.2)
    times, alphas, energies = arp_knot_energy(curve, mu=mu, v=v)
    print(f"  ARP energy (μ={mu}, v={v}):")
    print(f"    E(0) = {energies[0]:.4f}")
    print(f"    E_min = {energies.min():.4f}  at t={times[np.argmin(energies)]:.2f}")
    print(f"    E_final = {energies[-1]:.4f}")
    _hr()


def cmd_hopf(args):
    """Compute Hopf link invariants at given coupling α."""
    alpha = getattr(args, "alpha", 1.0)
    mu = getattr(args, "mu", 0.8)
    v = getattr(args, "v", 1.2)

    ca, cb = hopf_link_curves(alpha=alpha, n_pts=200)

    print(f"\n  Hopf Link (α={alpha:.3f})")
    _hr()

    lk = linking_number(ca, cb)
    print(f"  Linking number:  {lk:.4f}")

    wr_a = writhe(ca)
    wr_b = writhe(cb)
    print(f"  Writhe A:        {wr_a:.4f}")
    print(f"  Writhe B:        {wr_b:.4f}")

    # ARP sweep
    print(f"\n  ARP α-sweep (μ={mu}, v={v}):")
    for t_step in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        a_t = alpha * np.exp(-mu * t_step) * np.cos(v * t_step)
        ca_t, cb_t = hopf_link_curves(alpha=max(0, a_t), n_pts=100)
        lk_t = linking_number(ca_t, cb_t)
        print(f"    t={t_step:.1f}  α(t)={a_t:+.4f}  Lk={lk_t:.4f}")
    _hr()


def cmd_energy(args):
    """Run fractal energy minimiser on 540-gen polyhedron."""
    depth = getattr(args, "depth", 3)
    n_steps = getattr(args, "steps", 100)
    mu = getattr(args, "mu", 0.8)
    v = getattr(args, "v", 1.2)

    print(f"\n  Building 540-gen polyhedron (depth={depth})...")
    nodes, edges, levels = polyhedron_540(depth=depth)
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}  Levels: {len(levels)}")

    e0 = branching_energy(nodes, edges)
    print(f"  Initial energy: {e0['total']:.4f}")
    print(f"    stretch={e0['stretch']:.3f} bend={e0['bend']:.3f} "
          f"rep={e0['repulsion']:.3f}")

    _hr()
    print(f"  Minimising (μ={mu}, v={v}, {n_steps} steps)...")
    opt = FractalEnergyMinimiser(
        nodes, edges,
        fixed={0},  # pin origin node
        mu=mu, v=v,
    )
    opt.run(n_steps, verbose=True)

    ef = opt.history[-1]
    print(f"\n  Final energy: {ef['total']:.4f}")
    print(f"    stretch={ef['stretch']:.3f} bend={ef['bend']:.3f} "
          f"rep={ef['repulsion']:.3f}")
    print(f"    α_final={ef['alpha']:+.4f}")
    if opt.converged():
        print("  ✓ Converged")
    else:
        print("  ⚠ Not yet converged — increase steps")
    _hr()


def cmd_egatl(args):
    """Run EGATL block-admittance simulation."""
    nx = getattr(args, "nx", 6)
    ny = getattr(args, "ny", 6)
    T = getattr(args, "T", 40.0)
    dt = getattr(args, "dt", 0.1)
    mass = getattr(args, "mass", -1.0)
    seed = getattr(args, "seed", 0)
    phase = getattr(args, "phase", "lifted")

    print(f"\n  EGATL Block-Admittance Simulation")
    _hr()
    print(f"  Lattice: {nx}×{ny}  mass={mass}  T={T}  dt={dt}")
    print(f"  Phase mode: {phase}")

    lattice = build_qwz_lattice(nx, ny, mass=mass)
    print(f"  Bonds: {len(lattice.bonds)}  Cells: {lattice.n_cells}")
    print(f"  Source cell: {lattice.source_cell}  Sink cell: {lattice.sink_cell}")

    out = egatl_simulate(lattice, T=T, dt=dt, seed=seed, phase_mode=phase)
    K = len(out["t"])
    state = out["final_state"]

    print(f"\n  Simulation complete — {K} steps")
    _hr()
    print(f"  Final S       = {out['S'][-1]:.4f}")
    print(f"  Final π_a     = {out['pi_a'][-1]:.4f}")
    print(f"  Flip rate     = {out['r_b'][-1]:.4f}")
    print(f"  GMRES fails   = {int(np.sum(out['solve_info'] != 0))}")
    g_final = out['g'][-1]
    print(f"  |g| range     = [{np.abs(g_final).min():.4f}, {np.abs(g_final).max():.4f}]")
    Yeff = effective_transfer(out['phi'][-1], lattice.source_cell, lattice.sink_cell)
    Bfrac = boundary_current_fraction(out['I_norm'][-1], lattice.bonds)
    Tfrac = top_edge_fraction(out['I_norm'][-1], lattice)
    print(f"  Transfer eff  = {Yeff:.4f}")
    print(f"  Boundary frac = {Bfrac:.4f}")
    print(f"  Top-edge frac = {Tfrac:.4f}")
    _hr()


def cmd_chern(args):
    """Compute discretised Chern number for QWZ model."""
    mass = getattr(args, "mass", -1.0)
    nk = getattr(args, "nk", 31)
    sweep = getattr(args, "sweep", False)

    print(f"\n  QWZ Chern Number (Fukui-Hatsugai-Suzuki)")
    _hr()

    if sweep:
        print(f"  Mass sweep  nk={nk}")
        for m_val in np.linspace(-3.0, 3.0, 13):
            c = egatl_chern(1.0, 1.0, m_val, nk)
            print(f"    m={m_val:+6.2f}  C={c:+.4f}  ({round(c)})")
    else:
        c = egatl_chern(1.0, 1.0, mass, nk)
        print(f"  mass={mass}  nk={nk}")
        print(f"  Chern number = {c:.6f}  (rounded: {round(c)})")
    _hr()


def cmd_benchmark(args):
    """Run leaderboard equation benchmarks."""
    print("\n  Running TopEquations leaderboard benchmarks...")
    _hr()
    run_benchmarks(verbose=True)


def cmd_famous(args):
    """Run all 14 famous equations (Phase-Lift adjusted)."""
    print("\n  Running 14 Famous Equations (Phase-Lift adjusted)...")
    _hr()
    run_famous(verbose=True)


def cmd_core(args):
    """Run all 14 canonical core equations."""
    print("\n  Running 14 Canonical Core Equations...")
    _hr()
    run_core(verbose=True)


def cmd_cosmo(args):
    """Run cosmological calculator for high-z galaxies."""
    print("\n  Running Cosmological Calculator (Planck 2018 ΛCDM)...")
    _hr()
    run_cosmo(verbose=True)


def cmd_cmb(args):
    """Run CMB evidence analysis."""
    print("\n  Running CMB Evidence Analysis...")
    _hr()
    run_cmb(verbose=True)


def cmd_ssh(args):
    """Run SSH scalar chain recovery protocol."""
    n_cells = getattr(args, "n_cells", 20)
    T = getattr(args, "T", 60.0)
    seed = getattr(args, "seed", 0)
    ablation = getattr(args, "ablation", False)

    print(f"\n  SSH Scalar Chain Benchmark")
    _hr()
    print(f"  n_cells={n_cells}  T={T}")

    if ablation:
        results = ssh_mod.compare_ablations(n_cells=n_cells, T=T, seed=seed)
        for name, (bench, out, summ) in results.items():
            print(f"\n  [{name}]")
            print(f"    Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
            print(f"    Dimerization pre/post: {summ['dimerization_pre']:.4f} → {summ['dimerization_post']:.4f}")
            print(f"    Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
            print(f"    GMRES fails: {summ['gmres_fails']}")
    else:
        bench, out = ssh_mod.run_recovery_protocol(
            n_cells=n_cells, T=T, seed=seed)
        summ = ssh_mod.summarize_recovery(out, bench, damage_time=20.0)
        print(f"  Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
        print(f"  Dimerization pre/post: {summ['dimerization_pre']:.4f} → {summ['dimerization_post']:.4f}")
        print(f"  Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    _hr()


def cmd_flux(args):
    """Run 2D scalar flux lattice recovery or Hofstadter sweep."""
    nx = getattr(args, "nx", 8)
    ny = getattr(args, "ny", 8)
    T = getattr(args, "T", 80.0)
    seed = getattr(args, "seed", 0)
    sweep = getattr(args, "sweep", False)

    print(f"\n  Scalar Flux Lattice Benchmark")
    _hr()

    if sweep:
        print(f"  Hofstadter butterfly sweep ({nx}×{ny})")
        result = flux_mod.hofstadter_sweep(nx=nx, ny=ny, n_alpha=21, seed=seed)
        for i in range(len(result["alphas"])):
            print(f"    α={result['alphas'][i]:.3f}  Yeff={result['Yeff_final'][i]:.4f}  S={result['S_final'][i]:.4f}")
    else:
        print(f"  Recovery protocol ({nx}×{ny})  T={T}")
        bench, out = flux_mod.run_recovery_protocol(nx=nx, ny=ny, T=T, seed=seed)
        summ = flux_mod.summarize_recovery(out, bench, damage_time=30.0)
        print(f"  Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
        print(f"  Top-edge pre/post: {summ['top_edge_pre']:.4f} → {summ['top_edge_post']:.4f}")
        print(f"  Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    _hr()


def cmd_kitaev(args):
    """Run Kitaev chain Majorana benchmark."""
    n_sites = getattr(args, "n_sites", 20)
    T = getattr(args, "T", 60.0)
    seed = getattr(args, "seed", 0)
    ablation = getattr(args, "ablation", False)

    print(f"\n  Kitaev Chain Benchmark (Majorana edge modes)")
    _hr()
    print(f"  n_sites={n_sites}  T={T}")

    if ablation:
        results = kitaev_mod.compare_ablations(n_sites=n_sites, T=T, seed=seed)
        for name, (chain, out, summ) in results.items():
            print(f"\n  [{name}]")
            print(f"    Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
            print(f"    Edge ratio pre/post: {summ['edge_ratio_pre']:.4f} → {summ['edge_ratio_post']:.4f}")
            print(f"    Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
            print(f"    GMRES fails: {summ['gmres_fails']}")
    else:
        chain, out = kitaev_mod.run_recovery_protocol(
            n_sites=n_sites, T=T, seed=seed)
        summ = kitaev_mod.summarize_recovery(out, chain, damage_time=20.0)
        print(f"  Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
        print(f"  Edge ratio pre/post: {summ['edge_ratio_pre']:.4f} → {summ['edge_ratio_post']:.4f}")
        print(f"  Topo gap: {summ.get('topo_gap', 'N/A')}")
        print(f"  Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    _hr()


def cmd_haldane(args):
    """Run Haldane honeycomb lattice benchmark."""
    nx = getattr(args, "nx", 6)
    ny = getattr(args, "ny", 6)
    T = getattr(args, "T", 60.0)
    seed = getattr(args, "seed", 0)
    ablation = getattr(args, "ablation", False)

    print(f"\n  Haldane Honeycomb Lattice Benchmark")
    _hr()
    print(f"  {nx}×{ny} unit cells  T={T}")

    if ablation:
        results = haldane_mod.compare_ablations(nx=nx, ny=ny, T=T, seed=seed)
        for name, (bench, out, summ) in results.items():
            print(f"\n  [{name}]")
            print(f"    Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
            print(f"    Boundary pre/post: {summ['boundary_pre']:.4f} → {summ['boundary_post']:.4f}")
            print(f"    NNN ratio pre/post: {summ['nnn_ratio_pre']:.4f} → {summ['nnn_ratio_post']:.4f}")
            print(f"    Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    else:
        bench, out = haldane_mod.run_recovery_protocol(
            nx=nx, ny=ny, T=T, seed=seed)
        summ = haldane_mod.summarize_recovery(out, bench, damage_time=20.0)
        print(f"  Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
        print(f"  Boundary pre/post: {summ['boundary_pre']:.4f} → {summ['boundary_post']:.4f}")
        print(f"  NNN ratio pre/post: {summ['nnn_ratio_pre']:.4f} → {summ['nnn_ratio_post']:.4f}")
        print(f"  Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    _hr()


def cmd_ricemele(args):
    """Run Rice–Mele chain benchmark."""
    n_cells = getattr(args, "n_cells", 20)
    T = getattr(args, "T", 60.0)
    seed = getattr(args, "seed", 0)
    ablation = getattr(args, "ablation", False)
    sweep = getattr(args, "sweep", False)

    print(f"\n  Rice–Mele Chain Benchmark")
    _hr()

    if sweep:
        print(f"  Zak phase sweep (n_cells={n_cells})")
        result = rice_mele_mod.zak_phase_sweep(n_cells=n_cells, seed=seed)
        for i in range(len(result["angles"])):
            ang = result["angles"][i]
            print(f"    θ={ang:.3f}  Yeff={result['Yeff_final'][i]:.4f}  "
                  f"dim={result['dimerization'][i]:.4f}  "
                  f"φ_imb={result['phase_imbalance'][i]:.4f}")
    elif ablation:
        results = rice_mele_mod.compare_ablations(n_cells=n_cells, T=T, seed=seed)
        for name, (bench, out, summ) in results.items():
            print(f"\n  [{name}]")
            print(f"    Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
            print(f"    Dimerization pre/post: {summ['dimerization_pre']:.4f} → {summ['dimerization_post']:.4f}")
            print(f"    Phase imbalance pre/post: {summ['phase_imbalance_pre']:.4f} → {summ['phase_imbalance_post']:.4f}")
            print(f"    Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    else:
        print(f"  n_cells={n_cells}  T={T}")
        bench, out = rice_mele_mod.run_recovery_protocol(
            n_cells=n_cells, T=T, seed=seed)
        summ = rice_mele_mod.summarize_recovery(out, bench, damage_time=20.0)
        print(f"  Yeff recovery: {summ['Yeff_recovery_ratio']:.4f}")
        print(f"  Dimerization pre/post: {summ['dimerization_pre']:.4f} → {summ['dimerization_post']:.4f}")
        print(f"  Phase imbalance pre/post: {summ['phase_imbalance_pre']:.4f} → {summ['phase_imbalance_post']:.4f}")
        print(f"  Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
    _hr()


def cmd_ablation(args):
    """Run EGATL ablation comparison."""
    nx = getattr(args, "nx", 6)
    ny = getattr(args, "ny", 6)
    T = getattr(args, "T", 24.0)
    dt = getattr(args, "dt", 0.1)
    mass = getattr(args, "mass", -0.25)
    seed = getattr(args, "seed", 0)
    damage_time = getattr(args, "damage_time", 10.0)

    print(f"\n  EGATL Ablation Comparison")
    _hr()
    print(f"  Lattice: {nx}×{ny}  mass={mass}  T={T}  damage@t={damage_time}")
    print(f"  Configs: principal/fixed_π, lifted/fixed_π, lifted/adaptive_π\n")

    results = compare_ablations(nx, ny, T, dt, seed, damage_time, mass)

    for name, (lat, out, summ) in results.items():
        print(f"  [{name}]")
        print(f"    Transfer recovery: {summ['transfer_recovery']:.4f}")
        print(f"    Boundary pre/post: {summ['boundary_pre']:.4f} → {summ['boundary_post']:.4f}")
        print(f"    Chern   pre/post:  {summ['chern_pre']:.4f} → {summ['chern_post']:.4f}")
        print(f"    Final S={summ['final_S']:.3f}  π_a={summ['final_pi_a']:.3f}")
        print(f"    GMRES fails: {summ['gmres_fails']}")
        print()
    _hr()


def cmd_pack(args):
    """Run lattice packing optimiser."""
    n = getattr(args, "n", 64)
    mode = getattr(args, "mode", "random")
    n_steps = getattr(args, "steps", 200)
    radius = getattr(args, "radius", 0.5)
    box = getattr(args, "box", 8.0)
    mu = getattr(args, "mu", 0.8)
    v = getattr(args, "v", 1.2)

    print(f"\n  Lattice Packing (n={n}, mode={mode})")
    _hr()
    packer = LatticePacker(
        n_spheres=n, radius=radius, box_size=box,
        init_mode=mode, mu=mu, v=v,
    )
    print(f"  Initial density: "
          f"{sphere_packing_density(packer.centres, radius, box):.4f}")

    packer.run(n_steps, verbose=True)

    report = packer.packing_report()
    print(f"\n  Final Report:")
    for k, val in report.items():
        print(f"    {k}: {val}")
    _hr()


# ---- Interactive REPL ------------------------------------------------------

def repl():
    """Interactive solver REPL."""
    print("""
╔══════════════════════════════════════════════════╗
║   ARP Topological Solver — Interactive REPL      ║
║                                                  ║
║   Commands:                                      ║
║     seeds [n] [k]     E8 defect web stats        ║
║     knot <type>       Knot invariants             ║
║       trefoil | figure8 | torus <p> <q>          ║
║     hopf [alpha]      Hopf link at α              ║
║     energy [depth]    Fractal energy minimise     ║
║     pack [n] [mode]   Lattice packing optimise    ║
║     egatl [nx] [ny]   EGATL block simulation      ║
║     chern [mass]      QWZ Chern number            ║
║     ablation          EGATL ablation comparison   ║
║     ssh [n_cells]     SSH scalar chain             ║
║     flux [nx] [ny]    2D scalar flux lattice       ║
║     kitaev [n_sites]  Kitaev chain (Majorana)      ║
║     haldane [nx] [ny] Haldane honeycomb            ║
║     ricemele [n]      Rice–Mele chain              ║
║     benchmark         Leaderboard equation tests  ║
║     famous            14 Famous Equations (PL)    ║
║     core              14 Core Equations            ║
║     cosmo             Cosmological calculator      ║
║     cmb               CMB evidence analysis        ║
║     quit / exit       Exit                        ║
╚══════════════════════════════════════════════════╝
""")

    class _NS:
        pass

    while True:
        try:
            line = input("arp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        args = _NS()
        args.mu = 0.8
        args.v = 1.2

        if cmd == "seeds":
            args.n = int(parts[1]) if len(parts) > 1 else 120
            args.k = int(parts[2]) if len(parts) > 2 else 6
            cmd_seeds(args)

        elif cmd == "knot":
            args.knot_type = parts[1] if len(parts) > 1 else "trefoil"
            if args.knot_type == "torus":
                args.p = int(parts[2]) if len(parts) > 2 else 2
                args.q = int(parts[3]) if len(parts) > 3 else 3
            cmd_knot(args)

        elif cmd == "hopf":
            args.alpha = float(parts[1]) if len(parts) > 1 else 1.0
            cmd_hopf(args)

        elif cmd == "energy":
            args.depth = int(parts[1]) if len(parts) > 1 else 3
            args.steps = int(parts[2]) if len(parts) > 2 else 100
            cmd_energy(args)

        elif cmd == "pack":
            args.n = int(parts[1]) if len(parts) > 1 else 64
            args.mode = parts[2] if len(parts) > 2 else "random"
            args.steps = int(parts[3]) if len(parts) > 3 else 200
            args.radius = 0.5
            args.box = 8.0
            cmd_pack(args)

        elif cmd == "egatl":
            args.nx = int(parts[1]) if len(parts) > 1 else 6
            args.ny = int(parts[2]) if len(parts) > 2 else 6
            args.T = float(parts[3]) if len(parts) > 3 else 40.0
            args.dt = 0.1
            args.mass = -1.0
            args.seed = 0
            args.phase = "lifted"
            cmd_egatl(args)

        elif cmd == "chern":
            args.mass = float(parts[1]) if len(parts) > 1 else -1.0
            args.nk = int(parts[2]) if len(parts) > 2 else 31
            args.sweep = len(parts) > 1 and parts[-1] == "sweep"
            cmd_chern(args)

        elif cmd == "ablation":
            args.nx = int(parts[1]) if len(parts) > 1 else 6
            args.ny = int(parts[2]) if len(parts) > 2 else 6
            args.T = 24.0
            args.dt = 0.1
            args.mass = -0.25
            args.seed = 0
            args.damage_time = 10.0
            cmd_ablation(args)

        elif cmd == "benchmark":
            cmd_benchmark(args)

        elif cmd == "famous":
            cmd_famous(args)

        elif cmd == "core":
            cmd_core(args)

        elif cmd == "cosmo":
            cmd_cosmo(args)

        elif cmd == "cmb":
            cmd_cmb(args)

        elif cmd == "ssh":
            args.n_cells = int(parts[1]) if len(parts) > 1 else 20
            args.T = float(parts[2]) if len(parts) > 2 else 60.0
            args.seed = 0
            args.ablation = len(parts) > 1 and parts[-1] == "ablation"
            cmd_ssh(args)

        elif cmd == "flux":
            args.nx = int(parts[1]) if len(parts) > 1 else 8
            args.ny = int(parts[2]) if len(parts) > 2 else 8
            args.T = 80.0
            args.seed = 0
            args.sweep = len(parts) > 1 and parts[-1] == "sweep"
            cmd_flux(args)

        elif cmd == "kitaev":
            args.n_sites = int(parts[1]) if len(parts) > 1 else 20
            args.T = float(parts[2]) if len(parts) > 2 else 60.0
            args.seed = 0
            args.ablation = len(parts) > 1 and parts[-1] == "ablation"
            cmd_kitaev(args)

        elif cmd == "haldane":
            args.nx = int(parts[1]) if len(parts) > 1 else 6
            args.ny = int(parts[2]) if len(parts) > 2 else 6
            args.T = 60.0
            args.seed = 0
            args.ablation = len(parts) > 1 and parts[-1] == "ablation"
            cmd_haldane(args)

        elif cmd == "ricemele":
            args.n_cells = int(parts[1]) if len(parts) > 1 else 20
            args.T = 60.0
            args.seed = 0
            args.ablation = len(parts) > 1 and parts[-1] == "ablation"
            args.sweep = len(parts) > 1 and parts[-1] == "sweep"
            cmd_ricemele(args)

        else:
            print(f"  Unknown command: {cmd}")
            print("  Try: seeds, knot, hopf, energy, pack, egatl, chern, ablation, ssh, flux, kitaev, haldane, ricemele, benchmark, famous, core, cosmo, cmb, quit")


# ---- CLI entry point -------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        repl()
        return

    parser = argparse.ArgumentParser(
        prog="arp-solver",
        description="ARP Topological Solver",
    )
    sub = parser.add_subparsers(dest="command")

    # seeds
    p_seeds = sub.add_parser("seeds", help="E8 seed statistics")
    p_seeds.add_argument("--n", type=int, default=120)
    p_seeds.add_argument("--k", type=int, default=6)

    # knot
    p_knot = sub.add_parser("knot", help="Knot invariants")
    p_knot.add_argument("knot_type", choices=["trefoil", "figure8", "torus"])
    p_knot.add_argument("p", nargs="?", type=int, default=2)
    p_knot.add_argument("q", nargs="?", type=int, default=3)
    p_knot.add_argument("--mu", type=float, default=0.8)
    p_knot.add_argument("--v", type=float, default=1.2)

    # hopf
    p_hopf = sub.add_parser("hopf", help="Hopf link invariants")
    p_hopf.add_argument("alpha", nargs="?", type=float, default=1.0)
    p_hopf.add_argument("--mu", type=float, default=0.8)
    p_hopf.add_argument("--v", type=float, default=1.2)

    # energy
    p_energy = sub.add_parser("energy", help="Fractal energy minimiser")
    p_energy.add_argument("--depth", type=int, default=3)
    p_energy.add_argument("--steps", type=int, default=100)
    p_energy.add_argument("--mu", type=float, default=0.8)
    p_energy.add_argument("--v", type=float, default=1.2)

    # pack
    p_pack = sub.add_parser("pack", help="Lattice packing optimiser")
    p_pack.add_argument("--n", type=int, default=64)
    p_pack.add_argument("--mode", choices=["random", "fcc", "e8"], default="random")
    p_pack.add_argument("--steps", type=int, default=200)
    p_pack.add_argument("--radius", type=float, default=0.5)
    p_pack.add_argument("--box", type=float, default=8.0)
    p_pack.add_argument("--mu", type=float, default=0.8)
    p_pack.add_argument("--v", type=float, default=1.2)

    # egatl
    p_egatl = sub.add_parser("egatl", help="EGATL block-admittance simulation")
    p_egatl.add_argument("--nx", type=int, default=6)
    p_egatl.add_argument("--ny", type=int, default=6)
    p_egatl.add_argument("--T", type=float, default=40.0)
    p_egatl.add_argument("--dt", type=float, default=0.1)
    p_egatl.add_argument("--mass", type=float, default=-1.0)
    p_egatl.add_argument("--seed", type=int, default=0)
    p_egatl.add_argument("--phase", choices=["lifted", "principal"], default="lifted")

    # chern
    p_chern = sub.add_parser("chern", help="QWZ Chern number")
    p_chern.add_argument("--mass", type=float, default=-1.0)
    p_chern.add_argument("--nk", type=int, default=31)
    p_chern.add_argument("--sweep", action="store_true")

    # ablation
    p_abl = sub.add_parser("ablation", help="EGATL ablation comparison")
    p_abl.add_argument("--nx", type=int, default=6)
    p_abl.add_argument("--ny", type=int, default=6)
    p_abl.add_argument("--T", type=float, default=24.0)
    p_abl.add_argument("--dt", type=float, default=0.1)
    p_abl.add_argument("--mass", type=float, default=-0.25)
    p_abl.add_argument("--seed", type=int, default=0)
    p_abl.add_argument("--damage-time", type=float, default=10.0)

    # benchmark
    sub.add_parser("benchmark", help="Run leaderboard equation benchmarks")

    # famous
    sub.add_parser("famous", help="Run 14 famous equations (Phase-Lift)")

    # core
    sub.add_parser("core", help="Run 14 canonical core equations")

    # cosmo
    sub.add_parser("cosmo", help="Cosmological calculator (high-z galaxies)")

    # cmb
    sub.add_parser("cmb", help="CMB evidence analysis")

    # ssh
    p_ssh = sub.add_parser("ssh", help="SSH scalar chain benchmark")
    p_ssh.add_argument("--n-cells", type=int, default=20)
    p_ssh.add_argument("--T", type=float, default=60.0)
    p_ssh.add_argument("--seed", type=int, default=0)
    p_ssh.add_argument("--ablation", action="store_true")

    # flux
    p_flux = sub.add_parser("flux", help="2D scalar flux lattice / Hofstadter")
    p_flux.add_argument("--nx", type=int, default=8)
    p_flux.add_argument("--ny", type=int, default=8)
    p_flux.add_argument("--T", type=float, default=80.0)
    p_flux.add_argument("--seed", type=int, default=0)
    p_flux.add_argument("--sweep", action="store_true")

    # kitaev
    p_kit = sub.add_parser("kitaev", help="Kitaev chain Majorana benchmark")
    p_kit.add_argument("--n-sites", type=int, default=20)
    p_kit.add_argument("--T", type=float, default=60.0)
    p_kit.add_argument("--seed", type=int, default=0)
    p_kit.add_argument("--ablation", action="store_true")

    # haldane
    p_hal = sub.add_parser("haldane", help="Haldane honeycomb lattice benchmark")
    p_hal.add_argument("--nx", type=int, default=6)
    p_hal.add_argument("--ny", type=int, default=6)
    p_hal.add_argument("--T", type=float, default=60.0)
    p_hal.add_argument("--seed", type=int, default=0)
    p_hal.add_argument("--ablation", action="store_true")

    # ricemele
    p_rm = sub.add_parser("ricemele", help="Rice-Mele chain benchmark")
    p_rm.add_argument("--n-cells", type=int, default=20)
    p_rm.add_argument("--T", type=float, default=60.0)
    p_rm.add_argument("--seed", type=int, default=0)
    p_rm.add_argument("--ablation", action="store_true")
    p_rm.add_argument("--sweep", action="store_true")

    args = parser.parse_args()
    cmd = args.command

    if cmd == "seeds":
        cmd_seeds(args)
    elif cmd == "knot":
        cmd_knot(args)
    elif cmd == "hopf":
        cmd_hopf(args)
    elif cmd == "energy":
        cmd_energy(args)
    elif cmd == "pack":
        cmd_pack(args)
    elif cmd == "egatl":
        cmd_egatl(args)
    elif cmd == "chern":
        cmd_chern(args)
    elif cmd == "ablation":
        cmd_ablation(args)
    elif cmd == "benchmark":
        cmd_benchmark(args)
    elif cmd == "famous":
        cmd_famous(args)
    elif cmd == "core":
        cmd_core(args)
    elif cmd == "cosmo":
        cmd_cosmo(args)
    elif cmd == "cmb":
        cmd_cmb(args)
    elif cmd == "ssh":
        cmd_ssh(args)
    elif cmd == "flux":
        cmd_flux(args)
    elif cmd == "kitaev":
        cmd_kitaev(args)
    elif cmd == "haldane":
        cmd_haldane(args)
    elif cmd == "ricemele":
        cmd_ricemele(args)
    else:
        repl()


if __name__ == "__main__":
    main()
