"""ARP Topological Solver — computational engine for AdaptiveCAD-Manim.

Pure numerics: knot invariants, fractal energy minimisation,
lattice packing optimisation, EGATL block-admittance dynamics.
"""

from .e8_seeds import E8Seeds
from .topo_invariants import (
    writhe,
    linking_number,
    crossing_sign,
    alexander_polynomial,
    jones_bracket,
    arp_knot_energy,
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
from .famous import run_all as run_famous
from .core import run_all as run_core
from .cosmo import run_all as run_cosmo
from .cmb import run_all as run_cmb
from .egatl import (
    EGATLParams,
    EntropyParams,
    RulerParams,
    QWZLattice,
    EGATLState,
    build_qwz_lattice,
    simulate,
    chern_number,
    proxy_chern_series,
    run_recovery_protocol,
    compare_ablations,
    summarize_recovery,
)

__all__ = [
    "E8Seeds",
    "writhe",
    "linking_number",
    "crossing_sign",
    "alexander_polynomial",
    "jones_bracket",
    "arp_knot_energy",
    "FractalEnergyMinimiser",
    "polyhedron_540",
    "branching_energy",
    "LatticePacker",
    "sphere_packing_density",
    "voronoi_energy",
    # EGATL block-admittance
    "EGATLParams",
    "EntropyParams",
    "RulerParams",
    "QWZLattice",
    "EGATLState",
    "build_qwz_lattice",
    "simulate",
    "chern_number",
    "proxy_chern_series",
    "run_recovery_protocol",
    "compare_ablations",
    "summarize_recovery",
    # Famous equations
    "run_famous",
    # Core equations
    "run_core",
    # Cosmological calculator
    "run_cosmo",
    # CMB evidence analysis
    "run_cmb",
]
