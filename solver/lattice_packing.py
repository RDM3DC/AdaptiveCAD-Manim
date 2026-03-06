"""Lattice packing optimiser for sphere/polyhedra arrangements.

Finds dense packings of spheres (or convex polyhedra) in a bounding
box using ARP-coupled gradient descent with Voronoi energy.

Supports:
- Random/E8-seeded initial configurations
- Packing density tracking (volume fraction)
- Voronoi energy (deviation from uniform cells)
- ARP α/μ modulated repulsion + attraction
- Real-time step-by-step iteration
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


_TAU = 2.0 * np.pi


# ---- ARP dynamics ----------------------------------------------------------

def _arp_decay(alpha_0: float, t: float, mu: float, v: float) -> float:
    return alpha_0 * np.exp(-mu * t) * np.cos(v * t)


# ---- Packing density -------------------------------------------------------

def sphere_packing_density(
    centres: np.ndarray,
    radius: float,
    box_size: float,
) -> float:
    """Compute packing fraction of N spheres in a cubic box.

    η = N · (4/3)πr³ / L³
    """
    n = len(centres)
    v_sphere = (4.0 / 3.0) * np.pi * radius ** 3
    v_box = box_size ** 3
    return n * v_sphere / v_box


# ---- Voronoi energy --------------------------------------------------------

def voronoi_energy(centres: np.ndarray, box_size: float) -> float:
    """Approximate Voronoi energy as variance of nearest-neighbour distances.

    Low variance → uniform cell sizes → better packing order.
    """
    n = len(centres)
    if n < 2:
        return 0.0

    nn_dists = np.full(n, np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            # Minimum-image distance (periodic box)
            diff = centres[i] - centres[j]
            diff = diff - box_size * np.round(diff / box_size)
            d = np.linalg.norm(diff)
            if d < nn_dists[i]:
                nn_dists[i] = d
            if d < nn_dists[j]:
                nn_dists[j] = d

    return float(np.var(nn_dists))


# ---- Coordination number ---------------------------------------------------

def coordination_numbers(
    centres: np.ndarray,
    cutoff: float,
    box_size: float,
) -> np.ndarray:
    """Count neighbours within cutoff for each centre (periodic box)."""
    n = len(centres)
    coord = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            diff = centres[i] - centres[j]
            diff = diff - box_size * np.round(diff / box_size)
            if np.linalg.norm(diff) < cutoff:
                coord[i] += 1
                coord[j] += 1
    return coord


# ---- Lattice packer -------------------------------------------------------

class LatticePacker:
    """ARP-coupled lattice packing optimiser.

    Gradient-descent on an energy that balances:
    - Hard-sphere repulsion (overlap penalty)
    - Attraction toward target spacing (lattice regularity)
    - Voronoi cell uniformity

    The coupling α decays via ARP: α(t) = α₀·exp(−μt)·cos(v·t),
    annealing from strong ordering to relaxed settling.

    Parameters
    ----------
    n_spheres : int
        Number of spheres to pack.
    radius : float
        Sphere radius.
    box_size : float
        Side length of the cubic periodic box.
    init_mode : str
        'random', 'fcc', or 'e8' (E8-seeded positions).
    alpha_0, mu, v : ARP decay parameters.
    lr : learning rate.
    """

    def __init__(
        self,
        n_spheres: int = 64,
        radius: float = 0.5,
        box_size: float = 8.0,
        init_mode: str = "random",
        alpha_0: float = 1.0,
        mu: float = 0.8,
        v: float = 1.2,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.n = n_spheres
        self.radius = radius
        self.box_size = box_size
        self.alpha_0 = alpha_0
        self.mu = mu
        self.v = v
        self.lr = lr

        rng = np.random.RandomState(seed)

        if init_mode == "fcc":
            self.centres = self._init_fcc()
        elif init_mode == "e8":
            self.centres = self._init_e8()
        else:
            self.centres = rng.uniform(0, box_size, size=(n_spheres, 3))

        self.history: List[Dict[str, float]] = []

    def _init_fcc(self) -> np.ndarray:
        """Face-centred-cubic initial lattice."""
        a = self.box_size / max(1, int(np.cbrt(self.n / 4)))
        pts = []
        offsets = np.array([
            [0, 0, 0],
            [a / 2, a / 2, 0],
            [a / 2, 0, a / 2],
            [0, a / 2, a / 2],
        ], dtype=np.float64)
        nx = int(np.ceil(self.box_size / a))
        for ix in range(nx):
            for iy in range(nx):
                for iz in range(nx):
                    base = np.array([ix, iy, iz], dtype=np.float64) * a
                    for off in offsets:
                        pt = (base + off) % self.box_size
                        pts.append(pt)
                        if len(pts) >= self.n:
                            return np.array(pts[:self.n])
        arr = np.array(pts[:self.n])
        if len(arr) < self.n:
            extra = np.random.uniform(
                0, self.box_size, size=(self.n - len(arr), 3)
            )
            arr = np.vstack([arr, extra])
        return arr

    def _init_e8(self) -> np.ndarray:
        """Seed from E8 root projections, scaled to fit box."""
        from .e8_seeds import E8Seeds
        seeds = E8Seeds(target_n=self.n)
        pts = seeds.points.copy()
        # Shift & scale to [0, box_size]
        pts -= pts.min(axis=0)
        mx = pts.max()
        if mx > 1e-12:
            pts *= (self.box_size * 0.8) / mx
        pts += self.box_size * 0.1  # centre in box
        if len(pts) < self.n:
            extra = np.random.uniform(
                0, self.box_size, size=(self.n - len(pts), 3)
            )
            pts = np.vstack([pts, extra])
        return pts[:self.n]

    def _alpha(self, step: int) -> float:
        t = step * 0.05
        return _arp_decay(self.alpha_0, t, self.mu, self.v)

    def step(self, iteration: int) -> Dict[str, float]:
        """One optimisation step: compute forces, update positions."""
        alpha = self._alpha(iteration)
        sigma = 2.0 * self.radius  # target min distance
        grad = np.zeros_like(self.centres)

        # Pairwise forces (periodic)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.centres[i] - self.centres[j]
                diff = diff - self.box_size * np.round(diff / self.box_size)
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    # Nudge apart
                    nudge = np.random.randn(3) * 0.01
                    grad[i] += nudge
                    grad[j] -= nudge
                    continue

                unit = diff / d

                # Overlap repulsion (hard core)
                if d < sigma:
                    overlap = sigma - d
                    force = overlap * 2.0 * unit
                    grad[i] -= force  # push apart
                    grad[j] += force

                # ARP-coupled attraction toward target spacing
                # When α is large → strong ordering pull
                # When α is small → relaxed
                target = sigma * 1.05
                if d < 4.0 * sigma:
                    pull = abs(alpha) * 0.5 * (d - target) * unit
                    grad[i] += pull
                    grad[j] -= pull

        # Apply gradient
        self.centres -= self.lr * grad
        # Periodic wrap
        self.centres %= self.box_size

        density = sphere_packing_density(self.centres, self.radius, self.box_size)
        v_energy = voronoi_energy(self.centres, self.box_size)
        coord = coordination_numbers(self.centres, sigma * 1.2, self.box_size)

        record = {
            "iteration": iteration,
            "alpha": alpha,
            "density": density,
            "voronoi_energy": v_energy,
            "mean_coord": float(np.mean(coord)),
            "min_dist": self._min_distance(),
        }
        self.history.append(record)
        return record

    def _min_distance(self) -> float:
        """Minimum pairwise distance (periodic)."""
        min_d = np.inf
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.centres[i] - self.centres[j]
                diff = diff - self.box_size * np.round(diff / self.box_size)
                d = np.linalg.norm(diff)
                if d < min_d:
                    min_d = d
        return float(min_d)

    def run(
        self, n_steps: int = 200, verbose: bool = False
    ) -> List[Dict[str, float]]:
        """Run n_steps of optimisation."""
        for i in range(n_steps):
            rec = self.step(i)
            if verbose and i % 20 == 0:
                print(
                    f"  step {i:4d}  α={rec['alpha']:+.4f}  "
                    f"η={rec['density']:.4f}  "
                    f"V_e={rec['voronoi_energy']:.5f}  "
                    f"coord={rec['mean_coord']:.1f}  "
                    f"d_min={rec['min_dist']:.4f}"
                )
        return self.history

    def packing_report(self) -> Dict:
        """Summary of final packing state."""
        sigma = 2.0 * self.radius
        coord = coordination_numbers(self.centres, sigma * 1.2, self.box_size)
        return {
            "n_spheres": self.n,
            "radius": self.radius,
            "box_size": self.box_size,
            "density": sphere_packing_density(
                self.centres, self.radius, self.box_size
            ),
            "voronoi_energy": voronoi_energy(self.centres, self.box_size),
            "mean_coordination": float(np.mean(coord)),
            "min_distance": self._min_distance(),
            "overlap_count": int(np.sum(coord[coord > 12])),
        }
