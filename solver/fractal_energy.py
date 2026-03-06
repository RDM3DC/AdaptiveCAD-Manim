"""Fractal energy minimiser for 540-generation polyhedra.

Builds a hierarchical branching tree (like a polyhedral skeleton),
assigns ARP-coupled energies to each edge/node, and runs gradient
descent to minimise the total fractal energy functional.

The 540-gen polyhedron is constructed by recursive stellations:
start from an icosahedron and branch each face 3 generations deep
with golden-ratio scaling, producing 540 terminal nodes.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict


_TAU = 2.0 * np.pi
_PHI = (1 + np.sqrt(5)) / 2  # golden ratio


# ---- ARP dynamics ----------------------------------------------------------

def _arp_decay(alpha_0: float, t: float, mu: float, v: float) -> float:
    """ARP filament decay: α(t) = α₀·exp(−μt)·cos(v·t)."""
    return alpha_0 * np.exp(-mu * t) * np.cos(v * t)


def _bloom_scale(t: float, mu: float = 0.8, v: float = 1.2) -> float:
    """Bloom dynamics scale factor."""
    return (1 - np.exp(-mu * t * 3)) * (1 + 0.1 * np.sin(v * t * 4))


# ---- Rotation helper (Rodrigues) ------------------------------------------

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


# ---- Icosahedron seed ------------------------------------------------------

def _icosahedron_vertices() -> np.ndarray:
    """12 vertices of a unit icosahedron."""
    verts = []
    for i in range(5):
        angle = _TAU * i / 5
        verts.append([np.cos(angle), np.sin(angle), 0.5])
    for i in range(5):
        angle = _TAU * i / 5 + _TAU / 10
        verts.append([np.cos(angle), np.sin(angle), -0.5])
    verts.append([0, 0, _PHI / 2])
    verts.append([0, 0, -_PHI / 2])
    verts = np.array(verts)
    # Normalise to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _icosahedron_faces() -> List[Tuple[int, int, int]]:
    """20 triangular faces of an icosahedron (vertex indices)."""
    faces = [
        (0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 4, 10), (4, 0, 10),
        (0, 1, 5), (1, 2, 6), (2, 3, 7), (3, 4, 8), (4, 0, 9),
        (5, 6, 1), (6, 7, 2), (7, 8, 3), (8, 9, 4), (9, 5, 0),
        (5, 6, 11), (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 5, 11),
    ]
    return faces


# ---- 540-gen polyhedron construction ---------------------------------------

def polyhedron_540(
    depth: int = 3,
    branch_factor: int = 3,
    scale_decay: float = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]]]:
    """Build a 540-generation polyhedral skeleton.

    Starts from a 12-vertex icosahedron, then recursively branches
    each face centroid into ``branch_factor`` child nodes at each
    depth level with golden-ratio arm decay.

    Parameters
    ----------
    depth : int
        Recursion depth (3 gives ~540 terminal nodes from 20 faces).
    branch_factor : int
        Children per face/node (default 3).
    scale_decay : float or None
        Length decay per level. Default: 1/φ.

    Returns
    -------
    nodes : (N, 3) positions
    edges : list of (parent_idx, child_idx)
    level_indices : list of lists, each level's node indices
    """
    if scale_decay is None:
        scale_decay = 1.0 / _PHI

    ico_v = _icosahedron_vertices()
    ico_f = _icosahedron_faces()

    nodes = list(ico_v)
    edges: List[Tuple[int, int]] = []
    level_indices: List[List[int]] = [list(range(12))]

    # Edges of icosahedron
    edge_set = set()
    for f in ico_f:
        for k in range(3):
            e = (min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3]))
            edge_set.add(e)
    edges.extend(list(edge_set))

    # Face centroids are the first tier of branches
    parents = []
    parent_dirs = []
    for f in ico_f:
        centroid = (ico_v[f[0]] + ico_v[f[1]] + ico_v[f[2]]) / 3.0
        centroid_dir = centroid / (np.linalg.norm(centroid) + 1e-15)
        idx = len(nodes)
        nodes.append(centroid * 1.2)  # push slightly outward
        parents.append(idx)
        parent_dirs.append(centroid_dir)
        # Connect centroid to face vertices
        for vi in f:
            edges.append((vi, idx))

    level_indices.append(list(parents))

    # Recursive branching
    arm_length = 0.5
    for level in range(depth):
        arm_length *= scale_decay
        new_parents = []
        new_dirs = []
        level_idx = []

        for pi, p_dir in zip(parents, parent_dirs):
            p_pos = np.array(nodes[pi])

            # Build a perpendicular frame
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(p_dir, ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            n1 = np.cross(p_dir, ref)
            n1 = n1 / (np.linalg.norm(n1) + 1e-15)

            for b in range(branch_factor):
                angle = _TAU * b / branch_factor
                rot = _rotation_matrix(p_dir, angle)
                branch_dir = (
                    np.cos(0.6) * p_dir + np.sin(0.6) * (rot @ n1)
                )
                branch_dir = branch_dir / (np.linalg.norm(branch_dir) + 1e-15)

                child_pos = p_pos + arm_length * branch_dir
                child_idx = len(nodes)
                nodes.append(child_pos)
                edges.append((pi, child_idx))
                new_parents.append(child_idx)
                new_dirs.append(branch_dir)
                level_idx.append(child_idx)

        parents = new_parents
        parent_dirs = new_dirs
        level_indices.append(level_idx)

    return np.array(nodes), edges, level_indices


# ---- Energy functionals ---------------------------------------------------

def branching_energy(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    alpha: float = 1.0,
    repulsion_weight: float = 0.1,
) -> Dict[str, float]:
    """Compute the fractal energy of a branching skeleton.

    E = E_stretch + α·E_bend + w·E_repulsion

    - E_stretch : Σ_edges (|e| - target_len)²
    - E_bend    : Σ_vertices (1 - cos θ_ij) for adjacent edge pairs
    - E_repulsion : Σ_pairs 1/|r_ij|² (short-range, cutoff at 5× median edge)

    Parameters
    ----------
    nodes : (N, 3)
    edges : list of (i, j)
    alpha : ARP coupling (scales bending term)
    repulsion_weight : scale for Coulomb repulsion

    Returns
    -------
    dict with keys 'stretch', 'bend', 'repulsion', 'total'
    """
    n = len(nodes)
    edge_lens = np.array([
        np.linalg.norm(nodes[i] - nodes[j]) for i, j in edges
    ])
    target_len = float(np.median(edge_lens)) if len(edge_lens) > 0 else 1.0

    # Stretch
    e_stretch = float(np.sum((edge_lens - target_len) ** 2))

    # Bend: for each node, cost of angle between adjacent edges
    adj: Dict[int, List[int]] = {}
    for i, j in edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    e_bend = 0.0
    for node_i in range(n):
        nb = adj.get(node_i, [])
        if len(nb) < 2:
            continue
        for a_idx in range(len(nb)):
            for b_idx in range(a_idx + 1, len(nb)):
                va = nodes[nb[a_idx]] - nodes[node_i]
                vb = nodes[nb[b_idx]] - nodes[node_i]
                la = np.linalg.norm(va)
                lb = np.linalg.norm(vb)
                if la < 1e-12 or lb < 1e-12:
                    continue
                cos_th = np.clip(np.dot(va, vb) / (la * lb), -1, 1)
                e_bend += 1.0 - cos_th

    # Repulsion: soft log-barrier on direct edges only
    e_rep = 0.0
    sigma = target_len * 0.5  # equilibrium separation
    for i, j in edges:
        d = np.linalg.norm(nodes[i] - nodes[j])
        if d < sigma and d > 1e-12:
            e_rep += -np.log(d / sigma)
        elif d <= 1e-12:
            e_rep += 20.0  # capped penalty for collapsed edges

    total = e_stretch + alpha * e_bend + repulsion_weight * e_rep

    return {
        "stretch": e_stretch,
        "bend": e_bend,
        "repulsion": e_rep,
        "total": total,
    }


# ---- Fractal energy minimiser ---------------------------------------------

class FractalEnergyMinimiser:
    """Gradient-descent minimiser with ARP-modulated coupling.

    Runs iterative relaxation on node positions while α decays
    via the ARP schedule: α(t) = α₀·exp(−μt)·cos(v·t).

    Parameters
    ----------
    nodes : (N, 3) initial positions
    edges : list of (i, j) pairs
    fixed : optional set of node indices to keep pinned
    alpha_0, mu, v : ARP parameters
    lr : learning rate
    repulsion_weight : weight of Coulomb term
    """

    def __init__(
        self,
        nodes: np.ndarray,
        edges: List[Tuple[int, int]],
        fixed: Optional[set] = None,
        alpha_0: float = 1.0,
        mu: float = 0.8,
        v: float = 1.2,
        lr: float = 0.005,
        repulsion_weight: float = 0.1,
    ):
        self.nodes = nodes.astype(np.float64).copy()
        self.edges = edges
        self.fixed = fixed or set()
        self.alpha_0 = alpha_0
        self.mu = mu
        self.v = v
        # Auto-scale learning rate to node count so large graphs don't blow up
        self.lr = lr if lr is not None else 0.005 * min(1.0, 100.0 / max(len(nodes), 1))
        self.repulsion_weight = repulsion_weight

        self._adj: Dict[int, List[int]] = {}
        for i, j in edges:
            self._adj.setdefault(i, []).append(j)
            self._adj.setdefault(j, []).append(i)

        self.history: List[Dict[str, float]] = []

    def _alpha(self, step: int) -> float:
        t = step * 0.1
        return _arp_decay(self.alpha_0, t, self.mu, self.v)

    def step(self, iteration: int) -> Dict[str, float]:
        """Execute one gradient-descent step.

        Returns energy dict for this iteration.
        """
        alpha = self._alpha(iteration)
        n = len(self.nodes)
        grad = np.zeros_like(self.nodes)

        # Compute median edge length for target
        edge_lens = np.array([
            np.linalg.norm(self.nodes[i] - self.nodes[j])
            for i, j in self.edges
        ])
        target = float(np.median(edge_lens)) if len(edge_lens) > 0 else 1.0

        # Stretch gradient: ∂/∂x_i Σ (|e| - L)²
        for i, j in self.edges:
            diff = self.nodes[i] - self.nodes[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            force = 2.0 * (dist - target) * diff / dist
            grad[i] += force
            grad[j] -= force

        # Bend gradient (simplified: push neighbours toward equal angles)
        for node_i in range(n):
            nb = self._adj.get(node_i, [])
            if len(nb) < 2:
                continue
            for a_idx in range(len(nb)):
                for b_idx in range(a_idx + 1, len(nb)):
                    va = self.nodes[nb[a_idx]] - self.nodes[node_i]
                    vb = self.nodes[nb[b_idx]] - self.nodes[node_i]
                    la = np.linalg.norm(va)
                    lb = np.linalg.norm(vb)
                    if la < 1e-12 or lb < 1e-12:
                        continue
                    va_n = va / la
                    vb_n = vb / lb
                    # Grad of (1 - cos θ) w.r.t. node_i pushes
                    # towards flattening the angle
                    mid = (va_n + vb_n)
                    mid_n = np.linalg.norm(mid)
                    if mid_n > 1e-12:
                        grad[node_i] -= alpha * mid / mid_n * 0.1

        # Repulsion gradient: log-barrier on edges with d < sigma
        sigma = target * 0.5
        for i, j in self.edges:
            diff = self.nodes[i] - self.nodes[j]
            d = np.linalg.norm(diff)
            if d < sigma and d > 1e-10:
                # Grad of -log(d/σ) = -1/d · (diff/d)
                force = self.repulsion_weight * diff / (d * d)
                grad[i] -= force  # push apart
                grad[j] += force

        # Update (skip fixed nodes, clip gradient for stability)
        max_grad = 1.0
        for i in range(n):
            if i not in self.fixed:
                g_norm = np.linalg.norm(grad[i])
                if g_norm > max_grad:
                    grad[i] *= max_grad / g_norm
                self.nodes[i] -= self.lr * grad[i]

        energy = branching_energy(
            self.nodes, self.edges, abs(alpha), self.repulsion_weight
        )
        energy["alpha"] = alpha
        energy["iteration"] = iteration
        self.history.append(energy)
        return energy

    def run(self, n_steps: int = 100, verbose: bool = False) -> List[Dict[str, float]]:
        """Run n_steps of optimisation."""
        for i in range(n_steps):
            e = self.step(i)
            if verbose and i % 10 == 0:
                print(
                    f"  step {i:4d}  α={e['alpha']:+.4f}  "
                    f"E={e['total']:.4f}  "
                    f"(str={e['stretch']:.3f} bend={e['bend']:.3f} "
                    f"rep={e['repulsion']:.3f})"
                )
        return self.history

    def converged(self, window: int = 10, tol: float = 1e-4) -> bool:
        """Check if energy has converged over the last ``window`` steps."""
        if len(self.history) < window:
            return False
        recent = [h["total"] for h in self.history[-window:]]
        return (max(recent) - min(recent)) < tol
