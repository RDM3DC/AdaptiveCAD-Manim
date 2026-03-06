"""E8 root system seed generator for the ARP topological solver.

Generates the 240 roots of E8 in 8D, projects to 3D via golden-ratio
Coxeter projection, builds k-NN defect webs, and feeds seed graphs
into the knot/packing/fractal solvers.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


class E8Seeds:
    """E8 root system as seed data for topological computations.

    Parameters
    ----------
    target_n : int
        Number of nodes after farthest-point subsampling (default 120).
    k_neighbours : int
        k for nearest-neighbour edge graph (default 6).
    fit_radius : float
        Scale projected points to fit within this radius (default 2.8).
    """

    def __init__(
        self,
        target_n: int = 120,
        k_neighbours: int = 6,
        fit_radius: float = 2.8,
    ):
        self.target_n = target_n
        self.k_neighbours = k_neighbours
        self.fit_radius = fit_radius

        self._roots_8d = self._generate_e8_roots()
        self._pts_3d_full = self._project_8d_to_3d(self._roots_8d)
        self._pts_3d, self._selected_idx = self._subsample(
            self._pts_3d_full, target_n
        )
        self._edges = self._nearest_edges(self._pts_3d, k_neighbours)

    # ---- public API --------------------------------------------------------

    @property
    def roots_8d(self) -> np.ndarray:
        """Full 240×8 root vectors."""
        return self._roots_8d

    @property
    def points(self) -> np.ndarray:
        """Subsampled 3D projected positions, shape (target_n, 3)."""
        return self._pts_3d

    @property
    def points_full(self) -> np.ndarray:
        """All 240 projected positions in 3D."""
        return self._pts_3d_full

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """k-NN edge list for the subsampled points."""
        return self._edges

    @property
    def adjacency(self) -> np.ndarray:
        """Symmetric adjacency matrix for subsampled graph."""
        n = len(self._pts_3d)
        A = np.zeros((n, n), dtype=np.int8)
        for i, j in self._edges:
            A[i, j] = 1
            A[j, i] = 1
        return A

    @property
    def degree_sequence(self) -> np.ndarray:
        """Degree of each node in the defect web."""
        return self.adjacency.sum(axis=1)

    def edge_lengths(self) -> np.ndarray:
        """Euclidean length of each edge."""
        return np.array([
            np.linalg.norm(self._pts_3d[i] - self._pts_3d[j])
            for i, j in self._edges
        ])

    def radial_shells(self, n_shells: int = 4) -> List[np.ndarray]:
        """Partition nodes into radial shells by distance from origin."""
        r = np.linalg.norm(self._pts_3d, axis=1)
        edges_r = np.linspace(0, r.max() + 1e-8, n_shells + 1)
        shells = []
        for lo, hi in zip(edges_r[:-1], edges_r[1:]):
            mask = (r >= lo) & (r < hi)
            shells.append(np.where(mask)[0])
        return shells

    def cycles(self, max_length: int = 6) -> List[List[int]]:
        """Find short cycles in the defect web (up to max_length).

        Uses DFS back-edge detection.  Returns unique cycles sorted
        by length.
        """
        adj = {}
        for i, j in self._edges:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)

        found = []
        n = len(self._pts_3d)

        for start in range(n):
            # BFS-like bounded DFS
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_length:
                    continue
                for nb in adj.get(node, []):
                    if nb == start and len(path) >= 3:
                        cycle = tuple(sorted(path))
                        if cycle not in [tuple(sorted(c)) for c in found]:
                            found.append(list(path))
                    elif nb not in path:
                        stack.append((nb, path + [nb]))

        found.sort(key=len)
        return found

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _generate_e8_roots() -> np.ndarray:
        roots = []
        # D8 roots: ±e_i ± e_j
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (1, -1):
                    for sj in (1, -1):
                        r = np.zeros(8)
                        r[i] = si
                        r[j] = sj
                        roots.append(r)
        # Half-spin: (±½)^8 with even # minus signs
        for bits in range(256):
            signs = np.array(
                [(bits >> k) & 1 for k in range(8)], dtype=np.float64
            )
            signs = 1.0 - 2.0 * signs
            if np.sum(signs < 0) % 2 == 0:
                roots.append(signs * 0.5)
        return np.array(roots)

    def _project_8d_to_3d(self, roots_8d: np.ndarray) -> np.ndarray:
        phi = (1 + np.sqrt(5)) / 2
        proj = np.array([
            [1, phi, 0, -1 / phi, 1, 0, -1, phi],
            [phi, 0, 1, 1, -1 / phi, -1, phi, 0],
            [0, 1, phi, 0, 1, phi, 0, -1 / phi],
        ]) / np.sqrt(8)
        pts = (proj @ roots_8d.T).T
        mx = np.max(np.linalg.norm(pts, axis=1))
        pts *= self.fit_radius / (mx + 1e-12)
        return pts

    @staticmethod
    def _subsample(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(pts) <= n:
            return pts.copy(), np.arange(len(pts))
        sel = [0]
        md = np.linalg.norm(pts - pts[0], axis=1)
        for _ in range(n - 1):
            f = int(np.argmax(md))
            sel.append(f)
            nd = np.linalg.norm(pts - pts[f], axis=1)
            md = np.minimum(md, nd)
            md[sel] = -1
        idx = np.array(sel)
        return pts[idx].copy(), idx

    @staticmethod
    def _nearest_edges(
        pts: np.ndarray, k: int = 6
    ) -> List[Tuple[int, int]]:
        n = len(pts)
        d = np.zeros((n, n))
        for i in range(n):
            d[i] = np.linalg.norm(pts - pts[i], axis=1)
            d[i, i] = np.inf
        kth = []
        for i in range(n):
            sd = np.sort(d[i])
            kth.append(sd[min(k - 1, len(sd) - 1)])
        cutoff = float(np.median(kth)) * 1.1
        edges: set = set()
        for i in range(n):
            for j in np.where(d[i] < cutoff)[0][:k]:
                edges.add((min(i, j), max(i, j)))
        return list(edges)

    def __repr__(self) -> str:
        return (
            f"E8Seeds(n={len(self._pts_3d)}, edges={len(self._edges)}, "
            f"k={self.k_neighbours})"
        )
