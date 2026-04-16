"""
Convex Optimization Placer

Two-stage algorithm:
  1. Quadratic Placement (QP): solve for macro positions that minimize
     weighted sum of squared pairwise distances for connected macros.
     Formulated as a sparse linear system via the graph Laplacian:
         (L_mm + lambda*I) * x_m = -L_mf * x_f + lambda * x_m0
     Separates identically for x and y. Solved with numpy's dense solver
     (n_macros <= 600, so 600x600 systems are trivial).

  2. Legalization: minimum-displacement overlap removal. Processes macros
     largest-first, searching outward from QP positions for the nearest
     overlap-free location.

  Conservative fallback: a fast approximate proxy cost (HPWL + density)
  compares the QP+legalized result against the initial placement. If QP
  degraded the result, the initial placement is kept instead. This ensures
  we never do worse than the starting point.

Usage:
    uv run evaluate submissions/convex_opt/placer.py
    uv run evaluate submissions/convex_opt/placer.py --all
    uv run evaluate submissions/convex_opt/placer.py -b ibm03
"""

import math
import random

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Nets larger than this use a star model instead of full clique expansion
# to avoid O(k^2) blowup on high-fanout (e.g. clock) nets.
MAX_CLIQUE_SIZE = 20


class ConvexOptPlacer:
    """
    Convex (quadratic) placement + legalization + SA refinement.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    lambda_anchor : float
        Spring constant for the anchor term (fraction of mean Laplacian diagonal).
        Keeps the QP well-conditioned and pulls macros toward their initial
        positions when few fixed-position nodes exist.
    sa_iters : int
        Number of SA refinement steps after legalization.
    """

    def __init__(self, seed: int = 42, lambda_anchor: float = 30.0, sa_iters: int = 0):
        self.seed = seed
        self.lambda_anchor = lambda_anchor
        self.sa_iters = sa_iters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros

        pos = benchmark.macro_positions.numpy().copy().astype(np.float64)  # [n_total, 2]
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)           # [n_total, 2]
        fixed_mask = benchmark.macro_fixed.numpy()                          # [n_total] bool
        port_pos = benchmark.port_positions.numpy().astype(np.float64)     # [n_ports, 2]

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Movable: non-fixed hard macros only.
        # Soft macros stay at their initial positions (already co-optimized).
        movable = np.zeros(n_total, dtype=bool)
        movable[:n_hard] = ~fixed_mask[:n_hard]

        # --- Stage 1: QP solve with multiple lambda values ---
        # Try a range of anchor strengths. Higher lambda = smaller QP displacement,
        # closer to initial positions. We legalize each candidate and compare.
        candidates = []
        for lam in [30.0, 100.0, 400.0]:
            self.lambda_anchor = lam
            pos_cand = self._qp_solve(pos.copy(), movable, sizes, benchmark, n_total, port_pos, cw, ch)
            pos_cand[:n_hard] = self._legalize(
                pos_cand[:n_hard].copy(), ~fixed_mask[:n_hard], sizes[:n_hard], cw, ch,
            )
            candidates.append((self._fast_proxy(pos_cand, benchmark, n_total, port_pos), pos_cand))
        self.lambda_anchor = 30.0  # restore default

        # --- Stage 2: Also legalize the initial positions as a fallback ---
        pos_init_legal = pos.copy()
        pos_init_legal[:n_hard] = self._legalize(
            pos[:n_hard].copy(), ~fixed_mask[:n_hard], sizes[:n_hard], cw, ch,
        )
        candidates.append((self._fast_proxy(pos_init_legal, benchmark, n_total, port_pos), pos_init_legal))

        # Pick the candidate with lowest approximate proxy cost
        pos = min(candidates, key=lambda c: c[0])[1]

        # --- Stage 3: SA refinement (optional) ---
        if self.sa_iters > 0:
            pos[:n_hard] = self._sa_refine(
                pos[:n_hard].copy(),
                ~fixed_mask[:n_hard],
                sizes[:n_hard],
                benchmark,
                n_hard, n_total,
                cw, ch,
            )

        return torch.tensor(pos, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Fast proxy cost (for conservative fallback comparison)
    # ------------------------------------------------------------------

    def _fast_proxy(self, pos, benchmark, n_total, port_pos):
        """
        Fast approximate proxy cost: HPWL + 0.5 * density.

        Omits congestion (expensive to compute without plc), but the
        WL + density approximation is enough to detect regressions where
        QP clustering increases density without meaningful WL gain.
        """
        wl = self._fast_hpwl(pos, benchmark, n_total, port_pos)
        density = self._fast_density(pos, benchmark)
        return wl + 0.5 * density

    def _fast_hpwl(self, pos, benchmark, n_total, port_pos):
        """Normalized HPWL over all nets (approximates plc.get_cost())."""
        canvas_area = benchmark.canvas_width * benchmark.canvas_height
        total = 0.0
        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = nodes_tensor.numpy()
            w = float(benchmark.net_weights[net_i])
            xs, ys = [], []
            for n in nodes:
                n = int(n)
                if n < n_total:
                    xs.append(pos[n, 0])
                    ys.append(pos[n, 1])
                else:
                    p = n - n_total
                    if p < len(port_pos):
                        xs.append(port_pos[p, 0])
                        ys.append(port_pos[p, 1])
            if len(xs) >= 2:
                total += w * (max(xs) - min(xs) + max(ys) - min(ys))
        return total / canvas_area

    def _fast_density(self, pos, benchmark):
        """
        Grid-based density cost: average density of top 10% cells.
        Approximates plc.get_density_cost() without the Python evaluator.
        """
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        rows = benchmark.grid_rows
        cols = benchmark.grid_cols
        cell_w = cw / cols
        cell_h = ch / rows
        cell_area = cell_w * cell_h
        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)

        density_grid = np.zeros(rows * cols, dtype=np.float64)

        for i in range(n_hard):
            x, y = pos[i, 0], pos[i, 1]
            hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
            xl, xr = x - hw, x + hw
            yl, yr = y - hh, y + hh

            c_start = max(0, int(xl / cell_w))
            c_end = min(cols - 1, int(xr / cell_w))
            r_start = max(0, int(yl / cell_h))
            r_end = min(rows - 1, int(yr / cell_h))

            for r in range(r_start, r_end + 1):
                cy_lo, cy_hi = r * cell_h, (r + 1) * cell_h
                oy = max(0.0, min(yr, cy_hi) - max(yl, cy_lo))
                if oy <= 0:
                    continue
                for c in range(c_start, c_end + 1):
                    cx_lo, cx_hi = c * cell_w, (c + 1) * cell_w
                    ox = max(0.0, min(xr, cx_hi) - max(xl, cx_lo))
                    if ox > 0:
                        density_grid[r * cols + c] += ox * oy / cell_area

        density_grid.sort()
        thresh = int(0.9 * len(density_grid))
        return float(density_grid[thresh:].mean())

    # ------------------------------------------------------------------
    # Stage 1: Quadratic Placement
    # ------------------------------------------------------------------

    def _qp_solve(self, pos, movable, sizes, benchmark, n_total, port_pos, cw, ch):
        """
        Build and solve the Laplacian linear system.

        For each net with nodes [n1, n2, ...] (hard macros, soft macros, ports):
        - Expand to weighted edges via clique (or star for large nets)
        - Build Laplacian L partitioned into movable (m) and fixed (f)
        - Solve: (L_mm + λI) x_m = Σ_{j fixed} w_ij * x_j + λ * x_m0

        The spring term λI handles:
        - Near-singular systems (few fixed nodes)
        - Keeps macros near their initial positions as a prior
        """
        movable_idx = np.where(movable)[0]
        n_mov = len(movable_idx)
        if n_mov == 0:
            return pos

        mov_map = {int(idx): i for i, idx in enumerate(movable_idx)}

        # Dense Laplacian for movable macros (n_mov <= ~540, trivially small)
        L_mm = np.zeros((n_mov, n_mov), dtype=np.float64)
        # Accumulate L_mf * x_f and L_mf * y_f directly
        rhs_x = np.zeros(n_mov, dtype=np.float64)
        rhs_y = np.zeros(n_mov, dtype=np.float64)

        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = nodes_tensor.numpy().tolist()
            w = float(benchmark.net_weights[net_i])
            k = len(nodes)
            if k < 2:
                continue

            # Generate weighted edges
            if k > MAX_CLIQUE_SIZE:
                # Star model: connect anchor to all others
                anchor = nodes[0]
                edges = [(anchor, nodes[j], w / (k - 1)) for j in range(1, k)]
            else:
                # Full clique, total weight = w
                ew = 2.0 * w / (k * (k - 1))
                edges = [
                    (nodes[a], nodes[b], ew)
                    for a in range(k)
                    for b in range(a + 1, k)
                ]

            for ni, nj, pw in edges:
                ni, nj = int(ni), int(nj)
                ri_mov = ni in mov_map
                rj_mov = nj in mov_map

                if ri_mov:
                    ri = mov_map[ni]
                    L_mm[ri, ri] += pw
                    if rj_mov:
                        rj = mov_map[nj]
                        L_mm[rj, rj] += pw
                        L_mm[ri, rj] -= pw
                        L_mm[rj, ri] -= pw
                    else:
                        xj, yj = self._fixed_pos(nj, pos, port_pos, n_total)
                        rhs_x[ri] += pw * xj
                        rhs_y[ri] += pw * yj
                elif rj_mov:
                    rj = mov_map[nj]
                    L_mm[rj, rj] += pw
                    xi, yi = self._fixed_pos(ni, pos, port_pos, n_total)
                    rhs_x[rj] += pw * xi
                    rhs_y[rj] += pw * yi

        # Spring anchor: pulls toward initial positions (regularization)
        mean_diag = L_mm.diagonal().mean()
        if mean_diag <= 0:
            mean_diag = 1.0
        lam = self.lambda_anchor * mean_diag
        A = L_mm + np.eye(n_mov) * lam

        x_m0 = pos[movable_idx, 0]
        y_m0 = pos[movable_idx, 1]
        bx = rhs_x + lam * x_m0
        by = rhs_y + lam * y_m0

        try:
            x_mov = np.linalg.solve(A, bx)
            y_mov = np.linalg.solve(A, by)
        except np.linalg.LinAlgError:
            # Fallback: keep initial positions
            return pos

        # Replace NaNs/Infs with initial positions
        x_mov = np.where(np.isfinite(x_mov), x_mov, x_m0)
        y_mov = np.where(np.isfinite(y_mov), y_mov, y_m0)

        # Clamp to canvas (positions are centers)
        hw = sizes[movable_idx, 0] / 2
        hh = sizes[movable_idx, 1] / 2
        x_mov = np.clip(x_mov, hw, cw - hw)
        y_mov = np.clip(y_mov, hh, ch - hh)

        pos = pos.copy()
        pos[movable_idx, 0] = x_mov
        pos[movable_idx, 1] = y_mov
        return pos

    @staticmethod
    def _fixed_pos(global_idx, pos, port_pos, n_total):
        """Return (x, y) for a fixed node (macro or port)."""
        if global_idx < n_total:
            return pos[global_idx, 0], pos[global_idx, 1]
        p = global_idx - n_total
        if p < len(port_pos):
            return port_pos[p, 0], port_pos[p, 1]
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Stage 2: Legalization
    # ------------------------------------------------------------------

    def _legalize(self, pos, movable, sizes, cw, ch):
        """
        Minimum-displacement legalization for hard macros.

        Processes macros largest-area-first. For each macro, searches
        outward from its QP position in expanding grid rings until a
        position is found that has no overlap with already-placed macros.
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2  # [n, n]
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2  # [n, n]
        n = len(pos)

        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue

            # Check if current position is already legal
            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                conflict = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                conflict[idx] = False
                if not conflict.any():
                    placed[idx] = True
                    continue

            # Search outward from QP position
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].copy()
            best_d = float("inf")

            for r in range(1, 200):
                found = False
                for dxr in range(-r, r + 1):
                    for dyr in range(-r, r + 1):
                        if abs(dxr) != r and abs(dyr) != r:
                            continue  # Only ring perimeter
                        cx = np.clip(pos[idx, 0] + dxr * step, half_w[idx], cw - half_w[idx])
                        cy = np.clip(pos[idx, 1] + dyr * step, half_h[idx], ch - half_h[idx])
                        if placed.any():
                            dx = np.abs(cx - legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            c = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                            c[idx] = False
                            if c.any():
                                continue
                        d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if d < best_d:
                            best_d = d
                            best_p = np.array([cx, cy])
                            found = True
                if found:
                    break

            legal[idx] = best_p
            placed[idx] = True

        return legal

    # ------------------------------------------------------------------
    # Stage 3: SA Refinement
    # ------------------------------------------------------------------

    def _sa_refine(self, pos, movable, sizes, benchmark, n_hard, n_total, cw, ch):
        """
        Simulated annealing on weighted Manhattan wirelength (hard-hard edges).

        Move types (equal probability):
          - Shift: Gaussian random displacement, scaled by temperature
          - Swap: exchange with a neighbor macro (connectivity-biased)
          - Pull: move fraction of the way toward a connected macro
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2

        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        # Build hard-macro-only weighted edges
        edge_dict: dict = {}
        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = [int(n) for n in nodes_tensor.numpy() if n < n_hard]
            if len(nodes) < 2:
                continue
            w = float(benchmark.net_weights[net_i])
            k = len(nodes)
            if k > MAX_CLIQUE_SIZE:
                anchor = nodes[0]
                for j in nodes[1:]:
                    a, b = (anchor, j) if anchor < j else (j, anchor)
                    edge_dict[(a, b)] = edge_dict.get((a, b), 0.0) + w / (k - 1)
            else:
                ew = w / (k - 1)
                for a in range(k):
                    for b in range(a + 1, k):
                        ni, nj = nodes[a], nodes[b]
                        if ni > nj:
                            ni, nj = nj, ni
                        edge_dict[(ni, nj)] = edge_dict.get((ni, nj), 0.0) + ew

        if not edge_dict:
            return pos

        edges = np.array(list(edge_dict.keys()), dtype=np.int32)   # [E, 2]
        ew = np.array(list(edge_dict.values()), dtype=np.float64)   # [E]

        # Neighbor list for connectivity-biased moves
        neighbors: list = [[] for _ in range(n_hard)]
        for ni, nj in edge_dict:
            neighbors[ni].append(nj)
            neighbors[nj].append(ni)

        def wl_cost():
            dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
            dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
            return (ew * (dx + dy)).sum()

        def has_overlap(idx):
            gap = 0.05
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            ov = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            ov[idx] = False
            return ov.any()

        current_cost = wl_cost()
        best_pos = pos.copy()
        best_cost = current_cost

        T_start = max(cw, ch) * 0.10
        T_end = max(cw, ch) * 0.001

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac

            move_type = random.random()
            i = int(random.choice(movable_idx))
            ox, oy = pos[i, 0], pos[i, 1]

            if move_type < 0.45:
                # Shift: Gaussian displacement scaled by temperature
                sigma = T * (0.2 + 0.8 * (1 - frac))
                pos[i, 0] = np.clip(ox + random.gauss(0, sigma), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(oy + random.gauss(0, sigma), half_h[i], ch - half_h[i])

            elif move_type < 0.75:
                # Swap: exchange positions (prefer connected neighbors)
                if neighbors[i] and random.random() < 0.7:
                    cands = [j for j in neighbors[i] if movable[j]]
                    j = int(random.choice(cands)) if cands else int(random.choice(movable_idx))
                else:
                    j = int(random.choice(movable_idx))

                if i == j:
                    continue
                ojx, ojy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(ojx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(ojy, half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(ox, half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(oy, half_h[j], ch - half_h[j])

                if has_overlap(i) or has_overlap(j):
                    pos[i, 0] = ox;  pos[i, 1] = oy
                    pos[j, 0] = ojx; pos[j, 1] = ojy
                    continue

                new_cost = wl_cost()
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_pos = pos.copy()
                else:
                    pos[i, 0] = ox;  pos[i, 1] = oy
                    pos[j, 0] = ojx; pos[j, 1] = ojy
                continue

            else:
                # Pull: move toward a connected macro
                if not neighbors[i]:
                    continue
                j = int(random.choice(neighbors[i]))
                alpha = random.uniform(0.05, 0.35)
                pos[i, 0] = np.clip(ox + alpha * (pos[j, 0] - ox), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(oy + alpha * (pos[j, 1] - oy), half_h[i], ch - half_h[i])

            # Overlap check for shift/pull moves
            if has_overlap(i):
                pos[i, 0] = ox
                pos[i, 1] = oy
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = ox
                pos[i, 1] = oy

        return best_pos
