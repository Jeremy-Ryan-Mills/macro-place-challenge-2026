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

    def __init__(self, seed: int = 42, lambda_anchor: float = 30.0, sa_iters: int = 200_000):
        self.seed = seed
        self.lambda_anchor = lambda_anchor
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros

        pos = (
            benchmark.macro_positions.numpy().copy().astype(np.float64)
        )  # [n_total, 2]
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)  # [n_total, 2]
        fixed_mask = benchmark.macro_fixed.numpy()  # [n_total] bool
        port_pos = benchmark.port_positions.numpy().astype(np.float64)  # [n_ports, 2]

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Movable: non-fixed hard macros only.
        # Soft macros stay at their initial positions (already co-optimized).
        movable = np.zeros(n_total, dtype=bool)
        movable[:n_hard] = ~fixed_mask[:n_hard]

        def _qp_legalize_candidates(anchor_pos, lambdas):
            """Run QP+legalize for each lambda, return list of (cost, pos)."""
            cands = []
            for lam in lambdas:
                self.lambda_anchor = lam
                p = self._qp_solve(
                    anchor_pos.copy(), movable, sizes, benchmark, n_total, port_pos, cw, ch
                )
                p[:n_hard] = self._legalize(
                    p[:n_hard].copy(), ~fixed_mask[:n_hard], sizes[:n_hard], cw, ch
                )
                cands.append((self._fast_proxy(p, benchmark, n_total, port_pos), p))
            self.lambda_anchor = 30.0
            return cands

        # Round 1: try a range of anchor strengths from the initial positions.
        candidates = _qp_legalize_candidates(pos, [30.0, 100.0, 400.0])

        # Also legalize the initial positions as a fallback.
        pos_init_legal = pos.copy()
        pos_init_legal[:n_hard] = self._legalize(
            pos[:n_hard].copy(), ~fixed_mask[:n_hard], sizes[:n_hard], cw, ch
        )
        candidates.append(
            (self._fast_proxy(pos_init_legal, benchmark, n_total, port_pos), pos_init_legal)
        )

        best_cost, best_pos = min(candidates, key=lambda c: c[0])

        # Iterative refinement: use the best legal positions as the new anchor
        # and re-run QP.  Each iteration, QP finds a WL minimum near a legal
        # solution, then legalization nudges it back to feasibility — this
        # feedback loop steadily improves quality.
        for _ in range(2):
            new_cands = _qp_legalize_candidates(best_pos, [30.0, 100.0, 400.0])
            new_cost, new_pos = min(new_cands, key=lambda c: c[0])
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = new_pos
            else:
                break  # no improvement — stop early

        pos = best_pos

        # SA refinement — compare before/after by proxy cost and keep the better one.
        if self.sa_iters > 0:
            sa_result = self._sa_refine(
                pos[:n_hard].copy(),
                ~fixed_mask[:n_hard],
                sizes[:n_hard],
                benchmark,
                n_hard,
                n_total,
                port_pos,
                cw,
                ch,
            )
            sa_pos = pos.copy()
            sa_pos[:n_hard] = sa_result
            sa_cost = self._fast_proxy(sa_pos, benchmark, n_total, port_pos)
            if sa_cost < best_cost:
                pos = sa_pos

        return torch.tensor(pos, dtype=torch.float32)

    # Fast proxy cost (for conservative fallback comparison)

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
        """Normalized HPWL over all nets — same normalization as plc.get_cost():
        total_wirelength / ((canvas_width + canvas_height) * net_count).
        """
        norm = (benchmark.canvas_width + benchmark.canvas_height) * benchmark.num_nets
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
        return total / norm

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
        n_total_macros = benchmark.num_macros  # hard + soft

        for i in range(n_total_macros):
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

    # Quadratic Placement

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
                    (nodes[a], nodes[b], ew) for a in range(k) for b in range(a + 1, k)
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

    # Legalization

    def _legalize(self, pos, movable, sizes, cw, ch):
        """
        Minimum-displacement legalization for hard macros.

        Two-phase search:
          1. Coarse ring search (step = max(w,h)*0.25) to locate the first
             ring that contains a legal position.
          2. Fine-grained local search within ±coarse_step of the best
             coarse position, using step = min(w,h)*0.08 (≥0.03 μm floor).
             This tightens the final placement toward the true nearest legal
             point without iterating many small rings from the origin.
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2  # [n, n]
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2  # [n, n]
        n = len(pos)

        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()

        def _no_conflict(cx, cy, idx):
            if not placed.any():
                return True
            dx = np.abs(cx - legal[:, 0])
            dy = np.abs(cy - legal[:, 1])
            c = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            c[idx] = False
            return not c.any()

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue

            # Check if current position is already legal
            if _no_conflict(legal[idx, 0], legal[idx, 1], idx):
                placed[idx] = True
                continue

            # ── Phase 1: coarse ring search ──────────────────────────────────
            step_c = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].copy()
            best_d = float("inf")
            found_ring = False

            for r in range(1, 200):
                for dxr in range(-r, r + 1):
                    for dyr in range(-r, r + 1):
                        if abs(dxr) != r and abs(dyr) != r:
                            continue
                        cx = np.clip(pos[idx, 0] + dxr * step_c,
                                     half_w[idx], cw - half_w[idx])
                        cy = np.clip(pos[idx, 1] + dyr * step_c,
                                     half_h[idx], ch - half_h[idx])
                        if not _no_conflict(cx, cy, idx):
                            continue
                        d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if d < best_d:
                            best_d = d
                            best_p = np.array([cx, cy])
                            found_ring = True
                if found_ring:
                    break

            # ── Phase 2: fine-grained local refinement ───────────────────────
            # Scan a dense grid within ±step_c of the best coarse position.
            # This finds the true nearest legal point on a sub-coarse grid.
            step_f = max(min(sizes[idx, 0], sizes[idx, 1]) * 0.08, 0.03)
            x_lo = max(half_w[idx],      best_p[0] - step_c)
            x_hi = min(cw - half_w[idx], best_p[0] + step_c)
            y_lo = max(half_h[idx],      best_p[1] - step_c)
            y_hi = min(ch - half_h[idx], best_p[1] + step_c)

            # Cap fine grid to ≤ 40 × 40 points to bound runtime
            n_x = min(40, max(1, int((x_hi - x_lo) / step_f) + 1))
            n_y = min(40, max(1, int((y_hi - y_lo) / step_f) + 1))
            xs = np.linspace(x_lo, x_hi, n_x)
            ys = np.linspace(y_lo, y_hi, n_y)

            for cx in xs:
                for cy in ys:
                    if not _no_conflict(cx, cy, idx):
                        continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])

            legal[idx] = best_p
            placed[idx] = True

        return legal

    # SA Refinement

    def _sa_refine(self, pos, movable, sizes, benchmark, n_hard, n_total, port_pos, cw, ch):
        """
        Simulated annealing on combined WL + density cost.

        Cost = WL(hard↔hard, hard↔soft, hard↔port) + w_dens * sum_of_squares(density_grid)

        WL uses O(degree) incremental delta updates.
        Density uses an incrementally-maintained grid (soft macros fixed, hard macros movable).
        w_dens is set so that density contributes ~50% of the initial total cost,
        preventing SA from clustering macros to reduce WL at the expense of density.

        Move types: Shift (45%), Swap (30%), Pull (25%)
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2

        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        # ── Build weighted adjacency: hard-macro ↔ {hard macro, soft macro, port} ──
        #
        # Each entry: (endpoint_x, endpoint_y, weight)
        # For hard-macro endpoints we store their index so we can look up pos[idx]
        # dynamically.  For fixed endpoints (soft macros, ports) we snapshot their
        # positions once — they never move during SA.
        #
        # adj_hard[i]  = list of (hard_macro_j_index, weight)   — movable ends
        # adj_fixed[i] = list of (fixed_x, fixed_y, weight)     — immovable ends

        all_macro_pos = benchmark.macro_positions.numpy().astype(np.float64)
        adj_hard:  list = [[] for _ in range(n_hard)]  # (j, w) pairs
        adj_fixed: list = [[] for _ in range(n_hard)]  # (x, y, w) triples

        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = nodes_tensor.numpy().tolist()
            w = float(benchmark.net_weights[net_i])
            hard_nodes = [int(n) for n in raw if n < n_hard]
            if len(hard_nodes) < 1:
                continue

            # Fixed endpoints on this net: soft macros + ports
            fixed_pts = []
            for n in raw:
                n = int(n)
                if n_hard <= n < n_total:
                    # soft macro — use initial (fixed) position
                    fixed_pts.append((all_macro_pos[n, 0], all_macro_pos[n, 1]))
                elif n >= n_total:
                    p = n - n_total
                    if p < len(port_pos):
                        fixed_pts.append((port_pos[p, 0], port_pos[p, 1]))

            k_h = len(hard_nodes)
            k_f = len(fixed_pts)
            k_total = k_h + k_f

            if k_total < 2:
                continue

            # Generate weighted edges (clique or star for large nets)
            if k_total > MAX_CLIQUE_SIZE:
                # Star model: hard anchor → all others
                anchor = hard_nodes[0]
                ew = w / (k_total - 1)
                for j in hard_nodes[1:]:
                    a, b = (anchor, j) if anchor < j else (j, anchor)
                    # Record as hard–hard edge
                    adj_hard[a].append((b, ew))
                    adj_hard[b].append((a, ew))
                for fx, fy in fixed_pts:
                    adj_fixed[anchor].append((fx, fy, ew))
            else:
                ew = 2.0 * w / (k_total * (k_total - 1))
                # hard–hard edges
                for ai in range(k_h):
                    for bi in range(ai + 1, k_h):
                        ni, nj = hard_nodes[ai], hard_nodes[bi]
                        adj_hard[ni].append((nj, ew))
                        adj_hard[nj].append((ni, ew))
                # hard–fixed edges
                for ni in hard_nodes:
                    for fx, fy in fixed_pts:
                        adj_fixed[ni].append((fx, fy, ew))

        # Collapse into numpy arrays per macro for fast vectorised delta computation
        macro_hard_j:  list = []   # macro_hard_j[i]  = int32 array of hard neighbours
        macro_hard_w:  list = []   # macro_hard_w[i]  = float64 weights
        macro_fixed_x: list = []   # macro_fixed_x[i] = float64 array of fixed x coords
        macro_fixed_y: list = []
        macro_fixed_w: list = []

        for i in range(n_hard):
            if adj_hard[i]:
                js, ws = zip(*adj_hard[i])
                macro_hard_j.append(np.array(js, dtype=np.int32))
                macro_hard_w.append(np.array(ws, dtype=np.float64))
            else:
                macro_hard_j.append(np.empty(0, dtype=np.int32))
                macro_hard_w.append(np.empty(0, dtype=np.float64))

            if adj_fixed[i]:
                fxs, fys, fws = zip(*adj_fixed[i])
                macro_fixed_x.append(np.array(fxs, dtype=np.float64))
                macro_fixed_y.append(np.array(fys, dtype=np.float64))
                macro_fixed_w.append(np.array(fws, dtype=np.float64))
            else:
                macro_fixed_x.append(np.empty(0, dtype=np.float64))
                macro_fixed_y.append(np.empty(0, dtype=np.float64))
                macro_fixed_w.append(np.empty(0, dtype=np.float64))

        # Neighbour list (hard–hard only) for connectivity-biased swap/pull moves
        neighbors: list = [list(macro_hard_j[i]) for i in range(n_hard)]

        # ── WL delta helpers ─────────────────────────────────────────────────────

        def delta_move(i, ox, oy, nx, ny) -> float:
            """
            Cost delta for moving macro i from (ox,oy) to (nx,ny).
            Handles both hard-neighbour and fixed-endpoint contributions.
            Also accounts for the change seen from i's hard neighbours (their
            edge to i changes too — captured via symmetric adjacency).
            """
            delta = 0.0
            # Hard neighbours of i: edge (i,j) — counted from i's side
            if len(macro_hard_j[i]):
                js = macro_hard_j[i]
                ox_arr = pos[js, 0]
                oy_arr = pos[js, 1]
                delta += float((macro_hard_w[i] * (
                    np.abs(nx - ox_arr) + np.abs(ny - oy_arr) -
                    np.abs(ox - ox_arr) - np.abs(oy - oy_arr)
                )).sum())
            # Fixed endpoints of i
            if len(macro_fixed_x[i]):
                delta += float((macro_fixed_w[i] * (
                    np.abs(nx - macro_fixed_x[i]) + np.abs(ny - macro_fixed_y[i]) -
                    np.abs(ox - macro_fixed_x[i]) - np.abs(oy - macro_fixed_y[i])
                )).sum())
            return delta

        def delta_swap_wl(i, j, ox_i, oy_i, ox_j, oy_j) -> float:
            """WL delta for swapping i and j (call AFTER applying swap in pos)."""
            nx_i, ny_i = pos[i, 0], pos[i, 1]
            nx_j, ny_j = pos[j, 0], pos[j, 1]
            delta = 0.0
            if len(macro_hard_j[i]):
                js = macro_hard_j[i]; mask = js != j
                js_f, w_f = js[mask], macro_hard_w[i][mask]
                if len(js_f):
                    delta += float((w_f * (
                        np.abs(nx_i - pos[js_f, 0]) + np.abs(ny_i - pos[js_f, 1]) -
                        np.abs(ox_i - pos[js_f, 0]) - np.abs(oy_i - pos[js_f, 1])
                    )).sum())
            if len(macro_fixed_x[i]):
                delta += float((macro_fixed_w[i] * (
                    np.abs(nx_i - macro_fixed_x[i]) + np.abs(ny_i - macro_fixed_y[i]) -
                    np.abs(ox_i - macro_fixed_x[i]) - np.abs(oy_i - macro_fixed_y[i])
                )).sum())
            if len(macro_hard_j[j]):
                js = macro_hard_j[j]; mask = js != i
                js_f, w_f = js[mask], macro_hard_w[j][mask]
                if len(js_f):
                    delta += float((w_f * (
                        np.abs(nx_j - pos[js_f, 0]) + np.abs(ny_j - pos[js_f, 1]) -
                        np.abs(ox_j - pos[js_f, 0]) - np.abs(oy_j - pos[js_f, 1])
                    )).sum())
            if len(macro_fixed_x[j]):
                delta += float((macro_fixed_w[j] * (
                    np.abs(nx_j - macro_fixed_x[j]) + np.abs(ny_j - macro_fixed_y[j]) -
                    np.abs(ox_j - macro_fixed_x[j]) - np.abs(oy_j - macro_fixed_y[j])
                )).sum())
            return delta

        def has_overlap(idx):
            gap = 0.05
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            ov = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            ov[idx] = False
            return ov.any()

        # ── Incremental density grid ─────────────────────────────────────────
        #
        # Tracks macro area coverage per grid cell.  Soft macros contribute a
        # fixed baseline; only hard macro movements update the grid.  We use
        # sum-of-squares as the density penalty: cheap to update incrementally
        # and strongly penalises hotspots.

        cols_d = benchmark.grid_cols
        rows_d = benchmark.grid_rows
        cell_w_d = cw / cols_d
        cell_h_d = ch / rows_d
        cell_area_d = cell_w_d * cell_h_d
        all_sizes_d = benchmark.macro_sizes.numpy().astype(np.float64)

        def _cells(x, y, mw, mh):
            hw_, hh_ = mw / 2, mh / 2
            xl_, xr_ = x - hw_, x + hw_
            yl_, yr_ = y - hh_, y + hh_
            c_s = max(0, int(xl_ / cell_w_d))
            c_e = min(cols_d - 1, int(xr_ / cell_w_d))
            r_s = max(0, int(yl_ / cell_h_d))
            r_e = min(rows_d - 1, int(yr_ / cell_h_d))
            result = []
            for r in range(r_s, r_e + 1):
                oy_ = max(0.0, min(yr_, (r + 1) * cell_h_d) - max(yl_, r * cell_h_d))
                if oy_ <= 0:
                    continue
                for c in range(c_s, c_e + 1):
                    ox_ = max(0.0, min(xr_, (c + 1) * cell_w_d) - max(xl_, c * cell_w_d))
                    if ox_ > 0:
                        result.append((r * cols_d + c, ox_ * oy_ / cell_area_d))
            return result

        density_grid = np.zeros(rows_d * cols_d, dtype=np.float64)
        for i_sm in range(n_hard, benchmark.num_macros):   # soft macros (fixed)
            for cell, frac in _cells(all_macro_pos[i_sm, 0], all_macro_pos[i_sm, 1],
                                     all_sizes_d[i_sm, 0], all_sizes_d[i_sm, 1]):
                density_grid[cell] += frac
        for i_hm in range(n_hard):                         # hard macros (current pos)
            for cell, frac in _cells(pos[i_hm, 0], pos[i_hm, 1],
                                     sizes[i_hm, 0], sizes[i_hm, 1]):
                density_grid[cell] += frac

        def _dens_changes(x_old, y_old, x_new, y_new, mw, mh):
            """Dict of {cell: net_density_change} for one macro move."""
            ch_dict = {}
            for cell, f in _cells(x_old, y_old, mw, mh):
                ch_dict[cell] = ch_dict.get(cell, 0.0) - f
            for cell, f in _cells(x_new, y_new, mw, mh):
                ch_dict[cell] = ch_dict.get(cell, 0.0) + f
            return ch_dict

        def _delta_sos(ch_dict):
            """Delta in sum-of-squares density for a given change dict."""
            d = 0.0
            for cell, dc in ch_dict.items():
                dg = density_grid[cell]
                d += (dg + dc) ** 2 - dg ** 2
            return d

        def _apply_changes(ch_dict):
            for cell, dc in ch_dict.items():
                density_grid[cell] += dc

        # ── Initialise combined cost ─────────────────────────────────────────

        init_wl = 0.0
        for i_c in range(n_hard):
            if len(macro_hard_j[i_c]):
                js = macro_hard_j[i_c]
                init_wl += float((macro_hard_w[i_c] * (
                    np.abs(pos[i_c, 0] - pos[js, 0]) + np.abs(pos[i_c, 1] - pos[js, 1])
                )).sum())
            if len(macro_fixed_x[i_c]):
                init_wl += float((macro_fixed_w[i_c] * (
                    np.abs(pos[i_c, 0] - macro_fixed_x[i_c]) +
                    np.abs(pos[i_c, 1] - macro_fixed_y[i_c])
                )).sum())

        init_sos = float((density_grid ** 2).sum())
        # w_dens: weight so density starts contributing 50% of total cost
        w_dens = (0.5 * init_wl / init_sos) if init_sos > 1e-12 else 0.0

        current_wl  = init_wl
        current_sos = init_sos
        current_cost = current_wl + w_dens * current_sos
        best_pos  = pos.copy()
        best_cost = current_cost

        T_start = max(cw, ch) * 0.10
        T_end   = max(cw, ch) * 0.001

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac

            move_type = random.random()
            i = int(random.choice(movable_idx))
            ox, oy = pos[i, 0], pos[i, 1]
            j = None
            ojx = ojy = 0.0
            ch_dict = None

            if move_type < 0.45:
                # Shift
                sigma = T * (0.2 + 0.8 * (1 - frac))
                nx = np.clip(ox + random.gauss(0, sigma), half_w[i], cw - half_w[i])
                ny = np.clip(oy + random.gauss(0, sigma), half_h[i], ch - half_h[i])
                pos[i, 0] = nx; pos[i, 1] = ny
                if has_overlap(i):
                    pos[i, 0] = ox; pos[i, 1] = oy
                    continue
                delta_wl = delta_move(i, ox, oy, nx, ny)
                ch_dict  = _dens_changes(ox, oy, nx, ny, sizes[i, 0], sizes[i, 1])

            elif move_type < 0.75:
                # Swap (prefer connected neighbours)
                if neighbors[i] and random.random() < 0.7:
                    cands = [nb for nb in neighbors[i] if movable[nb]]
                    j = int(random.choice(cands)) if cands else int(random.choice(movable_idx))
                else:
                    j = int(random.choice(movable_idx))
                if i == j:
                    continue
                ojx, ojy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(ojx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(ojy, half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(ox,  half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(oy,  half_h[j], ch - half_h[j])
                if has_overlap(i) or has_overlap(j):
                    pos[i, 0] = ox;  pos[i, 1] = oy
                    pos[j, 0] = ojx; pos[j, 1] = ojy
                    continue
                delta_wl = delta_swap_wl(i, j, ox, oy, ojx, ojy)
                # Merge both moves' density changes simultaneously
                c1 = _dens_changes(ox, oy, pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1])
                c2 = _dens_changes(ojx, ojy, pos[j, 0], pos[j, 1], sizes[j, 0], sizes[j, 1])
                ch_dict = dict(c1)
                for cell, dc in c2.items():
                    ch_dict[cell] = ch_dict.get(cell, 0.0) + dc

            else:
                # Pull toward a connected node
                if not neighbors[i]:
                    continue
                nb = int(random.choice(neighbors[i]))
                alpha = random.uniform(0.05, 0.35)
                nx = np.clip(ox + alpha * (pos[nb, 0] - ox), half_w[i], cw - half_w[i])
                ny = np.clip(oy + alpha * (pos[nb, 1] - oy), half_h[i], ch - half_h[i])
                pos[i, 0] = nx; pos[i, 1] = ny
                if has_overlap(i):
                    pos[i, 0] = ox; pos[i, 1] = oy
                    continue
                delta_wl = delta_move(i, ox, oy, nx, ny)
                ch_dict  = _dens_changes(ox, oy, nx, ny, sizes[i, 0], sizes[i, 1])

            delta_sos  = _delta_sos(ch_dict)
            delta      = delta_wl + w_dens * delta_sos

            # Metropolis acceptance
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_wl  += delta_wl
                current_sos += delta_sos
                current_cost = current_wl + w_dens * current_sos
                _apply_changes(ch_dict)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos  = pos.copy()
            else:
                pos[i, 0] = ox; pos[i, 1] = oy
                if j is not None:
                    pos[j, 0] = ojx; pos[j, 1] = ojy

        return best_pos
