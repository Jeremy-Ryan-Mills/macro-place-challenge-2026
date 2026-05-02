"""
DREAMPlace-style macro placer.

Optimizes macro positions via gradient descent with a growing density penalty,
then legalizes and refines with SA.

Key idea: density penalty starts small so the optimizer can minimize WL freely
(with overlaps allowed), then grows exponentially to force macros to spread
out. This is the core DREAMPlace insight applied to macro placement.

Pipeline:
  1. Gradient descent: smooth WL + lambda*density, lambda grows over stages
  2. Legalize: minimum-displacement overlap removal
  3. SA: RUDY-tracked refinement (WL + density + congestion)

Nets are capped at 8000 by weight in both gradient and SA phases to keep
each step tractable on large benchmarks. Density uses a coarse 32×32 grid.

WL approximation: log-sum-exp of macro/port coordinates per net.
Density: separable Gaussian kernels evaluated on the placement grid.
Optimizer: Adam with cosine LR decay and exponential lambda schedule.

Usage:
    uv run evaluate submissions/dreamplace/placer.py
    uv run evaluate submissions/dreamplace/placer.py --all
    uv run evaluate submissions/dreamplace/placer.py -b ibm03
"""

import math
import random

import numpy as np
import torch

from macro_place.benchmark import Benchmark

MAX_CLIQUE_SIZE = 20


class DreamPlacer:
    def __init__(
        self,
        seed: int = 42,
        n_outer: int = 5,     # number of projection + lr-update checkpoints
        n_inner: int = 100,   # Adam steps per checkpoint (total = n_outer * n_inner)
        lambda_min: float = 1e-3,
        lambda_max: float = 4.0,
        sa_iters: int = 60_000,
    ):
        self.seed = seed
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        pos = benchmark.macro_positions.numpy().copy().astype(np.float64)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        fixed_mask = benchmark.macro_fixed.numpy()
        port_pos = benchmark.port_positions.numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        movable_hard = ~fixed_mask[:n_hard]
        movable_idx = np.where(movable_hard)[0]

        if len(movable_idx) == 0:
            return torch.tensor(pos, dtype=torch.float32)

        nt = self._build_net_tensors(
            benchmark, movable_idx, n_hard, n_total, port_pos, cw, ch
        )

        grad_hard = self._gradient_phase(
            pos[:n_hard].copy(), movable_idx, sizes[:n_hard], nt, cw, ch
        )

        best_pos = pos.copy()
        best_pos[:n_hard] = self._legalize(grad_hard, movable_hard, sizes[:n_hard], cw, ch)
        best_cost = self._fast_proxy(best_pos, benchmark, n_total, port_pos, nt)

        # Also keep legalized initial positions as fallback
        init_legal = pos.copy()
        init_legal[:n_hard] = self._legalize(
            pos[:n_hard].copy(), movable_hard, sizes[:n_hard], cw, ch
        )
        init_cost = self._fast_proxy(init_legal, benchmark, n_total, port_pos, nt)
        if init_cost < best_cost:
            best_cost = init_cost
            best_pos = init_legal

        if self.sa_iters > 0:
            sa_result = self._sa_refine(
                best_pos[:n_hard].copy(),
                movable_hard,
                sizes[:n_hard],
                benchmark,
                n_hard,
                n_total,
                port_pos,
                cw,
                ch,
            )
            sa_pos = best_pos.copy()
            sa_pos[:n_hard] = sa_result
            sa_cost = self._fast_proxy(sa_pos, benchmark, n_total, port_pos, nt)
            if sa_cost < best_cost:
                best_pos = sa_pos

        return torch.tensor(best_pos, dtype=torch.float32)

    # Net tensor construction

    def _build_net_tensors(
        self, benchmark, movable_idx, n_hard, n_total, port_pos, cw, ch
    ):
        """
        Precompute all tensors needed by the gradient phase.

        Returns a dict containing:
          - Padded batch tensors for batched LSE wirelength computation
          - Gaussian density tensors (cell centers, per-macro sigma, soft-macro baseline)
          - Per-macro canvas bounds
        """
        n_mov = len(movable_idx)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        all_pos = benchmark.macro_positions.numpy().astype(np.float64)
        glob_to_loc = {int(g): l for l, g in enumerate(movable_idx)}

        raw_mov, raw_fixed, raw_w = [], [], []
        for nodes_tensor, w_t in zip(benchmark.net_nodes, benchmark.net_weights):
            nodes = [int(n) for n in nodes_tensor.numpy()]
            if len(nodes) < 2:
                continue
            w = float(w_t)
            mov_local = [glob_to_loc[n] for n in nodes if n in glob_to_loc]
            fixed_pts = []
            for n in nodes:
                if n not in glob_to_loc:
                    if n < n_total:
                        fixed_pts.append([float(all_pos[n, 0]), float(all_pos[n, 1])])
                    else:
                        p = n - n_total
                        if p < len(port_pos):
                            fixed_pts.append(
                                [float(port_pos[p, 0]), float(port_pos[p, 1])]
                            )
            if len(mov_local) + len(fixed_pts) < 2:
                continue
            raw_mov.append(mov_local)
            raw_fixed.append(fixed_pts)
            raw_w.append(w)

        # Cap gradient nets by weight to keep each step tractable
        MAX_GRAD_NETS = 8000
        if len(raw_w) > MAX_GRAD_NETS:
            order = sorted(range(len(raw_w)), key=lambda i: -raw_w[i])[:MAX_GRAD_NETS]
            raw_mov = [raw_mov[i] for i in order]
            raw_fixed = [raw_fixed[i] for i in order]
            raw_w = [raw_w[i] for i in order]

        n_nets = len(raw_w)

        if n_nets == 0:
            # No nets involving movable macros — return dummy tensors
            mov_slots = torch.zeros(1, 1, dtype=torch.int64)
            fixed_xy_pad = torch.zeros(1, 1, 2)
            is_movable = torch.zeros(1, 1, dtype=torch.bool)
            is_valid = torch.zeros(1, 1, dtype=torch.bool)
            net_weights_t = torch.zeros(1)
        else:
            MAX_K = min(
                max(len(m) + len(f) for m, f in zip(raw_mov, raw_fixed)), 64
            )
            mov_slots = torch.full((n_nets, MAX_K), n_mov, dtype=torch.int64)
            fixed_xy_pad = torch.zeros(n_nets, MAX_K, 2, dtype=torch.float32)
            is_movable = torch.zeros(n_nets, MAX_K, dtype=torch.bool)
            is_valid = torch.zeros(n_nets, MAX_K, dtype=torch.bool)
            net_weights_t = torch.tensor(raw_w, dtype=torch.float32)

            for i, (ml, fl) in enumerate(zip(raw_mov, raw_fixed)):
                k_m = min(len(ml), MAX_K)
                k_f = min(len(fl), MAX_K - k_m)
                if k_m > 0:
                    mov_slots[i, :k_m] = torch.tensor(ml[:k_m], dtype=torch.int64)
                    is_movable[i, :k_m] = True
                    is_valid[i, :k_m] = True
                if k_f > 0:
                    fixed_xy_pad[i, k_m : k_m + k_f] = torch.tensor(
                        fl[:k_f], dtype=torch.float32
                    )
                    is_valid[i, k_m : k_m + k_f] = True

        # Density grid setup — coarser grid to keep gradient step fast
        MAX_DENS_DIM = 32
        rows = min(benchmark.grid_rows, MAX_DENS_DIM)
        cols = min(benchmark.grid_cols, MAX_DENS_DIM)
        n_cells = rows * cols
        cell_w = cw / cols
        cell_h = ch / rows

        # Cell centers in row-major order: [n_cells]
        cx_arr = (np.arange(cols) + 0.5) * cell_w
        cy_arr = (np.arange(rows) + 0.5) * cell_h
        cell_cx = torch.tensor(np.tile(cx_arr, rows), dtype=torch.float32)
        cell_cy = torch.tensor(np.repeat(cy_arr, cols), dtype=torch.float32)

        # Sigma = macro full size. Wide Gaussian = effective spreading pressure.
        sigma_mov = torch.tensor(sizes[movable_idx], dtype=torch.float32).clamp(min=1e-3)
        inv_sx = 1.0 / sigma_mov[:, 0]
        inv_sy = 1.0 / sigma_mov[:, 1]

        # Precompute scaled cell centers to avoid repeated division in the inner loop
        # scaled_cc_x[i, c] = cell_cx[c] / sigma_x[i]
        scaled_cc_x = inv_sx[:, None] * cell_cx[None, :]  # [n_mov, n_cells]
        scaled_cc_y = inv_sy[:, None] * cell_cy[None, :]

        # Fixed density contribution from soft macros (never changes during optimization)
        soft_dens = torch.zeros(n_cells, dtype=torch.float32)
        for i_sm in range(n_hard, n_total):
            sx = float(all_pos[i_sm, 0])
            sy = float(all_pos[i_sm, 1])
            swx = max(float(benchmark.macro_sizes[i_sm, 0]), 1e-3)
            swy = max(float(benchmark.macro_sizes[i_sm, 1]), 1e-3)
            dx = (cell_cx - sx) / swx
            dy = (cell_cy - sy) / swy
            soft_dens = soft_dens + torch.exp(-0.5 * (dx * dx + dy * dy))

        # Per-macro canvas bounds for projection and boundary penalty
        hw_arr = sizes[movable_idx, 0] / 2
        hh_arr = sizes[movable_idx, 1] / 2
        hw_t = torch.tensor(hw_arr, dtype=torch.float32)
        hh_t = torch.tensor(hh_arr, dtype=torch.float32)
        x_lo = hw_t.clone()
        x_hi = torch.tensor(cw - hw_arr, dtype=torch.float32)
        y_lo = hh_t.clone()
        y_hi = torch.tensor(ch - hh_arr, dtype=torch.float32)

        alpha = float(max(cw, ch)) / 100.0
        wl_norm = float((cw + ch) * benchmark.num_nets)

        # Store raw net data for vectorized fast HPWL (avoids re-iterating all nets)
        hpwl_movable_idx = raw_mov    # list of lists of local movable indices
        hpwl_fixed_xy = raw_fixed     # list of lists of [x, y] fixed points
        hpwl_weights = np.array(raw_w, dtype=np.float64)

        return dict(
            mov_slots=mov_slots,
            fixed_xy_pad=fixed_xy_pad,
            is_movable=is_movable,
            is_valid=is_valid,
            net_weights_t=net_weights_t,
            scaled_cc_x=scaled_cc_x,
            scaled_cc_y=scaled_cc_y,
            inv_sx=inv_sx,
            inv_sy=inv_sy,
            soft_dens=soft_dens,
            hw_t=hw_t,
            hh_t=hh_t,
            x_lo=x_lo,
            x_hi=x_hi,
            y_lo=y_lo,
            y_hi=y_hi,
            alpha=alpha,
            wl_norm=wl_norm,
            n_cells=n_cells,
            n_mov=n_mov,
            hpwl_movable_idx=hpwl_movable_idx,
            hpwl_fixed_xy=hpwl_fixed_xy,
            hpwl_weights=hpwl_weights,
            movable_idx=movable_idx,
        )

    # Gradient phase

    def _gradient_phase(self, pos_hard, movable_idx, sizes_hard, nt, cw, ch):
        """
        Density-annealed gradient descent using a single Adam optimizer.

        Total steps = n_outer * n_inner. Both the density weight (lambda) and
        learning rate are scheduled over the full run:
          - lambda: exponential from lambda_min → lambda_max
          - lr: cosine decay from lr_max → lr_min

        Every n_inner steps, positions are projected back to canvas bounds
        to prevent the boundary penalty from dominating.
        """
        n_mov = nt["n_mov"]
        NEG_INF = -1e9

        mov_slots = nt["mov_slots"]
        fixed_xy_pad = nt["fixed_xy_pad"]
        is_movable = nt["is_movable"]
        is_valid = nt["is_valid"]
        net_weights_t = nt["net_weights_t"]
        scaled_cc_x = nt["scaled_cc_x"]
        scaled_cc_y = nt["scaled_cc_y"]
        inv_sx = nt["inv_sx"]
        inv_sy = nt["inv_sy"]
        soft_dens = nt["soft_dens"]
        hw_t = nt["hw_t"]
        hh_t = nt["hh_t"]
        x_lo = nt["x_lo"]
        x_hi = nt["x_hi"]
        y_lo = nt["y_lo"]
        y_hi = nt["y_hi"]
        alpha = nt["alpha"]
        wl_norm = nt["wl_norm"]
        n_cells = nt["n_cells"]

        safe_slots = mov_slots.clamp(0, n_mov - 1)

        x = torch.tensor(pos_hard[movable_idx], dtype=torch.float32, requires_grad=True)

        lr_max = max(cw, ch) / 50.0
        lr_min = max(cw, ch) / 500.0
        optimizer = torch.optim.Adam([x], lr=lr_max)

        total_steps = self.n_outer * self.n_inner

        for step in range(total_steps):
            t = step / max(total_steps - 1, 1)

            lam = self.lambda_min * (self.lambda_max / self.lambda_min) ** t

            # Cosine lr decay: large step early (cluster for WL), small later (fine-tune density)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()

            # Batched log-sum-exp wirelength
            gathered = x[safe_slots]  # [n_nets, MAX_K, 2]
            all_xy = torch.where(is_movable[:, :, None], gathered, fixed_xy_pad)

            all_x = all_xy[:, :, 0].masked_fill(~is_valid, NEG_INF)
            neg_x = (-all_xy[:, :, 0]).masked_fill(~is_valid, NEG_INF)
            all_y = all_xy[:, :, 1].masked_fill(~is_valid, NEG_INF)
            neg_y = (-all_xy[:, :, 1]).masked_fill(~is_valid, NEG_INF)

            wl = (
                net_weights_t
                * (
                    alpha * torch.logsumexp(all_x / alpha, dim=1)
                    + alpha * torch.logsumexp(neg_x / alpha, dim=1)
                    + alpha * torch.logsumexp(all_y / alpha, dim=1)
                    + alpha * torch.logsumexp(neg_y / alpha, dim=1)
                )
            ).sum() / wl_norm

            # Separable Gaussian density: [n_mov, n_cells] → [n_cells]
            xs = (x[:, 0] * inv_sx)[:, None] - scaled_cc_x
            ys = (x[:, 1] * inv_sy)[:, None] - scaled_cc_y
            density = torch.exp(-0.5 * (xs * xs + ys * ys)).sum(0) + soft_dens
            dens_loss = (density * density).sum() / n_cells

            # Quadratic boundary penalty
            bnd = (
                torch.relu(hw_t - x[:, 0]).pow(2)
                + torch.relu(x[:, 0] - (cw - hw_t)).pow(2)
                + torch.relu(hh_t - x[:, 1]).pow(2)
                + torch.relu(x[:, 1] - (ch - hh_t)).pow(2)
            ).sum()

            loss = wl + lam * dens_loss + 10.0 * bnd
            loss.backward()
            optimizer.step()

            # Project to canvas every n_inner steps
            if (step + 1) % self.n_inner == 0:
                with torch.no_grad():
                    x[:, 0] = torch.max(torch.min(x[:, 0], x_hi), x_lo)
                    x[:, 1] = torch.max(torch.min(x[:, 1], y_hi), y_lo)

        result = pos_hard.copy()
        result[movable_idx] = x.detach().numpy()
        return result

    # Legalization

    def _legalize(self, pos, movable, sizes, cw, ch):
        """
        Minimum-displacement legalization for hard macros.

        Two-phase search per macro (largest-area-first order):
          Phase 1: coarse ring search to find the nearest ring with a legal slot
          Phase 2: vectorized fine-grained search within that ring
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2
        n = len(pos)

        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()

        def no_conflict(cx, cy, idx):
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
            if no_conflict(legal[idx, 0], legal[idx, 1], idx):
                placed[idx] = True
                continue

            step_c = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].copy()
            best_d = float("inf")
            found_ring = False

            for r in range(1, 200):
                for dxr in range(-r, r + 1):
                    for dyr in range(-r, r + 1):
                        if abs(dxr) != r and abs(dyr) != r:
                            continue
                        cx = np.clip(
                            pos[idx, 0] + dxr * step_c, half_w[idx], cw - half_w[idx]
                        )
                        cy = np.clip(
                            pos[idx, 1] + dyr * step_c, half_h[idx], ch - half_h[idx]
                        )
                        if not no_conflict(cx, cy, idx):
                            continue
                        d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if d < best_d:
                            best_d = d
                            best_p = np.array([cx, cy])
                            found_ring = True
                if found_ring:
                    break

            # Fine grid search around the best coarse position
            step_f = max(min(sizes[idx, 0], sizes[idx, 1]) * 0.08, 0.03)
            x_lo = max(half_w[idx], best_p[0] - step_c)
            x_hi = min(cw - half_w[idx], best_p[0] + step_c)
            y_lo = max(half_h[idx], best_p[1] - step_c)
            y_hi = min(ch - half_h[idx], best_p[1] + step_c)

            n_x = min(40, max(1, int((x_hi - x_lo) / step_f) + 1))
            n_y = min(40, max(1, int((y_hi - y_lo) / step_f) + 1))
            gx = np.linspace(x_lo, x_hi, n_x)
            gy = np.linspace(y_lo, y_hi, n_y)
            gx2, gy2 = np.meshgrid(gx, gy)
            pts_x = gx2.ravel()
            pts_y = gy2.ravel()

            pmask = placed.copy()
            pmask[idx] = False
            pidx = np.where(pmask)[0]

            if len(pidx) == 0:
                dists2 = (pts_x - pos[idx, 0]) ** 2 + (pts_y - pos[idx, 1]) ** 2
                k = int(np.argmin(dists2))
                if dists2[k] < best_d:
                    best_p = np.array([pts_x[k], pts_y[k]])
            else:
                px_p = legal[pidx, 0]
                py_p = legal[pidx, 1]
                sx_p = sep_x[idx, pidx] + 0.05
                sy_p = sep_y[idx, pidx] + 0.05

                dx2d = np.abs(pts_x[:, np.newaxis] - px_p[np.newaxis, :])
                dy2d = np.abs(pts_y[:, np.newaxis] - py_p[np.newaxis, :])
                valid = ~((dx2d < sx_p) & (dy2d < sy_p)).any(axis=1)

                if valid.any():
                    vx = pts_x[valid]
                    vy = pts_y[valid]
                    dists2 = (vx - pos[idx, 0]) ** 2 + (vy - pos[idx, 1]) ** 2
                    k = int(np.argmin(dists2))
                    if dists2[k] < best_d:
                        best_p = np.array([vx[k], vy[k]])

            legal[idx] = best_p
            placed[idx] = True

        return legal

    # SA refinement

    def _sa_refine(
        self, pos, movable, sizes, benchmark, n_hard, n_total, port_pos, cw, ch
    ):
        """
        SA refinement on WL + density + RUDY routing congestion.

        Cost = WL + w_dens * density_SOS + w_cong * congestion_SOS

        WL and density are updated incrementally per move.
        RUDY congestion is updated incrementally via bbox tracking per net.
        Only nets with at least one hard macro are tracked (fixed nets contribute
        a constant baseline folded into the initial demand grid).

        Move types: shift (45%), swap (30%), pull toward neighbor (25%).
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2

        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        all_pos = benchmark.macro_positions.numpy().astype(np.float64)

        # Build WL adjacency lists (clique/star expansion per net)
        adj_hard = [[] for _ in range(n_hard)]
        adj_fixed = [[] for _ in range(n_hard)]

        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = nodes_tensor.numpy().tolist()
            w = float(benchmark.net_weights[net_i])
            hard_nodes = [int(n) for n in raw if n < n_hard]
            if not hard_nodes:
                continue

            fixed_pts = []
            for n in raw:
                n = int(n)
                if n_hard <= n < n_total:
                    fixed_pts.append((all_pos[n, 0], all_pos[n, 1]))
                elif n >= n_total:
                    p = n - n_total
                    if p < len(port_pos):
                        fixed_pts.append((port_pos[p, 0], port_pos[p, 1]))

            k_total = len(hard_nodes) + len(fixed_pts)
            if k_total < 2:
                continue

            if k_total > MAX_CLIQUE_SIZE:
                anchor = hard_nodes[0]
                ew = w / (k_total - 1)
                for j in hard_nodes[1:]:
                    adj_hard[anchor].append((j, ew))
                    adj_hard[j].append((anchor, ew))
                for fx, fy in fixed_pts:
                    adj_fixed[anchor].append((fx, fy, ew))
            else:
                ew = 2.0 * w / (k_total * (k_total - 1))
                for ai in range(len(hard_nodes)):
                    for bi in range(ai + 1, len(hard_nodes)):
                        ni, nj = hard_nodes[ai], hard_nodes[bi]
                        adj_hard[ni].append((nj, ew))
                        adj_hard[nj].append((ni, ew))
                for ni in hard_nodes:
                    for fx, fy in fixed_pts:
                        adj_fixed[ni].append((fx, fy, ew))

        macro_hard_j, macro_hard_w = [], []
        macro_fixed_x, macro_fixed_y, macro_fixed_w = [], [], []

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

        neighbors = [list(macro_hard_j[i]) for i in range(n_hard)]

        def delta_wl(i, ox, oy, nx, ny):
            d = 0.0
            if len(macro_hard_j[i]):
                js = macro_hard_j[i]
                d += float(
                    (
                        macro_hard_w[i]
                        * (
                            np.abs(nx - pos[js, 0])
                            + np.abs(ny - pos[js, 1])
                            - np.abs(ox - pos[js, 0])
                            - np.abs(oy - pos[js, 1])
                        )
                    ).sum()
                )
            if len(macro_fixed_x[i]):
                d += float(
                    (
                        macro_fixed_w[i]
                        * (
                            np.abs(nx - macro_fixed_x[i])
                            + np.abs(ny - macro_fixed_y[i])
                            - np.abs(ox - macro_fixed_x[i])
                            - np.abs(oy - macro_fixed_y[i])
                        )
                    ).sum()
                )
            return d

        def has_overlap(idx):
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            ov = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05)
            ov[idx] = False
            return ov.any()

        # Incremental density grid
        cols_d = benchmark.grid_cols
        rows_d = benchmark.grid_rows
        cell_w_d = cw / cols_d
        cell_h_d = ch / rows_d
        cell_area_d = cell_w_d * cell_h_d
        all_sizes_d = benchmark.macro_sizes.numpy().astype(np.float64)

        def macro_cells(x, y, mw, mh):
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
                    ox_ = max(
                        0.0, min(xr_, (c + 1) * cell_w_d) - max(xl_, c * cell_w_d)
                    )
                    if ox_ > 0:
                        result.append((r * cols_d + c, ox_ * oy_ / cell_area_d))
            return result

        density_grid = np.zeros(rows_d * cols_d, dtype=np.float64)
        for i_sm in range(n_hard, benchmark.num_macros):
            for cell, frac in macro_cells(
                all_pos[i_sm, 0], all_pos[i_sm, 1],
                all_sizes_d[i_sm, 0], all_sizes_d[i_sm, 1],
            ):
                density_grid[cell] += frac
        for i_hm in range(n_hard):
            for cell, frac in macro_cells(
                pos[i_hm, 0], pos[i_hm, 1], sizes[i_hm, 0], sizes[i_hm, 1]
            ):
                density_grid[cell] += frac

        def dens_changes(x_old, y_old, x_new, y_new, mw, mh):
            ch_dict = {}
            for cell, f in macro_cells(x_old, y_old, mw, mh):
                ch_dict[cell] = ch_dict.get(cell, 0.0) - f
            for cell, f in macro_cells(x_new, y_new, mw, mh):
                ch_dict[cell] = ch_dict.get(cell, 0.0) + f
            return ch_dict

        def delta_sos(ch_dict):
            d = 0.0
            for cell, dc in ch_dict.items():
                d += (density_grid[cell] + dc) ** 2 - density_grid[cell] ** 2
            return d

        def apply_dens(ch_dict):
            for cell, dc in ch_dict.items():
                density_grid[cell] += dc

        # RUDY incremental routing demand grid
        h_cap = float(benchmark.hroutes_per_micron) * cell_w_d
        v_cap = float(benchmark.vroutes_per_micron) * cell_h_d

        # Per routing-net data for demand tracking
        rn_hard = []        # hard macro indices in this net
        rn_fx = []          # fixed endpoint x coords
        rn_fy = []          # fixed endpoint y coords
        rn_wt = []          # net weight
        rn_bbox = []        # current [xl, xr, yl, yr]
        macro_to_rn = [[] for _ in range(n_hard)]

        # Collect all nets with hard macros, then keep only the top N by weight.
        # This keeps per-move RUDY cost manageable on dense netlists.
        MAX_RUDY_NETS = 8000
        rudy_candidates = []
        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = [int(n) for n in nodes_tensor.numpy()]
            w_net = float(benchmark.net_weights[net_i])
            hn = [n for n in raw if n < n_hard]
            if not hn:
                continue
            fx, fy = [], []
            for n in raw:
                if n >= n_hard:
                    if n < n_total:
                        fx.append(float(all_pos[n, 0]))
                        fy.append(float(all_pos[n, 1]))
                    else:
                        p = n - n_total
                        if p < len(port_pos):
                            fx.append(float(port_pos[p, 0]))
                            fy.append(float(port_pos[p, 1]))
            rudy_candidates.append((w_net, hn, fx, fy))

        rudy_candidates.sort(key=lambda x: -x[0])
        for w_net, hn, fx, fy in rudy_candidates[:MAX_RUDY_NETS]:
            rn_idx = len(rn_hard)
            rn_hard.append(np.array(hn, dtype=np.int32))
            rn_fx.append(np.array(fx))
            rn_fy.append(np.array(fy))
            rn_wt.append(w_net)
            rn_bbox.append([0.0, 0.0, 0.0, 0.0])
            for hi in hn:
                macro_to_rn[hi].append(rn_idx)

        def rn_current_bbox(rn_idx):
            hn = rn_hard[rn_idx]
            xs = list(pos[hn, 0])
            ys = list(pos[hn, 1])
            if len(rn_fx[rn_idx]):
                xs.extend(rn_fx[rn_idx])
                ys.extend(rn_fy[rn_idx])
            return [min(xs), max(xs), min(ys), max(ys)]

        def rudy_ch(xl, xr, yl, yr, xl2, xr2, yl2, yr2, w):
            """Demand change dict for one net bbox update (old → new)."""
            ch = {}
            dx = max(xr - xl, cell_w_d)
            dy = max(yr - yl, cell_h_d)
            dx2 = max(xr2 - xl2, cell_w_d)
            dy2 = max(yr2 - yl2, cell_h_d)
            h_rm = -w * cell_h_d / dy / h_cap
            v_rm = -w * cell_w_d / dx / v_cap
            h_ad = w * cell_h_d / dy2 / h_cap
            v_ad = w * cell_w_d / dx2 / v_cap
            for xl_b, xr_b, yl_b, yr_b, dh, dv in [
                (xl, xr, yl, yr, h_rm, v_rm),
                (xl2, xr2, yl2, yr2, h_ad, v_ad),
            ]:
                c_s = max(0, int(xl_b / cell_w_d))
                c_e = min(cols_d - 1, int(xr_b / cell_w_d))
                r_s = max(0, int(yl_b / cell_h_d))
                r_e = min(rows_d - 1, int(yr_b / cell_h_d))
                for r in range(r_s, r_e + 1):
                    for c in range(c_s, c_e + 1):
                        k = r * cols_d + c
                        prev = ch.get(k, (0.0, 0.0))
                        ch[k] = (prev[0] + dh, prev[1] + dv)
            return ch

        def delta_cong_sos(cong_ch):
            d = 0.0
            for k, (dh, dv) in cong_ch.items():
                d += (
                    (h_dem[k] + dh) ** 2 - h_dem[k] ** 2
                    + (v_dem[k] + dv) ** 2 - v_dem[k] ** 2
                )
            return d

        def apply_cong(cong_ch):
            for k, (dh, dv) in cong_ch.items():
                h_dem[k] += dh
                v_dem[k] += dv

        h_dem = np.zeros(rows_d * cols_d, dtype=np.float64)
        v_dem = np.zeros(rows_d * cols_d, dtype=np.float64)

        # Fixed nets (no hard macros): contribute constant baseline demand
        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = [int(n) for n in nodes_tensor.numpy()]
            if any(n < n_hard for n in raw):
                continue
            w_net = float(benchmark.net_weights[net_i])
            xs, ys = [], []
            for n in raw:
                if n < n_total:
                    xs.append(float(all_pos[n, 0]))
                    ys.append(float(all_pos[n, 1]))
                else:
                    p = n - n_total
                    if p < len(port_pos):
                        xs.append(float(port_pos[p, 0]))
                        ys.append(float(port_pos[p, 1]))
            if len(xs) < 2:
                continue
            xl, xr, yl, yr = min(xs), max(xs), min(ys), max(ys)
            dx = max(xr - xl, cell_w_d)
            dy = max(yr - yl, cell_h_d)
            hc = w_net * cell_h_d / dy / h_cap
            vc = w_net * cell_w_d / dx / v_cap
            c_s = max(0, int(xl / cell_w_d))
            c_e = min(cols_d - 1, int(xr / cell_w_d))
            r_s = max(0, int(yl / cell_h_d))
            r_e = min(rows_d - 1, int(yr / cell_h_d))
            for r in range(r_s, r_e + 1):
                for c in range(c_s, c_e + 1):
                    h_dem[r * cols_d + c] += hc
                    v_dem[r * cols_d + c] += vc

        # Initialize movable-net bboxes and add their demand
        for rn_idx in range(len(rn_hard)):
            rn_bbox[rn_idx] = rn_current_bbox(rn_idx)
            xl, xr, yl, yr = rn_bbox[rn_idx]
            dx = max(xr - xl, cell_w_d)
            dy = max(yr - yl, cell_h_d)
            hc = rn_wt[rn_idx] * cell_h_d / dy / h_cap
            vc = rn_wt[rn_idx] * cell_w_d / dx / v_cap
            c_s = max(0, int(xl / cell_w_d))
            c_e = min(cols_d - 1, int(xr / cell_w_d))
            r_s = max(0, int(yl / cell_h_d))
            r_e = min(rows_d - 1, int(yr / cell_h_d))
            for r in range(r_s, r_e + 1):
                for c in range(c_s, c_e + 1):
                    h_dem[r * cols_d + c] += hc
                    v_dem[r * cols_d + c] += vc

        # Initial costs and weights
        init_wl = sum(
            float(
                (
                    macro_hard_w[i]
                    * (
                        np.abs(pos[i, 0] - pos[macro_hard_j[i], 0])
                        + np.abs(pos[i, 1] - pos[macro_hard_j[i], 1])
                    )
                ).sum()
            )
            if len(macro_hard_j[i])
            else 0.0
            for i in range(n_hard)
        )
        init_sos = float((density_grid**2).sum())
        w_dens = (0.5 * init_wl / init_sos) if init_sos > 1e-12 else 0.0

        init_cong = float((h_dem**2).sum() + (v_dem**2).sum())
        w_cong = (0.5 * init_wl / init_cong) if init_cong > 1e-12 else 0.0

        current_wl = init_wl
        current_sos = init_sos
        current_cong = init_cong
        current_cost = current_wl + w_dens * current_sos + w_cong * current_cong
        best_pos_sa = pos.copy()
        best_cost_sa = current_cost

        T_start = max(cw, ch) * 0.10
        T_end = max(cw, ch) * 0.001

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac

            i = int(random.choice(movable_idx))
            ox, oy = pos[i, 0], pos[i, 1]
            j = None
            ojx = ojy = 0.0
            ch_dict = None

            r = random.random()

            if r < 0.45:
                # Shift
                sigma = T * (0.2 + 0.8 * (1 - frac))
                nx = np.clip(ox + random.gauss(0, sigma), half_w[i], cw - half_w[i])
                ny = np.clip(oy + random.gauss(0, sigma), half_h[i], ch - half_h[i])
                pos[i, 0] = nx
                pos[i, 1] = ny
                if has_overlap(i):
                    pos[i, 0] = ox
                    pos[i, 1] = oy
                    continue
                dw = delta_wl(i, ox, oy, nx, ny)
                ch_dict = dens_changes(ox, oy, nx, ny, sizes[i, 0], sizes[i, 1])

            elif r < 0.75:
                # Swap (prefer connected neighbors)
                if neighbors[i] and random.random() < 0.7:
                    cands = [nb for nb in neighbors[i] if movable[nb]]
                    j = (
                        int(random.choice(cands))
                        if cands
                        else int(random.choice(movable_idx))
                    )
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
                    pos[i, 0] = ox
                    pos[i, 1] = oy
                    pos[j, 0] = ojx
                    pos[j, 1] = ojy
                    continue
                dw = delta_wl(i, ox, oy, pos[i, 0], pos[i, 1]) + delta_wl(
                    j, ojx, ojy, pos[j, 0], pos[j, 1]
                )
                c1 = dens_changes(ox, oy, pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1])
                c2 = dens_changes(
                    ojx, ojy, pos[j, 0], pos[j, 1], sizes[j, 0], sizes[j, 1]
                )
                ch_dict = dict(c1)
                for cell, dc in c2.items():
                    ch_dict[cell] = ch_dict.get(cell, 0.0) + dc

            else:
                # Pull toward a connected neighbor
                if not neighbors[i]:
                    continue
                nb = int(random.choice(neighbors[i]))
                alpha_ = random.uniform(0.05, 0.35)
                nx = np.clip(
                    ox + alpha_ * (pos[nb, 0] - ox), half_w[i], cw - half_w[i]
                )
                ny = np.clip(
                    oy + alpha_ * (pos[nb, 1] - oy), half_h[i], ch - half_h[i]
                )
                pos[i, 0] = nx
                pos[i, 1] = ny
                if has_overlap(i):
                    pos[i, 0] = ox
                    pos[i, 1] = oy
                    continue
                dw = delta_wl(i, ox, oy, nx, ny)
                ch_dict = dens_changes(ox, oy, nx, ny, sizes[i, 0], sizes[i, 1])

            ds = delta_sos(ch_dict)

            # RUDY congestion delta: check affected routing nets for bbox changes
            aff_rn = set(macro_to_rn[i])
            if j is not None:
                aff_rn |= set(macro_to_rn[j])
            cong_ch = {}
            new_bboxes = {}
            for rn_idx in aff_rn:
                ob = rn_bbox[rn_idx]
                eps = 1e-6
                i_at_bnd = (
                    abs(ox - ob[0]) < eps or abs(ox - ob[1]) < eps
                    or abs(oy - ob[2]) < eps or abs(oy - ob[3]) < eps
                )
                j_at_bnd = j is not None and (
                    abs(ojx - ob[0]) < eps or abs(ojx - ob[1]) < eps
                    or abs(ojy - ob[2]) < eps or abs(ojy - ob[3]) < eps
                )
                if not (i_at_bnd or j_at_bnd):
                    new_bboxes[rn_idx] = ob
                    continue
                nb_bb = rn_current_bbox(rn_idx)
                new_bboxes[rn_idx] = nb_bb
                if nb_bb == ob:
                    continue
                for k, (dh, dv) in rudy_ch(*ob, *nb_bb, rn_wt[rn_idx]).items():
                    cdh, cdv = cong_ch.get(k, (0.0, 0.0))
                    cong_ch[k] = (cdh + dh, cdv + dv)
            dc = delta_cong_sos(cong_ch)

            delta = dw + w_dens * ds + w_cong * dc

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_wl += dw
                current_sos += ds
                current_cong += dc
                current_cost = current_wl + w_dens * current_sos + w_cong * current_cong
                apply_dens(ch_dict)
                apply_cong(cong_ch)
                for rn_idx, nb_bb in new_bboxes.items():
                    rn_bbox[rn_idx] = nb_bb
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pos_sa = pos.copy()
            else:
                pos[i, 0] = ox
                pos[i, 1] = oy
                if j is not None:
                    pos[j, 0] = ojx
                    pos[j, 1] = ojy

        return best_pos_sa

    # Fast proxy cost (used to compare candidates)

    def _fast_proxy(self, pos, benchmark, n_total, port_pos, nt=None):
        wl = self._fast_hpwl(pos, benchmark, n_total, port_pos, nt)
        density = self._fast_density(pos, benchmark)
        return wl + 0.5 * density

    def _fast_hpwl(self, pos, benchmark, n_total, port_pos, nt=None):
        norm = (benchmark.canvas_width + benchmark.canvas_height) * benchmark.num_nets

        # Use cached net data from _build_net_tensors when available (fast path)
        if nt is not None and len(nt["hpwl_weights"]) > 0:
            movable_idx = nt["movable_idx"]
            total = 0.0
            for mov_locals, fixed_xy, w in zip(
                nt["hpwl_movable_idx"], nt["hpwl_fixed_xy"], nt["hpwl_weights"]
            ):
                xs = [pos[movable_idx[l], 0] for l in mov_locals]
                ys = [pos[movable_idx[l], 1] for l in mov_locals]
                for fx, fy in fixed_xy:
                    xs.append(fx)
                    ys.append(fy)
                if len(xs) >= 2:
                    total += float(w) * (max(xs) - min(xs) + max(ys) - min(ys))
            return total / norm

        # Fallback: iterate all nets (slow for large benchmarks)
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
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        rows = benchmark.grid_rows
        cols = benchmark.grid_cols
        cell_w = cw / cols
        cell_h = ch / rows
        cell_area = cell_w * cell_h
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)

        density_grid = np.zeros(rows * cols, dtype=np.float64)

        for i in range(benchmark.num_macros):
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
