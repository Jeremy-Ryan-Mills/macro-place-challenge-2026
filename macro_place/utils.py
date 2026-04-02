"""
Utility functions for placement validation and visualization.
"""

import sys

import torch
from typing import Tuple, List, Optional

from macro_place.benchmark import Benchmark


def validate_placement(
    placement: torch.Tensor, benchmark: Benchmark, check_overlaps: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate placement legality.

    Checks:
    - All macros within canvas bounds
    - No NaN/Inf values
    - Correct shape
    - Fixed macros at original positions
    - No macro overlaps (optional, can be slow for large designs)

    Args:
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object
        check_overlaps: If True, check for macro-to-macro overlaps (default: True)

    Returns:
        (is_valid, violations)
    """
    violations = []

    # Check shape
    if placement.shape != (benchmark.num_macros, 2):
        violations.append(
            f"Shape mismatch: expected {(benchmark.num_macros, 2)}, got {placement.shape}"
        )
        return False, violations

    # Check for NaN/Inf
    if torch.isnan(placement).any():
        violations.append("Placement contains NaN values")
    if torch.isinf(placement).any():
        violations.append("Placement contains Inf values")

    # Check bounds
    x_coords = placement[:, 0]
    y_coords = placement[:, 1]
    widths = benchmark.macro_sizes[:, 0]
    heights = benchmark.macro_sizes[:, 1]

    x_min = x_coords - widths / 2
    x_max = x_coords + widths / 2
    y_min = y_coords - heights / 2
    y_max = y_coords + heights / 2

    if (x_min < 0).any() or (x_max > benchmark.canvas_width).any():
        violations.append("Macros outside horizontal canvas bounds")
    if (y_min < 0).any() or (y_max > benchmark.canvas_height).any():
        violations.append("Macros outside vertical canvas bounds")

    # Check fixed macros
    fixed_mask = benchmark.macro_fixed
    if fixed_mask.any():
        original_pos = benchmark.macro_positions[fixed_mask]
        new_pos = placement[fixed_mask]
        if not torch.allclose(original_pos, new_pos, atol=1e-3):
            violations.append("Fixed macros have been moved")

    # Check overlaps among hard macros only (soft macros naturally overlap)
    if check_overlaps:
        overlap_count = 0
        num_hard = benchmark.num_hard_macros
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                # Get bounding boxes
                lx_i, ux_i = x_min[i].item(), x_max[i].item()
                ly_i, uy_i = y_min[i].item(), y_max[i].item()
                lx_j, ux_j = x_min[j].item(), x_max[j].item()
                ly_j, uy_j = y_min[j].item(), y_max[j].item()

                # Check if boxes overlap (NOT just touching)
                # No overlap if: one box is completely to the left, right, above, or below the other
                if not (lx_i >= ux_j or ux_i <= lx_j or ly_i >= uy_j or uy_i <= ly_j):
                    overlap_count += 1
                    if overlap_count <= 5:  # Only report first 5 to avoid spam
                        violations.append(f"Macros {i} and {j} overlap")

        if overlap_count > 5:
            violations.append(f"... and {overlap_count - 5} more overlaps")

    return len(violations) == 0, violations


def visualize_placement(
    placement: torch.Tensor,
    benchmark: Benchmark,
    save_path: Optional[str] = None,
    plc=None,
):
    """
    Visualize placement (requires matplotlib).

    Args:
        placement: [num_macros, 2] positions
        benchmark: Benchmark data
        save_path: Optional path to save figure
        plc: Optional PlacementCost object (enables net connectivity drawing)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print(
            "Error: matplotlib not installed. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw canvas
    ax.add_patch(
        Rectangle(
            (0, 0),
            benchmark.canvas_width,
            benchmark.canvas_height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )

    # Draw macros
    num_hard = getattr(benchmark, "num_hard_macros", benchmark.num_macros)
    for i in range(benchmark.num_macros):
        x, y = placement[i].tolist()
        w, h = benchmark.macro_sizes[i].tolist()

        x_min = x - w / 2
        y_min = y - h / 2

        is_soft = i >= num_hard
        color = (
            "red"
            if benchmark.macro_fixed[i]
            else "lightsteelblue"
            if is_soft
            else "blue"
        )
        alpha = 0.25 if is_soft else 0.5
        linestyle = "dashed" if is_soft else "solid"

        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                w,
                h,
                fill=True,
                facecolor=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
                linestyle=linestyle,
            )
        )

    # Draw hard macro pins as small circles at macro_center + offset
    if benchmark.macro_pin_offsets:
        all_pin_x = []
        all_pin_y = []
        for i, offsets in enumerate(benchmark.macro_pin_offsets):
            if offsets.shape[0] == 0:
                continue
            cx, cy = placement[i].tolist()
            all_pin_x.extend((cx + offsets[:, 0]).tolist())
            all_pin_y.extend((cy + offsets[:, 1]).tolist())
        if all_pin_x:
            ax.scatter(
                all_pin_x,
                all_pin_y,
                s=3,
                c="darkslateblue",
                zorder=6,
            )

    # Draw I/O pins as small circles
    if benchmark.port_positions.shape[0] > 0:
        pin_x = benchmark.port_positions[:, 0].tolist()
        pin_y = benchmark.port_positions[:, 1].tolist()
        ax.scatter(
            pin_x,
            pin_y,
            s=8,
            c="green",
            zorder=5,
            edgecolors="darkgreen",
            linewidths=0.3,
        )

    # Draw net connections as star topology (average center → each pin)
    if plc is not None and hasattr(plc, "nets"):
        from matplotlib.collections import LineCollection

        lines = []
        for driver_name, sink_names in plc.nets.items():
            if driver_name not in plc.mod_name_to_indices:
                continue
            coords = []
            driver_idx = plc.mod_name_to_indices[driver_name]
            dx, dy = plc.modules_w_pins[driver_idx].get_pos()
            coords.append((dx, dy))
            for sink_name in sink_names:
                if sink_name not in plc.mod_name_to_indices:
                    continue
                sink_idx = plc.mod_name_to_indices[sink_name]
                sx, sy = plc.modules_w_pins[sink_idx].get_pos()
                coords.append((sx, sy))
            if len(coords) < 2:
                continue
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            for cx, cy in coords:
                lines.append([(avg_x, avg_y), (cx, cy)])
        if lines:
            lc = LineCollection(
                lines, colors="gray", alpha=0.05, linewidths=0.5, zorder=1
            )
            ax.add_collection(lc)

    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_title(f"Placement: {benchmark.name}")

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor="blue", alpha=0.5, edgecolor="black", label="Hard macros"),
        Patch(
            facecolor="lightsteelblue",
            alpha=0.1,
            edgecolor="black",
            linestyle="dashed",
            label="Soft macros",
        ),
        Patch(facecolor="red", alpha=0.3, edgecolor="black", label="Fixed macros"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markeredgecolor="darkorange",
            markersize=5,
            label="Macro pins",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="darkgreen",
            markersize=6,
            label="I/O pins",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
