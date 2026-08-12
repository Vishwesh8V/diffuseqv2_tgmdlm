import os
import json
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Publication styling dictionary
PUBLICATION_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "FreeSerif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "figure.figsize": (4.0, 3.2),
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "legend.frameon": False,
}

def apply_publication_style():
    plt.rcParams.update(PUBLICATION_RCPARAMS)

def plot_single_position(
    r,
    logsnr_values,
    loss_values,
    is_solver_t,
    n_examples,
    solver_steps,
    out_dir,
    show_title=False
):
    """
    Plots Loss vs log-SNR for a single relative token position.
    Saves pos{r:03d}.png and pos{r:03d}.pdf in out_dir/per_position/
    """
    apply_publication_style()

    # Sort points by log-SNR ascending for a clean left-to-right curve
    sort_idx = np.argsort(logsnr_values)
    logsnr_sorted = logsnr_values[sort_idx]
    loss_sorted = loss_values[sort_idx]
    solver_sorted = is_solver_t[sort_idx]

    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    # Base curve
    ax.plot(
        logsnr_sorted,
        loss_sorted,
        "-",
        color="#1f77b4",
        lw=1.6,
        label="Loss curve",
        zorder=2,
    )

    # DPM-Solver step dots
    solver_mask = solver_sorted
    if np.any(solver_mask):
        ax.scatter(
            logsnr_sorted[solver_mask],
            loss_sorted[solver_mask],
            marker="o",
            s=42,
            facecolor="#d62728",
            edgecolor="black",
            linewidth=0.6,
            label=f"DPM-Solver steps (N={solver_steps})",
            zorder=3,
        )

    ax.set_xlabel("log SNR")
    ax.set_ylabel("Loss (per-token MSE)")
    if show_title:
        ax.set_title(f"Relative Token Position {r}")

    # Unobtrusive annotation of sample count
    ax.text(
        0.04,
        0.06,
        f"$n = {n_examples}$",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
        zorder=4,
    )

    ax.legend(loc="upper right")

    per_pos_dir = os.path.join(out_dir, "per_position")
    os.makedirs(per_pos_dir, exist_ok=True)

    png_path = os.path.join(per_pos_dir, f"pos{r:03d}.png")
    pdf_path = os.path.join(per_pos_dir, f"pos{r:03d}.pdf")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

def plot_overview_grid(
    t_grid,
    loss_r,
    logsnr_r,
    is_solver_t,
    n_r,
    min_examples,
    solver_steps,
    out_dir,
    max_positions=0,
    show_title=True
):
    """
    Plots a multi-panel overview grid of Loss vs log-SNR across all valid token positions.
    Saves overview_grid.png and overview_grid.pdf in out_dir/
    """
    apply_publication_style()

    R_max = loss_r.shape[1]
    valid_positions = [r for r in range(R_max) if n_r[r] >= min_examples]
    if max_positions > 0:
        valid_positions = valid_positions[:max_positions]

    n_plots = len(valid_positions)
    if n_plots == 0:
        print("No valid positions found for overview grid.")
        return

    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 2.8 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    h_curve, h_solver = None, None

    for idx, r in enumerate(valid_positions):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        logsnr_values = logsnr_r[:, r]
        loss_values = loss_r[:, r]

        sort_idx = np.argsort(logsnr_values)
        logsnr_sorted = logsnr_values[sort_idx]
        loss_sorted = loss_values[sort_idx]
        solver_sorted = is_solver_t[sort_idx]

        (line,) = ax.plot(
            logsnr_sorted,
            loss_sorted,
            "-",
            color="#1f77b4",
            lw=1.4,
            zorder=2,
        )
        if h_curve is None:
            h_curve = line

        solver_mask = solver_sorted
        if np.any(solver_mask):
            pts = ax.scatter(
                logsnr_sorted[solver_mask],
                loss_sorted[solver_mask],
                marker="o",
                s=28,
                facecolor="#d62728",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            if h_solver is None:
                h_solver = pts

        ax.set_title(f"Pos {r} ($n={n_r[r]}$)", fontsize=10)
        if row == nrows - 1:
            ax.set_xlabel("log SNR", fontsize=9)
        if col == 0:
            ax.set_ylabel("Loss (MSE)", fontsize=9)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    handles = []
    labels = []
    if h_curve is not None:
        handles.append(h_curve)
        labels.append("Loss curve")
    if h_solver is not None:
        handles.append(h_solver)
        labels.append(f"DPM-Solver steps (N={solver_steps})")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=10,
        )

    plt.tight_layout()
    png_path = os.path.join(out_dir, "overview_grid.png")
    pdf_path = os.path.join(out_dir, "overview_grid.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def save_raw_data(
    out_dir,
    t_grid,
    loss_r,
    logsnr_r,
    n_r,
    solver_t_indices,
    is_solver_t,
    max_cap_len,
    args_dict
):
    """
    Saves raw numeric curves/point data as curve_data.npz, curve_data.csv, and run_manifest.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save NPZ
    npz_path = os.path.join(out_dir, "curve_data.npz")
    np.savez(
        npz_path,
        t_grid=t_grid,
        loss_r=loss_r,
        logsnr_r=logsnr_r,
        n_r=n_r,
        solver_t_indices=np.array(solver_t_indices),
        is_solver_t=is_solver_t,
        max_cap_len=max_cap_len,
    )

    # 2. Save CSV (long format: position, t, logsnr, loss, n_examples, is_solver_step)
    csv_path = os.path.join(out_dir, "curve_data.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("position,t,logsnr,loss,n_examples,is_solver_step\n")
        T_len, R_max = loss_r.shape
        for r in range(R_max):
            n_ex = n_r[r]
            for t_idx in range(T_len):
                t_val = t_grid[t_idx]
                logsnr_val = logsnr_r[t_idx, r]
                loss_val = loss_r[t_idx, r]
                is_solv = 1 if is_solver_t[t_idx] else 0
                f.write(f"{r},{t_val},{logsnr_val:.6f},{loss_val:.6f},{n_ex},{is_solv}\n")

    # 3. Save run_manifest.json
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = "unknown"

    sanitized_args = {}
    for k, v in args_dict.items():
        if isinstance(v, np.ndarray):
            sanitized_args[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            sanitized_args[k] = v.item()
        elif isinstance(v, (set, tuple)):
            sanitized_args[k] = list(v)
        else:
            sanitized_args[k] = v

    manifest = {
        "git_commit": git_commit,
        "args": sanitized_args,
    }
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def plot_overall_sequence(
    t_grid,
    loss_r,
    logsnr_r,
    is_solver_t,
    n_r,
    solver_steps,
    out_dir,
    show_title=False
):
    """
    Plots overall (token-averaged across all caption positions) Loss vs log-SNR.
    Saves overall_sequence.png and overall_sequence.pdf in out_dir/
    """
    apply_publication_style()

    weights = n_r.astype(np.float64)
    total_w = np.sum(weights)
    if total_w == 0:
        return

    overall_loss = np.sum(loss_r * weights[None, :], axis=1) / total_w
    overall_logsnr = np.sum(logsnr_r * weights[None, :], axis=1) / total_w

    sort_idx = np.argsort(overall_logsnr)
    logsnr_sorted = overall_logsnr[sort_idx]
    loss_sorted = overall_loss[sort_idx]
    solver_sorted = is_solver_t[sort_idx]

    fig, ax = plt.subplots(figsize=(4.5, 3.4))

    ax.plot(
        logsnr_sorted,
        loss_sorted,
        "-",
        color="#1f77b4",
        lw=1.8,
        label="Sequence-averaged Loss curve",
        zorder=2,
    )

    solver_mask = solver_sorted
    if np.any(solver_mask):
        ax.scatter(
            logsnr_sorted[solver_mask],
            loss_sorted[solver_mask],
            marker="o",
            s=48,
            facecolor="#d62728",
            edgecolor="black",
            linewidth=0.6,
            label=f"DPM-Solver steps (N={solver_steps})",
            zorder=3,
        )

    ax.set_xlabel("log SNR")
    ax.set_ylabel("Overall Loss (MSE)")
    if show_title:
        ax.set_title("Overall Caption-Averaged Loss vs. log-SNR")

    ax.legend(loc="upper right")

    png_path = os.path.join(out_dir, "overall_sequence.png")
    pdf_path = os.path.join(out_dir, "overall_sequence.pdf")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
