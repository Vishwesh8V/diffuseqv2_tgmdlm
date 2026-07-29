"""
plot_alphas.py

Consumes the per-sentence `*_diagnostics.npz` files produced by sample_seq2seq.py
(--save_trajectories) and, for each sentence, plots:

  1. ||pred_xstart_t|| per caption token, across timesteps
  2. sqrt(alpha_cumprod_t) per caption token, across timesteps
  3. entropy_t per caption token, across timesteps

Usage:
    python plot_alphas.py --diag_dir generation_outputs/.../ema_....samples \
                           --out_dir plots/ \
                           --step_interval 100 \
                           --exclude_tokens 0 1 2
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def downsample_indices(num_steps, step_interval):
    idx = list(range(0, num_steps, step_interval))
    if idx[-1] != num_steps - 1:
        idx.append(num_steps - 1)
    return np.array(idx)


def timesteps_for_indices(idx, num_steps):
    return (num_steps - 1) - idx


def map_to_schedule_rows(idx, num_steps, num_timesteps):
    if num_steps == num_timesteps:
        return idx
    scale = (num_timesteps - 1) / max(num_steps - 1, 1)
    return np.round(idx * scale).astype(int)


def make_plot(x_vals, series, labels, title, ylabel, out_path):
    n = len(labels)
    cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ax.plot(
            x_vals,
            series[:, i],
            label=label,
            color=cmap(i / max(n - 1, 1)),
            linewidth=1.5,
        )
    ax.set_xlabel("timestep t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        ncol=max(1, n // 25),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_file(npz_path, out_dir, step_interval, exclude_tokens):
    data = np.load(npz_path, allow_pickle=True)

    pred_xstart_trajectory = data["pred_xstart_trajectory"]
    alpha_trajectory = data["alpha_trajectory"]
    entropy_trajectory = data["entropy_trajectory"]
    position_labels = [str(l) for l in data["position_labels"]]
    reference = str(data["reference"][0]) if "reference" in data else ""

    keep_mask = np.ones(len(position_labels), dtype=bool)
    for i in exclude_tokens:
        if 0 <= i < len(position_labels):
            keep_mask[i] = False

    position_labels = [
        lbl for lbl, keep in zip(position_labels, keep_mask) if keep
    ]

    num_steps = pred_xstart_trajectory.shape[0]
    num_timesteps = alpha_trajectory.shape[0]

    idx = downsample_indices(num_steps, step_interval)
    t_vals = timesteps_for_indices(idx, num_steps)
    schedule_rows = map_to_schedule_rows(idx, num_steps, num_timesteps)

    base = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    pred_xstart_norm = np.linalg.norm(pred_xstart_trajectory[idx], axis=-1)
    pred_xstart_norm = pred_xstart_norm[:, keep_mask]
    make_plot(
        t_vals,
        pred_xstart_norm,
        position_labels,
        title=f"||pred_xstart_t|| per token\n{reference}",
        ylabel="||pred_xstart_t||",
        out_path=os.path.join(out_dir, f"{base}_pred_xstart_norm.png"),
    )

    sqrt_alpha = np.sqrt(alpha_trajectory[schedule_rows])
    sqrt_alpha = sqrt_alpha[:, keep_mask]
    make_plot(
        t_vals,
        sqrt_alpha,
        position_labels,
        title=f"sqrt(alpha_cumprod_t) per token (adaptive schedule)\n{reference}",
        ylabel="sqrt(alpha_cumprod_t)",
        out_path=os.path.join(out_dir, f"{base}_sqrt_alpha.png"),
    )

    entropy_vals = entropy_trajectory[idx]
    entropy_vals = entropy_vals[:, keep_mask]
    make_plot(
        t_vals,
        entropy_vals,
        position_labels,
        title=f"predicted-token entropy per position\n{reference}",
        ylabel="entropy (nats)",
        out_path=os.path.join(out_dir, f"{base}_entropy.png"),
    )

    print(f"[{base}] saved 3 plots ({len(position_labels)} tokens, {len(idx)} timestep samples)")


def main():
    parser = argparse.ArgumentParser(description="Plot per-token diffusion diagnostics.")
    parser.add_argument(
        "--diag_dir",
        type=str,
        required=True,
        help="directory containing *_diagnostics.npz files (searched recursively)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="trajectory_plots",
        help="directory to write the plots into",
    )
    parser.add_argument(
        "--step_interval",
        type=int,
        default=100,
        help="plot one point every N diffusion steps (default 100)",
    )
    parser.add_argument(
        "--exclude_tokens",
        type=int,
        nargs="*",
        default=[],
        help="Token indices to exclude from all plots (0-based).",
    )

    args = parser.parse_args()

    if os.path.isfile(args.diag_dir):
        npz_files = [args.diag_dir]
    else:
        pattern = os.path.join(args.diag_dir, "**", "*_diagnostics.npz")
        npz_files = sorted(glob.glob(pattern, recursive=True))

    if not npz_files:
        print(f"No *_diagnostics.npz files found under {args.diag_dir}")
        return

    print(f"Found {len(npz_files)} sentence diagnostics files.")
    for npz_path in npz_files:
        process_file(
            npz_path,
            args.out_dir,
            args.step_interval,
            args.exclude_tokens,
        )


if __name__ == "__main__":
    main()
