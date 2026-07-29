import os
"""
analyze_benchmark.py

Reads the timing CSV produced by run_benchmark.py and produces:
  - <out_prefix>_summary.csv : mean/std sec-per-sentence and speedup-vs-normal,
    per method/step-count (also printed as a table)
  - <out_prefix>_bar.png     : average sec/sentence, normal vs each DPM step count
  - <out_prefix>_line.png    : sec/sentence vs DPM step count (log-x), with the
    normal baseline as a horizontal reference line, annotated with speedup
    multipliers at each point

repeat_idx == 0 (the warmup run) is discarded before any averaging.

Usage:
    python analyze_benchmark.py --timing_csv benchmark_timing.csv --out_prefix benchmark
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_and_aggregate(csv_path):
    # First try exactly what the user passed.
    if not os.path.isfile(csv_path):
        # Then try one directory above this script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback = os.path.join(script_dir, "..", csv_path)

        if os.path.isfile(fallback):
            csv_path = fallback
        else:
            raise FileNotFoundError(
                f"Could not find timing CSV.\n"
                f"Tried:\n"
                f"  {os.path.abspath(csv_path)}\n"
                f"  {os.path.abspath(fallback)}"
            )
    df = pd.read_csv(csv_path)
    df = df[df["repeat_idx"] != 0].copy()  # drop warmup runs

    # collapse multiple batches within a single run (repeat) into one row per run,
    # since a "run" = one full pass over the (possibly multi-batch) test split
    per_run = (
        df.groupby(["method", "steps", "repeat_idx"])
        .agg(elapsed_sec=("elapsed_sec", "sum"), num_sentences=("num_sentences", "sum"))
        .reset_index()
    )
    per_run["sec_per_sentence"] = per_run["elapsed_sec"] / per_run["num_sentences"]

    summary = (
        per_run.groupby(["method", "steps"])["sec_per_sentence"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_sec_per_sentence", "std": "std_sec_per_sentence", "count": "n_repeats"})
    )
    return summary


def add_speedup(summary):
    normal_rows = summary[summary["method"] == "normal"]
    if normal_rows.empty:
        raise ValueError("No 'normal' baseline rows found in the timing CSV -- did the normal-inference run complete?")
    baseline = normal_rows["mean_sec_per_sentence"].iloc[0]
    summary = summary.copy()
    summary["speedup_vs_normal"] = baseline / summary["mean_sec_per_sentence"]
    return summary, baseline


def print_table(summary):
    cols = ["method", "steps", "mean_sec_per_sentence", "std_sec_per_sentence", "n_repeats", "speedup_vs_normal"]
    print(summary[cols].sort_values(["method", "steps"]).to_string(index=False))


def make_bar_chart(summary, out_path):
    plot_df = summary.sort_values(["method", "steps"])
    labels = [
        "normal\n(2000 steps)" if m == "normal" else f"dpm\n({s} steps)"
        for m, s in zip(plot_df["method"], plot_df["steps"])
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 5))
    ax.bar(labels, plot_df["mean_sec_per_sentence"], yerr=plot_df["std_sec_per_sentence"], capsize=4)
    ax.set_ylabel("seconds / sentence")
    ax.set_title("Inference time per sentence: normal vs. DPM-Solver")
    for i, (v, sp) in enumerate(zip(plot_df["mean_sec_per_sentence"], plot_df["speedup_vs_normal"])):
        ax.text(i, v, f"{sp:.1f}x", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_line_chart(summary, baseline, out_path):
    dpm = summary[summary["method"] == "dpm"].sort_values("steps")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(dpm["steps"], dpm["mean_sec_per_sentence"], yerr=dpm["std_sec_per_sentence"],
                marker="o", capsize=4, label="DPM-Solver")
    ax.axhline(baseline, color="gray", linestyle="--", label="normal (2000 steps)")
    for x, y, sp in zip(dpm["steps"], dpm["mean_sec_per_sentence"], dpm["speedup_vs_normal"]):
        ax.annotate(f"{sp:.1f}x", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("DPM-Solver steps (log scale)")
    ax.set_ylabel("seconds / sentence")
    ax.set_title("DPM-Solver speed vs. step count, relative to normal inference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timing_csv", type=str, default="benchmark_timing.csv")
    p.add_argument("--out_prefix", type=str, default="benchmark")
    args = p.parse_args()

    summary = load_and_aggregate(args.timing_csv)
    summary, baseline = add_speedup(summary)

    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print_table(summary)

    make_bar_chart(summary, f"{args.out_prefix}_bar.png")
    make_line_chart(summary, baseline, f"{args.out_prefix}_line.png")

    print(f"\nSaved: {args.out_prefix}_summary.csv, {args.out_prefix}_bar.png, {args.out_prefix}_line.png")


if __name__ == "__main__":
    main()
