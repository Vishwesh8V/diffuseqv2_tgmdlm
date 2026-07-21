import numpy as np
import matplotlib.pyplot as plt

def plot_inference_diagnostics(npz_path, tokenizer):
    data = np.load(npz_path, allow_pickle=True)
    token_traj = data['token_trajectory']   # (num_steps, caption_len)
    entropy_traj = data['entropy_trajectory']
    reference = str(data['reference'][0])
    
    num_steps, caption_len = token_traj.shape
    steps = np.arange(num_steps)  # 0 = most noisy (t=2000), -1 = clean (t=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # --- Plot 1: Entropy per token per step ---
    im = ax1.imshow(entropy_traj.T, aspect='auto', cmap='viridis',
                    origin='upper', interpolation='nearest')
    ax1.set_xlabel("Denoising Step (0=noisy → N=clean)")
    ax1.set_ylabel("Caption Token Position")
    ax1.set_title(f"Per-token Entropy During Inference\n{reference[:80]}")
    plt.colorbar(im, ax=ax1, label='Entropy')

    # --- Plot 2: When does each token stabilize? ---
    # Find first step where token prediction stops changing
    stabilize_step = np.full(caption_len, num_steps - 1)
    for pos in range(caption_len):
        for step in range(num_steps - 2, -1, -1):
            if token_traj[step, pos] != token_traj[-1, pos]:
                stabilize_step[pos] = step + 1
                break

    ax2.bar(range(caption_len), stabilize_step, color='steelblue', alpha=0.7)
    ax2.set_xlabel("Caption Token Position")
    ax2.set_ylabel("Step at which token stabilizes")
    ax2.set_title("Token Stabilization Step (higher = stabilizes later = harder to predict)")
    ax2.axhline(y=num_steps * 0.5, color='red', linestyle='--', alpha=0.5, label='50% mark')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(npz_path.replace('.npz', '_plot.png'))
    print(f"Saved diagnostic plot")


def plot_schedule_trajectories(npz_path):
    """
    Plot the (adaptive, per-token) alpha_cumprod and beta noise-schedule trajectories
    for one test sentence, as saved by sample_seq2seq.py when run with --save_trajectories.

    This is a DIFFERENT diagnostic from plot_inference_diagnostics above: it visualizes
    the static noise schedule itself (loaded via --time_schedule_path), not how the
    model's predictions evolve during sampling.
    """
    data = np.load(npz_path, allow_pickle=True)
    alpha_traj = data['alpha_trajectory']   # (num_timesteps, caption_len), row0=noisiest(t=T) -> row-1=clean(t=0)
    beta_traj = data['beta_trajectory']     # same shape/orientation
    position_labels = [str(x) for x in data['position_labels']]
    reference = str(data['reference'][0])

    num_steps, caption_len = alpha_traj.shape

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, max(6, 0.35 * caption_len) * 2))

    for ax, traj, name, cmap in [(ax1, alpha_traj, 'alpha_cumprod', 'viridis'),
                                  (ax2, beta_traj, 'beta', 'magma')]:
        im = ax.imshow(traj.T, aspect='auto', cmap=cmap, origin='upper', interpolation='nearest')
        ax.set_xlabel("Diffusion Timestep (0=noisiest, t=T  →  N=clean, t=0)")
        ax.set_ylabel("Caption Token")
        ax.set_yticks(range(caption_len))
        ax.set_yticklabels(position_labels, fontsize=8)
        ax.set_title(f"Per-token {name} schedule\n{reference[:80]}")
        plt.colorbar(im, ax=ax, label=name)

    plt.tight_layout()
    out_path = npz_path.replace('.npz', '_schedule_plot.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved schedule trajectory plot to {out_path}")


def plot_word_entropy_trajectories(npz_path, word=None):
    """
    Plot the per-word entropy trajectory for ONE test sentence, as saved by
    sample_seq2seq.py when run with --save_trajectories (one .npz per sentence).

    - word=None: plots every word in the sentence as its own curve. This is
      the main "entropy trajectory of each word in the sentence" view.
    - word="Hydrogen": isolates just that word's occurrence(s) in THIS
      sentence (case-insensitive match against the saved position_labels),
      with no comparison against other sentences or words.

    Since decode already writes one .npz per sentence regardless of how many
    lines are in the test file, "picking one sentence" just means pointing
    this function at that sentence's file.
    """
    data = np.load(npz_path, allow_pickle=True)
    entropy_traj = data['entropy_trajectory']   # (num_steps, caption_len), 0=noisiest(t=T) -> -1=clean(t=0)
    position_labels = [str(x) for x in data['position_labels']]
    reference = str(data['reference'][0])
    num_steps, caption_len = entropy_traj.shape
    steps = np.arange(num_steps)

    if word is not None:
        matches = [k for k, lbl in enumerate(position_labels) if lbl.lower() == word.lower()]
        if not matches:
            print(f"'{word}' not found among this sentence's token labels: {position_labels}")
            print("(if your tokenizer splits words into subword pieces, e.g. 'Hydro' + '##gen', "
                  "a whole-word match may not exist as a single position)")
            return
        indices, title = matches, f"Entropy trajectory for '{word}'"
    else:
        indices, title = list(range(caption_len)), "Entropy trajectory for every word in the sentence"

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('tab20' if len(indices) > 10 else 'tab10')
    for i, k in enumerate(indices):
        label = position_labels[k] if word is None else f"{position_labels[k]} (pos {k})"
        ax.plot(steps, entropy_traj[:, k], label=label, color=cmap(i % cmap.N), linewidth=1.8)

    ax.set_xlabel("Denoising Step (0=noisiest, t=T  →  N=clean, t=0)")
    ax.set_ylabel("Entropy")
    ax.set_title(f"{title}\n{reference[:80]}")
    ax.legend(fontsize=8, ncol=2 if len(indices) > 8 else 1, loc='best')
    plt.tight_layout()

    suffix = f"_{word}" if word else "_allwords"
    out_path = npz_path.replace('.npz', f'_entropy{suffix}.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved word-entropy plot to {out_path}")


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, default=None,
                         help="path to ONE '*_sentN_diagnostics.npz' file. Required for --kind word_entropy; "
                              "also works as a single-file alternative to --npz_glob for schedule/inference.")
    parser.add_argument('--npz_glob', type=str, default=None,
                         help="glob pattern matching one or more '*_diagnostics.npz' files, "
                              "e.g. 'generation_outputs/.../*_diagnostics.npz' (batch mode, schedule/inference only)")
    parser.add_argument('--kind', type=str, default='word_entropy',
                         choices=['schedule', 'inference', 'word_entropy', 'both'],
                         help="'schedule' = alpha/beta heatmap, 'inference' = token/entropy heatmap + "
                              "stabilization bar chart, 'word_entropy' = per-word entropy line plot for ONE sentence")
    parser.add_argument('--word', type=str, default=None,
                         help="only for --kind word_entropy: isolate this word's occurrence(s) in the "
                              "sentence instead of plotting every word")
    args = parser.parse_args()

    if args.kind == 'word_entropy':
        if not args.npz_path:
            raise SystemExit("--kind word_entropy needs --npz_path pointing at one sentence's _sentN_diagnostics.npz")
        plot_word_entropy_trajectories(args.npz_path, word=args.word)
    else:
        npz_paths = sorted(glob.glob(args.npz_glob)) if args.npz_glob else ([args.npz_path] if args.npz_path else [])
        if not npz_paths:
            print("Nothing to plot -- pass --npz_glob (batch) or --npz_path (single file)")
        for path in npz_paths:
            if args.kind in ('schedule', 'both'):
                plot_schedule_trajectories(path)
            if args.kind in ('inference', 'both'):
                # plot_inference_diagnostics needs a tokenizer only for its signature;
                # it doesn't actually decode anything further, the reference string is
                # already saved in the npz, so this is safe to call with None.
                plot_inference_diagnostics(path, tokenizer=None)