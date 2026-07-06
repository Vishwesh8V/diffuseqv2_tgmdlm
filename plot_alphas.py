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