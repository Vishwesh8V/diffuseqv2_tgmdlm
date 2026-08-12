"""
Evaluate Loss vs. log-SNR curves per caption-token position with DPM-Solver step overlay.
"""

import os
import sys
import json
import time
import glob
import re
import argparse
import numpy as np
import torch as th
from transformers import set_seed

# Ensure repo root and scripts directory are on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffuseq.text_datasets import load_data_text
from diffuseq.gaussian_diffusion import _extract_into_tensor
from diffuseq.utils import dist_util, logger
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer,
)
from dpm_solver_pytorch import NoiseScheduleVP, DPM_Solver
from loss_logsnr_plotting import plot_single_position, plot_overview_grid, plot_overall_sequence, save_raw_data


def create_argparser():
    defaults = dict(
        model_path="",
        time_schedule_path="",
        split="test",
        batch_size=64,
        num_t_grid=50,
        solver_steps=20,
        solver_order=2,
        solver_skip_type="time_uniform",
        n_noise_seeds=3,
        min_examples_per_position=5,
        max_positions=0,
        out_dir="loss_logsnr_plots",
        seed=102,
        titles=False,
        exclude_eos=False,
    )
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser(
        description="Evaluate Loss vs log-SNR per caption token position with DPM-Solver step overlay."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # 1. Infer model_path from time_schedule_path if not explicitly provided
    if not args.model_path and args.time_schedule_path:
        schedule_dir = os.path.dirname(args.time_schedule_path)
        if schedule_dir:
            step_match = re.search(r"step_(\d+)", os.path.basename(args.time_schedule_path))
            step_suffix = f"_{step_match.group(1)}.pt" if step_match else ".pt"

            pt_files = glob.glob(os.path.join(schedule_dir, "*.pt"))
            if pt_files:
                matching_files = [f for f in pt_files if f.endswith(step_suffix)]
                if matching_files:
                    args.model_path = matching_files[0]
                else:
                    ema_files = [f for f in pt_files if "ema" in os.path.basename(f)]
                    if ema_files:
                        args.model_path = ema_files[0]
                    else:
                        args.model_path = pt_files[0]
                logger.log(f"### Inferred model_path from time_schedule_path: {args.model_path}")

    # 2. Load training_args.json for model architecture details
    config_dir = ""
    if args.model_path:
        config_dir = os.path.split(args.model_path)[0]
    elif args.time_schedule_path:
        config_dir = os.path.split(args.time_schedule_path)[0]

    config_path = os.path.join(config_dir, "training_args.json") if config_dir else "training_args.json"

    if not os.path.exists(config_path):
        logger.log(f"### training_args.json not found at {config_path}. Searching recursively...")
        candidate_paths = glob.glob("**/training_args.json", recursive=True)
        candidate_paths = [p for p in candidate_paths if "venv" not in p and ".conda" not in p]
        if candidate_paths:
            config_path = candidate_paths[0]
            logger.log(f"### Found training_args.json at: {config_path}")
        else:
            raise FileNotFoundError(f"Could not find training_args.json. Checked: {config_path}")

    _cli_overrides = {
        "model_path": args.model_path,
        "time_schedule_path": args.time_schedule_path,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_t_grid": args.num_t_grid,
        "solver_steps": args.solver_steps,
        "solver_order": args.solver_order,
        "solver_skip_type": args.solver_skip_type,
        "n_noise_seeds": args.n_noise_seeds,
        "min_examples_per_position": args.min_examples_per_position,
        "max_positions": args.max_positions,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "titles": args.titles,
        "exclude_eos": args.exclude_eos,
    }

    with open(config_path, "rb") as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)

    # Restore CLI overrides so training_args doesn't overwrite CLI inputs
    for k, v in _cli_overrides.items():
        if v is not None and ((isinstance(v, str) and v != "") or not isinstance(v, str)):
            setattr(args, k, v)

    # 3. Create Model & Diffusion
    device = dist_util.dev()
    args.device = device

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    if args.model_path:
        logger.log(f"### Loading model state dict from {args.model_path}...")
        model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    else:
        logger.log("### WARNING: model_path is empty. Using random weights.")

    if args.time_schedule_path:
        logger.log(f"### Loading time schedule from {args.time_schedule_path}...")
        diffusion._load_time_schedule(args.time_schedule_path)

    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)
    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb = model_emb.to(device)

    # 4. Load Dataset
    set_seed(args.seed)
    logger.log(f"### Loading dataset split: {args.split}")
    data_test = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),
        loop=False,
    )

    all_input_ids = []
    all_input_mask = []
    for batch, cond in data_test:
        all_input_ids.append(cond["input_ids"])
        all_input_mask.append(cond["input_mask"])

    input_ids_full = th.cat(all_input_ids, dim=0)   # (M, seq_len)
    input_mask_full = th.cat(all_input_mask, dim=0) # (M, seq_len)
    M = input_ids_full.shape[0]
    logger.log(f"### Loaded {M} total test examples from split '{args.split}'")

    # 5. Caption Bookkeeping
    mask_bool = (input_mask_full == 1)
    cap_start = mask_bool.float().argmax(dim=1).long()  # first 1 index per example
    cap_len = input_mask_full.sum(dim=1).long()          # total 1s in input_mask

    if args.exclude_eos:
        cap_len = th.clamp(cap_len - 1, min=0)

    valid_ex = (cap_len > 0)
    if not valid_ex.all():
        logger.log(f"### Excluded {(~valid_ex).sum().item()} examples with 0 caption tokens.")
        input_ids_full = input_ids_full[valid_ex]
        input_mask_full = input_mask_full[valid_ex]
        cap_start = cap_start[valid_ex]
        cap_len = cap_len[valid_ex]
        M = input_ids_full.shape[0]

    max_cap_len = int(cap_len.max().item())
    if args.max_positions > 0:
        max_cap_len = min(max_cap_len, args.max_positions)

    n_r = np.zeros(max_cap_len, dtype=int)
    for r in range(max_cap_len):
        n_r[r] = (cap_len > r).sum().item()

    logger.log(f"### Max caption position depth: {max_cap_len}")
    logger.log(f"### Sample counts per position r=0..{min(5, max_cap_len-1)}: {n_r[:min(6, max_cap_len)]}")

    # 6. Extract DPM-Solver actual discrete timesteps
    betas_for_solver = diffusion.betas
    if betas_for_solver.ndim == 2:
        logger.log(f"### Averaging 2D betas {betas_for_solver.shape} across tokens for DPM-Solver schedule")
        betas_for_solver = betas_for_solver.mean(axis=1)

    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=th.from_numpy(betas_for_solver))
    dummy_solver = DPM_Solver(lambda *a, **k: None, noise_schedule, algorithm_type="dpmsolver++")
    t_T, t_0 = noise_schedule.T, 1.0 / noise_schedule.total_N
    t_continuous = dummy_solver.get_time_steps(
        skip_type=args.solver_skip_type, t_T=t_T, t_0=t_0, N=args.solver_steps, device="cpu"
    )
    N = noise_schedule.total_N
    solver_t_indices = th.clamp(th.round(t_continuous * N).long() - 1, 0, diffusion.num_timesteps - 1)
    solver_t_indices = sorted(set(solver_t_indices.tolist()))
    logger.log(f"### Extracted DPM-Solver discrete timesteps ({len(solver_t_indices)} unique points): {solver_t_indices}")

    # 7. Construct Union Timestep Grid
    base_grid = np.unique(np.round(np.linspace(0, diffusion.num_timesteps - 1, args.num_t_grid)).astype(int))
    t_grid = [int(x) for x in np.unique(np.concatenate([base_grid, np.array(solver_t_indices)]))]
    is_solver_t = np.isin(t_grid, solver_t_indices)
    T_len = len(t_grid)
    logger.log(f"### Total timesteps in evaluation grid: {T_len}")

    # 8. Compute log-SNR for each (t, r)
    logsnr_r = np.zeros((T_len, max_cap_len), dtype=np.float64)
    alphas_cp = np.asarray(diffusion.alphas_cumprod)  # (num_timesteps, seq_len) or (num_timesteps,)

    for t_idx, t_val in enumerate(t_grid):
        t_val = int(t_val)
        for r in range(max_cap_len):
            valid_mask = (cap_len > r)
            if not valid_mask.any():
                continue
            cols = (cap_start + r)[valid_mask].cpu().numpy()
            if alphas_cp.ndim == 2:
                ac = alphas_cp[t_val, cols]
            else:
                ac = np.full_like(cols, alphas_cp[t_val], dtype=np.float64)
            ac = np.clip(ac, 1e-12, 1.0 - 1e-12)
            logsnr_vals = np.log(ac) - np.log(1.0 - ac)
            logsnr_r[t_idx, r] = np.mean(logsnr_vals)

    # 9. Compute per-timestep loss across test set
    loss_r = np.zeros((T_len, max_cap_len), dtype=np.float64)
    logger.log(f"### Starting loss evaluation across {T_len} timesteps...")
    start_eval_time = time.time()

    for t_idx, t_val in enumerate(t_grid):
        t_val = int(t_val)
        t_start_time = time.time()

        loss_r_sum = np.zeros(max_cap_len, dtype=np.float64)
        loss_r_count = np.zeros(max_cap_len, dtype=np.int64)

        for micro_start in range(0, M, args.batch_size):
            micro_end = min(micro_start + args.batch_size, M)
            ids_batch = input_ids_full[micro_start:micro_end].to(device)
            mask_batch = input_mask_full[micro_start:micro_end].to(device)
            c_start_batch = cap_start[micro_start:micro_end].to(device)
            c_len_batch = cap_len[micro_start:micro_end].to(device)
            bs_curr = ids_batch.shape[0]

            with th.no_grad():
                x_start_mean = model.get_embeds(ids_batch)

            std = _extract_into_tensor(
                diffusion.sqrt_one_minus_alphas_cumprod,
                th.tensor([0], device=device),
                x_start_mean.shape,
            )

            for seed_i in range(args.n_noise_seeds):
                seed_val = int(args.seed * 10000 + seed_i * 100 + t_val)
                g = th.Generator(device=device).manual_seed(seed_val)
                x_start = diffusion._get_x_start(x_start_mean, std)
                noise = th.randn(x_start.shape, generator=g, device=device)
                t_batch = th.full((bs_curr,), t_val, device=device, dtype=th.long)

                x_t = diffusion.q_sample(
                    x_start, t_batch, noise=noise, mask=mask_batch, mean_embed=model.mean_embed
                )

                with th.no_grad():
                    model_output = model(x_t, diffusion._scale_timesteps(t_batch))

                    if t_val == 0:
                        model_out_x_start = diffusion._x0_helper(model_output, x_t, t_batch)["pred_xstart"]
                        mse_per_token = ((x_start_mean - model_out_x_start) ** 2).mean(dim=-1)
                    else:
                        mse_per_token = ((x_start - model_output) ** 2).mean(dim=-1)

                for r in range(max_cap_len):
                    valid_mask = (c_len_batch > r)
                    if not valid_mask.any():
                        continue
                    valid_indices = th.where(valid_mask)[0]
                    target_cols = c_start_batch[valid_indices] + r
                    losses_r_batch = mse_per_token[valid_indices, target_cols]
                    loss_r_sum[r] += losses_r_batch.sum().item()
                    loss_r_count[r] += valid_indices.numel()

        for r in range(max_cap_len):
            if loss_r_count[r] > 0:
                loss_r[t_idx, r] = loss_r_sum[r] / loss_r_count[r]

        elapsed_t = time.time() - t_start_time
        if t_idx == 0:
            est_total = elapsed_t * T_len
            logger.log(f"### Timestep 1/{T_len} (t={t_val}) done in {elapsed_t:.2f}s. Est total time: {est_total/60:.2f} min.")
        elif (t_idx + 1) % 10 == 0 or (t_idx + 1) == T_len:
            elapsed_total = time.time() - start_eval_time
            logger.log(f"### Progress: {t_idx+1}/{T_len} timesteps evaluated ({elapsed_total/60:.2f} min elapsed)")

    # 10. Save Data & Plot Figures
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    save_raw_data(
        out_dir=out_dir,
        t_grid=t_grid,
        loss_r=loss_r,
        logsnr_r=logsnr_r,
        n_r=n_r,
        solver_t_indices=solver_t_indices,
        is_solver_t=is_solver_t,
        max_cap_len=max_cap_len,
        args_dict=args.__dict__,
    )

    for r in range(max_cap_len):
        if n_r[r] >= args.min_examples_per_position:
            plot_single_position(
                r=r,
                logsnr_values=logsnr_r[:, r],
                loss_values=loss_r[:, r],
                is_solver_t=is_solver_t,
                n_examples=n_r[r],
                solver_steps=args.solver_steps,
                out_dir=out_dir,
                show_title=args.titles,
            )

    plot_overview_grid(
        t_grid=t_grid,
        loss_r=loss_r,
        logsnr_r=logsnr_r,
        is_solver_t=is_solver_t,
        n_r=n_r,
        min_examples=args.min_examples_per_position,
        solver_steps=args.solver_steps,
        out_dir=out_dir,
        max_positions=args.max_positions,
        show_title=args.titles,
    )

    plot_overall_sequence(
        t_grid=t_grid,
        loss_r=loss_r,
        logsnr_r=logsnr_r,
        is_solver_t=is_solver_t,
        n_r=n_r,
        solver_steps=args.solver_steps,
        out_dir=out_dir,
        show_title=args.titles,
    )

    logger.log(f"### Completed successfully! All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
