"""
Generate a large batch of samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
from torch.cuda.amp import autocast

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)
from timing_utils import append_timing_row  # BENCHMARK: shared CSV row logger

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0.0, rejection_rate=0.0, note='none', time_schedule_path='', save_trajectories=False,
                     timing_csv='benchmark_timing.csv', repeat_idx=0,
                     K=20, J_path='', profile_batches=10, smooth_sigma=5.0, T_subset=0, solver_order=2)  # BENCHMARK: new CLI args
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False, start_n=0)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# FIX: this decorator was missing. Without it, every forward pass through
# base_model_fn/hooked_model_fn builds an autograd graph it never uses --
# pure overhead that has nothing to do with DPM-Solver's actual algorithmic
# speed, and would unfairly handicap it in a timing comparison against the
# normal script (which does have @th.no_grad()).
@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    
    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # Try to infer model_path from time_schedule_path if not provided
    if not args.model_path and args.time_schedule_path:
        import glob
        import re
        schedule_dir = os.path.dirname(args.time_schedule_path)
        if schedule_dir:
            # Look for step number in time_schedule_path, e.g. "step_120000"
            step_match = re.search(r'step_(\d+)', os.path.basename(args.time_schedule_path))
            step_suffix = f"_{step_match.group(1)}.pt" if step_match else ".pt"
            
            # Find all .pt files in schedule_dir
            pt_files = glob.glob(os.path.join(schedule_dir, "*.pt"))
            if pt_files:
                # Prioritize files matching step_suffix
                matching_files = [f for f in pt_files if f.endswith(step_suffix)]
                if matching_files:
                    args.model_path = matching_files[0]
                else:
                    # Fallback: prioritize ema checkpoints
                    ema_files = [f for f in pt_files if "ema" in os.path.basename(f)]
                    if ema_files:
                        args.model_path = ema_files[0]
                    else:
                        args.model_path = pt_files[0]
                logger.log(f"### Inferred model_path from time_schedule_path: {args.model_path}")

    # load configurations.
    config_dir = ""
    if args.model_path:
        config_dir = os.path.split(args.model_path)[0]
    elif args.time_schedule_path:
        config_dir = os.path.split(args.time_schedule_path)[0]

    config_path = os.path.join(config_dir, "training_args.json") if config_dir else "training_args.json"
    print(f"Loading training configuration from: {config_path}")

    # If the file still doesn't exist, search recursively in the workspace
    if not os.path.exists(config_path):
        import glob
        logger.log(f"### training_args.json not found at {config_path}. Searching recursively...")
        candidate_paths = glob.glob("**/training_args.json", recursive=True)
        candidate_paths = [p for p in candidate_paths if "venv" not in p and ".conda" not in p]
        if candidate_paths:
            config_path = candidate_paths[0]
            logger.log(f"### Found training_args.json at: {config_path}")
        else:
            raise FileNotFoundError(
                f"Could not find training_args.json. Checked: {config_path} and recursive search. "
                "Please make sure it exists in the workspace or specify a valid --model_path."
            )

    _cli_time_schedule_path = args.time_schedule_path
    _cli_save_trajectories = args.save_trajectories
    _cli_timing_csv = args.timing_csv          # BENCHMARK
    _cli_repeat_idx = args.repeat_idx          # BENCHMARK
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    # Restore inference-time overrides
    if _cli_time_schedule_path:
        args.time_schedule_path = _cli_time_schedule_path
    args.save_trajectories = _cli_save_trajectories
    args.timing_csv = _cli_timing_csv          # BENCHMARK
    args.repeat_idx = _cli_repeat_idx          # BENCHMARK

    # FIX: this whole block was a duplicate of the update() above, and it ran
    # AFTER the CLI-override restore -- silently re-overwriting
    # time_schedule_path/save_trajectories/timing_csv/repeat_idx right back
    # to the training-time config values and undoing the restore. Removed.
    # (was: training_args['batch_size'] = args.batch_size; args.__dict__.update(training_args))

    logger.log("### Creating model and diffusion...")
    args.device = dist_util.dev()
    # args.denoise_rate = 0.0
    print('#'*10, args.clamp_step)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    if args.model_path:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
    else:
        logger.log("### WARNING: model_path is empty. Skipping loading state dict (model will use random weights).")

    if args.time_schedule_path:
        logger.log(f"### Loading time schedule from {args.time_schedule_path}...")
        diffusion._load_time_schedule(args.time_schedule_path)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    # Move the embedding used during DPM sampling onto the GPU
    model_emb = model_emb.to(dist_util.dev())

    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    SOLVER_STEP = args.step

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}' if args.model_path else "random_weights.ema_0.9999_000000.pt"
    
    if '.ema' in model_base_name:
        model_name_prefix = model_base_name.split('.ema')[0]
        ema_suffix = model_base_name.split('.ema')[1]
    else:
        model_name_prefix = model_base_name
        ema_suffix = "_no_ema"

    out_dir = os.path.join(args.out_dir, f"{model_name_prefix}")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"ema{ema_suffix}.samples")
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f"seed{args.seed2}_solverstep{SOLVER_STEP}_{args.note}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    idx = 0
    try:
        while True:
            batch, cond = next(data_valid)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1
            # Add your if idx >= 500: break here if needed
    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(dist_util.dev())  # FIX: load_data_text received model_emb.cpu(), so it is still on CPU here;
                                    # move it back to GPU before denoised_fn_round uses it inside the DPM loop.

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data[args.start_n:])
    else:
        iterator = iter(all_test_data[args.start_n:])
    
    from tqdm import tqdm
    print('Start from ...', args.start_n)
    all_test_data = all_test_data[args.start_n:]

    from dpm_solver_token_adaptive import TokenAdaptiveDPM_Solver, build_token_timestep_matrix, token_model_wrapper

    if args.J_path and os.path.exists(args.J_path):
        J = np.load(args.J_path)
        print(f"Loaded J from {args.J_path}")
    else:
        print("Profiling to build J matrix...")
        data_profiler = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            deterministic=True,
            data_args=args,
            split=args.split,
            loaded_vocab=tokenizer,
            model_emb=model_emb.cpu(),
            loop=False
        )
        J = build_token_timestep_matrix(
            model=model, diffusion=diffusion,
            data_loader=data_profiler,
            K=args.K, device=dist_util.dev(),
            T_subset=args.T_subset or None,
            smooth_sigma=args.smooth_sigma,
            profile_batches=args.profile_batches,
        )
        save_path = args.J_path or "J_adaptive.npy"
        np.save(save_path, J)
        print(f"Saved J to {save_path}")

    J_tensor = th.from_numpy(J).long().to(dist_util.dev())

    # NEW: Lists to store our trajectories during the solver loop
    samples_traj = []
    pred_xstart_traj = []

    alphas_cumprod_t = th.from_numpy(diffusion.alphas_cumprod).float().to(dist_util.dev())
    model_fn = token_model_wrapper(model, alphas_cumprod_t, model_kwargs={})

    def rounding_corrector(x0, t=None):
        rounded_x0, _ = denoised_fn_round(args, model_emb, x0, t)
        return rounded_x0

    dpm_solver = TokenAdaptiveDPM_Solver(
        model_fn=model_fn,
        alphas_cumprod_2d=alphas_cumprod_t,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=rounding_corrector,
    )

    def sample_x_at_J0(x_start, t_ids_row0, diffusion, noise, device):
        ac = th.from_numpy(diffusion.alphas_cumprod).float().to(device)
        L = x_start.shape[1]
        ac_t = ac[t_ids_row0, th.arange(L, device=device)]
        alpha = th.sqrt(ac_t)[None, :, None]
        sigma = th.sqrt(1.0 - ac_t)[None, :, None]
        return alpha * x_start + sigma * noise


    sentence_counter = 0  # BENCHMARK: needed by the trajectory-saving block below (was referenced but never initialized)
    batch_counter = 0     # BENCHMARK: running count of real batches, for timing rows

    for cond in tqdm(all_test_data):

        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
            continue
                
        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = sample_x_at_J0(x_start, J_tensor[0], diffusion, noise, dist_util.dev())
        x_noised = th.where(input_ids_mask==0, x_start, x_noised)
        
        # Clear trajectories for the current batch
        samples_traj.clear()
        pred_xstart_traj.clear()        

        th.cuda.synchronize()
        t0 = time.time()
        with autocast():
            x_sample = dpm_solver.sample(
                x_noised,
                J=J_tensor,
                order=args.solver_order,
                x_start=x_start,
                input_ids_mask=input_ids_mask,
            )
        th.cuda.synchronize()
        elapsed = time.time() - t0

        if rank == 0:
            append_timing_row(args.timing_csv, {
                "method": "dpm",
                "steps": SOLVER_STEP,
                "repeat_idx": args.repeat_idx,
                "batch_idx": batch_counter,
                "num_sentences": x_start.shape[0],
                "elapsed_sec": elapsed,
                "sec_per_sentence": elapsed / x_start.shape[0],
                "checkpoint": os.path.basename(args.model_path),
                "seed": args.seed2,
                "note": args.note,
            })
        batch_counter += 1

        # print(x_sample[0].shape) # samples for each step [128, 128]

        sample = x_sample
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []


        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()
        # print('decoding for seq2seq', )
        # print(arr.shape)

        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices

        # Use the original simple len_x formula:
        # input_ids_mask_ori is 0 for source (SMILES) positions, 1 for target+pad.
        # sum(input_mask) = number of target+pad slots, so seq_len - sum = SMILES length.
        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
            print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
        fout.close()

        # ADDED 24/06/2026: NEW CHANGE TO GET HEATMAP/trajectories of alpha values for each token in input sequence
        # samples/pred_xstart_list are lists of length num_steps, each element shape (bsz, seq_len, hidden_dim).
        # NOTE: --bsz is commonly the whole test set in one batch (e.g. 50), so we loop over every
        # item in the batch here -- indexing only [0] would silently produce diagnostics for just
        # one sentence and skip the rest.

        if args.save_trajectories:
            bsz = input_ids_x.shape[0]
            for b in range(bsz):
                smiles_len = (input_ids_mask_ori[b] == 0).sum().item()  # source (SMILES) positions

                # Trim to the real (non-pad) caption length FIRST, so every array below
                # — token/entropy/zt trajectories, alpha/beta schedule, and labels — is
                # sliced to the exact same [smiles_len : smiles_len + caption_len] window.
                caption_ids = input_ids_x[b, smiles_len:].squeeze(-1).tolist()
                while len(caption_ids) > 0 and caption_ids[-1] == tokenizer.pad_token_id:
                    caption_ids.pop()
                caption_len = len(caption_ids)

                if tokenizer.tokenizer == 'dual' or isinstance(tokenizer.tokenizer, dict):
                    position_labels = [tokenizer.rev_tokenizer.get(x, '[UNK]') for x in caption_ids]
                else:
                    position_labels = tokenizer.tokenizer.convert_ids_to_tokens(caption_ids)

                cap_slice = slice(smiles_len, smiles_len + caption_len)

                # --- per-step denoising trajectory (needs the sampling loop) ---
                zt_trajectory = []
                token_trajectory = []
                entropy_trajectory = []
                pred_xstart_trajectory = [] 

                for zt, pred_x0 in zip(samples_traj, pred_xstart_traj):
                    zt_trajectory.append(zt[b, cap_slice].cpu().numpy())              # noisy embedding
                    pred_xstart_trajectory.append(pred_x0[b, cap_slice].cpu().numpy())
                    step_logits = model.get_logits(pred_x0[b:b+1])                    # logits from clean-x0 estimate
                    step_tokens = th.argmax(step_logits, dim=-1)
                    token_trajectory.append(step_tokens[0, cap_slice].cpu().numpy())

                    probs = th.softmax(step_logits[0, cap_slice], dim=-1)
                    ent = -(probs * th.log(probs + 1e-10)).sum(dim=-1)
                    entropy_trajectory.append(ent.cpu().numpy())

                token_trajectory = np.array(token_trajectory)      # (num_steps, caption_len)
                entropy_trajectory = np.array(entropy_trajectory)  # (num_steps, caption_len)

                ref_tokens = tokenizer.decode_token(input_ids_x[b, smiles_len:])

                # --- alpha / beta noise-schedule trajectory (static, from --time_schedule_path) ---
                # diffusion.alphas_cumprod has shape (num_timesteps, full_seq_len). The column
                # axis is ABSOLUTE position in source+caption (it broadcasts against
                # x_start_mean.shape == [bsz, full_seq_len, hidden]), so caption columns start
                # at `smiles_len`, not at 0 — that was the earlier bug.
                # Row 0 = t=0 (clean) ... row num_timesteps-1 = noisiest; flip to match the
                # noisiest-first -> clean-last order of `samples`/`pred_xstart_list`.
                alpha_traj = diffusion.alphas_cumprod[:, cap_slice].copy()   # (num_timesteps, caption_len)
                alpha_prev = np.vstack([alpha_traj[:1], alpha_traj[:-1]])
                beta_traj = 1.0 - (alpha_traj / alpha_prev)

                alpha_traj = alpha_traj[::-1]
                beta_traj = beta_traj[::-1]

                sent_idx = args.start_n + sentence_counter
                diag_path = out_path.replace('.json', f'_sent{sent_idx}_diagnostics.npz')
                np.savez(diag_path,
                    token_trajectory=token_trajectory,
                    entropy_trajectory=entropy_trajectory,
                    zt_trajectory=np.array(zt_trajectory),
                    alpha_trajectory=alpha_traj,
                    beta_trajectory=beta_traj,
                    pred_xstart_trajectory=np.array(pred_xstart_trajectory),
                    position_labels=np.array(position_labels),
                    smiles_len=smiles_len,
                    reference=np.array([ref_tokens])
                )
                print(f"Saved diagnostics to {diag_path}")
                sentence_counter += 1


    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')

if __name__ == "__main__":
    main()