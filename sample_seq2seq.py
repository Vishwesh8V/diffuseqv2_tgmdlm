"""
Generate a large batch of image samples from a model and save them as a large
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
    #ADDED top_p from 0 to 0.0
    defaults = dict(model_path='', step=0, out_dir='', top_p=0.0, rejection_rate=0.0, note='none', time_schedule_path='', save_trajectories=False,
                     timing_csv='benchmark_timing.csv', repeat_idx=0)  # BENCHMARK: new CLI args
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False, start_n=0)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



@th.no_grad()
def main():
    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    # Save inference-time CLI args BEFORE training_args.json overwrites them.
    # args.__dict__.update(training_args) would silently replace --time_schedule_path
    # with the value stored at training time (often empty or a stale path).
    _cli_time_schedule_path = args.time_schedule_path
    _cli_save_trajectories = args.save_trajectories
    _cli_timing_csv = args.timing_csv          # BENCHMARK
    _cli_repeat_idx = args.repeat_idx          # BENCHMARK

    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    # Restore inference-time overrides so the CLI wins over the training config.
    if _cli_time_schedule_path:
        args.time_schedule_path = _cli_time_schedule_path
    args.save_trajectories = _cli_save_trajectories
    args.timing_csv = _cli_timing_csv          # BENCHMARK
    args.repeat_idx = _cli_repeat_idx          # BENCHMARK
    args.device = f"cuda:{CUDA_VISIBLE_DEVICES}"

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    if args.time_schedule_path:
        logger.log(f"### Loading time schedule from {args.time_schedule_path}...")
        diffusion._load_time_schedule(args.time_schedule_path)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)
    # model_emb, tokenizer = load_model_emb(args, tokenizer)
    
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args).eval().requires_grad_(False)

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
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_{args.note}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    idx = 0

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1
            if idx >= 500:
                break

    except StopIteration:
        print('### End of reading iteration...')
    
    model_emb.to(dist_util.dev())

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    sentence_counter = 0  # running count of real (non-dummy) sentences processed, for diagnostic filenames
    batch_counter = 0     # BENCHMARK: running count of real batches, for timing rows

    for cond in iterator:

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
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        # BENCHMARK: time only the core sampling call. th.cuda.synchronize() before
        # AND after is required -- CUDA kernel launches are async, so a bare
        # time.time() around an unsynchronized call mostly measures launch
        # overhead, not actual GPU compute time.
        th.cuda.synchronize()
        t0 = time.time()
        with autocast():
            samples, pred_xstart_list = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap
            )
        th.cuda.synchronize()
        elapsed = time.time() - t0

        if rank == 0:
            append_timing_row(args.timing_csv, {
                "method": "normal",
                "steps": args.step,
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

                for zt, pred_x0 in zip(samples, pred_xstart_list):
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

        sample = samples[-1]

        # print('decoding for seq2seq', )
        # print(sample.shape)

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        # tokenizer = load_tokenizer(args)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            non_zero_indices = (input_mask == 1).nonzero()
            len_x = non_zero_indices[0].item() if len(non_zero_indices) > 0 else (args.seq_len - sum(input_mask).tolist())
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            non_zero_indices = (input_mask == 1).nonzero()
            len_x = non_zero_indices[0].item() if len(non_zero_indices) > 0 else (args.seq_len - sum(input_mask).tolist())
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, 'a')
                for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                fout.close()
            dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')

if __name__ == "__main__":
    main()