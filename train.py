"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    #NEW CHANGE -----
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
    #-----------------
    args.save_dir = args.checkpoint_path # Important for adaptive noisin
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight # use model's weights as init
    )
    next(data)

    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight # using the same embedding wight with tranining data
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    args.device = dist_util.dev()
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    args.device = ""
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    if args.resume_checkpoint != 'none' and args.resume_checkpoint != '':
        try:
            from train_util import parse_resume_step_from_filename
            resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
            resume_dir = os.path.dirname(args.resume_checkpoint)
            potential_npy = os.path.join(resume_dir, f'alpha_cumprod_step_{resume_step}.npy')
            if os.path.exists(potential_npy):
                logger.log(f"### Resuming adaptive schedule from {potential_npy}")
                diffusion._load_time_schedule(potential_npy)
            else:
                logger.log(f"### Warning: No matching schedule found at {potential_npy}. Starting with default.")
        except Exception as e:
            logger.log(f"### Error attempting to resume schedule: {e}")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
