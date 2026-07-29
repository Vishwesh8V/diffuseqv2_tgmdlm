import os, sys, glob
import argparse
import random
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--model_dir', type=str, default='', help='path to the folder of diffusion model (ignored if --checkpoint is given)')
    parser.add_argument('--checkpoint', type=str, default='', help='path to ONE specific checkpoint .pt file to test. If set, this overrides '
                         '--model_dir/--pattern entirely -- only this exact checkpoint runs, instead of the default "latest checkpoint" pick.')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--step', type=int, default=2000, help='if less than diffusion training steps, like 1000, use ddim sampling')
    parser.add_argument('--clamp_step', type=int, default=0, help='clamp start step')
    parser.add_argument('--rejection_rate', type=float, default=0.0, help='reject tokens once it does not change')
    parser.add_argument('--note', type=str, default='none', help='note')

    parser.add_argument('--bsz', type=int, default=50, help='batch size')
    parser.add_argument('--start_n', type=int, default=0, help='start batch iteration')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help='dataset split used to decode')
    #ADDED top_p to type float from int 
    parser.add_argument('--top_p', type=float, default=-1, help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema', help='training pattern')
    parser.add_argument('--time_schedule_path', type=str, required=True, help='path to the .npy alpha schedule file')
    parser.add_argument('--save_trajectories', action='store_true', help='dump per-sentence alpha/beta noise-schedule + token/entropy denoising trajectories (one .npz per test sentence) for diagnostic plotting via plot_alphas.py')
    # BENCHMARK: new flags forwarded through to sample_seq2seq_dpmSolver.py for the DPM-vs-normal speed benchmark
    parser.add_argument('--timing_csv', type=str, default='benchmark_timing.csv', help='path to append per-run timing rows for the benchmark')
    parser.add_argument('--repeat_idx', type=int, default=0, help='repeat index for this run (0 = warmup, excluded from analysis)')

    args = parser.parse_args()

    # Resolve --checkpoint to an absolute path BEFORE the chdir below.
    checkpoint_abs = os.path.abspath(args.checkpoint) if args.checkpoint else ''

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    output_lst = []

    if checkpoint_abs:
        checkpoint_iter = [[checkpoint_abs]]
    else:
        checkpoint_iter = []
        for lst in glob.glob(args.model_dir):
            print(lst)
            checkpoint_iter.append(sorted(glob.glob(f"{lst}/{args.pattern}*pt"))[::-1][:1])

    out_dir = 'generation_outputs'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for checkpoints in checkpoint_iter:
        for checkpoint_one in checkpoints:
            # Always use the explicitly provided schedule pat
            current_schedule = args.time_schedule_path

            # FIX: the line `f'--note {args.note}'` used to have NO trailing backslash,
            # which silently terminated the COMMAND assignment right there -- the
            # `--time_schedule_path` and `--save_trajectories` lines below it were being
            # evaluated as a separate, discarded expression and never reached the
            # subprocess. Added the missing backslash so every piece actually gets forwarded.
            COMMAND = f'python -m torch.distributed.launch --nproc_per_node=1 --master_port=12{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)} --use_env sample_seq2seq_dpmSolver.py ' \
            f'--model_path {checkpoint_one} --step {args.step} ' \
            f'--batch_size {args.bsz} --start_n {args.start_n} --seed2 {args.seed} --split {args.split} ' \
            f'--out_dir {out_dir} --top_p {args.top_p} ' \
            f'--rejection_rate {args.rejection_rate} --clamp_step {args.clamp_step} '\
            f'--note {args.note}'\
            f' --time_schedule_path {current_schedule}'\
            f' --timing_csv {args.timing_csv} --repeat_idx {args.repeat_idx}'\
            f'{" --save_trajectories True" if args.save_trajectories else ""}'
            print(COMMAND)
            
            os.system(COMMAND)
    
    print('#'*30, 'decoding finished...')
