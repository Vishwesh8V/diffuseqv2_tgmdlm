import os, sys, glob
import argparse
import random
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--model_dir', type=str, default='', help='path to the folder of diffusion model')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='random seeds for 5 inference runs (default: [123, 101, 102, 103, 104])')
    parser.add_argument('--seed', type=int, default=None, help='single random seed (backward compatibility, 5 seeds including 123 will run)')
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
    args = parser.parse_args()

    # Determine 5 seeds ensuring seed 123 is included
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
        for default_s in [123, 101, 102, 103, 104]:
            if default_s not in seeds:
                seeds.append(default_s)
            if len(seeds) == 5:
                break
    else:
        seeds = [123, 101, 102, 103, 104]

    if 123 not in seeds:
        if len(seeds) >= 5:
            seeds[0] = 123
        else:
            seeds.insert(0, 123)

    cur_seed = 100
    while len(seeds) < 5:
        if cur_seed not in seeds:
            seeds.append(cur_seed)
        cur_seed += 1

    print(f"Running inference on 5 seeds: {seeds}")

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    output_lst = []
    for lst in glob.glob(args.model_dir):
        print(lst)
        checkpoints = sorted(glob.glob(f"{lst}/{args.pattern}*pt"))[::-1]

        out_dir = 'generation_outputs'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        for checkpoint_one in checkpoints:
            # Always use the explicitly provided schedule pat
            current_schedule = args.time_schedule_path

            for seed in seeds:
                COMMAND = f'python -m torch.distributed.launch --nproc_per_node=1 --master_port=12{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)} --use_env sample_seq2seq.py ' \
                f'--model_path {checkpoint_one} --step {args.step} ' \
                f'--batch_size {args.bsz} --start_n {args.start_n} --seed2 {seed} --split {args.split} ' \
                f'--out_dir {out_dir} --top_p {args.top_p} ' \
                f'--rejection_rate {args.rejection_rate} --clamp_step {args.clamp_step} '\
                f'--note {args.note}'\
                f' --time_schedule_path {current_schedule}'\
                f'{" --save_trajectories True" if args.save_trajectories else ""}'

                print(COMMAND)
                
                os.system(COMMAND)
    
    print('#'*30, 'decoding finished...')