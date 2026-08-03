"""
run_benchmark.py

Drives:
  - sample_seq2seq.py          (normal inference, fixed at --normal_steps, e.g. 2000)
  - sample_seq2seq_dpmSolver.py (DPM-Solver, swept across --dpm_steps)

For each configuration, runs (1 + --num_repeats) times: repeat_idx=0 is a
discarded warmup (first CUDA call in a process has extra init overhead),
repeat_idx=1..num_repeats are the real, averaged timings.

All timing rows land in one shared CSV (default benchmark_timing.csv),
appended to by sample_seq2seq.py / sample_seq2seq_dpmSolver.py via
timing_utils.append_timing_row. Run analyze_benchmark.py afterwards.

Usage:
    python run_benchmark.py \
        --model_path path/to/checkpoint.pt \
        --time_schedule_path path/to/schedule.npy \
        --num_repeats 5 \
        --dpm_steps 10 12 15 20 25 50 100
"""
import argparse
import os
import sys
import random


def run(cmd):
    print("### RUNNING:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"### WARNING: command exited with code {ret} -- check output above")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True,
                    help='Path to the single .pt checkpoint file you want to benchmark.')
    p.add_argument('--time_schedule_path', type=str, required=True,
                    help='Path to the .npy noise schedule file matching the checkpoint.')
    p.add_argument('--normal_steps', type=int, default=2000,
                    help='full-schedule step count for the normal-inference baseline')
    p.add_argument('--dpm_steps', type=int, nargs='+', default=[10, 12, 15, 20, 25, 50, 100])
    p.add_argument('--num_repeats', type=int, default=5,
                    help='real (non-warmup) repeats per configuration; 1 extra warmup run is added automatically')
    p.add_argument('--bsz', type=int, default=50)
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--seed', type=int, default=101)
    #ADDED top_p to type float from int 
    p.add_argument('--top_p', type=float, default=-1, help='top p used in sampling, default is off')
    p.add_argument('--rejection_rate', type=float, default=0.0, help='reject tokens once it does not change')
    p.add_argument('--timing_csv', type=str, default='benchmark_timing.csv')
    p.add_argument('--skip_normal', action='store_true', help='skip the normal baseline (e.g. re-running just the DPM sweep)')
    p.add_argument('--skip_dpm', action='store_true', help='skip the DPM sweep (e.g. re-running just the normal baseline)')
    args = p.parse_args()

    # set working dir to the upper folder (repository root)
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    total_runs = 1 + args.num_repeats  # + 1 warmup at repeat_idx=0

    # Ensure output dir exists
    if not os.path.isdir('generation_outputs'):
        os.mkdir('generation_outputs')

    if not args.skip_normal:
        print(f"\n=== Normal baseline: {args.normal_steps} steps, {total_runs} runs (1 warmup + {args.num_repeats} timed) ===")
        for repeat_idx in range(total_runs):
            port = f"12{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}"
            cmd = (
                f"python -m torch.distributed.launch --nproc_per_node=1 "
                f"--master_port={port} --use_env "
                f"sample_seq2seq.py "
                f"--model_path '{args.model_path}' "
                f"--step {args.normal_steps} "
                f"--batch_size {args.bsz} "
                f"--seed2 {args.seed} "
                f"--split {args.split} "
                f"--time_schedule_path '{args.time_schedule_path}' "
                f"--top_p {args.top_p} "
                f"--rejection_rate {args.rejection_rate} "
                f"--timing_csv '{args.timing_csv}' "
                f"--repeat_idx {repeat_idx} "
                f"--note bench_normal_r{repeat_idx} "
                f"--out_dir generation_outputs"
            )
            run(cmd)

    if not args.skip_dpm:
        for steps in args.dpm_steps:
            print(f"\n=== DPM-Solver: {steps} steps, {total_runs} runs (1 warmup + {args.num_repeats} timed) ===")
            for repeat_idx in range(total_runs):
                port = f"12{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}"
                cmd = (
                    f"python -m torch.distributed.launch --nproc_per_node=1 "
                    f"--master_port={port} --use_env "
                    f"sample_seq2seq_dpmSolver.py "
                    f"--model_path '{args.model_path}' "
                    f"--step {steps} "
                    f"--batch_size {args.bsz} "
                    f"--seed2 {args.seed} "
                    f"--split {args.split} "
                    f"--time_schedule_path '{args.time_schedule_path}' "
                    f"--top_p {args.top_p} "
                    f"--rejection_rate {args.rejection_rate} "
                    f"--timing_csv '{args.timing_csv}' "
                    f"--repeat_idx {repeat_idx} "
                    f"--note bench_dpm{steps}_r{repeat_idx} "
                    f"--out_dir generation_outputs"
                )
                run(cmd)

    print(f"\n### Benchmark sweep finished. Timing data in: {args.timing_csv}")
    print(f"### Next: python analyze_benchmark.py --timing_csv {args.timing_csv}")


if __name__ == "__main__":
    main()
