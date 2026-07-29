"""
run_benchmark.py

Drives:
  - run_decode.py      (normal inference, fixed at --normal_steps, e.g. 2000)
  - run_decode_solver.py  (DPM-Solver, swept across --dpm_steps)

For each configuration, runs (1 + --num_repeats) times: repeat_idx=0 is a
discarded warmup (first CUDA call in a process has extra init overhead),
repeat_idx=1..num_repeats are the real, averaged timings.

All timing rows land in one shared CSV (default benchmark_timing.csv),
appended to by sample_seq2seq.py / sample_seq2seq_dpmSolver.py via
timing_utils.append_timing_row. Run analyze_benchmark.py afterwards.

Usage:
    python run_benchmark.py \
        --model_dir path/to/checkpoint_folder \
        --time_schedule_path path/to/schedule.npy \
        --num_repeats 5 \
        --dpm_steps 10 12 15 20 25 50 100
"""
import argparse
import os


def run(cmd):
    print("### RUNNING:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"### WARNING: command exited with code {ret} -- check output above")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True,
                    help='glob pattern / folder passed straight through to run_decode.py --model_dir. '
                         'Point this at a folder containing exactly ONE checkpoint you want timed -- '
                         'run_decode.py (normal) will otherwise loop over every matching checkpoint, '
                         'multiplying your run count for no benchmarking benefit.')
    p.add_argument('--normal_run_decode', type=str, default='run_decode.py')
    p.add_argument('--dpm_run_decode', type=str, default='run_decode_solver.py')
    p.add_argument('--normal_steps', type=int, default=2000,
                    help='full-schedule step count for the normal-inference baseline')
    p.add_argument('--dpm_steps', type=int, nargs='+', default=[10, 12, 15, 20, 25, 50, 100])
    p.add_argument('--num_repeats', type=int, default=5,
                    help='real (non-warmup) repeats per configuration; 1 extra warmup run is added automatically')
    p.add_argument('--bsz', type=int, default=50)
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--time_schedule_path', type=str, required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--timing_csv', type=str, default='benchmark_timing.csv')
    p.add_argument('--skip_normal', action='store_true', help='skip the normal baseline (e.g. re-running just the DPM sweep)')
    p.add_argument('--skip_dpm', action='store_true', help='skip the DPM sweep (e.g. re-running just the normal baseline)')
    args = p.parse_args()

    total_runs = 1 + args.num_repeats  # + 1 warmup at repeat_idx=0

    if not args.skip_normal:
        print(f"\n=== Normal baseline: {args.normal_steps} steps, {total_runs} runs (1 warmup + {args.num_repeats} timed) ===")
        for repeat_idx in range(total_runs):
            cmd = (
                f"python {args.normal_run_decode} --model_dir '{args.model_dir}' "
                f"--seed {args.seed} --step {args.normal_steps} --bsz {args.bsz} "
                f"--split {args.split} --time_schedule_path {args.time_schedule_path} "
                f"--timing_csv {args.timing_csv} --repeat_idx {repeat_idx} "
                f"--note bench_normal_r{repeat_idx}"
            )
            run(cmd)

    if not args.skip_dpm:
        for steps in args.dpm_steps:
            print(f"\n=== DPM-Solver: {steps} steps, {total_runs} runs (1 warmup + {args.num_repeats} timed) ===")
            for repeat_idx in range(total_runs):
                cmd = (
                    f"python {args.dpm_run_decode} --model_dir '{args.model_dir}' "
                    f"--seed {args.seed} --step {steps} --bsz {args.bsz} "
                    f"--split {args.split} --time_schedule_path {args.time_schedule_path} "
                    f"--timing_csv {args.timing_csv} --repeat_idx {repeat_idx} "
                    f"--note bench_dpm{steps}_r{repeat_idx}"
                )
                run(cmd)

    print(f"\n### Benchmark sweep finished. Timing data in: {args.timing_csv}")
    print(f"### Next: python analyze_benchmark.py --timing_csv {args.timing_csv}")


if __name__ == "__main__":
    main()
