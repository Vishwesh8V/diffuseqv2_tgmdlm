"""
timing_utils.py

Shared helper for the DPM-Solver vs. normal-inference speed benchmark.
Both sample_seq2seq.py (normal) and sample_seq2seq_dpmSolver.py (DPM) import
`append_timing_row` and call it once per timed sampling call. Because
run_benchmark.py drives many separate process invocations (one per
method/step-count/repeat), all rows accumulate into a single shared CSV that
analyze_benchmark.py later loads with pandas.

Place this file in the same directory as sample_seq2seq.py / basic_utils.py
so the `import timing_utils` in both scripts resolves.
"""
import csv
import os
import time

FIELDNAMES = [
    "method", "steps", "repeat_idx", "batch_idx", "num_sentences",
    "elapsed_sec", "sec_per_sentence", "checkpoint", "seed", "note", "timestamp",
]


def append_timing_row(csv_path, row: dict):
    """
    Append one row to `csv_path`, writing the header first if the file is new.
    `row` need not contain every field in FIELDNAMES -- missing ones are
    written as empty strings, and `timestamp` is filled in automatically if
    not provided.
    """
    row = dict(row)
    row.setdefault("timestamp", time.time())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
