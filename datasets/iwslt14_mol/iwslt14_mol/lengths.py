"""
Token length statistics for molecule JSONL dataset.
Usage: python token_length_stats.py --data path/to/data.jsonl
"""

import argparse
import json
import numpy as np
from transformers import AutoTokenizer
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL file")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="HuggingFace tokenizer name")
    parser.add_argument("--src_key", default="src", help="Key for source field")
    parser.add_argument("--trg_key", default="trg", help="Key for target field")
    return parser.parse_args()

def load_jsonl(path, src_key, trg_key):
    srcs, trgs = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                srcs.append(obj[src_key])
                trgs.append(obj[trg_key])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipping line {i}: {e}")
    return srcs, trgs

def token_lengths(texts, tokenizer):
    lengths = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(ids))
    return np.array(lengths)

def print_stats(name, lengths, thresholds=[64, 128, 256, 512]):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Count      : {len(lengths):,}")
    print(f"  Min        : {lengths.min()}")
    print(f"  Max        : {lengths.max()}")
    print(f"  Mean       : {lengths.mean():.1f}")
    print(f"  Median     : {int(np.median(lengths))}")
    print(f"  Std        : {lengths.std():.1f}")
    print()
    print("  Percentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"    p{p:<3} : {int(np.percentile(lengths, p))}")
    print()
    print("  % of examples fitting within seq_len cutoffs:")
    for t in thresholds:
        pct = 100.0 * (lengths <= t).sum() / len(lengths)
        bar = "#" * int(pct / 2)
        print(f"    <= {t:>4} tokens : {pct:5.1f}%  {bar}")

def print_combined_stats(src_lengths, trg_lengths, thresholds=[64, 128, 256, 512]):
    combined = src_lengths + trg_lengths  # src + trg concatenated length
    print(f"\n{'='*50}")
    print(f"  COMBINED (src + trg, as seen by model)")
    print(f"{'='*50}")
    print(f"  Count      : {len(combined):,}")
    print(f"  Min        : {combined.min()}")
    print(f"  Max        : {combined.max()}")
    print(f"  Mean       : {combined.mean():.1f}")
    print(f"  Median     : {int(np.median(combined))}")
    print()
    print("  % of examples fitting within seq_len cutoffs:")
    for t in thresholds:
        pct = 100.0 * (combined <= t).sum() / len(combined)
        bar = "#" * int(pct / 2)
        print(f"    <= {t:>4} tokens : {pct:5.1f}%  {bar}")
    print()
    recommended = int(np.percentile(combined, 95))
    # Round up to nearest power of 2
    import math
    recommended_pow2 = 2 ** math.ceil(math.log2(recommended))
    print(f"  Recommended seq_len (p95, rounded to power of 2): {recommended_pow2}")

def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading data from: {args.data}")
    srcs, trgs = load_jsonl(args.data, args.src_key, args.trg_key)
    print(f"Loaded {len(srcs):,} examples")

    print("\nTokenizing (this may take a moment)...")
    src_lengths = token_lengths(srcs, tokenizer)
    trg_lengths = token_lengths(trgs, tokenizer)

    print_stats("SOURCE (SMILES)", src_lengths)
    print_stats("TARGET (description)", trg_lengths)
    print_combined_stats(src_lengths, trg_lengths)

    # Flag the worst offenders
    combined = src_lengths + trg_lengths
    n_over = {t: int((combined > t).sum()) for t in [128, 256, 512]}
    print(f"\n  Examples truncated at common seq_len values:")
    for t, n in n_over.items():
        print(f"    seq_len={t:<4}: {n:,} examples truncated ({100*n/len(combined):.1f}%)")

if __name__ == "__main__":
    main()