import os
import sys
import glob
import json
import re
import argparse
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu_single_file(file_path):
    data = []
    
    # Read and parse the JSON / JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in '{file_path}': {e}")
        return None

    # Prepare tokenized references and candidates
    references_tokenized = [[item["reference"].split()] for item in data if "reference" in item]
    candidates_tokenized = [item["recover"].split() for item in data if "recover" in item]

    if not references_tokenized or not candidates_tokenized:
        print(f"No valid reference/recover data found in '{file_path}'.")
        return None

    # Calculate Standard BLEU-4
    standard_score = corpus_bleu(references_tokenized, candidates_tokenized)
    
    # Calculate Smoothed BLEU-4
    smoothie = SmoothingFunction().method1
    smoothed_score = corpus_bleu(references_tokenized, candidates_tokenized, smoothing_function=smoothie)

    return {
        "file_path": file_path,
        "num_samples": len(data),
        "standard_bleu": standard_score,
        "smoothed_bleu": smoothed_score
    }

def find_seed_files(file_path):
    """
    Given a file path, finds all matching files in the same directory that share
    the same name structure but have different random seed numbers (e.g. seed123, seed101).
    """
    if '*' in file_path or '?' in file_path:
        matched = sorted(glob.glob(file_path))
        if matched:
            return matched

    dirname = os.path.dirname(file_path) or '.'
    basename = os.path.basename(file_path)

    # Match seed<digits> pattern in basename
    seed_match = re.search(r'seed\d+', basename)
    if seed_match:
        prefix = basename[:seed_match.start()]
        suffix = basename[seed_match.end():]
        base_suffix = re.sub(r'\.jsonl?$', '', suffix)
        pattern = re.compile(rf"^{re.escape(prefix)}seed\d+{re.escape(base_suffix)}\.jsonl?$")

        try:
            candidates = os.listdir(dirname)
        except OSError:
            candidates = []

        matched_files = [
            os.path.join(dirname, f) for f in sorted(candidates) if pattern.match(f)
        ]
        if matched_files:
            return matched_files

    if os.path.exists(file_path):
        return [file_path]
    
    return []

def calculate_bleu_from_jsonl(file_path):
    matched_files = find_seed_files(file_path)
    if not matched_files:
        print(f"Error: No matching seed files found for path '{file_path}'.")
        return None

    results = []
    for fp in matched_files:
        res = compute_bleu_single_file(fp)
        if res is not None:
            results.append(res)

    if not results:
        print("No valid evaluation results obtained.")
        return None

    std_scores = [r["standard_bleu"] for r in results]
    smooth_scores = [r["smoothed_bleu"] for r in results]

    mean_std = float(np.mean(std_scores))
    sd_std = float(np.std(std_scores, ddof=1)) if len(std_scores) > 1 else 0.0

    mean_smooth = float(np.mean(smooth_scores))
    sd_smooth = float(np.std(smooth_scores, ddof=1)) if len(smooth_scores) > 1 else 0.0

    print(f"\n{'='*65}")
    print(f"BLEU-4 Evaluation Summary across {len(results)} seed file(s):")
    print(f"{'='*65}")
    for r in results:
        fname = os.path.basename(r['file_path'])
        print(f"  - {fname}: Standard BLEU-4 = {r['standard_bleu']:.4f}, Smoothed BLEU-4 = {r['smoothed_bleu']:.4f} ({r['num_samples']} samples)")
    print(f"{'-'*65}")
    print(f"Standard Corpus BLEU-4: {mean_std:.4f} +/- {sd_std:.4f} (mean +/- sd)")
    print(f"Smoothed Corpus BLEU-4: {mean_smooth:.4f} +/- {sd_smooth:.4f} (mean +/- sd)")
    print(f"{'='*65}\n")

    return {
        "mean_standard": mean_std,
        "sd_standard": sd_std,
        "mean_smoothed": mean_smooth,
        "sd_smoothed": sd_smooth,
        "individual_results": results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean +/- sd BLEU-4 score across seed files.")
    parser.add_argument("path", type=str, nargs="?", default=None, help="Path to a seed json/jsonl file or pattern")
    parser.add_argument("--path", type=str, dest="path_opt", default=None, help="Path to a seed json/jsonl file or pattern")
    args = parser.parse_args()

    target_path = args.path or args.path_opt
    if target_path:
        calculate_bleu_from_jsonl(target_path)
    else:
        # Default test line if no CLI argument provided
        default_file = '../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_best_ckpt_200samp_200k.json'
        calculate_bleu_from_jsonl(default_file)