"""
Evaluation script for diffuseqv2_tgmdlm generation outputs.

Computes:
  - chrF++            (sacrebleu, word_order=2)
  - METEOR            (nltk)
  - BERTScore F1       (bert_score, same model_type convention as scripts/eval_seq2seq.py)
  - MAUVE              (mauve-text)

Reads every *.jsonl line-delimited output file in the given folder (each line:
{"source": ..., "reference": ..., "recover": ...}), scores each file separately,
then reports the mean +/- standard deviation of each metric across all files.

Usage:
    python scripts/eval_generation_metrics.py --folder generation_outputs/<model>/<ckpt>.samples
    python scripts/eval_generation_metrics.py --folder generation_outputs/<model>/<ckpt>.samples --mbr
    python scripts/eval_generation_metrics.py --folder generation_outputs/<model>/<ckpt>.samples --out results.json
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import torch

import sacrebleu
from bert_score import score as bert_score_fn

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

try:
    import mauve
except ImportError:
    mauve = None


def ensure_nltk_resources():
    """meteor_score needs wordnet (+ omw-1.4 for lemmatization); punkt is not
    required since we tokenize with a plain .split(), but nltk.translate.meteor
    used to also touch punkt on older versions, so we grab it defensively."""
    for pkg, path in [
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("punkt", "tokenizers/punkt"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def get_bleu(recover, reference):
    return sentence_bleu(
        [reference.split()], recover.split(),
        smoothing_function=SmoothingFunction().method4,
    )


def select_best(sentences):
    """Same MBR candidate-selection rule as eval_seq2seq.py: pick the
    candidate with highest total self-BLEU against the other candidates."""
    self_bleu = [[] for _ in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            self_bleu[i].append(get_bleu(s1, s2))
    for i in range(len(sentences)):
        self_bleu[i][i] = 0
    idx = np.argmax(np.sum(self_bleu, -1))
    return sentences[idx]


def clean(text, args, is_recover=False):
    text = text.replace(args.eos, "").replace(args.sos, "").replace(args.sep, "")
    if is_recover:
        text = text.replace(args.pad, "")
    return text.strip()


def compute_metrics(recovers, references, tag, bertscore_model, mauve_device_id):
    """Compute chrF++, METEOR, BERTScore F1, and MAUVE over a list of
    recover/reference sentence pairs, returning a dict of results."""
    results = {"tag": tag, "n": len(recovers)}

    # --- chrF++ (sacrebleu; word_order=2 => chrF++) ---
    chrf_metric = sacrebleu.CHRF(word_order=2)
    per_sent_chrf = [
        chrf_metric.sentence_score(rec, [ref]).score
        for rec, ref in zip(recovers, references)
    ]
    corpus_chrf = chrf_metric.corpus_score(recovers, [references]).score
    results["chrf++_sentence_avg"] = float(np.mean(per_sent_chrf))
    results["chrf++_corpus"] = float(corpus_chrf)

    # --- METEOR ---
    ensure_nltk_resources()
    per_sent_meteor = [
        meteor_score([ref.split()], rec.split())
        for rec, ref in zip(recovers, references)
    ]
    results["meteor"] = float(np.mean(per_sent_meteor))

    # --- BERTScore F1 ---
    P, R, F1 = bert_score_fn(
        recovers, references, model_type=bertscore_model, lang="en", verbose=True
    )
    results["bertscore_f1"] = float(torch.mean(F1))

    # --- MAUVE ---
    if mauve is None:
        results["mauve"] = None
        results["mauve_note"] = "mauve-text not installed; pip install mauve-text"
    elif len(recovers) < 2:
        results["mauve"] = None
        results["mauve_note"] = "MAUVE needs a reasonably sized sample set (skipped, n<2)"
    else:
        mauve_out = mauve.compute_mauve(
            p_text=recovers,
            q_text=references,
            device_id=mauve_device_id,
            verbose=False,
        )
        results["mauve"] = float(mauve_out.mauve)

    return results


def print_results(results):
    print("*" * 30)
    print(f"[{results['tag']}]  n={results['n']}")
    print(f"  chrF++ (sentence avg) : {results['chrf++_sentence_avg']:.4f}")
    print(f"  chrF++ (corpus)       : {results['chrf++_corpus']:.4f}")
    print(f"  METEOR                : {results['meteor']:.4f}")
    print(f"  BERTScore F1          : {results['bertscore_f1']:.4f}")
    if results.get("mauve") is not None:
        print(f"  MAUVE                 : {results['mauve']:.4f}")
    else:
        print(f"  MAUVE                 : skipped ({results.get('mauve_note', 'n/a')})")


AGGREGATE_METRIC_KEYS = [
    "chrf++_sentence_avg",
    "chrf++_corpus",
    "meteor",
    "bertscore_f1",
    "mauve",
]


def aggregate_across_files(all_results):
    """Given one results dict per file, compute mean +/- sd for each metric
    across files. Files where a metric is missing/None (e.g. MAUVE skipped)
    are excluded from that metric's aggregate, not treated as zero."""
    agg = {}
    for key in AGGREGATE_METRIC_KEYS:
        values = [r[key] for r in all_results if r.get(key) is not None]
        n_used = len(values)
        if n_used == 0:
            agg[key] = {"mean": None, "sd": None, "n_files": 0}
        else:
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n_used > 1 else 0.0
            agg[key] = {"mean": mean, "sd": sd, "n_files": n_used}
    return agg


def print_aggregate(agg, n_files):
    print("\n" + "=" * 30)
    print(f"AGGREGATE across {n_files} file(s) (mean +/- sd)")
    print("=" * 30)
    labels = {
        "chrf++_sentence_avg": "chrF++ (sentence avg)",
        "chrf++_corpus": "chrF++ (corpus)      ",
        "meteor": "METEOR               ",
        "bertscore_f1": "BERTScore F1         ",
        "mauve": "MAUVE                ",
    }
    for key, label in labels.items():
        stat = agg[key]
        if stat["mean"] is None:
            print(f"  {label} : n/a (no files produced this metric)")
        else:
            print(f"  {label} : {stat['mean']:.4f} +/- {stat['sd']:.4f}  (n_files={stat['n_files']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chrF++ / METEOR / BERTScore-F1 / MAUVE eval")
    parser.add_argument("--folder", type=str, default="", help="path to the folder of decoded texts (evaluates every *.jsonl file in it)")
    parser.add_argument("--mbr", action="store_true", help="MBR-select one candidate per source before scoring (single combined result, no aggregate)")
    parser.add_argument("--sos", type=str, default="[CLS]", help="start token of the sentence")
    parser.add_argument("--eos", type=str, default="[EOS]", help="end token of the sentence")
    parser.add_argument("--sep", type=str, default="[SEP]", help="sep token of the sentence")
    parser.add_argument("--pad", type=str, default="[PAD]", help="pad token of the sentence")
    parser.add_argument("--bertscore_model", type=str, default="microsoft/deberta-xlarge-mnli",
                         help="bert_score model_type (matches eval_seq2seq.py default)")
    parser.add_argument("--mauve_device_id", type=int, default=-1,
                         help="GPU id for MAUVE's feature extractor, -1 for CPU")
    parser.add_argument("--out", type=str, default="", help="optional path to dump results as JSON")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.json")))
    if not files:
        sys.exit(f"No *.json files found under {args.folder}")
    print(f"Found {len(files)} .json file(s) to evaluate:")
    for f in files:
        print(f"  - {f}")

    sample_num = 0
    with open(files[0], "r") as f:
        for _ in f:
            sample_num += 1

    sentence_dict = {i: [] for i in range(sample_num)}
    reference_dict = {i: [] for i in range(sample_num)}

    all_results = []

    for path in files:
        print(path)
        recovers, references = [], []

        with open(path, "r") as f:
            cnt = 0
            for row in f:
                obj = json.loads(row)
                reference = clean(obj["reference"].strip(), args)
                recover = clean(obj["recover"].strip(), args, is_recover=True)

                recovers.append(recover)
                references.append(reference)

                sentence_dict[cnt].append(recover)
                reference_dict[cnt].append(reference)
                cnt += 1

        if not args.mbr:
            res = compute_metrics(
                recovers, references, tag=os.path.basename(path),
                bertscore_model=args.bertscore_model,
                mauve_device_id=args.mauve_device_id,
            )
            print_results(res)
            all_results.append(res)

    if args.mbr:
        print("*" * 30)
        print("MBR...")
        print("*" * 30)
        recovers, references = [], []
        for k, v in sentence_dict.items():
            if len(v) == 0 or len(reference_dict[k]) == 0:
                continue
            recovers.append(select_best(v))
            references.append(reference_dict[k][0])

        res = compute_metrics(
            recovers, references, tag="mbr",
            bertscore_model=args.bertscore_model,
            mauve_device_id=args.mauve_device_id,
        )
        print_results(res)
        all_results.append(res)
        output = {"per_file": all_results}
    else:
        agg = aggregate_across_files(all_results)
        print_aggregate(agg, n_files=len(all_results))
        output = {"per_file": all_results, "aggregate": agg}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.out}")