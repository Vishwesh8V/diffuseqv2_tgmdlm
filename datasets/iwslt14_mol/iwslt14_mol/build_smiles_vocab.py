"""
build_smiles_vocab.py
─────────────────────
Scan your train/valid/test.jsonl files, collect every unique SMILES token
that appears in the 'trg' field, and write a vocab.json file that
SmilesVocab (text_datasets.py) can load.

Usage:
    python build_smiles_vocab.py \
        --data_dir ./datasets/my_data \
        --out      ./datasets/smiles_vocab.json \
        --min_freq 1

The tokenisation pattern is identical to the one inside SmilesVocab._split_smiles,
so the vocab is guaranteed to cover every token the model will ever see at
train / inference time.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ── same regex as SmilesVocab._split_smiles ──────────────────────────────────
PATTERN = re.compile(
    r"(\[[^\]]+]"       # bracketed atoms  e.g. [NH3+], [nH]
    r"|Br?"             # Br or B
    r"|Cl?"             # Cl or C
    r"|N|O|S|P|F|I"
    r"|Si|Se|se|As|Te"  # two-char atoms
    r"|@@|@"            # chirality
    r"|%\d{2}"          # ring closures >= 10  e.g. %10
    r"|[a-z]"           # aromatic atoms (c,n,o,s,p,b)
    r"|.)"              # everything else char by char
)

# Reserved special tokens — must come first and keep these IDs fixed.
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


def split_smiles(smi: str):
    return PATTERN.findall(smi)


def build_vocab(data_dir: str, min_freq: int = 1) -> dict:
    counter: Counter = Counter()
    data_dir = Path(data_dir)

    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"  scanning {path} …")
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                smiles = obj.get("src", "").strip()
                if smiles:
                    counter.update(split_smiles(smiles))

    # Filter by minimum frequency
    vocab_tokens = [tok for tok, cnt in sorted(counter.items()) if cnt >= min_freq]

    # Build final token→id dict: specials first, then sorted corpus tokens
    token2id = {}
    for tok in SPECIAL_TOKENS:
        token2id[tok] = len(token2id)
    for tok in vocab_tokens:
        if tok not in token2id:
            token2id[tok] = len(token2id)

    return token2id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing train/valid/test.jsonl")
    parser.add_argument("--out", default="./datasets/smiles_vocab.json",
                        help="Where to write the vocab JSON file")
    parser.add_argument("--min_freq", type=int, default=1,
                        help="Minimum token frequency to include (default 1 = keep all)")
    args = parser.parse_args()

    print(f"\nBuilding SMILES vocab from {args.data_dir} …")
    token2id = build_vocab(args.data_dir, args.min_freq)
    print(f"  vocab size: {len(token2id)}  (including {len(SPECIAL_TOKENS)} special tokens)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(token2id, f, indent=2)
    print(f"  written → {args.out}\n")

    # Quick sanity check
    print("  First 10 tokens:")
    for tok, idx in list(token2id.items())[:10]:
        print(f"    {idx:4d}  {tok!r}")


if __name__ == "__main__":
    main()
