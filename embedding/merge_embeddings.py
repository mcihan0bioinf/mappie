#!/usr/bin/env python3
"""Merge each HIPPIE PPI's two per-protein embeddings into one interaction
vector per op, written to combine_embeddings/combinations/{source}_{merge}.pkl
for train_autoencoder.py. `multiply` is the op actually used at inference;
the rest exist for the model_selection/ grid search that picked it.

Filters HIPPIE to score >= HIPPIE_SCORE_THRESH (default 0.64, the primary
reference set: 199,137 PPIs / 15,503 proteins) before merging, and writes
hippie_pairs_064.csv / hippie_pairs_top10.csv (>=0.82, the top-10% comparison
subset) for train_autoencoder.py's confidence-filter grid."""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HIPPIE_FILE = os.environ.get("HIPPIE_FULL_TXT", "data/hippie/hippie_current.txt")
EMBEDDING_DIR = Path(os.environ.get("ESM_EMBEDDING_DIR", "embeddings/esm"))
OUTPUT_DIR = Path(os.environ.get("MERGED_EMBEDDING_DIR", "combine_embeddings/combinations"))
SOURCE_NAME = os.environ.get("EMBEDDING_SOURCE_NAME", "esm")
REPR_LAYER = int(os.environ.get("ESM_REPR_LAYER", "33"))
HIPPIE_SCORE_THRESH = float(os.environ.get("HIPPIE_SCORE_THRESH", "0.64"))
HIPPIE_TOP10_THRESH = float(os.environ.get("HIPPIE_TOP10_THRESH", "0.82"))
FILTER_LIST_DIR = Path(os.environ.get("HIPPIE_FILTER_LIST_DIR", "."))

MERGE_OPS = {
    "average": lambda a, b: (a + b) / 2,
    "multiply": lambda a, b: a * b,
    "difference": lambda a, b: a - b,
    "concat_ab": lambda a, b: np.concatenate([a, b]),
    "concat_ba": lambda a, b: np.concatenate([b, a]),
}


def load_embedding(protein_id: str, cache: dict) -> np.ndarray | None:
    if protein_id in cache:
        return cache[protein_id]
    path = EMBEDDING_DIR / f"{protein_id}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    tensor = data["mean_representations"][REPR_LAYER] if isinstance(data, dict) else data
    vec = tensor.numpy()
    cache[protein_id] = vec
    return vec


def main():
    df = pd.read_csv(HIPPIE_FILE, sep="\t", header=None, names=["p1", "id1", "p2", "id2", "score", "info"])
    df = df[df["score"] >= HIPPIE_SCORE_THRESH].reset_index(drop=True)
    pairs = list(zip(df["p1"], df["p2"]))
    print(f"HIPPIE score >= {HIPPIE_SCORE_THRESH}: {len(pairs):,} PPIs")

    FILTER_LIST_DIR.mkdir(parents=True, exist_ok=True)
    keys = [f"{a}_{b}" for a, b in pairs]
    pd.Series(keys, name="ppikey").to_csv(FILTER_LIST_DIR / "hippie_pairs_064.csv", index=False)
    top10_keys = [k for k, s in zip(keys, df["score"]) if s >= HIPPIE_TOP10_THRESH]
    pd.Series(top10_keys, name="ppikey").to_csv(FILTER_LIST_DIR / "hippie_pairs_top10.csv", index=False)
    print(f"Wrote filter lists: {len(keys):,} (>{HIPPIE_SCORE_THRESH}), {len(top10_keys):,} (>={HIPPIE_TOP10_THRESH})")

    cache: dict[str, np.ndarray] = {}
    merged = {op: {} for op in MERGE_OPS}
    skipped = 0

    for protA, protB in tqdm(pairs, desc="Merging PPI embeddings"):
        embA = load_embedding(protA, cache)
        embB = load_embedding(protB, cache)
        if embA is None or embB is None:
            skipped += 1
            continue
        key = f"{protA}_{protB}"
        for op, fn in MERGE_OPS.items():
            merged[op][key] = fn(embA, embB)

    if skipped:
        print(f"Skipped {skipped} pairs missing a per-protein embedding.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for op, vectors in merged.items():
        out_path = OUTPUT_DIR / f"{SOURCE_NAME}_{op}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(vectors, f)
        print(f"Saved {len(vectors)} pairs: {out_path}")


if __name__ == "__main__":
    main()
