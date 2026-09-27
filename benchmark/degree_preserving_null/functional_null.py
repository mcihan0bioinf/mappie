#!/usr/bin/env python3
"""Functional-recovery degree-preserving null comparison (Fig. S5B)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(os.environ.get("MAPPIE_REPO_ROOT", Path(__file__).resolve().parents[2]))))
from core_algorithm.run_enrichment import run_enrichment_db, bh_adjust, DB_FILE

MAPPIE_ROOT = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE = Path(os.environ.get("MAPPIE_LATENT_INDEX", str(MAPPIE_ROOT / "data_processed/latent_index.npz")))
ANNOT_DB = Path(os.environ.get("MAPPIE_ANNOTATION_DB", DB_FILE))
NULL_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "results"

N_NULL = 20
N_SAMPLE_REAL = 1500
N_SAMPLE_NULL = 200
K_VALUES = [10, 25, 50, 100, 200, 500]
PADJ = 0.05
RNG_SEED = 42


def parse_ppikey(key: str) -> tuple[str, str]:
    p1, p2 = key.split("_HUMAN_", 1)
    return p1 + "_HUMAN", p2


def own_terms(ppikey: str, db_path: str) -> set[tuple[str, str]]:
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute("SELECT term_type, term FROM annotations WHERE ppikey=?", (ppikey,)).fetchall()
    con.close()
    return set(rows)


def significant_terms(neigh_ppis: list[str], db_path: str) -> set[tuple[str, str]]:
    raw = run_enrichment_db(neigh_ppis, db_path, padj_cutoff=None)
    if not raw:
        return set()
    pvals = [r["p_value"] for r in raw]
    padj = bh_adjust(pvals)
    return {(r["term_type"], r["term"]) for r, p in zip(raw, padj) if p < PADJ}


def run_arm(keys, vecs, db_path, n_sample, rng_seed=RNG_SEED):
    pos_to_prots = [parse_ppikey(k) for k in keys]
    rng = np.random.default_rng(rng_seed)
    selected = rng.choice(len(keys), min(n_sample, len(keys)), replace=False)

    max_k = max(K_VALUES)
    records = []
    for si in selected:
        own = own_terms(keys[si], db_path)
        if not own:
            continue

        qv = vecs[si]
        sims = vecs @ qv
        sims[si] = -np.inf
        top_idx = np.argpartition(-sims, max_k)[:max_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        rec = {"ppikey": keys[si]}
        neighbor_keys, prev_k = [], 0
        for k in K_VALUES:
            neighbor_keys.extend(keys[j] for j in top_idx[prev_k:k])
            prev_k = k
            sig = significant_terms(neighbor_keys, db_path)
            rec[f"overall_k{k}"] = len(own & sig) / len(own)
        records.append(rec)
    return pd.DataFrame(records)


def summarize_k(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"k": k, "overall": df[f"overall_k{k}"].mean()} for k in K_VALUES])


def empirical_pvalue(observed, null_vals):
    r = int((null_vals >= observed).sum())
    return (r + 1) / (len(null_vals) + 1)


def main():
    d = np.load(LATENT_FILE, allow_pickle=True)
    real_keys = list(np.asarray(d["keys"], dtype=str))
    real_vecs = d["vecs"].astype(np.float32)
    print(f"[Real network] {len(real_keys):,} PPIs")

    real_df = run_arm(real_keys, real_vecs, str(ANNOT_DB), N_SAMPLE_REAL)
    real_summary = summarize_k(real_df).assign(network="real")

    null_summaries = []
    for seed in range(N_NULL):
        dn = np.load(NULL_DIR / f"null_latent_seed_{seed}.npz", allow_pickle=True)
        null_keys = list(np.asarray(dn["keys"], dtype=str))
        null_vecs = dn["vecs"].astype(np.float32)
        df = run_arm(null_keys, null_vecs, str(ANNOT_DB), N_SAMPLE_NULL, rng_seed=RNG_SEED + seed)
        null_summaries.append(summarize_k(df).assign(network=f"null_seed_{seed}"))
        print(f"  [seed {seed}] done")

    all_summary = pd.concat([real_summary] + null_summaries, ignore_index=True)
    all_summary.to_csv(OUT_DIR / "functional_null_per_network.tsv", sep="\t", index=False)
    print(f"[Output] saved functional_null_per_network.tsv ({len(all_summary)} rows)")

    null_only = all_summary[all_summary["network"] != "real"]
    rows = []
    for k in K_VALUES:
        obs = all_summary.loc[(all_summary["network"] == "real") & (all_summary["k"] == k), "overall"].iloc[0]
        vals = null_only.loc[null_only["k"] == k, "overall"].values
        rows.append({
            "k": k, "observed_real": obs, "null_mean": vals.mean(), "null_std": vals.std(),
            "exceeds_all_null": bool(obs > vals.max()), "empirical_pvalue": empirical_pvalue(obs, vals),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "functional_null_summary.tsv", sep="\t", index=False)
    print("[Output] saved functional_null_summary.tsv")
    print(summary.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
