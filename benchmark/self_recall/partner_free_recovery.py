#!/usr/bin/env python3
"""Functional recovery excluding shared-protein neighbours (Fig. S6)."""
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
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_SAMPLE_PARTNER_FREE = 1000
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


def run_arm(keys, vecs, db_path, n_sample, partner_free):
    pos_to_prots = [parse_ppikey(k) for k in keys]
    rng = np.random.default_rng(RNG_SEED)
    selected = rng.choice(len(keys), min(n_sample, len(keys)), replace=False)

    max_k = max(K_VALUES)
    records = []
    for si in selected:
        a, b = pos_to_prots[si]
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
            for j in top_idx[prev_k:k]:
                pa, pb = pos_to_prots[j]
                if partner_free and (pa in (a, b) or pb in (a, b)):
                    continue  # shares a protein with the query -- excluded
                neighbor_keys.append(keys[j])
            prev_k = k
            sig = significant_terms(neighbor_keys, db_path)
            rec[f"overall_k{k}"] = len(own & sig) / len(own)
        records.append(rec)
    return pd.DataFrame(records)


def summarize_k(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"k": k, "overall": df[f"overall_k{k}"].mean()} for k in K_VALUES])


def main():
    d = np.load(LATENT_FILE, allow_pickle=True)
    real_keys = list(np.asarray(d["keys"], dtype=str))
    real_vecs = d["vecs"].astype(np.float32)
    print(f"[Real network] {len(real_keys):,} PPIs")

    df_pf = run_arm(real_keys, real_vecs, str(ANNOT_DB), N_SAMPLE_PARTNER_FREE, partner_free=True)
    df_normal = run_arm(real_keys, real_vecs, str(ANNOT_DB), N_SAMPLE_PARTNER_FREE, partner_free=False)
    out = pd.concat([
        summarize_k(df_pf).assign(mode="partner_free"),
        summarize_k(df_normal).assign(mode="normal_same_n"),
    ], ignore_index=True)
    out.to_csv(OUT_DIR / "functional_partner_free.tsv", sep="\t", index=False)
    print("[Output] saved functional_partner_free.tsv")
    print(out.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
