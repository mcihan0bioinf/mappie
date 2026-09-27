#!/usr/bin/env python3
"""Dimension-matched PCA control (Fig. S4)."""
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(os.environ.get("MAPPIE_REPO_ROOT", Path(__file__).resolve().parents[2]))))
from core_algorithm.project_ppi import SCALER_FILE, EMBEDDINGS_DB

MAPPIE_ROOT = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE = Path(os.environ.get("MAPPIE_LATENT_INDEX", str(MAPPIE_ROOT / "data_processed/latent_index.npz")))
UNIPROT_XREF_TSV = Path(os.environ.get("UNIPROT_XREF_DOMAINS_TSV", "data/uniprot/uniprot_xref_domains.tsv"))
THREEDID_CSV = Path(os.environ.get("THREEDID_DOMAIN_PAIRS_CSV", "data/3did/3did_domain_pairs_with_pfam.csv"))
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PCA_DIM = 128
K_PURITY = 15
MIN_SIZE = 30
SAMPLE_PER_TERM = 300
DEV_FRACTION = 0.80
N_SEEDS = 20
RNG_SEED = 42


def parse_ppikey(key: str) -> tuple[str, str]:
    p1, p2 = key.split("_HUMAN_", 1)
    return p1 + "_HUMAN", p2


def build_arms():
    d = np.load(LATENT_FILE, allow_pickle=True)
    keys = list(np.asarray(d["keys"], dtype=str))
    ae_latent = d["vecs"].astype(np.float32)  # already L2-normalized production latent

    con = sqlite3.connect(f"file:{EMBEDDINGS_DB}?mode=ro", uri=True)
    prot_emb = {pid: np.frombuffer(blob, dtype=np.float32).copy()
                for pid, blob in con.execute("SELECT protein_id, embedding FROM embeddings")}
    con.close()

    merged = np.empty((len(keys), 1280), dtype=np.float32)
    for i, k in enumerate(keys):
        a, b = parse_ppikey(k)
        merged[i] = prot_emb[a] * prot_emb[b]

    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(merged)

    pca = PCA(n_components=PCA_DIM, whiten=False, random_state=RNG_SEED)
    pca_latent = pca.fit_transform(X_scaled).astype(np.float32)

    def l2norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (X / n).astype(np.float32)

    return keys, {"pca_128d": l2norm(pca_latent), "ae_128d": ae_latent}


def load_ddi_annotators():
    df = pd.read_csv(UNIPROT_XREF_TSV, sep="\t")
    protein_pfams = {}
    for entry_name, pfam_field in zip(df["Entry Name"], df["Pfam"]):
        if pd.isna(pfam_field) or not str(pfam_field).strip():
            continue
        accs = {a for a in str(pfam_field).split(";") if a}
        if accs:
            protein_pfams[entry_name] = accs
    ddi = pd.read_csv(THREEDID_CSV)
    ddi_map = defaultdict(set)
    for pf1_raw, pf2_raw in zip(ddi["domain1_pfam"], ddi["domain2_pfam"]):
        pf1 = str(pf1_raw).split(".")[0] if pd.notna(pf1_raw) else None
        pf2 = str(pf2_raw).split(".")[0] if pd.notna(pf2_raw) else None
        if pf1 and pf2:
            ddi_map[pf1].add(pf2)
            ddi_map[pf2].add(pf1)
    return protein_pfams, ddi_map


def assign_ddi_terms(keys, protein_pfams, ddi_map):
    out = []
    for key in keys:
        p1, p2 = parse_ppikey(key)
        pfams_a, pfams_b = protein_pfams.get(p1), protein_pfams.get(p2)
        if not pfams_a or not pfams_b:
            out.append(None)
            continue
        terms = {"--".join(sorted((pa, pb)))
                 for pa in pfams_a for pb in ddi_map.get(pa, set()) & pfams_b}
        out.append(terms if terms else None)
    return out


def main():
    keys, arms = build_arms()
    protein_pfams, ddi_map = load_ddi_annotators()
    ddi_sets = assign_ddi_terms(keys, protein_pfams, ddi_map)

    term_to_indices = defaultdict(list)
    for i, terms in enumerate(ddi_sets):
        if terms:
            for t in terms:
                term_to_indices[t].append(i)
    usable_terms = [t for t, idxs in term_to_indices.items() if len(idxs) >= MIN_SIZE]
    print(f"[DDI] {len(usable_terms)} DDI terms with >={MIN_SIZE} PPIs (fixed universe, both arms)")

    nbr_idx_by_arm = {}
    for arm, vecs in arms.items():
        nn = NearestNeighbors(n_neighbors=K_PURITY + 1, algorithm="auto", n_jobs=-1).fit(vecs)
        _, nbr_idx = nn.kneighbors(vecs)
        nbr_idx_by_arm[arm] = nbr_idx
        print(f"  [{arm}] kNN fit done")

    N = len(keys)

    def per_term_purity(arm, terms_subset, rng):
        rows = []
        nbr_idx = nbr_idx_by_arm[arm]
        for term in terms_subset:
            members = term_to_indices[term]
            sample = members if len(members) <= SAMPLE_PER_TERM else \
                list(rng.choice(members, SAMPLE_PER_TERM, replace=False))
            hits = total = 0
            for i in sample:
                for j in nbr_idx[i][1:]:
                    total += 1
                    if ddi_sets[j] and term in ddi_sets[j]:
                        hits += 1
            rows.append((term, hits / total if total else np.nan))
        return dict(rows)

    full_purity = {arm: per_term_purity(arm, usable_terms, np.random.default_rng(RNG_SEED))
                   for arm in arms}

    per_seed_rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(usable_terms)
        heldout_terms = list(shuffled[int(round(DEV_FRACTION * len(shuffled))):])
        for arm in arms:
            purities = [full_purity[arm][t] for t in heldout_terms]
            per_seed_rows.append({"seed": seed, "arm": arm, "n_heldout_terms": len(heldout_terms),
                                   "mean_purity_knn": float(np.nanmean(purities))})

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUT_DIR / "pca_vs_ae_per_seed.tsv", sep="\t", index=False)
    print(f"[Output] saved pca_vs_ae_per_seed.tsv ({len(per_seed_df)} rows)")

    pivot = per_seed_df.pivot(index="seed", columns="arm", values="mean_purity_knn")
    diff = pivot["ae_128d"] - pivot["pca_128d"]
    stat, p = wilcoxon(pivot["ae_128d"], pivot["pca_128d"], alternative="greater")
    summary = pd.DataFrame([{
        "ae_mean": pivot["ae_128d"].mean(), "pca_mean": pivot["pca_128d"].mean(),
        "mean_diff_ae_minus_pca": diff.mean(), "n_seeds_ae_gt_pca": int((diff > 0).sum()),
        "n_seeds": len(diff), "wilcoxon_statistic": stat, "wilcoxon_pvalue_onesided": p,
    }])
    summary.to_csv(OUT_DIR / "pca_vs_ae_summary.tsv", sep="\t", index=False)
    print("[Output] saved pca_vs_ae_summary.tsv")
    print(summary.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
