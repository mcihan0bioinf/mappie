#!/usr/bin/env python3
"""DDI local purity under degree-preserving randomization (Fig. S5A)
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors

MAPPIE_ROOT = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE = Path(os.environ.get("MAPPIE_LATENT_INDEX", str(MAPPIE_ROOT / "data_processed/latent_index.npz")))
UNIPROT_XREF_TSV = Path(os.environ.get("UNIPROT_XREF_DOMAINS_TSV", "data/uniprot/uniprot_xref_domains.tsv"))
THREEDID_CSV = Path(os.environ.get("THREEDID_DOMAIN_PAIRS_CSV", "data/3did/3did_domain_pairs_with_pfam.csv"))
NULL_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "results"

N_NULL = 20
K_PURITY = 15
MIN_SIZE = 30
N_SUBSAMPLES = 10
MIN_M_SIZE_MATCHED = 5
RNG_SEED = 42


def parse_ppikey(key: str) -> tuple[str, str]:
    p1, p2 = key.split("_HUMAN_", 1)
    return p1 + "_HUMAN", p2


def strip_pfam_version(pfam_id):
    s = str(pfam_id).strip() if pfam_id is not None else ""
    return None if not s or s.lower() == "nan" else s.split(".")[0]


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
        pf1, pf2 = strip_pfam_version(pf1_raw), strip_pfam_version(pf2_raw)
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


def build_network(keys, vecs, protein_pfams, ddi_map):
    ddi_sets = assign_ddi_terms(keys, protein_pfams, ddi_map)
    pos_to_prots = [parse_ppikey(k) for k in keys]
    term_to_indices = defaultdict(list)
    for i, terms in enumerate(ddi_sets):
        if terms:
            for t in terms:
                term_to_indices[t].append(i)

    nn = NearestNeighbors(n_neighbors=K_PURITY + 50, algorithm="auto", n_jobs=-1).fit(vecs)
    _, nbr_idx = nn.kneighbors(vecs)  # oversample neighbours; partner-sharing ones get filtered below
    return {
        "keys": keys, "ddi_sets": ddi_sets, "pos_to_prots": pos_to_prots,
        "term_to_indices": term_to_indices, "nbr_idx": nbr_idx,
    }


def partner_free_purity_for_sample(sample, net, term, rng):
    hits = total = 0
    for i in sample:
        qa, qb = net["pos_to_prots"][i]
        n_kept = 0
        for j in net["nbr_idx"][i]:
            if j == i:
                continue
            ja, jb = net["pos_to_prots"][j]
            if qa in (ja, jb) or qb in (ja, jb):
                continue  # shares a protein with the query -- excluded
            total += 1
            if net["ddi_sets"][j] and term in net["ddi_sets"][j]:
                hits += 1
            n_kept += 1
            if n_kept >= K_PURITY:
                break
    return hits / total if total else np.nan


def size_matched_partner_free(term, real_net, null_net, rng):
    real_members = real_net["term_to_indices"].get(term, [])
    null_members = null_net["term_to_indices"].get(term, [])
    m = min(len(real_members), len(null_members))
    if m < MIN_M_SIZE_MATCHED:
        return None
    real_vals, null_vals = [], []
    for _ in range(N_SUBSAMPLES):
        real_sample = rng.choice(real_members, m, replace=False)
        null_sample = rng.choice(null_members, m, replace=False)
        real_vals.append(partner_free_purity_for_sample(real_sample, real_net, term, rng))
        null_vals.append(partner_free_purity_for_sample(null_sample, null_net, term, rng))
    return {"m": m, "real": float(np.nanmean(real_vals)), "null": float(np.nanmean(null_vals))}


def main():
    protein_pfams, ddi_map = load_ddi_annotators()

    d = np.load(LATENT_FILE, allow_pickle=True)
    real_keys, real_vecs = list(np.asarray(d["keys"], dtype=str)), d["vecs"].astype(np.float32)
    print(f"[Real network] {len(real_keys):,} PPIs")
    real_net = build_network(real_keys, real_vecs, protein_pfams, ddi_map)
    candidate_terms = [t for t, idxs in real_net["term_to_indices"].items() if len(idxs) >= MIN_SIZE]
    print(f"  {len(candidate_terms)} candidate DDI terms (>={MIN_SIZE} PPIs)")

    rows = []
    for seed in range(N_NULL):
        null_npz = NULL_DIR / f"null_latent_seed_{seed}.npz"
        dn = np.load(null_npz, allow_pickle=True)
        null_keys, null_vecs = list(np.asarray(dn["keys"], dtype=str)), dn["vecs"].astype(np.float32)
        null_net = build_network(null_keys, null_vecs, protein_pfams, ddi_map)

        rng = np.random.default_rng(RNG_SEED + seed)
        for term in candidate_terms:
            result = size_matched_partner_free(term, real_net, null_net, rng)
            if result:
                rows.append({"seed": seed, "ddi_term": term, **result})
        print(f"  [seed {seed}] done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ddi_null_size_matched_partner_free.tsv", sep="\t", index=False)
    print(f"[Output] saved ddi_null_size_matched_partner_free.tsv ({len(df)} rows)")

    per_term = df.groupby("ddi_term")[["real", "null"]].mean()
    n_above = int((per_term["real"] > per_term["null"]).sum())
    stat, p = wilcoxon(per_term["real"], per_term["null"], alternative="greater")
    summary = pd.DataFrame([{
        "n_terms": len(per_term), "n_terms_real_gt_null": n_above,
        "frac_terms_real_gt_null": n_above / len(per_term),
        "mean_real": per_term["real"].mean(), "mean_null": per_term["null"].mean(),
        "wilcoxon_statistic": stat, "wilcoxon_pvalue_onesided": p,
    }])
    summary.to_csv(OUT_DIR / "ddi_null_size_matched_partner_free_summary.tsv", sep="\t", index=False)
    print("[Output] saved ddi_null_size_matched_partner_free_summary.tsv")
    print(summary.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
