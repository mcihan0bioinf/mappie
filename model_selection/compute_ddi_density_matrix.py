#!/usr/bin/env python3
"""Per candidate latent space, per DDI d with PPI set P(d): contrast(d) =
mean(||z_p - centroid(P(d))||) / global mean over DDIs. Tight clustering ->
contrast < 1; select_latent_space.py ranks candidates by proportion of DDIs
below 1."""
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

DDI_COUNTS_CSV = os.environ.get("DDI_COUNTS_CSV", "data/3did/3did_domain_pairs_with_pfam.csv")
PFAM_BINARY_MATRIX_CSV = os.environ.get("PFAM_BINARY_MATRIX_CSV", "data/3did/protein_pfam_binary_matrix.csv")
LATENT_DIR = os.environ.get("CANDIDATE_LATENT_DIR", "data/optimize_encoder")
OUTPUT_CSV = os.environ.get("DDI_DENSITY_MATRIX_CSV", "results/ddi_density_matrix.csv")


def strip_pfam_version(pfam_id) -> str | None:
    s = str(pfam_id).strip() if pfam_id is not None else ""
    if not s or s.lower() == "nan":
        return None
    return s.split(".")[0]


def build_ddi_to_protein_pairs(ddi_df: pd.DataFrame, pfam_matrix: pd.DataFrame) -> dict[tuple[str, str], list]:
    ddi_df = ddi_df.copy()
    ddi_df["pfam1"] = ddi_df["domain1_pfam"].map(strip_pfam_version)
    ddi_df["pfam2"] = ddi_df["domain2_pfam"].map(strip_pfam_version)
    ddi_df = ddi_df.dropna(subset=["pfam1", "pfam2"])
    ddi_df["domain_pair"] = ddi_df.apply(
        lambda row: tuple(sorted([row["pfam1"], row["pfam2"]])), axis=1
    )
    ddi_list = ddi_df["domain_pair"].drop_duplicates().tolist()

    domain_to_proteins = {
        domain: set(pfam_matrix.index[pfam_matrix[domain] == 1])
        for domain in pfam_matrix.columns
    }

    ddi_to_protein_pairs = {}
    for d1, d2 in ddi_list:
        if d1 not in domain_to_proteins or d2 not in domain_to_proteins:
            ddi_to_protein_pairs[(d1, d2)] = []
            continue
        ddi_to_protein_pairs[(d1, d2)] = [
            (a, b) for a in domain_to_proteins[d1] for b in domain_to_proteins[d2] if a != b
        ]
    return ddi_to_protein_pairs


def score_latent_space(latent_path: str, ddi_to_protein_pairs: dict) -> tuple[str, dict] | None:
    latent_df = pd.read_csv(latent_path, index_col=0)
    if latent_df.shape[1] == 0:
        return None

    available = set(latent_df.index)
    ddi_distances = {}
    for (d1, d2), prot_pairs in ddi_to_protein_pairs.items():
        vecs = []
        for p1, p2 in prot_pairs:
            key1, key2 = f"{p1}_{p2}", f"{p2}_{p1}"
            if key1 in available:
                vecs.append(latent_df.loc[key1].values)
            elif key2 in available:
                vecs.append(latent_df.loc[key2].values)
        if len(vecs) < 2:
            ddi_distances[f"{d1}_{d2}"] = np.nan
            continue
        vecs = np.stack(vecs)
        centroid = vecs.mean(axis=0)
        ddi_distances[f"{d1}_{d2}"] = np.mean(np.linalg.norm(vecs - centroid, axis=1))

    valid = [v for v in ddi_distances.values() if not np.isnan(v) and v > 0]
    if not valid:
        return None
    global_mean = np.mean(valid)

    row = {ddi: (dist / global_mean if global_mean > 0 else 0) for ddi, dist in ddi_distances.items()}
    return os.path.basename(latent_path), row


def main():
    print("Loading DDI and Pfam data...")
    ddi_df = pd.read_csv(DDI_COUNTS_CSV)
    pfam_matrix = pd.read_csv(PFAM_BINARY_MATRIX_CSV, index_col=0)
    ddi_to_protein_pairs = build_ddi_to_protein_pairs(ddi_df, pfam_matrix)

    latent_files = sorted(glob(os.path.join(LATENT_DIR, "*latent*.csv")))
    print(f"Scoring {len(latent_files)} candidate latent spaces...")

    results = []
    for path in tqdm(latent_files):
        r = score_latent_space(path, ddi_to_protein_pairs)
        if r is not None:
            results.append(r)

    all_ddis = sorted({ddi for _, row in results for ddi in row})
    matrix = pd.DataFrame(index=all_ddis)
    for latent_name, row in results:
        matrix[latent_name] = pd.Series(row)
    matrix = matrix.fillna(0.0).T

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    matrix.to_csv(OUTPUT_CSV)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
