#!/usr/bin/env python3
"""Held-out validation of DDI-guided model selection, plus cross-PLM selection stability (Fig. S2, S3).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DENSITY_MATRIX_CSV = os.environ.get("DDI_DENSITY_MATRIX_CSV", "results/ddi_density_matrix.csv")
PROSTT5_DENSITY_MATRIX_CSV = os.environ.get("PROSTT5_DENSITY_MATRIX_CSV", "results/ddi_density_matrix_prostt5.csv")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

SELECTED_CONFIG = "esm_multiply_ld128_064_latent.csv"
FILTER_TIER = "064"
MAX_DIM = 512  # ld1024 was only ever scored for ProstT5, not comparable across all three sources
DEV_FRACTION = 0.80
N_SEEDS = 50


def load_model_matrix(model: str) -> pd.DataFrame:
    """One model's own tier-064, ld<=512 candidates, columns = raw DDI contrast scores."""
    csv = PROSTT5_DENSITY_MATRIX_CSV if model == "prostt5" else DENSITY_MATRIX_CSV
    m = pd.read_csv(csv, index_col=0)
    m = m[~m.index.str.startswith("both_")]  # leave out the ESM+ProtBERT combination experiment
    m = m[m.index.str.startswith(f"{model}_")]
    m = m[m.index.str.contains(f"_{FILTER_TIER}_")]
    m = m[~m.index.str.contains("_ld1024_")]
    return m


def proportion_under_1(matrix: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The exact model-selection statistic from select_latent_space.py, restricted to `cols`."""
    return (matrix[cols] < 1).mean(axis=1)


def run_fig_s2(esm_matrix: pd.DataFrame):
    print("[Fig. S2] production ESM-2 held-out validation, 50 random 80/20 DDI splits ...")
    ddi_cols = esm_matrix.columns[(esm_matrix == 0).sum(axis=0) == 0].tolist()
    print(f"  {len(ddi_cols)} DDI columns usable by every ESM-2 candidate")

    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(ddi_cols)
        n_dev = int(round(DEV_FRACTION * len(shuffled)))
        dev_cols, heldout_cols = list(shuffled[:n_dev]), list(shuffled[n_dev:])

        dev_scores = proportion_under_1(esm_matrix, dev_cols)
        best_config = dev_scores.idxmax()
        heldout_scores = proportion_under_1(esm_matrix, heldout_cols)

        rows.append({
            "seed": seed,
            "selected_config": best_config,
            "selected_matches_production": best_config == SELECTED_CONFIG,
            "dev_score_selected": dev_scores[best_config],
            "heldout_score_selected": heldout_scores[best_config],
            "production_dev_score": dev_scores.get(SELECTED_CONFIG, np.nan),
            "production_heldout_score": heldout_scores.get(SELECTED_CONFIG, np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "heldout_esm_per_split.csv", index=False)
    print(f"  [Output] saved heldout_esm_per_split.csv ({len(df)} rows)")
    print(f"  selected == production in {df['selected_matches_production'].sum()}/{N_SEEDS} splits")
    print(f"  mean held-out score (production config): {df['production_heldout_score'].mean():.4f}")


def run_fig_s3():
    print("\n[Fig. S3] cross-PLM selection stability, same 50 splits, same statistic ...")
    matrices = {model: load_model_matrix(model) for model in ["esm", "protbert", "prostt5"]}
    for model, m in matrices.items():
        print(f"  [{model}] {m.shape[0]} candidates")

    # A DDI column must be usable (nonzero) by every candidate of every model,
    # so all three embedding sources are compared on an identical DDI universe.
    shared_cols = None
    for m in matrices.values():
        usable = set(m.columns[(m == 0).sum(axis=0) == 0])
        shared_cols = usable if shared_cols is None else (shared_cols & usable)
    ddi_cols = sorted(shared_cols)
    print(f"  {len(ddi_cols)} DDI columns usable by every candidate of every model")
    matrices = {model: m[ddi_cols] for model, m in matrices.items()}

    top_config = {
        "esm": "esm_multiply_ld128_064_latent.csv",
        "protbert": "protbert_multiply_ld128_064_latent.csv",
        "prostt5": "prostt5_multiply_ld128_064_latent.csv",
    }

    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(ddi_cols)
        n_dev = int(round(DEV_FRACTION * len(shuffled)))
        dev_cols, heldout_cols = list(shuffled[:n_dev]), list(shuffled[n_dev:])

        for model, matrix in matrices.items():
            dev_scores = proportion_under_1(matrix, dev_cols)
            own_best = dev_scores.idxmax()
            heldout_scores = proportion_under_1(matrix, heldout_cols)
            cfg = top_config[model]
            rows.append({
                "seed": seed, "model": model,
                "own_dev_selected_config": own_best,
                "own_dev_selected_is_multiply_ld128": own_best == cfg,
                "multiply_ld128_heldout_score": heldout_scores[cfg],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "heldout_cross_plm_per_split.csv", index=False)
    print(f"  [Output] saved heldout_cross_plm_per_split.csv ({len(df)} rows)")

    summary = df.groupby("model").agg(
        n_splits=("seed", "count"),
        n_splits_multiply_ld128_is_own_top=("own_dev_selected_is_multiply_ld128", "sum"),
        mean_heldout_score=("multiply_ld128_heldout_score", "mean"),
        std_heldout_score=("multiply_ld128_heldout_score", "std"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "heldout_cross_plm_summary.csv", index=False)
    print("  [Output] saved heldout_cross_plm_summary.csv")
    print(summary.to_string(index=False))


def main():
    esm_matrix = load_model_matrix("esm")
    run_fig_s2(esm_matrix)
    run_fig_s3()
    print("\nDone.")


if __name__ == "__main__":
    main()
