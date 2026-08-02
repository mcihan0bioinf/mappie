#!/usr/bin/env python3
"""Ranks candidate latent spaces by proportion of DDIs with contrast < 1
(from compute_ddi_density_matrix.py), highlighting the selected config
(ESM-2, multiply, 128-d, >0.64 filter). Behind Supp. Fig. 1."""
import os
from pathlib import Path

import pandas as pd

DENSITY_MATRIX_CSV = os.environ.get("DDI_DENSITY_MATRIX_CSV", "results/ddi_density_matrix.csv")
SELECTED_CONFIG = "esm_multiply_ld128_064_latent.csv"
FILTER_TIER = "064"  # HIPPIE > 0.64; the tier scored consistently across all candidates


def main():
    matrix = pd.read_csv(DENSITY_MATRIX_CSV, index_col=0)
    matrix = matrix[~matrix.index.str.startswith("both_")]  # leave-out ESM+ProtBERT combination experiment

    # A DDI column is only usable for comparison if every candidate latent
    # space had at least one valid protein pair for it.
    matrix_clean = matrix.loc[:, (matrix == 0).sum(axis=0) == 0]
    print(f"{matrix.shape[1]} DDI columns -> {matrix_clean.shape[1]} after dropping any-zero columns")

    proportion_under_1 = (matrix_clean < 1).mean(axis=1)

    ranking = pd.DataFrame({
        "latent_space": proportion_under_1.index,
        "proportion_under_1": proportion_under_1.values,
    })
    ranking["model"] = ranking["latent_space"].str.extract(r"^(esm|protbert)_")
    ranking["merge"] = ranking["latent_space"].str.extract(r"^(?:esm|protbert)_(average|multiply|difference|concat_ab)_")
    ranking["dim"] = ranking["latent_space"].str.extract(r"_ld(\d+)_").astype(int)
    ranking["tier"] = ranking["latent_space"].str.extract(r"_(064|all|top10)_latent")
    ranking["selected"] = ranking["latent_space"] == SELECTED_CONFIG

    ranking = ranking[ranking["tier"] == FILTER_TIER].reset_index(drop=True)
    ranking = ranking.sort_values("proportion_under_1", ascending=False).reset_index(drop=True)

    for model in sorted(ranking["model"].unique()):
        sub = ranking[ranking["model"] == model]
        print(f"{model}: n={len(sub)}, median={sub['proportion_under_1'].median():.3f}, "
              f"max={sub['proportion_under_1'].max():.3f}")

    sel_rank = ranking.index[ranking["selected"]].tolist()
    print(f"Selected config ({SELECTED_CONFIG}) overall rank (1-indexed): "
          f"{[r + 1 for r in sel_rank]} of {len(ranking)}")

    out_csv = Path(os.environ.get("LATENT_SELECTION_RANKING_CSV", "results/latent_space_ranking.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_csv, index=False)
    print(f"Saved ranking table: {out_csv}")


if __name__ == "__main__":
    main()
