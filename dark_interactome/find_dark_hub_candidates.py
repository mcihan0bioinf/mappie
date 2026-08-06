#!/usr/bin/env python3
"""Project both-dark pairs into MAPPIE's latent space, enrich, roll up into
candidate dark hub proteins (recurring across >=3 partners; Fig. 5)."""
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

K_NEIGHBOURS = int(os.environ.get("K_NEIGHBOURS", "100"))
ADJ_P = float(os.environ.get("ADJ_P", "0.05"))
ANNOTATION_DB = os.environ.get("MAPPIE_ANNOTATION_DB", "../../mappie/data_processed/uniprot/ppi_annotation_long_ddi.db")
LATENT_INDEX_NPZ = os.environ.get("MAPPIE_LATENT_INDEX", "../../mappie/data_processed/latent_index.npz")
MAPPIE_REPO_ROOT = os.environ.get("MAPPIE_REPO_ROOT", str(Path(__file__).resolve().parents[1]))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results")))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", str(ROOT / "logs")))

GO_TERM_RE = re.compile(r"^(.*?)\s*\[(GO:\d+)\]$")
GENERIC_TERMS = {
    "plasma membrane", "cell surface", "extracellular region", "extracellular space",
    "membrane", "nucleus", "nucleoplasm", "cytoplasm", "cytosol", "endoplasmic reticulum",
    "endoplasmic reticulum membrane", "Golgi membrane", "Golgi apparatus",
}


def rpath(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


class DropLog:
    def __init__(self, stage: str, logs_dir: Path):
        self.path = Path(logs_dir) / f"dropped_{stage}.tsv"
        self._rows: list[dict] = []

    def append(self, entity: str, reason: str, **extra):
        self._rows.append({"entity": entity, "reason": reason, **extra})

    def __len__(self):
        return len(self._rows)

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            with open(self.path, "w") as f:
                f.write("entity\treason\n")
            return
        fieldnames = list({k for row in self._rows for k in row.keys()})
        fieldnames = ["entity", "reason"] + [f for f in fieldnames if f not in ("entity", "reason")]
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            for row in self._rows:
                w.writerow(row)


def load_latent_index(latent_index_npz: str):
    data = np.load(latent_index_npz, allow_pickle=True)
    return data["vecs"], data["keys"]


def build_neigh_tree(vecs: np.ndarray):
    from sklearn.neighbors import NearestNeighbors

    tree = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    tree.fit(vecs)
    return tree


def project_and_enrich(both_dark: pd.DataFrame):
    pairs = both_dark.dropna(subset=["mnemonicA", "mnemonicB"]).reset_index(drop=True)
    droplog_proj = DropLog("project_and_enrich_failures", LOGS_DIR)
    if len(pairs) < len(both_dark):
        for _, row in both_dark[both_dark["mnemonicA"].isna() | both_dark["mnemonicB"].isna()].iterrows():
            droplog_proj.append(row["pair_id"], "missing_mnemonic")

    mappie_root = str(Path(MAPPIE_REPO_ROOT).resolve())
    if mappie_root not in sys.path:
        sys.path.insert(0, mappie_root)
    from core_algorithm import project_ppi as npc
    from core_algorithm.run_enrichment import run_enrichment_db, bh_adjust

    print("[project_and_enrich] loading MAPPIE models (ESM-2 650M, scaler, autoencoder, UMAP)...")
    t0 = time.time()
    npc.init_models()
    print(f"[project_and_enrich] models ready in {time.time()-t0:.1f}s")

    ref_vecs, ref_keys = load_latent_index(str(rpath(LATENT_INDEX_NPZ)))
    tree = build_neigh_tree(ref_vecs)
    print(f"[project_and_enrich] latent index: {ref_vecs.shape[0]:,} known pairs")

    latents = np.zeros((len(pairs), ref_vecs.shape[1]), dtype=np.float32)
    ok_mask = np.zeros(len(pairs), dtype=bool)
    for i, row in pairs.iterrows():
        try:
            _, _, latent = npc.project_ppi(row["uniprotA"], row["uniprotB"], input_type="uniprot")
            latents[i] = np.array(latent, dtype=np.float32)
            ok_mask[i] = True
        except Exception as exc:
            droplog_proj.append(row["pair_id"], f"projection_failed: {exc}")
        if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
            print(f"[project_and_enrich] projected {i+1:,}/{len(pairs):,}", flush=True)
    droplog_proj.write()

    pairs_ok = pairs[ok_mask].reset_index(drop=True)
    latents_ok = latents[ok_mask]

    k = K_NEIGHBOURS
    dists, idxs = tree.kneighbors(latents_ok, n_neighbors=k + 1)
    own_keyset = set(f"{a}_{b}" for a, b in zip(pairs_ok["mnemonicA"], pairs_ok["mnemonicB"]))
    db_path = str(rpath(ANNOTATION_DB))
    adj_p = ADJ_P

    hit_rows = []
    for i in range(len(pairs_ok)):
        neigh_keys = [ref_keys[j] for j in idxs[i]]
        neigh_keys = [kk for kk in neigh_keys if kk not in own_keyset][:k]
        results = run_enrichment_db(neigh_keys, db_path, padj_cutoff=None)
        if not results:
            continue
        padj = bh_adjust([r["p_value"] for r in results])
        pair_id = pairs_ok.loc[i, "pair_id"]
        for r, p_adj in zip(results, padj):
            if p_adj <= adj_p:
                hit_rows.append({
                    "pair_id": pair_id,
                    "uniprotA": pairs_ok.loc[i, "uniprotA"], "uniprotB": pairs_ok.loc[i, "uniprotB"],
                    "term_type": r["term_type"], "term": r["term"],
                    "observed": r["observed"], "expected": r["expected"],
                    "p_value": r["p_value"], "p_adj": p_adj,
                })
        if (i + 1) % 200 == 0 or (i + 1) == len(pairs_ok):
            print(f"[project_and_enrich] enriched {i+1:,}/{len(pairs_ok):,}", flush=True)

    hits = pd.DataFrame(hit_rows)
    hits.to_csv(RESULTS_DIR / "both_dark_enrichment_hits.tsv", sep="\t", index=False)
    print(f"[project_and_enrich] {len(hits):,} significant (pair, term) hits across "
          f"{hits['pair_id'].nunique() if len(hits) else 0:,}/{len(pairs_ok):,} pairs")
    return hits, pairs_ok


def find_convergent_terms(hits: pd.DataFrame, both_dark: pd.DataFrame,
                           min_observed: int = 3, min_supporting_pairs: int = 3, top_n_proteins: int = 20):
    hits["fold"] = hits["observed"] / hits["expected"].replace(0, 0.01)
    func = hits[hits["term_type"].isin(["GO_BP", "GO_MF", "GO_CC"]) & (hits["observed"] >= min_observed)].copy()
    func["term_name"] = func["term"].str.split(r" \[GO:").str[0]
    func = func[~func["term_name"].isin(GENERIC_TERMS)]

    long_a = func[["uniprotA", "term_type", "term_name", "pair_id", "fold"]].rename(columns={"uniprotA": "protein"})
    long_b = func[["uniprotB", "term_type", "term_name", "pair_id", "fold"]].rename(columns={"uniprotB": "protein"})
    long = pd.concat([long_a, long_b], ignore_index=True)

    conv = (long.groupby(["protein", "term_type", "term_name"])
            .agg(n_supporting_pairs=("pair_id", "nunique"), mean_fold=("fold", "mean"), max_fold=("fold", "max"))
            .reset_index())
    conv = conv[conv["n_supporting_pairs"] >= min_supporting_pairs]
    conv = conv.sort_values(["n_supporting_pairs", "mean_fold"], ascending=False)
    conv.to_csv(RESULTS_DIR / "bioplex_dark_convergent_terms.tsv", sep="\t", index=False)

    mnem_a = both_dark[["uniprotA", "mnemonicA", "knownness_A"]].rename(
        columns={"uniprotA": "protein", "mnemonicA": "mnemonic", "knownness_A": "knownness"})
    mnem_b = both_dark[["uniprotB", "mnemonicB", "knownness_B"]].rename(
        columns={"uniprotB": "protein", "mnemonicB": "mnemonic", "knownness_B": "knownness"})
    mnem = pd.concat([mnem_a, mnem_b]).drop_duplicates("protein").set_index("protein")

    n_partners = (pd.concat([
        both_dark[["uniprotA", "pair_id"]].rename(columns={"uniprotA": "protein"}),
        both_dark[["uniprotB", "pair_id"]].rename(columns={"uniprotB": "protein"}),
    ]).groupby("protein")["pair_id"].nunique())
    n_novel_partners = (pd.concat([
        both_dark[~both_dark["in_map"]][["uniprotA", "pair_id"]].rename(columns={"uniprotA": "protein"}),
        both_dark[~both_dark["in_map"]][["uniprotB", "pair_id"]].rename(columns={"uniprotB": "protein"}),
    ]).groupby("protein")["pair_id"].nunique())

    rows = []
    for protein, grp in conv.groupby("protein"):
        top = grp.sort_values(["n_supporting_pairs", "mean_fold"], ascending=False).head(5)
        narrative = "; ".join(
            f"{r.term_name} ({r.n_supporting_pairs} partners, {r.mean_fold:.0f}x avg fold)"
            for r in top.itertuples())
        rows.append({
            "protein": protein,
            "mnemonic": mnem.loc[protein, "mnemonic"] if protein in mnem.index else "",
            "knownness": mnem.loc[protein, "knownness"] if protein in mnem.index else None,
            "n_dark_dark_partners": int(n_partners.get(protein, 0)),
            "n_novel_partners": int(n_novel_partners.get(protein, 0)),
            "n_converging_terms": len(grp),
            "top_converging_terms": narrative,
        })
    per_protein = pd.DataFrame(rows).sort_values(
        ["n_converging_terms", "n_dark_dark_partners"], ascending=False).head(top_n_proteins)
    per_protein.to_csv(RESULTS_DIR / "bioplex_dark_convergent_proteins.tsv",
                        sep="\t", index=False)

    print(f"[find_convergent_terms] {len(conv):,} (protein, term) convergent hits "
          f"(>={min_supporting_pairs} independent partners each); "
          f"{per_protein.shape[0]:,} candidate dark hub proteins in the top-{top_n_proteins} rollup")
    return conv, per_protein


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    both_dark = pd.read_csv(RESULTS_DIR / "bioplex_both_dark_pairs.tsv", sep="\t")
    hits, pairs_ok = project_and_enrich(both_dark)
    find_convergent_terms(hits, both_dark)


if __name__ == "__main__":
    main()
