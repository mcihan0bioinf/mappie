#!/usr/bin/env python3
"""CORUM (FCG) per-complex neighbourhood recovery: pool k=100 latent
neighbours across a complex's MAPPIE-covered subunit pairs, run MAPPIE's
enrichment, and check recall of the complex's own FCG GO terms."""

import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(os.environ.get("MAPPIE_REPO_ROOT", Path(__file__).resolve().parents[2]))))
from core_algorithm.run_enrichment import run_enrichment_db, bh_adjust

FCG_FILE = Path(os.environ.get("CORUM_FCG_TXT", str(Path(__file__).parent / "data/corum_fcg.txt")))
LATENT_FILE  = Path(os.environ.get("MAPPIE_LATENT_INDEX", "../mappie/data_processed/latent_index.npz"))
HIPPIE_FULL  = Path(os.environ.get("HIPPIE_FULL_TXT", "data/hippie/hippie_current.txt"))
UNIPROT_TSV  = Path(os.environ.get("UNIPROT_PROTEOME_TSV", "data/uniprot/uniprotkb_proteome_UP000005640.tsv"))
ANNOT_DB     = Path(os.environ.get("MAPPIE_ANNOTATION_DB", "../mappie/data_processed/uniprot/ppi_annotation_long_ddi.db"))
OUTPUT_DIR   = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RNG          = np.random.default_rng(42)
K_NEIGHBOURS = 100
MIN_HIPPIE   = 2   # "more than one PPI reconstructed" from HIPPIE
PADJ_CUTOFF  = 0.05
_go_id_re    = re.compile(r"(GO:\d+)")

print("1. UniProt accession -> mnemonic map")
uni = pd.read_csv(UNIPROT_TSV, sep="\t", usecols=["Entry", "Entry Name"])
acc_to_mnem = dict(zip(uni["Entry"], uni["Entry Name"]))
print(f"  {len(acc_to_mnem):,} entries")

print("\n2. Parsing corum_fcg.txt (Human) -- per-complex subunit + GO ID lists")
fcg = pd.read_csv(FCG_FILE, sep="\t")
fcg_human = fcg[fcg["Organism"] == "Human"]

complex_subunits: dict[int, set] = defaultdict(set)
complex_go: dict[int, set] = defaultdict(set)
complex_name: dict[int, str] = {}
complex_root: dict[int, str] = {}
complex_fcg_name: dict[int, str] = {}

for _, row in fcg_human.iterrows():
    cid = int(row["ComplexID"])
    complex_name[cid] = str(row["ComplexName"])
    complex_root[cid] = str(row["Root"])
    complex_fcg_name[cid] = str(row["Functional Complex Group"])
    accs = [a.strip() for a in str(row.get("Subunits(UniProt IDs)", "")).split(";") if a.strip()]
    for acc in accs:
        mnem = acc_to_mnem.get(acc)
        if mnem:
            complex_subunits[cid].add(mnem)
    go_ids = _go_id_re.findall(str(row.get("GO ID", "")))
    complex_go[cid].update(go_ids)

print(f"  {len(complex_subunits):,} human complexes with subunits mapped")
print(f"  median GO terms/complex: {np.median([len(v) for v in complex_go.values()]):.0f}")

print("\n3. Full HIPPIE pair set (all confidence levels)")
hippie_raw = pd.read_csv(HIPPIE_FULL, sep="\t", header=None,
                          names=["prot1", "id1", "prot2", "id2", "score", "info"])
hippie_pairs: set[str] = set()
for p1, p2 in zip(hippie_raw["prot1"], hippie_raw["prot2"]):
    a, b = sorted([str(p1), str(p2)])
    hippie_pairs.add(f"{a}_{b}")
print(f"  Unique HIPPIE pairs: {len(hippie_pairs):,}")


def canonical(a, b):
    a2, b2 = sorted([a, b])
    return f"{a2}_{b2}"


print("\n4. MAPPIE latent")
_z = np.load(LATENT_FILE, allow_pickle=True)
_keys = list(np.asarray(_z["keys"], dtype=str))
lat_norm = _z["vecs"].astype(np.float32)
latent_keys = set(_keys)
key_to_row = {k: i for i, k in enumerate(_keys)}
n_all = len(_keys)
print(f"  {n_all:,} PPIs x {lat_norm.shape[1]}d")

print("\n5. Building intra-complex PPI working set (qualifying complexes only)")
intra_ppi_to_cids: dict[str, set] = defaultdict(set)
complex_hippie_pairs: dict[int, list] = {}

for cid, prots in complex_subunits.items():
    if len(prots) < 2:
        continue
    hip_pairs = []
    for a, b in combinations(sorted(prots), 2):
        if canonical(a, b) in hippie_pairs:
            hip_pairs.append((a, b))
    if len(hip_pairs) < MIN_HIPPIE:
        continue
    complex_hippie_pairs[cid] = hip_pairs
    for a, b in hip_pairs:
        k1, k2 = f"{a}_{b}", f"{b}_{a}"
        fk = k1 if k1 in latent_keys else (k2 if k2 in latent_keys else None)
        if fk:
            intra_ppi_to_cids[fk].add(cid)

print(f"  {len(complex_hippie_pairs):,} complexes with >={MIN_HIPPIE} HIPPIE-confirmed intra-complex pairs")
intra_keys_all = list(intra_ppi_to_cids.keys())
intra_rows_all = np.array([key_to_row[k] for k in intra_keys_all])
intra_vecs_all = lat_norm[intra_rows_all]
n_intra = len(intra_keys_all)
print(f"  MAPPIE-covered intra-complex PPIs: {n_intra:,}")
intra_key_index = {k: i for i, k in enumerate(intra_keys_all)}

print(f"\n6. kNN (k={K_NEIGHBOURS}) for all intra-complex PPIs")
nn_model = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, metric="cosine", algorithm="brute", n_jobs=-1)
nn_model.fit(lat_norm)
BATCH = 1000
nbr_idx = np.zeros((n_intra, K_NEIGHBOURS + 1), dtype=np.int32)
for start in range(0, n_intra, BATCH):
    end = min(start + BATCH, n_intra)
    _, idx = nn_model.kneighbors(intra_vecs_all[start:end])
    nbr_idx[start:end] = idx
    if start % 10000 == 0:
        print(f"    {end}/{n_intra} ...")
pos_to_key = _keys


def get_neighbours(i):
    self_pos = intra_rows_all[i]
    raw = nbr_idx[i]
    mask = raw != self_pos
    return raw[mask][:K_NEIGHBOURS]


print("\n7. Per-complex enrichment recall")
rows = []
for qi, (cid, hip_pairs) in enumerate(complex_hippie_pairs.items()):
    corum_go = complex_go.get(cid, set())
    mappie_local_idx = []
    for a, b in hip_pairs:
        k1, k2 = f"{a}_{b}", f"{b}_{a}"
        fk = k1 if k1 in intra_key_index else (k2 if k2 in intra_key_index else None)
        if fk:
            mappie_local_idx.append(intra_key_index[fk])
    n_mappie = len(mappie_local_idx)

    if n_mappie >= 1 and corum_go:
        nbr_ppikeys = set()
        for li in mappie_local_idx:
            for nb_pos in get_neighbours(li):
                nbr_ppikeys.add(pos_to_key[int(nb_pos)])
        enrich_results = run_enrichment_db(list(nbr_ppikeys), str(ANNOT_DB), padj_cutoff=None)
        if enrich_results:
            pvals = [r["p_value"] for r in enrich_results]
            padjs = bh_adjust(pvals)
            for r, padj in zip(enrich_results, padjs):
                r["padj"] = padj
            sig_results = [r for r in enrich_results if r["padj"] < PADJ_CUTOFF]
        else:
            sig_results = []
        sig_go_ids = set()
        for r in sig_results:
            if r["term_type"] in ("GO_BP", "GO_MF", "GO_CC"):
                m = _go_id_re.search(r["term"])
                if m:
                    sig_go_ids.add(m.group(1))
        n_covered = len(corum_go & sig_go_ids)
        recall = n_covered / len(corum_go)
    else:
        n_covered = 0
        recall = float("nan")

    rows.append({
        "complex_id": cid,
        "complex_name": complex_name.get(cid, ""),
        "root": complex_root.get(cid, ""),
        "fcg_subcategory": complex_fcg_name.get(cid, ""),
        "n_subunits": len(complex_subunits[cid]),
        "n_hippie_ppis": len(hip_pairs),
        "n_mappie_ppis": n_mappie,
        "n_corum_go_terms": len(corum_go),
        "n_nbr_go_covered": n_covered,
        "nbr_go_recall": round(recall, 4) if recall == recall else float("nan"),
    })
    if qi % 200 == 0:
        print(f"  {qi:,}/{len(complex_hippie_pairs):,} complexes ...")

df = pd.DataFrame(rows)
out_csv = OUTPUT_DIR / "corum_fcg_go_recall_table.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

scored = df.dropna(subset=["nbr_go_recall"])
print(f"\n{len(df):,} complexes total, {len(scored):,} with a valid recall score")
print(f"Mean recall: {scored['nbr_go_recall'].mean():.3f}")
print(f"Median recall: {scored['nbr_go_recall'].median():.3f}")

root_stats = scored.groupby("root").agg(n_complexes=("nbr_go_recall", "count"), mean_recall=("nbr_go_recall", "mean")).sort_values("mean_recall", ascending=False)
print("\nBy Root category:")
print(root_stats)
root_stats.to_csv(OUTPUT_DIR / "corum_fcg_go_recall_by_root.csv")

sub_stats = scored.groupby("fcg_subcategory").agg(n_complexes=("nbr_go_recall", "count"), mean_recall=("nbr_go_recall", "mean"), root=("root", "first")).sort_values("mean_recall", ascending=False)
print("\nBy subcategory:")
print(sub_stats)
sub_stats.to_csv(OUTPUT_DIR / "corum_fcg_go_recall_by_subcategory.csv")

print("\nAll done.")
