#!/usr/bin/env python3
"""Does a PPI's own neighbourhood recover its own annotated function, pooled
across all 8 term types (GO_BP/MF/CC, KEGG, REACTOME, PFAM, INTERPRO, RHEA),
for MAPPIE vs. the combined (multiplied) ESM-2 pair embedding without AE
compression, swept across k (Fig. 2C). Default N_SAMPLE=10000 matches the
manuscript's reported sample."""

import sys
import time
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import warnings
import os
warnings.filterwarnings("ignore")

MAPPIE_ROOT = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE = MAPPIE_ROOT / "data_processed/latent_index.npz"
EMB_DB      = MAPPIE_ROOT / "data_processed/protein_embeddings.db"
UNIPROT_TSV = Path(os.environ.get("UNIPROT_PROTEOME_TSV", "data/uniprot/uniprotkb_proteome_UP000005640.tsv"))
GAF_FILE    = Path(os.environ.get("GOA_HUMAN_GAF", "data/go/goa_human.gaf"))
ANNOT_DB    = MAPPIE_ROOT / "data_processed/uniprot/ppi_annotation_long_ddi.db"
KEGG_PATHWAY_LINKS = Path(os.environ.get("KEGG_HSA_PATHWAY_LINKS", str(Path(__file__).parent / "annotation_sources/kegg_hsa_pathway_links.txt")))
KEGG_UNIPROT_CONV  = Path(os.environ.get("KEGG_HSA_UNIPROT_CONV", str(Path(__file__).parent / "annotation_sources/kegg_hsa_uniprot_conv.txt")))
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
K_VALUES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10, 25, 50, 100, 200, 500]
GO_EVIDENCE_TIER = sys.argv[3] if len(sys.argv) > 3 else "strict_ht"
PADJ = 0.05
GO_CLASSES = ["GO_BP", "GO_MF", "GO_CC"]
DB_DERIVED_TYPES = ["REACTOME", "PFAM", "INTERPRO", "RHEA"]
ALL_CLASSES = GO_CLASSES + ["KEGG"] + DB_DERIVED_TYPES
ASPECT_TO_TYPE = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}
SUFFIX = f"_n{N_SAMPLE}_{GO_EVIDENCE_TIER}"

t_start = time.time()
print(f"N_SAMPLE={N_SAMPLE}  K_VALUES={K_VALUES}  GO_EVIDENCE_TIER={GO_EVIDENCE_TIER}")

print("STEP 1: UniProt accession -> mnemonic map")
uni = pd.read_csv(UNIPROT_TSV, sep="\t", usecols=["Entry", "Entry Name"])
acc_to_mnem = dict(zip(uni["Entry"], uni["Entry Name"]))

EVIDENCE_TIERS = {
    "super_strict": {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},
    "strict_ht":    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"},
}
STRICT_EVIDENCE_CODES = EVIDENCE_TIERS[GO_EVIDENCE_TIER]
print("STEP 2: Parsing GOA GAF ...")
protein_go: dict[str, set[tuple[str, str]]] = defaultdict(set)
with open(GAF_FILE) as f:
    for line in f:
        if line.startswith("!"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue
        acc, go_id, evidence, aspect = parts[1], parts[4], parts[6], parts[8]
        if evidence not in STRICT_EVIDENCE_CODES:
            continue
        mnem = acc_to_mnem.get(acc)
        if not mnem:
            continue
        tt = ASPECT_TO_TYPE.get(aspect)
        if not tt:
            continue
        protein_go[mnem].add((tt, go_id))
print(f"  {len(protein_go):,} proteins with >=1 GO term")

print("STEP 2b: Deriving protein-level REACTOME/PFAM/INTERPRO by pair-intersection ...")
ann_con = sqlite3.connect(f"file:{ANNOT_DB}?mode=ro", uri=True)
for tt in DB_DERIVED_TYPES:
    pair_terms: dict[str, set[str]] = defaultdict(set)
    for ppikey, term in ann_con.execute("SELECT ppikey, term FROM annotations WHERE term_type=?", (tt,)):
        pair_terms[ppikey].add(term)
    protein_pairsets: dict[str, list[set[str]]] = defaultdict(list)
    for ppikey, terms in pair_terms.items():
        parts = ppikey.split("_HUMAN_")
        if len(parts) != 2:
            continue
        a, b = parts[0] + "_HUMAN", parts[1]
        protein_pairsets[a].append(terms)
        protein_pairsets[b].append(terms)
    n_derived = 0
    for prot, sets_list in protein_pairsets.items():
        if len(sets_list) < 3:
            continue
        core = set.intersection(*sets_list)
        if core:
            protein_go[prot].update((tt, term) for term in core)
            n_derived += 1
    print(f"  {tt}: derived terms for {n_derived:,} proteins")
ann_con.close()

print("STEP 2c: Building protein-level KEGG pathway sets from KEGG's own REST API ...")
gene_to_pathways = defaultdict(set)
with open(KEGG_PATHWAY_LINKS) as f:
    for line in f:
        gene, pathway = line.rstrip("\n").split("\t")
        gene_to_pathways[gene].add(pathway.removeprefix("path:"))
gene_to_acc = defaultdict(list)
with open(KEGG_UNIPROT_CONV) as f:
    for line in f:
        gene, acc = line.rstrip("\n").split("\t")
        gene_to_acc[gene].append(acc.removeprefix("up:"))
n_kegg = 0
for gene, pathways in gene_to_pathways.items():
    if not pathways:
        continue
    for acc in gene_to_acc.get(gene, []):
        mnem = acc_to_mnem.get(acc)
        if mnem:
            protein_go[mnem].update(("KEGG", p) for p in pathways)
            n_kegg += 1
print(f"  KEGG: added real pathway terms for {n_kegg:,} protein records")
print(f"  protein_go covers {len(protein_go):,} proteins across {len(ALL_CLASSES)} term types")

term_stats_protein = defaultdict(int)
for terms in protein_go.values():
    for tt_term in terms:
        term_stats_protein[tt_term] += 1
term_stats_protein = dict(term_stats_protein)
N_BG_PROTEIN = len(protein_go)

_term_to_id = {t: i for i, t in enumerate(term_stats_protein.keys())}
_id_to_term = list(term_stats_protein.keys())
_K_ARR = np.array([term_stats_protein[t] for t in _id_to_term], dtype=np.int64)
_protein_term_ids = {
    p: np.array([_term_to_id[t] for t in terms if t in _term_to_id], dtype=np.int64)
    for p, terms in protein_go.items()
}

print("STEP 3: Loading MAPPIE latent index ...")
z = np.load(LATENT_FILE, allow_pickle=True)
keys = list(np.asarray(z["keys"], dtype=str))
lat_n = z["vecs"].astype(np.float32)
key_to_i = {k: i for i, k in enumerate(keys)}
n_all = len(keys)
pos_to_prots = []
for k_ in keys:
    parts = k_.split("_HUMAN_")
    pos_to_prots.append((parts[0] + "_HUMAN", parts[1]) if len(parts) == 2 else (None, None))
print(f"  {n_all:,} MAPPIE PPIs")

print("STEP 3b: Building combined (multiplied) ESM-2 vectors, no AE ...")
conn = sqlite3.connect(f"file:{EMB_DB}?mode=ro", uri=True)
prot_emb = {}
for pid, blob in conn.execute("SELECT protein_id, embedding FROM embeddings"):
    prot_emb[pid] = np.frombuffer(blob, dtype=np.float32).copy()
conn.close()
esm_vecs = np.zeros((n_all, 1280), dtype=np.float32)
esm_valid = np.zeros(n_all, dtype=bool)
for i, (a, b) in enumerate(pos_to_prots):
    if a in prot_emb and b in prot_emb:
        v = prot_emb[a] * prot_emb[b]
        nrm = np.linalg.norm(v)
        if nrm > 0:
            esm_vecs[i] = v / nrm
            esm_valid[i] = True
print(f"  {esm_valid.sum():,} valid combined ESM-2 PPI vectors")


def fast_enrichment_protein(proteins_):
    arrs = [_protein_term_ids[p] for p in proteins_ if p in _protein_term_ids]
    if not arrs:
        return set()
    all_ids = np.concatenate(arrs)
    if all_ids.size == 0:
        return set()
    term_ids, xs = np.unique(all_ids, return_counts=True)
    Ks = _K_ARR[term_ids]
    valid = Ks > 0
    if not valid.any():
        return set()
    n = len(arrs)
    pvals = np.ones(len(term_ids))
    pvals[valid] = hypergeom.sf(xs[valid] - 1, N_BG_PROTEIN, Ks[valid], n)
    m = valid.sum()
    order = np.argsort(pvals[valid])
    sorted_p = pvals[valid][order]
    ranks = np.arange(1, m + 1)
    padj_sorted = np.minimum.accumulate((sorted_p * m / ranks)[::-1])[::-1]
    padj = np.ones(len(term_ids))
    padj[np.where(valid)[0][order]] = padj_sorted
    sig_ids = term_ids[valid & (padj < PADJ)]
    return {_id_to_term[i] for i in sig_ids}


def overall_recall(sig, own_ids):
    if not own_ids:
        return float("nan")
    return len(own_ids & sig) / len(own_ids)


def recall_by_class(sig, own_by_class):
    out = {}
    for cls, own_ids in own_by_class.items():
        if not own_ids:
            continue
        sig_ids = {t for tt, t in sig if tt == cls}
        out[cls] = len(own_ids & sig_ids) / len(own_ids)
    return out


print("STEP 4: Sampling queries with >=1 annotated term ...")
qualify_keys = []
for k_, (a, b) in zip(keys, pos_to_prots):
    if a is None or not esm_valid[key_to_i[k_]]:
        continue
    if (protein_go.get(a, set()) | protein_go.get(b, set())):
        qualify_keys.append(k_)
print(f"  {len(qualify_keys):,} of {n_all:,} MAPPIE PPIs have >=1 annotated term and valid ESM-2 vector")
sample_idx = RNG.choice(len(qualify_keys), min(N_SAMPLE, len(qualify_keys)), replace=False)
selected_keys = [qualify_keys[i] for i in sample_idx]

max_k = max(K_VALUES)
print(f"STEP 5: Full per-query computation for {len(selected_keys):,} queries, k in {K_VALUES}, MAPPIE + combined ESM-2 (no AE) ...")
records = []
t_loop = time.time()
for qi, tkey in enumerate(selected_keys):
    parts = tkey.split("_HUMAN_")
    a, b = parts[0] + "_HUMAN", parts[1]
    self_pos = key_to_i[tkey]
    own = protein_go.get(a, set()) | protein_go.get(b, set())
    if not own:
        continue
    own_ids = own  # keep (term_type, term) tuples -- matches fast_enrichment_protein's sig format
    own_by_class = defaultdict(set)
    for tt, go_id in own:
        own_by_class[tt].add(go_id)

    rec = {"ppikey": tkey}
    for arm_name, vecs in [("mappie", lat_n), ("esm", esm_vecs)]:
        qv = vecs[self_pos]
        sims = vecs @ qv
        sims[self_pos] = -np.inf
        top_idx = np.argpartition(-sims, max_k)[:max_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        sel_proteins = set()
        prev_k = 0
        for k in K_VALUES:
            for j in top_idx[prev_k:k]:
                pa, pb = pos_to_prots[j]
                if pa == a and pb == b:
                    continue
                sel_proteins.add(pa)
                sel_proteins.add(pb)
            prev_k = k
            sig = fast_enrichment_protein(sel_proteins)
            rec[f"{arm_name}_overall_k{k}"] = overall_recall(sig, own_ids)
            for cls, val in recall_by_class(sig, own_by_class).items():
                rec[f"{arm_name}_{cls}_k{k}"] = val

    records.append(rec)
    if qi % 200 == 0:
        elapsed = time.time() - t_loop
        rate = (qi + 1) / elapsed if elapsed > 0 else 0
        print(f"  {qi:,}/{len(selected_keys):,}  ({elapsed:.0f}s elapsed, {rate:.2f} q/s)")

res_df = pd.DataFrame(records)
print(f"\nLoop done: {len(selected_keys):,} queries x {len(K_VALUES)} k-values x 2 arms, {time.time()-t_loop:.0f}s total")
raw_csv = OUT_DIR / f"self_recall_by_k_by_cat_raw{SUFFIX}.csv"
res_df.to_csv(raw_csv, index=False)
print(f"Saved: {raw_csv}")

PLOT_CATS = ["overall"] + ALL_CLASSES
summary_rows = []
for k in K_VALUES:
    row = {"k": k}
    for arm_name in ["mappie", "esm"]:
        for cls in PLOT_CATS:
            col = f"{arm_name}_{cls}_k{k}"
            row[f"{arm_name}_{cls}"] = res_df[col].mean() if col in res_df.columns else float("nan")
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows).set_index("k")
print(summary_df)
summary_csv = OUT_DIR / f"self_recall_by_k_by_cat_summary{SUFFIX}.csv"
summary_df.to_csv(summary_csv)
print(f"Saved: {summary_csv}")

print(f"\nTotal script time: {time.time()-t_start:.1f}s")
print("Done.")
