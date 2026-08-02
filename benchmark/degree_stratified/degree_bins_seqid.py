#!/usr/bin/env python3
"""Recomputes only the SeqID arm against degree_bins_fixed_edges.py's cached
queries/bins/budgets; HIPPIE/MAPPIE/ESM-2 columns are copied verbatim.

  default  raw %identity, MMseqs2 default params
  strict   raw %identity, +coverage filter and 0.25 min-seq-id floor
  rank     default matrix, ranked by rank-sum instead of raw %identity
  evalue   default matrix, ranked by -log10(E-value) instead of %identity
"""
import sqlite3
import warnings
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, rankdata
from scipy.sparse import coo_matrix

warnings.filterwarnings("ignore")

MAPPIE_ROOT = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE = MAPPIE_ROOT / "data_processed/latent_index.npz"
UNIPROT_TSV = Path(os.environ.get("UNIPROT_PROTEOME_TSV", "data/uniprot/uniprotkb_proteome_UP000005640.tsv"))
GAF_FILE = Path(os.environ.get("GOA_HUMAN_GAF", "data/go/goa_human.gaf"))
ANNOT_DB = MAPPIE_ROOT / "data_processed/uniprot/ppi_annotation_long_ddi.db"
KEGG_PATHWAY_LINKS = Path(os.environ.get("KEGG_HSA_PATHWAY_LINKS", str(Path(__file__).parent / "annotation_sources/kegg_hsa_pathway_links.txt")))
KEGG_UNIPROT_CONV = Path(os.environ.get("KEGG_HSA_UNIPROT_CONV", str(Path(__file__).parent / "annotation_sources/kegg_hsa_uniprot_conv.txt")))
SEQ_BASELINE_DIR = Path(__file__).parent.parent / "sequence_identity_baseline/results"

OUT_DIR = Path(__file__).parent / "results"
BASE_SUFFIX = "_n500_mappie_universe_strict_ht"
PADJ = 0.05
GO_CLASSES = ["GO_BP", "GO_MF", "GO_CC"]
DB_DERIVED_TYPES = ["REACTOME", "PFAM", "INTERPRO", "RHEA"]
EXTRA_TYPES = ["KEGG"] + DB_DERIVED_TYPES
ALL_CLASSES = GO_CLASSES + EXTRA_TYPES
ASPECT_TO_TYPE = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}
EVIDENCE_TIERS = {
    "super_strict": {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},
    "strict_ht":    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"},
}
STRICT_EVIDENCE_CODES = EVIDENCE_TIERS["strict_ht"]
MIN_NBRS = 1

VARIANTS = {
    "default": dict(tsv=SEQ_BASELINE_DIR / "seq_identity_default.tsv", suffix="_default", scoring="pident"),
    "strict":  dict(tsv=SEQ_BASELINE_DIR / "seq_identity_strict_minid.tsv", suffix="_seqidstrict", scoring="pident"),
    "rank":    dict(tsv=SEQ_BASELINE_DIR / "seq_identity_default.tsv", suffix="_defaultrank", scoring="rank"),
    "evalue":  dict(tsv=SEQ_BASELINE_DIR / "seq_identity_default_eval.tsv", suffix="_defaultevalue", scoring="evalue"),
}


def load_protein_go():
    print("Loading UniProt accession -> mnemonic map...")
    uni = pd.read_csv(UNIPROT_TSV, sep="\t", usecols=["Entry", "Entry Name"])
    acc_to_mnem = dict(zip(uni["Entry"], uni["Entry Name"]))

    print("Parsing GOA GAF (evidence tier=strict_ht)...")
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
            tt = ASPECT_TO_TYPE.get(aspect)
            if mnem and tt:
                protein_go[mnem].add((tt, go_id))
    print(f"  {len(protein_go):,} proteins with >=1 strict-evidence GO term")

    print("Deriving protein-level REACTOME/PFAM/INTERPRO/RHEA by pair-intersection...")
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
        for prot, sets_list in protein_pairsets.items():
            if len(sets_list) < 3:
                continue
            core = set.intersection(*sets_list)
            if core:
                protein_go[prot].update((tt, term) for term in core)
    ann_con.close()

    print("Building protein-level KEGG pathway sets from the KEGG REST mapping...")
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
    for gene, pathways in gene_to_pathways.items():
        for acc in gene_to_acc.get(gene, []):
            mnem = acc_to_mnem.get(acc)
            if mnem:
                protein_go[mnem].update(("KEGG", p) for p in pathways)

    return protein_go, acc_to_mnem


def build_term_index(protein_go):
    term_stats = defaultdict(int)
    for terms in protein_go.values():
        for tt_term in terms:
            term_stats[tt_term] += 1
    id_to_term = list(term_stats.keys())
    term_to_id = {t: i for i, t in enumerate(id_to_term)}
    k_arr = np.array([term_stats[t] for t in id_to_term], dtype=np.int64)
    protein_term_ids = {
        p: np.array([term_to_id[t] for t in terms if t in term_to_id], dtype=np.int64)
        for p, terms in protein_go.items()
    }
    return id_to_term, k_arr, protein_term_ids, len(protein_go)


def bh_adjust_vec(pvals):
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(adj, 0, 1)


def load_identity_matrix(tsv_path, scoring, pos_to_prots):
    if scoring == "evalue":
        id_df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "pident", "evalue"])
        vals = -np.log10(np.clip(id_df["evalue"].to_numpy(dtype=np.float64), 1e-300, None)).astype(np.float32)
        vals = np.clip(vals, 0, 300)
    else:
        id_df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "pident"])
        vals = id_df["pident"].to_numpy(dtype=np.float32)

    proteins = sorted(set(id_df["query"]) | set(id_df["target"]))
    prot_to_idx = {p: i for i, p in enumerate(proteins)}
    n_prot = len(proteins)
    rows = id_df["query"].map(prot_to_idx).to_numpy()
    cols = id_df["target"].map(prot_to_idx).to_numpy()
    identity_mat = coo_matrix((vals, (rows, cols)), shape=(n_prot, n_prot)).tocsr()

    protA_idx = np.array([prot_to_idx.get(a, -1) if a else -1 for a, b in pos_to_prots], dtype=np.int64)
    protB_idx = np.array([prot_to_idx.get(b, -1) if b else -1 for a, b in pos_to_prots], dtype=np.int64)
    seqid_valid = (protA_idx >= 0) & (protB_idx >= 0)
    protA_safe = np.clip(protA_idx, 0, n_prot - 1)
    protB_safe = np.clip(protB_idx, 0, n_prot - 1)
    print(f"  {n_prot:,} proteins, {seqid_valid.sum():,} PPIs with both proteins covered, {len(id_df):,} pairwise hits")

    row_cache = {}
    def get_row(idx):
        if idx not in row_cache:
            row_cache[idx] = identity_mat.getrow(idx).toarray().ravel()
        return row_cache[idx]

    return get_row, protA_idx, protB_idx, protA_safe, protB_safe, seqid_valid


def rank_candidates(scoring, self_pos, idxA, idxB, exclude_prots, get_row, protA_safe, protB_safe,
                     seqid_valid, pos_to_prots, POS_A, POS_B):
    rowA, rowB = get_row(idxA), get_row(idxB)
    if scoring == "rank":
        rank_A = rankdata(-rowA, method="average")
        rank_B = rankdata(-rowB, method="average")
        score = np.minimum(rank_A[protA_safe] + rank_B[protB_safe], rank_A[protB_safe] + rank_B[protA_safe])
        score[~seqid_valid] = np.inf
        score[self_pos] = np.inf
        order = np.argsort(score)
    else:
        score = np.maximum(rowA[protA_safe] + rowB[protB_safe], rowA[protB_safe] + rowB[protA_safe])
        score[~seqid_valid] = -1.0
        score[self_pos] = -1.0
        order = np.argsort(-score)
    keep = ~(np.isin(POS_A[order], list(exclude_prots)) | np.isin(POS_B[order], list(exclude_prots)))
    return order[keep]


def protein_budgeted_ranked(ranked_positions, target_n, pos_to_prots):
    seen, out = set(), []
    for j in ranked_positions:
        if len(seen) >= target_n:
            break
        pa, pb = pos_to_prots[j]
        out.append(j)
        seen.add(pa); seen.add(pb)
    return out


def fast_enrichment_protein(proteins_, protein_term_ids, k_arr, id_to_term, n_bg):
    arrs = [protein_term_ids[p] for p in proteins_ if p in protein_term_ids]
    n = len(arrs)
    if n == 0:
        return set(), n
    all_ids = np.concatenate(arrs) if arrs else np.array([], dtype=np.int64)
    if all_ids.size == 0:
        return set(), n
    term_ids, xs = np.unique(all_ids, return_counts=True)
    Ks = k_arr[term_ids]
    valid = Ks > 0
    if not valid.any():
        return set(), n
    pvals = np.ones(len(term_ids))
    pvals[valid] = hypergeom.sf(xs[valid] - 1, n_bg, Ks[valid], n)
    padj = bh_adjust_vec(pvals)
    sig_ids = term_ids[valid & (padj < PADJ)]
    return {id_to_term[i] for i in sig_ids}, n


def recall_by_class(sig, own_by_class):
    out = {}
    pooled_hit = pooled_total = go_hit = go_total = 0
    for cls, own_ids in own_by_class.items():
        if not own_ids:
            continue
        sig_ids = {t for tt, t in sig if tt == cls}
        hit = len(own_ids & sig_ids)
        out[cls] = hit / len(own_ids)
        pooled_hit += hit; pooled_total += len(own_ids)
        if cls in GO_CLASSES:
            go_hit += hit; go_total += len(own_ids)
    out["pooled"] = pooled_hit / pooled_total if pooled_total > 0 else float("nan")
    out["GO_ALL"] = go_hit / go_total if go_total > 0 else float("nan")
    return out


def run_variant(name, spec, protein_go, term_index, keys, key_to_i, pos_to_prots, POS_A, POS_B):
    print(f"\n=== variant: {name} ===")
    id_to_term, k_arr, protein_term_ids, n_bg = term_index
    print(f"Loading identity matrix ({spec['scoring']} scoring): {spec['tsv']}")
    get_row, protA_idx, protB_idx, protA_safe, protB_safe, seqid_valid = load_identity_matrix(
        spec["tsv"], spec["scoring"], pos_to_prots)

    base_csv = OUT_DIR / f"degree_bins_raw{BASE_SUFFIX}.csv"
    base_df = pd.read_csv(base_csv, index_col=0)
    print(f"  {len(base_df):,} cached queries loaded from {base_csv}")

    records, dropped = [], 0
    for qi, (tkey, row) in enumerate(base_df.iterrows()):
        parts = tkey.split("_HUMAN_")
        if len(parts) != 2:
            dropped += 1
            continue
        a, b = parts[0] + "_HUMAN", parts[1]
        self_pos = key_to_i.get(tkey)
        if self_pos is None or not seqid_valid[self_pos]:
            dropped += 1
            continue
        own = protein_go.get(a, set()) | protein_go.get(b, set())
        if not own:
            dropped += 1
            continue
        own_by_class = defaultdict(set)
        for tt, go_id in own:
            own_by_class[tt].add(go_id)

        exclude = {a, b}
        ranked = rank_candidates(spec["scoring"], self_pos, protA_idx[self_pos], protB_idx[self_pos],
                                  exclude, get_row, protA_safe, protB_safe, seqid_valid, pos_to_prots, POS_A, POS_B)
        if len(ranked) < MIN_NBRS:
            dropped += 1
            continue

        sel = protein_budgeted_ranked(ranked, row["p_h"], pos_to_prots)
        sel_proteins = set()
        for j in sel:
            pa, pb = pos_to_prots[j]
            sel_proteins.add(pa); sel_proteins.add(pb)
        sig, _ = fast_enrichment_protein(sel_proteins, protein_term_ids, k_arr, id_to_term, n_bg)

        rec = {"ppikey": tkey, "bin": row["bin"], "p_h": row["p_h"]}
        for cls, val in recall_by_class(sig, own_by_class).items():
            rec[f"seqid_{cls}"] = val
        for col in base_df.columns:
            if col.startswith(("hippie_", "mappie_", "esm_")):
                rec[col] = row[col]
        records.append(rec)
        if qi % 200 == 0:
            print(f"  {qi:,}/{len(base_df):,}")

    res_df = pd.DataFrame(records)
    print(f"Done: {len(res_df):,} kept, {dropped:,} dropped")
    new_suffix = BASE_SUFFIX + spec["suffix"]
    res_df.to_csv(OUT_DIR / f"degree_bins_raw{new_suffix}.csv", index=False)

    bin_order = list(dict.fromkeys(base_df["bin"]))
    summary_rows = []
    for b in bin_order:
        sub = res_df[res_df["bin"] == b]
        for cls in ALL_CLASSES + ["pooled", "GO_ALL"]:
            row_out = {"bin": b, "n": len(sub), "mean_p_h": sub["p_h"].mean() if len(sub) else float("nan"),
                       "category": cls}
            for label, prefix in [("HIPPIE", "hippie"), ("MAPPIE", "mappie"), ("ESM-2 alone", "esm"), ("SeqID", "seqid")]:
                col = f"{prefix}_{cls}"
                row_out[label] = sub[col].mean() if col in sub.columns and len(sub) else float("nan")
            row_out["MAPPIE - HIPPIE"] = row_out.get("MAPPIE", float("nan")) - row_out.get("HIPPIE", float("nan"))
            summary_rows.append(row_out)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / f"degree_bins_summary{new_suffix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")


def main():
    protein_go, acc_to_mnem = load_protein_go()
    term_index = build_term_index(protein_go)

    print("Loading MAPPIE latent index (PPI universe / exclusion bookkeeping)...")
    z = np.load(LATENT_FILE, allow_pickle=True)
    keys = list(np.asarray(z["keys"], dtype=str))
    key_to_i = {k: i for i, k in enumerate(keys)}
    pos_to_prots = []
    for k in keys:
        parts = k.split("_HUMAN_")
        pos_to_prots.append((parts[0] + "_HUMAN", parts[1]) if len(parts) == 2 else (None, None))
    POS_A = np.array([p[0] for p in pos_to_prots], dtype=object)
    POS_B = np.array([p[1] for p in pos_to_prots], dtype=object)
    print(f"  {len(keys):,} MAPPIE PPIs")

    for name, spec in VARIANTS.items():
        run_variant(name, spec, protein_go, term_index, keys, key_to_i, pos_to_prots, POS_A, POS_B)


if __name__ == "__main__":
    main()
