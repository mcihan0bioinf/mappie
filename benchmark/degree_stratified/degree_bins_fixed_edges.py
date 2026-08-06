#!/usr/bin/env python3
"""Degree-stratified recall (Fig. 3A): bins 1-10, 10-25, 25-75, 75-100, >100
partners, up to 500 queries per bin. Budget per query is its own HIPPIE
degree p_h, one evaluation each (no k sweep).

Run once per HIPPIE_MODE:
  mappie_universe     -- HIPPIE restricted to proteins MAPPIE embeds
  mappie_interactions -- HIPPIE restricted to score >= 0.64

Usage: degree_bins_fixed_edges.py <HIPPIE_MODE> <N_PER_BIN>
  e.g. degree_bins_fixed_edges.py mappie_universe 500
"""

import sys
import time
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.sparse import coo_matrix
import warnings
import os
warnings.filterwarnings("ignore")

MAPPIE_ROOT  = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE  = MAPPIE_ROOT / "data_processed/latent_index.npz"
EMB_DB       = MAPPIE_ROOT / "data_processed/protein_embeddings.db"
UNIPROT_TSV  = Path(os.environ.get("UNIPROT_PROTEOME_TSV", "data/uniprot/uniprotkb_proteome_UP000005640.tsv"))
GAF_FILE     = Path(os.environ.get("GOA_HUMAN_GAF", "data/go/goa_human.gaf"))
HIPPIE_FULL  = Path(os.environ.get("HIPPIE_FULL_TXT", "data/hippie/hippie_current.txt"))
IDENTITY_TSV = Path(__file__).parent.parent / "sequence_identity_baseline/results/seq_identity_default.tsv"
ANNOT_DB     = MAPPIE_ROOT / "data_processed/uniprot/ppi_annotation_long_ddi.db"
KEGG_PATHWAY_LINKS = Path(os.environ.get("KEGG_HSA_PATHWAY_LINKS", str(Path(__file__).parent / "annotation_sources/kegg_hsa_pathway_links.txt")))
KEGG_UNIPROT_CONV  = Path(os.environ.get("KEGG_HSA_UNIPROT_CONV", str(Path(__file__).parent / "annotation_sources/kegg_hsa_uniprot_conv.txt")))
OUT_DIR      = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
HIPPIE_MODE = sys.argv[1] if len(sys.argv) > 1 else "mappie_universe"
N_PER_BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 500
GO_EVIDENCE_TIER = sys.argv[3] if len(sys.argv) > 3 else "strict_ht"  # <-- "super_strict" (experimental only) or "strict_ht" (+ high-throughput)
BIN_EDGES = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 10, 25, 75, 100, 100000]   # -> bins 1-10, 10-25, 25-75, 75-100, 100+
BIN_LABELS = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]}" for i in range(len(BIN_EDGES) - 1)]
MIN_NBRS = int(sys.argv[5]) if len(sys.argv) > 5 else 10
EXCLUDE_KEYS_FILE = sys.argv[6] if len(sys.argv) > 6 else None  # optional: file of ppikeys (one per line) to exclude from sampling, so a second tier run doesn't reuse the same PPIs
PADJ = 0.05
GO_CLASSES = ["GO_BP", "GO_MF", "GO_CC"]
DB_DERIVED_TYPES = ["REACTOME", "PFAM", "INTERPRO", "RHEA"]
EXTRA_TYPES = ["KEGG"] + DB_DERIVED_TYPES
ALL_CLASSES = GO_CLASSES + EXTRA_TYPES
ASPECT_TO_TYPE = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}
SUFFIX = f"_n{N_PER_BIN}_{HIPPIE_MODE}_{GO_EVIDENCE_TIER}"

t_start = time.time()
print(f"HIPPIE_MODE={HIPPIE_MODE}  N_PER_BIN={N_PER_BIN}  bins={BIN_LABELS}")

print("STEP 1: UniProt accession -> mnemonic map")
uni = pd.read_csv(UNIPROT_TSV, sep="\t", usecols=["Entry", "Entry Name"])
acc_to_mnem = dict(zip(uni["Entry"], uni["Entry Name"]))

EVIDENCE_TIERS = {
    "super_strict": {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},
    "strict_ht":    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"},
}
STRICT_EVIDENCE_CODES = EVIDENCE_TIERS[GO_EVIDENCE_TIER]
print(f"STEP 2: Parsing GOA GAF (GO_EVIDENCE_TIER={GO_EVIDENCE_TIER!r}) ...")
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
print(f"  {len(protein_go):,} proteins with >=1 strict-evidence term")

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

# Vectorized enrichment encoding (dict-counting per query is the bottleneck at scale).
_term_to_id = {t: i for i, t in enumerate(term_stats_protein.keys())}
_id_to_term = list(term_stats_protein.keys())
_K_ARR = np.array([term_stats_protein[t] for t in _id_to_term], dtype=np.int64)
_protein_term_ids = {
    p: np.array([_term_to_id[t] for t in terms if t in _term_to_id], dtype=np.int64)
    for p, terms in protein_go.items()
}

print(f"STEP 3: Loading HIPPIE network (HIPPIE_MODE={HIPPIE_MODE!r}) ...")
emb_con = sqlite3.connect(f"file:{EMB_DB}?mode=ro", uri=True)
mappie_proteins = set(r[0] for r in emb_con.execute("SELECT protein_id FROM embeddings"))
emb_con.close()
hippie_raw = pd.read_csv(HIPPIE_FULL, sep="\t", header=None,
                          names=["prot1", "id1", "prot2", "id2", "score", "info"])
hippie_raw = hippie_raw.dropna(subset=["prot1", "prot2"])
hippie_raw = hippie_raw[hippie_raw["prot1"].apply(lambda x: isinstance(x, str)) &
                         hippie_raw["prot2"].apply(lambda x: isinstance(x, str))]
if HIPPIE_MODE == "mappie_universe":
    hippie_raw = hippie_raw[hippie_raw["prot1"].isin(mappie_proteins) & hippie_raw["prot2"].isin(mappie_proteins)]
elif HIPPIE_MODE == "mappie_interactions":
    hippie_raw = hippie_raw[hippie_raw["score"] >= 0.64]
elif HIPPIE_MODE != "full":
    raise ValueError(f"Unknown HIPPIE_MODE: {HIPPIE_MODE!r}")
hippie_adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
for p1, p2, sc in zip(hippie_raw["prot1"], hippie_raw["prot2"], hippie_raw["score"]):
    if p1 == p2:
        continue
    hippie_adj[p1].append((p2, float(sc)))
    hippie_adj[p2].append((p1, float(sc)))
print(f"  {len(hippie_raw):,} rows, {len(hippie_adj):,} proteins")


def one_hop_partners(a, b):
    partners = set()
    for p, _ in hippie_adj.get(a, []):
        if p == b or p not in protein_go:
            continue
        partners.add(p)
    for p, _ in hippie_adj.get(b, []):
        if p == a or p not in protein_go:
            continue
        partners.add(p)
    return partners


print("STEP 4: Loading MAPPIE latent index ...")
z = np.load(LATENT_FILE, allow_pickle=True)
keys = list(np.asarray(z["keys"], dtype=str))
lat_n = z["vecs"].astype(np.float32)
key_to_i = {k: i for i, k in enumerate(keys)}
n_all = len(keys)
pos_to_prots = []
for k in keys:
    parts = k.split("_HUMAN_")
    pos_to_prots.append((parts[0] + "_HUMAN", parts[1]) if len(parts) == 2 else (None, None))
POS_A = np.array([p[0] for p in pos_to_prots], dtype=object)
POS_B = np.array([p[1] for p in pos_to_prots], dtype=object)
print(f"  {n_all:,} MAPPIE PPIs")

print("STEP 5: Building ESM-2-alone vectors ...")
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
print(f"  {esm_valid.sum():,} valid ESM-2-alone PPI vectors")

print("STEP 6: Loading sequence-identity matrix ...")
id_df = pd.read_csv(IDENTITY_TSV, sep="\t", header=None, names=["query", "target", "pident"])
proteins = sorted(set(id_df["query"]) | set(id_df["target"]))
prot_to_idx = {p: i for i, p in enumerate(proteins)}
n_prot = len(proteins)
rows = id_df["query"].map(prot_to_idx).to_numpy()
cols = id_df["target"].map(prot_to_idx).to_numpy()
vals = id_df["pident"].to_numpy(dtype=np.float32)
identity_mat = coo_matrix((vals, (rows, cols)), shape=(n_prot, n_prot)).tocsr()
row_cache = {}
def get_id_row(idx):
    if idx not in row_cache:
        row_cache[idx] = identity_mat.getrow(idx).toarray().ravel()
    return row_cache[idx]
protA_idx = np.array([prot_to_idx.get(a, -1) if a else -1 for a, b in pos_to_prots], dtype=np.int64)
protB_idx = np.array([prot_to_idx.get(b, -1) if b else -1 for a, b in pos_to_prots], dtype=np.int64)
seqid_valid = (protA_idx >= 0) & (protB_idx >= 0)
protA_safe = np.clip(protA_idx, 0, n_prot - 1)
protB_safe = np.clip(protB_idx, 0, n_prot - 1)
print(f"  {n_prot:,} proteins, {seqid_valid.sum():,} PPIs with both proteins covered")

print("STEP 7: Cheap pre-analysis -- HIPPIE degree (p_h) for a large candidate pool ...")
ppi_to_terms: dict[str, set] = {}
for k_, (a, b) in zip(keys, pos_to_prots):
    if a is None:
        continue
    terms = protein_go.get(a, set()) | protein_go.get(b, set())
    if terms:
        ppi_to_terms[k_] = terms
qualify_keys = list(ppi_to_terms.keys())
print(f"  {len(qualify_keys):,} of {n_all:,} MAPPIE PPIs have >=1 annotated term")

if EXCLUDE_KEYS_FILE and Path(EXCLUDE_KEYS_FILE).exists():
    with open(EXCLUDE_KEYS_FILE) as f:
        exclude_keys = set(line.strip() for line in f if line.strip())
    before = len(qualify_keys)
    qualify_keys = [k for k in qualify_keys if k not in exclude_keys]
    print(f"  excluded {before - len(qualify_keys):,} PPIs already used in a prior run ({EXCLUDE_KEYS_FILE})")

CANDIDATE_POOL_SIZE = len(qualify_keys) if min(BIN_EDGES) < 10 else min(40000, len(qualify_keys))
cand_idx = RNG.choice(len(qualify_keys), CANDIDATE_POOL_SIZE, replace=False)
candidate_keys = [qualify_keys[i] for i in cand_idx]

p_h_cheap = {}
for tkey in candidate_keys:
    parts = tkey.split("_HUMAN_")
    if len(parts) != 2:
        continue
    a, b = parts[0] + "_HUMAN", parts[1]
    p_h_cheap[tkey] = len(one_hop_partners(a, b))
p_h_series = pd.Series(p_h_cheap)

selected_keys = []
bin_of_key = {}
for i, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
    eligible = p_h_series[(p_h_series >= lo) & (p_h_series < hi)].index.tolist()
    RNG.shuffle(eligible)
    chosen = eligible[:N_PER_BIN]
    print(f"  bin [{lo},{hi}): {len(eligible):,} eligible in pool, selected {len(chosen):,}")
    for tkey in chosen:
        bin_of_key[tkey] = BIN_LABELS[i]
    selected_keys.extend(chosen)
print(f"  {len(selected_keys):,} total queries selected across {len(BIN_LABELS)} bins")

def bh_adjust_vec(pvals):
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out


def fast_enrichment_protein(proteins_):
    arrs = [_protein_term_ids[p] for p in proteins_ if p in _protein_term_ids]
    n = len(arrs)
    if n == 0:
        return set(), n
    all_ids = np.concatenate(arrs) if arrs else np.array([], dtype=np.int64)
    if all_ids.size == 0:
        return set(), n
    term_ids, xs = np.unique(all_ids, return_counts=True)
    Ks = _K_ARR[term_ids]
    valid = Ks > 0
    if not valid.any():
        return set(), n
    pvals = np.ones(len(term_ids))
    pvals[valid] = hypergeom.sf(xs[valid] - 1, N_BG_PROTEIN, Ks[valid], n)
    padj = bh_adjust_vec(pvals)
    sig_ids = term_ids[valid & (padj < PADJ)]
    return {_id_to_term[i] for i in sig_ids}, n


def recall_by_class(sig, own_by_class):
    out = {}
    pooled_hit = 0
    pooled_total = 0
    go_hit = 0
    go_total = 0
    for cls, own_ids in own_by_class.items():
        if not own_ids:
            continue
        sig_ids = {t for tt, t in sig if tt == cls}
        hit = len(own_ids & sig_ids)
        out[cls] = hit / len(own_ids)
        pooled_hit += hit
        pooled_total += len(own_ids)
        if cls in GO_CLASSES:
            go_hit += hit
            go_total += len(own_ids)
    out["pooled"] = pooled_hit / pooled_total if pooled_total > 0 else float("nan")
    out["GO_ALL"] = go_hit / go_total if go_total > 0 else float("nan")
    return out


def embedding_full_order(self_pos, vecs, valid_mask, exclude_prots):
    qv = vecs[self_pos]
    sims = vecs @ qv
    sims[~valid_mask] = -np.inf
    sims[self_pos] = -np.inf
    order = np.argsort(-sims)
    keep = ~(np.isin(POS_A[order], list(exclude_prots)) | np.isin(POS_B[order], list(exclude_prots)))
    return order[keep]


def seqid_full_order(self_pos, idxA, idxB, exclude_prots):
    rowA = get_id_row(idxA)
    rowB = get_id_row(idxB)
    score = np.maximum(rowA[protA_safe] + rowB[protB_safe], rowA[protB_safe] + rowB[protA_safe])
    score[~seqid_valid] = -1.0
    score[self_pos] = -1.0
    order = np.argsort(-score)
    keep = ~(np.isin(POS_A[order], list(exclude_prots)) | np.isin(POS_B[order], list(exclude_prots)))
    return order[keep]


def protein_budgeted_ranked(ranked_positions, target_n):
    seen = set()
    out = []
    for j in ranked_positions:
        if len(seen) >= target_n:
            break
        pa, pb = pos_to_prots[j]
        out.append(j)
        seen.add(pa); seen.add(pb)
    return out


print(f"\nSTEP 8: Full per-query computation for {len(selected_keys):,} queries "
      f"(one evaluation each, budget = own actual p_h) ...")
records = []
dropped = 0
t_loop = time.time()
for qi, tkey in enumerate(selected_keys):
    parts = tkey.split("_HUMAN_")
    a, b = parts[0] + "_HUMAN", parts[1]
    self_pos = key_to_i[tkey]
    exclude = {a, b}
    own = protein_go.get(a, set()) | protein_go.get(b, set())
    if not own:
        dropped += 1
        continue
    own_by_class = defaultdict(set)
    for tt, go_id in own:
        own_by_class[tt].add(go_id)

    h_partners = one_hop_partners(a, b)
    p_h = len(h_partners)
    if p_h < MIN_NBRS:
        dropped += 1
        continue

    mappie_full = embedding_full_order(self_pos, lat_n, np.ones(n_all, dtype=bool), exclude)
    esm_full = embedding_full_order(self_pos, esm_vecs, esm_valid, exclude) if esm_valid[self_pos] else np.array([], dtype=int)
    if not seqid_valid[self_pos]:
        dropped += 1
        continue
    seqid_full = seqid_full_order(self_pos, protA_idx[self_pos], protB_idx[self_pos], exclude)
    if len(mappie_full) < MIN_NBRS or len(esm_full) < MIN_NBRS or len(seqid_full) < MIN_NBRS:
        dropped += 1
        continue

    sig_h, _ = fast_enrichment_protein(h_partners)
    rec = {"ppikey": tkey, "bin": bin_of_key[tkey], "p_h": p_h}
    for cls, val in recall_by_class(sig_h, own_by_class).items():
        rec[f"hippie_{cls}"] = val

    for arm_name, ranked in [("mappie", mappie_full), ("esm", esm_full), ("seqid", seqid_full)]:
        sel = protein_budgeted_ranked(ranked, p_h)
        sel_proteins = set()
        for j in sel:
            pa, pb = pos_to_prots[j]
            sel_proteins.add(pa); sel_proteins.add(pb)
        sig, _ = fast_enrichment_protein(sel_proteins)
        for cls, val in recall_by_class(sig, own_by_class).items():
            rec[f"{arm_name}_{cls}"] = val

    records.append(rec)
    if qi % 100 == 0:
        elapsed = time.time() - t_loop
        rate = (qi + 1) / elapsed if elapsed > 0 else 0
        print(f"  {qi:,}/{len(selected_keys):,}  ({elapsed:.0f}s elapsed, {rate:.2f} q/s)")

res_df = pd.DataFrame(records)
print(f"\nLoop done: {len(res_df):,} kept, {dropped:,} dropped, {time.time()-t_loop:.0f}s total")
raw_csv = OUT_DIR / f"degree_bins_raw{SUFFIX}.csv"
res_df.to_csv(raw_csv, index=False)
print(f"Saved: {raw_csv}")

summary_rows = []
for b in BIN_LABELS:
    sub = res_df[res_df["bin"] == b]
    for cls in ALL_CLASSES + ["pooled", "GO_ALL"]:
        row = {"bin": b, "n": len(sub), "mean_p_h": sub["p_h"].mean() if len(sub) else float("nan"), "category": cls}
        for label, prefix in [("HIPPIE", "hippie"), ("MAPPIE", "mappie"), ("ESM-2 alone", "esm"), ("SeqID", "seqid")]:
            col = f"{prefix}_{cls}"
            row[label] = sub[col].mean() if col in sub.columns and len(sub) else float("nan")
        row["MAPPIE - HIPPIE"] = row.get("MAPPIE", float("nan")) - row.get("HIPPIE", float("nan"))
        summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / f"degree_bins_summary{SUFFIX}.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved: {summary_csv}")
print(summary_df[summary_df["category"] == "GO_ALL"].to_string(index=False))

print(f"\nTotal script time: {time.time()-t_start:.1f}s")
print("Done.")
