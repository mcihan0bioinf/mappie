#!/usr/bin/env python3
"""
degree_bins_no_homologs.py

Redo of the degree-stratified recall analysis (degree_bins_fixed_edges.py)
but with sequence homologs of the query proteins EXCLUDED from the
candidate pool for ALL FOUR methods, not just SeqID:

  HIPPIE       -- homologs of A or B are stripped out of the first-degree
                  neighbourhood before computing enrichment.
  MAPPIE       -- homologs excluded from the latent-space nearest-neighbour
                  ranking before the top-k unique proteins are taken.
  ESM-2 alone  -- same exclusion applied to the raw ESM-2 cosine ranking.
  SeqID        -- same exclusion (on top of already being the seq-identity
                  method itself -- this removes the trivial "we ranked by
                  identity, of course the top hits are near-duplicates of
                  the query" case that inflates SeqID's own numbers).

Homology cutoff: 30% sequence identity (Rost, B. 1999, "Twilight zone of
protein sequence alignments", Protein Engineering 12(2):85-94) -- the
standard, literature-supported threshold above which two sequences are
confidently homologous regardless of alignment length. Identity is taken
from the same DEFAULT-parameter MMseqs2 all-vs-all matrix used for the
"_default" SeqID variant (seq_identity_default.tsv).

Because homolog-exclusion shrinks HIPPIE's own first-degree neighbourhood
too, the per-query BUDGET k is redefined as the size of the
homolog-filtered HIPPIE neighbourhood -- this keeps the "same number of
unique proteins across all four methods" design intact, just on the
homolog-free population. Queries whose filtered HIPPIE neighbourhood
drops below MIN_NBRS_FILTERED are dropped.

Two query-selection modes:
  default   -- reuses the same 2,156 cached queries/bins as
               degree_bins_fixed_edges.py, for direct comparability.
  --finebins -- draws a fresh sample against finer degree bins
               (1-10, 10-20, ..., 90-100, 100+) and additionally records
               the per-bin standard deviation of recall, not just the mean.

Usage: degree_bins_no_homologs.py [CUTOFF] [--finebins] [N_PER_BIN]
  CUTOFF defaults to 30.0 (%identity); N_PER_BIN (finebins only) defaults to 500
"""
import sys
import time
import sqlite3
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.sparse import coo_matrix
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

FINEBINS = "--finebins" in sys.argv
_posargs = [a for a in sys.argv[1:] if a != "--finebins"]
CUTOFF = float(_posargs[0]) if len(_posargs) > 0 else 30.0
N_PER_BIN = int(_posargs[1]) if len(_posargs) > 1 else 500
RNG = np.random.default_rng(42)

BASE_SUFFIX = "_n500_mappie_universe_strict_ht"
if FINEBINS:
    BIN_EDGES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100000]
    BIN_LABELS = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]}" for i in range(len(BIN_EDGES) - 1)]
    NEW_SUFFIX = f"_n{N_PER_BIN}_mappie_universe_strict_ht_nohomolog{int(CUTOFF)}_finebins"
else:
    NEW_SUFFIX = f"{BASE_SUFFIX}_nohomolog{int(CUTOFF)}"
MIN_NBRS_FILTERED = 2  # drop only if k<2 (0 or 1 homolog-free HIPPIE partners); k=2 is kept
PADJ = 0.05
GO_CLASSES = ["GO_BP", "GO_MF", "GO_CC"]
DB_DERIVED_TYPES = ["REACTOME", "PFAM", "INTERPRO", "RHEA"]
EXTRA_TYPES = ["KEGG"] + DB_DERIVED_TYPES
ALL_CLASSES = GO_CLASSES + EXTRA_TYPES
ASPECT_TO_TYPE = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}
GO_EVIDENCE_TIER = "strict_ht"
EVIDENCE_TIERS = {
    "super_strict": {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},
    "strict_ht":    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"},
}
STRICT_EVIDENCE_CODES = EVIDENCE_TIERS[GO_EVIDENCE_TIER]

t_start = time.time()
print(f"FINEBINS={FINEBINS}  CUTOFF={CUTOFF}%  NEW_SUFFIX={NEW_SUFFIX!r}")

print("STEP 1: UniProt accession -> mnemonic map")
uni = pd.read_csv(UNIPROT_TSV, sep="\t", usecols=["Entry", "Entry Name"])
acc_to_mnem = dict(zip(uni["Entry"], uni["Entry Name"]))

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

print("STEP 2b: Deriving protein-level REACTOME/PFAM/INTERPRO/RHEA by pair-intersection ...")
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
print(f"  protein_go covers {len(protein_go):,} proteins across {len(ALL_CLASSES)} term types (pre-MAPPIE-universe restriction)")

print("STEP 2d: Restricting enrichment background to MAPPIE's embedded protein universe ...")
emb_con = sqlite3.connect(f"file:{EMB_DB}?mode=ro", uri=True)
mappie_proteins = set(r[0] for r in emb_con.execute("SELECT protein_id FROM embeddings"))
emb_con.close()
before_n = len(protein_go)
protein_go = defaultdict(set, {p: terms for p, terms in protein_go.items() if p in mappie_proteins})
print(f"  dropped {before_n - len(protein_go):,} GO/annotated proteins not in MAPPIE's {len(mappie_proteins):,}-protein "
      f"embedded universe -- background is now exactly MAPPIE's referenceable proteins ({len(protein_go):,})")

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

print("STEP 3: Loading HIPPIE network (HIPPIE_MODE='mappie_universe') ...")
hippie_raw = pd.read_csv(HIPPIE_FULL, sep="\t", header=None,
                          names=["prot1", "id1", "prot2", "id2", "score", "info"])
hippie_raw = hippie_raw.dropna(subset=["prot1", "prot2"])
hippie_raw = hippie_raw[hippie_raw["prot1"].apply(lambda x: isinstance(x, str)) &
                         hippie_raw["prot2"].apply(lambda x: isinstance(x, str))]
hippie_raw = hippie_raw[hippie_raw["prot1"].isin(mappie_proteins) & hippie_raw["prot2"].isin(mappie_proteins)]
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

print(f"STEP 6: Loading DEFAULT-parameter MMseqs2 identity matrix (homolog cutoff = {CUTOFF}%) ...")
id_df = pd.read_csv(IDENTITY_TSV, sep="\t", header=None, names=["query", "target", "pident"])
proteins = sorted(set(id_df["query"]) | set(id_df["target"]))
prot_to_idx = {p: i for i, p in enumerate(proteins)}
n_prot = len(proteins)
rows_ = id_df["query"].map(prot_to_idx).to_numpy()
cols_ = id_df["target"].map(prot_to_idx).to_numpy()
vals_ = id_df["pident"].to_numpy(dtype=np.float32)
identity_mat = coo_matrix((vals_, (rows_, cols_)), shape=(n_prot, n_prot)).tocsr()
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
validA = protA_idx >= 0
validB = protB_idx >= 0
print(f"  {n_prot:,} proteins, {seqid_valid.sum():,} PPIs with both proteins covered")

if FINEBINS:
    print("STEP 7: Fresh sample against finer degree bins -- HIPPIE degree (p_h) for a large candidate pool ...")
    ppi_to_terms: dict[str, set] = {}
    for k_, (a, b) in zip(keys, pos_to_prots):
        if a is None:
            continue
        terms = protein_go.get(a, set()) | protein_go.get(b, set())
        if terms:
            ppi_to_terms[k_] = terms
    qualify_keys = list(ppi_to_terms.keys())
    print(f"  {len(qualify_keys):,} of {n_all:,} MAPPIE PPIs have >=1 annotated term")

    CANDIDATE_POOL_SIZE = len(qualify_keys)  # min bin edge is 1 (<10), so use the full population
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
    bin_order = BIN_LABELS
else:
    print("STEP 7: Loading cached query set (same queries/bins as degree_bins_fixed_edges.py) ...")
    base_csv = OUT_DIR / f"degree_bins_raw{BASE_SUFFIX}.csv"
    base_df = pd.read_csv(base_csv, index_col=0)
    bin_of_key = dict(zip(base_df.index, base_df["bin"]))
    selected_keys = list(base_df.index)
    bin_order = list(dict.fromkeys(base_df["bin"]))
    print(f"  {len(selected_keys):,} cached queries loaded from {base_csv}")


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


print(f"\nSTEP 8: Full per-query computation for {len(selected_keys):,} queries, "
      f"homologs (>= {CUTOFF}% identity to A or B) excluded from ALL FOUR arms ...")
records = []
dropped = 0
dropped_no_homology_data = 0
t_loop = time.time()
for qi, tkey in enumerate(selected_keys):
    parts = tkey.split("_HUMAN_")
    a, b = parts[0] + "_HUMAN", parts[1]
    self_pos = key_to_i.get(tkey)
    if self_pos is None:
        dropped += 1
        continue
    own = protein_go.get(a, set()) | protein_go.get(b, set())
    if not own:
        dropped += 1
        continue
    own_by_class = defaultdict(set)
    for tt, go_id in own:
        own_by_class[tt].add(go_id)

    idxA = prot_to_idx.get(a)
    idxB = prot_to_idx.get(b)
    if idxA is None or idxB is None:
        dropped_no_homology_data += 1
        continue
    rowA = get_id_row(idxA)
    rowB = get_id_row(idxB)
    homolog_mask = (rowA >= CUTOFF) | (rowB >= CUTOFF)  # over identity-vocab proteins; includes A,B themselves

    # HIPPIE arm: strip homologs out of the real first-degree neighbourhood.
    h_partners_all = one_hop_partners(a, b)
    h_partners_filtered = {p for p in h_partners_all
                            if (prot_to_idx.get(p) is None) or (not homolog_mask[prot_to_idx[p]])}
    k = len(h_partners_filtered)
    if k < MIN_NBRS_FILTERED:
        dropped += 1
        continue

    # Vectorized homolog exclusion mask over the full MAPPIE PPI universe --
    # a candidate PPI position is excluded if EITHER of its two proteins is
    # a homolog of query-protein A or B (so no homolog protein can sneak
    # into the selected set via its non-homolog partner).
    excl_by_ppi = np.zeros(n_all, dtype=bool)
    excl_by_ppi[validA] |= homolog_mask[protA_safe[validA]]
    excl_by_ppi[validB] |= homolog_mask[protB_safe[validB]]
    excl_by_ppi[self_pos] = True

    # MAPPIE arm
    sims = lat_n @ lat_n[self_pos]
    sims[excl_by_ppi] = -np.inf
    mappie_order = np.argsort(-sims)

    # ESM-2-alone arm
    if esm_valid[self_pos]:
        esims = esm_vecs @ esm_vecs[self_pos]
        esims[~esm_valid] = -np.inf
        esims[excl_by_ppi] = -np.inf
        esm_order = np.argsort(-esims)
    else:
        esm_order = np.array([], dtype=int)

    # SeqID arm (default MMseqs2 identity, sum-of-best-pairing score)
    score = np.maximum(rowA[protA_safe] + rowB[protB_safe], rowA[protB_safe] + rowB[protA_safe])
    score[~seqid_valid] = -1.0
    score[excl_by_ppi] = -np.inf
    seqid_order = np.argsort(-score)

    if len(mappie_order) < k or len(esm_order) < k or len(seqid_order) < k:
        dropped += 1
        continue

    sig_h, _ = fast_enrichment_protein(h_partners_filtered)
    rec = {"ppikey": tkey, "bin": bin_of_key[tkey], "k": k}
    for cls, val in recall_by_class(sig_h, own_by_class).items():
        rec[f"hippie_{cls}"] = val

    for arm_name, ranked in [("mappie", mappie_order), ("esm", esm_order), ("seqid", seqid_order)]:
        sel = protein_budgeted_ranked(ranked, k)
        sel_proteins = set()
        for j in sel:
            pa, pb = pos_to_prots[j]
            sel_proteins.add(pa); sel_proteins.add(pb)
        sig, _ = fast_enrichment_protein(sel_proteins)
        for cls, val in recall_by_class(sig, own_by_class).items():
            rec[f"{arm_name}_{cls}"] = val

    records.append(rec)
    step = 200 if FINEBINS else 100
    if qi % step == 0:
        elapsed = time.time() - t_loop
        rate = (qi + 1) / elapsed if elapsed > 0 else 0
        print(f"  {qi:,}/{len(selected_keys):,}  ({elapsed:.0f}s elapsed, {rate:.2f} q/s)")

res_df = pd.DataFrame(records)
print(f"\nLoop done: {len(res_df):,} kept, {dropped:,} dropped (k<{MIN_NBRS_FILTERED} or ranking too short), "
      f"{dropped_no_homology_data:,} dropped (no identity data)  ({time.time()-t_loop:.0f}s total)")
raw_csv = OUT_DIR / f"degree_bins_raw{NEW_SUFFIX}.csv"
res_df.to_csv(raw_csv, index=False)
print(f"Saved: {raw_csv}")

summary_rows = []
for b in bin_order:
    sub = res_df[res_df["bin"] == b]
    for cls in ALL_CLASSES + ["pooled", "GO_ALL"]:
        row = {"bin": b, "n": len(sub), "mean_k": sub["k"].mean() if len(sub) else float("nan"), "category": cls}
        for label, prefix in [("HIPPIE", "hippie"), ("MAPPIE", "mappie"), ("ESM-2 alone", "esm"), ("SeqID", "seqid")]:
            col = f"{prefix}_{cls}"
            if col in sub.columns and len(sub):
                row[label] = sub[col].mean()
                if FINEBINS:
                    row[f"{label}_std"] = sub[col].std()
            else:
                row[label] = float("nan")
                if FINEBINS:
                    row[f"{label}_std"] = float("nan")
        row["MAPPIE - HIPPIE"] = row.get("MAPPIE", float("nan")) - row.get("HIPPIE", float("nan"))
        summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / f"degree_bins_summary{NEW_SUFFIX}.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved: {summary_csv}")
print(summary_df[summary_df["category"] == "GO_ALL"].to_string(index=False))

print(f"\nTotal script time: {time.time()-t_start:.1f}s")
print("Done.")
