#!/usr/bin/env python3
"""For each PPI (A,B) in STRING ∩ MAPPIE: Network = STRING partners of A/B at
each confidence threshold; k-NN = k nearest MAPPIE neighbours of (A,B).
coverage@k, precision@k, MRR, after excluding neighbours sharing a protein
with (A,B). Also degree-stratified fold enrichment (low/med/high terciles).

Random baseline is exact, not sampled: for query (A,B) with pool size M and
per-protein degree d(p) in the valid pool, expected coverage@k is the mean
over network proteins of [1 - hypergeom.pmf(0, M, d(p), k)].
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
from scipy.stats import hypergeom
import warnings
import os
warnings.filterwarnings("ignore")

LATENT_FILE  = Path(os.environ.get("MAPPIE_LATENT_INDEX", "../mappie/data_processed/latent_index.npz"))
STRING_PAIRS = Path(__file__).parent / "results/string_mappie_pairs.csv"
OUTPUT_DIR   = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else None  # optional: subsample queries for a fast proof-of-concept run
RNG = np.random.default_rng(42)

K_MAX           = 3001   # buffer for shared-protein exclusion + protein-budgeted walk (reaching k=500 unique proteins needs ~700 interactions on average, up to ~2300 in the tail)
K_SWEEP         = [10, 25, 50, 100, 200, 500]
BATCH           = 500
CONF_THRESHOLDS = {"≥900": 900, "≥700": 700, "≥400": 400, "≥150": 150}
DEGREE_BINS     = ["low", "med", "high"]
DEGREE_K        = [10, 100, 500]
PRIMARY_CONF    = "≥900"   # threshold used for degree stratification

print("Loading MAPPIE latent 128d ...")
_z       = np.load(LATENT_FILE, allow_pickle=True)
keys     = list(np.asarray(_z["keys"], dtype=str))
lat_n    = _z["vecs"].astype(np.float32)
key_to_i = {k: i for i, k in enumerate(keys)}
latent_set = set(keys)
n_all    = len(keys)
print(f"  {n_all:,} PPIs × {lat_n.shape[1]}d")

# position → (protA, protB)
pos_to_prots = []
for k in keys:
    parts = k.split("_HUMAN_")
    if len(parts) == 2:
        pos_to_prots.append((parts[0] + "_HUMAN", parts[1]))
    else:
        pos_to_prots.append((None, None))

print("Loading STRING-MAPPIE pairs ...")
str_df = pd.read_csv(STRING_PAIRS)
print(f"  {len(str_df):,} query PPIs in STRING∩MAPPIE")

# protein → {partner: score}
prot_string: dict[str, dict[str, int]] = defaultdict(dict)
for _, row in str_df.iterrows():
    p1, p2, sc = row["prot1"], row["prot2"], int(row["combined_score"])
    prot_string[p1][p2] = sc
    prot_string[p2][p1] = sc

query_keys = list(str_df["ppikey"])
if N_SAMPLE is not None and N_SAMPLE < len(query_keys):
    sample_idx = RNG.choice(len(query_keys), N_SAMPLE, replace=False)
    query_keys = [query_keys[i] for i in sample_idx]
    print(f"  Subsampled to {len(query_keys):,} queries for a fast proof-of-concept run")
n_queries  = len(query_keys)

print("Computing per-protein degree (over all MAPPIE PPIs) and query-degree terciles ...")
deg_counter: Counter = Counter()
for a, b in pos_to_prots:
    if a: deg_counter[a] += 1
    if b: deg_counter[b] += 1

def edge_exists(p: str, q: str) -> bool:
    return f"{p}_{q}" in latent_set or f"{q}_{p}" in latent_set

query_deg = np.full(n_queries, np.nan)
for qi, ppikey in enumerate(query_keys):
    parts = ppikey.split("_HUMAN_")
    if len(parts) != 2:
        continue
    pA, pB = parts[0] + "_HUMAN", parts[1]
    query_deg[qi] = max(deg_counter.get(pA, 0), deg_counter.get(pB, 0))

valid_deg = ~np.isnan(query_deg)
p33, p66 = np.percentile(query_deg[valid_deg], [100 / 3, 200 / 3])

def _deg_bin(d: float) -> str | None:
    if np.isnan(d):
        return None
    if d <= p33:
        return "low"
    if d <= p66:
        return "med"
    return "high"

query_bin = np.array([_deg_bin(d) for d in query_deg], dtype=object)
print(f"  Degree terciles: p33={p33:.1f}, p66={p66:.1f}")
for b in DEGREE_BINS:
    print(f"    {b:>4}: {int((query_bin == b).sum()):,} queries")

print(f"Fitting kNN (k={K_MAX}) ...")
nn = NearestNeighbors(n_neighbors=K_MAX, metric="cosine", algorithm="brute", n_jobs=-1)
nn.fit(lat_n)

print("Computing coverage@k, precision@k, MRR, exact random baseline, degree-stratified fold ...")
cns = list(CONF_THRESHOLDS.keys())

recall_sum    = {cn: {k: 0.0 for k in K_SWEEP} for cn in cns}
recall_count  = {cn: {k: 0   for k in K_SWEEP} for cn in cns}
prec_sum      = {cn: {k: 0.0 for k in K_SWEEP} for cn in cns}
prec_count    = {cn: {k: 0   for k in K_SWEEP} for cn in cns}
rr_lists      = {cn: [] for cn in cns}   # per-query reciprocal rank, 0 if no hit
rand_sum      = {cn: {k: 0.0 for k in K_SWEEP} for cn in cns}
rand_count    = {cn: {k: 0   for k in K_SWEEP} for cn in cns}

# Unfiltered (no shared-protein exclusion) parallel accumulators: a neighbour
# PPI that contains query protein A or B is allowed to count here, so e.g. a
# neighbour (A,C) where C is a true STRING partner of A contributes C to
# coverage. Reported alongside the exclusion-based numbers, not instead of
# them — see METHODS.md §4 for why both views are kept.
recall_sum_u   = {cn: {k: 0.0 for k in K_SWEEP} for cn in cns}
recall_count_u = {cn: {k: 0   for k in K_SWEEP} for cn in cns}
prec_sum_u     = {cn: {k: 0.0 for k in K_SWEEP} for cn in cns}
prec_count_u   = {cn: {k: 0   for k in K_SWEEP} for cn in cns}
rr_lists_u     = {cn: [] for cn in cns}
rand_sum_u     = {cn: {k: 0.0 for k in K_SWEEP} for cn in cns}
rand_count_u   = {cn: {k: 0   for k in K_SWEEP} for cn in cns}

# degree-stratified accumulators, PRIMARY_CONF only (now exact, every query contributes)
deg_cov_sum  = {b: {k: 0.0 for k in DEGREE_K} for b in DEGREE_BINS}
deg_cov_cnt  = {b: {k: 0   for k in DEGREE_K} for b in DEGREE_BINS}
deg_rand_sum = {b: {k: 0.0 for k in DEGREE_K} for b in DEGREE_BINS}
deg_rand_cnt = {b: {k: 0   for k in DEGREE_K} for b in DEGREE_BINS}

for start in range(0, n_queries, BATCH):
    end        = min(start + BATCH, n_queries)
    batch_keys = query_keys[start:end]
    b_rows     = np.array([key_to_i[k] for k in batch_keys])
    _, nbr_idx = nn.kneighbors(lat_n[b_rows])

    for bi, ppikey in enumerate(batch_keys):
        global_qi = start + bi
        self_pos = b_rows[bi]
        parts = ppikey.split("_HUMAN_")
        if len(parts) != 2:
            continue
        pA, pB = parts[0] + "_HUMAN", parts[1]
        query_prots = {pA, pB}
        deg_bin = query_bin[global_qi]

        # STRING network proteins at each threshold (nested: ≥900 ⊆ ≥700 ⊆ ≥400 ⊆ ≥150)
        network: dict[str, set[str]] = {}
        for cn, thresh in CONF_THRESHOLDS.items():
            net = set()
            for prot in [pA, pB]:
                for partner, sc in prot_string.get(prot, {}).items():
                    if sc >= thresh and partner not in (pA, pB):
                        net.add(partner)
            network[cn] = net

        # Exact random baseline: M (valid pool size) and per-protein d(p), computed
        # once per query over the union of all threshold networks (≥150 superset)
        M_query = n_all - (deg_counter.get(pA, 0) + deg_counter.get(pB, 0) - 1)
        all_net_prots = network["≥150"]
        d_valid = {p: max(0, deg_counter.get(p, 0) - edge_exists(p, pA) - edge_exists(p, pB))
                   for p in all_net_prots}

        # Unfiltered random baseline: pool is every PPI except self (no removal
        # of PPIs containing A/B), so M_u = n_all - 1 and d(p) is the raw degree.
        M_query_u = n_all - 1
        d_valid_u = {p: deg_counter.get(p, 0) for p in all_net_prots}

        # Shared-protein exclusion: remove self and any neighbour sharing a protein with query
        filtered_nbrs = []
        unfiltered_nbrs = []
        for j in nbr_idx[bi]:
            if j == self_pos:
                continue
            unfiltered_nbrs.append(j)
            a, b = pos_to_prots[j]
            if a in query_prots or b in query_prots:
                continue
            filtered_nbrs.append(j)

        # Reciprocal rank of first positive neighbour (per threshold)
        for cn in cns:
            net = network[cn]
            if not net:
                continue
            found_rank = None
            for rank, pos in enumerate(filtered_nbrs, start=1):
                a, b = pos_to_prots[pos]
                nbr_prots = set()
                if a: nbr_prots.add(a)
                if b: nbr_prots.add(b)
                if net & nbr_prots:
                    found_rank = rank
                    break
            rr_lists[cn].append((1.0 / found_rank) if found_rank is not None else 0.0)

            # Unfiltered MRR: (A,C)-type neighbours are allowed to contribute C
            found_rank_u = None
            for rank, pos in enumerate(unfiltered_nbrs, start=1):
                a, b = pos_to_prots[pos]
                nbr_prots = set()
                if a: nbr_prots.add(a)
                if b: nbr_prots.add(b)
                if net & nbr_prots:
                    found_rank_u = rank
                    break
            rr_lists_u[cn].append((1.0 / found_rank_u) if found_rank_u is not None else 0.0)

        # Coverage@k and precision@k over filtered neighbours, + exact random baseline.
        # k is a TARGET NUMBER OF UNIQUE PROTEINS, not a fixed interaction count --
        # walk the ranked neighbour list, adding interactions one at a time, until
        # the accumulated unique-protein set reaches k (compensates for overlap:
        # empirically ~40-60% of 2k proteins are unique due to hub repeats).
        knn_prots = set()
        pos_ptr = 0
        for k in K_SWEEP:
            while len(knn_prots) < k and pos_ptr < len(filtered_nbrs):
                pos = filtered_nbrs[pos_ptr]
                a, b = pos_to_prots[pos]
                if a: knn_prots.add(a)
                if b: knn_prots.add(b)
                pos_ptr += 1
            n_interactions_used = pos_ptr

            # ALL interactions consumed so far, for precision denominator
            all_k_nbrs = filtered_nbrs[:n_interactions_used]
            for cn in cns:
                net = network[cn]
                if not net:
                    continue
                # Coverage
                cov_val = len(net & knn_prots) / len(net)
                recall_sum[cn][k]   += cov_val
                recall_count[cn][k] += 1
                # Precision: fraction of the k PPIs with at least one protein in net
                n_pos = sum(1 for pos in all_k_nbrs
                            if (pos_to_prots[pos][0] in net or pos_to_prots[pos][1] in net))
                denom = len(all_k_nbrs)
                if denom > 0:
                    prec_sum[cn][k]   += n_pos / denom
                    prec_count[cn][k] += 1

                # Exact random-baseline coverage: mean over net of P(protein appears).
                # Uses n_interactions_used (actual draws), not the target protein count k.
                d_arr = np.fromiter((d_valid[p] for p in net), dtype=np.float64, count=len(net))
                rand_cov = float(np.nan_to_num(1.0 - hypergeom.pmf(0, M_query, d_arr, n_interactions_used)).mean())
                rand_sum[cn][k]   += rand_cov
                rand_count[cn][k] += 1

                if cn == PRIMARY_CONF and k in DEGREE_K and deg_bin is not None:
                    deg_cov_sum[deg_bin][k]  += cov_val
                    deg_cov_cnt[deg_bin][k]  += 1
                    deg_rand_sum[deg_bin][k] += rand_cov
                    deg_rand_cnt[deg_bin][k] += 1

        # Unfiltered coverage@k / precision@k: (A,C)-type neighbours allowed.
        # Same protein-budgeted walk as the filtered version above.
        knn_prots_u = set()
        pos_ptr_u = 0
        for k in K_SWEEP:
            while len(knn_prots_u) < k and pos_ptr_u < len(unfiltered_nbrs):
                pos = unfiltered_nbrs[pos_ptr_u]
                a, b = pos_to_prots[pos]
                if a: knn_prots_u.add(a)
                if b: knn_prots_u.add(b)
                pos_ptr_u += 1
            n_interactions_used_u = pos_ptr_u

            all_k_nbrs_u = unfiltered_nbrs[:n_interactions_used_u]
            for cn in cns:
                net = network[cn]
                if not net:
                    continue
                cov_val_u = len(net & knn_prots_u) / len(net)
                recall_sum_u[cn][k]   += cov_val_u
                recall_count_u[cn][k] += 1
                n_pos_u = sum(1 for pos in all_k_nbrs_u
                              if (pos_to_prots[pos][0] in net or pos_to_prots[pos][1] in net))
                denom_u = len(all_k_nbrs_u)
                if denom_u > 0:
                    prec_sum_u[cn][k]   += n_pos_u / denom_u
                    prec_count_u[cn][k] += 1

                d_arr_u = np.fromiter((d_valid_u[p] for p in net), dtype=np.float64, count=len(net))
                rand_cov_u = float(np.nan_to_num(1.0 - hypergeom.pmf(0, M_query_u, d_arr_u, n_interactions_used_u)).mean())
                rand_sum_u[cn][k]   += rand_cov_u
                rand_count_u[cn][k] += 1

    if start % 10000 == 0:
        print(f"  {end:,}/{n_queries:,}")

print("\n=== EXCLUSION IMPACT (≥900) ===")
print("Unfiltered = neighbour PPIs containing A or B are allowed to contribute their other")
print("protein to coverage/precision/MRR (e.g. neighbour (A,C) can recover C). Excluded =")
print("the stricter default reported elsewhere in this suite. Both computed exactly, same queries.")
print(f"{'k':>6}  {'excluded':>10}  {'unfiltered':>10}  {'delta':>10}")
for k in K_SWEEP:
    excl_v = recall_sum["≥900"][k] / recall_count["≥900"][k] if recall_count["≥900"][k] > 0 else float("nan")
    unf_v  = recall_sum_u["≥900"][k] / recall_count_u["≥900"][k] if recall_count_u["≥900"][k] > 0 else float("nan")
    delta = unf_v - excl_v
    print(f"{k:>6}  {excl_v:>10.4f}  {unf_v:>10.4f}  {delta:>+10.4f}")

mrr_u_stats = {}
for cn in cns:
    rr_u = np.array(rr_lists_u[cn])
    hit_mask_u = rr_u > 0
    mrr_u_stats[cn] = {
        "n": len(rr_u),
        "hit_fraction": float(hit_mask_u.mean()) if len(rr_u) > 0 else float("nan"),
        "mrr_all": float(rr_u.mean()) if len(rr_u) > 0 else float("nan"),
        "mrr_hit": float(rr_u[hit_mask_u].mean()) if hit_mask_u.any() else 0.0,
    }
print(f"\nUnfiltered MRR (≥900): hit_fraction={mrr_u_stats['≥900']['hit_fraction']:.4f}  "
      f"mrr_all={mrr_u_stats['≥900']['mrr_all']:.4f}  mrr_hit={mrr_u_stats['≥900']['mrr_hit']:.4f}")

print(f"\n=== STRING protein coverage@k ({len(query_keys):,} queries in STRING∩MAPPIE) ===")
print("(random baseline computed exactly via hypergeometric expectation, not sampled)")
print(f"{'k':>6}" + "".join(f"  {cn:>10}" for cn in cns) + f"  {'random≥900':>12}  {'fold':>8}")
for k in K_SWEEP:
    row = f"{k:>6}"
    for cn in cns:
        m = recall_sum[cn][k] / recall_count[cn][k] if recall_count[cn][k] > 0 else float("nan")
        row += f"  {m:>10.4f}"
    r = rand_sum["≥900"][k] / rand_count["≥900"][k] if rand_count["≥900"][k] > 0 else float("nan")
    m900 = recall_sum["≥900"][k] / recall_count["≥900"][k] if recall_count["≥900"][k] > 0 else float("nan")
    fold = m900 / r if r > 0 else float("nan")
    row += f"  {r:>12.4f}  {fold:>7.1f}×"
    print(row)

print(f"\n=== STRING precision@k ({len(query_keys):,} queries) ===")
print(f"{'k':>6}" + "".join(f"  {cn:>10}" for cn in cns))
for k in K_SWEEP:
    row = f"{k:>6}"
    for cn in cns:
        p = prec_sum[cn][k] / prec_count[cn][k] if prec_count[cn][k] > 0 else float("nan")
        row += f"  {p:>10.4f}"
    print(row)

print("\n=== STRING MRR, hit fraction, hit-conditioned MRR ===")
print(f"{'threshold':>12}  {'n':>8}  {'hit_fraction':>13}  {'mrr_all':>10}  {'mrr_hit':>10}")
mrr_stats = {}
for cn in cns:
    rr = np.array(rr_lists[cn])
    hit_mask = rr > 0
    hit_fraction = float(hit_mask.mean()) if len(rr) > 0 else float("nan")
    mrr_all = float(rr.mean()) if len(rr) > 0 else float("nan")
    mrr_hit = float(rr[hit_mask].mean()) if hit_mask.any() else 0.0
    mrr_stats[cn] = {"n": len(rr), "hit_fraction": hit_fraction, "mrr_all": mrr_all, "mrr_hit": mrr_hit}
    print(f"{cn:>12}  {len(rr):>8,}  {hit_fraction:>13.4f}  {mrr_all:>10.4f}  {mrr_hit:>10.4f}")

print(f"\n=== Degree-stratified fold enrichment ({PRIMARY_CONF}), exact random baseline ===")
print(f"{'bin':>6}" + "".join(f"  cov@{k:<4}" for k in DEGREE_K) +
      "".join(f"  rand@{k:<4}" for k in DEGREE_K) + "".join(f"  fold@{k:<4}" for k in DEGREE_K))
deg_table = {}
for b in DEGREE_BINS:
    row = f"{b:>6}"
    deg_table[b] = {}
    for k in DEGREE_K:
        cov = deg_cov_sum[b][k] / deg_cov_cnt[b][k] if deg_cov_cnt[b][k] > 0 else float("nan")
        row += f"  {cov:>8.4f}"
        deg_table[b][k] = {"coverage": cov}
    for k in DEGREE_K:
        rand = deg_rand_sum[b][k] / deg_rand_cnt[b][k] if deg_rand_cnt[b][k] > 0 else float("nan")
        row += f"  {rand:>9.4f}"
        deg_table[b][k]["random"] = rand
    for k in DEGREE_K:
        cov = deg_table[b][k]["coverage"]
        rand = deg_table[b][k]["random"]
        fold = cov / rand if rand and rand > 0 else float("nan")
        row += f"  {fold:>9.2f}×"
        deg_table[b][k]["fold"] = fold
    print(row)

high_fold_500 = deg_table["high"][500]["fold"]
high_fold_10  = deg_table["high"][10]["fold"]
summary_ok = all(deg_table["high"][k]["fold"] > 1 for k in DEGREE_K if not np.isnan(deg_table["high"][k]["fold"]))
print(f"\nSUMMARY: high-degree fold-enrichment stays above 1 at all k ∈ {DEGREE_K}: "
      f"{'YES' if summary_ok else 'NO'} (k=10: {high_fold_10:.2f}×, k=500: {high_fold_500:.2f}×) "
      "→ signal is not purely a hub artefact." if summary_ok else
      f"\nSUMMARY: high-degree fold-enrichment does NOT stay above 1 at all k ∈ {DEGREE_K} "
      f"(k=10: {high_fold_10:.2f}×, k=500: {high_fold_500:.2f}×).")

rows = []
for cn in cns:
    for k in K_SWEEP:
        m = recall_sum[cn][k] / recall_count[cn][k] if recall_count[cn][k] > 0 else float("nan")
        p = prec_sum[cn][k] / prec_count[cn][k] if prec_count[cn][k] > 0 else float("nan")
        r = rand_sum[cn][k] / rand_count[cn][k] if rand_count[cn][k] > 0 else float("nan")
        rows.append({"conf": cn, "k": k,
                     "mean_coverage": round(m, 4),
                     "mean_precision": round(p, 4),
                     "random_coverage_exact": round(r, 4),
                     "n_ppis": recall_count[cn][k]})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "string_recall_curve.csv", index=False)

rows_u = []
for cn in cns:
    for k in K_SWEEP:
        m = recall_sum_u[cn][k] / recall_count_u[cn][k] if recall_count_u[cn][k] > 0 else float("nan")
        p = prec_sum_u[cn][k] / prec_count_u[cn][k] if prec_count_u[cn][k] > 0 else float("nan")
        r = rand_sum_u[cn][k] / rand_count_u[cn][k] if rand_count_u[cn][k] > 0 else float("nan")
        rows_u.append({"conf": cn, "k": k,
                        "mean_coverage": round(m, 4),
                        "mean_precision": round(p, 4),
                        "random_coverage_exact": round(r, 4),
                        "n_ppis": recall_count_u[cn][k]})
pd.DataFrame(rows_u).to_csv(OUTPUT_DIR / "string_recall_curve_unfiltered.csv", index=False)

mrr_rows = [{"conf": cn,
             "n": mrr_stats[cn]["n"],
             "hit_fraction": round(mrr_stats[cn]["hit_fraction"], 4),
             "mrr_all": round(mrr_stats[cn]["mrr_all"], 4),
             "mrr_hit": round(mrr_stats[cn]["mrr_hit"], 4)} for cn in cns]
pd.DataFrame(mrr_rows).to_csv(OUTPUT_DIR / "string_mrr.csv", index=False)

mrr_rows_u = [{"conf": cn,
               "n": mrr_u_stats[cn]["n"],
               "hit_fraction": round(mrr_u_stats[cn]["hit_fraction"], 4),
               "mrr_all": round(mrr_u_stats[cn]["mrr_all"], 4),
               "mrr_hit": round(mrr_u_stats[cn]["mrr_hit"], 4)} for cn in cns]
pd.DataFrame(mrr_rows_u).to_csv(OUTPUT_DIR / "string_mrr_unfiltered.csv", index=False)

deg_rows = []
for b in DEGREE_BINS:
    for k in DEGREE_K:
        deg_rows.append({"bin": b, "k": k,
                          "coverage": round(deg_table[b][k]["coverage"], 4),
                          "random": round(deg_table[b][k]["random"], 4),
                          "fold": round(deg_table[b][k]["fold"], 2)})
pd.DataFrame(deg_rows).to_csv(OUTPUT_DIR / "string_degree_stratified.csv", index=False)

md = OUTPUT_DIR / "string_recall_curve_results.md"
with open(md, "w") as f:
    f.write("# STRING Network Protein Coverage\n\n")
    f.write("## Method\n\nFor each PPI (A,B) in STRING∩MAPPIE: "
            "network = proteins connected to A or B in STRING at each confidence threshold. "
            "Shared-protein exclusion: neighbours sharing a protein with (A,B) are removed. "
            "coverage@k = fraction of network proteins found among proteins in k nearest MAPPIE neighbours. "
            "precision@k = fraction of k neighbour PPIs with at least one protein in network. "
            "Random baseline is computed **exactly** via the hypergeometric expectation "
            "(no sampling): for each network protein p with d(p) valid PPIs containing it out of "
            "a valid pool of size M, P(p appears in k random draws) = 1 - hypergeom.pmf(0, M, d(p), k), "
            "averaged over network proteins.\n\n")
    f.write(f"Queries: {len(query_keys):,} PPIs in STRING∩MAPPIE\n\n")
    f.write("## Coverage@k\n\n")
    f.write("| k | ≥900 | ≥700 | ≥400 | ≥150 | random (exact) | fold |\n|---|---|---|---|---|---|---|\n")
    for k in K_SWEEP:
        vals = [f"{recall_sum[cn][k]/recall_count[cn][k]:.4f}" if recall_count[cn][k] > 0 else "nan" for cn in cns]
        r = rand_sum['≥900'][k]/rand_count['≥900'][k] if rand_count["≥900"][k] > 0 else float("nan")
        m900 = recall_sum["≥900"][k]/recall_count["≥900"][k] if recall_count["≥900"][k] > 0 else float("nan")
        fold = f"{m900/r:.1f}×" if r > 0 else "nan"
        f.write(f"| {k} | {' | '.join(vals)} | {r:.4f} | {fold} |\n")
    f.write("\n## Precision@k\n\n")
    f.write("| k | ≥900 | ≥700 | ≥400 | ≥150 |\n|---|---|---|---|---|\n")
    for k in K_SWEEP:
        vals = [f"{prec_sum[cn][k]/prec_count[cn][k]:.4f}" if prec_count[cn][k] > 0 else "nan" for cn in cns]
        f.write(f"| {k} | {' | '.join(vals)} |\n")
    f.write("\n## MRR, hit fraction, hit-conditioned MRR\n\n")
    f.write(
        "`hit_fraction` = fraction of queries that recover ≥1 network partner within the "
        f"{K_MAX-1}-neighbour search horizon. `mrr_all` = plain MRR (0 for non-hits, same as "
        "before). `mrr_hit` = MRR computed only over queries that had a hit.\n\n"
    )
    f.write("| threshold | n | hit_fraction | mrr_all | mrr_hit |\n|---|---|---|---|---|\n")
    for cn in cns:
        s = mrr_stats[cn]
        f.write(f"| {cn} | {s['n']:,} | {s['hit_fraction']:.4f} | {s['mrr_all']:.4f} | {s['mrr_hit']:.4f} |\n")
    f.write(f"\n## Degree-stratified fold enrichment ({PRIMARY_CONF})\n\n")
    f.write(
        "Query-protein degree = number of MAPPIE PPIs the protein appears in (over all "
        f"{n_all:,} PPIs). Queries binned into terciles by max(deg(A), deg(B)): "
        f"p33={p33:.1f}, p66={p66:.1f}. Both coverage and the random baseline are computed "
        "exactly for every query in the bin (no sampling).\n\n"
    )
    f.write("| degree bin | n queries | k | coverage | random (exact) | fold |\n|---|---|---|---|---|---|\n")
    for b in DEGREE_BINS:
        n_b = int((query_bin == b).sum())
        for k in DEGREE_K:
            t = deg_table[b][k]
            f.write(f"| {b} | {n_b:,} | {k} | {t['coverage']:.4f} | {t['random']:.4f} | {t['fold']:.2f}× |\n")
    f.write(
        f"\n**Summary**: high-degree fold-enrichment stays above 1 at all k ∈ {DEGREE_K}: "
        f"{'YES' if summary_ok else 'NO'} (k=10: {high_fold_10:.2f}×, k=500: {high_fold_500:.2f}×) "
        "→ signal is not purely a hub artefact.\n" if summary_ok else
        f"\n**Summary**: high-degree fold-enrichment does NOT stay above 1 at all k ∈ {DEGREE_K} "
        f"(k=10: {high_fold_10:.2f}×, k=500: {high_fold_500:.2f}×).\n"
    )
print(f"Saved: {md}")
print("Done.")
