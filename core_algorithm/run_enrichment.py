import os
import csv
import json
import pickle
import sqlite3
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom as _hypergeom_dist


ROOT_DIR = Path(__file__).resolve().parent.parent
ANN_FILE = str(ROOT_DIR / "data_processed/uniprot/ppi_annotation_long_ddi.csv")
DB_FILE  = str(ROOT_DIR / "data_processed/uniprot/ppi_annotation_long_ddi.db")
CACHE_VERSION = 1

PFAM_NAME_FILE = ROOT_DIR / "data_processed/pfam_mapping.csv"
_PFAM_NAME_MAP = None


def _load_pfam_names():
    """Pfam accession -> human-readable description, loaded once and cached.
    Source: data_processed/pfam_mapping.csv (pfam_name, pfam_ac, pfam_description),
    ported from Pfam-A.hmm.dat so DDI terms (raw "PFxxxxx--PFyyyyy" pairs) can be
    shown as readable domain names instead of bare accessions. Lives here (not in
    app.py) so both app.py and enrichment_plots.py can import it without a
    circular import."""
    global _PFAM_NAME_MAP
    if _PFAM_NAME_MAP is None:
        mapping = {}
        try:
            with open(PFAM_NAME_FILE, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    acc = (row.get("pfam_ac") or "").split(".")[0]
                    if acc and acc not in mapping:
                        mapping[acc] = row.get("pfam_description") or row.get("pfam_name") or acc
        except Exception:
            mapping = {}
        _PFAM_NAME_MAP = mapping
    return _PFAM_NAME_MAP


def pretty_ddi_term(term):
    """Raw DDI term "PF00069--PF00134" -> "Protein kinase domain – Cyclin, N-terminal domain".
    Falls back to the raw accession for any Pfam ID not found in the mapping,
    so an unmapped/new accession degrades gracefully instead of erroring."""
    if not term:
        return term
    s = str(term)
    if "--" not in s:
        return s
    a, b = s.split("--", 1)
    names = _load_pfam_names()
    na, nb = names.get(a, a), names.get(b, b)
    return f"{na} (self-interaction)" if a == b else f"{na} – {nb}"


def hypergeom_sf(k, N, K, n):
    # P(X >= k) using scipy's C implementation (single-value wrapper for CLI use)
    return float(_hypergeom_dist.sf(k - 1, N, K, n))


def _hypergeom_sf_vec(k_arr, N, K_arr, n):
    """Vectorized P(X >= k) for arrays of k and K values."""
    return _hypergeom_dist.sf(k_arr - 1, N, K_arr, n)


def bh_adjust(pvals):
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        val = pvals[i] * m / rank
        if val < prev:
            prev = val
        adj[i] = min(1.0, prev)
    return adj


def _cache_path_for_ann(ann_path):
    return Path(ann_path).with_suffix(".cache.pkl")


def _cache_enabled():
    return os.getenv("MAPPIE_ENRICH_CACHE", "1").strip().lower() not in ("0", "false", "no")


def _load_terms_from_csv(ann_path):
    term_to_ppis = {}
    all_ppis = set()
    with open(ann_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term_type = row["term_type"].strip()
            term = row["term"].strip()
            ppikey = row["ppikey"].strip()
            term_to_ppis.setdefault((term_type, term), set()).add(ppikey)
            all_ppis.add(ppikey)
    return term_to_ppis, all_ppis


def load_terms(ann_path):
    ann = Path(ann_path)
    cache_path = _cache_path_for_ann(ann_path)
    use_cache = _cache_enabled()

    if use_cache and cache_path.exists():
        try:
            ann_stat = ann.stat()
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if (
                isinstance(payload, dict)
                and payload.get("version") == CACHE_VERSION
                and payload.get("ann_mtime_ns") == ann_stat.st_mtime_ns
                and payload.get("ann_size") == ann_stat.st_size
            ):
                term_to_ppis = payload.get("term_to_ppis", {})
                all_ppis = payload.get("all_ppis", set())
                print(f"Loaded enrichment cache: {cache_path}")
                return term_to_ppis, all_ppis
            print(f"ℹEnrichment cache stale, rebuilding: {cache_path}")
        except Exception as e:
            print(f"Failed reading enrichment cache ({cache_path}): {e}")

    term_to_ppis, all_ppis = _load_terms_from_csv(ann_path)

    if use_cache:
        try:
            ann_stat = ann.stat()
            payload = {
                "version": CACHE_VERSION,
                "ann_mtime_ns": ann_stat.st_mtime_ns,
                "ann_size": ann_stat.st_size,
                "term_to_ppis": term_to_ppis,
                "all_ppis": all_ppis,
            }
            with cache_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Wrote enrichment cache: {cache_path}")
        except Exception as e:
            print(f"Failed writing enrichment cache ({cache_path}): {e}")
    return term_to_ppis, all_ppis


def load_ppi_terms(ann_path):
    """Return mapping: ppikey -> list of (term_type, term)."""
    ppi_terms = {}
    with open(ann_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ppikey = row["ppikey"].strip()
            term_type = row["term_type"].strip()
            term = row["term"].strip()
            if not ppikey or not term:
                continue
            ppi_terms.setdefault(ppikey, []).append((term_type, term))
    return ppi_terms


def run_enrichment_from_cache(neigh_ppis, term_to_ppis, all_ppis, padj_cutoff=None):
    neigh_ppis = set(neigh_ppis)
    neigh_ppis = {p for p in neigh_ppis if p in all_ppis}
    n = len(neigh_ppis)
    if n == 0 or not term_to_ppis:
        return []

    N = len(all_ppis)

    keys, k_list, K_list, obs_ppis_list = [], [], [], []
    for (term_type, term), ppis in term_to_ppis.items():
        K = len(ppis)
        if K == 0:
            continue
        hits = neigh_ppis & ppis
        k = len(hits)
        if k == 0:
            continue
        keys.append((term_type, term))
        k_list.append(k)
        K_list.append(K)
        obs_ppis_list.append(sorted(hits))

    if not keys:
        return []

    k_arr = np.array(k_list, dtype=np.int64)
    K_arr = np.array(K_list, dtype=np.int64)
    pvals = _hypergeom_sf_vec(k_arr, N, K_arr, n)

    results = []
    for i, (term_type, term) in enumerate(keys):
        pval = float(pvals[i])
        K = K_list[i]
        k = k_list[i]
        obs = obs_ppis_list[i]
        results.append({
            "term_type": term_type,
            "term": term,
            "observed": k,
            "expected": round(n * (K / N), 4),
            "p_value": pval,
            "observed_ppikeys": "|".join(obs),
            "observed_ppis": obs,
        })

    if padj_cutoff is not None:
        results = [r for r in results if r["p_value"] <= padj_cutoff]
    results.sort(key=lambda r: r["p_value"])
    return results


def run_enrichment(neigh_ppis, ann_path, padj_cutoff):
    term_to_ppis, all_ppis = load_terms(ann_path)
    return run_enrichment_from_cache(neigh_ppis, term_to_ppis, all_ppis, padj_cutoff)


def run_counts_from_cache(neigh_ppis, ppi_terms):
    neigh_ppis = [p for p in neigh_ppis if p in ppi_terms]
    n = len(neigh_ppis)
    if n == 0 or not ppi_terms:
        return []
    term_counts = {}
    term_to_ppis = {}
    for p in neigh_ppis:
        for term_type, term in ppi_terms.get(p, []):
            key = (term_type, term)
            term_counts[key] = term_counts.get(key, 0) + 1
            term_to_ppis.setdefault(key, []).append(p)

    results = []
    for (term_type, term), k in term_counts.items():
        observed_ppis = sorted(term_to_ppis.get((term_type, term), []))
        results.append({
            "term": term,
            "term_type": term_type,
            "observed": k,
            "fraction": f"{k}/{n}",
            "observed_ppikeys": "|".join(observed_ppis),
            "observed_ppis": observed_ppis,
            "observed_hits": k,
            "category": term_type,
        })

    results.sort(key=lambda r: r["observed"], reverse=True)
    return results


def _open_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    con.execute("PRAGMA cache_size = -32768")
    return con


def load_term_index(db_path: str) -> dict[str, set[str]]:
    """Return {term_type: {term, ...}} — just the category/term names, no ppikey sets.
    Lightweight: loads only term_stats (thousands of rows, not millions).
    """
    cat_to_terms: dict[str, set[str]] = {}
    with _open_db(db_path) as con:
        for term_type, term in con.execute("SELECT term_type, term FROM term_stats"):
            cat_to_terms.setdefault(term_type, set()).add(term)
    return cat_to_terms


def get_n_background(db_path: str) -> int:
    """Total unique ppikeys in the annotation background."""
    with _open_db(db_path) as con:
        return con.execute("SELECT value FROM meta WHERE key='total_ppikeys'").fetchone()[0]


def get_term_ppikeys(db_path: str, term_type: str, term: str) -> set[str]:
    """Return the set of ppikeys annotated with a specific term."""
    with _open_db(db_path) as con:
        rows = con.execute(
            "SELECT DISTINCT ppikey FROM annotations WHERE term_type=? AND term=?",
            (term_type, term),
        ).fetchall()
    return {r[0] for r in rows}


# In-memory cache of term_stats (87K rows, ~8 MB) — loaded once per process.
_TERM_STATS: dict[tuple[str, str], int] = {}
_N_BACKGROUND: int = 0
_DB_STATS_LOADED: str = ""  # db_path that was loaded


def _ensure_stats_loaded(db_path: str) -> None:
    global _TERM_STATS, _N_BACKGROUND, _DB_STATS_LOADED
    if _DB_STATS_LOADED == db_path:
        return
    with _open_db(db_path) as con:
        _TERM_STATS = {
            (tt, t): cnt
            for tt, t, cnt in con.execute("SELECT term_type, term, ppikey_count FROM term_stats")
        }
        _N_BACKGROUND = con.execute(
            "SELECT value FROM meta WHERE key='total_ppikeys'"
        ).fetchone()[0]
    _DB_STATS_LOADED = db_path


# Persistent read-only connection — opened once per process, reused for all queries.
_DB_CON: sqlite3.Connection | None = None
_DB_CON_PATH: str = ""


def _get_db_con(db_path: str) -> sqlite3.Connection:
    global _DB_CON, _DB_CON_PATH
    if _DB_CON is None or _DB_CON_PATH != db_path:
        if _DB_CON is not None:
            try:
                _DB_CON.close()
            except Exception:
                pass
        _DB_CON = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        _DB_CON.execute("PRAGMA cache_size = -65536")
        _DB_CON_PATH = db_path
    return _DB_CON


def run_enrichment_db(neigh_ppis, db_path: str, padj_cutoff=None) -> list[dict]:
    """Hypergeometric enrichment via SQLite with vectorized p-value computation."""
    neigh_set = set(neigh_ppis)
    if not neigh_set:
        return []

    _ensure_stats_loaded(db_path)
    N = _N_BACKGROUND
    if N == 0:
        return []

    con = _get_db_con(db_path)
    ppikey_list = list(neigh_set)
    if len(ppikey_list) <= 999:
        # Direct IN clause — 200x faster than temp table JOIN for typical k values.
        placeholders = ",".join("?" * len(ppikey_list))
        raw = con.execute(
            f"SELECT term_type, term, ppikey FROM annotations WHERE ppikey IN ({placeholders})",
            ppikey_list,
        ).fetchall()
    else:
        # Fall back to temp table for very large neighbor sets (>999).
        con2 = sqlite3.connect(db_path, check_same_thread=False)
        try:
            con2.execute("PRAGMA cache_size = -65536")
            con2.execute("CREATE TEMP TABLE _neigh (ppikey TEXT PRIMARY KEY)")
            con2.executemany("INSERT OR IGNORE INTO _neigh VALUES (?)", [(p,) for p in ppikey_list])
            raw = con2.execute("""
                SELECT a.term_type, a.term, a.ppikey
                FROM annotations a
                INNER JOIN _neigh t ON a.ppikey = t.ppikey
            """).fetchall()
        finally:
            con2.close()

    if not raw:
        return []

    # Aggregate hits per term.
    term_ppis: dict[tuple[str, str], set[str]] = defaultdict(set)
    for term_type, term, ppikey in raw:
        term_ppis[(term_type, term)].add(ppikey)

    valid_neigh = {ppikey for _, _, ppikey in raw}
    n = len(valid_neigh)
    if n == 0:
        return []

    # Build parallel arrays for vectorized hypergeometric test.
    keys, k_list, K_list, obs_ppis_list = [], [], [], []
    for (term_type, term), ppis in term_ppis.items():
        K = _TERM_STATS.get((term_type, term), 0)
        if K == 0:
            continue
        keys.append((term_type, term))
        k_list.append(len(ppis))
        K_list.append(K)
        obs_ppis_list.append(sorted(ppis))

    if not keys:
        return []

    k_arr = np.array(k_list, dtype=np.int64)
    K_arr = np.array(K_list, dtype=np.int64)
    pvals = _hypergeom_sf_vec(k_arr, N, K_arr, n)

    results = []
    for i, (term_type, term) in enumerate(keys):
        pval = float(pvals[i])
        K = K_list[i]
        k = k_list[i]
        results.append({
            "term_type": term_type,
            "term": term,
            "observed": k,
            "expected": round(n * (K / N), 4),
            "p_value": pval,
            "observed_ppikeys": "|".join(obs_ppis_list[i]),
            "observed_ppis": obs_ppis_list[i],
        })

    if padj_cutoff is not None:
        results = [r for r in results if r["p_value"] <= padj_cutoff]
    results.sort(key=lambda r: r["p_value"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi-file", required=True)
    ap.add_argument("--ann-file", default=ANN_FILE)
    ap.add_argument("--padj", type=float, default=0.1)
    args = ap.parse_args()

    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"Annotation file not found: {args.ann_file}")
    if not os.path.exists(args.ppi_file):
        raise FileNotFoundError(f"PPI file not found: {args.ppi_file}")

    with open(args.ppi_file, "r", encoding="utf-8") as f:
        neigh_ppis = [line.strip() for line in f if line.strip()]

    results = run_enrichment(neigh_ppis, args.ann_file, args.padj)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
