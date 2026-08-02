#!/usr/bin/env python3
"""
build_annotation_db.py

Convert ppi_annotation_long_ddi.csv (build_annotation_table.py's output)
into the SQLite database core_algorithm/run_enrichment.py queries at
inference time -- a flat table plus precomputed per-term counts, so
enrichment lookups don't need the whole annotation set in memory.
"""
import csv
import os
import sqlite3
from pathlib import Path

CSV_PATH = os.environ.get("PPI_ANNOTATION_CSV", "data_processed/uniprot/ppi_annotation_long_ddi.csv")
DB_PATH = os.environ.get("MAPPIE_ANNOTATION_DB", "data_processed/uniprot/ppi_annotation_long_ddi.db")
BATCH_SIZE = 50_000


def build(csv_path: str, db_path: str) -> None:
    csv_path, db_path = Path(csv_path), Path(db_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    db_path.unlink(missing_ok=True)
    print(f"Building {db_path} from {csv_path} ...")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;
        PRAGMA cache_size   = -65536;

        CREATE TABLE annotations (
            ppikey    TEXT NOT NULL,
            term_type TEXT NOT NULL,
            term      TEXT NOT NULL
        );
    """)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch: list[tuple[str, str, str]] = []
        total = 0
        for row in reader:
            batch.append((row["ppikey"].strip(), row["term_type"].strip(), row["term"].strip()))
            if len(batch) >= BATCH_SIZE:
                cur.executemany("INSERT INTO annotations VALUES (?,?,?)", batch)
                total += len(batch)
                batch.clear()
                if total % 1_000_000 == 0:
                    con.commit()
                    print(f"  {total:,} rows inserted...")
        if batch:
            cur.executemany("INSERT INTO annotations VALUES (?,?,?)", batch)
            total += len(batch)
        con.commit()

    print(f"  Total rows: {total:,}")
    print("Building indexes...")
    cur.executescript("""
        CREATE INDEX idx_ann_ppikey ON annotations(ppikey);
        CREATE INDEX idx_ann_term   ON annotations(term_type, term);
    """)

    print("Building term_stats table...")
    cur.executescript("""
        CREATE TABLE term_stats AS
            SELECT term_type, term, COUNT(DISTINCT ppikey) AS ppikey_count
            FROM annotations
            GROUP BY term_type, term;
        CREATE UNIQUE INDEX idx_ts ON term_stats(term_type, term);
    """)

    print("Building meta table...")
    cur.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value INTEGER);")
    cur.execute("INSERT INTO meta SELECT 'total_ppikeys', COUNT(DISTINCT ppikey) FROM annotations")
    con.commit()

    total_ppikeys = cur.execute("SELECT value FROM meta WHERE key='total_ppikeys'").fetchone()[0]
    unique_terms = cur.execute("SELECT COUNT(*) FROM term_stats").fetchone()[0]
    con.close()

    print("Done.")
    print(f"  DB size:        {db_path.stat().st_size / 1e9:.2f} GB")
    print(f"  Unique terms:   {unique_terms:,}")
    print(f"  Unique ppikeys: {total_ppikeys:,}")


if __name__ == "__main__":
    build(CSV_PATH, DB_PATH)
