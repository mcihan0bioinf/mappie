#!/usr/bin/env python3
"""Clean BioPlex, split in-map vs. novel edges, keep both-dark pairs (Unknome knownness <= threshold)."""
import csv
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

BIOPLEX_TSV = os.environ.get("BIOPLEX_TSV", "data/BioPlex_293T_Network_10K_Dec_2019.tsv")
BIOPLEX_PINT_MIN = float(os.environ.get("BIOPLEX_PINT_MIN", "0.75"))
DARKNESS_METRIC = os.environ.get("DARKNESS_METRIC", "unknome")
DARKNESS_THRESHOLD = float(os.environ.get("DARKNESS_THRESHOLD", "1.0"))
UNKNOME_TABLE = os.environ.get("UNKNOME_TABLE", "data/unknome_protein_table_18_Mar_2026.tsv")
UNKNOME_TAXON_ID = int(os.environ.get("UNKNOME_TAXON_ID", "9606"))
UNIPROT_ID_MAP_TSV = os.environ.get("UNIPROT_PROTEOME_TSV", "data/uniprot/uniprotkb_proteome_UP000005640.tsv")
REFERENCE_MAP_CSV = os.environ.get("MAPPIE_REFERENCE_MAP_CSV", "../../mappie/data_processed/esm_umap_hippie_064_server.csv")
INTERMEDIATE_DIR = Path(os.environ.get("INTERMEDIATE_DIR", str(ROOT / "intermediate")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results")))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", str(ROOT / "logs")))

BIOPLEX_CLEAN_COLS = [
    "edge_id", "geneA", "geneB", "uniprotA_raw", "uniprotB_raw", "uniprotA", "uniprotB",
    "symbolA", "symbolB", "pW", "pNI", "pInt", "isoform_collapsed_A", "isoform_collapsed_B",
]
BOTH_DARK_COLS = [
    "pair_id", "edge_id", "uniprotA", "uniprotB", "mnemonicA", "mnemonicB",
    "knownness_A", "knownness_B", "darkness_metric", "darkness_threshold", "in_map",
]

ISOFORM_RE = re.compile(r"^([A-Za-z0-9]+)-(\d+)$")


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


def strip_isoform(accession: str) -> tuple[str, bool]:
    """'Q9NR80-4' -> ('Q9NR80', True); 'P12345' -> ('P12345', False)."""
    m = ISOFORM_RE.match(accession)
    if m:
        return m.group(1), True
    return accession, False


def load_acc_to_mnemonic(uniprot_tsv: str) -> dict[str, str]:
    df = pd.read_csv(uniprot_tsv, sep="\t", usecols=["Entry", "Entry Name"])
    return dict(zip(df["Entry"], df["Entry Name"]))


def load_unknome_knownness(unknome_table: str, taxon_id: int) -> dict[str, float]:
    """UniProt accession -> knownness score, human only."""
    df = pd.read_csv(unknome_table, sep="\t")
    df = df[df["taxon_id"] == taxon_id]
    acc_to_knownness: dict[str, float] = {}
    for accessions, knownness in zip(df["uniprot_accessions"], df["knownness"]):
        if pd.isna(accessions):
            continue
        for acc in str(accessions).split(";"):
            acc = acc.strip()
            if not acc:
                continue
            canonical, _ = strip_isoform(acc)
            acc_to_knownness.setdefault(canonical, knownness)
    return acc_to_knownness


def compute_darkness() -> dict[str, float]:
    if DARKNESS_METRIC == "unknome":
        return load_unknome_knownness(str(rpath(UNKNOME_TABLE)), UNKNOME_TAXON_ID)
    raise ValueError(f"Unknown darkness_metric: {DARKNESS_METRIC!r}")


def is_dark(knownness: float, threshold: float) -> bool:
    return knownness <= threshold


def split_ppikey(ppikey: str) -> tuple[str, str]:
    """ppikey is '{mnemonicA}_HUMAN_{mnemonicB}'."""
    parts = ppikey.split("_HUMAN_")
    if len(parts) != 2:
        raise ValueError(f"Cannot split ppikey: {ppikey!r}")
    return parts[0] + "_HUMAN", parts[1]


def build_reference_pairset(reference_map_csv: str) -> set[frozenset]:
    df = pd.read_csv(reference_map_csv)
    pairs = set()
    for ppikey in df["ppikey"]:
        try:
            a, b = split_ppikey(ppikey)
        except ValueError:
            continue
        pairs.add(frozenset({a, b}))
    return pairs


def load_clean_bioplex() -> pd.DataFrame:
    for d in (INTERMEDIATE_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rpath(BIOPLEX_TSV), sep="\t")
    df["edge_id"] = range(len(df))

    droplog = DropLog("load_clean_bioplex", LOGS_DIR)
    keep_mask, uniprotA, uniprotB, isoform_A, isoform_B = [], [], [], [], []
    for _, row in df.iterrows():
        a_raw, b_raw = row["UniprotA"], row["UniprotB"]
        if pd.isna(a_raw) or pd.isna(b_raw) or not str(a_raw).strip() or not str(b_raw).strip():
            droplog.append(str(row["edge_id"]), "empty_or_malformed_uniprot_id",
                            geneA=row.get("GeneA"), geneB=row.get("GeneB"))
            keep_mask.append(False); uniprotA.append(None); uniprotB.append(None)
            isoform_A.append(False); isoform_B.append(False)
            continue
        if row["pInt"] < BIOPLEX_PINT_MIN:
            droplog.append(str(row["edge_id"]), "below_pint_min", pInt=row["pInt"])
            keep_mask.append(False); uniprotA.append(None); uniprotB.append(None)
            isoform_A.append(False); isoform_B.append(False)
            continue
        a_canon, a_iso = strip_isoform(str(a_raw))
        b_canon, b_iso = strip_isoform(str(b_raw))
        uniprotA.append(a_canon); uniprotB.append(b_canon)
        isoform_A.append(a_iso); isoform_B.append(b_iso)
        keep_mask.append(True)

    df["uniprotA_raw"], df["uniprotB_raw"] = df["UniprotA"], df["UniprotB"]
    df["uniprotA"], df["uniprotB"] = uniprotA, uniprotB
    df["isoform_collapsed_A"], df["isoform_collapsed_B"] = isoform_A, isoform_B
    df["geneA"], df["geneB"] = df["GeneA"], df["GeneB"]
    df["symbolA"], df["symbolB"] = df["SymbolA"], df["SymbolB"]

    clean = df[keep_mask][BIOPLEX_CLEAN_COLS].reset_index(drop=True)
    clean.to_csv(INTERMEDIATE_DIR / "bioplex_clean.tsv", sep="\t", index=False)
    droplog.write()

    n_isoform = int(clean["isoform_collapsed_A"].sum() + clean["isoform_collapsed_B"].sum())
    print(f"[load_clean_bioplex] kept {len(clean):,} / {len(df):,} edges "
          f"({len(droplog):,} dropped); {n_isoform:,} isoform accessions collapsed to canonical")
    return clean


def split_inmap_novel(bioplex_clean: pd.DataFrame) -> pd.DataFrame:
    acc_to_mnem = load_acc_to_mnemonic(UNIPROT_ID_MAP_TSV)
    ref_pairset = build_reference_pairset(REFERENCE_MAP_CSV)
    droplog = DropLog("split_inmap_novel", LOGS_DIR)

    mnemA, mnemB, in_map, matched = [], [], [], []
    for _, row in bioplex_clean.iterrows():
        ma = acc_to_mnem.get(row["uniprotA"])
        mb = acc_to_mnem.get(row["uniprotB"])
        if ma is None:
            droplog.append(row["uniprotA"], "accession_not_in_uniprot_id_map", edge_id=row["edge_id"], side="A")
        if mb is None:
            droplog.append(row["uniprotB"], "accession_not_in_uniprot_id_map", edge_id=row["edge_id"], side="B")
        mnemA.append(ma); mnemB.append(mb)
        if ma is not None and mb is not None and frozenset({ma, mb}) in ref_pairset:
            in_map.append(True); matched.append(f"{ma}_{mb}")
        else:
            in_map.append(False); matched.append(None)

    out = bioplex_clean.copy()
    out["mnemonicA"], out["mnemonicB"] = mnemA, mnemB
    out["in_map"], out["matched_ppikey"] = in_map, matched
    out.to_csv(INTERMEDIATE_DIR / "bioplex_split.tsv", sep="\t", index=False)
    droplog.write()

    n_in_map = int(out["in_map"].sum())
    print(f"[split_inmap_novel] {n_in_map:,} in-map edges, {len(out) - n_in_map:,} novel edges "
          f"({len(droplog):,} accessions unmapped to a mnemonic)")
    return out


def label_both_dark_pairs(bioplex_clean: pd.DataFrame, bioplex_split: pd.DataFrame) -> pd.DataFrame:
    mnem_map = bioplex_split.set_index("edge_id")[["mnemonicA", "mnemonicB", "in_map"]]
    knownness = compute_darkness()
    threshold = DARKNESS_THRESHOLD
    droplog = DropLog("label_both_dark_pairs", LOGS_DIR)

    rows = []
    for _, row in bioplex_clean.iterrows():
        ua, ub = row["uniprotA"], row["uniprotB"]
        ka, kb = knownness.get(ua), knownness.get(ub)
        if ka is None or kb is None:
            droplog.append(f"{ua}_{ub}", "not_in_unknome_table", edge_id=row["edge_id"])
            continue
        if not (is_dark(ka, threshold) and is_dark(kb, threshold)):
            droplog.append(f"{ua}_{ub}", "not_both_dark", edge_id=row["edge_id"])
            continue
        mnem_row = mnem_map.loc[row["edge_id"]]
        rows.append({
            "pair_id": f"bp_bothdark_{row['edge_id']}", "edge_id": row["edge_id"],
            "uniprotA": ua, "uniprotB": ub,
            "mnemonicA": mnem_row["mnemonicA"], "mnemonicB": mnem_row["mnemonicB"],
            "knownness_A": ka, "knownness_B": kb,
            "darkness_metric": DARKNESS_METRIC, "darkness_threshold": threshold,
            "in_map": bool(mnem_row["in_map"]),
        })

    both_dark = pd.DataFrame(rows, columns=BOTH_DARK_COLS)
    both_dark.to_csv(RESULTS_DIR / "bioplex_both_dark_pairs.tsv", sep="\t", index=False)
    droplog.write()

    n_proteins = len(set(both_dark["uniprotA"]) | set(both_dark["uniprotB"]))
    n_in_map = int(both_dark["in_map"].sum())
    print(f"[label_both_dark_pairs] {len(both_dark):,} both-dark pairs ({n_proteins:,} distinct dark proteins) "
          f"at threshold={threshold}; {n_in_map:,} already in MAPPIE, "
          f"{len(both_dark) - n_in_map:,} novel ({len(droplog):,} edges dropped)")
    return both_dark


def main():
    clean = load_clean_bioplex()
    split = split_inmap_novel(clean)
    label_both_dark_pairs(clean, split)


if __name__ == "__main__":
    main()
