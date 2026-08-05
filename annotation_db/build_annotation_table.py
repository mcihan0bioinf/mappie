#!/usr/bin/env python3
"""Assembles ppi_annotation_long_ddi.csv, one row per (ppikey, proteinA,
proteinB, term_type, term):

  GO_BP / GO_MF / GO_CC   "<name> [GO:ID]"
  GO_ID                   bare "GO:ID" (every GO_BP/MF/CC hit also gets this)
  KEGG                    KEGG pathway ID the protein participates in, e.g. "hsa04010"
  PFAM / INTERPRO / REACTOME / RHEA   bare cross-reference ID
  DDI                     "PfamA--PfamB", one row per documented domain-domain
                           interaction (3did) between a Pfam of proteinA and a
                           Pfam of proteinB -- a PPI can carry several

A PPI carries a protein-level term if either partner carries it.
"""
import csv
import os
import re
import time
from collections import defaultdict
from io import StringIO

import pandas as pd
import requests

HIPPIE_FILE = os.environ.get("HIPPIE_FULL_TXT", "data/hippie/hippie_current.txt")
UNIPROT_PROTEOME_TSV = os.environ.get("UNIPROT_PROTEOME_TSV", "data/uniprot/uniprotkb_proteome_UP000005640.tsv")
GOA_HUMAN_GAF = os.environ.get("GOA_HUMAN_GAF", "data/go/goa_human.gaf")
GO_BASIC_OBO = os.environ.get("GO_BASIC_OBO", "data/go/go-basic.obo")
UNIPROT_XREF_TSV = os.environ.get("UNIPROT_XREF_TSV", "data/uniprot/uniprot_xref_domains.tsv")
KEGG_HSA_UNIPROT_CONV = os.environ.get("KEGG_HSA_UNIPROT_CONV", "data/kegg/kegg_hsa_uniprot_conv.txt")
KEGG_HSA_PATHWAY_LINKS = os.environ.get("KEGG_HSA_PATHWAY_LINKS", "data/kegg/kegg_hsa_pathway_links.txt")
THREEDID_DOMAIN_PAIRS_CSV = os.environ.get("THREEDID_DOMAIN_PAIRS_CSV", "data/3did/3did_domain_pairs_with_pfam.csv")
OUTPUT_CSV = os.environ.get("PPI_ANNOTATION_CSV", "data_processed/uniprot/ppi_annotation_long_ddi.csv")

GO_ASPECT_TO_TERM_TYPE = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}
_GO_ID_RE = re.compile(r"^id: (GO:\d+)")
_GO_NAME_RE = re.compile(r"^name: (.+)")

UNIPROT_XREF_FIELDS = ["accession", "id", "xref_pfam", "xref_interpro", "xref_reactome", "xref_rhea"]
UNIPROT_BATCH_SIZE = 100
UNIPROT_REQUEST_DELAY = 1.0


def fetch_uniprot_xrefs(hippie_path: str, output_tsv: str) -> None:
    """Pfam/InterPro/Reactome/Rhea from the UniProt REST API, batched by ID."""
    if os.path.exists(output_tsv):
        print(f"Using cached cross-references: {output_tsv}")
        return

    hippie = pd.read_csv(hippie_path, sep="\t", header=None)
    protein_ids = sorted(pd.unique(hippie[[0, 2]].values.ravel()))
    print(f"Fetching UniProt cross-references for {len(protein_ids)} proteins...")

    batches = []
    for i in range(0, len(protein_ids), UNIPROT_BATCH_SIZE):
        batch = protein_ids[i:i + UNIPROT_BATCH_SIZE]
        print(f"  batch {i // UNIPROT_BATCH_SIZE + 1}/{-(-len(protein_ids) // UNIPROT_BATCH_SIZE)} "
              f"({len(batch)} proteins)")
        query = " OR ".join(f"id:{entry}" for entry in batch)
        response = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": query, "fields": ",".join(UNIPROT_XREF_FIELDS), "format": "tsv", "size": UNIPROT_BATCH_SIZE},
            timeout=60,
        )
        response.raise_for_status()
        batches.append(pd.read_csv(StringIO(response.text), sep="\t"))
        time.sleep(UNIPROT_REQUEST_DELAY)

    result = pd.concat(batches, ignore_index=True)
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved {len(result)} rows: {output_tsv}")


def parse_gaf(gaf_path: str) -> dict[str, set[tuple[str, str]]]:
    """UniProt accession -> {(term_type, go_id), ...}, all evidence codes."""
    protein_terms: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with open(gaf_path) as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            acc, go_id, aspect = parts[1], parts[4], parts[8]
            term_type = GO_ASPECT_TO_TERM_TYPE.get(aspect)
            if not term_type:
                continue
            protein_terms[acc].add((term_type, go_id))
    return protein_terms


def load_go_id_to_name(obo_path: str) -> dict[str, str]:
    """GO:0006081 -> 'aldehyde metabolic process'."""
    id_to_name: dict[str, str] = {}
    cur_id = None
    with open(obo_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur_id = None
                continue
            if m := _GO_ID_RE.match(line):
                cur_id = m.group(1)
            elif m := _GO_NAME_RE.match(line):
                if cur_id:
                    id_to_name[cur_id] = m.group(1)
    return id_to_name


def load_acc_to_mnemonic(uniprot_tsv: str) -> dict[str, str]:
    df = pd.read_csv(uniprot_tsv, sep="\t", usecols=["Entry", "Entry Name"])
    return dict(zip(df["Entry"], df["Entry Name"]))


def load_go_terms_by_mnemonic(gaf_path: str, obo_path: str, acc_to_mnem: dict[str, str]
                               ) -> dict[str, set[tuple[str, str]]]:
    by_acc = parse_gaf(gaf_path)
    id_to_name = load_go_id_to_name(obo_path)

    by_mnem: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for acc, terms in by_acc.items():
        mnem = acc_to_mnem.get(acc)
        if not mnem:
            continue
        for term_type, go_id in terms:
            name = id_to_name.get(go_id, go_id)
            by_mnem[mnem].add((term_type, f"{name} [{go_id}]"))
            by_mnem[mnem].add(("GO_ID", go_id))
    return by_mnem


def load_xref_terms(xref_tsv: str) -> dict[str, set[tuple[str, str]]]:
    """UniProt's raw fields come back as ';'-delimited ID lists, e.g. 'PF00018;PF07653;'."""
    df = pd.read_csv(xref_tsv, sep="\t")
    by_mnem: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for _, row in df.iterrows():
        mnem = row.get("Entry Name") or row.get("id")
        if not isinstance(mnem, str):
            continue
        for col, term_type in [("Pfam", "PFAM"), ("InterPro", "INTERPRO"),
                                ("Reactome", "REACTOME"), ("Rhea", "RHEA")]:
            raw = row.get(col)
            if not isinstance(raw, str):
                continue
            for term in raw.split(";"):
                term = term.strip()
                if term:
                    by_mnem[mnem].add((term_type, term))
    return by_mnem


def load_kegg_terms(uniprot_conv_path: str, pathway_links_path: str, acc_to_mnem: dict[str, str]
                     ) -> dict[str, set[tuple[str, str]]]:
    """KEGG pathway membership per protein, via gene -> pathway (kegg_hsa_pathway_links.txt)
    and gene -> UniProt accession (kegg_hsa_uniprot_conv.txt)."""
    gene_to_pathways: dict[str, set[str]] = defaultdict(set)
    with open(pathway_links_path) as f:
        for line in f:
            gene, pathway = line.rstrip("\n").split("\t")
            gene_to_pathways[gene].add(pathway.removeprefix("path:"))

    by_mnem: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with open(uniprot_conv_path) as f:
        for line in f:
            gene, up_id = line.rstrip("\n").split("\t")
            acc = up_id.removeprefix("up:")
            mnem = acc_to_mnem.get(acc)
            if not mnem:
                continue
            for pathway in gene_to_pathways.get(gene, ()):
                by_mnem[mnem].add(("KEGG", pathway))
    return by_mnem


def strip_pfam_version(pfam_id) -> str | None:
    s = str(pfam_id).strip() if pfam_id is not None else ""
    if not s or s.lower() == "nan":
        return None
    return s.split(".")[0]


def load_ddi_map(threedid_csv: str) -> dict[str, set[str]]:
    """Pfam accession -> set of Pfam accessions it's documented to interact with (3did)."""
    ddi = pd.read_csv(threedid_csv)
    ddi_map: dict[str, set[str]] = defaultdict(set)
    for pf1_raw, pf2_raw in zip(ddi["domain1_pfam"], ddi["domain2_pfam"]):
        pf1, pf2 = strip_pfam_version(pf1_raw), strip_pfam_version(pf2_raw)
        if pf1 and pf2:
            ddi_map[pf1].add(pf2)
            ddi_map[pf2].add(pf1)
    return ddi_map


def assign_ddi_terms(pairs: list[tuple[str, str]], protein_pfams: dict[str, set[str]],
                      ddi_map: dict[str, set[str]]) -> dict[tuple[str, str], set[str]]:
    """{(protA, protB): {"PfamA--PfamB", ...}} -- every documented domain-domain
    interaction between a Pfam of protA and a Pfam of protB, both partner orders."""
    ddi_by_pair: dict[tuple[str, str], set[str]] = {}
    for protA, protB in pairs:
        pfams_a, pfams_b = protein_pfams.get(protA), protein_pfams.get(protB)
        if not pfams_a or not pfams_b:
            continue
        terms = set()
        for pf_a in pfams_a:
            for pf_b in ddi_map.get(pf_a, set()) & pfams_b:
                terms.add("--".join(sorted((pf_a, pf_b))))
        if terms:
            ddi_by_pair[(protA, protB)] = terms
    return ddi_by_pair


def main():
    print("Loading HIPPIE PPIs...")
    hippie = pd.read_csv(HIPPIE_FILE, sep="\t", header=None)
    pairs = list(zip(hippie[0], hippie[2]))

    fetch_uniprot_xrefs(HIPPIE_FILE, UNIPROT_XREF_TSV)

    print("Loading accession -> mnemonic mapping...")
    acc_to_mnem = load_acc_to_mnemonic(UNIPROT_PROTEOME_TSV)

    print("Loading GO terms (GAF + OBO names)...")
    go_terms = load_go_terms_by_mnemonic(GOA_HUMAN_GAF, GO_BASIC_OBO, acc_to_mnem)

    print("Loading Pfam/InterPro/Reactome/Rhea cross-references...")
    xref_terms = load_xref_terms(UNIPROT_XREF_TSV)

    print("Loading KEGG pathway membership...")
    kegg_terms = load_kegg_terms(KEGG_HSA_UNIPROT_CONV, KEGG_HSA_PATHWAY_LINKS, acc_to_mnem)

    print("Loading 3did domain-domain interactions...")
    ddi_map = load_ddi_map(THREEDID_DOMAIN_PAIRS_CSV)
    protein_pfams = {mnem: {t for tt, t in terms if tt == "PFAM"} for mnem, terms in xref_terms.items()}
    ddi_pairs = assign_ddi_terms(pairs, protein_pfams, ddi_map)

    protein_terms: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for mnem, terms in go_terms.items():
        protein_terms[mnem] |= terms
    for mnem, terms in xref_terms.items():
        protein_terms[mnem] |= terms
    for mnem, terms in kegg_terms.items():
        protein_terms[mnem] |= terms

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    n_rows = 0
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ppikey", "proteinA", "proteinB", "term_type", "term"])
        for protA, protB in pairs:
            ppikey = f"{protA}_{protB}"
            terms = protein_terms.get(protA, set()) | protein_terms.get(protB, set())
            for term_type, term in terms:
                writer.writerow([ppikey, protA, protB, term_type, term])
                n_rows += 1
            for ddi_term in ddi_pairs.get((protA, protB), ()):
                writer.writerow([ppikey, protA, protB, "DDI", ddi_term])
                n_rows += 1

    print(f"Wrote {n_rows} rows: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
