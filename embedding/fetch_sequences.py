#!/usr/bin/env python3
"""Fetch every HIPPIE protein's sequence from UniProt into one multi-FASTA
for compute_esm_embeddings.py. Resumable via a pickle cache."""
import os
import pickle
import time

import pandas as pd
import requests
from tqdm import tqdm

HIPPIE_FILE = os.environ.get("HIPPIE_FULL_TXT", "data/hippie/hippie_current.txt")
SEQUENCE_CACHE_FILE = "cached_sequences.pkl"
FASTA_FILE = "fetched_sequences.fasta"
FASTA_DELAY = 0.34  # seconds between requests, to respect UniProt rate limiting


def fetch_uniprot_seq(uniprot_id: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            lines = response.text.splitlines()
            return "".join(lines[1:])
    except requests.RequestException:
        return None
    return None


def main():
    df = pd.read_csv(HIPPIE_FILE, sep="\t", header=None)
    protein_ids = pd.unique(df[[0, 2]].values.ravel())

    if os.path.exists(SEQUENCE_CACHE_FILE):
        print(f"Loading cached sequences from {SEQUENCE_CACHE_FILE}...")
        with open(SEQUENCE_CACHE_FILE, "rb") as f:
            sequences = pickle.load(f)
    else:
        sequences = {}
        print(f"Fetching {len(protein_ids)} protein sequences from UniProt...")
        for pid in tqdm(protein_ids):
            seq = fetch_uniprot_seq(pid)
            if seq:
                sequences[pid] = seq
            time.sleep(FASTA_DELAY)
        print(f"Fetched {len(sequences)} valid sequences.")

        with open(SEQUENCE_CACHE_FILE, "wb") as f:
            pickle.dump(sequences, f)
        print(f"Saved sequences: {SEQUENCE_CACHE_FILE}")

    with open(FASTA_FILE, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")
    print(f"Saved multi-FASTA: {FASTA_FILE}")


if __name__ == "__main__":
    main()
