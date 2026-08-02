#!/usr/bin/env python3
"""Run ESM-2 (esm2_t33_650M_UR50D) over fetch_sequences.py's FASTA, save one
mean-pooled per-residue embedding per protein. ProtBERT comparison embeddings
are computed the same way via transformers.BertModel("Rostlab/prot_bert")."""
import os

import torch
import esm
from Bio import SeqIO
from tqdm import tqdm

FASTA = os.environ.get("EMBEDDING_FASTA", "fetched_sequences.fasta")
OUTPUT_DIR = os.environ.get("ESM_EMBEDDING_DIR", "embeddings/esm")
MODEL_NAME = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
BATCH_SIZE = 16


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(FASTA, "fasta")]
    print(f"Embedding {len(sequences)} sequences...")

    for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch = sequences[i:i + BATCH_SIZE]
        labels, _, tokens = batch_converter(batch)
        with torch.no_grad():
            out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        reps = out["representations"][REPR_LAYER]

        for j, (label, seq) in enumerate(batch):
            mean_embedding = reps[j, 1:len(seq) + 1].mean(0).cpu()
            torch.save({"label": label, "mean_representations": {REPR_LAYER: mean_embedding}},
                       os.path.join(OUTPUT_DIR, f"{label}.pt"))

    print("Done.")


if __name__ == "__main__":
    main()
