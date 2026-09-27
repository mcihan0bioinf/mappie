#!/usr/bin/env python3
"""Generates 20 degree-preserving randomized networks and projects each
through the existing MAPPIE model. Feeds ddi_null.py and functional_controls.py."""
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(os.environ.get("MAPPIE_REPO_ROOT", Path(__file__).resolve().parents[2]))))
from core_algorithm.project_ppi import Autoencoder, SCALER_FILE, MODEL_FILE, EMBEDDINGS_DB

MAPPIE_ROOT = Path(os.environ.get("MAPPIE_ROOT", "../mappie"))
LATENT_FILE = Path(os.environ.get("MAPPIE_LATENT_INDEX", str(MAPPIE_ROOT / "data_processed/latent_index.npz")))
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_NULL = 20
SWAP_NSWAP_FACTOR = 10  # double_edge_swap nswap = factor * n_edges, standard mixing heuristic
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_ppikey(key: str) -> tuple[str, str]:
    p1, p2 = key.split("_HUMAN_", 1)
    return p1 + "_HUMAN", p2


def load_real_network() -> nx.Graph:
    d = np.load(LATENT_FILE, allow_pickle=True)
    keys = list(np.asarray(d["keys"], dtype=str))
    edges = [parse_ppikey(k) for k in keys]
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"[Real network] {G.number_of_nodes():,} proteins, {G.number_of_edges():,} edges")
    return G


def randomize_network(G: nx.Graph, seed: int) -> nx.Graph:
    Gr = G.copy()
    n_edges = Gr.number_of_edges()
    nx.double_edge_swap(Gr, nswap=SWAP_NSWAP_FACTOR * n_edges, max_tries=100 * n_edges, seed=seed)

    same_nodes = set(Gr.nodes()) == set(G.nodes())
    same_edge_count = Gr.number_of_edges() == n_edges
    deg_real = dict(G.degree())
    deg_null = dict(Gr.degree())
    max_deg_diff = max(abs(deg_real[n] - deg_null.get(n, 0)) for n in G.nodes())
    print(f"  [seed {seed}] {Gr.number_of_edges():,} edges, same_node_set={same_nodes}, "
          f"same_edge_count={same_edge_count}, max_degree_difference={max_deg_diff}")
    assert same_nodes and same_edge_count and max_deg_diff == 0, \
        f"seed {seed}: degree-preserving randomization invariant violated"
    return Gr


def load_protein_embeddings(proteins: set[str]) -> dict[str, np.ndarray]:
    con = sqlite3.connect(f"file:{EMBEDDINGS_DB}?mode=ro", uri=True)
    placeholders = ",".join("?" * len(proteins))
    emb = {}
    for pid, blob in con.execute(
        f"SELECT protein_id, embedding FROM embeddings WHERE protein_id IN ({placeholders})",
        list(proteins),
    ):
        emb[pid] = np.frombuffer(blob, dtype=np.float32).copy()
    con.close()
    return emb


def project_network(G: nx.Graph, prot_emb: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Multiply -> production scaler -> production (frozen) autoencoder -> L2-normalize."""
    keys, merged = [], []
    for a, b in G.edges():
        if a in prot_emb and b in prot_emb:
            keys.append(f"{a}_{b}")
            merged.append(prot_emb[a] * prot_emb[b])
    merged = np.stack(merged).astype(np.float32)

    scaler = joblib.load(SCALER_FILE)
    X = scaler.transform(merged)

    model = Autoencoder(input_dim=1280, latent_dim=128)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval().to(DEVICE)

    with torch.no_grad():
        z = model.encoder(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z = (z / norms).astype(np.float32)
    return keys, z


def main():
    G = load_real_network()
    prot_emb = load_protein_embeddings(set(G.nodes()))
    print(f"[Embeddings] {len(prot_emb):,} / {G.number_of_nodes():,} proteins have a cached ESM-2 embedding")

    for seed in range(N_NULL):
        Gr = randomize_network(G, seed)
        edge_df = pd.DataFrame(list(Gr.edges()), columns=["protein1", "protein2"])
        edge_df.to_csv(OUT_DIR / f"null_network_seed_{seed}.csv", index=False)

        keys, latents = project_network(Gr, prot_emb)
        np.savez_compressed(OUT_DIR / f"null_latent_seed_{seed}.npz",
                             keys=np.array(keys), vecs=latents)
        print(f"  [seed {seed}] projected {len(keys):,} PPIs -> null_latent_seed_{seed}.npz")

    print("Done.")


if __name__ == "__main__":
    main()
