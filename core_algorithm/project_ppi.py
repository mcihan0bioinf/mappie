import os
import hashlib
import threading
from functools import lru_cache
from pathlib import Path
import requests
import torch
import esm
import numpy as np
import joblib


ROOT_DIR = Path(__file__).resolve().parent.parent
ESM_MODEL_PATH = str(ROOT_DIR / "data_raw/esm_model/esm2_t33_650M_UR50D.pt")

merge = "multiply"
latent_dim = 128
filter_key = "064"

AE_BASE = str(ROOT_DIR / "data_raw/esm_model")
basename = f"esm_{merge}_ld{latent_dim}_{filter_key}"

SCALER_FILE = f"{AE_BASE}/{basename}_scaler.pkl"
MODEL_FILE  = f"{AE_BASE}/{basename}_model_best.pt"

UMAP_BASE = str(ROOT_DIR / "models/umap_models")
UMAP_NEIGHBORS = 50
DEFAULT_UMAP_MODEL_FILE = ROOT_DIR / "models/umap_models" / f"umap_{UMAP_NEIGHBORS}_{basename}_latent_model.joblib"
DEFAULT_SERVER_UMAP_MODEL_FILE = ROOT_DIR / "models/umap_models" / f"umap_{UMAP_NEIGHBORS}_{basename}_latent_model_server.joblib"
UMAP_MODEL_FILE = os.getenv(
    "MAPPIE_UMAP_MODEL_FILE",
    str(DEFAULT_SERVER_UMAP_MODEL_FILE if DEFAULT_SERVER_UMAP_MODEL_FILE.exists() else DEFAULT_UMAP_MODEL_FILE),
)

MAX_LEN = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDINGS_DB = str(ROOT_DIR / "data_processed/protein_embeddings.db")


_esm_model = None
_alphabet = None
_batch_converter = None
_scaler = None
_ae_model = None
_umap_model = None
_model_load_lock = threading.Lock()
_esm_inference_lock = threading.Lock()  # ESM model is not thread-safe for concurrent forward passes

# Precomputed embedding DB (optional — speeds up known proteins)
_emb_db_con = None
_emb_db_lock = threading.Lock()


def _id_variants(protein_id: str) -> list[str]:
    """Lookup variants so callers can omit the '_HUMAN' species suffix or use
    either case. Real UniProt accessions (e.g. 'Q14653') pass through
    unchanged in addition to the '_HUMAN'-suffixed variant, so this never
    breaks accession-based lookups — it just tries one extra candidate."""
    pid = protein_id.strip()
    variants = [pid]
    upper = pid.upper()
    if upper != pid:
        variants.append(upper)
    if not upper.endswith("_HUMAN"):
        variants.append(upper + "_HUMAN")
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _get_precomputed_embedding(protein_id: str) -> np.ndarray | None:
    """Look up a precomputed ESM embedding by protein_id. Returns None if not found.
    Tries the raw id, its uppercase form, and the uppercase form with '_HUMAN'
    appended, so lookups work whether or not the caller included the suffix."""
    global _emb_db_con
    import sqlite3
    db_path = EMBEDDINGS_DB
    if not Path(db_path).exists():
        return None
    try:
        with _emb_db_lock:
            if _emb_db_con is None:
                _emb_db_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
            for cand in _id_variants(protein_id):
                row = _emb_db_con.execute(
                    "SELECT embedding FROM embeddings WHERE protein_id=?", (cand,)
                ).fetchone()
                if row is not None:
                    return np.frombuffer(row[0], dtype=np.float32).copy()
        return None
    except Exception:
        return None


def _models_ready():
    return all(
        part is not None
        for part in (_esm_model, _alphabet, _batch_converter, _scaler, _ae_model, _umap_model)
    )


def fetch_uniprot_seq(uniprot_id):
    """Fetch a sequence from UniProt by accession or entry name (mnemonic).
    UniProt's REST API requires the full entry name including the species
    suffix (e.g. 'IRF4_HUMAN'), not the bare gene symbol ('IRF4') — so a
    '_HUMAN'-suffixed variant is tried as well, letting callers omit it."""
    for cand in _id_variants(uniprot_id):
        url = f"https://rest.uniprot.org/uniprotkb/{cand}.fasta"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.text.startswith(">"):
                return "".join(r.text.splitlines()[1:])
        except requests.RequestException:
            pass
    return None


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        if latent_dim == 512:
            hidden = [512]
        elif latent_dim == 256:
            hidden = [512, 256]
        elif latent_dim == 128:
            hidden = [512, 256, 128]
        else:
            raise ValueError()

        enc = []
        prev = input_dim
        for h in hidden:
            enc.append(torch.nn.Linear(prev, h))
            enc.append(torch.nn.ReLU())
            prev = h
        enc.pop()
        self.encoder = torch.nn.Sequential(*enc)

        dec = []
        rev = list(reversed(hidden))
        prev = rev[0]
        for h in rev[1:]:
            dec.append(torch.nn.Linear(prev, h))
            dec.append(torch.nn.ReLU())
            prev = h
        dec.append(torch.nn.Linear(prev, input_dim))
        self.decoder = torch.nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def init_models():
    global _esm_model, _alphabet, _batch_converter
    global _scaler, _ae_model, _umap_model

    if _models_ready():
        return

    with _model_load_lock:
        if _models_ready():
            return

        print("Loading ESM-2...")
        model_data = torch.load(ESM_MODEL_PATH, map_location=device, weights_only=False)
        _esm_model, _alphabet = esm.pretrained.load_model_and_alphabet_core(
            "esm2_t33_650M_UR50D", model_data
        )
        _batch_converter = _alphabet.get_batch_converter()
        _esm_model.eval().to(device)

        print("Loading scaler + autoencoder...")
        _scaler = joblib.load(SCALER_FILE)

        input_dim = 1280
        _ae_model = Autoencoder(input_dim, latent_dim)
        _ae_model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        _ae_model.eval().to(device)

        print("Loading UMAP...")
        _umap_model = joblib.load(UMAP_MODEL_FILE)

        print("Models ready.")


_esm_cache: dict[str, np.ndarray] = {}  # seq_hash -> embedding, bounded to 256 entries
_ESM_CACHE_MAX = 256


def compute_esm_embedding(seq, label="query"):
    if len(seq) > MAX_LEN:
        raise ValueError(f"Sequence too long ({len(seq)} aa > {MAX_LEN})")

    seq_hash = hashlib.md5(seq.encode()).hexdigest()
    if seq_hash in _esm_cache:
        return _esm_cache[seq_hash]

    batch = [(label, seq)]
    _, _, tokens = _batch_converter(batch)
    tokens = tokens.to(device)

    with _esm_inference_lock, torch.no_grad():
        out = _esm_model(tokens, repr_layers=[33], return_contacts=False)

    reps = out["representations"][33]
    mean_emb = reps[0, 1:len(seq)+1].mean(0).cpu().numpy()

    if len(_esm_cache) >= _ESM_CACHE_MAX:
        _esm_cache.pop(next(iter(_esm_cache)))
    _esm_cache[seq_hash] = mean_emb
    return mean_emb


def project_ppi(protA, protB, input_type="sequence"):
    """
    protA, protB:
        - raw sequences        (input_type="sequence")
        - UniProt accessions  (input_type="uniprot")

    Returns:
        umap_x, umap_y, latent_vec (list of 128 floats, L2-normalised)
    """

    init_models()

    # ---- Try precomputed DB first (uniprot input only) ----
    embA = embB = None
    if input_type == "uniprot":
        embA = _get_precomputed_embedding(protA)
        embB = _get_precomputed_embedding(protB)

    # ---- For any embedding not in DB, fetch sequence and compute ----
    need_seqA = embA is None
    need_seqB = embB is None

    seqA = seqB = None
    if need_seqA or need_seqB:
        if input_type == "uniprot":
            results_seq = [None, None]
            errors_seq  = [None, None]

            def _fetch(idx, uid):
                seq = fetch_uniprot_seq(uid)
                if seq is None:
                    errors_seq[idx] = ValueError(f"Failed to fetch UniProt ID: {uid}")
                else:
                    results_seq[idx] = seq

            threads = []
            if need_seqA:
                threads.append(threading.Thread(target=_fetch, args=(0, protA)))
            if need_seqB:
                threads.append(threading.Thread(target=_fetch, args=(1, protB)))
            for t in threads: t.start()
            for t in threads: t.join()

            if errors_seq[0]: raise errors_seq[0]
            if errors_seq[1]: raise errors_seq[1]
            if need_seqA: seqA = results_seq[0]
            if need_seqB: seqB = results_seq[1]

        elif input_type == "sequence":
            seqA = protA
            seqB = protB
        else:
            raise ValueError("input_type must be 'sequence' or 'uniprot'")

    # ---- ESM: compute only what's missing, in parallel ----
    if need_seqA or need_seqB:
        results_emb = [embA, embB]
        errors_emb  = [None, None]

        def _embed(idx, seq, label):
            try:
                results_emb[idx] = compute_esm_embedding(seq, label)
            except Exception as e:
                errors_emb[idx] = e

        threads = []
        if need_seqA:
            threads.append(threading.Thread(target=_embed, args=(0, seqA, "A")))
        if need_seqB:
            threads.append(threading.Thread(target=_embed, args=(1, seqB, "B")))
        for t in threads: t.start()
        for t in threads: t.join()

        if errors_emb[0]: raise errors_emb[0]
        if errors_emb[1]: raise errors_emb[1]
        embA, embB = results_emb[0], results_emb[1]

    # ---- Merge ----
    merged = embA * embB

    # ---- Scale ----
    X_scaled = _scaler.transform(merged.reshape(1, -1))

    # ---- Encode ----
    with torch.no_grad():
        z = _ae_model.encoder(
            torch.tensor(X_scaled, dtype=torch.float32, device=device)
        )

    latent = z.cpu().numpy()

    # ---- L2-normalise (matches latent_index.npz convention) ----
    norm = np.linalg.norm(latent)
    latent_norm = (latent / norm if norm > 0 else latent).squeeze()

    # ---- UMAP ----
    xy = _umap_model.transform(latent).squeeze()

    return float(xy[0]), float(xy[1]), latent_norm.tolist()
