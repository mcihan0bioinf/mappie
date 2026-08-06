# MAPPIE — Map of Protein-Protein Interaction Embeddings
<img width="313.9" height="92" alt="mappie" src="https://github.com/user-attachments/assets/c8d33f1e-ec31-4d1d-9b1d-9942b7d6afe5" />


Code accompanying the manuscript:

> **A map of human protein-protein interaction embeddings for functional
> discovery**

MAPPIE (Map of Protein–Protein Interaction Embeddings) is a deep learning framework that learns an interaction-level latent space from protein sequence embeddings. Each PPI is represented by combining the ESM-2 embeddings of its two partners and projecting them through an autoencoder into a compact latent vector. 
In the resulting map, a PPI's local neighborhood reflects shared functional properties independent of network topology. Functional context is inferred by enrichment analysis over neighboring interactions. Candidate or newly predicted PPIs can be projected into the space and annotated from their nearest neighbors, enabling interaction-level function inference without relying on direct network connectivity.

The MAPPIE web server is freely available at [http://cbdm-01.zdv.uni-mainz.de/~mcihan/mappie/](https://cbdm-01.zdv.uni-mainz.de/~mcihan/mappie/). 
## Layout

Folders follow the manuscript's Methods, in pipeline order.

- `embedding/`
- `annotation_db/`
- `core_algorithm/`
- `training/`
- `model_selection/`
- `benchmark/`
- `dark_interactome/`

All example MAPPIE visualizations can be reproduced directly from the
public web server.
See the manuscript Methods for full detail.

## Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data & model availability

Large inputs for the full pipeline can be accessed either where codes indicate, from the manuscripts methods or from these sources:

- HIPPIE v2.3 — http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/
- ESM-2 650M (`esm2_t33_650M_UR50D`) via `fair-esm`/Hugging Face; ProtBERT
  from Hugging Face
- Trained autoencoder/UMAP models — reproduce with `training/`
- GO, Reactome, Pfam, InterPro, Rhea, 3did DDIs — public releases cited in
  the manuscript; `model_selection/` additionally needs 3did's per-domain
  protein membership (`ddi_counts.csv`, `protein_pfam_binary_matrix.csv`)
- KEGG hsa pathway links / UniProt ID mapping (`benchmark/degree_stratified`,
  `benchmark/self_recall`, `annotation_db/`) — KEGG REST API `https://rest.kegg.jp/link/pathway/hsa`
  and `https://rest.kegg.jp/conv/uniprot/hsa`
- CORUM 5.3 complex list with Functional Complex Group annotation
  (`benchmark/complex_recovery_corum/data/corum_fcg.txt`) — from
  corum.org/download
- STRING v12 — public download, used for `benchmark/`
- BioPlex v3.0 293T, Unknome — public downloads, used for
  `dark_interactome/`

Each script resolves its inputs from an environment variable, falling
back to a small relative default (e.g. `data/hippie/hippie_current.txt`,
`../mappie/data_processed/latent_index.npz`) if unset:

The variable name and default for each script are near the top of the
file (`os.environ.get("...", "...")`).

If you use MAPPIE in your work, please cite:
> **A map of human protein-protein interactions embeddings for functional
> discovery** Mert Cihan, Ute Distler, Miguel A. Andrade Navarro (TBD)

## Contact
For questions, feedback or problems please contact mcihan.bioinf@gmail.com.
