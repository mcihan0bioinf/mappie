import pandas as pd
import numpy as np
import umap
import os
import joblib


np.random.seed(42)

latent_file = "esm_multiply_ld128_064_latent.csv"

base_dir = os.path.join(os.environ.get("MAPPIE_TRAINING_BASE", "data"), "optimize_encoder")
output_dir = os.path.join(base_dir, "umap")
os.makedirs(output_dir, exist_ok=True)

# For the full 199,137-PPI reference set
neighbors_list = [30, 50, 100]   # keep 50 as final
min_dist = 0.1
n_components = 2
metric = "euclidean"
random_state = 42


path = os.path.join(base_dir, latent_file)
print(f"Loading latent space: {latent_file}")

df = pd.read_csv(path, index_col=0)
X = df.values.astype(np.float32)

if np.isnan(X).any():
    raise ValueError("NaNs found in latent space")

print(f"   Shape: {X.shape}")


for n in neighbors_list:
    print(f"\nUMAP n_neighbors={n}")

    reducer = umap.UMAP(
        n_neighbors=n,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )

    embedding = reducer.fit_transform(X)

    umap_df = pd.DataFrame(
        embedding,
        columns=["UMAP1", "UMAP2"],
        index=df.index
    )

    name = os.path.splitext(latent_file)[0]

    out_csv   = os.path.join(output_dir, f"umap_{n}_{name}.csv")
    out_model = os.path.join(output_dir, f"umap_{n}_{name}_model.joblib")

    umap_df.to_csv(out_csv)
    joblib.dump(reducer, out_model)

    print(f"   Saved: {out_csv}")
    print(f"   Saved: {out_model}")

print("\nUMAP COMPLETE for the full 199,137-PPI reference latent space.")
