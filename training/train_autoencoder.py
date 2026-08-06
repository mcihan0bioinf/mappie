# Grid search: embedding source x merge op x latent dim x HIPPIE confidence
# filter. Trains an autoencoder per config, keeps the lowest-val-loss checkpoint.

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


base_path = os.environ.get("MAPPIE_TRAINING_BASE", "data")
combine_path = os.path.join(base_path, "combine_embeddings/combinations")

output_dir = os.path.join(base_path, "optimize_encoder")
os.makedirs(output_dir, exist_ok=True)

# hippie_pairs_064.csv / hippie_pairs_top10.csv are written by
# embedding/merge_embeddings.py (HIPPIE_FILTER_LIST_DIR=base_path to place them here)
filter_files = {
    "064": os.path.join(base_path, "hippie_pairs_064.csv"),
    "top10": os.path.join(base_path, "hippie_pairs_top10.csv"),
    "all": None,
}

# embedding source -> (pickle prefix, per-protein embedding dimensionality)
embedding_sources = {
    "esm": 1280,
    "protbert": 1024,
}

# merge operation -> whether it doubles the per-protein dimensionality
merge_methods = {
    "average": False,
    "multiply": False,
    "difference": False,
    "concat_ab": True,
    "concat_ba": True,
}

latent_dims = [512, 256, 128]

batch_size = 1024
num_epochs = 20
learning_rate = 1e-3
val_split = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


HIDDEN_SIZES = [512, 256]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Only cascade through hidden sizes strictly larger than latent_dim,
        # so the encoder shrinks monotonically down to the bottleneck instead
        # of shrinking then expanding back up when latent_dim >= a hidden size.
        hidden = [h for h in HIDDEN_SIZES if h > latent_dim]

        enc_dims = [input_dim] + hidden + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim] + hidden[::-1] + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


for source, per_protein_dim in embedding_sources.items():
    for merge, doubles_dim in merge_methods.items():
        embedding_file = os.path.join(combine_path, f"{source}_{merge}.pkl")
        if not os.path.exists(embedding_file):
            print(f"Skipping {source}/{merge}: {embedding_file} not found")
            continue

        input_dim = per_protein_dim * 2 if doubles_dim else per_protein_dim

        print(f"\nLoading {source} / {merge} (input_dim={input_dim})...")
        with open(embedding_file, "rb") as f:
            data = pickle.load(f)
        all_keys = sorted(data.keys())

        for filter_key, filter_path in filter_files.items():
            print(f"\nFilter: {filter_key}")
            if filter_path:
                filter_df = pd.read_csv(filter_path)
                filter_pairs = set(filter_df.iloc[:, 0])
                keys = [k for k in all_keys if k in filter_pairs]
            else:
                keys = all_keys

            print(f"Total pairs to use: {len(keys)}")
            X = np.stack([data[k] for k in keys])

            # Split raw (unscaled) data first, then fit the scaler on the
            # training split only, to avoid leaking validation statistics
            # into standardization.
            X_train_raw, X_val_raw = train_test_split(
                X, test_size=val_split, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_scaled = scaler.transform(X)

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                batch_size=batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
                batch_size=batch_size,
            )
            full_loader = DataLoader(
                TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)),
                batch_size=batch_size,
            )

            for latent_dim in latent_dims:
                print(f"Training: {source} | {merge} | latent={latent_dim} | filter={filter_key}")

                basename = f"{source}_{merge}_ld{latent_dim}_{filter_key}"
                scaler_file = os.path.join(output_dir, f"{basename}_scaler.pkl")
                latent_file = os.path.join(output_dir, f"{basename}_latent.csv")
                best_model_file = os.path.join(output_dir, f"{basename}_model_best.pt")
                joblib.dump(scaler, scaler_file)

                model = Autoencoder(input_dim, latent_dim).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn = nn.MSELoss()

                best_val_loss = float("inf")
                best_latents = None

                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0
                    for batch in train_loader:
                        x = batch[0].to(device)
                        recon, _ = model(x)
                        loss = loss_fn(recon, x)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total_loss += loss.item() * len(x)
                    avg_train = total_loss / len(X_train)

                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            x = batch[0].to(device)
                            recon, _ = model(x)
                            val_loss += loss_fn(recon, x).item() * len(x)
                    avg_val = val_loss / len(X_val)

                    print(f"Epoch {epoch+1}/{num_epochs} | Train={avg_train:.4f} | Val={avg_val:.4f}")

                    if avg_val < best_val_loss:
                        best_val_loss = avg_val
                        torch.save(model.state_dict(), best_model_file)

                        model.eval()
                        latent_all = []
                        with torch.no_grad():
                            for batch in full_loader:
                                x = batch[0].to(device)
                                _, z = model(x)
                                latent_all.append(z.cpu())
                        best_latents = torch.cat(latent_all).numpy()

                pd.DataFrame(best_latents, index=keys).to_csv(latent_file)
                print(f"Saved: {latent_file} (best val loss={best_val_loss:.4f})")

print("\nGrid search complete.")
