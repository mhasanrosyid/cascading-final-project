"""Inspect persistent CNN false negatives at W1."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

# Config
WINDOW = "1.25"
DATA_DIR = Path(f"C:/Users/M Hasan Rosyid/Desktop/PowerSystemDataset/Windows/Temporal/cnn_data/cnn_window_0.95_{WINDOW}")
SEED = 42
TAU = 0.249

# Load arrays
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")
meta = pd.read_csv(DATA_DIR / "meta.csv")
N, T, D = X.shape

print(f"Loaded W{WINDOW}: X={X.shape}, y={y.shape}, meta={len(meta)} rows")
assert len(meta) == N, "meta.csv length mismatch with X.npy"

# Reproduce training split
all_indices = np.arange(N)
trainval_idx, test_idx = train_test_split(
    all_indices, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Test set size: {len(test_idx)}")

X_test = X[test_idx]
y_test = y[test_idx]
meta_test = meta.iloc[test_idx].reset_index(drop=True)

# Standardise from checkpoint
ckpt = torch.load(DATA_DIR / "cnn_model.pt", weights_only=False, map_location="cpu")
mean, std = ckpt["mean"], ckpt["std"]
X_test_s = (X_test - mean) / std

X_test_t = torch.FloatTensor(X_test_s).permute(0, 2, 1)
test_loader = DataLoader(TensorDataset(X_test_t, torch.FloatTensor(y_test)),
                         batch_size=256, shuffle=False)

# Model
class CascadeCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)

device = torch.device("cpu")
model = CascadeCNN(in_channels=D).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
all_probs = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
y_proba = np.concatenate(all_probs)

# Identify FNs
y_pred = (y_proba >= TAU).astype(int)
fn_mask = (y_test == 1) & (y_pred == 0)
n_fn = fn_mask.sum()
print(f"\nFNs at tau={TAU}: {n_fn}")

fn_meta = meta_test[fn_mask].copy()
fn_meta["proba"] = y_proba[fn_mask]

print(f"\nFN probability distribution:")
print(f"  Min: {y_proba[fn_mask].min():.4f}")
print(f"  Max: {y_proba[fn_mask].max():.4f}")
print(f"  Mean: {y_proba[fn_mask].mean():.4f}")
print(f"  Median: {np.median(y_proba[fn_mask]):.4f}")

# Parse base field
def parse_base(base_str):
    line_part, rest = base_str.split("_load=", 1)
    load_str, wind_part = rest.split("_wind=", 1)
    w1, w2, w3 = wind_part.split("_")
    return {
        "line": line_part,
        "load": float(load_str),
        "wind1": float(w1),
        "wind2": float(w2),
        "wind3": float(w3),
    }

parsed = pd.DataFrame([parse_base(b) for b in fn_meta["base"]])
fn_full = pd.concat([fn_meta.reset_index(drop=True),
                     parsed.reset_index(drop=True)], axis=1)
print(f"\n=== FN OPERATING CONDITIONS ===")
print(fn_full[["line", "load", "wind1", "wind2", "wind3", "proba"]].to_string())

# Compare against cascades
all_casc_test = meta_test[y_test == 1].copy()
parsed_all = pd.DataFrame([parse_base(b) for b in all_casc_test["base"]])
all_casc_full = pd.concat([all_casc_test.reset_index(drop=True),
                           parsed_all.reset_index(drop=True)], axis=1)

print(f"\n=== COMPARISON: FN vs all cascading test samples ===")
print(f"\nLoad - FN mean={fn_full['load'].mean():.4f}, std={fn_full['load'].std():.4f}")
print(f"Load - All casc mean={all_casc_full['load'].mean():.4f}, std={all_casc_full['load'].std():.4f}")
print(f"\nWind1 - FN mean={fn_full['wind1'].mean():.4f}, all={all_casc_full['wind1'].mean():.4f}")
print(f"Wind2 - FN mean={fn_full['wind2'].mean():.4f}, all={all_casc_full['wind2'].mean():.4f}")
print(f"Wind3 - FN mean={fn_full['wind3'].mean():.4f}, all={all_casc_full['wind3'].mean():.4f}")

print(f"\n=== Faulted-line distribution among FNs ===")
print(fn_full["line"].value_counts())

print(f"\n=== Faulted-line distribution among all cascading test samples ===")
print(all_casc_full["line"].value_counts())

# Save output
fn_full.to_csv(DATA_DIR / "fn_inspection.csv", index=False)
print(f"\nSaved fn_inspection.csv with {len(fn_full)} rows")