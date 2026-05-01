import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Configuration
WINDOW = "3.20"
DATA_DIR = fr"C:\Users/M Hasan Rosyid/Desktop/PowerSystemDataset/Windows/Temporal/cnn_data/cnn_window_0.95_{WINDOW}"
CHECKPOINT_PATH = f"{DATA_DIR}/cnn_model.pt"
SEED = 42
THRESHOLD = 0.5
BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Channel groups
GROUPS = {
    "Bus voltage":          list(range(0, 59)),
    "Line current":         list(range(59, 93)),
    "Generator speed":      list(range(93, 103)),
    "Rotor angle":          list(range(103, 113)),
    "Gen active power":     list(range(113, 123)),
    "Gen reactive power":   list(range(123, 133)),
    "NSG active power":     list(range(133, 136)),
    "NSG reactive power":   list(range(136, 139)),
    "Electrical frequency": list(range(139, 156)),
    "Excitation current":   list(range(156, 166)),
    "Tap position":         list(range(166, 183)),
    "Line active power":    list(range(183, 217)),
    "Line reactive power":  list(range(217, 251)),
}

# Coverage check
all_idx = sorted([i for g in GROUPS.values() for i in g])
assert all_idx == list(range(251)), "Channel indices must cover 0-250 exactly once"
print(f"Channel groups verified: {len(GROUPS)} groups covering 251 channels.")


# CNN model
class CascadeCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)


# Load data
print(f"\nLoading data from {DATA_DIR}...")
X = np.load(f"{DATA_DIR}/X.npy")
y = np.load(f"{DATA_DIR}/y.npy")
print(f"X shape: {X.shape}, y shape: {y.shape}")

N, T, D = X.shape
assert D == 251, f"Expected 251 channels, got {D}"

# Test split
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Test set: {X_test.shape[0]} samples "
      f"({(y_test == 0).sum()} non-cascade, {(y_test == 1).sum()} cascade)")

# Load checkpoint
print(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

mean = ckpt["mean"]
std = ckpt["std"]
print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

# Standardise data
X_test_std = (X_test - mean) / std
X_test_t = torch.FloatTensor(X_test_std).permute(0, 2, 1)
print(f"Standardised test tensor shape: {X_test_t.shape}")

# Load model
model = CascadeCNN(in_channels=D).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# Predict probabilities
def predict_probs(X_tensor, model, batch_size=BATCH_SIZE):
    probs = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(DEVICE)
            logits = model(batch)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)

    return np.concatenate(probs)


# Evaluate metrics
def evaluate(y_true, y_prob, threshold=THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision_noncasc": prec[0],
        "recall_noncasc": rec[0],
        "precision_casc": prec[1],
        "recall_casc": rec[1],
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# Baseline evaluation
print("\n" + "=" * 60)
print(f"BASELINE, Window {WINDOW} s")
print("=" * 60)

baseline_probs = predict_probs(X_test_t, model)
baseline = evaluate(y_test, baseline_probs)

print(f"Accuracy:            {baseline['accuracy']:.4f}")
print(f"Cascading recall:    {baseline['recall_casc']:.4f}")
print(f"Cascading precision: {baseline['precision_casc']:.4f}")
print(f"Confusion matrix:    TN={baseline['TN']}, FP={baseline['FP']}, "
      f"FN={baseline['FN']}, TP={baseline['TP']}")

# Group ablation
print("\n" + "=" * 60)
print("CHANNEL-GROUP ABLATION")
print("=" * 60)

results = []

for group_name, indices in GROUPS.items():
    X_ablated = X_test_t.clone()
    X_ablated[:, indices, :] = 0.0

    probs = predict_probs(X_ablated, model)
    metrics = evaluate(y_test, probs)

    row = {
        "group": group_name,
        "n_channels": len(indices),
        "accuracy": metrics["accuracy"],
        "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
        "recall_casc": metrics["recall_casc"],
        "delta_recall_casc": metrics["recall_casc"] - baseline["recall_casc"],
        "precision_casc": metrics["precision_casc"],
        "delta_precision_casc": metrics["precision_casc"] - baseline["precision_casc"],
        "FN": metrics["FN"],
        "FP": metrics["FP"],
        "delta_FN": metrics["FN"] - baseline["FN"],
        "delta_FP": metrics["FP"] - baseline["FP"],
    }

    results.append(row)

    print(f"  {group_name:25s} (n={len(indices):3d}): "
          f"acc={metrics['accuracy']:.4f} (Δ={row['delta_accuracy']:+.4f}) | "
          f"R_casc={metrics['recall_casc']:.4f} (Δ={row['delta_recall_casc']:+.4f}) | "
          f"P_casc={metrics['precision_casc']:.4f} (Δ={row['delta_precision_casc']:+.4f})")

# Rank results
df = pd.DataFrame(results).sort_values("delta_recall_casc")

print("\n" + "=" * 60)
print("RANKED BY RECALL DROP")
print("=" * 60)

print(df[[
    "group",
    "n_channels",
    "recall_casc",
    "delta_recall_casc",
    "precision_casc",
    "delta_precision_casc",
    "FN",
    "delta_FN",
    "FP",
    "delta_FP",
]].to_string(index=False))

# Save results
out_path = f"{DATA_DIR}/channel_ablation_W{WINDOW.replace('.', '_')}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# Top groups
print("\n" + "=" * 60)
print("TOP 5 GROUPS BY RECALL DROP")
print("=" * 60)

top5 = df.head(5)

for _, row in top5.iterrows():
    print(f"  {row['group']:25s} | ΔR_casc = {row['delta_recall_casc']:+.4f} | "
          f"ΔP_casc = {row['delta_precision_casc']:+.4f} | "
          f"ΔFN = {row['delta_FN']:+d}")