import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
WINDOW = "1.45"
DATA_DIR = f"/workspace/Windows/{WINDOW}"

EPOCHS = 150
PATIENCE = 20
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 256
SEED = 42

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load data
X = np.load(f"{DATA_DIR}/X.npy")
y = np.load(f"{DATA_DIR}/y.npy")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

N, T, D = X.shape

# Data split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.1,
    random_state=SEED,
    stratify=y_trainval
)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Channel standardisation
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Tensor conversion
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
X_val_t = torch.FloatTensor(X_val).permute(0, 2, 1)
X_test_t = torch.FloatTensor(X_test).permute(0, 2, 1)

y_train_t = torch.FloatTensor(y_train)
y_val_t = torch.FloatTensor(y_val)
y_test_t = torch.FloatTensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False
)

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


model = CascadeCNN(in_channels=D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"\nDevice: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss weighting
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()

pos_weight = torch.FloatTensor([n_neg / n_pos]).to(device)
print(f"pos_weight: {pos_weight.item():.3f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=EPOCHS
)

# Training setup
best_val_loss = float("inf")
patience_counter = 0
best_state = None

print(f"\nTraining for up to {EPOCHS} epochs\n")

for epoch in range(EPOCHS):
    # Train step
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss.item() * len(yb)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        train_correct += (preds == yb).sum().item()
        train_total += len(yb)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validation step
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item() * len(yb)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_total += len(yb)

    val_loss /= val_total
    val_acc = val_correct / val_total

    scheduler.step()

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS} | "
        f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
        f"LR={optimiser.param_groups[0]['lr']:.2e}"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

# Final evaluation
print("\n" + "=" * 60)
print(f"FINAL EVALUATION, Window {WINDOW} s")
print("=" * 60)

model.load_state_dict(best_state)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        logits = model(xb)
        preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()

        all_preds.append(preds)
        all_labels.append(yb.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)

print(f"\nConfusion matrix:\n{confusion_matrix(y_true, y_pred)}")
print(f"\n{classification_report(y_true, y_pred, digits=4)}")

# Save checkpoint
save_path = f"{DATA_DIR}/cnn_model.pt"

torch.save({
    "model_state_dict": best_state,
    "mean": mean,
    "std": std,
    "window": WINDOW,
    "sampling_rate_hz": 100,
    "input_channels": D,
    "sequence_length": T,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "batch_size_train": BATCH_SIZE_TRAIN,
    "batch_size_eval": BATCH_SIZE_EVAL,
    "seed": SEED,
    "best_val_loss": best_val_loss,
}, save_path)

print(f"Saved model checkpoint to {save_path}")

# Inference latency
print("\n" + "=" * 60)
print("INFERENCE LATENCY")
print("=" * 60)

model.eval()

dummy = torch.randn(1, D, T).to(device)

# Warmup runs
for _ in range(50):
    with torch.no_grad():
        model(dummy)

if device.type == "cuda":
    torch.cuda.synchronize()

times = []

for _ in range(1000):
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    times.append(t1 - t0)

times = np.array(times) * 1000

print(f"Per-sample inference: {times.mean():.3f} ms ± {times.std():.3f} ms")
print(f"Throughput: {1000 / times.mean():.0f} samples/sec")