from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Configuration
DATA_DIR = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\Windows\tabular_v3_window_0.95_1.25"
)

# Load data
dfs = [pd.read_parquet(p) for p in sorted(DATA_DIR.glob("features_part*.parquet"))]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded: {df.shape}")


# Extract group
def extract_operating_point(base_name: str) -> str:
    match = re.search(r"(load=.+)", str(base_name))
    return match.group(1) if match else str(base_name)


df["group"] = df["meta__base"].apply(extract_operating_point)

n_groups = df["group"].nunique()
print(f"Unique operating points (groups): {n_groups}")
print(f"Simulations per group (expected ~34): {len(df) / n_groups:.1f}")

# Prepare features
y = df["flag"].astype(int)

X = df.drop(
    columns=[c for c in df.columns if c.startswith("meta__")]
    + ["flag", "y", "group"],
    errors="ignore",
)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

groups = df["group"].values


# Train and evaluate
def train_and_evaluate(X_train, X_test, y_train, y_test, label=""):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")
    print(f"  Train cascade rate: {y_train.mean():.1%}")
    print(f"  Test cascade rate:  {y_test.mean():.1%}")
    print(f"\nConfusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n{classification_report(y_test, y_pred, digits=4)}")

    return acc


# Random split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

acc_random = train_and_evaluate(
    X_train_r, X_test_r, y_train_r, y_test_r,
    label="RANDOM STRATIFIED SPLIT"
)

# Grouped split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train_g = X.iloc[train_idx]
X_test_g = X.iloc[test_idx]
y_train_g = y.iloc[train_idx]
y_test_g = y.iloc[test_idx]

# Check overlap
train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])
overlap = train_groups & test_groups
print(f"\nGroup overlap check: {len(overlap)} groups in both sets (should be 0)")

acc_grouped = train_and_evaluate(
    X_train_g, X_test_g, y_train_g, y_test_g,
    label="GROUP-AWARE SPLIT"
)

# Compare results
print("\n" + "=" * 70)
print("  COMPARISON")
print("=" * 70)
print(f"  Random split accuracy:  {acc_random:.4f}")
print(f"  Grouped split accuracy: {acc_grouped:.4f}")
print(f"  Drop:                   {acc_random - acc_grouped:.4f}")

if acc_random - acc_grouped > 0.05:
    print("\n  Significant drop (>5%).")
    print("  Possible operating-point leakage.")
    print("  Consider grouped splits.")
elif acc_random - acc_grouped > 0.02:
    print("\n  Moderate drop (2-5%).")
    print("  Some leakage may exist.")
    print("  Grouped splits recommended.")
else:
    print("\n  Minimal drop (<2%).")
    print("  Strong operating-point generalisation.")