from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import re

# Configuration
DATA_DIR = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\Windows\tabular_v3_window_0.95_1.25"
)

# Load data
dfs = [pd.read_parquet(p) for p in sorted(DATA_DIR.glob("features_part*.parquet"))]
df = pd.concat(dfs, ignore_index=True)

print(f"Loaded: {df.shape}")


# Extract fault line
def extract_faulted_line(base_name: str) -> str:
    match = re.search(r"(Line \d+ - \d+)", str(base_name))
    return match.group(1) if match else "Unknown"


df["faulted_line"] = df["meta__base"].apply(extract_faulted_line)

# Prepare features
y = df["flag"].astype(int)

X = df.drop(
    columns=[c for c in df.columns if c.startswith("meta__")]
    + ["flag", "y", "faulted_line"],
    errors="ignore",
)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

faulted_lines = df["faulted_line"].values

# Train-test split
X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
    X, y, faulted_lines,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train XGBoost
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

overall_acc = accuracy_score(y_test, y_pred)
print(f"\nOverall test accuracy: {overall_acc:.4f}")

# Line breakdown
results = []

for line in sorted(set(lines_test)):
    mask = lines_test == line

    y_true_line = y_test.values[mask]
    y_pred_line = y_pred[mask]

    n_total = len(y_true_line)
    n_cascade = int(y_true_line.sum())
    n_non_cascade = n_total - n_cascade
    cascade_rate = n_cascade / n_total if n_total > 0 else 0

    acc = accuracy_score(y_true_line, y_pred_line)

    false_neg = int(((y_true_line == 1) & (y_pred_line == 0)).sum())
    false_pos = int(((y_true_line == 0) & (y_pred_line == 1)).sum())

    results.append({
        "faulted_line": line,
        "n_test": n_total,
        "n_cascade": n_cascade,
        "n_non_cascade": n_non_cascade,
        "cascade_rate": cascade_rate,
        "accuracy": acc,
        "false_neg": false_neg,
        "false_pos": false_pos,
    })

rdf = pd.DataFrame(results).sort_values("cascade_rate")

# Print results
print("\n" + "=" * 100)
print("PERFORMANCE BY FAULTED LINE")
print("=" * 100)
print(f"\n{'Faulted Line':<18s}  {'Tests':>5s}  {'Casc':>5s}  {'Non-C':>5s}  "
      f"{'Casc%':>6s}  {'Acc':>6s}  {'FN':>4s}  {'FP':>4s}  {'Category'}")
print("-" * 100)

for _, row in rdf.iterrows():
    cr = row["cascade_rate"]

    if cr >= 0.9:
        cat = "Almost always cascades"
    elif cr <= 0.1:
        cat = "Almost never cascades"
    else:
        cat = "Mixed"

    print(f"{row['faulted_line']:<18s}  {row['n_test']:5d}  {row['n_cascade']:5d}  "
          f"{row['n_non_cascade']:5d}  {cr:5.1%}  {row['accuracy']:5.1%}  "
          f"{row['false_neg']:4d}  {row['false_pos']:4d}  {cat}")

# Summary statistics
always = rdf[rdf["cascade_rate"] >= 0.9]
never = rdf[rdf["cascade_rate"] <= 0.1]
mixed = rdf[(rdf["cascade_rate"] > 0.1) & (rdf["cascade_rate"] < 0.9)]

print(f"\n\n{'Category':<30s}  {'Lines':>5s}  {'Test samples':>12s}  {'Mean accuracy':>14s}")
print("-" * 70)

for label, subset in [("Almost always cascades", always),
                       ("Almost never cascades", never),
                       ("Mixed (ambiguous)", mixed)]:
    if len(subset) == 0:
        print(f"{label:<30s}  {'0':>5s}  {'-':>12s}  {'-':>14s}")
    else:
        n_lines = len(subset)
        n_samples = int(subset["n_test"].sum())
        w_acc = (subset["accuracy"] * subset["n_test"]).sum() / subset["n_test"].sum()
        print(f"{label:<30s}  {n_lines:5d}  {n_samples:12d}  {w_acc:13.1%}")

print(f"\n{'Overall':<30s}  {len(rdf):5d}  {int(rdf['n_test'].sum()):12d}  {overall_acc:13.1%}")

# Diagnostic summary
print("\n\n" + "=" * 100)
print("DIAGNOSTIC SUMMARY")
print("=" * 100)

if len(mixed) > 0:
    mixed_acc = (mixed["accuracy"] * mixed["n_test"]).sum() / mixed["n_test"].sum()
    mixed_samples = int(mixed["n_test"].sum())

    print(f"\nThe {len(mixed)} 'mixed' lines ({mixed_samples} test samples) have a "
          f"weighted accuracy of {mixed_acc:.1%}.")
    print("If this is much lower than the overall accuracy, the model is largely")
    print("memorising fault identity rather than learning transient dynamics.")
    print(f"\n  Overall accuracy:      {overall_acc:.1%}")
    print(f"  Mixed-lines accuracy:  {mixed_acc:.1%}")
    print(f"  Gap:                   {overall_acc - mixed_acc:.1%}")
else:
    print("\nNo 'mixed' lines found — every faulted line either almost always")
    print("or almost never cascades. The model may just be identifying which")
    print("line was faulted.")

# Save results
out_path = DATA_DIR / "diagnosis_by_faulted_line.csv"
rdf.to_csv(out_path, index=False)

print(f"\nSaved per-line breakdown to: {out_path}")