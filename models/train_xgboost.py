from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import xgboost as xgb
import joblib

DATA_DIR = Path(r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\Windows\Tabular\tabular_v3_window_0.95_2.20")

# Load parquet shards
dfs = [pd.read_parquet(p) for p in sorted(DATA_DIR.glob("features_part*.parquet"))]
df = pd.concat(dfs, ignore_index=True)

print("Loaded:", df.shape)
print("Label counts:\n", df["flag"].value_counts())

# Split features labels
y = df["flag"].astype(int)

X = df.drop(
    columns=[c for c in df.columns if c.startswith("meta__")] + ["flag", "y"],
    errors="ignore"
)

# Ensure numeric values
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Class imbalance
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / max(pos, 1)

print(f"Train neg={neg}, pos={pos}, scale_pos_weight={scale_pos_weight:.3f}")

# Train XGBoost
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

# Evaluate model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 30 features:")
print(importances.head(30).to_string())

# Save outputs
joblib.dump(model, "xgboost_flag01.joblib")
importances.head(200).to_csv("xgboost_top200_features.csv")

print("\nSaved:")
print("- xgboost_flag01.joblib")
print("- xgboost_top200_features.csv")

# Bootstrap intervals
def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, ci=95,
                 seed=42, **metric_kwargs):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    point = metric_fn(y_true, y_score, **metric_kwargs)

    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_score[idx], **metric_kwargs)

    alpha = (100 - ci) / 2
    lo = np.percentile(scores, alpha)
    hi = np.percentile(scores, 100 - alpha)

    return point, lo, hi


print("\n--- 95% Bootstrap CIs (n=2000) ---\n")

# Accuracy
p, lo, hi = bootstrap_ci(y_test, y_pred, accuracy_score)
print(f"Accuracy:                {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Cascade metrics
p, lo, hi = bootstrap_ci(y_test, y_pred, precision_score,
                         pos_label=1, zero_division=0)
print(f"Precision (cascading):   {p:.4f}  [{lo:.4f}, {hi:.4f}]")

p, lo, hi = bootstrap_ci(y_test, y_pred, recall_score, pos_label=1)
print(f"Recall (cascading):      {p:.4f}  [{lo:.4f}, {hi:.4f}]")

p, lo, hi = bootstrap_ci(y_test, y_pred, f1_score, pos_label=1)
print(f"F1 (cascading):          {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Non-cascade metrics
p, lo, hi = bootstrap_ci(y_test, y_pred, precision_score,
                         pos_label=0, zero_division=0)
print(f"Precision (non-casc):    {p:.4f}  [{lo:.4f}, {hi:.4f}]")

p, lo, hi = bootstrap_ci(y_test, y_pred, recall_score, pos_label=0)
print(f"Recall (non-casc):       {p:.4f}  [{lo:.4f}, {hi:.4f}]")

p, lo, hi = bootstrap_ci(y_test, y_pred, f1_score, pos_label=0)
print(f"F1 (non-casc):           {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# ROC-AUC
p, lo, hi = bootstrap_ci(y_test, y_proba, roc_auc_score)
print(f"ROC-AUC:                 {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Save predictions
np.savez(
    "xgboost_W3_predictions.npz",
    y_test=y_test.values,
    y_pred=y_pred,
    y_proba=y_proba,
)

print("\nSaved: xgboost_W3_predictions.npz")