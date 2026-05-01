from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DATA_DIR = Path(r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\Windows\Tabular\tabular_v3_window_0.95_3.20")

# Load parquet shards
dfs = [pd.read_parquet(p) for p in sorted(DATA_DIR.glob("features_part*.parquet"))]
df = pd.concat(dfs, ignore_index=True)

# Features and labels
y = df["flag"].astype(int)
X = df.drop(
    columns=[c for c in df.columns if c.startswith("meta__")] + ["flag", "y"],
    errors="ignore"
)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model pipeline
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs"
    ))
])
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

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

# Cascade precision
p, lo, hi = bootstrap_ci(y_test, y_pred, precision_score, 
                         pos_label=1, zero_division=0)
print(f"Precision (cascading):   {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Cascade recall
p, lo, hi = bootstrap_ci(y_test, y_pred, recall_score, pos_label=1)
print(f"Recall (cascading):      {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Cascade F1
p, lo, hi = bootstrap_ci(y_test, y_pred, f1_score, pos_label=1)
print(f"F1 (cascading):          {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Non-cascade precision
p, lo, hi = bootstrap_ci(y_test, y_pred, precision_score, 
                         pos_label=0, zero_division=0)
print(f"Precision (non-casc):    {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Non-cascade recall
p, lo, hi = bootstrap_ci(y_test, y_pred, recall_score, pos_label=0)
print(f"Recall (non-casc):       {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Non-cascade F1
p, lo, hi = bootstrap_ci(y_test, y_pred, f1_score, pos_label=0)
print(f"F1 (non-casc):           {p:.4f}  [{lo:.4f}, {hi:.4f}]")

# ROC-AUC
p, lo, hi = bootstrap_ci(y_test, y_proba, roc_auc_score)
print(f"ROC-AUC:                 {p:.4f}  [{lo:.4f}, {hi:.4f}]")