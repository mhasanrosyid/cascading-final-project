from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
FAULT_T = 1.0
POST_CLEAR_T = 1.15
T_START = 0.95
TARGET_DT = 0.01

RESULTS_SUFFIX = "results_cascs"

WINDOWS = [
    (1.25, "cnn_window_0.95_1.25"),
    (1.45, "cnn_window_0.95_1.45"),
    (1.70, "cnn_window_0.95_1.70"),
    (2.20, "cnn_window_0.95_2.20"),
    (3.20, "cnn_window_0.95_3.20"),
]

# Paths
DATA_FOLDER = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\in1\lhs_results"
)

SUMMARY_CSV = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\results_summary_lhs.csv"
)

BASE_OUT_DIR = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\cnn_data"
)


# Load labels
def load_flag_map_from_summary(summary_csv: Path) -> dict[str, int]:
    s = pd.read_csv(summary_csv, engine="python", on_bad_lines="skip")
    s = s.dropna(subset=["Line ", "Load", "Wind1", "Wind2", "Wind3", "flag"]).copy()
    s["Line "] = s["Line "].astype(str).str.strip()
    s["flag"] = pd.to_numeric(s["flag"], errors="coerce").astype(int)

    def fmt(x):
        return ("{:.4f}".format(float(x))).rstrip("0").rstrip(".")

    s["base"] = (
        s["Line "]
        + "_load=" + s["Load"].map(fmt)
        + "_wind="
        + s["Wind1"].map(fmt) + "_"
        + s["Wind2"].map(fmt) + "_"
        + s["Wind3"].map(fmt)
    )

    return dict(zip(s["base"], s["flag"]))


# Load CSV
def load_powerfactory_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1])

    flat_cols = []

    for a, b in df.columns:
        a = str(a).strip()
        b = str(b).strip()

        if "Time in s" in b:
            flat_cols.append("Time in s")
        else:
            flat_cols.append(f"{a}__{b}")

    df.columns = flat_cols
    return df


# Preprocess simulation
def preprocess_simulation(df: pd.DataFrame, t_end: float) -> np.ndarray | None:
    t = pd.to_numeric(df["Time in s"], errors="coerce")
    df = df.loc[t.notna()].copy()
    t = t.loc[t.notna()].to_numpy(dtype=float)

    margin = 0.01
    mask = (t >= T_START - margin) & (t <= t_end + margin)

    if mask.sum() < 10:
        return None

    df_w = df.loc[mask].reset_index(drop=True)
    t_raw = pd.to_numeric(df_w["Time in s"], errors="coerce").to_numpy(dtype=float)

    X_raw = df_w.drop(columns=["Time in s"])
    signal_cols = X_raw.columns.tolist()

    for c in signal_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    X_raw = X_raw.ffill().bfill().fillna(0.0)

    t_full = np.arange(T_START, t_end + TARGET_DT * 0.5, TARGET_DT)

    D = len(signal_cols)
    X_full = np.zeros((len(t_full), D), dtype=np.float32)
    raw_vals = X_raw.to_numpy(dtype=float)

    for d in range(D):
        X_full[:, d] = np.interp(t_full, t_raw, raw_vals[:, d])

    pre_mask = t_full < FAULT_T

    if pre_mask.sum() > 0:
        baseline = X_full[pre_mask].mean(axis=0)
    else:
        baseline = X_full[0]

    X_full -= baseline

    post_mask = t_full >= POST_CLEAR_T
    X_post = X_full[post_mask]

    return X_post


# File helpers
def is_results_file(path: Path) -> bool:
    return path.stem.endswith(RESULTS_SUFFIX)


def get_base_stem(path: Path) -> str:
    stem = path.stem

    if stem.endswith(RESULTS_SUFFIX):
        return stem[: -len(RESULTS_SUFFIX)]

    return stem


# Main routine
def main():
    print("Loading flag map...")
    flag_map = load_flag_map_from_summary(SUMMARY_CSV)
    print(f"Flag map: {len(flag_map)} entries")

    # Discover files
    all_csvs = sorted(DATA_FOLDER.rglob("*.csv"))
    data_files = [p for p in all_csvs if not is_results_file(p)]
    print(f"Found {len(data_files)} data CSV files\n")

    # Valid files
    valid_files = []

    for data_path in data_files:
        base = get_base_stem(data_path)

        if "_" in base and base.split("_", 1)[0].isdigit():
            base = base.split("_", 1)[1]

        flag = flag_map.get(base, None)

        if flag in (0, 1):
            valid_files.append((data_path, base, flag))

    print(f"Valid labelled files: {len(valid_files)}")

    # Channel names
    df_sample = load_powerfactory_csv(valid_files[0][0])
    channel_names = [c for c in df_sample.columns if c != "Time in s"]
    D = len(channel_names)
    print(f"Channels (D): {D}")

    # Process windows
    for t_end, folder_name in WINDOWS:

        T_post = len(np.arange(POST_CLEAR_T, t_end + TARGET_DT * 0.5, TARGET_DT))
        print(f"\n{'=' * 70}")
        print(f"Window: [0.95, {t_end}] s → post-clearance [{POST_CLEAR_T}, {t_end}] s")
        print(f"Expected shape per sample: ({T_post}, {D})")

        size_gb = len(valid_files) * T_post * D * 4 / 1e9
        print(f"Estimated X.npy size: {size_gb:.1f} GB")
        print(f"{'=' * 70}")

        out_dir = BASE_OUT_DIR / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Existing outputs
        if (out_dir / "X.npy").exists() and (out_dir / "y.npy").exists():
            print(f"Already exists, skipping. Delete {out_dir} to regenerate.")
            continue

        N = len(valid_files)
        X_all = np.zeros((N, T_post, D), dtype=np.float32)
        y_all = np.zeros(N, dtype=np.int32)
        meta_rows = []

        processed = 0
        failed = 0

        for i, (data_path, base, flag) in enumerate(valid_files):
            try:
                df_raw = load_powerfactory_csv(data_path)
                X_sample = preprocess_simulation(df_raw, t_end)

                if X_sample is None:
                    failed += 1
                    continue

                # Trim length
                if X_sample.shape[0] > T_post:
                    X_sample = X_sample[:T_post]

                elif X_sample.shape[0] < T_post:
                    # Pad final value
                    pad = np.tile(X_sample[-1:], (T_post - X_sample.shape[0], 1))
                    X_sample = np.vstack([X_sample, pad])

                X_all[processed] = X_sample
                y_all[processed] = flag
                meta_rows.append({"idx": processed, "base": base, "flag": flag})
                processed += 1

            except Exception as e:
                failed += 1

                if failed <= 5:
                    print(f"  [FAIL] {data_path.name}: {repr(e)}")

            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{N} | processed={processed} | failed={failed}")

        # Trim arrays
        X_all = X_all[:processed]
        y_all = y_all[:processed]

        # Save arrays
        np.save(out_dir / "X.npy", X_all)
        np.save(out_dir / "y.npy", y_all)

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(out_dir / "meta.csv", index=False)

        # Save channel names
        pd.Series(channel_names).to_csv(out_dir / "channel_names.csv", index=False)

        print(f"\n  Saved to {out_dir}:")
        print(f"    X.npy:  {X_all.shape} ({X_all.nbytes / 1e9:.2f} GB)")
        print(f"    y.npy:  {y_all.shape}")
        print(f"    Labels: {(y_all == 0).sum()} non-cascade, {(y_all == 1).sum()} cascade")
        print(f"    Failed: {failed}")


if __name__ == "__main__":
    main()