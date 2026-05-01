from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
FAULT_T = 1.0
POST_CLEAR_T = 1.15

T_START = 0.95
T_END = 3.2

RAW_DT = 0.001
TARGET_DT = 0.01

EPS = 1e-12
CHUNK_SIZE = 500
RESULTS_SUFFIX = "results_cascs"


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


# Slice and resample
def slice_and_resample(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    t = pd.to_numeric(df["Time in s"], errors="coerce")
    df = df.loc[t.notna()].copy()
    t = t.loc[t.notna()].to_numpy(dtype=float)

    margin = 0.01
    mask = (t >= T_START - margin) & (t <= T_END + margin)
    df_w = df.loc[mask].reset_index(drop=True)
    t_raw = pd.to_numeric(df_w["Time in s"], errors="coerce").to_numpy(dtype=float)

    X_raw = df_w.drop(columns=["Time in s"])
    signal_cols = X_raw.columns.tolist()

    for c in signal_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    X_raw = X_raw.ffill().bfill().fillna(0.0)

    t_uniform = np.arange(T_START, T_END + TARGET_DT * 0.5, TARGET_DT)

    X_uniform = np.zeros((len(t_uniform), len(signal_cols)))

    for d, col in enumerate(signal_cols):
        X_uniform[:, d] = np.interp(t_uniform, t_raw, X_raw[col].to_numpy(dtype=float))

    X_out = pd.DataFrame(X_uniform, columns=signal_cols)
    return t_uniform, X_out


# Channel features
def channel_summary_features(t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    pre_mask = (t >= T_START) & (t < FAULT_T)
    x_pre = x[pre_mask] if np.any(pre_mask) else x[:max(1, len(x) // 10)]
    baseline = float(np.mean(x_pre))

    post_mask = (t >= POST_CLEAR_T) & (t <= T_END)
    t_post = t[post_mask]
    x_post = x[post_mask]

    if len(x_post) < 2:
        return {k: 0.0 for k in [
            "min_dev", "max_dev", "ptp", "max_abs_dev",
            "rms_dev", "auc_abs_dev", "max_abs_dx",
            "t_at_max_abs_dev", "final_dev",
        ]}

    dev = x_post - baseline
    abs_dev = np.abs(dev)

    dx = np.diff(x_post) / np.diff(t_post)
    max_abs_dx = float(np.max(np.abs(dx))) if len(dx) else 0.0

    dt = float(t_post[1] - t_post[0]) if len(t_post) >= 2 else 0.0

    idx_peak = int(np.argmax(abs_dev))
    t_at_peak = float(t_post[idx_peak] - FAULT_T)

    return {
        "min_dev": float(np.min(dev)),
        "max_dev": float(np.max(dev)),
        "ptp": float(np.max(dev) - np.min(dev)),
        "max_abs_dev": float(np.max(abs_dev)),
        "rms_dev": float(np.sqrt(np.mean(dev ** 2))),
        "auc_abs_dev": float(np.sum(abs_dev) * dt),
        "max_abs_dx": max_abs_dx,
        "t_at_max_abs_dev": t_at_peak,
        "final_dev": float(dev[-1]),
    }


# Featurise simulation
def featurise_data_csv(data_csv_path: Path) -> dict:
    df_raw = load_powerfactory_csv(data_csv_path)
    t, X = slice_and_resample(df_raw)

    row = {}

    for col in X.columns:
        feats = channel_summary_features(t, X[col].to_numpy(dtype=float))

        for k, v in feats.items():
            row[f"{col}__{k}"] = v

    return row


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


# File helpers
def is_results_file(path: Path) -> bool:
    return path.stem.endswith(RESULTS_SUFFIX)


def get_base_stem(path: Path) -> str:
    stem = path.stem

    if stem.endswith(RESULTS_SUFFIX):
        return stem[: -len(RESULTS_SUFFIX)]

    return stem


def load_done_set(done_json: Path) -> set[str]:
    if not done_json.exists():
        return set()

    try:
        return set(json.loads(done_json.read_text()))
    except Exception:
        return set()


def save_done_set(done_json: Path, done: set[str]) -> None:
    done_json.write_text(json.dumps(sorted(done), indent=2))


# Build feature shards
def build_features_for_folder(
    data_folder: Path,
    out_dir: Path,
    flag_map: dict[str, int],
    chunk_size: int = CHUNK_SIZE,
) -> None:
    data_folder = data_folder.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    done_json = out_dir / "done.json"
    done = load_done_set(done_json)

    all_csvs = sorted(data_folder.rglob("*.csv"))
    data_files = [p for p in all_csvs if not is_results_file(p)]

    print(f"Found {len(data_files)} data CSV files in: {data_folder}")
    print(f"Output directory: {out_dir}")
    print(f"Already done: {len(done)}")
    print(f"Window: [{T_START}, {T_END}] s  |  Post-clearance start: {POST_CLEAR_T} s")

    buffer_rows = []
    part_idx = 0

    existing_parts = sorted(out_dir.glob("features_part*.parquet"))

    if existing_parts:
        last = existing_parts[-1].stem

        try:
            part_idx = int(last.replace("features_part", "")) + 1
        except Exception:
            part_idx = len(existing_parts)

    processed_now = 0
    skipped = 0
    failed = 0

    for i, data_path in enumerate(data_files, 1):
        base = get_base_stem(data_path)

        if "_" in base and base.split("_", 1)[0].isdigit():
            base = base.split("_", 1)[1]

        flag = flag_map.get(base, None)

        if flag not in (0, 1):
            skipped += 1
            continue

        if base in done:
            skipped += 1
            continue

        try:
            feat_row = featurise_data_csv(data_path)

            # Metadata fields
            feat_row["meta__base"] = base
            feat_row["meta__data_file"] = str(data_path)
            feat_row["meta__t_start"] = T_START
            feat_row["meta__t_end"] = T_END
            feat_row["meta__post_clear"] = POST_CLEAR_T
            feat_row["meta__dt"] = TARGET_DT

            feat_row["flag"] = flag
            feat_row["y"] = flag

            buffer_rows.append(feat_row)
            done.add(base)
            processed_now += 1

        except Exception as e:
            failed += 1
            (out_dir / "errors.log").open("a", encoding="utf-8").write(
                f"[FAIL] {data_path} :: {repr(e)}\n"
            )

        if len(buffer_rows) >= chunk_size:
            df = pd.DataFrame(buffer_rows)
            out_path = out_dir / f"features_part{part_idx:03d}.parquet"
            df.to_parquet(out_path, index=False)
            save_done_set(done_json, done)

            print(f"  Wrote {len(buffer_rows)} rows -> {out_path.name} | total done={len(done)}")

            buffer_rows.clear()
            part_idx += 1

        if i % 200 == 0:
            print(f"Progress: scanned={i}/{len(data_files)} | new={processed_now} | skipped={skipped} | failed={failed}")

    if buffer_rows:
        df = pd.DataFrame(buffer_rows)
        out_path = out_dir / f"features_part{part_idx:03d}.parquet"
        df.to_parquet(out_path, index=False)
        save_done_set(done_json, done)

        print(f"  Wrote {len(buffer_rows)} rows -> {out_path.name} | total done={len(done)}")

    print("\nDONE")
    print(f"New processed this run: {processed_now}")
    print(f"Skipped (already done or no label): {skipped}")
    print(f"Failed (logged): {failed}")
    print(f"Shards in: {out_dir}")


if __name__ == "__main__":

    DATA_FOLDER = Path(
        r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\in1\lhs_results"
    )

    OUT_DIR = Path(
        r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\tabulardata_flag01_v3"
    )

    SUMMARY_CSV = Path(
        r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\results_summary_lhs.csv"
    )

    flag_map = load_flag_map_from_summary(SUMMARY_CSV)

    # Sanity check
    first_file = next(iter(sorted(DATA_FOLDER.glob("*.csv"))))
    base = first_file.stem

    if "_" in base and base.split("_", 1)[0].isdigit():
        base = base.split("_", 1)[1]

    print("first file base:", base)
    print("flag lookup:", flag_map.get(base, None))

    build_features_for_folder(DATA_FOLDER, OUT_DIR, flag_map)