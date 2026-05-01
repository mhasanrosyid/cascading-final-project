import pandas as pd
import numpy as np
from pathlib import Path
import random

# Configuration
DATA_DIR = Path(r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\in1\lhs_results")

NUM_FILES_TO_SCAN = 40000

ZERO_THRESHOLD = 1e-4
FAULT_TIME = 1.0


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


# Find leaky columns
def find_leaky_columns():
    all_files = list(DATA_DIR.glob("*.csv"))
    data_files = [f for f in all_files if "results_cascs" not in f.name and "summary" not in f.name]

    if len(data_files) > NUM_FILES_TO_SCAN:
        data_files = random.sample(data_files, NUM_FILES_TO_SCAN)

    print(f"Scanning {len(data_files)} files to identify leaky features...")

    leaky_features = set()

    for i, file_path in enumerate(data_files):
        try:
            df = load_powerfactory_csv(file_path)

            # Post-fault data
            t = pd.to_numeric(df["Time in s"], errors="coerce").to_numpy()
            post_fault_mask = t > FAULT_TIME
            df_post = df.iloc[post_fault_mask]

            for col in df_post.columns:
                if col == "Time in s" or "Position" in col:
                    continue

                vals = df_post[col].to_numpy()

                if len(vals) == 0:
                    continue

                # Zero-ending signal
                if abs(vals[-1]) < ZERO_THRESHOLD:
                    leaky_features.add(col)

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data_files)}... Found {len(leaky_features)} leaky features so far.")

    # Save blacklist
    leaky_features_list = sorted(list(leaky_features))

    out_file = Path("leaky_features.txt")

    with open(out_file, "w") as f:
        for feature in leaky_features_list:
            f.write(f"{feature}\n")

    print("\n--- SCAN COMPLETE ---")
    print(f"Identified {len(leaky_features_list)} features that drop to zero.")
    print(f"Blacklist saved to: {out_file.absolute()}")

    print("\nSample of leaky features found:")

    for f in leaky_features_list[:10]:
        print(f" - {f}")


if __name__ == "__main__":
    find_leaky_columns()