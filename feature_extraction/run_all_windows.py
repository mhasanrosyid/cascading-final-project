"""
run_all_windows.py
==================
Runs extract_features_v3.py logic for all six window configurations
in sequence, outputting each to a separate folder.

Windows:
  Predictive (before most cascades):
    - 0.95 - 1.25 s   (0.4% of cascades started)
    - 0.95 - 1.45 s   (before cascade ramp-up)
    - 0.95 - 1.70 s   (~75% of cascades not yet started)

  Original (detection / mixed):
    - 0.95 - 1.25 s   (same as above, shared)
    - 0.95 - 2.20 s   (~62% of cascades already started)
    - 0.95 - 3.20 s   (~98.5% of cascades already started)

Since 1.25 s appears in both sets, there are 5 unique runs.

Usage:
    python run_all_windows.py
"""

from pathlib import Path
import new_extract as feat

# ──────────────────── PATHS ────────────────────

DATA_FOLDER = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\in1\lhs_results"
)

SUMMARY_CSV = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\results_summary_lhs.csv"
)

BASE_OUT_DIR = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset"
)

# ──────────────────── WINDOW CONFIGS ────────────────────

WINDOWS = [
    # (t_end, folder_suffix, description)
    (1.25, "window_0.95_1.25", "Predictive — 0.4% cascades started"),
    (1.45, "window_0.95_1.45", "Predictive — before cascade ramp-up"),
    (1.70, "window_0.95_1.70", "Predictive — ~75% cascades not started"),
    (2.20, "window_0.95_2.20", "Mixed — ~62% cascades already started"),
    (3.20, "window_0.95_3.20", "Detection — ~98.5% cascades already started"),
]

# ──────────────────── MAIN ────────────────────

def main():
    # Load flag map once (shared across all runs)
    print("Loading flag map from summary CSV...")
    flag_map = feat.load_flag_map_from_summary(SUMMARY_CSV)
    print(f"Flag map contains {len(flag_map)} entries\n")

    for t_end, folder_suffix, desc in WINDOWS:
        print("=" * 70)
        print(f"RUNNING: T_END = {t_end} s")
        print(f"  {desc}")
        print("=" * 70)

        # Override the module-level config
        feat.T_END = t_end

        out_dir = BASE_OUT_DIR / f"tabular_v3_{folder_suffix}"

        feat.build_features_for_folder(
            data_folder=DATA_FOLDER,
            out_dir=out_dir,
            flag_map=flag_map,
        )

        print(f"\nCompleted T_END = {t_end} s -> {out_dir}\n\n")


if __name__ == "__main__":
    main()
