from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Configuration
DATA_FOLDER = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\Dataset_Project\LHS\LHS\in1\lhs_results"
)

OUT_DIR = Path(
    r"C:\Users\M Hasan Rosyid\Desktop\PowerSystemDataset\cascade_timing_analysis"
)

RESULTS_SUFFIX = "results_cascs"


# Extract fault line
def extract_faulted_line(filename_stem: str) -> str:
    name = filename_stem.replace(RESULTS_SUFFIX, "")
    match = re.match(r"^\d+_(Line \d+ - \d+)", name)

    if match:
        return match.group(1)

    match = re.search(r"(Line \d+ - \d+)", name)

    if match:
        return match.group(1)

    return ""


# Parse cascade file
def parse_cascs_file(path: Path) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except Exception:
        return []

    if not raw:
        return []

    events = []

    # Tuple extraction
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, raw)

    for m in matches:
        parts = [p.strip().strip("'\"") for p in m.split(",")]

        # Two-field tuple
        if len(parts) == 2:
            component = parts[0].strip()

            try:
                t = float(parts[1].strip())
            except ValueError:
                continue

            events.append({"component": component, "reason": "", "time": t})

        # Three-field tuple
        elif len(parts) == 3:
            component = parts[0].strip()
            reason = parts[1].strip()

            try:
                t = float(parts[2].strip())
            except ValueError:
                continue

            events.append({"component": component, "reason": reason, "time": t})

    events.sort(key=lambda e: e["time"])
    return events


# Check initiating event
def is_initiating_clearance(event: dict, faulted_line: str) -> bool:
    if not faulted_line:
        return False

    return event["component"].startswith(faulted_line)


# Main routine
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Locate files
    cascs_files = sorted(DATA_FOLDER.rglob(f"*{RESULTS_SUFFIX}*.csv"))
    print(f"Found {len(cascs_files)} results_cascs files")

    if len(cascs_files) == 0:
        print("No results_cascs files found. Check DATA_FOLDER path.")
        return

    # Sample parsing
    print("\n── Sample file ──")
    sample = cascs_files[0]
    print(f"File: {sample.name}")

    faulted = extract_faulted_line(sample.stem)
    print(f"Faulted line: {faulted}")

    events = parse_cascs_file(sample)

    for e in events:
        is_init = is_initiating_clearance(e, faulted)
        tag = " [INITIATING - excluded]" if is_init else ""
        print(f"  t={e['time']:.2f}s  {e['component']:30s}  {e['reason']}{tag}")

    # Process files
    records = []

    for i, cfile in enumerate(cascs_files, 1):
        events = parse_cascs_file(cfile)

        if not events:
            continue

        faulted_line = extract_faulted_line(cfile.stem)

        # Remove fault clearance
        cascade_events = [
            e for e in events
            if not is_initiating_clearance(e, faulted_line)
        ]

        if not cascade_events:
            continue

        first = cascade_events[0]

        # Clean simulation name
        stem = cfile.stem.replace(RESULTS_SUFFIX, "")

        if "_" in stem and stem.split("_", 1)[0].isdigit():
            stem = stem.split("_", 1)[1]

        records.append({
            "simulation": stem,
            "faulted_line": faulted_line,
            "first_cascade_time": first["time"],
            "first_cascade_component": first["component"],
            "first_cascade_reason": first["reason"],
            "num_cascade_events": len(cascade_events),
        })

        if i % 2000 == 0:
            print(f"Processed {i}/{len(cascs_files)} | cascading: {len(records)}")

    print(f"\nSimulations with cascading events: {len(records)}")

    if not records:
        print("No cascading events found.")
        return

    df = pd.DataFrame(records)
    times = df["first_cascade_time"].values

    # Summary statistics
    print("\n" + "=" * 60)
    print("FIRST CASCADE EVENT TIME DISTRIBUTION")
    print("(excludes initiating fault clearance)")
    print("=" * 60)
    print(f"  Count:       {len(times)}")
    print(f"  Min:         {np.min(times):.3f} s")
    print(f"  5th pctl:    {np.percentile(times, 5):.3f} s")
    print(f"  10th pctl:   {np.percentile(times, 10):.3f} s")
    print(f"  25th pctl:   {np.percentile(times, 25):.3f} s")
    print(f"  Median:      {np.median(times):.3f} s")
    print(f"  75th pctl:   {np.percentile(times, 75):.3f} s")
    print(f"  90th pctl:   {np.percentile(times, 90):.3f} s")
    print(f"  95th pctl:   {np.percentile(times, 95):.3f} s")
    print(f"  Max:         {np.max(times):.3f} s")
    print(f"  Mean:        {np.mean(times):.3f} s")
    print(f"  Std:         {np.std(times):.3f} s")

    # Window counts
    print("\n── Cascades occurring BEFORE window end ──")
    print("  (model sees the trip itself, not predicting it)")

    for t_end in [1.15, 1.25, 2.2, 3.2, 4.0, 5.0, 10.0]:
        n_before = np.sum(times <= t_end)
        pct = 100.0 * n_before / len(times)
        print(f"  First cascade <= {t_end:5.2f} s:  {n_before:6d}  ({pct:5.1f}%)")

    # Lead times
    lead_times = times - 3.2
    n_positive = np.sum(lead_times > 0)

    print(f"\n── Lead time (first cascade - 3.2 s window end) ──")
    print(f"  Cascades AFTER 3.2 s: {n_positive} ({100 * n_positive / len(times):.1f}%)")

    if n_positive > 0:
        pos_leads = lead_times[lead_times > 0]
        print(f"  Min lead time:    {np.min(pos_leads):.3f} s")
        print(f"  Median lead time: {np.median(pos_leads):.3f} s")
        print(f"  Mean lead time:   {np.mean(pos_leads):.3f} s")

    # Save CSV
    csv_path = OUT_DIR / "first_cascade_times.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(times, bins=120, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Time of first cascading event (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Full range")
    axes[0].axvline(x=3.2, color="red", ls="--", lw=1.5, label="Window end (3.2 s)")
    axes[0].legend()

    t_zoom = times[times <= 10]
    axes[1].hist(t_zoom, bins=100, edgecolor="black", alpha=0.7, color="darkorange")
    axes[1].set_xlabel("Time of first cascading event (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Zoomed: 0–10 s")

    for t_end, col in [(1.15, "green"), (1.25, "blue"), (2.2, "purple"), (3.2, "red")]:
        axes[1].axvline(x=t_end, ls="--", lw=1.2, color=col, label=f"{t_end} s")

    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "cascade_time_distribution.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved histogram to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()