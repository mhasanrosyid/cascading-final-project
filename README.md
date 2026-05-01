# Cascading Failure Detection in Power Systems

This is the codebase for my third-year individual project at the University of Manchester, supervised by Dr Panagiotis Papadopoulos. The project trains machine learning models to detect whether a fault in a power network will propagate into a cascading failure, using only a short observation window of post-fault PMU-style measurements from a modified IEEE-39 bus system simulated in PowerFactory.

The full report goes into the methodology, results and limitations in detail. This README is just here to explain what's in the repo and how the pieces fit together.

## What the project does

Given a multivariate time-series of 251 monitored signals around an initiating three-phase fault at t = 1.0 s, the models output a binary prediction of cascading or non-cascading. Three models are compared:

- **Logistic regression**, a linear baseline on hand-crafted summary features.
- **XGBoost**, a non-linear tree-based baseline on the same features.
- **1D-CNN**, the primary model, trained end-to-end on the raw resampled time-series.

Each model is evaluated across five observation windows ranging from 0.30 s (predictive, almost no cascade trips visible yet) up to 2.25 s (identification, most cascades already in progress). The point of doing this is to characterise how performance changes as you trade lead time against direct cascade evidence.

## Repository layout

```
exploration/              Notebooks for inspecting the dataset
cascade_timing_analysis/  Working out when cascades actually start
feature_extraction/       Building the tabular feature matrix
models/                   Training scripts for LogReg, XGBoost and CNN
FourierTransform.ipynb    Spectral analysis used to pick the 100 Hz resampling rate
find_leak.py              Sanity checks for train/test leakage
diagnose_group_leakage.py Per-group leakage diagnostic
diagnose_by_faulted_line.py  Checks for fault-location signature leakage
top_features.txt          Ranked XGBoost feature importances
```

A short tour of each folder:

**`exploration/`** contains Jupyter notebooks without a single fixed objective. These are the ones I used early on to get a feel for the dataset, looking at things like class balance across the four flag values, distributions of system load and wind generation, and signal integrity checks. A lot of the figures in the early sections of the report came out of here.

**`cascade_timing_analysis/`** extracts the time of the first cascading protection action from each cascading simulation's `results_cascs` file. The cumulative distribution this produced is what defines the predictive vs identification regimes used throughout the report (W1 to W5).

**`feature_extraction/`** holds the scripts that take the raw CSV outputs from PowerFactory, align them to the fault time, resample to 100 Hz, crop to the chosen observation window, and compute the 9 summary statistics per channel (peak deviation, RMS deviation, area under absolute deviation, max rate of change, and so on). Output is written to disk as Parquet shards of 500 rows so the pipeline can resume cleanly if it gets interrupted halfway through.

**`models/`** contains three training scripts, one per model. Logistic regression and XGBoost were run locally on CPU. The CNN was trained on Runpod, using one GPU at a time, either an RTX 3090 or an RTX 4090 depending on what was available.

**Loose files at the root.** `find_leak.py`, `diagnose_group_leakage.py` and `diagnose_by_faulted_line.py` are sanity checks for data leakage between the training and test sets, including the more subtle case where the model could pick up the identity of the initially faulted line rather than the system response. `FourierTransform.ipynb` is the spectral analysis I used to justify the 100 Hz resampling rate (Section 3.4.3 of the report).

## Data

The raw PowerFactory dataset and the generated feature files aren't tracked in this repo because they're far too large for GitHub. The two places they would normally live are:

- A `windows/` directory containing the windowed inputs. This means Parquet shards for the tabular feature matrix used by LogReg and XGBoost, and NumPy and CSV files for the raw multivariate time-series used by the CNN.
- A `results/` directory containing per-window metrics, confusion matrices, and the CNN model checkpoints.

The dataset itself is from Nakas and Papadopoulos (2025).

## Running things

Order of operations, if you wanted to reproduce the pipeline end-to-end:

1. Run the exploration notebooks to confirm the dataset is present and looks right.
2. Run `cascade_timing_analysis/` to generate the first-cascade-time distribution.
3. Run the scripts in `feature_extraction/` to produce the windowed Parquet and NumPy files for each of W1 to W5.
4. Run the leakage diagnostics (`find_leak.py`, `diagnose_group_leakage.py`, `diagnose_by_faulted_line.py`) before training anything.
5. Train the models from `models/`. LogReg and XGBoost are quick on CPU. The CNN expects a CUDA GPU.

If anything in here is unclear or broken, feel free to open an issue.
