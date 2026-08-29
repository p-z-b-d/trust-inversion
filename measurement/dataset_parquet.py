"""
dataset_parquet.py — Parquet loader for the canonical Pi 4B HPC dataset.

The orchestrator writes per-trace .parquet files with raw HPC counters at
1ms sampling resolution (config.SAMPLING_INTERVAL_MS = 1). HMD literature
(Demme 2013, Sayadi 2021) classifies on coarser windows — 10-100ms is
standard. This loader aggregates raw counters into HMD-standard windows
before computing per-window derived rates.

Drop in next to dataset.py and hmd_vanilla.py. Usage:
    from dataset_parquet import load_real_parquet
    ds = load_real_parquet(
        "./data/traces",
        window_aggregation_factor=100,   # 1ms -> 100ms windows
    )

Derived rate definitions (matches analyze_dataset.py's DERIVED_FEATURES,
applied after aggregation):
    IPC          = sum(instructions)  / sum(cpu-cycles)        over the window
    cache_miss   = sum(cache-misses)  / sum(cache-references)  over the window
    branch_miss  = sum(branch-misses) / sum(branch-instructions) over the window

Per-window rate where denominator is 0 is treated as 0 (consistent with
analyze_dataset.safe_rate). NaN/inf from any other edge case -> 0.

Methodology note for the paper
------------------------------
The choice of aggregation factor is a deliberate methodology decision, not
just an optimization. HMD literature universally classifies on windows of
>= 10ms; 1ms is unusually fine and would produce per-window features
dominated by counter quantization noise. Aggregating to 100ms also matches
the Demme 2013 baseline window choice that the vanilla HMD line builds on.
"""

from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd

from dataset import (
    Trace, Dataset,
    CLASS_NAMES, CANONICAL_STATS,
)


DEFAULT_FEATURES = ["IPC", "cache_miss", "branch_miss"]
DEFAULT_AGGREGATION_FACTOR = 100   # 1ms -> 100ms; standard HMD window size

DERIVED_FROM_RAW = {
    "IPC":         ("instructions",  "cpu-cycles"),
    "cache_miss":  ("cache-misses",  "cache-references"),
    "branch_miss": ("branch-misses", "branch-instructions"),
}


# ===========================================================================
# Aggregation
# ===========================================================================

def _aggregate_raw_counters(
    df: pd.DataFrame,
    factor: int,
    required_cols: set,
) -> pd.DataFrame:
    """Sum every `factor` consecutive rows of the required counter columns.

    Returns a new DataFrame with aggregated counter sums; rows past the last
    complete group of `factor` are truncated (canonical traces have >=8000
    samples so this truncation is at most factor-1 samples).
    """
    if factor <= 1:
        return df[list(required_cols)].copy()

    n = len(df)
    n_agg = n // factor
    if n_agg == 0:
        return pd.DataFrame({c: [] for c in required_cols})

    truncated = df.iloc[: n_agg * factor]
    out = {}
    for col in required_cols:
        vals = truncated[col].astype(float).values.reshape(n_agg, factor)
        out[col] = vals.sum(axis=1)
    return pd.DataFrame(out)


def _compute_per_window_rate(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Per-window safe division: rate = num/denom where denom>0, else 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(denom > 0, num / denom, 0.0)
    return np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)


# ===========================================================================
# Loader
# ===========================================================================

def load_real_parquet(
    traces_root: str,
    feature_names=None,
    classes=None,
    skip_classes=None,
    window_aggregation_factor: int = DEFAULT_AGGREGATION_FACTOR,
    verbose: bool = True,
) -> Dataset:
    """Load real Pi traces from parquet, aggregate to HMD windows, compute rates.

    Parameters
    ----------
    traces_root : str
        Directory containing per-class subdirectories of .parquet traces.
    feature_names : list of str, optional
        Subset of {"IPC", "cache_miss", "branch_miss"} to load. Default: all.
    classes : list of str, optional
        If given, only load these class names.
    skip_classes : list of str, optional
        Default ["B4_idle"] (counters all zero by design).
    window_aggregation_factor : int, default 100
        Number of raw 1ms samples to sum into one logical HMD window.
        100 -> 100ms windows (Demme/Sayadi standard).
        Set to 1 to disable aggregation.
    verbose : bool
        Print per-class load counts and windowing summary.

    Returns
    -------
    Dataset
    """
    feature_names = list(feature_names or DEFAULT_FEATURES)
    if skip_classes is None:
        skip_classes = ["B4_idle"]
    if classes is None:
        classes = [c for c in CLASS_NAMES if c not in skip_classes]

    for feat in feature_names:
        if feat not in DERIVED_FROM_RAW:
            raise KeyError(
                f"Unknown derived feature {feat!r}. "
                f"Supported: {sorted(DERIVED_FROM_RAW)}"
            )

    required_raw_cols = set()
    for feat in feature_names:
        num, denom = DERIVED_FROM_RAW[feat]
        required_raw_cols.add(num)
        required_raw_cols.add(denom)

    if verbose:
        print(f"  Loading from: {traces_root}")
        print(f"  Aggregation factor: {window_aggregation_factor} "
              f"(1ms -> {window_aggregation_factor}ms windows)")
        print(f"  Features: {feature_names}")
        print()

    traces = []
    n_loaded_per_class = {}
    first_trace_logged = False

    for cls in classes:
        cls_dir = os.path.join(traces_root, cls)
        if not os.path.isdir(cls_dir):
            if verbose:
                print(f"  WARN: class dir not found, skipping: {cls_dir}")
            continue

        parquet_files = sorted(glob.glob(os.path.join(cls_dir, "*.parquet")))
        n_loaded = 0
        n_skipped = 0

        for pq_path in parquet_files:
            try:
                df = pd.read_parquet(pq_path)
            except Exception as e:
                if verbose:
                    print(f"  WARN: failed to read {pq_path}: {e}")
                n_skipped += 1
                continue

            if len(df) == 0:
                n_skipped += 1
                continue

            missing = required_raw_cols - set(df.columns)
            if missing:
                if verbose:
                    print(f"  WARN: {os.path.basename(pq_path)} "
                          f"missing columns {sorted(missing)}; skipping")
                n_skipped += 1
                continue

            df_agg = _aggregate_raw_counters(
                df, window_aggregation_factor, required_raw_cols,
            )
            if len(df_agg) == 0:
                n_skipped += 1
                continue

            rates = {}
            for feat in feature_names:
                num_col, denom_col = DERIVED_FROM_RAW[feat]
                num = df_agg[num_col].values
                denom = df_agg[denom_col].values
                rates[feat] = _compute_per_window_rate(num, denom)

            windows = np.stack([rates[f] for f in feature_names], axis=1)

            if not first_trace_logged:
                first_trace_logged = True
                if verbose:
                    print(f"  First trace: {len(df)} raw samples -> "
                          f"{windows.shape[0]} aggregated windows")
                    print()

            trace_id = os.path.splitext(os.path.basename(pq_path))[0]
            traces.append(Trace(
                class_name=cls,
                label=CANONICAL_STATS[cls]["label"],
                trace_id=trace_id,
                windows=windows,
                feature_names=list(feature_names),
            ))
            n_loaded += 1

        n_loaded_per_class[cls] = n_loaded
        if verbose:
            extra = f"  ({n_skipped} skipped)" if n_skipped else ""
            print(f"  {cls}: loaded {n_loaded} traces{extra}")

    if verbose:
        non_empty = sum(1 for v in n_loaded_per_class.values() if v > 0)
        total_windows = sum(t.n_windows for t in traces) if traces else 0
        print()
        print(f"  Total: {len(traces)} traces across {non_empty} non-empty classes")
        print(f"  Total windows: {total_windows}")

    return Dataset(traces=traces, feature_names=list(feature_names))


if __name__ == "__main__":
    ds = load_real_parquet(
        "./data/traces",
        window_aggregation_factor=100,
    )
    print()
    print(ds.summary())
