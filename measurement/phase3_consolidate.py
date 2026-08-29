"""
Phase 3 closeout — consolidated operating-point figure + Pi inference timing
=============================================================================

Runs all four HPC classifiers under consistent CV (5-fold, cv_seed=0) on real
Pi data and produces:

  1. `phase3_operating_points.png` — two-panel publication figure:
       Left:  per-window (TPR, FPR) scatter for each classifier across folds
       Right: per-attack-class trace TPR comparison

  2. `phase3_runtime.txt` — per-window inference latency on Cortex-A72,
       measured on a held-out 20% test fold with 5 timing repeats.

  3. Terminal: consolidated operating-point table (the numbers that feed the
     Phase 4 composition grid).

This is the narrative anchor for Section 5 of the paper. After this figure
the reader knows what we measured, before Section 6 (composition grid)
explains what happens when these classifiers are composed with trust schemes.

Run from ~/phase3_classifiers/ on the Pi:
    source ~/research_venv/bin/activate
    python3 phase3_consolidate.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ATTACK_CLASSES, stratified_kfold_traces
from dataset_parquet import load_real_parquet
from hmd_vanilla import (
    VanillaHMDConfig, fit_vanilla_hmd, cross_validate_vanilla_hmd,
    summarize_cv, windows_and_labels,
)
from hmd_rhmd import RHMDConfig, fit_rhmd, cross_validate_rhmd
from hmd_mtd import MTDHMDConfig, fit_mtd_hmd, cross_validate_mtd_hmd


# ===========================================================================
# Classifier configurations to compare
# ===========================================================================

@dataclass
class ClassifierSpec:
    label: str
    short_label: str         # used in figure legend (shorter)
    color: str
    marker: str
    kind: str                # "vanilla", "rhmd", "mtd"
    cfg_kwargs: dict

CLASSIFIER_SPECS = [
    ClassifierSpec(
        label="Vanilla LogReg",
        short_label="Vanilla LogReg",
        color="tab:red", marker="o",
        kind="vanilla",
        cfg_kwargs={"classifier": "logreg", "seed": 0},
    ),
    ClassifierSpec(
        label="Vanilla RF",
        short_label="Vanilla RF",
        color="tab:blue", marker="s",
        kind="vanilla",
        cfg_kwargs={"classifier": "rf", "seed": 0},
    ),
    ClassifierSpec(
        label="RHMD (n=5)",
        short_label="RHMD n=5",
        color="tab:green", marker="^",
        kind="rhmd",
        cfg_kwargs={
            "n_classifiers": 5,
            "base_classifiers": ["logreg", "rf", "svm", "mlp"],
            "bootstrap_train": True,
            "seed": 0,
            "inference_seed": 1000,
        },
    ),
    ClassifierSpec(
        label="MTD (switch=5)",
        short_label="MTD switch=5",
        color="tab:purple", marker="D",
        kind="mtd",
        cfg_kwargs={
            "base_classifier": "rf",
            "switching_interval": 5,
            "switching_policy": "round_robin",
            "seed": 0,
            "switching_seed": 2000,
        },
    ),
]


# ===========================================================================
# CV runners (dispatch by kind)
# ===========================================================================

def run_cv(spec: ClassifierSpec, dataset, n_folds=5, cv_seed=0):
    if spec.kind == "vanilla":
        cfg = VanillaHMDConfig(**spec.cfg_kwargs)
        return cross_validate_vanilla_hmd(
            dataset, cfg=cfg, n_folds=n_folds, cv_seed=cv_seed, verbose=False,
        )
    if spec.kind == "rhmd":
        cfg = RHMDConfig(**spec.cfg_kwargs)
        return cross_validate_rhmd(
            dataset, cfg=cfg, n_folds=n_folds, cv_seed=cv_seed, verbose=False,
        )
    if spec.kind == "mtd":
        cfg = MTDHMDConfig(**spec.cfg_kwargs)
        return cross_validate_mtd_hmd(
            dataset, cfg=cfg, n_folds=n_folds, cv_seed=cv_seed, verbose=False,
        )
    raise ValueError(f"Unknown kind: {spec.kind}")


def fit_for_timing(spec: ClassifierSpec, train_traces):
    """Return a fitted classifier with a predict_proba_windows(X) method."""
    if spec.kind == "vanilla":
        cfg = VanillaHMDConfig(**spec.cfg_kwargs)
        return fit_vanilla_hmd(train_traces, cfg=cfg)
    if spec.kind == "rhmd":
        cfg = RHMDConfig(**spec.cfg_kwargs)
        return fit_rhmd(train_traces, cfg=cfg)
    if spec.kind == "mtd":
        cfg = MTDHMDConfig(**spec.cfg_kwargs)
        return fit_mtd_hmd(train_traces, cfg=cfg)
    raise ValueError(f"Unknown kind: {spec.kind}")


# ===========================================================================
# Inference timing
# ===========================================================================

def time_inference(fit, X_test, n_repeats: int = 5) -> dict:
    """Median per-window inference latency in microseconds.

    Warmup pass discarded; median of n_repeats reported (robust to OS jitter).
    """
    # Warmup — populate any sklearn caches, JIT-like warmup
    _ = fit.predict_proba_windows(X_test)

    times_s = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = fit.predict_proba_windows(X_test)
        t1 = time.perf_counter()
        times_s.append(t1 - t0)

    times_s = np.asarray(times_s)
    median_s = float(np.median(times_s))
    min_s = float(times_s.min())
    n_windows = int(X_test.shape[0])

    return {
        "n_windows": n_windows,
        "median_total_s": median_s,
        "min_total_s": min_s,
        "median_per_window_us": (median_s / n_windows) * 1e6,
        "min_per_window_us": (min_s / n_windows) * 1e6,
        "n_repeats": n_repeats,
    }


# ===========================================================================
# Main: run CV, time inference, plot, save
# ===========================================================================

def main():
    # 1. Load real Pi data
    print("=" * 78)
    print("PHASE 3 CLOSEOUT — consolidated operating-point + Pi runtime")
    print("=" * 78)
    ds = load_real_parquet(
        "./data/traces",
        window_aggregation_factor=100,
        verbose=True,
    )
    print()

    # 2. Run CV for each classifier
    cv_results: dict[str, list] = {}
    summaries: dict[str, dict] = {}
    for spec in CLASSIFIER_SPECS:
        print(f"  CV: {spec.label} ...", flush=True)
        t0 = time.perf_counter()
        results = run_cv(spec, ds, n_folds=5, cv_seed=0)
        t1 = time.perf_counter()
        cv_results[spec.label] = results
        summaries[spec.label] = summarize_cv(results)
        print(f"    done in {t1 - t0:.1f}s — "
              f"win (TPR, FPR) = ({summaries[spec.label]['window_tpr'][0]:.4f}, "
              f"{summaries[spec.label]['window_fpr'][0]:.4f}), "
              f"tr AUC = {summaries[spec.label]['trace_auc'][0]:.4f}")

    # 3. Time inference on a held-out fold (use first CV fold for consistency)
    print()
    print("=" * 78)
    print("Pi INFERENCE TIMING — Cortex-A72")
    print("=" * 78)
    # Use the same fold-0 train/test split as the CV
    fold_iter = list(stratified_kfold_traces(ds, n_folds=5, seed=0))
    train_traces, test_traces = fold_iter[0]
    X_test, _, _ = windows_and_labels(test_traces, binary=True)
    print(f"  Train traces: {len(train_traces)}    "
          f"Test traces: {len(test_traces)}    "
          f"Test windows: {X_test.shape[0]}")
    print()

    timings: dict[str, dict] = {}
    for spec in CLASSIFIER_SPECS:
        print(f"  Timing: {spec.label} ...", flush=True)
        fit = fit_for_timing(spec, train_traces)
        t = time_inference(fit, X_test, n_repeats=5)
        timings[spec.label] = t
        print(f"    per-window: median {t['median_per_window_us']:.1f}µs, "
              f"best {t['min_per_window_us']:.1f}µs "
              f"(over {t['n_repeats']} repeats × {t['n_windows']} windows)")

    # 4. Consolidated operating-point table
    print()
    print("=" * 84)
    print("PHASE 3 OPERATING-POINT SPECTRUM  (feeds Phase 4 composition grid)")
    print("=" * 84)
    print(f"{'Classifier':<22s} {'win TPR':>10s} {'win FPR':>10s} "
          f"{'tr AUC':>10s} {'µs/win':>10s} {'µs/win (best)':>15s}")
    print("-" * 84)
    for spec in CLASSIFIER_SPECS:
        s = summaries[spec.label]
        t = timings[spec.label]
        print(f"{spec.label:<22s} "
              f"{s['window_tpr'][0]:>10.4f} {s['window_fpr'][0]:>10.4f} "
              f"{s['trace_auc'][0]:>10.4f} "
              f"{t['median_per_window_us']:>10.1f} "
              f"{t['min_per_window_us']:>15.1f}")

    # 5. Write runtime to file (for paper reference)
    with open("phase3_runtime.txt", "w") as f:
        f.write("Phase 3 Pi-side inference timing — Cortex-A72\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test windows per inference call: {timings[CLASSIFIER_SPECS[0].label]['n_windows']}\n")
        f.write(f"Repeats per classifier: 5 (median reported)\n\n")
        f.write(f"{'Classifier':<22s} {'median µs/win':>15s} {'best µs/win':>15s}\n")
        f.write("-" * 60 + "\n")
        for spec in CLASSIFIER_SPECS:
            t = timings[spec.label]
            f.write(f"{spec.label:<22s} "
                    f"{t['median_per_window_us']:>15.1f} "
                    f"{t['min_per_window_us']:>15.1f}\n")
    print()
    print("Runtime table saved: phase3_runtime.txt")

    # 6. Consolidated figure
    plot_operating_points(cv_results, summaries,
                          output_path="phase3_operating_points.png")


# ===========================================================================
# Plotting
# ===========================================================================

def plot_operating_points(
    cv_results: dict, summaries: dict,
    output_path: str = "phase3_operating_points.png",
) -> None:
    """Two-panel figure: ROC-style scatter (left) + per-attack TPR (right)."""
    fig, (ax_roc, ax_per_attack) = plt.subplots(1, 2, figsize=(15, 6))

    # ----- Left panel: per-window ROC-style scatter -----
    # Each classifier: small dots per fold + large marker at mean.
    for spec in CLASSIFIER_SPECS:
        results = cv_results[spec.label]
        fold_tpr = np.array([r.window_metrics["tpr"] for r in results])
        fold_fpr = np.array([r.window_metrics["fpr"] for r in results])

        # Per-fold light dots
        ax_roc.scatter(fold_fpr, fold_tpr,
                       color=spec.color, marker=spec.marker, s=45, alpha=0.45,
                       edgecolors="none")
        # Mean as large outlined marker
        mean_tpr = fold_tpr.mean()
        mean_fpr = fold_fpr.mean()
        legend_label = (
            f"{spec.short_label}\n"
            f"({mean_tpr:.3f}, {mean_fpr:.3f})"
        )
        ax_roc.scatter([mean_fpr], [mean_tpr],
                       color=spec.color, marker=spec.marker, s=220,
                       edgecolors="black", linewidths=1.5,
                       label=legend_label, zorder=3)

    # Random-classifier diagonal
    ax_roc.plot([0, 1], [0, 1], "k:", alpha=0.4, linewidth=1, label="Random")
    ax_roc.set_xlim(-0.02, 1.0)
    ax_roc.set_ylim(0.5, 1.02)
    ax_roc.set_xlabel("Per-window FPR")
    ax_roc.set_ylabel("Per-window TPR")
    ax_roc.set_title("Per-window operating points\n"
                     "(5 small dots per classifier = per-fold; large = mean)")
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ----- Right panel: per-attack-class trace TPR -----
    n_attacks = len(ATTACK_CLASSES)
    width = 0.8 / len(CLASSIFIER_SPECS)
    x_pos = np.arange(n_attacks)

    for i, spec in enumerate(CLASSIFIER_SPECS):
        summary = summaries[spec.label]
        tprs = [summary["per_attack_tpr"].get(c, (np.nan, 0))[0]
                for c in ATTACK_CLASSES]
        ax_per_attack.bar(x_pos + i * width - 0.4 + width / 2, tprs,
                          width=width,
                          label=spec.short_label, color=spec.color,
                          edgecolor="black", linewidth=0.5)

    ax_per_attack.set_xticks(x_pos)
    ax_per_attack.set_xticklabels(
        [c.replace("_", "\n", 1) for c in ATTACK_CLASSES], fontsize=9,
    )
    ax_per_attack.set_ylabel("Per-attack-class trace TPR")
    ax_per_attack.set_title("Per-attack detection across classifiers")
    ax_per_attack.set_ylim(0, 1.08)
    ax_per_attack.grid(alpha=0.3, axis="y")
    ax_per_attack.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Phase 3 closeout — HPC classifier operating-point spectrum on real Pi traces\n"
        "Feeds Phase 4 composition grid: four distinct (TPR, FPR) points × four trust schemes",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


if __name__ == "__main__":
    main()
