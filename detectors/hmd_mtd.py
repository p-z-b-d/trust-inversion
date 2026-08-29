"""
Phase 3 — MTD-HMD (Moving Target Defense)
==========================================

Reimplementation of Kuruvila et al., "Defending Hardware-based Malware
Detectors against Adversarial Attacks" (2020), adapted to the canonical
9-class dataset.

Core defense mechanism
----------------------
Where RHMD randomizes WHICH CLASSIFIER answers each query, MTD-HMD
randomizes WHICH FEATURES the classifier looks at. The defender maintains
a pool of base classifiers, each trained on a different feature subset.
At runtime, a switching policy periodically rotates which subset is active.
Adversarial attacks crafted against one subset go stale after rotation.

Defense rationale (not tested here): unpredictable feature visibility makes
trace-perturbation attacks unreliable, because the attacker doesn't know
which counters the detector is currently inspecting.

For our 3-feature setup
-----------------------
Available features: IPC, cache_miss, branch_miss.
All 7 non-empty subsets (3 singletons + 3 pairs + 1 triple) used by default.
This is the maximum subset diversity available. Kuruvila's original paper
uses 8+ counters with 3-4-counter subsets; we have less subset space.
Document this constraint in the methodology section — feature engineering
(Stage E or supplementary) would expand it.

What Stage D measures (and what it does NOT)
--------------------------------------------
DOES: per-attack TPR profile under clean inputs; per-window operating point
for the Phase 4 composition grid; per-subset standalone performance
(showing which subsets are weak individually).

DOES NOT: adversarial-robustness evaluation. That's Kuruvila's canonical
claim; in our paper MTD-HMD is a baseline that will be defeated by
behavioral mimicry — for a structural reason: the mimicry input is
benign-shaped across ALL counters, so every subset agrees it looks benign.

Design choices baked in
-----------------------
- Single base classifier (RF, default). Diversity comes from features.
- Linear window indexing for the switching schedule across the full
  concatenated test stream (mimics continuous-monitoring deployment).
- Round-robin policy. Random-uniform is parameterizable in MTDHMDConfig
  for future work but not used in the main sweep.
- Switching RNG separate from training RNG.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score

from dataset import (
    Trace, Dataset, ATTACK_CLASSES,
    stratified_kfold_traces,
)
from hmd_vanilla import (
    VanillaHMDConfig, VanillaHMDFit, fit_vanilla_hmd,
    windows_and_labels, aggregate_per_trace,
    binary_metrics, per_attack_class_tpr,
    FoldResult, summarize_cv,
    cross_validate_vanilla_hmd, print_summary,
)


# ===========================================================================
# MTD-HMD configuration
# ===========================================================================

@dataclass
class MTDHMDConfig:
    """MTD-HMD ensemble configuration.

    Attributes
    ----------
    feature_subsets : list of tuples, optional
        Each tuple is a subset of feature names. If None, defaults to all
        non-empty subsets of the dataset's feature_names (7 subsets for a
        3-feature dataset).
    base_classifier : str
        Shared base classifier type for every subset. RF by default — LogReg
        fails on A5 (see PAPER_NOTES §5.1) so using it here would conflate
        "MTD doesn't help" with "base classifier was already broken."
    switching_interval : int
        Number of consecutive windows during which a single subset stays
        active before the next rotation. 1 ≈ per-window switching;
        larger values ≈ slower rotation.
    switching_policy : str
        "round_robin"     — deterministic cycle through subsets (default)
        "random_uniform"  — random subset per switching block (future work)
    use_scaler : bool
        Per-subset StandardScaler. Passed through to each base fit.
    seed : int
        Master training seed. Per-subset seeds derived as seed + k*1000.
    switching_seed : int
        Independent seed for switching RNG (only used by random_uniform).
    """

    feature_subsets: list[tuple[str, ...]] | None = None
    base_classifier: str = "rf"
    switching_interval: int = 5
    switching_policy: str = "round_robin"
    use_scaler: bool = True
    seed: int = 0
    switching_seed: int = 2000


@dataclass
class MTDHMDFit:
    """Trained MTD-HMD ensemble: one base classifier per feature subset."""

    cfg: MTDHMDConfig
    feature_subsets: list[tuple[str, ...]]
    subset_indices: list[list[int]]   # column indices into the X feature matrix
    fits: list[VanillaHMDFit]
    switching_rng: np.random.Generator

    def _active_subset_per_window(self, n_windows: int) -> np.ndarray:
        """Compute the active-subset index for each window in [0, n_windows).

        Round-robin: window i uses subset (i // interval) % n_subsets.
        Random-uniform: each block of `interval` windows samples a subset
                        uniformly at random; deterministic given switching_rng.
        """
        interval = max(1, int(self.cfg.switching_interval))
        n_subsets = len(self.fits)
        block_idx = np.arange(n_windows) // interval

        if self.cfg.switching_policy == "round_robin":
            return block_idx % n_subsets
        if self.cfg.switching_policy == "random_uniform":
            n_blocks = int(block_idx.max()) + 1 if n_windows > 0 else 0
            block_choices = self.switching_rng.integers(
                0, n_subsets, size=n_blocks,
            )
            return block_choices[block_idx]
        raise ValueError(f"Unknown switching_policy: {self.cfg.switching_policy}")

    def predict_proba_windows(self, X: np.ndarray) -> np.ndarray:
        """Per-window prediction: each window routed to its currently-active subset."""
        n = X.shape[0]
        if n == 0:
            return np.zeros(0)
        which = self._active_subset_per_window(n)
        probs = np.zeros(n)
        for k, fit in enumerate(self.fits):
            mask = (which == k)
            if mask.any():
                X_sub = X[mask][:, self.subset_indices[k]]
                probs[mask] = fit.predict_proba_windows(X_sub)
        return probs


# ===========================================================================
# Helpers
# ===========================================================================

def _all_non_empty_subsets(feature_names: list[str]) -> list[tuple[str, ...]]:
    """Return all 2^n - 1 non-empty subsets of `feature_names` as tuples.

    For 3 features: 3 singletons + 3 pairs + 1 triple = 7 subsets.
    """
    out: list[tuple[str, ...]] = []
    for r in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, r):
            out.append(combo)
    return out


def _subset_traces(
    traces: list[Trace],
    indices: list[int],
    subset_feature_names: tuple[str, ...],
) -> list[Trace]:
    """Build new Trace objects whose `windows` are column-restricted to `indices`."""
    return [
        Trace(
            class_name=t.class_name,
            label=t.label,
            trace_id=t.trace_id,
            windows=t.windows[:, indices],
            feature_names=list(subset_feature_names),
        )
        for t in traces
    ]


# ===========================================================================
# Training
# ===========================================================================

def fit_mtd_hmd(train_traces: list[Trace], cfg: MTDHMDConfig | None = None) -> MTDHMDFit:
    cfg = cfg or MTDHMDConfig()
    base_feature_names = list(train_traces[0].feature_names)

    feature_subsets = (
        cfg.feature_subsets
        if cfg.feature_subsets is not None
        else _all_non_empty_subsets(base_feature_names)
    )

    # Validate
    for subset in feature_subsets:
        for f in subset:
            if f not in base_feature_names:
                raise KeyError(
                    f"Subset {subset!r} references unknown feature {f!r}. "
                    f"Available: {base_feature_names}"
                )

    subset_indices: list[list[int]] = []
    fits: list[VanillaHMDFit] = []
    for k, subset in enumerate(feature_subsets):
        indices = [base_feature_names.index(f) for f in subset]
        subset_indices.append(indices)

        sub_traces = _subset_traces(train_traces, indices, subset)
        sub_cfg = VanillaHMDConfig(
            classifier=cfg.base_classifier,
            seed=cfg.seed + k * 1000,
            use_scaler=cfg.use_scaler,
        )
        fits.append(fit_vanilla_hmd(sub_traces, cfg=sub_cfg))

    return MTDHMDFit(
        cfg=cfg,
        feature_subsets=list(feature_subsets),
        subset_indices=subset_indices,
        fits=fits,
        switching_rng=np.random.default_rng(cfg.switching_seed),
    )


# ===========================================================================
# Cross-validation
# ===========================================================================

def cross_validate_mtd_hmd(
    dataset: Dataset,
    cfg: MTDHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    trace_threshold: float = 0.5,
    verbose: bool = True,
) -> list[FoldResult]:
    """5-fold stratified CV at the trace level; mirrors cross_validate_rhmd."""
    cfg = cfg or MTDHMDConfig()
    results: list[FoldResult] = []

    for fold_idx, (train_traces, test_traces) in enumerate(
        stratified_kfold_traces(dataset, n_folds=n_folds, seed=cv_seed)
    ):
        fold_cfg = dataclasses.replace(
            cfg, switching_seed=cfg.switching_seed + fold_idx * 10_000,
        )
        fit = fit_mtd_hmd(train_traces, cfg=fold_cfg)

        X_test, y_test_win, tids = windows_and_labels(test_traces, binary=True)
        probs_win = fit.predict_proba_windows(X_test)
        preds_win = (probs_win >= 0.5).astype(int)
        win_m = binary_metrics(y_test_win, preds_win)

        attack_frac, y_test_trace = aggregate_per_trace(
            test_traces, probs_win, tids,
        )
        preds_trace = (attack_frac >= trace_threshold).astype(int)
        trace_m = binary_metrics(y_test_trace, preds_trace)

        try:
            trace_auc = float(roc_auc_score(y_test_trace, attack_frac))
        except ValueError:
            trace_auc = float("nan")

        per_atk_tpr = per_attack_class_tpr(
            test_traces, attack_frac, threshold=trace_threshold,
        )

        results.append(FoldResult(
            fold_idx=fold_idx,
            window_metrics=win_m,
            trace_metrics=trace_m,
            trace_auc=trace_auc,
            per_attack_tpr=per_atk_tpr,
            n_train_windows=int(len(train_traces) * train_traces[0].n_windows),
            n_test_windows=int(len(test_traces) * test_traces[0].n_windows),
        ))

        if verbose:
            print(f"  fold {fold_idx}: win TPR={win_m['tpr']:.3f} "
                  f"FPR={win_m['fpr']:.3f}  "
                  f"trace TPR={trace_m['tpr']:.3f} FPR={trace_m['fpr']:.3f}  "
                  f"trace AUC={trace_auc:.4f}")
    return results


# ===========================================================================
# Per-subset diagnostic (standalone, no rotation)
# ===========================================================================

def evaluate_per_subset(
    dataset: Dataset,
    cfg: MTDHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Evaluate each subset classifier standalone, no MTD rotation.

    Useful diagnostic: shows which subsets are weak individually (typically
    singletons) — provides context for whether MTD's rotation gains anything
    over the best subset alone.

    Returns
    -------
    dict mapping subset -> summary dict (same shape as summarize_cv output)
    """
    cfg = cfg or MTDHMDConfig()
    base_feature_names = list(dataset.traces[0].feature_names)
    feature_subsets = (
        cfg.feature_subsets
        if cfg.feature_subsets is not None
        else _all_non_empty_subsets(base_feature_names)
    )

    out: dict[tuple[str, ...], dict] = {}
    for subset in feature_subsets:
        indices = [base_feature_names.index(f) for f in subset]
        sub_traces = _subset_traces(dataset.traces, indices, subset)
        sub_dataset = Dataset(traces=sub_traces, feature_names=list(subset))

        if verbose:
            print(f"\n----- standalone: subset {subset} -----")
        v_cfg = VanillaHMDConfig(
            classifier=cfg.base_classifier,
            seed=cfg.seed,
            use_scaler=cfg.use_scaler,
        )
        results = cross_validate_vanilla_hmd(
            sub_dataset, cfg=v_cfg, n_folds=n_folds, cv_seed=cv_seed,
            verbose=verbose,
        )
        out[subset] = summarize_cv(results)
    return out


# ===========================================================================
# Switching-interval sweep
# ===========================================================================

def sweep_switching_interval(
    dataset: Dataset,
    intervals: list[int] = (1, 5, 10, 50),
    base_cfg: MTDHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run cross_validate_mtd_hmd for each switching_interval value.

    Returns
    -------
    dict mapping interval -> {'fold_results': list[FoldResult], 'summary': dict}
    """
    base_cfg = base_cfg or MTDHMDConfig()
    out: dict = {}
    for interval in intervals:
        if verbose:
            print(f"\n----- MTD switching_interval = {interval} windows -----")
        cfg_i = dataclasses.replace(base_cfg, switching_interval=interval)
        results = cross_validate_mtd_hmd(
            dataset, cfg=cfg_i, n_folds=n_folds, cv_seed=cv_seed,
            verbose=verbose,
        )
        summary = summarize_cv(results)
        out[interval] = {"fold_results": results, "summary": summary}
    return out


# ===========================================================================
# Reporting
# ===========================================================================

def print_per_subset_table(per_subset_results: dict):
    print("\n" + "=" * 84)
    print("PER-SUBSET STANDALONE CLASSIFIER PERFORMANCE (no rotation)")
    print("=" * 84)
    print(f"{'Subset':<32s} {'tr TPR':>10s} {'tr FPR':>10s} "
          f"{'tr AUC':>10s} {'A5 TPR':>10s}")
    print("-" * 84)
    for subset, summary in per_subset_results.items():
        tm, _ = summary["trace_tpr"]
        tf, _ = summary["trace_fpr"]
        ta, _ = summary["trace_auc"]
        a5, _ = summary["per_attack_tpr"].get("A5_encrypt_loop", (float("nan"), 0))
        subset_str = "{" + ", ".join(subset) + "}"
        print(f"{subset_str:<32s} {tm:>10.4f} {tf:>10.4f} {ta:>10.4f} {a5:>10.4f}")


def print_sweep_table(sweep_results: dict, vanilla_summary: dict | None = None):
    """Comparison table: each switching interval vs (optionally) vanilla baseline."""
    print("\n" + "=" * 84)
    print(f"{'Method':<24s} {'win TPR':>10s} {'win FPR':>10s} "
          f"{'tr TPR':>10s} {'tr FPR':>10s} {'tr AUC':>10s}")
    print("-" * 84)

    if vanilla_summary is not None:
        wm, _ = vanilla_summary["window_tpr"]
        wf, _ = vanilla_summary["window_fpr"]
        tm, _ = vanilla_summary["trace_tpr"]
        tf, _ = vanilla_summary["trace_fpr"]
        ta, _ = vanilla_summary["trace_auc"]
        print(f"{'Vanilla RF (reference)':<24s} "
              f"{wm:>10.4f} {wf:>10.4f} {tm:>10.4f} {tf:>10.4f} {ta:>10.4f}")

    for interval in sorted(sweep_results.keys()):
        s = sweep_results[interval]["summary"]
        wm, _ = s["window_tpr"]
        wf, _ = s["window_fpr"]
        tm, _ = s["trace_tpr"]
        tf, _ = s["trace_fpr"]
        ta, _ = s["trace_auc"]
        print(f"{'MTD switch=' + str(interval):<24s} "
              f"{wm:>10.4f} {wf:>10.4f} {tm:>10.4f} {tf:>10.4f} {ta:>10.4f}")

    # Per-attack TPR breakdown
    print("\n" + "=" * 84)
    print("PER-ATTACK-CLASS TPR (mean across folds)")
    print("=" * 84)
    header = f"{'Method':<24s}"
    for cls in ATTACK_CLASSES:
        header += f"  {cls[:14]:>14s}"
    print(header)
    print("-" * (24 + 16 * len(ATTACK_CLASSES)))

    if vanilla_summary is not None:
        row = f"{'Vanilla RF (reference)':<24s}"
        for cls in ATTACK_CLASSES:
            m, _ = vanilla_summary["per_attack_tpr"].get(cls, (float("nan"), 0))
            row += f"  {m:>14.4f}"
        print(row)

    for interval in sorted(sweep_results.keys()):
        s = sweep_results[interval]["summary"]
        row = f"{'MTD switch=' + str(interval):<24s}"
        for cls in ATTACK_CLASSES:
            m, _ = s["per_attack_tpr"].get(cls, (float("nan"), 0))
            row += f"  {m:>14.4f}"
        print(row)


def plot_sweep(
    sweep_results: dict,
    vanilla_rf_summary: dict | None = None,
    output_path: str = "phase3_mtd_sweep.png",
    title_suffix: str = "real Pi traces (100ms aggregated windows)",
) -> None:
    """Two-panel: trace AUC vs switching_interval (left), per-attack TPR (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intervals = sorted(sweep_results.keys())
    aucs = [sweep_results[i]["summary"]["trace_auc"][0] for i in intervals]
    auc_stds = [sweep_results[i]["summary"]["trace_auc"][1] for i in intervals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ----- Panel 1: trace AUC vs switching_interval -----
    ax1.errorbar(intervals, aucs, yerr=auc_stds,
                 marker="s", markersize=10, linewidth=2, capsize=4,
                 color="tab:purple", label="MTD-HMD")
    if vanilla_rf_summary is not None:
        v_auc, v_std = vanilla_rf_summary["trace_auc"]
        ax1.axhline(v_auc, color="gray", linestyle="--", linewidth=1.5,
                    label=f"Vanilla RF baseline ({v_auc:.4f})")
        ax1.fill_between([min(intervals) * 0.5, max(intervals) * 2],
                         v_auc - v_std, v_auc + v_std,
                         color="gray", alpha=0.15)
    ax1.set_xscale("log")
    ax1.set_xticks(intervals)
    ax1.set_xticklabels([str(i) for i in intervals])
    ax1.set_xlim(min(intervals) * 0.6, max(intervals) * 1.6)
    ax1.set_xlabel("Switching interval (windows; log scale)")
    ax1.set_ylabel("Trace-level AUC (mean ± std across 5 folds)")
    ax1.set_title("Trace AUC vs switching interval")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="lower right")

    # ----- Panel 2: per-attack TPR vs interval -----
    n_attacks = len(ATTACK_CLASSES)
    width = 0.8 / len(intervals)
    x_pos = np.arange(n_attacks)
    for i, interval in enumerate(intervals):
        per_atk = sweep_results[interval]["summary"]["per_attack_tpr"]
        tprs = [per_atk.get(c, (float("nan"), 0))[0] for c in ATTACK_CLASSES]
        ax2.bar(x_pos + i * width - 0.4 + width / 2, tprs,
                width=width, label=f"interval={interval}",
                edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace("_", "\n", 1) for c in ATTACK_CLASSES],
                         rotation=0, fontsize=9)
    ax2.set_ylabel("Per-attack-class trace TPR")
    ax2.set_title("Per-attack detection by switching interval")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="lower left", title="Switching interval")

    fig.suptitle(
        f"MTD-HMD switching-interval sweep — 5-fold CV on {title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")


# ===========================================================================
# Driver
# ===========================================================================

if __name__ == "__main__":
    from dataset_parquet import load_real_parquet

    # Real Pi data, 100ms aggregated windows (Stage C established this).
    ds = load_real_parquet(
        "./data/traces",
        window_aggregation_factor=100,
        verbose=True,
    )
    print()
    print(ds.summary())

    # Vanilla RF baseline (best single-classifier reference; RF chosen because
    # LogReg fails on A5 — see PAPER_NOTES §5.1).
    print("\n" + "=" * 78)
    print("Vanilla RF baseline — best single-classifier reference")
    print("=" * 78)
    v_cfg = VanillaHMDConfig(classifier="rf", seed=0)
    v_results = cross_validate_vanilla_hmd(ds, cfg=v_cfg, verbose=True)
    v_summary = summarize_cv(v_results)
    print_summary(v_summary, title="Vanilla RF (reference)")

    # Per-subset standalone diagnostic — shows which subsets are weak.
    print("\n" + "=" * 78)
    print("Per-subset standalone classifiers (no rotation, diagnostic)")
    print("=" * 78)
    base_cfg = MTDHMDConfig(
        base_classifier="rf",
        switching_policy="round_robin",
        seed=0,
        switching_seed=2000,
    )
    per_subset = evaluate_per_subset(ds, cfg=base_cfg, verbose=False)
    print_per_subset_table(per_subset)

    # MTD switching-interval sweep
    print("\n" + "=" * 78)
    print("MTD-HMD switching-interval sweep — interval ∈ {1, 5, 10, 50}")
    print("=" * 78)
    sweep = sweep_switching_interval(
        ds,
        intervals=[1, 5, 10, 50],
        base_cfg=base_cfg,
        verbose=True,
    )

    print_sweep_table(sweep, vanilla_summary=v_summary)
    plot_sweep(sweep, vanilla_rf_summary=v_summary,
               output_path="phase3_mtd_sweep.png")
