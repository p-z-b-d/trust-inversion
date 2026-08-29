"""
Phase 3 — RHMD (Random HMD ensemble)
======================================

Reimplementation of Khasawneh et al., "RHMD: Evasion-Resilient Hardware
Malware Detectors", MICRO 2017, adapted to the canonical 9-class dataset.

Core defense mechanism
----------------------
Train N diverse base classifiers. At each per-window inference, randomly
select one of the N to make the prediction. An adversary who does not know
which classifier will answer cannot reliably craft adversarial inputs that
fool the ensemble.

Diversity mechanisms in this implementation
-------------------------------------------
With only 3 derived features (IPC, cache_miss_rate, branch_miss_rate) the
canonical RHMD "feature-subset diversity" mechanism is constrained. We
provide diversity through:

  1. **Heterogeneous base architectures** — cycle through {LogReg, RF, SVM,
     MLP}. Different inductive biases produce decision surfaces that disagree
     differently for adversarial inputs.
  2. **Bagging** — bootstrap-sample the training traces per ensemble member.
     Provides training-distribution diversity even with identical architecture.
  3. **Per-classifier random seed** — affects RF/MLP/SVC randomization;
     no-op for LogReg with deterministic solver but cheap to set.

Feature-subset diversity is deferred to Stage E (would require window-level
feature engineering — per-window stats, deltas, interactions — to expand
the 3-feature space to ~10–12).

What Stage C does NOT cover
---------------------------
- Adversarial-robustness evaluation. This is the canonical RHMD claim, but
  in OUR paper RHMD is a baseline that will be defeated by behavioral
  mimicry for a *structural* reason (ensemble diversity doesn't help when
  every member agrees the input looks benign). Trace-perturbation evasion
  is Bahador 2020 territory — distinct threat model, prior work.
- Feature-subset diversity. Deferred.
- Validation-weighted selection. Stub provided; default is uniform.

CV protocol
-----------
5-fold stratified CV at the TRACE level (no per-trace leakage). Same harness
as `cross_validate_vanilla_hmd`. Per-fold inference-RNG offset for
reproducibility across folds.

Output is a list[FoldResult] drop-in compatible with `summarize_cv` and
`print_summary` from hmd_vanilla.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from dataset import (
    Dataset, ATTACK_CLASSES,
    stratified_kfold_traces,
)
from hmd_vanilla import (
    VanillaHMDConfig, VanillaHMDFit, fit_vanilla_hmd,
    windows_and_labels, aggregate_per_trace,
    binary_metrics, per_attack_class_tpr,
    FoldResult, summarize_cv,
)


# ===========================================================================
# RHMD configuration
# ===========================================================================

@dataclass
class RHMDConfig:
    """RHMD ensemble configuration.

    Attributes
    ----------
    n_classifiers : int
        Ensemble size. Sweep over {3, 5, 7} in Stage C.
    base_classifiers : list[str]
        Cycled-through list of base classifier types. For n_classifiers > len,
        the list cycles (e.g. n=7 with 4 types → logreg, rf, svm, mlp, logreg,
        rf, svm).
    bootstrap_train : bool
        If True, each ensemble member is trained on a bootstrap sample of the
        training traces (with replacement). Bagging mechanism.
    selection_policy : str
        "uniform"            — uniform random selection (Khasawneh canonical)
        "validation_weighted"— weighted by held-out validation accuracy (stub)
    use_scaler : bool
        Passed through to each base classifier.
    seed : int
        Master training seed. Per-member seeds derived as seed + k*1000.
    inference_seed : int
        Separate seed controlling per-window classifier selection at inference.
        Kept independent of training seed so we can re-roll inference draws
        without retraining.
    """

    n_classifiers: int = 5
    base_classifiers: list[str] = field(
        default_factory=lambda: ["logreg", "rf", "svm", "mlp"]
    )
    bootstrap_train: bool = True
    selection_policy: str = "uniform"
    use_scaler: bool = True
    seed: int = 0
    inference_seed: int = 1000

    def base_classifier_for_member(self, k: int) -> str:
        """Cycle through the base_classifiers list."""
        return self.base_classifiers[k % len(self.base_classifiers)]


@dataclass
class RHMDFit:
    """Trained RHMD ensemble: holds N fitted base classifiers + selection policy."""

    cfg: RHMDConfig
    fits: list[VanillaHMDFit]
    selection_weights: np.ndarray   # length-n_classifiers probability vector
    inference_rng: np.random.Generator

    def predict_proba_windows(self, X: np.ndarray) -> np.ndarray:
        """Per-window random selection: each row independently routed to one
        base classifier, sampled from selection_weights."""
        n = X.shape[0]
        if n == 0:
            return np.zeros(0)
        chosen = self.inference_rng.choice(
            len(self.fits), size=n, p=self.selection_weights,
        )
        probs = np.zeros(n)
        # Vectorize per chosen-classifier (no Python-level per-row loop)
        for k, fit in enumerate(self.fits):
            mask = (chosen == k)
            if mask.any():
                probs[mask] = fit.predict_proba_windows(X[mask])
        return probs

    def reseed_inference(self, seed: int) -> None:
        """Re-seed the inference RNG (e.g., for multi-draw CIs without retraining)."""
        self.inference_rng = np.random.default_rng(seed)


# ===========================================================================
# Training
# ===========================================================================

def fit_rhmd(train_traces, cfg: RHMDConfig | None = None) -> RHMDFit:
    cfg = cfg or RHMDConfig()
    train_rng = np.random.default_rng(cfg.seed)
    fits: list[VanillaHMDFit] = []

    for k in range(cfg.n_classifiers):
        sub_cfg = VanillaHMDConfig(
            classifier=cfg.base_classifier_for_member(k),
            seed=cfg.seed + k * 1000,
            use_scaler=cfg.use_scaler,
        )
        if cfg.bootstrap_train:
            # Bootstrap sample at the TRACE level (with replacement).
            # Window-level bootstrap would be wrong: it would break the
            # per-trace-leakage discipline if applied across the CV boundary.
            idx = train_rng.integers(0, len(train_traces), size=len(train_traces))
            sub_traces = [train_traces[i] for i in idx]
        else:
            sub_traces = train_traces
        fits.append(fit_vanilla_hmd(sub_traces, cfg=sub_cfg))

    # Selection weights
    if cfg.selection_policy == "uniform":
        weights = np.ones(cfg.n_classifiers) / cfg.n_classifiers
    elif cfg.selection_policy == "validation_weighted":
        raise NotImplementedError(
            "validation_weighted selection requires a held-out validation "
            "pass; implement in Stage E or supplementary if needed."
        )
    else:
        raise ValueError(f"Unknown selection_policy: {cfg.selection_policy}")

    return RHMDFit(
        cfg=cfg, fits=fits,
        selection_weights=weights,
        inference_rng=np.random.default_rng(cfg.inference_seed),
    )


# ===========================================================================
# Cross-validation driver
# ===========================================================================

def cross_validate_rhmd(
    dataset: Dataset,
    cfg: RHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    trace_threshold: float = 0.5,
    verbose: bool = True,
) -> list[FoldResult]:
    """5-fold stratified CV at the trace level, mirrors cross_validate_vanilla_hmd."""
    cfg = cfg or RHMDConfig()
    results: list[FoldResult] = []

    for fold_idx, (train_traces, test_traces) in enumerate(
        stratified_kfold_traces(dataset, n_folds=n_folds, seed=cv_seed)
    ):
        # Offset inference seed per fold for deterministic-but-independent draws
        fold_cfg = dataclasses.replace(
            cfg, inference_seed=cfg.inference_seed + fold_idx * 10_000,
        )
        fit = fit_rhmd(train_traces, cfg=fold_cfg)

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
# Ensemble-size sweep
# ===========================================================================

def sweep_ensemble_size(
    dataset: Dataset,
    n_values: list[int] = (3, 5, 7),
    base_cfg: RHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run cross_validate_rhmd for each n_classifiers value.

    Returns
    -------
    dict mapping n -> {'fold_results': list[FoldResult], 'summary': dict}
    """
    base_cfg = base_cfg or RHMDConfig()
    out: dict = {}
    for n in n_values:
        if verbose:
            print(f"\n----- RHMD ensemble size n = {n} -----")
        cfg_n = dataclasses.replace(base_cfg, n_classifiers=n)
        results = cross_validate_rhmd(
            dataset, cfg=cfg_n, n_folds=n_folds, cv_seed=cv_seed,
            verbose=verbose,
        )
        summary = summarize_cv(results)
        out[n] = {"fold_results": results, "summary": summary}
    return out


# ===========================================================================
# Reporting
# ===========================================================================

def print_sweep_table(sweep_results: dict, vanilla_summary: dict | None = None):
    """Comparison table: each ensemble size vs (optionally) the vanilla baseline."""
    print("\n" + "=" * 84)
    print(f"{'Method':<22s} {'win TPR':>10s} {'win FPR':>10s} "
          f"{'tr TPR':>10s} {'tr FPR':>10s} {'tr AUC':>10s}")
    print("-" * 84)

    if vanilla_summary is not None:
        wm, _ = vanilla_summary["window_tpr"]
        wf, _ = vanilla_summary["window_fpr"]
        tm, _ = vanilla_summary["trace_tpr"]
        tf, _ = vanilla_summary["trace_fpr"]
        ta, _ = vanilla_summary["trace_auc"]
        print(f"{'Vanilla (reference)':<22s} "
              f"{wm:>10.4f} {wf:>10.4f} {tm:>10.4f} {tf:>10.4f} {ta:>10.4f}")

    for n in sorted(sweep_results.keys()):
        s = sweep_results[n]["summary"]
        wm, _ = s["window_tpr"]
        wf, _ = s["window_fpr"]
        tm, _ = s["trace_tpr"]
        tf, _ = s["trace_fpr"]
        ta, _ = s["trace_auc"]
        print(f"{'RHMD n=' + str(n):<22s} "
              f"{wm:>10.4f} {wf:>10.4f} {tm:>10.4f} {tf:>10.4f} {ta:>10.4f}")

    # Per-attack TPR breakdown
    print("\n" + "=" * 84)
    print("PER-ATTACK-CLASS TPR (mean across folds)")
    print("=" * 84)
    header = f"{'Method':<22s}"
    for cls in ATTACK_CLASSES:
        header += f"  {cls[:14]:>14s}"
    print(header)
    print("-" * (22 + 16 * len(ATTACK_CLASSES)))

    if vanilla_summary is not None:
        row = f"{'Vanilla (reference)':<22s}"
        for cls in ATTACK_CLASSES:
            m, _ = vanilla_summary["per_attack_tpr"].get(cls, (float("nan"), 0))
            row += f"  {m:>14.4f}"
        print(row)

    for n in sorted(sweep_results.keys()):
        s = sweep_results[n]["summary"]
        row = f"{'RHMD n=' + str(n):<22s}"
        for cls in ATTACK_CLASSES:
            m, _ = s["per_attack_tpr"].get(cls, (float("nan"), 0))
            row += f"  {m:>14.4f}"
        print(row)


def plot_sweep(
    sweep_results: dict,
    vanilla_summary: dict | None = None,
    output_path: str = "phase3_rhmd_sweep.png",
    title_suffix: str = "synthetic-stats-matched data (re-run on real Pi traces before closeout)",
) -> None:
    """Two-panel: trace AUC vs n (left), per-attack TPR vs n (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(sweep_results.keys())
    aucs = [sweep_results[n]["summary"]["trace_auc"][0] for n in ns]
    auc_stds = [sweep_results[n]["summary"]["trace_auc"][1] for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ----- Panel 1: trace AUC vs ensemble size -----
    ax1.errorbar(ns, aucs, yerr=auc_stds,
                 marker="o", markersize=10, linewidth=2, capsize=4,
                 color="tab:blue", label="RHMD")
    if vanilla_summary is not None:
        v_auc, v_std = vanilla_summary["trace_auc"]
        ax1.axhline(v_auc, color="gray", linestyle="--", linewidth=1.5,
                    label=f"Vanilla baseline ({v_auc:.4f})")
        ax1.fill_between([min(ns) - 0.5, max(ns) + 0.5],
                         v_auc - v_std, v_auc + v_std,
                         color="gray", alpha=0.15)
    ax1.set_xlabel("RHMD ensemble size  $n_{classifiers}$")
    ax1.set_ylabel("Trace-level AUC (mean ± std across 5 folds)")
    ax1.set_title("Trace AUC vs ensemble size")
    ax1.set_xticks(ns)
    ax1.set_xlim(min(ns) - 0.5, max(ns) + 0.5)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    # ----- Panel 2: per-attack TPR vs n -----
    n_attacks = len(ATTACK_CLASSES)
    width = 0.8 / len(ns)
    x_pos = np.arange(n_attacks)
    for i, n in enumerate(ns):
        per_atk = sweep_results[n]["summary"]["per_attack_tpr"]
        tprs = [per_atk.get(c, (float("nan"), 0))[0] for c in ATTACK_CLASSES]
        ax2.bar(x_pos + i * width - 0.4 + width / 2, tprs,
                width=width, label=f"n={n}",
                edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace("_", "\n", 1) for c in ATTACK_CLASSES],
                         rotation=0, fontsize=9)
    ax2.set_ylabel("Per-attack-class trace TPR")
    ax2.set_title("Per-attack detection by ensemble size")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="lower left", title="Ensemble size")

    fig.suptitle(
        f"RHMD ensemble-size sweep — 5-fold CV on {title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")


# ===========================================================================
# Driver: synthetic-data smoke run
# ===========================================================================

if __name__ == "__main__":
    from dataset import generate_synthetic
    from hmd_vanilla import cross_validate_vanilla_hmd, print_summary

    ds = generate_synthetic(seed=42)
    print(ds.summary())

    # Vanilla baseline (LogReg) for reference. Stage B already established
    # that RF is stronger on synthetic data; LogReg is the harder baseline
    # because of the A5-IPC-overlap issue (binary linear failure on A5).
    print("\n" + "=" * 78)
    print("Vanilla HMD baseline (LogReg) — Stage B reference")
    print("=" * 78)
    v_cfg = VanillaHMDConfig(classifier="logreg", seed=0)
    v_results = cross_validate_vanilla_hmd(ds, cfg=v_cfg, verbose=True)
    v_summary = summarize_cv(v_results)
    print_summary(v_summary, title="Vanilla HMD (LogReg, reference)")

    # RHMD ensemble-size sweep over {3, 5, 7}
    print("\n" + "=" * 78)
    print("RHMD ensemble-size sweep — n ∈ {3, 5, 7}")
    print("=" * 78)
    base_rhmd_cfg = RHMDConfig(
        base_classifiers=["logreg", "rf", "svm", "mlp"],
        bootstrap_train=True,
        selection_policy="uniform",
        seed=0,
        inference_seed=1000,
    )
    sweep = sweep_ensemble_size(
        ds,
        n_values=[3, 5, 7],
        base_cfg=base_rhmd_cfg,
        verbose=True,
    )

    print_sweep_table(sweep, vanilla_summary=v_summary)
    plot_sweep(sweep, vanilla_summary=v_summary,
               output_path="phase3_rhmd_sweep.png")
