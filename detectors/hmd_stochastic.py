"""
Phase 3 — D5: Stochastic-HMD with Inference-Time Noise Injection
================================================================

Reimplementation of Pundir et al. 2021 "Stochastic-HMDs" defense, in
software-equivalent form. Wraps a base classifier (vanilla RF) with
Gaussian noise injection at inference time.

Defense mechanism
-----------------
For each test window:
  1. Pre-scale features using the base classifier's StandardScaler (once).
  2. Sample K independent Gaussian noise vectors ~ N(0, sigma^2 * I) in
     standardized feature space.
  3. Add each noise vector to the scaled features and call the base model's
     predict_proba K times.
  4. Average the K probability outputs.

Rationale (Pundir 2021): by perturbing the input with random noise the
defender doesn't commit to, adversarial perturbations crafted for a specific
input become less effective. The attacker doesn't know which noisy version
the classifier will see at inference time.

For our composition story
-------------------------
Stochastic-HMD's defense operates at the per-classifier input layer. A5
(composition-aware mimicry) operates at the aggregation layer. We expect D5
to perform like a slightly-noisier vanilla RF on clean data, and to fail
similarly under A5 — adversarial robustness at one layer doesn't carry to
another. Phase 1's per-edge noise null result is consistent: aggregation
crushes per-edge noise (AUC stayed at 1.0 for sigma up to 1.5), so D5's
mechanism is orthogonal to the composition vulnerability.

Parameters
----------
- base_classifier: "rf" (vanilla RF — LogReg is broken on A5 so wrapping it
  conflates "D5 doesn't help" with "base classifier was already broken")
- noise_sigma: sweep over {0.05, 0.1, 0.2, 0.5} in standardized units
- n_noise_samples (K): 10, fixed
- Same dataset as D1/D3/D4 (3 derived features, 238 traces / 8 classes via
  the standard load_real_parquet loader)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from dataset import ATTACK_CLASSES, stratified_kfold_traces
from hmd_vanilla import (
    VanillaHMDConfig, VanillaHMDFit, fit_vanilla_hmd,
    cross_validate_vanilla_hmd,
    windows_and_labels, aggregate_per_trace,
    binary_metrics, per_attack_class_tpr,
    FoldResult, summarize_cv, print_summary,
)


# ===========================================================================
# Config and Fit dataclasses
# ===========================================================================

@dataclass
class StochasticHMDConfig:
    """Stochastic-HMD wrapper configuration."""
    base_classifier: str = "rf"
    noise_sigma: float = 0.1     # standard deviation of N(0, sigma^2 * I) noise
    n_noise_samples: int = 10    # K — averages per prediction
    use_scaler: bool = True      # base classifier's StandardScaler
    seed: int = 0
    noise_seed: int = 3000       # independent of training seed


@dataclass
class StochasticHMDFit:
    """Trained Stochastic-HMD: base VanillaHMDFit + a noise RNG."""
    cfg: StochasticHMDConfig
    base_fit: VanillaHMDFit
    noise_rng: np.random.Generator

    def predict_proba_windows(self, X: np.ndarray) -> np.ndarray:
        """Per-window attack probability under K-sample noise averaging.

        Pre-scales X once with the base StandardScaler, then samples K
        Gaussian noise vectors in standardized space, runs predict_proba
        K times, averages.
        """
        n = X.shape[0]
        if n == 0:
            return np.zeros(0, dtype=float)

        # Apply the base scaler once (cheaper than scaling inside the loop)
        if self.base_fit.scaler is not None:
            X_scaled = self.base_fit.scaler.transform(X)
        else:
            X_scaled = np.asarray(X, dtype=float)

        K = max(1, int(self.cfg.n_noise_samples))
        sigma = float(self.cfg.noise_sigma)

        probs_accum = np.zeros(n, dtype=float)
        for _ in range(K):
            noise = self.noise_rng.normal(
                loc=0.0, scale=sigma, size=X_scaled.shape,
            ).astype(X_scaled.dtype)
            X_noisy = X_scaled + noise
            # base model's predict_proba returns shape (n, 2); column 1 = attack
            probs_accum += self.base_fit.model.predict_proba(X_noisy)[:, 1]
        return probs_accum / K


# ===========================================================================
# Training
# ===========================================================================

def fit_stochastic_hmd(
    train_traces: list,
    cfg: StochasticHMDConfig | None = None,
) -> StochasticHMDFit:
    """Train base RF on train_traces; wrap with noise RNG for inference."""
    cfg = cfg or StochasticHMDConfig()

    base_cfg = VanillaHMDConfig(
        classifier=cfg.base_classifier,
        seed=cfg.seed,
        use_scaler=cfg.use_scaler,
    )
    base_fit = fit_vanilla_hmd(train_traces, cfg=base_cfg)

    return StochasticHMDFit(
        cfg=cfg,
        base_fit=base_fit,
        noise_rng=np.random.default_rng(cfg.noise_seed),
    )


# ===========================================================================
# Cross-validation
# ===========================================================================

def cross_validate_stochastic_hmd(
    dataset,
    cfg: StochasticHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    trace_threshold: float = 0.5,
    verbose: bool = True,
) -> list[FoldResult]:
    """5-fold stratified CV at the trace level. Mirrors cross_validate_mtd_hmd."""
    cfg = cfg or StochasticHMDConfig()
    results: list[FoldResult] = []

    for fold_idx, (train_traces, test_traces) in enumerate(
        stratified_kfold_traces(dataset, n_folds=n_folds, seed=cv_seed)
    ):
        # Per-fold noise seed: keeps each fold's noise reproducible while
        # producing distinct noise sequences across folds.
        fold_cfg = dataclasses.replace(
            cfg, noise_seed=cfg.noise_seed + fold_idx * 10_000,
        )
        fit = fit_stochastic_hmd(train_traces, cfg=fold_cfg)

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
# Noise-sigma sweep
# ===========================================================================

def sweep_noise_sigma(
    dataset,
    sigmas: list[float] = (0.05, 0.1, 0.2, 0.5),
    base_cfg: StochasticHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run cross_validate_stochastic_hmd for each sigma value."""
    base_cfg = base_cfg or StochasticHMDConfig()
    out: dict = {}
    for sigma in sigmas:
        if verbose:
            print(f"\n----- Stochastic-HMD noise sigma = {sigma:.2f} "
                  f"(K = {base_cfg.n_noise_samples}) -----")
        cfg_s = dataclasses.replace(base_cfg, noise_sigma=float(sigma))
        results = cross_validate_stochastic_hmd(
            dataset, cfg=cfg_s, n_folds=n_folds, cv_seed=cv_seed,
            verbose=verbose,
        )
        out[float(sigma)] = {"fold_results": results, "summary": summarize_cv(results)}
    return out


# ===========================================================================
# Reporting
# ===========================================================================

def print_sweep_table(sweep_results: dict, vanilla_summary: dict | None = None):
    """Comparison table across sigma values."""
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

    for sigma in sorted(sweep_results.keys()):
        s = sweep_results[sigma]["summary"]
        wm, _ = s["window_tpr"]
        wf, _ = s["window_fpr"]
        tm, _ = s["trace_tpr"]
        tf, _ = s["trace_fpr"]
        ta, _ = s["trace_auc"]
        print(f"{'D5 sigma=' + f'{sigma:.2f}':<24s} "
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

    for sigma in sorted(sweep_results.keys()):
        s = sweep_results[sigma]["summary"]
        row = f"{'D5 sigma=' + f'{sigma:.2f}':<24s}"
        for cls in ATTACK_CLASSES:
            m, _ = s["per_attack_tpr"].get(cls, (float("nan"), 0))
            row += f"  {m:>14.4f}"
        print(row)


def plot_sweep(
    sweep_results: dict,
    vanilla_rf_summary: dict | None = None,
    output_path: str = "phase3_stochastic_sweep.png",
    title_suffix: str = "real Pi traces (100ms aggregated windows)",
) -> None:
    """Two-panel: trace AUC vs sigma (left), per-attack TPR (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = sorted(sweep_results.keys())
    aucs = [sweep_results[s]["summary"]["trace_auc"][0] for s in sigmas]
    auc_stds = [sweep_results[s]["summary"]["trace_auc"][1] for s in sigmas]

    # Window-level FPR is the more interesting metric for this defense
    win_fprs = [sweep_results[s]["summary"]["window_fpr"][0] for s in sigmas]
    win_fpr_stds = [sweep_results[s]["summary"]["window_fpr"][1] for s in sigmas]
    win_tprs = [sweep_results[s]["summary"]["window_tpr"][0] for s in sigmas]
    win_tpr_stds = [sweep_results[s]["summary"]["window_tpr"][1] for s in sigmas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ----- Panel 1: per-window TPR and FPR vs sigma (twin y-axes) -----
    color_tpr = "tab:green"
    color_fpr = "tab:red"

    ax1.errorbar(sigmas, win_tprs, yerr=win_tpr_stds,
                 marker="o", markersize=9, linewidth=2, capsize=4,
                 color=color_tpr, label="D5 TPR (window)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Noise sigma (standardized feature units, log scale)")
    ax1.set_ylabel("Per-window TPR", color=color_tpr)
    ax1.tick_params(axis="y", labelcolor=color_tpr)
    ax1.set_ylim(0.4, 1.02)

    if vanilla_rf_summary is not None:
        v_tpr, _ = vanilla_rf_summary["window_tpr"]
        ax1.axhline(v_tpr, color=color_tpr, linestyle="--", linewidth=1.2,
                    alpha=0.6, label=f"Vanilla RF TPR ({v_tpr:.4f})")

    ax1b = ax1.twinx()
    ax1b.errorbar(sigmas, win_fprs, yerr=win_fpr_stds,
                  marker="s", markersize=9, linewidth=2, capsize=4,
                  color=color_fpr, label="D5 FPR (window)")
    ax1b.set_ylabel("Per-window FPR", color=color_fpr)
    ax1b.tick_params(axis="y", labelcolor=color_fpr)
    ax1b.set_ylim(0, 1.05)

    if vanilla_rf_summary is not None:
        v_fpr, _ = vanilla_rf_summary["window_fpr"]
        ax1b.axhline(v_fpr, color=color_fpr, linestyle="--", linewidth=1.2,
                     alpha=0.6, label=f"Vanilla RF FPR ({v_fpr:.4f})")

    ax1.set_title("Per-window operating point vs noise sigma")
    ax1.grid(alpha=0.3, which="both")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", fontsize=9, framealpha=0.9)

    # ----- Panel 2: per-attack TPR vs sigma -----
    n_attacks = len(ATTACK_CLASSES)
    width = 0.8 / len(sigmas)
    x_pos = np.arange(n_attacks)
    for i, sigma in enumerate(sigmas):
        per_atk = sweep_results[sigma]["summary"]["per_attack_tpr"]
        tprs = [per_atk.get(c, (float("nan"), 0))[0] for c in ATTACK_CLASSES]
        ax2.bar(x_pos + i * width - 0.4 + width / 2, tprs,
                width=width, label=f"sigma={sigma:.2f}",
                edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace("_", "\n", 1) for c in ATTACK_CLASSES],
                         fontsize=9)
    ax2.set_ylabel("Per-attack-class trace TPR")
    ax2.set_title("Per-attack detection by noise sigma")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="lower left", title="Noise sigma", fontsize=9)

    fig.suptitle(
        f"D5: Stochastic-HMD noise-sigma sweep — 5-fold CV on {title_suffix}",
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

    print("=" * 78)
    print("Phase 3 — D5: Stochastic-HMD (Pundir 2021-class)")
    print("=" * 78)
    print()

    # Standard derived-feature loader (3 features, 238 traces, 8 classes)
    ds = load_real_parquet(
        "./data/traces",
        window_aggregation_factor=100,
        verbose=True,
    )
    print()
    print(ds.summary())

    # ----- Vanilla RF baseline (reference, K=1 sigma=0 limit) -----
    print("\n" + "=" * 78)
    print("Vanilla RF baseline — D5's underlying classifier without noise")
    print("=" * 78)
    v_cfg = VanillaHMDConfig(classifier="rf", seed=0)
    t0 = time.perf_counter()
    v_results = cross_validate_vanilla_hmd(ds, cfg=v_cfg, verbose=True)
    t1 = time.perf_counter()
    v_summary = summarize_cv(v_results)
    print(f"  (vanilla RF CV completed in {t1 - t0:.1f}s)")
    print_summary(v_summary, title="Vanilla RF (reference)")

    # ----- D5 sigma sweep -----
    print("\n" + "=" * 78)
    print("D5 Stochastic-HMD — noise-sigma sweep, sigma ∈ {0.05, 0.1, 0.2, 0.5}")
    print(f"K = 10 noise samples per window")
    print("=" * 78)
    base_cfg = StochasticHMDConfig(
        base_classifier="rf",
        n_noise_samples=10,
        seed=0,
        noise_seed=3000,
    )
    t0 = time.perf_counter()
    sweep = sweep_noise_sigma(
        ds,
        sigmas=[0.05, 0.1, 0.2, 0.5],
        base_cfg=base_cfg,
        verbose=True,
    )
    t1 = time.perf_counter()
    print(f"\n(D5 full sweep CV completed in {t1 - t0:.1f}s)")

    # ----- Comparison table -----
    print_sweep_table(sweep, vanilla_summary=v_summary)

    # ----- Figure -----
    plot_sweep(sweep, vanilla_rf_summary=v_summary,
               output_path="phase3_stochastic_sweep.png")
