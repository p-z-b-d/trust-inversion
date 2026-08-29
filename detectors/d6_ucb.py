"""
D6 Module 3: UCB Defense-Assignment Controller
===============================================

Third component of D6. The UCB (Upper Confidence Bound) controller routes
each input to whichever base classifier has the best track record for the
A2C-predicted adversarial pattern.

Routing logic
-------------
For each test window:
  1. A2C (Module 2) predicts adversarial pattern p_pred.
  2. UCB selects base classifier c* = argmax_c mean_reward[p_pred, c].
  3. Base classifier c* predicts attack probability.

Training
--------
During training (offline on synthetic perturbation data with full ground
truth), UCB uses the standard sampling rule to balance exploration and
exploitation:
  c_t = argmax_c [ mean_reward[p_pred, c] + λ · sqrt(2 ln N_p / n_pulls[p_pred, c]) ]

where N_p is the total pulls under pattern p_pred. Each training window
generates one classifier selection, one reward observation, and one stats
update. With ~113K synthetic examples per fold and 18 (pattern, classifier)
cells, every cell gets thousands of pulls — UCB exploration is essentially
academic at this scale, but we implement it for paper-faithfulness with
He et al. 2024.

Inference
---------
Greedy — exploration term dropped. Routing table = argmax of mean_reward
along the classifier axis.

End-to-end D6 pipeline
----------------------
This module exposes the final D6 inference path:

  raw_window → A2C(state(b)) → predicted_pattern
             → UCB[predicted_pattern] → selected_classifier_idx
             → base_classifiers[idx].predict_proba(raw_window) → attack_prob
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import time

import numpy as np
import torch

from dataset import ATTACK_CLASSES
from hmd_vanilla import windows_and_labels
from d6_adversarial import (
    AdversarialMLPConfig, AdversarialMLPFit, fit_adversarial_mlp,
)
from d6_a2c import (
    PATTERN_NAMES, N_PATTERNS,
    A2CConfig, A2CFit, train_a2c,
    generate_synthetic_patterns, compute_state,
)


# ===========================================================================
# Config and Fit dataclasses
# ===========================================================================

@dataclass
class UCBConfig:
    n_patterns: int = N_PATTERNS
    n_classifiers: int = 3
    exploration_c: float = 1.4142    # sqrt(2) — standard UCB1
    seed: int = 0
    smoothing_prior: float = 0.5      # uninformative prior for never-pulled cells


@dataclass
class UCBFit:
    cfg: UCBConfig
    n_pulls: np.ndarray        # (n_patterns, n_classifiers) — total pulls
    reward_sums: np.ndarray    # (n_patterns, n_classifiers) — sum of rewards
    mean_rewards: np.ndarray   # (n_patterns, n_classifiers) — reward_sums / max(n_pulls, 1)
    training_steps: int

    @property
    def routing_table(self) -> np.ndarray:
        """For each pattern, the index of the best base classifier (argmax mean reward).

        Returns array of shape (n_patterns,) with values in {0, ..., n_classifiers - 1}.
        """
        return self.mean_rewards.argmax(axis=1)

    def predict_proba_windows(
        self,
        X_raw: np.ndarray,
        a2c_fit: A2CFit,
        base_classifiers: list[AdversarialMLPFit],
    ) -> np.ndarray:
        """End-to-end D6 inference: A2C → UCB → base classifier.

        Returns per-window attack probabilities, shape (n_windows,).
        """
        if X_raw.shape[0] == 0:
            return np.zeros(0, dtype=float)

        X_std = (X_raw - a2c_fit.feature_means) / a2c_fit.feature_stds

        # A2C: predict pattern for each window
        states = compute_state(X_std, base_classifiers)
        state_t = torch.from_numpy(states.astype(np.float32))
        a2c_fit.policy.eval()
        with torch.no_grad():
            logits = a2c_fit.policy(state_t)
            predicted_patterns = logits.argmax(dim=-1).numpy()

        # UCB: lookup routing for each predicted pattern
        routing = self.routing_table
        selected_c = routing[predicted_patterns]    # (n_windows,)

        # Get every base classifier's probability on the standardized input
        X_std_t = torch.from_numpy(X_std.astype(np.float32))
        all_probs = np.zeros((len(base_classifiers), X_std.shape[0]), dtype=float)
        for c_idx, clf in enumerate(base_classifiers):
            clf.model.eval()
            with torch.no_grad():
                logits_c = clf.model(X_std_t)
                all_probs[c_idx] = torch.sigmoid(logits_c).cpu().numpy()

        # Gather the routed classifier's prob per window
        n = X_std.shape[0]
        final_probs = all_probs[selected_c, np.arange(n)]
        return final_probs


# ===========================================================================
# UCB training
# ===========================================================================

def fit_ucb(
    a2c_fit: A2CFit,
    base_classifiers: list[AdversarialMLPFit],
    train_traces: list,
    cfg: UCBConfig | None = None,
    verbose: bool = True,
) -> UCBFit:
    """Train UCB statistics via single-pass sampling over synthetic patterns.

    Steps:
      1. Generate synthetic data (same as A2C training).
      2. A2C predicts pattern for each example.
      3. For each example in randomized order:
         a. UCB sampling rule selects a classifier.
         b. Classifier predicts; reward = (prediction == true_label).
         c. Update (n_pulls, reward_sums) for the (pred_pattern, classifier) cell.
      4. mean_rewards = reward_sums / n_pulls.
    """
    cfg = cfg or UCBConfig()

    # Generate the same synthetic data the A2C saw (for consistency)
    feature_means = a2c_fit.feature_means
    feature_stds = a2c_fit.feature_stds
    X_raw, y_raw, _ = windows_and_labels(train_traces, binary=True)
    X_std = (X_raw - feature_means) / feature_stds

    rng = np.random.default_rng(cfg.seed)
    X_perturbed, true_patterns, attack_labels = generate_synthetic_patterns(
        X_std, y_raw, base_classifiers, a2c_fit.cfg, rng=rng,
    )
    if verbose:
        print(f"  UCB training set: {X_perturbed.shape[0]} examples")

    # A2C predicts pattern for each example
    states = compute_state(X_perturbed, base_classifiers)
    state_t = torch.from_numpy(states.astype(np.float32))
    a2c_fit.policy.eval()
    with torch.no_grad():
        logits = a2c_fit.policy(state_t)
        predicted_patterns = logits.argmax(dim=-1).numpy()

    # Pre-compute every base classifier's prediction on every example
    X_perturbed_t = torch.from_numpy(X_perturbed.astype(np.float32))
    classifier_probs = np.zeros((len(base_classifiers), X_perturbed.shape[0]), dtype=float)
    for c_idx, clf in enumerate(base_classifiers):
        clf.model.eval()
        with torch.no_grad():
            logits_c = clf.model(X_perturbed_t)
            classifier_probs[c_idx] = torch.sigmoid(logits_c).cpu().numpy()
    classifier_preds = (classifier_probs >= 0.5).astype(int)
    classifier_correct = (classifier_preds == attack_labels[None, :].astype(int)).astype(float)
    # classifier_correct shape: (n_classifiers, n_examples) — reward for each (clf, ex) pair

    # UCB sampling pass — randomized order for IID assumption
    n_pulls = np.zeros((cfg.n_patterns, cfg.n_classifiers), dtype=float)
    reward_sums = np.zeros((cfg.n_patterns, cfg.n_classifiers), dtype=float)

    order = rng.permutation(X_perturbed.shape[0])

    for step, ex_idx in enumerate(order):
        p = int(predicted_patterns[ex_idx])

        # UCB selection rule
        N_p = n_pulls[p].sum()
        if N_p < cfg.n_classifiers:
            # Force-pull each classifier under this pattern at least once
            chosen = int(N_p)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ucb_bonus = cfg.exploration_c * np.sqrt(
                    2.0 * np.log(N_p) / np.maximum(n_pulls[p], 1e-9)
                )
            ucb_values = (reward_sums[p] / np.maximum(n_pulls[p], 1e-9)) + ucb_bonus
            chosen = int(ucb_values.argmax())

        reward = float(classifier_correct[chosen, ex_idx])
        n_pulls[p, chosen] += 1
        reward_sums[p, chosen] += reward

    mean_rewards = np.where(
        n_pulls > 0,
        reward_sums / np.maximum(n_pulls, 1),
        cfg.smoothing_prior,
    )

    if verbose:
        print(f"  UCB sampling completed in {len(order)} steps")

    return UCBFit(
        cfg=cfg,
        n_pulls=n_pulls,
        reward_sums=reward_sums,
        mean_rewards=mean_rewards,
        training_steps=len(order),
    )


# ===========================================================================
# Reporting
# ===========================================================================

def print_ucb_table(ucb_fit: UCBFit, epsilons: list[float] = [0.05, 0.10, 0.20]) -> None:
    """Print the UCB mean_reward table + the resulting routing decision."""
    print("\n" + "=" * 78)
    print("UCB Mean Reward Table  (rows = A2C-predicted pattern, "
          "cols = base classifier)")
    print("=" * 78)
    classifier_labels = [f"ε={e:.2f}" for e in epsilons]
    header = f"{'Pattern':<22s}"
    for lbl in classifier_labels:
        header += f"  {lbl:>10s}"
    header += f"  {'→ routed':>12s}"
    header += f"  {'n_pulls':>10s}"
    print(header)
    print("-" * 78)

    routing = ucb_fit.routing_table
    for p in range(N_PATTERNS):
        row = f"{PATTERN_NAMES[p]:<22s}"
        for c_idx in range(ucb_fit.cfg.n_classifiers):
            r = ucb_fit.mean_rewards[p, c_idx]
            row += f"  {r:>10.4f}"
        chosen_idx = int(routing[p])
        chosen_label = classifier_labels[chosen_idx]
        total_p = int(ucb_fit.n_pulls[p].sum())
        row += f"  {chosen_label:>12s}"
        row += f"  {total_p:>10d}"
        print(row)


def evaluate_d6_on_real_traces(
    ucb_fit: UCBFit,
    a2c_fit: A2CFit,
    base_classifiers: list[AdversarialMLPFit],
    test_traces: list,
    trace_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Evaluate end-to-end D6 on real (unperturbed) test traces.

    Reports the D6 operating point: per-window TPR/FPR, per-trace metrics,
    trace AUC, and per-attack-class detection.
    """
    from hmd_vanilla import (
        aggregate_per_trace, binary_metrics, per_attack_class_tpr,
    )
    from sklearn.metrics import roc_auc_score

    X_test, y_test_win, tids = windows_and_labels(test_traces, binary=True)
    probs_win = ucb_fit.predict_proba_windows(X_test, a2c_fit, base_classifiers)
    preds_win = (probs_win >= 0.5).astype(int)
    win_m = binary_metrics(y_test_win, preds_win)

    attack_frac, y_test_trace = aggregate_per_trace(test_traces, probs_win, tids)
    preds_trace = (attack_frac >= trace_threshold).astype(int)
    trace_m = binary_metrics(y_test_trace, preds_trace)
    try:
        trace_auc = float(roc_auc_score(y_test_trace, attack_frac))
    except ValueError:
        trace_auc = float("nan")

    per_atk_tpr = per_attack_class_tpr(
        test_traces, attack_frac, threshold=trace_threshold,
    )

    # Routing statistics on the real test set: which classifier got used how often?
    X_std = (X_test - a2c_fit.feature_means) / a2c_fit.feature_stds
    states = compute_state(X_std, base_classifiers)
    state_t = torch.from_numpy(states.astype(np.float32))
    a2c_fit.policy.eval()
    with torch.no_grad():
        predicted_patterns = a2c_fit.policy(state_t).argmax(dim=-1).numpy()
    routed_classifiers = ucb_fit.routing_table[predicted_patterns]
    from collections import Counter
    routing_dist = Counter(routed_classifiers.tolist())

    if verbose:
        print()
        print("=" * 78)
        print("D6 END-TO-END EVALUATION (real test traces, single fold)")
        print("=" * 78)
        print(f"  Per-window TPR: {win_m['tpr']:.4f}")
        print(f"  Per-window FPR: {win_m['fpr']:.4f}")
        print(f"  Per-trace  TPR: {trace_m['tpr']:.4f}")
        print(f"  Per-trace  FPR: {trace_m['fpr']:.4f}")
        print(f"  Per-trace AUC : {trace_auc:.4f}")
        print()
        print("  Per-attack-class detection rate (per-trace at threshold 0.5):")
        for cls in ATTACK_CLASSES:
            val = per_atk_tpr.get(cls, float("nan"))
            print(f"    {cls:<22s} {val:.4f}")
        print()
        print("  Routing distribution on REAL test windows:")
        for c_idx in range(ucb_fit.cfg.n_classifiers):
            count = routing_dist.get(c_idx, 0)
            frac = count / len(routed_classifiers) if len(routed_classifiers) else 0
            print(f"    ε={[0.05, 0.10, 0.20][c_idx]:.2f} classifier: "
                  f"{count:>6d} windows ({frac:.1%})")

    return {
        "window_tpr": win_m["tpr"],
        "window_fpr": win_m["fpr"],
        "trace_tpr": trace_m["tpr"],
        "trace_fpr": trace_m["fpr"],
        "trace_auc": trace_auc,
        "per_attack_tpr": per_atk_tpr,
        "routing_distribution": dict(routing_dist),
    }


# ===========================================================================
# Driver — full D6 pipeline standalone
# ===========================================================================

if __name__ == "__main__":
    from dataset_parquet import load_real_parquet
    from dataset import stratified_kfold_traces

    print("=" * 78)
    print("D6 Module 3 — UCB Defense-Assignment Controller")
    print("End-to-end D6 pipeline standalone evaluation")
    print("=" * 78)
    print()

    ds = load_real_parquet(
        "./data/traces",
        window_aggregation_factor=100,
        verbose=True,
    )
    print()
    print(ds.summary())
    print()

    fold_iter = list(stratified_kfold_traces(ds, n_folds=5, seed=0))
    train_traces, test_traces = fold_iter[0]
    print(f"Standalone evaluation on fold 0: "
          f"{len(train_traces)} train traces, {len(test_traces)} test traces")

    epsilons = [0.05, 0.10, 0.20]

    # ----- Step 1: train three base classifiers -----
    print()
    print("=" * 78)
    print("Step 1: Train three Module 1 base classifiers")
    print("=" * 78)
    t_step1 = time.perf_counter()
    base_classifiers: list[AdversarialMLPFit] = []
    for eps in epsilons:
        t0 = time.perf_counter()
        cfg_mlp = AdversarialMLPConfig(fgsm_epsilon=eps, seed=0)
        fit = fit_adversarial_mlp(train_traces, cfg=cfg_mlp, verbose=False)
        print(f"  Adv MLP ε={eps}: trained in {time.perf_counter() - t0:.1f}s  "
              f"(clean_acc={fit.clean_acc:.4f} adv_acc={fit.adv_acc:.4f})")
        base_classifiers.append(fit)
    print(f"  Module 1 total: {time.perf_counter() - t_step1:.1f}s")

    # ----- Step 2: train A2C -----
    print()
    print("=" * 78)
    print("Step 2: Train Module 2 A2C agent")
    print("=" * 78)
    cfg_a2c = A2CConfig(seed=0)
    t0 = time.perf_counter()
    a2c_fit = train_a2c(base_classifiers, train_traces, cfg=cfg_a2c, verbose=False)
    print(f"  A2C trained in {time.perf_counter() - t0:.1f}s "
          f"(final training pattern_acc={a2c_fit.training_history[-1]['pattern_acc']:.4f})")

    # ----- Step 3: train UCB -----
    print()
    print("=" * 78)
    print("Step 3: Train Module 3 UCB controller")
    print("=" * 78)
    cfg_ucb = UCBConfig(seed=0)
    t0 = time.perf_counter()
    ucb_fit = fit_ucb(
        a2c_fit, base_classifiers, train_traces, cfg=cfg_ucb, verbose=True,
    )
    print(f"  UCB trained in {time.perf_counter() - t0:.1f}s")

    # Show routing table
    print_ucb_table(ucb_fit, epsilons=epsilons)

    # ----- Step 4: evaluate D6 end-to-end on the real test fold -----
    eval_result = evaluate_d6_on_real_traces(
        ucb_fit, a2c_fit, base_classifiers, test_traces, verbose=True,
    )

    print()
    print("=" * 78)
    print(f"Module 3 SUMMARY:  D6 operating point (fold 0) = "
          f"(TPR={eval_result['window_tpr']:.4f}, "
          f"FPR={eval_result['window_fpr']:.4f}), "
          f"trace AUC={eval_result['trace_auc']:.4f}")
    print("=" * 78)
    print()
    print("Module 3 complete. Module 4 (5-fold CV harness + figure) is the last step.")
