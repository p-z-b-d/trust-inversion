"""
D6 Module 2: A2C Agent for Adversarial Pattern Prediction
==========================================================

Second component of D6. The A2C agent observes the per-window state and
predicts which of six adversarial perturbation patterns is being applied.
Module 3 (UCB) will use the predicted pattern to route the input to
whichever base classifier handles that pattern best.

State representation (decision (b), 2026-05-26)
-----------------------------------------------
Each window's state is 6-dimensional:
  state[0:3] — standardized window features (IPC, cache_miss, branch_miss)
  state[3:6] — probability outputs from the three Module 1 base classifiers
               (ε=0.05, ε=0.10, ε=0.20)

The base-classifier outputs let A2C use *disagreement* among classifiers as
evidence of perturbation — important because adversarial inputs typically
cause heterogeneous responses across robustness profiles.

Action space (6 discrete patterns)
----------------------------------
  0  no_attack            (clean input)
  1  small_fgsm           (FGSM ε=0.05 — small adversarial perturbation)
  2  medium_fgsm          (FGSM ε=0.10 — moderate perturbation)
  3  large_fgsm           (FGSM ε=0.20 — large perturbation)
  4  gaussian_noise       (random N(0, σ²) noise; off-FGSM-direction)
  5  distribution_shift   (mimicry-style; moves toward opposite class)

NOT in this list: AP5 (composition-aware mimicry) — that's the novel
attack we test against D6 in Phase 4. A2C will misclassify AP5-style
inputs as something else (probably 0 or 5), and UCB will route them to
a base classifier that doesn't defend against AP5. That misrouting is
the structural failure D6 will demonstrate.

Networks (~3.5K params total)
-----------------------------
Policy:  MLP(6 → 32 → 32 → 6)  softmax over actions
Value:   MLP(6 → 32 → 32 → 1)  scalar value baseline

Single-step A2C
---------------
Each window is an independent episode. For a single-step problem, A2C
reduces to value-baselined policy gradient — equivalent in expectation
to supervised classification with a learned baseline. We use the full
A2C loss (policy + value + entropy) for paper-faithfulness with He et
al. 2024's described architecture.

Loss:
  total = policy_loss + value_coef·value_loss − entropy_coef·H(π)
  policy_loss = −E[log π(a|s) · A(s,a)]   with A = R − V(s).detach()
  value_loss  = MSE(V(s), R)
  H(π)        = −Σ π log π                  (entropy bonus)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ATTACK_CLASSES
from hmd_vanilla import windows_and_labels
from d6_adversarial import (
    AdversarialMLPConfig, AdversarialMLPFit, fit_adversarial_mlp,
    fgsm_perturbation,
)


# ===========================================================================
# Pattern constants
# ===========================================================================

PATTERN_NAMES = [
    "no_attack",
    "small_fgsm",
    "medium_fgsm",
    "large_fgsm",
    "gaussian_noise",
    "distribution_shift",
]
N_PATTERNS = len(PATTERN_NAMES)


# ===========================================================================
# Networks
# ===========================================================================

class PolicyNetwork(nn.Module):
    """Softmax policy over discrete adversarial patterns."""

    def __init__(self, state_dim: int, n_actions: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits, shape (batch, n_actions)


class ValueNetwork(nn.Module):
    """Scalar state-value baseline V(s)."""

    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


# ===========================================================================
# Config and Fit dataclasses
# ===========================================================================

@dataclass
class A2CConfig:
    state_dim: int = 6                # 3 features + 3 base-classifier outputs
    n_actions: int = N_PATTERNS
    hidden_size: int = 32

    # Training
    n_epochs: int = 50                # 30 → 50: A2C needs longer convergence with
                                      #          stronger entropy regularization
    batch_size: int = 128
    lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.05        # 0.01 → 0.05: prevent policy collapse on
                                      #              under-represented patterns
                                      #              (e.g. gaussian_noise)
    ce_aux_coef: float = 0.5          # NEW: cross-entropy auxiliary loss weight.
                                      #      Adds dense supervised gradient on top of
                                      #      A2C's sparse policy-gradient signal.
                                      #      Modest deviation from pure He et al. A2C
                                      #      to address slow convergence on our small
                                      #      (3-feature) state space; documented in
                                      #      paper as imitation-regularization aux.
    seed: int = 0

    # Synthetic-pattern generator parameters — tuned to make patterns geometrically
    # distinct in the 3-feature space (avoid small_fgsm ≈ no_attack indistinguishability)
    fgsm_epsilon_small: float = 0.08  # 0.05 → 0.08: pull small_fgsm out of the
                                      #              "indistinguishable from clean"
                                      #              regime created by adversarial training
    fgsm_epsilon_medium: float = 0.10
    fgsm_epsilon_large: float = 0.20
    gaussian_sigma: float = 0.20      # 0.10 → 0.20: differentiate from sign-aligned
                                      #              FGSM at comparable magnitude
    distribution_shift_alpha: float = 0.5
    fgsm_target_classifier_idx: int = 1     # which base classifier's gradient drives FGSM


@dataclass
class A2CFit:
    cfg: A2CConfig
    policy: nn.Module
    value: nn.Module
    training_history: list[dict]
    feature_means: np.ndarray
    feature_stds: np.ndarray

    def predict_pattern(
        self,
        X_raw: np.ndarray,
        base_classifiers: list[AdversarialMLPFit],
    ) -> np.ndarray:
        """Predict pattern label for each window in X_raw (unstandardized).

        Returns (n_windows,) int64 array — argmax of policy.
        """
        X_std = (X_raw - self.feature_means) / self.feature_stds
        state = compute_state(X_std, base_classifiers)
        state_t = torch.from_numpy(state.astype(np.float32))
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(state_t)
            return logits.argmax(dim=-1).numpy()


# ===========================================================================
# Synthetic perturbation generator
# ===========================================================================

def generate_synthetic_patterns(
    X_std: np.ndarray,
    y_true: np.ndarray,
    base_classifiers: list[AdversarialMLPFit],
    cfg: A2CConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (x, y), generate 6 perturbed copies (one per pattern).

    Inputs
    ------
    X_std : (N, 3)  standardized features
    y_true : (N,)  binary attack labels (used for distribution_shift target)
    base_classifiers : list of 3 AdversarialMLPFit — used to compute FGSM direction
    cfg : A2CConfig
    rng : optional numpy RNG

    Returns
    -------
    X_perturbed : (6N, 3)
    pattern_labels : (6N,)
    attack_labels : (6N,)
    """
    rng = rng or np.random.default_rng(cfg.seed)
    N = X_std.shape[0]

    # FGSM direction against the medium-ε base classifier
    target_model = base_classifiers[cfg.fgsm_target_classifier_idx].model
    X_t = torch.from_numpy(X_std.astype(np.float32))
    y_t = torch.from_numpy(y_true.astype(np.float32))
    perturbation = fgsm_perturbation(target_model, X_t, y_t, epsilon=1.0)
    fgsm_sign = perturbation.numpy().astype(np.float32)  # already sign-scaled at ε=1

    # Cluster centers in standardized space (for distribution_shift)
    attack_mask = y_true == 1
    benign_mask = y_true == 0
    center_attack = X_std[attack_mask].mean(axis=0) if attack_mask.any() else np.zeros(X_std.shape[1])
    center_benign = X_std[benign_mask].mean(axis=0) if benign_mask.any() else np.zeros(X_std.shape[1])
    target_centers = np.where(
        y_true.reshape(-1, 1) == 1,
        center_benign.reshape(1, -1),
        center_attack.reshape(1, -1),
    ).astype(np.float32)

    # Build perturbed sets in this fixed order matching PATTERN_NAMES
    X_lists: list[np.ndarray] = []
    pattern_lists: list[np.ndarray] = []

    # 0 — no_attack: clean
    X_lists.append(X_std.astype(np.float32).copy())
    pattern_lists.append(np.zeros(N, dtype=np.int64))

    # 1 — small_fgsm
    X_lists.append((X_std + cfg.fgsm_epsilon_small * fgsm_sign).astype(np.float32))
    pattern_lists.append(np.ones(N, dtype=np.int64))

    # 2 — medium_fgsm
    X_lists.append((X_std + cfg.fgsm_epsilon_medium * fgsm_sign).astype(np.float32))
    pattern_lists.append(np.full(N, 2, dtype=np.int64))

    # 3 — large_fgsm
    X_lists.append((X_std + cfg.fgsm_epsilon_large * fgsm_sign).astype(np.float32))
    pattern_lists.append(np.full(N, 3, dtype=np.int64))

    # 4 — gaussian_noise
    noise = rng.normal(0.0, cfg.gaussian_sigma, size=X_std.shape).astype(np.float32)
    X_lists.append((X_std + noise).astype(np.float32))
    pattern_lists.append(np.full(N, 4, dtype=np.int64))

    # 5 — distribution_shift toward opposite class
    shifted = X_std + cfg.distribution_shift_alpha * (target_centers - X_std)
    X_lists.append(shifted.astype(np.float32))
    pattern_lists.append(np.full(N, 5, dtype=np.int64))

    X_perturbed = np.concatenate(X_lists, axis=0)
    pattern_labels = np.concatenate(pattern_lists)
    attack_labels = np.tile(y_true.astype(np.float32), N_PATTERNS)

    return X_perturbed, pattern_labels, attack_labels


# ===========================================================================
# State computation (decision (b))
# ===========================================================================

def compute_state(
    X_std: np.ndarray,
    base_classifiers: list[AdversarialMLPFit],
) -> np.ndarray:
    """6-dim state: [standardized features, p1, p2, p3] where pk is the k-th
    base classifier's attack probability on the (already-standardized) input.

    Inputs
    ------
    X_std : (N, 3)  ALREADY-STANDARDIZED features (consistent with base classifier scaling)
    base_classifiers : 3 AdversarialMLPFit objects

    Returns
    -------
    state : (N, 6)
    """
    X_t = torch.from_numpy(X_std.astype(np.float32))
    probs_cols: list[np.ndarray] = []
    for clf in base_classifiers:
        clf.model.eval()
        with torch.no_grad():
            logits = clf.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        probs_cols.append(probs.reshape(-1, 1))
    base_outputs = np.concatenate(probs_cols, axis=1).astype(np.float32)
    return np.concatenate([X_std.astype(np.float32), base_outputs], axis=1)


# ===========================================================================
# A2C training
# ===========================================================================

def train_a2c(
    base_classifiers: list[AdversarialMLPFit],
    train_traces: list,
    cfg: A2CConfig | None = None,
    verbose: bool = True,
) -> A2CFit:
    """Train A2C on synthetic patterns generated from train_traces."""
    cfg = cfg or A2CConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Use the first base classifier's standardization stats — all three were
    # trained on the same fold so their stats are equal.
    feature_means = base_classifiers[0].feature_means
    feature_stds = base_classifiers[0].feature_stds

    # Raw training data → standardized
    X_raw, y_raw, _ = windows_and_labels(train_traces, binary=True)
    X_std = (X_raw - feature_means) / feature_stds

    # Synthetic patterns
    rng = np.random.default_rng(cfg.seed)
    X_perturbed, pattern_labels, _ = generate_synthetic_patterns(
        X_std, y_raw, base_classifiers, cfg, rng=rng,
    )
    states = compute_state(X_perturbed, base_classifiers)

    if verbose:
        print(f"  Synthetic dataset: {states.shape[0]} examples, "
              f"{N_PATTERNS} patterns")

    policy = PolicyNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_size)
    value = ValueNetwork(cfg.state_dim, cfg.hidden_size)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=cfg.lr,
    )

    states_t = torch.from_numpy(states.astype(np.float32))
    patterns_t = torch.from_numpy(pattern_labels.astype(np.int64))
    n_samples = states_t.size(0)

    history: list[dict] = []
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_samples)

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_ce_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for i in range(0, n_samples, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            s_batch = states_t[idx]
            p_batch = patterns_t[idx]

            # Forward
            logits = policy(s_batch)
            log_probs_all = F.log_softmax(logits, dim=-1)
            probs_all = F.softmax(logits, dim=-1)

            # Sample actions stochastically (training-time exploration)
            dist = torch.distributions.Categorical(probs=probs_all)
            actions = dist.sample()

            # Reward: 1 if action matches the true pattern, else 0
            rewards = (actions == p_batch).float()

            # Value baseline
            values_pred = value(s_batch)
            advantages = (rewards - values_pred.detach())

            # Policy gradient with advantage baseline
            chosen_log_probs = log_probs_all.gather(
                1, actions.unsqueeze(-1)
            ).squeeze(-1)
            policy_loss = -(chosen_log_probs * advantages).mean()

            # Value regression
            value_loss = F.mse_loss(values_pred, rewards)

            # Entropy regularizer
            entropy = -(probs_all * log_probs_all).sum(dim=-1).mean()

            # Cross-entropy auxiliary (dense supervised gradient on the policy net)
            ce_loss = F.cross_entropy(logits, p_batch)

            total_loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
                + cfg.ce_aux_coef * ce_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_actions = logits.argmax(dim=-1)
                epoch_correct += (pred_actions == p_batch).sum().item()
                epoch_total += p_batch.size(0)
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_ce_loss += ce_loss.item()
                n_batches += 1

        history.append({
            "epoch": epoch + 1,
            "policy_loss": epoch_policy_loss / n_batches,
            "value_loss": epoch_value_loss / n_batches,
            "entropy": epoch_entropy / n_batches,
            "ce_loss": epoch_ce_loss / n_batches,
            "pattern_acc": epoch_correct / epoch_total,
        })

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            h = history[-1]
            print(f"    epoch {h['epoch']:>2d}  "
                  f"policy={h['policy_loss']:+.4f}  "
                  f"value={h['value_loss']:.4f}  "
                  f"ce={h['ce_loss']:.4f}  "
                  f"H={h['entropy']:.4f}  "
                  f"acc={h['pattern_acc']:.4f}")

    return A2CFit(
        cfg=cfg,
        policy=policy,
        value=value,
        training_history=history,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_pattern_classification(
    fit: A2CFit,
    base_classifiers: list[AdversarialMLPFit],
    test_traces: list,
    verbose: bool = True,
) -> dict:
    """Held-out pattern classification accuracy + per-pattern + confusion matrix."""
    X_raw, y_raw, _ = windows_and_labels(test_traces, binary=True)
    X_std = (X_raw - fit.feature_means) / fit.feature_stds

    rng = np.random.default_rng(fit.cfg.seed + 99999)
    X_perturbed, pattern_labels, _ = generate_synthetic_patterns(
        X_std, y_raw, base_classifiers, fit.cfg, rng=rng,
    )
    states = compute_state(X_perturbed, base_classifiers)
    state_t = torch.from_numpy(states.astype(np.float32))

    fit.policy.eval()
    with torch.no_grad():
        logits = fit.policy(state_t)
        predicted = logits.argmax(dim=-1).numpy()

    overall_acc = float((predicted == pattern_labels).mean())

    per_pattern: dict[str, tuple[float, int]] = {}
    for p in range(N_PATTERNS):
        mask = pattern_labels == p
        if mask.sum() > 0:
            acc = float((predicted[mask] == p).mean())
            per_pattern[PATTERN_NAMES[p]] = (acc, int(mask.sum()))

    confusion = np.zeros((N_PATTERNS, N_PATTERNS), dtype=int)
    for true_p, pred_p in zip(pattern_labels, predicted):
        confusion[true_p, pred_p] += 1

    if verbose:
        print()
        print("PER-PATTERN ACCURACY (held-out fold):")
        for name in PATTERN_NAMES:
            acc, count = per_pattern.get(name, (float("nan"), 0))
            print(f"  {name:<22s} {acc:.4f}  (n={count})")
        print(f"\n  Overall pattern classification accuracy: {overall_acc:.4f}")

        print()
        print("CONFUSION MATRIX (rows = true pattern, columns = predicted):")
        col_header = "  " + " ".join(f"{n[:6]:>7s}" for n in PATTERN_NAMES)
        print(f"{'true':<22s}{col_header}")
        for i, name in enumerate(PATTERN_NAMES):
            row = f"  {name:<20s}"
            for j in range(N_PATTERNS):
                row += f"{confusion[i, j]:>8d}"
            print(row)

    # Sanity check: predict patterns on real (un-perturbed) test windows
    real_states = compute_state(X_std, base_classifiers)
    real_state_t = torch.from_numpy(real_states.astype(np.float32))
    fit.policy.eval()
    with torch.no_grad():
        real_logits = fit.policy(real_state_t)
        real_predicted = real_logits.argmax(dim=-1).numpy()
    from collections import Counter
    real_pattern_dist = Counter(real_predicted.tolist())
    if verbose:
        print()
        print("Predicted pattern distribution on REAL (unperturbed) test windows:")
        print("  (should be heavily 'no_attack' — these are real clean inputs)")
        for p in range(N_PATTERNS):
            count = real_pattern_dist.get(p, 0)
            frac = count / len(real_predicted) if len(real_predicted) else 0
            print(f"    {PATTERN_NAMES[p]:<22s} {count:>6d}  ({frac:.1%})")

    return {
        "overall_acc": overall_acc,
        "per_pattern": per_pattern,
        "confusion": confusion,
        "real_window_pattern_dist": dict(real_pattern_dist),
    }


# ===========================================================================
# Driver — train all of D6 Module 1 + Module 2 + standalone evaluation
# ===========================================================================

if __name__ == "__main__":
    from dataset_parquet import load_real_parquet
    from dataset import stratified_kfold_traces

    print("=" * 78)
    print("D6 Module 2 — A2C Agent for Adversarial Pattern Prediction")
    print("State (b): window features + base classifier outputs (6-dim)")
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

    # Standalone evaluation: single train/test split (fold 0)
    fold_iter = list(stratified_kfold_traces(ds, n_folds=5, seed=0))
    train_traces, test_traces = fold_iter[0]
    print(f"Standalone evaluation on fold 0: "
          f"{len(train_traces)} train traces, {len(test_traces)} test traces")

    # ----- Step 1: train three base classifiers (Module 1) -----
    print()
    print("=" * 78)
    print("Step 1: Train three base classifiers (Module 1)")
    print("=" * 78)
    epsilons = [0.05, 0.10, 0.20]
    base_classifiers: list[AdversarialMLPFit] = []
    t_step1 = time.perf_counter()
    for eps in epsilons:
        t0 = time.perf_counter()
        cfg_mlp = AdversarialMLPConfig(fgsm_epsilon=eps, seed=0)
        fit = fit_adversarial_mlp(train_traces, cfg=cfg_mlp, verbose=False)
        print(f"  Adv MLP ε={eps}: trained in {time.perf_counter() - t0:.1f}s  "
              f"(clean_acc={fit.clean_acc:.4f} adv_acc={fit.adv_acc:.4f})")
        base_classifiers.append(fit)
    print(f"  Total Module 1 time: {time.perf_counter() - t_step1:.1f}s")

    # ----- Step 2: train A2C -----
    print()
    print("=" * 78)
    print("Step 2: Train A2C agent on synthetic adversarial patterns")
    print("=" * 78)
    cfg_a2c = A2CConfig(seed=0)
    t0 = time.perf_counter()
    a2c_fit = train_a2c(base_classifiers, train_traces, cfg=cfg_a2c, verbose=True)
    print(f"\n  A2C training time: {time.perf_counter() - t0:.1f}s")

    # ----- Step 3: evaluate pattern classification on held-out fold -----
    print()
    print("=" * 78)
    print("Step 3: Evaluate A2C pattern classification on held-out fold")
    print("=" * 78)
    eval_result = evaluate_pattern_classification(
        a2c_fit, base_classifiers, test_traces, verbose=True,
    )

    print()
    print("=" * 78)
    print(f"Module 2 sanity check: A2C overall pattern accuracy = "
          f"{eval_result['overall_acc']:.4f}")
    print("=" * 78)
    print()
    print("Module 2 complete. The trained A2C agent will be reused by Module 3 (UCB).")
