"""
D6 Module 1: Adversarially-Trained MLP Base Classifiers
========================================================

First component of D6 — the FGSM-trained base classifiers that the
A2C+UCB ensemble (later modules) will select among.

Reference: He, Homayoun, Sayadi, "Beyond Conventional Defenses: Proactive
and Adversarial-Resilient Hardware Malware Detection using Deep
Reinforcement Learning," DAC 2024.

Three small MLPs trained with progressively larger FGSM perturbation
magnitudes — each ends up with a different decision-boundary profile:

  MLP_eps_005:  ε = 0.05  (tight boundary, small-attack robustness)
  MLP_eps_010:  ε = 0.10  (medium boundary)
  MLP_eps_020:  ε = 0.20  (wide boundary, large-attack robustness)

Why three?
----------
The He et al. architecture relies on a UCB controller routing each input
to whichever base classifier has the best track record against the
adversarial pattern the A2C agent predicts is being applied. For this
routing to be meaningful, the base classifiers must have *different*
robustness profiles. Adversarial training with different ε values is the
canonical way to produce that diversity.

Architecture (per base classifier, ~370 params)
-----------------------------------------------
  Linear(3 → 16) → ReLU
  Linear(16 → 16) → ReLU → Dropout(0.1)
  Linear(16 → 1)                              # logit output

FGSM training procedure (mixed Madry-style)
-------------------------------------------
Each training batch:
  1. Compute adversarial perturbation x_adv = x + ε · sign(∇_x L) using
     current model parameters.
  2. Train on the union of clean (x, y) and adversarial (x_adv, y) batches.
  3. Mixed training preserves clean accuracy while building adversarial
     robustness — preferred over pure-adversarial training which often
     catastrophically forgets clean performance.

Input representation: same 3 derived features as D1/D3/D4/D5 (IPC,
cache_miss, branch_miss) — NOT the 6-counter time-series from D2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset import ATTACK_CLASSES, stratified_kfold_traces
from hmd_vanilla import (
    windows_and_labels, aggregate_per_trace,
    binary_metrics, per_attack_class_tpr,
    FoldResult, summarize_cv, print_summary,
)


# ===========================================================================
# Model
# ===========================================================================

class AdversarialMLP(nn.Module):
    """Small MLP for D6 base classifier — adversarially trainable via FGSM."""

    def __init__(
        self,
        n_features: int = 3,
        hidden_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)  # logits, shape (batch,)


# ===========================================================================
# FGSM perturbation
# ===========================================================================

def fgsm_perturbation(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Generate FGSM adversarial perturbation for input X with label y.

    Computes ∇_x BCEWithLogits(model(x), y) and returns the sign-perturbation
    scaled by ε. The model is set to eval() during gradient computation to
    disable dropout (so the perturbation is deterministic given x, y, model
    parameters). Returns a detached tensor — caller adds to X for use in
    training the *next* gradient step.
    """
    X = X.clone().detach().requires_grad_(True)
    was_training = model.training
    model.eval()
    try:
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss.backward()
        perturbation = epsilon * X.grad.sign()
    finally:
        if was_training:
            model.train()
    return perturbation.detach()


# ===========================================================================
# Config and Fit dataclasses
# ===========================================================================

@dataclass
class AdversarialMLPConfig:
    n_features: int = 3
    hidden_size: int = 16
    dropout: float = 0.1
    fgsm_epsilon: float = 0.1     # adversarial perturbation magnitude

    use_scaler: bool = True
    n_epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0


@dataclass
class AdversarialMLPFit:
    cfg: AdversarialMLPConfig
    model: nn.Module
    feature_means: np.ndarray   # shape (n_features,)
    feature_stds: np.ndarray    # shape (n_features,)
    training_history: list[dict]
    clean_acc: float            # final-epoch training accuracy on clean inputs
    adv_acc: float              # final-epoch training accuracy on FGSM-perturbed inputs

    def predict_proba_windows(self, X: np.ndarray) -> np.ndarray:
        """X: (n_windows, n_features) raw features → probs (n_windows,)."""
        if self.cfg.use_scaler:
            X = (X - self.feature_means) / self.feature_stds
        X_t = torch.from_numpy(X.astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ===========================================================================
# Training with FGSM augmentation (mixed-style)
# ===========================================================================

def fit_adversarial_mlp(
    train_traces: list,
    cfg: AdversarialMLPConfig | None = None,
    verbose: bool = False,
) -> AdversarialMLPFit:
    cfg = cfg or AdversarialMLPConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Gather (X, y) over all training windows
    X_train, y_train, _ = windows_and_labels(train_traces, binary=True)

    # Standardize per feature
    if cfg.use_scaler:
        feature_means = X_train.mean(axis=0).astype(np.float32)
        feature_stds = X_train.std(axis=0).astype(np.float32)
        feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
        X_train = (X_train - feature_means) / feature_stds
    else:
        feature_means = np.zeros(cfg.n_features, dtype=np.float32)
        feature_stds = np.ones(cfg.n_features, dtype=np.float32)

    model = AdversarialMLP(
        n_features=cfg.n_features,
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))

    n_samples = X_t.size(0)
    batch_size = cfg.batch_size
    epsilon = cfg.fgsm_epsilon

    history: list[dict] = []
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_samples)
        X_shuffled = X_t[perm]
        y_shuffled = y_t[perm]

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # 1. FGSM perturbation using current model state
            perturbation = fgsm_perturbation(model, X_batch, y_batch, epsilon)
            X_adv = X_batch + perturbation

            # 2. Mixed training: clean ∪ adversarial
            X_combined = torch.cat([X_batch, X_adv], dim=0)
            y_combined = torch.cat([y_batch, y_batch], dim=0)

            # 3. Standard training step (clears stale gradients from FGSM step)
            model.train()
            optimizer.zero_grad()
            logits = model(X_combined)
            loss = F.binary_cross_entropy_with_logits(logits, y_combined)
            loss.backward()
            optimizer.step()

            bs = X_combined.size(0)
            epoch_loss_sum += loss.item() * bs
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                epoch_correct += (preds == y_combined).sum().item()
                epoch_total += bs

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss_sum / epoch_total,
            "acc": epoch_correct / epoch_total,
        })
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            h = history[-1]
            print(f"    epoch {h['epoch']:>2d}  "
                  f"loss={h['loss']:.4f}  acc={h['acc']:.4f}")

    # Final clean and adversarial accuracy on training data — characterizes
    # the trained classifier's robustness profile.
    model.eval()
    with torch.no_grad():
        clean_logits = model(X_t)
        clean_preds = (torch.sigmoid(clean_logits) >= 0.5).float()
        clean_acc = float((clean_preds == y_t).float().mean().item())

    perturbation = fgsm_perturbation(model, X_t, y_t, epsilon)
    X_t_adv = X_t + perturbation
    with torch.no_grad():
        adv_logits = model(X_t_adv)
        adv_preds = (torch.sigmoid(adv_logits) >= 0.5).float()
        adv_acc = float((adv_preds == y_t).float().mean().item())

    return AdversarialMLPFit(
        cfg=cfg,
        model=model,
        feature_means=feature_means,
        feature_stds=feature_stds,
        training_history=history,
        clean_acc=clean_acc,
        adv_acc=adv_acc,
    )


# ===========================================================================
# Cross-validation (standalone characterization of base classifiers)
# ===========================================================================

def cross_validate_adversarial_mlp(
    dataset,
    cfg: AdversarialMLPConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    trace_threshold: float = 0.5,
    verbose: bool = True,
) -> list[FoldResult]:
    """5-fold CV mirroring the existing protocols.

    Useful as a sanity check: each base classifier should detect attacks
    well on clean inputs. The A2C+UCB ensemble layer (module 2+3) only
    adds value if these base classifiers are individually competent.
    """
    cfg = cfg or AdversarialMLPConfig()
    results: list[FoldResult] = []

    for fold_idx, (train_traces, test_traces) in enumerate(
        stratified_kfold_traces(dataset, n_folds=n_folds, seed=cv_seed)
    ):
        t_start = time.perf_counter()
        fold_cfg = dataclasses.replace(cfg, seed=cfg.seed + fold_idx * 10_000)
        fit = fit_adversarial_mlp(train_traces, cfg=fold_cfg, verbose=False)

        X_test, y_test_win, tids = windows_and_labels(test_traces, binary=True)
        probs_win = fit.predict_proba_windows(X_test)
        preds_win = (probs_win >= 0.5).astype(int)
        win_m = binary_metrics(y_test_win, preds_win)

        attack_frac, y_test_trace = aggregate_per_trace(test_traces, probs_win, tids)
        preds_trace = (attack_frac >= trace_threshold).astype(int)
        trace_m = binary_metrics(y_test_trace, preds_trace)
        try:
            trace_auc = float(roc_auc_score(y_test_trace, attack_frac))
        except ValueError:
            trace_auc = float("nan")

        per_atk_tpr = per_attack_class_tpr(test_traces, attack_frac, threshold=trace_threshold)

        elapsed = time.perf_counter() - t_start

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
            print(f"  fold {fold_idx}: trained in {elapsed:5.1f}s  "
                  f"clean_acc={fit.clean_acc:.4f} adv_acc={fit.adv_acc:.4f}  "
                  f"| win TPR={win_m['tpr']:.3f} FPR={win_m['fpr']:.3f}  "
                  f"trace AUC={trace_auc:.4f}")

    return results


# ===========================================================================
# Driver — characterize all three ε values standalone
# ===========================================================================

if __name__ == "__main__":
    from dataset_parquet import load_real_parquet

    print("=" * 78)
    print("D6 Module 1 — Adversarially-Trained MLP Base Classifiers")
    print("Three ε values, 5-fold CV per ε, standalone characterization")
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

    # Quick sanity probe: parameter count
    probe = AdversarialMLP(n_features=3, hidden_size=16, dropout=0.1)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Model parameter count: {n_params:,} per base classifier")
    print(f"Default PyTorch threads: {torch.get_num_threads()}")
    print()

    epsilons = [0.05, 0.10, 0.20]
    all_summaries: dict[float, dict] = {}
    overall_start = time.perf_counter()

    for eps in epsilons:
        print()
        print("=" * 78)
        print(f"Adversarial MLP with ε = {eps:.2f}")
        print("=" * 78)
        cfg = AdversarialMLPConfig(
            n_features=3,
            hidden_size=16,
            dropout=0.1,
            fgsm_epsilon=eps,
            n_epochs=30,
            batch_size=128,
            lr=1e-3,
            seed=0,
        )
        results = cross_validate_adversarial_mlp(ds, cfg=cfg, verbose=True)
        summary = summarize_cv(results)
        all_summaries[eps] = summary
        print_summary(summary, title=f"Adversarial MLP ε={eps:.2f}")

    overall_elapsed = time.perf_counter() - overall_start
    print()
    print("=" * 78)
    print(f"All three base classifiers trained in {overall_elapsed/60:.1f} min total")
    print("=" * 78)
    print()

    # Combined comparison table
    print("=" * 84)
    print(f"{'Base classifier':<24s} {'win TPR':>10s} {'win FPR':>10s} "
          f"{'tr TPR':>10s} {'tr FPR':>10s} {'tr AUC':>10s}")
    print("-" * 84)
    for eps in epsilons:
        s = all_summaries[eps]
        wm, _ = s["window_tpr"]; wf, _ = s["window_fpr"]
        tm, _ = s["trace_tpr"]; tf, _ = s["trace_fpr"]
        ta, _ = s["trace_auc"]
        print(f"{'Adv MLP ε=' + f'{eps:.2f}':<24s} "
              f"{wm:>10.4f} {wf:>10.4f} {tm:>10.4f} {tf:>10.4f} {ta:>10.4f}")

    print()
    print("Module 1 complete. The three trained classifiers will be reused")
    print("by Module 2 (A2C agent) and Module 3 (UCB controller).")
