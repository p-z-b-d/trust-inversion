"""
Phase 3 — D2: 1D CNN HMD (Sayadi-class)
========================================

Reimplementation of the Sayadi et al. (2018-2021) 1D-CNN hardware malware
detector class. Treats HPC counters as a multi-channel time series and
classifies each 100ms window from its 100 raw 1ms samples — the temporal
structure WITHIN the window is what D2 is designed to capture, in contrast
to D1/D3/D4 which use per-window summary statistics.

Methodological shift from D1/D3/D4
----------------------------------
- D1 vanilla RF, D3 RHMD, D4 MTD: input per window = (3,) scalar summary
  features (derived rates: IPC, cache-miss rate, branch-miss rate, computed
  from summed counters over the 100ms window).
- D2 1D CNN: input per window = (6, 100) tensor — 6 raw HPC counters
  (branch-instructions, branch-misses, cache-misses, cache-references,
  cpu-cycles, instructions) × 100 raw 1ms samples. The convolutions learn
  temporal patterns AND derived signal combinations the summary statistics
  flatten away. This matches canonical Sayadi-class methodology (Sayadi
  2018, 2020) which uses raw event counts as CNN input.

Architecture (~14.7K params)
----------------------------
  Conv1d(6 → 16, k=5, pad=2)  → ReLU → MaxPool1d(2)    # (6,100) → (16,50)
  Conv1d(16 → 16, k=5, pad=2) → ReLU → MaxPool1d(2)    # (16,50) → (16,25)
  Flatten                                              # (16*25)= 400
  Linear(400 → 32) → ReLU → Dropout(0.2)
  Linear(32 → 1)                                       # logits

Training: 30 epochs, batch 128, Adam lr=1e-3, BCEWithLogitsLoss.
Per-feature standardization computed on training fold, applied at inference.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score

from dataset import (
    ATTACK_CLASSES, CANONICAL_STATS,
    stratified_kfold_traces,
)

# Default feature ordering: 6 raw HPC counters from perf, as stored in the
# 1ms parquet traces. Sayadi-class CNN methodology takes raw counter values
# (not derived rates), letting the convolutional layers learn the relevant
# combinations internally. This is methodologically distinct from D1/D3/D4
# which use derived rates (IPC, cache_miss rate, branch_miss rate).
DEFAULT_FEATURES = [
    "branch-instructions",
    "branch-misses",
    "cache-misses",
    "cache-references",
    "cpu-cycles",
    "instructions",
]
from hmd_vanilla import (
    FoldResult, binary_metrics, per_attack_class_tpr,
    summarize_cv, print_summary,
)


# ===========================================================================
# Time-series Trace and Dataset
# ===========================================================================

@dataclass
class TimeSeriesTrace:
    """Like Trace but with a 3-D windows tensor: (n_windows, n_features, samples)."""
    class_name: str
    label: str           # "attack" or "benign"
    trace_id: str
    windows: np.ndarray  # (n_windows, n_features, window_samples)
    feature_names: list[str]

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]

    @property
    def is_attack(self) -> bool:
        return self.label == "attack"


@dataclass
class TimeSeriesDataset:
    traces: list[TimeSeriesTrace]
    feature_names: list[str]
    window_samples: int

    def summary(self) -> str:
        lines = [f"TimeSeriesDataset: {len(self.traces)} traces, "
                 f"features={self.feature_names}, "
                 f"window_samples={self.window_samples}"]
        class_counts: dict[str, int] = {}
        class_label: dict[str, str] = {}
        for t in self.traces:
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1
            class_label[t.class_name] = t.label
        for cls in sorted(class_counts.keys()):
            lines.append(f"  {cls:<22s} ({class_label[cls]:<13s})  "
                         f"n={class_counts[cls]}")
        return "\n".join(lines)


def load_real_parquet_timeseries(
    data_root: str | Path,
    window_samples: int = 100,
    feature_names: list[str] | None = None,
    verbose: bool = True,
) -> TimeSeriesDataset:
    """Load raw 1ms parquet samples and slice into per-window tensors.

    Each 100ms classification window becomes a (n_features, window_samples)
    tensor — no aggregation; the temporal structure within the window is
    preserved.
    """
    feature_names = feature_names or list(DEFAULT_FEATURES)
    data_root = Path(data_root)

    if verbose:
        print(f"  Loading time-series traces from: {data_root}")
        print(f"  Window samples per classification window: {window_samples} "
              f"(= {window_samples}ms at 1ms sampling)")
        print(f"  Features: {feature_names}")
        print()

    traces: list[TimeSeriesTrace] = []

    for cls in sorted(CANONICAL_STATS.keys()):
        cls_path = data_root / cls
        if not cls_path.exists():
            continue

        cls_label = CANONICAL_STATS[cls]["label"]
        n_loaded = 0
        for parquet_path in sorted(cls_path.glob("*.parquet")):
            df = pd.read_parquet(parquet_path)

            missing = [f for f in feature_names if f not in df.columns]
            if missing:
                if verbose:
                    print(f"    skipping {parquet_path.name} — "
                          f"missing features {missing}")
                continue

            n_samples = len(df)
            n_windows = n_samples // window_samples
            if n_windows < 1:
                continue

            X_raw = df[feature_names].to_numpy(dtype=np.float32)
            # Truncate to exact multiple of window_samples
            X_raw = X_raw[: n_windows * window_samples]

            # Reshape (n_total_samples, n_features) → (n_windows, samples, features)
            X_windowed = X_raw.reshape(n_windows, window_samples, len(feature_names))
            # Transpose to (n_windows, n_features, samples) — Conv1d expects channels-first
            X_windowed = X_windowed.transpose(0, 2, 1).astype(np.float32)

            traces.append(TimeSeriesTrace(
                class_name=cls,
                label=cls_label,
                trace_id=parquet_path.stem,
                windows=X_windowed,
                feature_names=list(feature_names),
            ))
            n_loaded += 1

        if verbose:
            print(f"  {cls}: loaded {n_loaded} traces")

    if verbose:
        total_traces = len(traces)
        total_windows = sum(t.n_windows for t in traces)
        n_classes = len({t.class_name for t in traces})
        print(f"\n  Total: {total_traces} traces across {n_classes} non-empty classes")
        print(f"  Total windows: {total_windows} "
              f"(each a {len(feature_names)}×{window_samples} tensor)")

    return TimeSeriesDataset(
        traces=traces,
        feature_names=list(feature_names),
        window_samples=window_samples,
    )


# ===========================================================================
# Model
# ===========================================================================

class CNN_HMD(nn.Module):
    """Sayadi-class 1D CNN — minimal architecture (~14K params)."""

    def __init__(
        self,
        n_features: int = 3,
        window_samples: int = 100,
        conv_channels: tuple = (16, 16),
        conv_kernel: int = 5,
        pool_size: int = 2,
        fc_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = conv_kernel // 2
        self.conv1 = nn.Conv1d(n_features, conv_channels[0],
                               kernel_size=conv_kernel, padding=pad)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1],
                               kernel_size=conv_kernel, padding=pad)
        self.pool2 = nn.MaxPool1d(pool_size)

        # After 2 pools of size pool_size: window_samples → window_samples / pool_size^2
        flat_size = conv_channels[1] * (window_samples // (pool_size ** 2))
        self.fc1 = nn.Linear(flat_size, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, window_samples)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)  # (batch,) logits
        return x


# ===========================================================================
# Config and Fit dataclasses
# ===========================================================================

@dataclass
class CNNHMDConfig:
    n_features: int = 6
    window_samples: int = 100
    conv_channels: tuple = (16, 16)
    conv_kernel: int = 5
    pool_size: int = 2
    fc_hidden: int = 32
    dropout: float = 0.2

    use_scaler: bool = True
    n_epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0
    n_threads: int = 0  # 0 = PyTorch default; set to 2 if Pi struggles


@dataclass
class CNNHMDFit:
    cfg: CNNHMDConfig
    model: nn.Module
    feature_means: np.ndarray   # (n_features,)
    feature_stds: np.ndarray    # (n_features,)
    training_history: list[dict]  # per-epoch {loss, acc}

    def predict_proba_windows(self, X: np.ndarray) -> np.ndarray:
        """X: (n_windows, n_features, window_samples) → probs (n_windows,)."""
        if self.cfg.use_scaler:
            X = (X - self.feature_means.reshape(1, -1, 1)) \
                / self.feature_stds.reshape(1, -1, 1)
        X_t = torch.from_numpy(X.astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ===========================================================================
# Helpers (time-series-aware)
# ===========================================================================

def windows_and_labels_ts(
    traces: list[TimeSeriesTrace], binary: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-trace windows into a single (N, F, S) tensor + labels + trace ids."""
    X_list, y_list, tid_list = [], [], []
    for t in traces:
        X_list.append(t.windows)
        y = 1 if t.is_attack else 0
        y_list.append(np.full(t.n_windows, y, dtype=np.int64))
        tid_list.append(np.array([t.trace_id] * t.n_windows, dtype=object))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    tids = np.concatenate(tid_list, axis=0)
    return X, y, tids


def aggregate_per_trace_ts(
    traces: list[TimeSeriesTrace],
    probs_win: np.ndarray,
    tids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace attack-fraction + true binary labels (same convention as hmd_vanilla)."""
    by_tid: dict[str, list[float]] = {}
    for p, t in zip(probs_win, tids):
        by_tid.setdefault(str(t), []).append(float(p))

    attack_frac = []
    y_trace = []
    for t in traces:
        if t.trace_id not in by_tid:
            continue
        ps = np.array(by_tid[t.trace_id])
        attack_frac.append(float((ps >= 0.5).mean()))
        y_trace.append(1 if t.is_attack else 0)
    return np.asarray(attack_frac), np.asarray(y_trace)


def per_attack_class_tpr_ts(
    traces: list[TimeSeriesTrace],
    attack_frac: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Per-attack-class trace TPR keyed by class_name.

    Returns dict[str, float] matching hmd_vanilla.per_attack_class_tpr.
    Per-fold scalar; summarize_cv aggregates across folds into (mean, std).
    """
    by_class: dict[str, list[int]] = {}

    # aggregate_per_trace_ts iterates `traces` in order, skipping any trace
    # whose trace_id isn't in by_tid. For our dataset every trace has windows
    # so the lengths align. Iterate `traces` and `attack_frac` in lockstep.
    fi = 0
    for t in traces:
        if fi >= len(attack_frac):
            break
        if t.is_attack:
            detected = int(attack_frac[fi] >= threshold)
            by_class.setdefault(t.class_name, []).append(detected)
        fi += 1

    out: dict[str, float] = {}
    for cls in ATTACK_CLASSES:
        vals = by_class.get(cls, [])
        out[cls] = float(np.mean(vals)) if vals else float("nan")
    return out


# ===========================================================================
# Training
# ===========================================================================

def fit_cnn_hmd(
    train_traces: list[TimeSeriesTrace],
    cfg: CNNHMDConfig | None = None,
    verbose: bool = False,
) -> CNNHMDFit:
    cfg = cfg or CNNHMDConfig()

    if cfg.n_threads > 0:
        torch.set_num_threads(cfg.n_threads)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_train, y_train, _ = windows_and_labels_ts(train_traces)

    # Per-feature standardization across (window × sample) axes
    if cfg.use_scaler:
        feature_means = X_train.mean(axis=(0, 2)).astype(np.float32)
        feature_stds = X_train.std(axis=(0, 2)).astype(np.float32)
        feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
        X_train = (X_train - feature_means.reshape(1, -1, 1)) \
                  / feature_stds.reshape(1, -1, 1)
    else:
        feature_means = np.zeros(cfg.n_features, dtype=np.float32)
        feature_stds = np.ones(cfg.n_features, dtype=np.float32)

    # Build model
    model = CNN_HMD(
        n_features=cfg.n_features,
        window_samples=cfg.window_samples,
        conv_channels=cfg.conv_channels,
        conv_kernel=cfg.conv_kernel,
        pool_size=cfg.pool_size,
        fc_hidden=cfg.fc_hidden,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=0)

    history: list[dict] = []
    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            bs = X_batch.size(0)
            epoch_loss_sum += loss.item() * bs
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                epoch_correct += (preds == y_batch).sum().item()
                epoch_total += bs

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss_sum / epoch_total,
            "acc": epoch_correct / epoch_total,
        })
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            h = history[-1]
            print(f"    epoch {h['epoch']:>2d}  "
                  f"loss={h['loss']:.4f}  acc={h['acc']:.4f}")

    return CNNHMDFit(
        cfg=cfg,
        model=model,
        feature_means=feature_means,
        feature_stds=feature_stds,
        training_history=history,
    )


# ===========================================================================
# Cross-validation
# ===========================================================================

def cross_validate_cnn_hmd(
    dataset: TimeSeriesDataset,
    cfg: CNNHMDConfig | None = None,
    n_folds: int = 5,
    cv_seed: int = 0,
    trace_threshold: float = 0.5,
    verbose: bool = True,
) -> list[FoldResult]:
    cfg = cfg or CNNHMDConfig()
    results: list[FoldResult] = []

    for fold_idx, (train_traces, test_traces) in enumerate(
        stratified_kfold_traces(dataset, n_folds=n_folds, seed=cv_seed)
    ):
        t_start = time.perf_counter()

        fold_cfg = dataclasses.replace(cfg, seed=cfg.seed + fold_idx * 10_000)
        fit = fit_cnn_hmd(train_traces, cfg=fold_cfg, verbose=False)

        X_test, y_test_win, tids = windows_and_labels_ts(test_traces)
        probs_win = fit.predict_proba_windows(X_test)
        preds_win = (probs_win >= 0.5).astype(int)
        win_m = binary_metrics(y_test_win, preds_win)

        attack_frac, y_test_trace = aggregate_per_trace_ts(
            test_traces, probs_win, tids,
        )
        preds_trace = (attack_frac >= trace_threshold).astype(int)
        trace_m = binary_metrics(y_test_trace, preds_trace)
        try:
            trace_auc = float(roc_auc_score(y_test_trace, attack_frac))
        except ValueError:
            trace_auc = float("nan")

        per_atk_tpr = per_attack_class_tpr_ts(
            test_traces, attack_frac, threshold=trace_threshold,
        )

        elapsed = time.perf_counter() - t_start
        final = fit.training_history[-1]

        results.append(FoldResult(
            fold_idx=fold_idx,
            window_metrics=win_m,
            trace_metrics=trace_m,
            trace_auc=trace_auc,
            per_attack_tpr=per_atk_tpr,
            n_train_windows=int(sum(t.n_windows for t in train_traces)),
            n_test_windows=int(sum(t.n_windows for t in test_traces)),
        ))

        if verbose:
            print(f"  fold {fold_idx}: trained in {elapsed:5.1f}s  "
                  f"(final loss={final['loss']:.4f} acc={final['acc']:.4f})  "
                  f"| win TPR={win_m['tpr']:.3f} FPR={win_m['fpr']:.3f}  "
                  f"trace TPR={trace_m['tpr']:.3f} FPR={trace_m['fpr']:.3f}  "
                  f"trace AUC={trace_auc:.4f}")

    return results


# ===========================================================================
# Plotting
# ===========================================================================

def plot_cnn_hmd(
    results: list[FoldResult],
    summary: dict,
    training_histories: list[list[dict]] | None = None,
    output_path: str = "phase3_cnn.png",
) -> None:
    """Two-panel figure: training curves (left) + per-attack TPR (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ----- Left panel: training loss curves -----
    if training_histories is not None:
        for fold_idx, hist in enumerate(training_histories):
            epochs = [h["epoch"] for h in hist]
            losses = [h["loss"] for h in hist]
            ax1.plot(epochs, losses, alpha=0.7, linewidth=1.5,
                     label=f"fold {fold_idx}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss (BCE)")
        ax1.set_title("Training loss per fold")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "training_histories not provided",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Training loss per fold")

    # ----- Right panel: per-attack-class TPR -----
    x_pos = np.arange(len(ATTACK_CLASSES))
    tprs = [summary["per_attack_tpr"].get(c, (float("nan"), 0))[0]
            for c in ATTACK_CLASSES]
    stds = [summary["per_attack_tpr"].get(c, (float("nan"), 0))[1]
            if isinstance(summary["per_attack_tpr"].get(c, (float("nan"), 0))[1],
                          (int, float)) else 0
            for c in ATTACK_CLASSES]
    ax2.bar(x_pos, tprs, color="tab:orange",
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace("_", "\n", 1) for c in ATTACK_CLASSES],
                         fontsize=9)
    ax2.set_ylabel("Per-attack-class trace TPR")
    ax2.set_title("Per-attack detection (D2 — 1D CNN HMD)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Phase 3 — D2: 1D CNN HMD (Sayadi-class) — 5-fold CV on real Pi traces\n"
        "Input: (3, 100) tensor per 100ms window (raw 1ms HPC samples)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# ===========================================================================
# Driver
# ===========================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("Phase 3 — D2: 1D CNN HMD (Sayadi-class)")
    print("=" * 78)
    print()

    # 1. Load time-series data
    ds = load_real_parquet_timeseries(
        "./data/traces",
        window_samples=100,
        verbose=True,
    )
    print()
    print(ds.summary())
    print()

    # 2. Config + model size check
    cfg = CNNHMDConfig(
        n_features=6,
        window_samples=100,
        conv_channels=(16, 16),
        conv_kernel=5,
        pool_size=2,
        fc_hidden=32,
        dropout=0.2,
        use_scaler=True,
        n_epochs=30,
        batch_size=128,
        lr=1e-3,
        seed=0,
    )

    probe_model = CNN_HMD(
        n_features=cfg.n_features,
        window_samples=cfg.window_samples,
        conv_channels=cfg.conv_channels,
        conv_kernel=cfg.conv_kernel,
        pool_size=cfg.pool_size,
        fc_hidden=cfg.fc_hidden,
        dropout=cfg.dropout,
    )
    n_params = sum(p.numel() for p in probe_model.parameters())
    n_trainable = sum(p.numel() for p in probe_model.parameters() if p.requires_grad)
    print(f"Model parameter count: {n_params:,} (trainable: {n_trainable:,})")
    print(f"Default PyTorch threads: {torch.get_num_threads()}")
    print()

    # 3. CV — also collect training histories for the figure
    print("=" * 78)
    print("5-fold CV training")
    print("=" * 78)
    t_overall = time.perf_counter()
    results: list[FoldResult] = []
    histories: list[list[dict]] = []

    for fold_idx, (train_traces, test_traces) in enumerate(
        stratified_kfold_traces(ds, n_folds=5, seed=0)
    ):
        t_start = time.perf_counter()
        fold_cfg = dataclasses.replace(cfg, seed=cfg.seed + fold_idx * 10_000)
        fit = fit_cnn_hmd(train_traces, cfg=fold_cfg, verbose=False)
        histories.append(fit.training_history)

        X_test, y_test_win, tids = windows_and_labels_ts(test_traces)
        probs_win = fit.predict_proba_windows(X_test)
        preds_win = (probs_win >= 0.5).astype(int)
        win_m = binary_metrics(y_test_win, preds_win)

        attack_frac, y_test_trace = aggregate_per_trace_ts(
            test_traces, probs_win, tids,
        )
        preds_trace = (attack_frac >= 0.5).astype(int)
        trace_m = binary_metrics(y_test_trace, preds_trace)
        try:
            trace_auc = float(roc_auc_score(y_test_trace, attack_frac))
        except ValueError:
            trace_auc = float("nan")
        per_atk_tpr = per_attack_class_tpr_ts(test_traces, attack_frac)

        elapsed = time.perf_counter() - t_start
        final = fit.training_history[-1]
        print(f"  fold {fold_idx}: trained in {elapsed:6.1f}s  "
              f"(final loss={final['loss']:.4f} acc={final['acc']:.4f})  "
              f"| win TPR={win_m['tpr']:.3f} FPR={win_m['fpr']:.3f}  "
              f"trace AUC={trace_auc:.4f}")

        results.append(FoldResult(
            fold_idx=fold_idx,
            window_metrics=win_m,
            trace_metrics=trace_m,
            trace_auc=trace_auc,
            per_attack_tpr=per_atk_tpr,
            n_train_windows=int(sum(t.n_windows for t in train_traces)),
            n_test_windows=int(sum(t.n_windows for t in test_traces)),
        ))

    total_elapsed = time.perf_counter() - t_overall
    print(f"\n  CV total wall time: {total_elapsed/60:.1f} min")

    # 4. Summary
    summary = summarize_cv(results)
    print_summary(summary, title="D2: 1D CNN HMD (Sayadi-class)")

    # 5. Figure
    plot_cnn_hmd(results, summary, training_histories=histories,
                 output_path="phase3_cnn.png")
