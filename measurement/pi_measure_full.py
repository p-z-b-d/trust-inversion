"""
pi_measure_full.py — Extended s_m / σ measurement on real Pi traces.

Improvements over measure_sm.py:
  1. Runs ALL 5 folds per defense (not just fold 0) -> CV error bars on
     s_a, s_b, s_m.
  2. Dumps raw per-window probability arrays for each (defense, fold) ->
     enables per-defense σ measurement and any future raw-distribution
     analysis off the Pi.
  3. Per-(defense, fold) checkpointing -> safe to interrupt and resume.

Outputs (relative to working directory):
  raw_arrays/{defense}_fold{k}.npz   per-fold raw arrays
                                     keys: probs (float32), is_attack (bool),
                                     class_name (U24), n_windows (int scalar)
  sm_results_cv.json                 per-(defense, fold) summaries +
                                     defense-level CV aggregates
  pi_measure_full.log                run log (when piped from shell wrapper)

Expected runtime on a Pi 4B (5 folds total):
  D1 RF:           ~10s    (cheap)
  D3 RHMD n=5:     ~30s
  D4 MTD:          ~20s
  D5 Stochastic:   ~20s
  D2 CNN:          ~25 min (5 folds x ~5 min)
  D6 DRL pipeline: ~70 min (5 folds x ~14 min)
  Total budget:    ~100 min, plus dataset load + archive.

Usage:
  source ~/research_venv/bin/activate
  cd ~/path/to/measurement/pipeline
  python3 pi_measure_full.py
  # or, unattended:
  bash pi_overnight_run.sh
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable, List, Tuple

import numpy as np

from dataset import ATTACK_CLASSES, stratified_kfold_traces
from hmd_vanilla import windows_and_labels


# ============================================================================
# Configuration
# ============================================================================

N_FOLDS = 5
CV_SEED = 0
PERCENTILES = [5, 10, 25, 50]
DATA_ROOT = "./data/traces"
RAW_DIR = "raw_arrays"
RESULTS_JSON = "sm_results_cv.json"

RUN_D2 = True
RUN_D6 = True

DEFENSES_DERIVED = ["D1_RF", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
DEFENSES_TIMESERIES = ["D2_CNN"]


# ============================================================================
# Utilities
# ============================================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_results() -> dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RESULTS_JSON)


def fold_key(defense: str, fold: int) -> str:
    return f"{defense}_fold{fold}"


def have_fold(results: dict, defense: str, fold: int) -> bool:
    return fold_key(defense, fold) in results


def collect_signals(predict_fn, test_traces, windows_fn) -> List[Tuple[float, bool, str]]:
    """Iterate test traces, predict per-window probs, tag by class.

    Returns list of (prob, is_attack, class_name) — same as measure_sm.py.
    """
    records = []
    for tr in test_traces:
        X, _, _ = windows_fn([tr])
        if X.shape[0] == 0:
            continue
        probs = np.asarray(predict_fn(X)).reshape(-1)
        is_atk = bool(tr.is_attack)
        cls = tr.class_name
        for p in probs:
            records.append((float(p), is_atk, cls))
    return records


def summarize(records, name: str, defense_key: str, fold: int) -> dict:
    """Per-fold summary stats. Mirrors measure_sm.py.summarize_signals
    structure for compatibility with downstream tools."""
    attack = np.array([p for p, a, _ in records if a], dtype=float)
    benign = np.array([p for p, a, _ in records if not a], dtype=float)

    s_a = float(attack.mean()) if attack.size else float("nan")
    s_b = float(benign.mean()) if benign.size else float("nan")

    sm = {}
    for p in PERCENTILES:
        if attack.size:
            cutoff = np.percentile(attack, p)
            below = attack[attack <= cutoff]
            sm[p] = float(below.mean()) if below.size else float("nan")
        else:
            sm[p] = float("nan")

    per_class = {}
    for cls in ATTACK_CLASSES:
        cp = np.array([p for p, _, c in records if c == cls], dtype=float)
        if cp.size:
            per_class[cls] = float(cp.mean())

    # Pre-computed σ statistics — saves re-loading raw arrays for the
    # common cases. Full distributions are saved to NPZ for any other use.
    sigma_pw = {
        "all": float(np.concatenate([attack, benign]).std(ddof=1))
               if (attack.size + benign.size) > 1 else float("nan"),
        "benign": float(benign.std(ddof=1)) if benign.size > 1 else float("nan"),
        "attack": float(attack.std(ddof=1)) if attack.size > 1 else float("nan"),
    }

    return {
        "name": name,
        "defense": defense_key,
        "fold": fold,
        "s_a": s_a,
        "s_b": s_b,
        "s_m": sm,
        "per_class": per_class,
        "sigma_per_window": sigma_pw,
        "n_attack_windows": int(attack.size),
        "n_benign_windows": int(benign.size),
    }


def dump_raw_npz(records, path: str) -> None:
    """Save raw per-window arrays so any future analysis can re-derive
    statistics without re-running the Pi pipeline."""
    if not records:
        return
    probs = np.asarray([r[0] for r in records], dtype=np.float32)
    is_atk = np.asarray([r[1] for r in records], dtype=bool)
    classes = np.asarray([r[2] for r in records], dtype="U24")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, probs=probs, is_attack=is_atk, class_name=classes)


# ============================================================================
# Per-defense fitters — mirror measure_sm.py but parameterized by fold
# ============================================================================

def fit_and_predict_d1(train_traces):
    from hmd_vanilla import VanillaHMDConfig, fit_vanilla_hmd
    d = fit_vanilla_hmd(train_traces, cfg=VanillaHMDConfig(classifier="rf", seed=0))
    return d.predict_proba_windows


def fit_and_predict_d3(train_traces):
    from hmd_rhmd import RHMDConfig, fit_rhmd
    d = fit_rhmd(train_traces, cfg=RHMDConfig(
        n_classifiers=5, selection_policy="uniform", inference_seed=7000))
    return d.predict_proba_windows


def fit_and_predict_d4(train_traces):
    from hmd_mtd import MTDHMDConfig, fit_mtd_hmd
    d = fit_mtd_hmd(train_traces, cfg=MTDHMDConfig())
    return d.predict_proba_windows


def fit_and_predict_d5(train_traces):
    from hmd_stochastic import StochasticHMDConfig, fit_stochastic_hmd
    d = fit_stochastic_hmd(train_traces, cfg=StochasticHMDConfig(
        noise_sigma=0.10, n_noise_samples=10, seed=0))
    return d.predict_proba_windows


def fit_and_predict_d6(train_traces):
    from d6_adversarial import AdversarialMLPConfig, fit_adversarial_mlp
    from d6_a2c import A2CConfig, train_a2c
    from d6_ucb import UCBConfig, fit_ucb
    base = [fit_adversarial_mlp(train_traces,
                                cfg=AdversarialMLPConfig(fgsm_epsilon=e, seed=0))
            for e in (0.05, 0.10, 0.20)]
    a2c = train_a2c(base, train_traces, cfg=A2CConfig(seed=0), verbose=False)
    ucb = fit_ucb(a2c, base, train_traces, cfg=UCBConfig(seed=0), verbose=False)
    return lambda X: ucb.predict_proba_windows(X, a2c, base)


DERIVED_FITTERS = {
    "D1_RF":         ("D1 Vanilla RF",          fit_and_predict_d1),
    "D3_RHMD":       ("D3 RHMD n=5",            fit_and_predict_d3),
    "D4_MTD":        ("D4 MTD-HMD",             fit_and_predict_d4),
    "D5_Stochastic": ("D5 Stochastic sigma=0.10", fit_and_predict_d5),
    "D6_DRL":        ("D6 DRL",                 fit_and_predict_d6),
}


def run_derived_defenses(results: dict) -> None:
    """All defenses using the derived-feature loader."""
    from dataset_parquet import load_real_parquet
    log("Loading derived-feature dataset ...")
    ds = load_real_parquet(DATA_ROOT, window_aggregation_factor=100, verbose=False)
    fold_iter = list(stratified_kfold_traces(ds, n_folds=N_FOLDS, seed=CV_SEED))
    log(f"Derived dataset folds prepared: {N_FOLDS} folds.")

    for defense_key, (label, fitter) in DERIVED_FITTERS.items():
        if defense_key == "D6_DRL" and not RUN_D6:
            log(f"Skipping {defense_key} (RUN_D6=False)")
            continue
        for k in range(N_FOLDS):
            if have_fold(results, defense_key, k):
                log(f"  skip {defense_key} fold {k} (cached)")
                continue
            t0 = time.perf_counter()
            log(f"  fitting {defense_key} fold {k} ...")
            train_traces, test_traces = fold_iter[k]
            predict_fn = fitter(train_traces)
            log(f"    fitted in {time.perf_counter()-t0:.1f}s; collecting signals ...")
            t1 = time.perf_counter()
            rec = collect_signals(predict_fn, test_traces, windows_and_labels)
            dump_raw_npz(rec, os.path.join(RAW_DIR, f"{defense_key}_fold{k}.npz"))
            summ = summarize(rec, label, defense_key, k)
            results[fold_key(defense_key, k)] = summ
            save_results(results)
            log(f"    done in {time.perf_counter()-t1:.1f}s  "
                f"s_a={summ['s_a']:.4f}  s_b={summ['s_b']:.4f}  "
                f"σ_pw[benign]={summ['sigma_per_window']['benign']:.4f}")


def run_timeseries_defenses(results: dict) -> None:
    """Time-series defenses (D2 CNN)."""
    if not RUN_D2:
        log("Skipping D2 (RUN_D2=False)")
        return
    from hmd_cnn import (
        load_real_parquet_timeseries, CNNHMDConfig, fit_cnn_hmd,
        windows_and_labels_ts,
    )
    log("Loading time-series dataset for D2 ...")
    ds_ts = load_real_parquet_timeseries(DATA_ROOT, window_samples=100, verbose=False)
    fold_iter = list(stratified_kfold_traces(ds_ts, n_folds=N_FOLDS, seed=CV_SEED))
    log(f"Time-series dataset folds prepared: {N_FOLDS} folds.")

    for k in range(N_FOLDS):
        if have_fold(results, "D2_CNN", k):
            log(f"  skip D2_CNN fold {k} (cached)")
            continue
        t0 = time.perf_counter()
        log(f"  fitting D2_CNN fold {k} ...")
        train_traces, test_traces = fold_iter[k]
        d2 = fit_cnn_hmd(train_traces, cfg=CNNHMDConfig())
        log(f"    fitted in {time.perf_counter()-t0:.1f}s; collecting signals ...")
        t1 = time.perf_counter()
        rec = collect_signals(d2.predict_proba_windows, test_traces, windows_and_labels_ts)
        dump_raw_npz(rec, os.path.join(RAW_DIR, f"D2_CNN_fold{k}.npz"))
        summ = summarize(rec, "D2 1D CNN", "D2_CNN", k)
        results[fold_key("D2_CNN", k)] = summ
        save_results(results)
        log(f"    done in {time.perf_counter()-t1:.1f}s  "
            f"s_a={summ['s_a']:.4f}  s_b={summ['s_b']:.4f}  "
            f"σ_pw[benign]={summ['sigma_per_window']['benign']:.4f}")


# ============================================================================
# CV aggregation -- mean ± SEM × 1.96 across the 5 folds
# ============================================================================

def aggregate_cv(results: dict) -> dict:
    """Build per-defense aggregates with CV error bars from the per-fold entries.

    Stored under keys like 'D3_RHMD_cv' alongside per-fold entries."""
    by_defense = {}
    for key, summ in results.items():
        if not key.endswith(tuple(f"_fold{k}" for k in range(N_FOLDS))):
            continue
        d = summ["defense"]
        by_defense.setdefault(d, []).append(summ)

    aggregates = {}
    for d, folds in by_defense.items():
        if len(folds) < 2:
            continue  # need >=2 folds for SEM
        def gather(getter):
            vals = np.array([getter(f) for f in folds], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size < 2:
                return {"mean": float("nan"), "ci95": float("nan"), "n": int(vals.size)}
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
            return {"mean": mean, "ci95": 1.96 * sem, "n": int(vals.size)}

        agg = {
            "name": folds[0]["name"],
            "defense": d,
            "n_folds_used": len(folds),
            "s_a": gather(lambda f: f["s_a"]),
            "s_b": gather(lambda f: f["s_b"]),
            "s_m": {p: gather(lambda f, p=p: f["s_m"].get(p, f["s_m"].get(str(p))))
                    for p in PERCENTILES},
            "sigma_per_window": {
                "all":    gather(lambda f: f["sigma_per_window"]["all"]),
                "benign": gather(lambda f: f["sigma_per_window"]["benign"]),
                "attack": gather(lambda f: f["sigma_per_window"]["attack"]),
            },
            "per_class": {
                cls: gather(lambda f, c=cls: f["per_class"].get(c, float("nan")))
                for cls in ATTACK_CLASSES
            },
        }
        aggregates[f"{d}_cv"] = agg
    return aggregates


def print_summary(results: dict) -> None:
    aggregates = {k: v for k, v in results.items() if k.endswith("_cv")}
    if not aggregates:
        log("No CV aggregates yet (need >=2 folds per defense).")
        return

    print()
    print("=" * 100)
    print("CV AGGREGATES (mean ± 95% CI across folds)")
    print("=" * 100)
    hdr = (f"{'Defense':<22s} {'n':>2s}  "
           f"{'s_a':>16s}  {'s_b':>16s}  "
           f"{'σ_pw[ben]':>14s}  {'σ_pw[atk]':>14s}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(aggregates.keys()):
        a = aggregates[key]
        sa, sb = a["s_a"], a["s_b"]
        spb, spa = a["sigma_per_window"]["benign"], a["sigma_per_window"]["attack"]
        def fmt(stat):
            return f"{stat['mean']:.4f} ± {stat['ci95']:.4f}"
        print(f"{a['name']:<22s} {a['n_folds_used']:>2d}  "
              f"{fmt(sa):>16s}  {fmt(sb):>16s}  "
              f"{fmt(spb):>14s}  {fmt(spa):>14s}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    log("=" * 70)
    log("pi_measure_full.py — extended s_m / σ measurement")
    log("=" * 70)
    log(f"N_FOLDS={N_FOLDS}  CV_SEED={CV_SEED}  PERCENTILES={PERCENTILES}")
    log(f"RAW_DIR={RAW_DIR}  RESULTS_JSON={RESULTS_JSON}")
    log(f"RUN_D2={RUN_D2}  RUN_D6={RUN_D6}")

    t0 = time.perf_counter()

    results = load_results()
    log(f"Loaded {len(results)} existing entries.")

    run_derived_defenses(results)
    run_timeseries_defenses(results)

    # Aggregate CV statistics and store them in the same JSON
    aggregates = aggregate_cv(results)
    results.update(aggregates)
    save_results(results)

    print_summary(results)

    log(f"Total runtime: {time.perf_counter()-t0:.1f}s")
    log("Done.")


if __name__ == "__main__":
    main()
