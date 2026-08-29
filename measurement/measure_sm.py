"""
measure_sm.py — Per-Edge Mimicry Signal (s_m) Measurement
==========================================================

For each defense D1-D6, measure the distribution of per-window attack
probabilities on REAL attack workloads. The AP4/AP5 mimicry attacker
preferentially transmits the lowest-signal attack windows, so:

    s_m(p) = mean attack-window probability among windows BELOW the p-th
             percentile of the attack-window probability distribution.

This is THE parameter that determines AP5 severity (PHASE4_SIGNAL_VALUES.md §4):
    AP5 attacker budget  m_max/E = (T_det - s_b) / (s_m - s_b)
As s_m -> s_b, the budget -> infinity (attacker hides on all edges).

Outputs per defense:
  - s_a : mean attack-window signal (AP1 baseline)
  - s_b : mean benign-window signal
  - s_m(p) for p in {5, 10, 25, 50}
  - per-attack-class mean signal (identifies the most-mimicable workload)

Single-fold (fold 0) measurement for speed; out-of-sample (test split).
D6 runs the full pipeline (~14 min) and can be disabled with RUN_D6 = False.
D2 uses the time-series loader; all others use the 3-derived-feature loader.
"""

from __future__ import annotations

import json
import time

import numpy as np

from dataset import ATTACK_CLASSES, stratified_kfold_traces
from hmd_vanilla import windows_and_labels


# ----- toggles -----
RUN_D2 = True    # CNN — time-series, ~5 min single-fold train
RUN_D6 = True    # full DRL pipeline — ~14 min single-fold train

PERCENTILES = [5, 10, 25, 50]
DATA_ROOT = "./data/traces"


# ===========================================================================
# Signal collection + summary
# ===========================================================================

def collect_signals(predict_fn, test_traces, windows_fn):
    """Iterate test traces, predict per-window probs, tag by class.

    predict_fn:  X -> probs (n_windows,)
    windows_fn:  [trace] -> (X, y, tids)
    Returns list of (prob, is_attack, class_name).
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


def summarize_signals(records: list, name: str) -> dict:
    attack = np.array([p for p, a, _ in records if a], dtype=float)
    benign = np.array([p for p, a, _ in records if not a], dtype=float)

    s_a = float(attack.mean()) if len(attack) else float("nan")
    s_b = float(benign.mean()) if len(benign) else float("nan")

    sm = {}
    for p in PERCENTILES:
        cutoff = np.percentile(attack, p)
        below = attack[attack <= cutoff]
        sm[p] = float(below.mean()) if len(below) else float("nan")

    per_class = {}
    for cls in ATTACK_CLASSES:
        cp = np.array([p for p, _, c in records if c == cls], dtype=float)
        if len(cp):
            per_class[cls] = float(cp.mean())

    return {
        "name": name,
        "s_a": s_a,
        "s_b": s_b,
        "s_m": sm,
        "per_class": per_class,
        "n_attack_windows": int(len(attack)),
        "n_benign_windows": int(len(benign)),
    }


def print_summary(summ: dict) -> None:
    print(f"\n--- {summ['name']} ---")
    print(f"  s_a (mean attack signal) : {summ['s_a']:.4f}  "
          f"(n={summ['n_attack_windows']})")
    print(f"  s_b (mean benign signal) : {summ['s_b']:.4f}  "
          f"(n={summ['n_benign_windows']})")
    print(f"  s_m by percentile cutoff (mean of attack windows below pct):")
    for p in PERCENTILES:
        print(f"    s_m({p:>2d}%) = {summ['s_m'][p]:.4f}")
    # most-mimicable class = lowest mean signal
    if summ["per_class"]:
        ranked = sorted(summ["per_class"].items(), key=lambda kv: kv[1])
        print(f"  per-attack-class mean signal (ascending = most mimicable first):")
        for cls, v in ranked:
            print(f"    {cls:<22s} {v:.4f}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    from dataset_parquet import load_real_parquet

    print("=" * 78)
    print("s_m MEASUREMENT — per-edge mimicry signal for all defenses")
    print("=" * 78)

    # --- derived-feature dataset, fold 0 ---
    ds = load_real_parquet(DATA_ROOT, window_aggregation_factor=100, verbose=True)
    train_traces, test_traces = list(stratified_kfold_traces(ds, n_folds=5, seed=0))[0]
    print(f"\nDerived-feature fold 0: {len(train_traces)} train, {len(test_traces)} test")

    results = {}

    import os
    SM_JSON = "sm_results.json"
    if os.path.exists(SM_JSON):
        with open(SM_JSON) as f:
            results = json.load(f)
        print(f"\nResuming: loaded {len(results)} existing result(s): "
              f"{', '.join(results.keys())}")

    def save_results():
        with open(SM_JSON, "w") as f:
            json.dump(results, f, indent=2)

    def have(key):
        return key in results

    def sm_get(s, p):
        """s_m value tolerant of int (fresh) or str (JSON-loaded) keys."""
        sm = s["s_m"]
        if p in sm:
            return sm[p]
        return sm[str(p)]

    # ---- D1: Vanilla RF ----
    if not have("D1_RF"):
        print("\n[D1] training vanilla RF ...")
        from hmd_vanilla import VanillaHMDConfig, fit_vanilla_hmd
        t0 = time.perf_counter()
        d1 = fit_vanilla_hmd(train_traces, cfg=VanillaHMDConfig(classifier="rf", seed=0))
        rec = collect_signals(d1.predict_proba_windows, test_traces, windows_and_labels)
        results["D1_RF"] = summarize_signals(rec, "D1 Vanilla RF")
        save_results()
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print_summary(results["D1_RF"])

    # ---- D3: RHMD ----
    if not have("D3_RHMD"):
        print("\n[D3] training RHMD ...")
        from hmd_rhmd import RHMDConfig, fit_rhmd
        t0 = time.perf_counter()
        d3 = fit_rhmd(train_traces, cfg=RHMDConfig(n_classifiers=5,
                                                   selection_policy="uniform",
                                                   inference_seed=7000))
        rec = collect_signals(d3.predict_proba_windows, test_traces, windows_and_labels)
        results["D3_RHMD"] = summarize_signals(rec, "D3 RHMD n=5")
        save_results()
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print_summary(results["D3_RHMD"])

    # ---- D4: MTD ----
    if not have("D4_MTD"):
        print("\n[D4] training MTD-HMD ...")
        from hmd_mtd import MTDHMDConfig, fit_mtd_hmd
        t0 = time.perf_counter()
        d4 = fit_mtd_hmd(train_traces, cfg=MTDHMDConfig())
        rec = collect_signals(d4.predict_proba_windows, test_traces, windows_and_labels)
        results["D4_MTD"] = summarize_signals(rec, "D4 MTD-HMD")
        save_results()
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print_summary(results["D4_MTD"])

    # ---- D5: Stochastic-HMD (sigma=0.10) ----
    if not have("D5_Stochastic"):
        print("\n[D5] training Stochastic-HMD (sigma=0.10) ...")
        from hmd_stochastic import StochasticHMDConfig, fit_stochastic_hmd
        t0 = time.perf_counter()
        d5 = fit_stochastic_hmd(train_traces,
                                cfg=StochasticHMDConfig(noise_sigma=0.10,
                                                        n_noise_samples=10, seed=0))
        rec = collect_signals(d5.predict_proba_windows, test_traces, windows_and_labels)
        results["D5_Stochastic"] = summarize_signals(rec, "D5 Stochastic sigma=0.10")
        save_results()
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print_summary(results["D5_Stochastic"])

    # ---- D6: full DRL pipeline ----
    if RUN_D6 and not have("D6_DRL"):
        print("\n[D6] training full DRL pipeline (base classifiers + A2C + UCB) ...")
        from d6_adversarial import AdversarialMLPConfig, fit_adversarial_mlp
        from d6_a2c import A2CConfig, train_a2c
        from d6_ucb import UCBConfig, fit_ucb
        t0 = time.perf_counter()
        base = [fit_adversarial_mlp(train_traces,
                                    cfg=AdversarialMLPConfig(fgsm_epsilon=e, seed=0))
                for e in (0.05, 0.10, 0.20)]
        a2c = train_a2c(base, train_traces, cfg=A2CConfig(seed=0), verbose=False)
        ucb = fit_ucb(a2c, base, train_traces, cfg=UCBConfig(seed=0), verbose=False)
        d6_predict = lambda X: ucb.predict_proba_windows(X, a2c, base)
        rec = collect_signals(d6_predict, test_traces, windows_and_labels)
        results["D6_DRL"] = summarize_signals(rec, "D6 DRL")
        save_results()
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print_summary(results["D6_DRL"])

    # ---- D2: 1D CNN (time-series) ----
    if RUN_D2 and not have("D2_CNN"):
        print("\n[D2] training 1D CNN (time-series) ...")
        from hmd_cnn import (
            load_real_parquet_timeseries, CNNHMDConfig, fit_cnn_hmd,
            windows_and_labels_ts,
        )
        t0 = time.perf_counter()
        ds_ts = load_real_parquet_timeseries(DATA_ROOT, window_samples=100, verbose=False)
        tr_ts, te_ts = list(stratified_kfold_traces(ds_ts, n_folds=5, seed=0))[0]
        d2 = fit_cnn_hmd(tr_ts, cfg=CNNHMDConfig())
        rec = collect_signals(d2.predict_proba_windows, te_ts, windows_and_labels_ts)
        results["D2_CNN"] = summarize_signals(rec, "D2 1D CNN")
        save_results()
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print_summary(results["D2_CNN"])

    # ===================================================================
    # Consolidated table + Theorem-1 AP5 budget
    # ===================================================================
    print("\n" + "=" * 90)
    print("CONSOLIDATED s_m TABLE")
    print("=" * 90)
    hdr = f"{'Defense':<22s} {'s_a':>7s} {'s_b':>7s}"
    for p in PERCENTILES:
        hdr += f" {'s_m'+str(p):>8s}"
    print(hdr)
    print("-" * 90)
    order = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
    for key in order:
        if key not in results:
            continue
        s = results[key]
        row = f"{s['name']:<22s} {s['s_a']:>7.4f} {s['s_b']:>7.4f}"
        for p in PERCENTILES:
            row += f" {sm_get(s, p):>8.4f}"
        print(row)

    # AP5 attacker budget at midpoint threshold T_det = (s_a + s_b)/2, using s_m(10%)
    print("\n" + "=" * 90)
    print("AP5 ATTACKER BUDGET  m_max/E = (T_det - s_b)/(s_m - s_b),  "
          "T_det = midpoint, s_m = s_m(10%)")
    print("(compare to AP1 baseline 0.5 at midpoint; higher = more edges attackable)")
    print("=" * 90)
    for key in order:
        if key not in results:
            continue
        s = results[key]
        s_a, s_b = s["s_a"], s["s_b"]
        s_m10 = sm_get(s, 10)
        T_det = (s_a + s_b) / 2.0
        denom = (s_m10 - s_b)
        if abs(denom) < 1e-9:
            budget = float("inf")
        else:
            budget = (T_det - s_b) / denom
        budget_str = "inf (mimics benign)" if budget == float("inf") else f"{budget:.3f}"
        # clamp interpretation: if s_m < T_det, attacker can attack ALL edges
        note = ""
        if s_m10 < T_det:
            note = "  <- s_m below threshold: ALL edges attackable undetected"
        print(f"  {s['name']:<22s} T_det={T_det:.3f}  s_m(10%)={s_m10:.4f}  "
              f"m_max/E={budget_str}{note}")

    # Save JSON for the design docs
    with open("sm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: sm_results.json")
