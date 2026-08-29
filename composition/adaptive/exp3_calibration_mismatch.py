"""
exp3_calibration_mismatch.py
============================
Second robustness axis: CALIBRATION MISMATCH (benign-noise drift).

Motivation
----------
Every scheme sets its FPR=1% threshold by calibrating on benign traffic assumed
to have per-edge noise sigma_cal = 0.15. In deployment, real benign noise drifts
(hotter nodes, noisier links, workload changes). The cross-check showed T_median
needs its threshold jammed 3.11 std into a compressed tail (benign_mu=0.044) to
hit 1% FPR, vs 2.56 for T_meanvar_z. A threshold jammed into a tail is brittle:
a small upward drift in real benign noise should blow up its false-positive rate.

This script MEASURES that, honestly and both-directionally:
  * calibrate each scheme's threshold at sigma_cal = 0.15 (benign)
  * then evaluate with TRUE benign noise sigma_real in {0.15, 0.18, ..., 0.30}
  * report, per scheme, at each drift level:
      - realized_benign_fpr  (target is 0.01; blow-up = brittle calibration)
      - attacker_tpr          (so we catch a scheme that "holds FPR" only by
                               also losing all detection — full operating point)
      - fpr_blowup_ratio      (realized_fpr / 0.01)

Fairness guards (so this cannot secretly favor T_meanvar_z):
  - identical drift levels, seeds, network topology for every scheme
  - identical calibration procedure (quantile(benign,0.99) at sigma_cal)
  - the attacker is each scheme's OWN optimal attacker from the cross-check,
    so detection numbers are comparable to that experiment
  - we report BOTH fpr and tpr; a scheme is only "robust to drift" if it keeps
    fpr near target AND retains detection.

A scheme is calibration-robust if realized FPR stays near 1% as sigma_real rises.
A scheme is calibration-brittle if FPR explodes (many false alarms) under drift.

Output:
  exp3_calibration_mismatch.csv
  console summary: FPR blow-up per scheme across drift.

Run: python exp3_calibration_mismatch.py [--seeds 100] [--smoke]
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_sim import (  # noqa: E402
    s_a as get_s_a, s_b as get_s_b,
    gen_er_digraph, assign_roles,
)
from trust_schemes import (  # noqa: E402
    T1_mean, T2_SL, T3_Beta, T4_EigenTrust, T5_TFLDT, T_max, T_median,
    make_T_meanvar_z,
)
from attacker_v1_optimal import best_response, emit_signal  # noqa: E402
from attacker_tmedian_optimal import signal_tmedian_optimal  # noqa: E402
from scheme_cost import cost_relative_to_T1  # noqa: E402

N = 20
P_EDGE = 0.30
RHO = 0.30
SIGMA_CAL = 0.15                      # calibration assumption
SIGMA_DRIFT = [0.15, 0.18, 0.21, 0.24, 0.27, 0.30]  # true benign noise levels


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def scheme_of(name, sb_cal, sigma_cal):
    return {
        "T1_mean": T1_mean, "T2_SL": T2_SL, "T3_Beta": T3_Beta,
        "T5_TFLDT": T5_TFLDT, "T_max": T_max, "T_median": T_median,
        "T4_EigenTrust": T4_EigenTrust,
        "T_meanvar_z": make_T_meanvar_z(sb_cal, sigma_cal),
    }[name]


def calibrate(scheme_fn, sb_cal, sigma_cal, n_cal=200):
    """Threshold at FPR=1% calibrated on benign traffic with sigma_cal noise.
    ALL edges emit benign (calibration assumes an all-honest network)."""
    benign = []
    for seed in range(n_cal):
        rng = np.random.default_rng(70_000 + seed)
        edges = gen_er_digraph(N, P_EDGE, rng)
        roles = assign_roles(N, RHO, rng)
        sig = {}
        for (i, j) in edges:
            sig[(i, j)] = clip01(sb_cal + rng.normal(0, sigma_cal))
        sc = scheme_fn(edges, sig, N)
        # every node is honest here, so all scores are benign calibration samples
        for i in range(N):
            if not np.isnan(sc[i]):
                benign.append(sc[i])
    return float(np.quantile(benign, 0.99))


def build_attack_signal(scheme_name, edges, roles, rng, sa, sb, sigma_real,
                        q, k_typ):
    """Each scheme's own optimal attacker, but with benign nodes emitting at the
    DRIFTED sigma_real (attackers still use their tuned strategy). Returns sig."""
    family = {"T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT"}
    sig = {}
    out = defaultdict(list)
    for (i, j) in edges:
        if roles[i] == 0:
            sig[(i, j)] = clip01(sb + rng.normal(0, sigma_real))  # DRIFTED benign
        else:
            out[i].append((i, j))

    for i, elist in out.items():
        E = len(elist)
        if scheme_name == "T_median":
            # largest-minority breakdown attack
            m = min(int(np.floor(0.49 * E)), int(np.floor((E - 1) / 2)))
            order = rng.permutation(E); atk = set(order[:m].tolist())
            for kk, e in enumerate(elist):
                base = sa if kk in atk else sb
                sig[e] = clip01(base + rng.normal(0, sigma_real))
        elif scheme_name == "T_max":
            # keep every edge low (cap near benign)
            for e in elist:
                sig[e] = clip01(sb + rng.normal(0, sigma_real))
        elif scheme_name == "T_meanvar_z":
            br = best_response(q, sa, sb, SIGMA_CAL, k_typ)
            bases = emit_signal(q, br["w_m"], br["m"], sa, E, rng)
            for e, b in zip(elist, bases):
                sig[e] = clip01(b + rng.normal(0, sigma_real))
        else:  # first-moment family + eigentrust: dilution
            m = int(round(q * E))
            order = rng.permutation(E); atk = set(order[:m].tolist())
            for kk, e in enumerate(elist):
                base = sa if kk in atk else sb
                sig[e] = clip01(base + rng.normal(0, sigma_real))
    return sig


def run(out_csv, n_seeds=100, smoke=False):
    defenses = ["D5_Stochastic"] if smoke else \
        ["D1_RF", "D3_RHMD", "D5_Stochastic"]  # representative spread
    schemes = ["T1_mean", "T_max", "T_median", "T_meanvar_z"] if smoke else \
        ["T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT", "T_max", "T_median",
         "T4_EigenTrust", "T_meanvar_z"]
    drifts = [0.15, 0.22, 0.30] if smoke else SIGMA_DRIFT
    if smoke:
        n_seeds = 25

    k_typ = max(2, int(round(P_EDGE * (N - 1))))
    q = 0.5  # meaningful attack yield for the detection side
    rows = []
    t0 = time.time()
    total = len(defenses) * len(schemes) * len(drifts)
    done = 0
    _last = -1

    for defense in defenses:
        sa, sb = get_s_a(defense), get_s_b(defense)
        sb_cal = sb
        for scheme_name in schemes:
            fn = scheme_of(scheme_name, sb_cal, SIGMA_CAL)
            # calibrate threshold ONCE at sigma_cal
            thr = calibrate(fn, sb_cal, SIGMA_CAL,
                            n_cal=(40 if smoke else 200))
            cvt = cost_relative_to_T1(scheme_name, k_typ, N=N)
            for sigma_real in drifts:
                fprs, tprs = [], []
                for seed in range(n_seeds):
                    rng = np.random.default_rng(seed)
                    edges = gen_er_digraph(N, P_EDGE, rng)
                    roles = assign_roles(N, RHO, rng)
                    sig = build_attack_signal(scheme_name, edges, roles, rng,
                                              sa, sb, sigma_real, q, k_typ)
                    sc = fn(edges, sig, N)
                    roles_arr = np.array(roles)
                    valid = ~np.isnan(sc)
                    benign = sc[valid & (roles_arr == 0)]
                    attack = sc[valid & (roles_arr == 1)]
                    if len(benign):
                        fprs.append(np.mean(benign > thr))  # realized FPR at drift
                    if len(attack):
                        tprs.append(np.mean(attack > thr))
                realized_fpr = float(np.mean(fprs)) if fprs else float("nan")
                tpr = float(np.mean(tprs)) if tprs else float("nan")
                rows.append({
                    "scheme": scheme_name, "defense": defense,
                    "sigma_cal": SIGMA_CAL, "sigma_real": sigma_real,
                    "drift": round(sigma_real - SIGMA_CAL, 3),
                    "realized_benign_fpr": round(realized_fpr, 5),
                    "fpr_blowup_ratio": round(realized_fpr / 0.01, 2)
                        if realized_fpr == realized_fpr else float("nan"),
                    "attacker_tpr": round(tpr, 5),
                    "cost_vs_T1": round(cvt, 2),
                })
                done += 1
                pct = int(100 * done / total)
                if pct != _last and pct % 10 == 0:
                    print(f"  ... {pct}% ({done}/{total})  t={time.time()-t0:.0f}s")
                    _last = pct

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    _summary(rows, schemes, drifts)
    print(f"\n  -> {out_csv}  ({len(rows)} cells)")


def _summary(rows, schemes, drifts):
    print("\n" + "=" * 92)
    print("  CALIBRATION-MISMATCH — realized benign FPR as true noise drifts above")
    print("  the calibration assumption (sigma_cal=0.15). Target FPR = 0.010.")
    print("  A brittle scheme's FPR explodes; a robust scheme's stays near target.")
    print("=" * 92)
    hdr = f"  {'Scheme':<15}" + "".join(f"s={d:>4}".rjust(9) for d in drifts) + f"{'blowup@max':>12}"
    print(hdr)
    print("  " + "-" * 88)
    for sch in schemes:
        line = f"  {sch:<15}"
        fpr_at_max = None
        for d in drifts:
            cells = [float(r["realized_benign_fpr"]) for r in rows
                     if r["scheme"] == sch and r["sigma_real"] == d]
            v = np.mean(cells) if cells else float("nan")
            line += f"{v:>9.3f}"
            fpr_at_max = v
        blowup = (fpr_at_max / 0.01) if fpr_at_max == fpr_at_max else float("nan")
        line += f"{blowup:>11.1f}x"
        print(line)
    print("  " + "-" * 88)
    print("  Values are realized benign FPR (target 0.010). blowup@max = FPR/target")
    print("  at the largest drift. Higher blowup = more brittle calibration.")
    print("  (attacker_tpr at each drift is in the CSV — a scheme must ALSO retain")
    print("   detection, not just hold FPR by collapsing scores.)")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if args.smoke:
        print("=== SMOKE MODE ===")
    print("EXP3 CALIBRATION-MISMATCH: benign-noise drift robustness")
    run(os.path.join(here, "exp3_calibration_mismatch.csv"),
        n_seeds=args.seeds, smoke=args.smoke)
    print("\nDone.")


if __name__ == "__main__":
    main()
