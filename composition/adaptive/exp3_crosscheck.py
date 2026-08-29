"""
exp3_crosscheck.py
==================
The FAIR comparison: each trust scheme is attacked by an adversary tuned to
THAT scheme's specific structure, not by one attacker tuned only to T_meanvar_z.

Why this exists
---------------
exp3_compare.py ran ONE attacker (V1, optimized against T_meanvar_z's
max(z_mu,z_var)) against every scheme. That is biased: it targets T_meanvar_z's
weakness while missing the OTHER schemes' weaknesses. On that biased test,
T_max looked stable -- but the attacker never tried the attack that beats a max
aggregator. This script fixes that by giving each scheme its own worst-case
attacker, so the comparison is apples-to-apples: every scheme faces the attack
built to beat it.

Scheme-specific optimal attackers
---------------------------------
  T1_mean / T2_SL / T3_Beta / T5_TFLDT  (first-moment family):
      Null the mean. Emit s_a on a q-fraction and a forged-low value on the
      rest, choosing the forge so the per-node MEAN sits at the benign mean.
      This is the dilution attack; it is the family's known weakness.

  T_max (worst-edge aggregator):
      T_max scores a node by its single highest edge. To evade, keep EVERY
      edge below the benign p99 line. The attacker's best move is to deliver
      payload on as FEW edges as possible while keeping even those under the
      max threshold -- i.e. it cannot raise any edge much. We sweep the
      attacker's per-edge cap and find the highest payload it can push while
      staying under T_max's line.

  T_median:
      Median resists a minority of corrupted edges but flips once >50% of edges
      are pulled. The tuned attacker corrupts a majority: emit the attack signal
      on >50% of edges so the median itself moves, while keeping each below the
      line as much as possible.

  T_meanvar_z:
      The V1 mean+variance optimal attacker (already built) -- its true worst case.

FPR-honesty
-----------
For every scheme we ALSO record, from an all-benign calibration:
  benign_mean, benign_std  -- where honest nodes sit
  threshold_p99            -- the FPR=1% cutoff
  thr_in_tail = (p99 - mean)/std  -- how extreme a cutoff the scheme needs
  realized_benign_fpr      -- actual benign flag rate at that cutoff (sanity)
A scheme that looks "robust" only because its benign scores are shifted up
(compressed dynamic range, e.g. T_max) is exposed here: high benign_mean and a
threshold jammed against the ceiling mean it cannot separate attack from benign,
even if the FPR-pinned TPR looks fine.

Outputs:
  exp3_crosscheck.csv   -- per (scheme, defense, q): tuned-attacker TPR + FPR-honesty
  console               -- one fair comparison table

Run:  python exp3_crosscheck.py [--seeds 100] [--smoke]
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
    s_a as get_s_a, s_b as get_s_b, midpoint,
    gen_er_digraph, assign_roles,
)
from trust_schemes import (  # noqa: E402
    T1_mean, T2_SL, T3_Beta, T4_EigenTrust, T5_TFLDT, T_max, T_median,
    make_T_meanvar_z,
)
from attacker_v1_optimal import best_response, emit_signal  # noqa: E402
from attacker_tmedian_optimal import (  # noqa: E402
    signal_tmedian_optimal, optimal_attack_fraction,
)
from scheme_cost import analytic_cost, cost_relative_to_T1  # noqa: E402

DEFENSES_ALL = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
Q_VALS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SIGMA = 0.15
N = 20
P_EDGE = 0.30
RHO = 0.30
FORGE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def scheme_of(name, sb_cal):
    return {
        "T1_mean": T1_mean, "T2_SL": T2_SL, "T3_Beta": T3_Beta,
        "T5_TFLDT": T5_TFLDT, "T_max": T_max, "T_median": T_median,
        "T4_EigenTrust": T4_EigenTrust,
        "T_meanvar_z": make_T_meanvar_z(sb_cal, SIGMA),
    }[name]


# ---------------------------------------------------------------------------
# FPR-honesty: characterize each scheme's benign distribution + threshold.
# ---------------------------------------------------------------------------
def fpr_honesty(scheme_fn, sb_cal, n_cal=200):
    benign = []
    for seed in range(n_cal):
        rng = np.random.default_rng(50_000 + seed)
        edges = gen_er_digraph(N, P_EDGE, rng)
        roles = assign_roles(N, RHO, rng)
        sig = {}
        for (i, j) in edges:
            sig[(i, j)] = clip01(sb_cal + rng.normal(0, SIGMA))
        sc = scheme_fn(edges, sig, N)
        for i in range(N):
            if roles[i] == 0 and not np.isnan(sc[i]):
                benign.append(sc[i])
    a = np.asarray(benign)
    p99 = float(np.quantile(a, 0.99))
    realized_fpr = float(np.mean(a > p99))
    return {
        "benign_mean": float(a.mean()), "benign_std": float(a.std()),
        "threshold_p99": p99,
        "thr_in_tail": float((p99 - a.mean()) / a.std()) if a.std() > 0 else float("nan"),
        "realized_benign_fpr": realized_fpr,
    }


# ---------------------------------------------------------------------------
# Scheme-specific tuned attackers. Each returns a per-edge signal dict.
# Benign nodes emit s_b + noise; attackers use the scheme-tuned strategy.
# ---------------------------------------------------------------------------
def _benign_signal(edges, roles, sb, rng):
    sig = {}
    for (i, j) in edges:
        if roles[i] == 0:
            sig[(i, j)] = clip01(sb + rng.normal(0, SIGMA))
    return sig


def attacker_firstmoment(edges, roles, rng, q, sa, sb, forge):
    """Dilution: q edges at s_a, rest at forge (null the mean)."""
    sig = _benign_signal(edges, roles, sb, rng)
    out = defaultdict(list)
    for (i, j) in edges:
        if roles[i] == 1:
            out[i].append((i, j))
    for i, elist in out.items():
        E = len(elist)
        m = int(round(q * E))
        order = rng.permutation(E)
        atk = set(order[:m].tolist())
        for kk, e in enumerate(elist):
            base = sa if kk in atk else forge
            sig[e] = clip01(base + rng.normal(0, SIGMA))
    return sig


def attacker_tmax(edges, roles, rng, q, sa, sb, cap):
    """T_max-tuned: keep EVERY attacker edge <= cap (since max flags the worst
    edge). Payload is delivered only up to `cap`, so the attacker is throttled by
    how high it dares push any single edge. q here selects how many edges carry
    the (capped) elevated value vs benign-forged."""
    sig = _benign_signal(edges, roles, sb, rng)
    out = defaultdict(list)
    for (i, j) in edges:
        if roles[i] == 1:
            out[i].append((i, j))
    for i, elist in out.items():
        E = len(elist)
        m = int(round(q * E))
        order = rng.permutation(E)
        atk = set(order[:m].tolist())
        for kk, e in enumerate(elist):
            # capped payload on attack edges, benign elsewhere; never exceed cap
            base = min(sa, cap) if kk in atk else sb
            sig[e] = clip01(base + rng.normal(0, SIGMA))
    return sig


def attacker_tmedian(edges, roles, rng, q, sa, sb, forge):
    """T_median-tuned: to move the median the attacker must corrupt a MAJORITY
    of edges. Force attack signal on max(q, 0.55) of edges."""
    q_eff = max(q, 0.55)
    return attacker_firstmoment(edges, roles, rng, q_eff, sa, sb, forge)


def attacker_v1_meanvar(edges, roles, rng, q, sa, sb, w_m, m):
    """T_meanvar_z-tuned: the V1 optimal mean+variance attacker."""
    sig = _benign_signal(edges, roles, sb, rng)
    out = defaultdict(list)
    for (i, j) in edges:
        if roles[i] == 1:
            out[i].append((i, j))
    for i, elist in out.items():
        E = len(elist)
        bases = emit_signal(q, w_m, m, sa, E, rng)
        for e, b in zip(elist, bases):
            sig[e] = clip01(b + rng.normal(0, SIGMA))
    return sig


# ---------------------------------------------------------------------------
# Evaluate a scheme against its tuned attacker at a given q.
# ---------------------------------------------------------------------------
def eval_scheme(scheme_name, defense, q, sa, sb, sb_cal, thr_p99, n_seeds):
    scheme_fn = scheme_of(scheme_name, sb_cal)
    k_typ = max(2, int(round(P_EDGE * (N - 1))))
    family = {"T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT"}

    # choose the tuned attacker + its best strategy for this scheme
    def run_attacker(strategy_arg):
        tprs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            edges = gen_er_digraph(N, P_EDGE, rng)
            roles = assign_roles(N, RHO, rng)
            if scheme_name in family:
                sig = attacker_firstmoment(edges, roles, rng, q, sa, sb, strategy_arg)
            elif scheme_name == "T_max":
                sig = attacker_tmax(edges, roles, rng, q, sa, sb, strategy_arg)
            elif scheme_name == "T_median":
                # OPTIMAL median attack: strategy_arg is the attack fraction.
                # signal_tmedian_optimal caps it at the largest minority so the
                # median stays benign (the breakdown-point attack).
                sig = signal_tmedian_optimal(edges, roles, rng, sa, sb,
                                             attack_frac=strategy_arg)
            elif scheme_name == "T_meanvar_z":
                w_m, m = strategy_arg
                sig = attacker_v1_meanvar(edges, roles, rng, q, sa, sb, w_m, m)
            else:  # T4_EigenTrust: dilution proxy (disqualified on cost anyway)
                sig = attacker_firstmoment(edges, roles, rng, q, sa, sb, strategy_arg)
            sc = scheme_fn(edges, sig, N)
            roles_arr = np.array(roles)
            valid = ~np.isnan(sc)
            benign = sc[valid & (roles_arr == 0)]
            attack = sc[valid & (roles_arr == 1)]
            if len(benign) and len(attack):
                thr = np.quantile(benign, 0.99)
                tprs.append(np.mean(attack > thr))
        return float(np.mean(tprs)) if tprs else float("nan"), \
               (1.96*np.std(tprs, ddof=1)/np.sqrt(len(tprs)) if len(tprs) > 1 else 0.0)

    # attacker minimizes TPR over its strategy knob
    if scheme_name == "T_meanvar_z":
        br = best_response(q, sa, sb, SIGMA, k_typ)
        tpr, ci = run_attacker((br["w_m"], br["m"]))
        best = {"tpr": tpr, "ci": ci, "strategy": f"w_m={br['w_m']:.2f},m={br['m']:.2f}"}
    elif scheme_name == "T_max":
        best = None
        for cap in [sb, sb+0.1, sb+0.2, (sa+sb)/2, sa]:
            tpr, ci = run_attacker(cap)
            if best is None or tpr < best["tpr"]:
                best = {"tpr": tpr, "ci": ci, "strategy": f"cap={cap:.2f}"}
    elif scheme_name == "T_median":
        # OPTIMAL: sweep attack fraction near the breakdown point. Enforce the
        # uniform meaningful-yield rule: only fractions delivering payload on
        # >= 0.2 of edges count. Report BOTH:
        #   - worst-case: lowest TPR at min-meaningful-yield (the rule's answer)
        #   - breakdown: TPR at the largest-minority (max payload while evading)
        curve = {}
        for frac in [0.2, 0.3, 0.4, 0.45, 0.49]:
            tpr, ci = run_attacker(frac)
            curve[frac] = (tpr, ci)
        # worst-case under the convention = min TPR over meaningful fractions
        wc_frac = min(curve, key=lambda f: curve[f][0])
        wc_tpr, wc_ci = curve[wc_frac]
        # breakdown = the largest-minority point (max payload, ~0.49)
        bd_tpr, bd_ci = curve[0.49]
        best = {"tpr": wc_tpr, "ci": wc_ci,
                "strategy": f"attack_frac={wc_frac:.2f}",
                "breakdown_tpr": bd_tpr, "breakdown_frac": 0.49}
    else:
        best = None
        for forge in FORGE_GRID:
            tpr, ci = run_attacker(forge)
            if best is None or tpr < best["tpr"]:
                best = {"tpr": tpr, "ci": ci, "strategy": f"forge={forge:.2f}"}
    return best


def run(out_csv, n_seeds=100, smoke=False):
    defenses = ["D5_Stochastic"] if smoke else DEFENSES_ALL
    schemes = ["T1_mean", "T_max", "T_median", "T_meanvar_z"] if smoke else \
        ["T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT", "T_max", "T_median",
         "T4_EigenTrust", "T_meanvar_z"]
    q_vals = [0.3, 0.5] if smoke else Q_VALS
    if smoke:
        n_seeds = 20

    k_typ = max(2, int(round(P_EDGE * (N - 1))))
    rows = []
    t0 = time.time()
    total = len(defenses) * len(schemes) * len(q_vals)
    done = 0
    _last = -1

    # FPR-honesty is defense-independent given sb_cal; compute per (scheme,defense)
    for defense in defenses:
        sa, sb = get_s_a(defense), get_s_b(defense)
        sb_cal = sb
        for scheme_name in schemes:
            fn = scheme_of(scheme_name, sb_cal)
            hon = fpr_honesty(fn, sb_cal, n_cal=(40 if smoke else 200))
            cost = analytic_cost(scheme_name, k_typ, N=N)
            cvt = cost_relative_to_T1(scheme_name, k_typ, N=N)
            for q in q_vals:
                best = eval_scheme(scheme_name, defense, q, sa, sb, sb_cal,
                                   hon["threshold_p99"], n_seeds)
                rows.append({
                    "scheme": scheme_name, "defense": defense, "q_attack_frac": q,
                    "tuned_attacker_tpr": round(best["tpr"], 6),
                    "tpr_ci95": round(best["ci"], 6),
                    "breakdown_tpr": (round(best["breakdown_tpr"], 6)
                                      if "breakdown_tpr" in best else ""),
                    "attacker_strategy": best["strategy"],
                    "benign_mean": round(hon["benign_mean"], 4),
                    "benign_std": round(hon["benign_std"], 4),
                    "threshold_p99": round(hon["threshold_p99"], 4),
                    "thr_in_tail_std": round(hon["thr_in_tail"], 3),
                    "realized_benign_fpr": round(hon["realized_benign_fpr"], 4),
                    "op_count": cost["op_count"],
                    "cost_vs_T1": round(cvt, 2),
                    "complexity": cost["complexity"],
                })
                done += 1
                pct = int(100*done/total)
                if pct != _last and pct % 10 == 0:
                    print(f"  ... {pct}% ({done}/{total})  t={time.time()-t0:.0f}s")
                    _last = pct

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    _summary(rows, defenses, schemes)
    print(f"\n  -> {out_csv}  ({len(rows)} cells)")


def _summary(rows, defenses, schemes):
    print("\n" + "=" * 96)
    print("  FAIR CROSS-CHECK — each scheme vs the attacker TUNED TO IT")
    print("  detection floor (q>=0.2, worst over defenses) + FPR-honesty")
    print("=" * 96)
    print(f"  {'Scheme':<15}{'cost/T1':>8}{'worst-case':>11}{'breakdown':>11}"
          f"{'mean floor':>11}{'benign_mu':>10}{'thr_tail':>9}{'realFPR':>9}")
    print("  " + "-" * 92)
    for sch in schemes:
        sub = [r for r in rows if r["scheme"] == sch]
        floors = []
        for d in defenses:
            ds = [r["tuned_attacker_tpr"] for r in sub if r["defense"] == d]
            if ds:
                floors.append(min(ds))
        if not floors:
            continue
        r0 = sub[0]
        # breakdown (T_median only): worst over defenses of the breakdown_tpr
        bds = [float(r["breakdown_tpr"]) for r in sub
               if r.get("breakdown_tpr") not in ("", None)]
        bd_str = f"{min(bds):>11.3f}" if bds else f"{'—':>11}"
        print(f"  {sch:<15}{r0['cost_vs_T1']:>7.1f}x{min(floors):>11.3f}{bd_str}"
              f"{np.mean(floors):>11.3f}{r0['benign_mean']:>10.3f}"
              f"{r0['thr_in_tail_std']:>9.2f}{r0['realized_benign_fpr']:>9.3f}")
    print("  " + "-" * 92)
    print("  worst-case = lowest detection at meaningful yield>=0.2 (uniform rule)")
    print("  breakdown  = T_median at largest-minority (max payload while evading)")
    print("  benign_mu  = where honest nodes sit (HIGH = compressed range, e.g. T_max)")
    print("  thr_tail   = std the FPR=1% cutoff sits above benign mean")
    print("  realFPR    = realized benign flag rate at the cutoff (should be ~0.01)")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if args.smoke:
        print("=== SMOKE MODE ===")
    print("EXP3 CROSS-CHECK: each scheme vs its own tuned attacker + FPR-honesty")
    run(os.path.join(here, "exp3_crosscheck.csv"), n_seeds=args.seeds,
        smoke=args.smoke)
    print("\nDone.")


if __name__ == "__main__":
    main()
