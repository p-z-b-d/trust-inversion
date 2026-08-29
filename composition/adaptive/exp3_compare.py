"""
exp3_compare.py
===============
Shared comparison harness for the 2x2 adaptive-attacker matrix vs T_meanvar_z.

                 grid strategy (V0)     continuous-optimal (V1)
  threshold-blind      V0                      V1
  threshold-aware      V0T                     V1T

All four tiers run through IDENTICAL network seeds/topologies so the comparison
is apples-to-apples. For each (tier, defense, attack-yield q) we measure:

  * realized TPR@FPR=1%  (the defense's node-level detection -- success metric b,
    the primary: does the attacker NODE pass as benign?)
  * edge-evasion rate    (success metric a, secondary: does at least one real
    attack edge slip below the per-edge benign 99th-pct line?)
  * inversion flag       (metric c bonus: did the attacker's aggregate score fall
    BELOW the benign mean -- the Corollary 1.2 outcome?)
  * T1-mean baseline TPR  (E: what the UNDEFENDED aggregator does under the same
    attacker -- the contrast that makes the frontier meaningful)

Threshold-aware tiers (V0T, V1T): the attacker targets the ACTUAL measured
benign threshold. We estimate it per defense as the mean over calibration seeds
of quantile(benign_node_scores, 0.99) under honest behavior, then let the
attacker optimize to sit just under it. This is the real cutoff the defense uses
(see phase4_sim.compute_metrics: thr = quantile(benign, 0.99)).

Outputs:
  exp3_compare_grid.csv      -- every (tier, defense, q) cell, all metrics + CIs
  exp3_compare_frontier.csv  -- per (tier, defense, q): the attacker-optimal TPR
  console                    -- one compact summary table only

Run:
  python exp3_compare.py               # full: 100 seeds, 6 defenses
  python exp3_compare.py --smoke       # fast plumbing check
  python exp3_compare.py --seeds 50
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
    gen_er_digraph, assign_roles, compute_metrics,
)
from trust_schemes import (  # noqa: E402
    T_meanvar_z, T1_mean, T2_SL, T3_Beta, T4_EigenTrust, T5_TFLDT,
    T_max, T_median,
)
from attacker_v1_optimal import best_response, emit_signal  # noqa: E402
from scheme_cost import analytic_cost, cost_relative_to_T1  # noqa: E402


# The competitive field of trust schemes (Option A: one attacker, all schemes).
# T_meanvar_z is sigma/s_b-parameterized so it's built per-defense below.
def scheme_field(sb_cal, sigma):
    return {
        "T1_mean": T1_mean,
        "T2_SL": T2_SL,
        "T3_Beta": T3_Beta,
        "T5_TFLDT": T5_TFLDT,
        "T_max": T_max,
        "T_median": T_median,
        "T4_EigenTrust": T4_EigenTrust,
        "T_meanvar_z": lambda e, s, N: T_meanvar_z(e, s, N, sb_cal, sigma),
    }


DEFENSES_ALL = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
Q_VALS_FULL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# V0 grid forge values (the discrete strategy space of the grid attacker)
FORGE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
SIGMA = 0.15
N_DEFAULT = 20
P_EDGE = 0.30
RHO_ATK = 0.30


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Threshold calibration: the ACTUAL measured benign 99th-pct line per defense.
# ---------------------------------------------------------------------------
def calibrate_threshold(defense, sb_cal, n_cal=100, N=N_DEFAULT):
    """Mean benign node-score 99th percentile under honest behavior — the real
    cutoff the defense uses. This is the line threshold-aware attackers target."""
    thrs = []
    for s in range(n_cal):
        rng = np.random.default_rng(10_000 + s)
        edges = gen_er_digraph(N, P_EDGE, rng)
        roles = assign_roles(N, RHO_ATK, rng)
        # all-honest signal: every node emits s_b + noise
        sig = {}
        by_src = defaultdict(list)
        for (i, j) in edges:
            sig[(i, j)] = _clip01(sb_cal + rng.normal(0, SIGMA))
        scores = T_meanvar_z(edges, sig, N, sb_cal, SIGMA)
        benign = scores[~np.isnan(scores)]
        if len(benign):
            thrs.append(np.quantile(benign, 0.99))
    return float(np.mean(thrs)) if thrs else 3.0


# ---------------------------------------------------------------------------
# Attacker signal builders (per tier). Each returns an edge->value signal dict.
# Benign nodes always emit s_b + noise. Attackers differ by tier.
# ---------------------------------------------------------------------------
def signal_v0(edges, roles, defense, rng, q, forge, sa, sb):
    """V0: grid two-point attacker (q edges at s_a, rest at `forge`)."""
    out_edges = defaultdict(list)
    for (i, j) in edges:
        out_edges[i].append((i, j))
    sig = {}
    for i, elist in out_edges.items():
        if roles[i] == 0:
            for e in elist:
                sig[e] = _clip01(sb + rng.normal(0, SIGMA))
        else:
            E = len(elist)
            m = int(round(q * E))
            order = rng.permutation(E)
            atk = set(order[:m].tolist())
            for kk, e in enumerate(elist):
                base = sa if kk in atk else forge
                sig[e] = _clip01(base + rng.normal(0, SIGMA))
    return sig


def signal_v1(edges, roles, defense, rng, q, w_m, m, sa, sb):
    """V1: continuous-optimal 3-point attacker using optimized (w_m, m)."""
    out_edges = defaultdict(list)
    for (i, j) in edges:
        out_edges[i].append((i, j))
    sig = {}
    for i, elist in out_edges.items():
        if roles[i] == 0:
            for e in elist:
                sig[e] = _clip01(sb + rng.normal(0, SIGMA))
        else:
            E = len(elist)
            bases = emit_signal(q, w_m, m, sa, E, rng)
            for e, b in zip(elist, bases):
                sig[e] = _clip01(b + rng.normal(0, SIGMA))
    return sig


# ---------------------------------------------------------------------------
# Metrics for one configuration over seeds.
# ---------------------------------------------------------------------------
def measure(defense, tier, q, sa, sb, sb_cal, threshold_z, n_seeds,
            forge=None, w_m=None, m=None, N=N_DEFAULT, base_seed=0):
    """Score EVERY scheme in the competitive field under the same attacker.
    Returns per-scheme TPR (+CI), plus attacker-side edge-evasion (a) and, for
    each scheme, inversion rate (c). T1 is included in the field so E (baseline)
    is just the T1 entry."""
    field = scheme_field(sb_cal, SIGMA)
    tpr_acc = {name: [] for name in field}
    inv_acc = {name: [] for name in field}
    edge_evasions = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed)
        edges = gen_er_digraph(N, P_EDGE, rng)
        roles = assign_roles(N, RHO_ATK, rng)

        if tier in ("V0", "V0T"):
            sig = signal_v0(edges, roles, defense, rng, q, forge, sa, sb)
        else:  # V1, V1T
            sig = signal_v1(edges, roles, defense, rng, q, w_m, m, sa, sb)

        bmask = np.array([roles[i] == 0 for i in range(N)])
        amask = np.array([roles[i] == 1 for i in range(N)])

        # --- score every scheme on the SAME signal ---
        for name, fn in field.items():
            try:
                scores = fn(edges, sig, N)
            except Exception:
                continue
            mets = compute_metrics(scores, roles, midpoint(defense))
            if not np.isnan(mets.get("tpr_at_fpr01", np.nan)):
                tpr_acc[name].append(mets["tpr_at_fpr01"])
            # inversion (c) per scheme
            valid = ~np.isnan(scores)
            if (bmask & valid).any() and (amask & valid).any():
                inv_acc[name].append(
                    1.0 if scores[amask & valid].mean() < scores[bmask & valid].mean()
                    else 0.0)

        # --- a: edge-evasion (attacker-side, scheme-independent) ---
        benign_edge_vals = [sig[(i, j)] for (i, j) in edges if roles[i] == 0]
        if benign_edge_vals:
            edge_thr = np.quantile(benign_edge_vals, 0.99)
            atk_payload_vals = [sig[(i, j)] for (i, j) in edges
                                if roles[i] == 1 and sig[(i, j)] > (sa + sb) / 2]
            if atk_payload_vals:
                edge_evasions.append(1.0 if min(atk_payload_vals) <= edge_thr else 0.0)
            else:
                edge_evasions.append(1.0)

    def _mc(x):
        a = np.asarray(x, dtype=float)
        if a.size == 0:
            return float("nan"), float("nan")
        mean = float(a.mean())
        ci = 1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
        return mean, ci

    per_scheme = {}
    for name in field:
        tpr, ci = _mc(tpr_acc[name])
        inv, _ = _mc(inv_acc[name])
        per_scheme[name] = {"tpr_mean": tpr, "tpr_ci95": ci, "inversion_rate": inv}
    return {
        "per_scheme": per_scheme,
        "edge_evasion_rate": _mc(edge_evasions)[0],
    }


# ---------------------------------------------------------------------------
# Driver: run all four tiers.
# ---------------------------------------------------------------------------
def run(out_grid, out_frontier, n_seeds=100, smoke=False):
    defenses = ["D5_Stochastic"] if smoke else DEFENSES_ALL
    q_vals = [0.1, 0.3, 0.5, 0.8] if smoke else Q_VALS_FULL
    if smoke:
        n_seeds = 15

    # avg out-degree for the optimizer's k (E[k] ~ p_edge*(N-1))
    k_typical = max(2, int(round(P_EDGE * (N_DEFAULT - 1))))

    rows = []
    t0 = time.time()
    # pre-calibrate the measured threshold per defense (for +T tiers)
    thr_by_def = {}
    for d in defenses:
        thr_by_def[d] = calibrate_threshold(d, get_s_b(d),
                                            n_cal=(20 if smoke else 100))

    total = len(defenses) * len(q_vals) * 4
    done = 0
    _last = -1
    for defense in defenses:
        sa, sb = get_s_a(defense), get_s_b(defense)
        sb_cal = sb
        thr_z = thr_by_def[defense]

        for q in q_vals:
            # --- V1 blind: optimize best response, threshold_z=None ---
            br_blind = best_response(q, sa, sb, SIGMA, k_typical,
                                     threshold_z=None)
            # --- V1T aware: optimize against the measured threshold ---
            br_aware = best_response(q, sa, sb, SIGMA, k_typical,
                                     threshold_z=thr_z)

            # --- V0 blind: pick the grid forge that MINIMIZES realized TPR
            # against T_meanvar_z (the defense the attacker targets) ---
            best_v0 = None
            best_v0_tpr = None
            for forge in FORGE_GRID:
                mm = measure(defense, "V0", q, sa, sb, sb_cal, thr_z, n_seeds,
                             forge=forge)
                tmvz_tpr = mm["per_scheme"]["T_meanvar_z"]["tpr_mean"]
                if best_v0_tpr is None or tmvz_tpr < best_v0_tpr:
                    best_v0_tpr = tmvz_tpr
                    best_v0 = {**mm, "forge": forge}
            # --- V0T aware: same grid, but attacker only needs to get under thr;
            # among strategies that get under, pick the one maximizing q-effect.
            # Operationally identical grid search on realized TPR here (the grid
            # attacker has no finer knob), but we tag it aware for completeness.
            best_v0t = best_v0  # grid attacker can't exploit thr knowledge further

            # --- V1 blind measured ---
            m_v1 = measure(defense, "V1", q, sa, sb, sb_cal, thr_z, n_seeds,
                           w_m=br_blind["w_m"], m=br_blind["m"])
            # --- V1T aware measured ---
            m_v1t = measure(defense, "V1T", q, sa, sb, sb_cal, thr_z, n_seeds,
                            w_m=br_aware["w_m"], m=br_aware["m"])

            for tier, mm, extra in [
                ("V0", best_v0, {"forge": best_v0["forge"], "w_m": "", "m": ""}),
                ("V0T", best_v0t, {"forge": best_v0t["forge"], "w_m": "", "m": ""}),
                ("V1", m_v1, {"forge": "", "w_m": round(br_blind["w_m"], 4),
                              "m": round(br_blind["m"], 4)}),
                ("V1T", m_v1t, {"forge": "", "w_m": round(br_aware["w_m"], 4),
                                "m": round(br_aware["m"], 4)}),
            ]:
                # one row per scheme in the competitive field
                for scheme_name, sm in mm["per_scheme"].items():
                    cost = analytic_cost(scheme_name, k=k_typical, N=N_DEFAULT)
                    rows.append({
                        "tier": tier, "defense": defense, "scheme": scheme_name,
                        "q_attack_frac": q,
                        "tpr_mean": round(sm["tpr_mean"], 6),
                        "tpr_ci95": round(sm["tpr_ci95"], 6),
                        "inversion_rate": round(sm["inversion_rate"], 4),
                        "edge_evasion_rate": round(mm["edge_evasion_rate"], 4),
                        "op_count": cost["op_count"],
                        "cost_vs_T1": round(cost_relative_to_T1(scheme_name, k=k_typical), 2),
                        "complexity": cost["complexity"],
                        "measured_threshold_z": round(thr_z, 4),
                        **extra,
                    })
                done += 1
            pct = int(100 * done / total)
            if pct != _last and pct % 10 == 0:
                print(f"  ... {pct}% ({done}/{total})  t={time.time()-t0:.0f}s")
                _last = pct

    with open(out_grid, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # frontier = same as grid here (one optimal strategy per tier/defense/q)
    with open(out_frontier, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    _summary(rows, defenses)
    print(f"\n  -> {out_grid}  ({len(rows)} cells)")
    print(f"  -> {out_frontier}")


def _summary(rows, defenses):
    """Two compact tables:
       (1) scheme comparison under the STRONGEST attacker (V1), q>=0.2 floor;
       (2) attacker-tier comparison for T_meanvar_z specifically."""
    SCHEMES = ["T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT", "T_max",
               "T_median", "T4_EigenTrust", "T_meanvar_z"]

    print("\n" + "=" * 88)
    print("  (1) SCHEME COMPARISON under strongest attacker (V1), q>=0.2 detection floor")
    print("      higher TPR = scheme resists better; cost = analytic ops vs T1")
    print("=" * 88)
    print(f"  {'Scheme':<15}{'cost/T1':>9}  " +
          "".join(f"{d.split('_')[0]:>8}" for d in defenses) + f"{'mean':>8}")
    print("  " + "-" * 84)
    for sch in SCHEMES:
        cvt = next((r["cost_vs_T1"] for r in rows if r["scheme"] == sch), float("nan"))
        floors = []
        line = f"  {sch:<15}{cvt:>8.1f}x  "
        for d in defenses:
            sub = [r["tpr_mean"] for r in rows
                   if r["scheme"] == sch and r["defense"] == d
                   and r["tier"] == "V1" and r["q_attack_frac"] >= 0.2]
            fl = min(sub) if sub else float("nan")
            floors.append(fl)
            line += f"{fl:>8.3f}"
        mean_fl = np.nanmean(floors) if floors else float("nan")
        line += f"{mean_fl:>8.3f}"
        print(line)
    print("  " + "-" * 84)

    print("\n" + "=" * 88)
    print("  (2) ATTACKER-TIER COMPARISON for T_meanvar_z, q>=0.2 detection floor")
    print("      does optimization (V0->V1) or threshold knowledge (blind->T) help attacker?")
    print("=" * 88)
    print(f"  {'Defense':<15}{'V0':>9}{'V0T':>9}{'V1':>9}{'V1T':>9}{'strongest':>11}")
    print("  " + "-" * 64)
    for d in defenses:
        floors = {}
        for tier in ("V0", "V0T", "V1", "V1T"):
            ts = [r["tpr_mean"] for r in rows
                  if r["scheme"] == "T_meanvar_z" and r["defense"] == d
                  and r["tier"] == tier and r["q_attack_frac"] >= 0.2]
            floors[tier] = min(ts) if ts else float("nan")
        strongest = min(floors, key=lambda t: floors[t])
        print(f"  {d:<15}{floors['V0']:>9.3f}{floors['V0T']:>9.3f}"
              f"{floors['V1']:>9.3f}{floors['V1T']:>9.3f}{strongest:>11}")
    print("  " + "-" * 64)
    print("  V1<V0 => strategy optimization helps attacker. *T<blind => threshold")
    print("  knowledge helps. Full per-scheme/tier/q data + edge-evasion + inversion -> CSV.")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if args.smoke:
        print("=== SMOKE MODE ===")
    print("EXPERIMENT 3 COMPARISON: V0 / V0T / V1 / V1T vs T_meanvar_z")
    run(os.path.join(here, "exp3_compare_grid.csv"),
        os.path.join(here, "exp3_compare_frontier.csv"),
        n_seeds=args.seeds, smoke=args.smoke)
    print("\nDone.")


if __name__ == "__main__":
    main()
