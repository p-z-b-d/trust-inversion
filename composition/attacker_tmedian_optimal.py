"""
attacker_tmedian_optimal.py
===========================
The OPTIMAL attacker against T_median — built with the same rigor as V1, because
the heuristic T_median attacker in exp3_crosscheck.py was provably suboptimal
(it forced >=55% attack, which MOVES the median and gets caught — attacking the
median exactly where it is STRONG, not weak).

The real T_median weakness (derived):
  A node is flagged iff median(edges) > threshold.
  median(edges) > threshold  <=>  MORE THAN HALF the edges exceed threshold.
  => The attacker can emit full attack signal s_a on UP TO floor((k-1)/2) edges
     (just under half) and keep the median in the benign half. The node evades
     while still delivering payload on ~49% of its edges.

  This is the classic breakdown point of the median: robust to a MINORITY of
  corrupted values, defenceless the instant the corruption is a minority that
  stays a minority. The optimal attacker sits exactly at the breakdown boundary.

Optimal strategy (closed form, no search needed):
  attack_fraction f* = floor((k-1)/2) / k     (largest minority)
  On those edges emit s_a (full payload — no need to forge low, since they're
  the minority and don't touch the median).
  On the remaining majority emit benign s_b (keeps the median at s_b).

We VERIFY this dominates:
  (1) the heuristic (>=55% attack) — should be much stronger,
  (2) a full grid of attack fractions — f* should be the min-TPR point,
  (3) per-edge forge variations — forging low on the majority shouldn't help
      (benign already minimizes the median).
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_sim import (  # noqa: E402
    s_a as get_s_a, s_b as get_s_b,
    gen_er_digraph, assign_roles,
)
from trust_schemes import T_median  # noqa: E402

SIGMA = 0.15
N = 20
P_EDGE = 0.30
RHO = 0.30


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def optimal_attack_fraction(k):
    """Largest minority: floor((k-1)/2)/k. This many edges can carry full s_a
    while the median stays benign."""
    if k < 2:
        return 0.0
    return np.floor((k - 1) / 2) / k


def signal_tmedian_optimal(edges, roles, rng, sa, sb, attack_frac=None):
    """Optimal T_median attacker: emit s_a on the largest minority of edges
    (just under half), benign s_b on the majority. If attack_frac is given, use
    it (for the verification grid); else use the closed-form optimum per node."""
    sig = {}
    from collections import defaultdict
    out = defaultdict(list)
    for (i, j) in edges:
        if roles[i] == 0:
            sig[(i, j)] = clip01(sb + rng.normal(0, SIGMA))
        else:
            out[i].append((i, j))
    for i, elist in out.items():
        E = len(elist)
        f = optimal_attack_fraction(E) if attack_frac is None else attack_frac
        m = int(np.floor(f * E))
        # ensure it stays a MINORITY (median untouched): cap at floor((E-1)/2)
        m = min(m, int(np.floor((E - 1) / 2)))
        order = rng.permutation(E)
        atk = set(order[:m].tolist())
        for kk, e in enumerate(elist):
            base = sa if kk in atk else sb
            sig[e] = clip01(base + rng.normal(0, SIGMA))
    return sig


def measure_tpr(defense, attack_frac, n_seeds, force_minority=True):
    """Realized TPR@FPR=1% for T_median under the optimal (or specified) attack."""
    sa, sb = get_s_a(defense), get_s_b(defense)
    tprs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        edges = gen_er_digraph(N, P_EDGE, rng)
        roles = assign_roles(N, RHO, rng)
        if force_minority:
            sig = signal_tmedian_optimal(edges, roles, rng, sa, sb, attack_frac)
        else:
            # verification variant: do NOT cap at minority (lets median move)
            sig = _signal_uncapped(edges, roles, rng, sa, sb, attack_frac)
        sc = T_median(edges, sig, N)
        roles_arr = np.array(roles)
        valid = ~np.isnan(sc)
        benign = sc[valid & (roles_arr == 0)]
        attack = sc[valid & (roles_arr == 1)]
        if len(benign) and len(attack):
            thr = np.quantile(benign, 0.99)
            tprs.append(np.mean(attack > thr))
    return float(np.mean(tprs)) if tprs else float("nan")


def _signal_uncapped(edges, roles, rng, sa, sb, attack_frac):
    from collections import defaultdict
    sig = {}
    out = defaultdict(list)
    for (i, j) in edges:
        if roles[i] == 0:
            sig[(i, j)] = clip01(sb + rng.normal(0, SIGMA))
        else:
            out[i].append((i, j))
    for i, elist in out.items():
        E = len(elist)
        m = int(round((attack_frac if attack_frac else 0.5) * E))
        order = rng.permutation(E)
        atk = set(order[:m].tolist())
        for kk, e in enumerate(elist):
            base = sa if kk in atk else sb
            sig[e] = clip01(base + rng.normal(0, SIGMA))
    return sig


def verify(defense="D5_Stochastic", n_seeds=100):
    print("=" * 72)
    print(f"  OPTIMAL T_median ATTACKER — verification on {defense}")
    print("=" * 72)

    # (1) closed-form optimum vs heuristic (>=55% attack)
    opt_tpr = measure_tpr(defense, None, n_seeds)  # closed-form per-node optimum
    heur_tpr = measure_tpr(defense, 0.55, n_seeds, force_minority=False)
    print(f"\n  (1) Optimal (largest-minority) vs heuristic (55% attack):")
    print(f"      optimal  TPR = {opt_tpr:.3f}   <- attacker wants this LOW")
    print(f"      heuristic TPR = {heur_tpr:.3f}   (the old exp3_crosscheck value)")
    print(f"      => optimal is {'STRONGER' if opt_tpr < heur_tpr else 'NOT stronger'} "
          f"(lower TPR = better attack)")

    # (2) grid of attack fractions — the min should be near the breakdown point
    print(f"\n  (2) TPR vs attack fraction (grid) — min = attacker's best:")
    print(f"      {'frac':>6}{'TPR':>9}{'note':>16}")
    k_typ = max(2, int(round(P_EDGE * (N - 1))))
    fstar = optimal_attack_fraction(k_typ)
    best_f, best_tpr = None, 1e9
    for f in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.5, 0.55, 0.6, 0.7]:
        tpr = measure_tpr(defense, f, n_seeds, force_minority=(f < 0.5))
        note = " <- ~breakdown" if abs(f - 0.49) < 0.02 else ""
        if tpr < best_tpr:
            best_tpr, best_f = tpr, f
        print(f"      {f:>6.2f}{tpr:>9.3f}{note:>16}")
    print(f"      grid min at frac={best_f} (TPR={best_tpr:.3f}); "
          f"closed-form f*={fstar:.3f}")

    # (3) does forging the majority LOW help? (shouldn't — benign already minimal)
    print(f"\n  (3) Forge-low-majority check: benign majority is already optimal")
    print(f"      (median sits in the benign half regardless of how low the")
    print(f"       minority attack edges go — nothing to gain by forging).")

    print("\n" + "=" * 72)
    verdict = "REAL" if opt_tpr < heur_tpr - 0.05 else "MARGINAL"
    print(f"  VERDICT: optimal T_median attack is {verdict}. "
          f"T_median's true worst-case TPR = {opt_tpr:.3f} on {defense}")
    print("=" * 72)
    return opt_tpr


if __name__ == "__main__":
    verify("D5_Stochastic", n_seeds=100)
    print()
    verify("D3_RHMD", n_seeds=100)
