"""
phase4_sim.py — Phase 4 composition simulator (MINIMAL CELL + validation)
=========================================================================

Pipeline for one grid cell:
  network generation  ->  per-edge signals (from real measured s_a/s_b/s_m)
  ->  trust aggregation  ->  per-node scores  ->  metrics

This minimal version implements:
  - directed Erdos-Renyi network generation
  - role assignment (attacker fraction rho_atk)
  - per-edge signal model for AP1 (baseline) and AP5 (composition mimicry)
  - T1 aggregation (per-edge mean over a node's outgoing edges)
  - 4 metrics (AUC, trust-gap, stealth-ratio, worst-case visibility)

Plus two validations that need no Pi:
  (A) Theorem 1 self-check: single attacker, sweep attack-edge fraction,
      confirm the aggregate crosses T_det exactly at the predicted m_max.
  (B) Dichotomy cross-check: D1 (not window-mimicable) vs D3 (fully mimicable)
      under an AP5 phi-sweep, confirming the simulator reproduces the s_m finding.

Signal values are loaded from sm_results.json (measured on real Pi traces).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Signal access
# ---------------------------------------------------------------------------

with open("sm_results.json") as f:
    SIG = json.load(f)


# ---------------------------------------------------------------------------
# CALIBRATION NOTE — per-edge signal noise (sigma)
# ---------------------------------------------------------------------------
# Each edge's signal is the mean of ~570 per-window classifier outputs, so its
# std ~= (window-level signal std) / sqrt(n_windows_per_edge). The minimal-cell
# tests showed this sigma is PIVOTAL: it decides whether AP5 gap-compression
# merely shrinks the operational margin (recoverable by recalibrating the
# threshold) or actually defeats detection at a fixed FPR. The value below is a
# placeholder; it must be calibrated from the real window-level variance
# (dump raw per-window probability arrays in one more measure_sm pass).
SIGMA_EDGE_DEFAULT = 0.03  # PLACEHOLDER — calibrate from raw per-window arrays


def s_a(defense):
    return SIG[defense]["s_a"]


def s_b(defense):
    return SIG[defense]["s_b"]


def s_m(defense, p):
    sm = SIG[defense]["s_m"]
    return sm[str(p)] if str(p) in sm else sm[p]


def midpoint(defense):
    return (s_a(defense) + s_b(defense)) / 2.0


# ---------------------------------------------------------------------------
# Network + roles
# ---------------------------------------------------------------------------

def gen_er_digraph(N, p_edge, rng):
    """Directed Erdos-Renyi: edge i->j present w.p. p_edge (no self-loops)."""
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < p_edge:
                edges.append((i, j))
    return edges


def assign_roles(N, rho_atk, rng):
    """1 = attacker, 0 = benign."""
    roles = np.zeros(N, dtype=int)
    n_atk = max(1, int(round(rho_atk * N)))
    atk_idx = rng.choice(N, size=n_atk, replace=False)
    roles[atk_idx] = 1
    return roles


# ---------------------------------------------------------------------------
# Per-edge signal model
#   signal(i->j) reflects source i's behavior as observed by j.
#   benign source:        s_b
#   attacker, AP1:        s_a on every outgoing edge (full attack, no evasion)
#   attacker, AP5:        mimicry signal s_m(p) on a phi-fraction of edges,
#                         s_b on the rest (composition-aware: keep aggregate low)
# Per-edge Gaussian noise (sigma) models finite-window concentration (~sqrt(570)).
# ---------------------------------------------------------------------------

def assign_edge_signals(edges, roles, defense, profile, rng,
                        p=None, phi=None, sigma=SIGMA_EDGE_DEFAULT,
                        inclique=None):
    out_edges = defaultdict(list)
    for (i, j) in edges:
        out_edges[i].append((i, j))

    sig = {}
    for i, elist in out_edges.items():
        E = len(elist)
        if roles[i] == 0:
            vals = {e: s_b(defense) for e in elist}
        elif profile == "AP1":
            vals = {e: s_a(defense) for e in elist}
        elif profile == "AP4":
            # Pure dilution: full attack signal (s_a) on phi*E edges, benign on rest.
            # Isolates the Theorem-1 aggregation-dilution mechanism (no window mimicry).
            sa = s_a(defense)
            sb = s_b(defense)
            m = int(round((phi if phi is not None else 0.5) * E))
            order = rng.permutation(E)
            atk_set = set(order[:m].tolist())
            vals = {e: (sa if k in atk_set else sb)
                    for k, e in enumerate(elist)}
        elif profile == "AP5":
            # Composition mimicry: mimicry signal (s_m(p)) on phi*E edges, benign on rest.
            # Combines window mimicry (low per-edge signal) with optional dilution.
            smv = s_m(defense, p)
            m = int(round((phi if phi is not None else 1.0) * E))
            order = rng.permutation(E)
            mimic_set = set(order[:m].tolist())
            vals = {e: (smv if k in mimic_set else s_b(defense))
                    for k, e in enumerate(elist)}
        elif profile == "AP6":
            # Collusion / mutual reputation boosting. All attackers form one clique.
            #   inclique=None (default): passive collusion — in-clique signal = s_b
            #     (colluders behave benignly toward each other, hardware-honest model).
            #   inclique=<value> (e.g., 0.0): active collusion — colluders forge a
            #     signal below s_b on in-clique edges, simulating direct manipulation
            #     of the trust report (the classical EigenTrust collusion model).
            sa = s_a(defense)
            inclique_val = s_b(defense) if inclique is None else float(inclique)
            vals = {}
            for e in elist:
                dst = e[1]
                vals[e] = inclique_val if roles[dst] == 1 else sa
        else:
            raise ValueError(f"unknown profile {profile}")

        for e, v in vals.items():
            sig[e] = float(np.clip(v + rng.normal(0, sigma), 0.0, 1.0))
    return sig


# ---------------------------------------------------------------------------
# Trust aggregation (T1 = per-edge mean over a node's outgoing edges)
# ---------------------------------------------------------------------------

def aggregate_T1(edges, sig, N):
    by_src = defaultdict(list)
    for e in edges:
        by_src[e[0]].append(sig[e])
    scores = np.full(N, np.nan)
    for i in range(N):
        if by_src[i]:
            scores[i] = float(np.mean(by_src[i]))
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(scores, roles, T_det):
    mask = ~np.isnan(scores)
    s = scores[mask]
    r = roles[mask]
    out = {"n_scored": int(mask.sum())}
    if len(set(r.tolist())) < 2:
        out.update(auc=float("nan"), gap=float("nan"), stealth=float("nan"),
                   tpr_at_fpr01=float("nan"), worst_vis=float("nan"))
        return out
    benign = s[r == 0]
    attack = s[r == 1]
    out["auc"] = float(roc_auc_score(r, s))
    out["gap"] = float(attack.mean() - benign.mean())
    # M2 stealth: frac attackers below the AP1-calibrated (fixed midpoint) threshold
    out["stealth"] = float(np.mean(attack < T_det))
    # M1 operational: recalibrate threshold to benign FPR=1%, measure attacker TPR
    thr = float(np.quantile(benign, 0.99))
    out["tpr_at_fpr01"] = float(np.mean(attack > thr))
    # M4 worst-case visibility: score of the most-hidden attacker
    out["worst_vis"] = float(attack.min())
    return out


# ---------------------------------------------------------------------------
# One cell (averaged over seeds)
# ---------------------------------------------------------------------------

def run_cell(defense, profile, scheme=None, N=20, p_edge=0.3, rho_atk=0.30,
             p=None, phi=None, sigma=SIGMA_EDGE_DEFAULT, n_seeds=30, base_seed=0,
             inclique=None):
    if scheme is None:
        from trust_schemes import T1_mean
        scheme = T1_mean
    T_det = midpoint(defense)
    acc = defaultdict(list)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        edges = gen_er_digraph(N, p_edge, rng)
        roles = assign_roles(N, rho_atk, rng)
        sig = assign_edge_signals(edges, roles, defense, profile, rng,
                                  p=p, phi=phi, sigma=sigma, inclique=inclique)
        scores = scheme(edges, sig, N)
        m = compute_metrics(scores, roles, T_det)
        for k, v in m.items():
            if not (isinstance(v, float) and np.isnan(v)):
                acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Validation A — Theorem 1 self-check
# ---------------------------------------------------------------------------

def validate_theorem1(defense, E=11):
    sa, sb, T = s_a(defense), s_b(defense), midpoint(defense)
    predicted_mmax = E * (T - sb) / (sa - sb)
    print(f"\n[Validation A] Theorem 1 self-check on {defense} "
          f"(E={E}, T_det=midpoint={T:.4f})")
    print(f"  predicted m_max = E*(T-s_b)/(s_a-s_b) = {predicted_mmax:.3f} edges "
          f"(largest m that evades = floor = {int(np.floor(predicted_mmax))})")
    crossing = None
    for m in range(E + 1):
        agg = (m * sa + (E - m) * sb) / E
        if crossing is None and agg > T:
            crossing = m
        if m in (0, int(np.floor(predicted_mmax)),
                 int(np.floor(predicted_mmax)) + 1, E):
            print(f"    m={m:2d}  aggregate={agg:.4f}  -> "
                  f"{'DETECTED' if agg > T else 'evades'}")
    ok = crossing == int(np.floor(predicted_mmax)) + 1
    print(f"  first detected at m={crossing}; predicted floor(m_max)+1="
          f"{int(np.floor(predicted_mmax))+1}  -> {'MATCH' if ok else 'MISMATCH'}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 74)
    print("MINIMAL CELL: D1_RF x T1(per-edge-mean) x AP1 x N=20")
    print("=" * 74)
    cell = run_cell("D1_RF", "AP1", N=20, p_edge=0.3, rho_atk=0.30, n_seeds=50)
    print(f"  T_det (midpoint) = {midpoint('D1_RF'):.4f}")
    for k, v in cell.items():
        print(f"  {k:12s} = {v:.4f}")
    print("  expected: AUC~1.0, gap~0.99, stealth~0 (AP1 = no evasion, "
          "attackers trivially caught)")

    # Validation A
    validate_theorem1("D1_RF", E=11)

    # Validation B — dichotomy under AP5 (phi = fraction of edges carrying mimicry)
    print("\n" + "=" * 74)
    print("[Validation B] AP5 dichotomy: D1 (tight) vs D3 (mimicable), N=20")
    print("phi = fraction of attacker edges carrying mimicry windows at p=10%")
    print("=" * 74)
    for defense in ["D1_RF", "D3_RHMD"]:
        smv = s_m(defense, 10)
        T = midpoint(defense)
        evades_all = smv < T
        print(f"\n  {defense}: s_m(10%)={smv:.4f}, T_det={T:.4f}  "
              f"-> single mimic edge {'EVADES' if evades_all else 'is visible'}")
        for phi in (0.25, 0.5, 0.75, 1.0):
            r = run_cell(defense, "AP5", N=20, p=10, phi=phi, n_seeds=50)
            print(f"    phi={phi:.2f}  AUC={r['auc']:.3f}  gap={r['gap']:+.3f}  "
                  f"stealth={r['stealth']:.3f}  TPR@FPR1%={r['tpr_at_fpr01']:.3f}  "
                  f"worst_vis={r['worst_vis']:.3f}")

    # Validation C — sigma (observation-length) sensitivity at full mimicry (phi=1.0)
    print("\n" + "=" * 74)
    print("[Validation C] sigma sweep under AP5 phi=1.0, p=10%  (N=20)")
    print("sigma_edge ~ (window-std)/sqrt(windows-per-edge): low=long obs, high=short obs")
    print("the crossover sigma* is where FPR-controlled detection (TPR@FPR1%) collapses")
    print("=" * 74)
    for defense in ["D3_RHMD", "D6_DRL", "D2_CNN"]:
        smv = s_m(defense, 10)
        print(f"\n  {defense}: AP5 attacker per-edge signal s_m(10%)={smv:.4f}, "
              f"benign s_b={s_b(defense):.4f}, gap={smv - s_b(defense):+.4f}")
        for sig in (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30):
            r = run_cell(defense, "AP5", N=20, p=10, phi=1.0, sigma=sig, n_seeds=100)
            print(f"    sigma={sig:.2f}  gap={r['gap']:+.3f}  "
                  f"AUC={r['auc']:.3f}  TPR@FPR1%={r['tpr_at_fpr01']:.3f}")

    # Validation D — defense aggregators against dilution (AP4) vs mimicry (AP5)
    print("\n" + "=" * 74)
    print("[Validation D] Defense aggregators: dilution (AP4) vs mimicry (AP5)")
    print("Setup: N=20, sigma=0.15 (moderate observation noise)")
    print("AP4 phi=0.30 (attack on 30% of edges, full s_a signal)")
    print("AP5 phi=1.00 p=10% (mimicry on every edge)")
    print("Expectation: tail-sensitive aggregators (T_max, T_quantile_90) DEFEAT")
    print("dilution but NOT mimicry; mean/median fail on both.")
    print("=" * 74)
    from trust_schemes import T1_mean, T_median, T_quantile, T_max
    schemes_to_test = [
        ("T1_mean",       T1_mean),
        ("T_median",      T_median),
        ("T_quantile_90", lambda e, s, N: T_quantile(e, s, N, q=0.9)),
        ("T_max",         T_max),
    ]
    SIGMA_D = 0.15
    for defense in ["D1_RF", "D3_RHMD"]:
        print(f"\n  {defense}  (s_a={s_a(defense):.3f}, s_b={s_b(defense):.3f}, "
              f"s_m(10%)={s_m(defense, 10):.3f}):")
        print(f"    {'scheme':<15s} {'AP1':>9s} {'AP4 dilute':>12s} {'AP5 mimicry':>13s}")
        print(f"    {'':<15s} {'TPR@FPR1%':>9s} {'TPR@FPR1%':>12s} {'TPR@FPR1%':>13s}")
        for name, sch in schemes_to_test:
            r1 = run_cell(defense, "AP1", scheme=sch, N=20, sigma=SIGMA_D, n_seeds=100)
            r4 = run_cell(defense, "AP4", scheme=sch, phi=0.30, N=20,
                          sigma=SIGMA_D, n_seeds=100)
            r5 = run_cell(defense, "AP5", scheme=sch, phi=1.0, p=10, N=20,
                          sigma=SIGMA_D, n_seeds=100)
            print(f"    {name:<15s} "
                  f"{r1['tpr_at_fpr01']:>9.3f} "
                  f"{r4['tpr_at_fpr01']:>12.3f} "
                  f"{r5['tpr_at_fpr01']:>13.3f}")
