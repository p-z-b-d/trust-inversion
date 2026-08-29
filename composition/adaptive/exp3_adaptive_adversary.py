"""
exp3_adaptive_adversary.py
==========================
Strengthening experiment 3: the ADAPTIVE ADVERSARY vs T_meanvar_z.

Reviewer objection this kills:
    "T_meanvar_z only works because your attackers don't know about the variance
     channel. An attacker who knows you threshold on variance will shape variance
     too, and your defense collapses."

The defense:
    Score(i) = max(z_mu, z_var)
      z_mu  = (mean(x) - s_b) / (sigma / sqrt(k))
      z_var = (var(x)  - sigma^2) / sqrt(2 sigma^4 / (k-1))
    A node is flagged if EITHER channel exceeds the FPR-calibrated threshold.
    So the attacker must keep BOTH channels low SIMULTANEOUSLY.

The fundamental tension (this is the paper's point):
    - To lower the mean while still attacking, the attacker must emit a mix of
      high (attack) and low (forged-benign) per-edge values. That mixture RAISES
      variance -> z_var fires.
    - To keep variance near sigma^2, the attacker must emit near-uniform values.
      Uniform-high raises the mean -> z_mu fires; uniform-low means NOT attacking.
    The two evasion goals are in direct conflict. T_meanvar_z is robust to the
    extent that this conflict has no simultaneous solution.

Adaptive attacker model (parameterized, white-box on the defense):
    The attacker knows s_b, sigma, and both thresholds. On its E outgoing edges
    it chooses a target emission mean m_target and a target spread. Specifically
    it emits a two-point distribution:
        fraction q of edges at a HIGH value h  (the actual attack signal it wants
                                                to deliver, = s_a),
        fraction (1-q) at a LOW forged value  l  (a fabricated near-benign report),
    then optionally adds a "variance-flattening" correction: it can instead emit
    a CONSTANT value c on all edges (q blended toward a degenerate point mass) to
    minimize variance at the cost of controllability.

    We expose two knobs and sweep them:
      * attack_frac q in [0, 1]   -- how many edges carry the real attack signal
      * forge_target l            -- the fabricated low value on the rest
    For each (q, l) the attacker analytically lands at some (mean, var); we sweep
    the plane and, for each seed/topology, compute T_meanvar_z's detection TPR.

    The attacker's BEST evasion for a given "attack yield" (how much true attack
    signal it delivers, measured as q, since only the q-fraction edges carry the
    payload) is the (q, l) minimizing max(z_mu, z_var). We report, per attack
    yield level, the attacker-optimal evasion TPR = the lowest TPR the attacker
    can force while still delivering that much attack.

Outputs:
    exp3_adaptive_grid.csv     -- full (q, l) plane, per defense, TPR + z-channels
    exp3_adaptive_frontier.csv -- attacker-optimal evasion frontier:
                                  for each attack-yield q, the min achievable TPR
                                  and which channel bound it.
Figure built separately by generate_strengthening_figures.py (fig_adaptive).

Run:
    python exp3_adaptive_adversary.py [--seeds 100] [--smoke]
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_sim import (  # noqa: E402
    s_a, s_b, midpoint,
    gen_er_digraph, assign_roles, compute_metrics,
)
from trust_schemes import T_meanvar_z, T1_mean  # noqa: E402


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def assign_adaptive_signals(edges, roles, defense, rng, q, forge_target,
                            sigma=0.15):
    """Adaptive attacker edge-signal assignment.

    Benign nodes: emit s_b + noise (honest).
    Attacker nodes: on each outgoing edge, emit the true attack signal s_a with
      probability q, else emit the forged low value `forge_target`. Both get the
      same per-edge observation noise sigma. This is the two-point evasion
      distribution the adaptive attacker uses to try to satisfy both channels.
    """
    out_edges = defaultdict(list)
    for (i, j) in edges:
        out_edges[i].append((i, j))

    sa = s_a(defense)
    sb = s_b(defense)
    sig = {}
    for i, elist in out_edges.items():
        E = len(elist)
        if roles[i] == 0:
            for e in elist:
                sig[e] = _clip01(sb + rng.normal(0, sigma))
        else:
            # attacker: q-fraction of edges carry the real payload s_a,
            # the rest carry the forged near-benign value.
            m = int(round(q * E))
            order = rng.permutation(E)
            atk_set = set(order[:m].tolist())
            for k, e in enumerate(elist):
                base = sa if k in atk_set else forge_target
                sig[e] = _clip01(base + rng.normal(0, sigma))
    return sig


def run_adaptive_cell(defense, scheme_fn, q, forge_target, N=20, p_edge=0.3,
                      rho_atk=0.30, sigma=0.15, n_seeds=100, base_seed=0):
    """Run one (q, forge_target) attacker configuration; return TPR@FPR=1% plus
    the mean z_mu / z_var the attacker achieves (diagnostic: which channel
    catches it)."""
    tprs = []
    zmu_list, zvar_list = [], []
    sb_cal = s_b(defense)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        edges = gen_er_digraph(N, p_edge, rng)
        roles = assign_roles(N, rho_atk, rng)
        sig = assign_adaptive_signals(edges, roles, defense, rng, q,
                                      forge_target, sigma=sigma)
        scores = scheme_fn(edges, sig, N)
        m = compute_metrics(scores, roles, midpoint(defense))
        tprs.append(m["tpr_at_fpr01"])

        # Diagnostic: recompute the attacker's own z-channels (mean over attackers)
        by_src = defaultdict(list)
        for (i, j) in edges:
            by_src[i].append(sig[(i, j)])
        amu, avar = [], []
        for i in range(N):
            if roles[i] == 1 and len(by_src[i]) >= 2:
                a = np.asarray(by_src[i])
                k = len(a)
                z_mu = (a.mean() - sb_cal) / (sigma / np.sqrt(k))
                z_var = (a.var(ddof=1) - sigma ** 2) / \
                        np.sqrt(2 * sigma ** 4 / (k - 1))
                amu.append(z_mu)
                avar.append(z_var)
        if amu:
            zmu_list.append(np.mean(amu))
            zvar_list.append(np.mean(avar))

    tpr = float(np.mean(tprs))
    tpr_ci = 1.96 * np.std(tprs, ddof=1) / np.sqrt(len(tprs)) if len(tprs) > 1 else 0.0
    return {
        "tpr_mean": tpr,
        "tpr_ci95": tpr_ci,
        "z_mu_mean": float(np.mean(zmu_list)) if zmu_list else float("nan"),
        "z_var_mean": float(np.mean(zvar_list)) if zvar_list else float("nan"),
    }


def build_adaptive_grid(out_grid, out_frontier, n_seeds=100, smoke=False):
    # Defenses to test: D5 (the inversion-regime defense, most interesting) plus
    # D3 (noisiest) and D1 (cleanest) to show generality. Smoke = D5 only.
    defenses = ["D5_Stochastic"] if smoke else \
        ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]

    # The (q, forge_target) plane.
    # q = fraction of edges carrying the real attack payload s_a.
    # forge_target = the fabricated low value on the remaining edges.
    if smoke:
        q_vals = [0.2, 0.5, 0.8]
        forge_vals = [0.0, 0.2]
        n_seeds = 12
    else:
        q_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        forge_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    rows = []
    total = len(defenses) * len(q_vals) * len(forge_vals)
    done = 0
    _last_pct = -1
    for defense in defenses:
        sb_cal = s_b(defense)
        sigma = 0.15
        Tmvz = lambda e, s, N: T_meanvar_z(e, s, N, sb_cal, sigma)
        for q in q_vals:
            for l in forge_vals:
                r = run_adaptive_cell(defense, Tmvz, q, l, sigma=sigma,
                                      n_seeds=n_seeds)
                binding = ("variance" if r["z_var_mean"] > r["z_mu_mean"]
                           else "mean")
                # Attack-capability accounting so the stealth/rate tradeoff is IN
                # the data, not just narrated. At attacker fraction rho_atk on N
                # nodes with ~p_edge*(N-1) out-edges each, delivered payload scales
                # with q. We record q directly and a normalized "attack_yield"
                # (q relative to full-attack q=1.0) so a reader can weight TPR by
                # how much the attacker actually accomplished.
                attack_yield = q  # fraction of attacker edges carrying real payload
                rows.append({
                    "defense": defense, "q_attack_frac": q, "forge_target": l,
                    "attack_yield": attack_yield,
                    "tpr_mean": round(r["tpr_mean"], 6),
                    "tpr_ci95": round(r["tpr_ci95"], 6),
                    "z_mu_mean": round(r["z_mu_mean"], 4),
                    "z_var_mean": round(r["z_var_mean"], 4),
                    "binding_channel": binding,
                    # evasion is only "effective" if attacker both evades AND
                    # attacks meaningfully; flag it for easy filtering downstream
                    "effective_evasion": bool(r["tpr_mean"] < 0.5 and q >= 0.2),
                })
                done += 1
                pct = int(100 * done / total)
                if pct != _last_pct and pct % 10 == 0:
                    print(f"  ... {pct}% ({done}/{total} cells)")
                    _last_pct = pct

    with open(out_grid, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_grid}  ({len(rows)} cells)")

    # ---- Attacker-optimal evasion frontier ----
    # For each (defense, q), find the forge_target that MINIMIZES TPR (best evasion
    # for that attack yield). That's the attacker's best move; the resulting TPR is
    # what the defense actually guarantees at that attack level.
    frontier = []
    by_dq = defaultdict(list)
    for r in rows:
        by_dq[(r["defense"], r["q_attack_frac"])].append(r)
    for (defense, q), cells in sorted(by_dq.items()):
        best = min(cells, key=lambda c: c["tpr_mean"])  # attacker minimizes TPR
        frontier.append({
            "defense": defense, "q_attack_frac": q,
            "best_evasion_tpr": best["tpr_mean"],
            "best_evasion_tpr_ci95": best["tpr_ci95"],
            "optimal_forge_target": best["forge_target"],
            "binding_channel": best["binding_channel"],
            "z_mu": best["z_mu_mean"], "z_var": best["z_var_mean"],
        })

    with open(out_frontier, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frontier[0].keys())
        w.writeheader()
        w.writerows(frontier)
    print(f"  -> {out_frontier}  ({len(frontier)} frontier points)")

    # Console summary: one compact, aligned table. Everything else is in the CSVs.
    yield_path = out_frontier.replace("frontier", "yield_frontier")
    with open(yield_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frontier[0].keys())
        w.writeheader()
        w.writerows(frontier)

    print("\n" + "=" * 78)
    print("  ADAPTIVE-ADVERSARY SUMMARY — attacker-optimal evasion vs T_meanvar_z")
    print("=" * 78)
    print(f"  {'Defense':<15}{'min TPR':>9}{'@ q':>6}{'q>=.2 floor':>13}"
          f"{'@ q':>6}{'@forge':>8}{'bound by':>11}")
    print("  " + "-" * 74)
    for defense in defenses:
        pts = [f for f in frontier if f["defense"] == defense]
        gmin = min(pts, key=lambda p: p["best_evasion_tpr"])
        meaningful = [p for p in pts if p["q_attack_frac"] >= 0.2]
        mfloor = min(meaningful, key=lambda p: p["best_evasion_tpr"])
        print(f"  {defense:<15}{gmin['best_evasion_tpr']:>9.3f}"
              f"{gmin['q_attack_frac']:>6.1f}{mfloor['best_evasion_tpr']:>13.3f}"
              f"{mfloor['q_attack_frac']:>6.1f}{mfloor['optimal_forge_target']:>8.2f}"
              f"{mfloor['binding_channel']:>11}")
    print("  " + "-" * 74)
    print("  min TPR = global best evasion (attacker throttles attack yield to reach it)")
    print("  q>=.2 floor = best evasion while still attacking >=20% of edges (the")
    print("               number that characterizes the defense under meaningful attack)")
    print("  Full per-cell data -> exp3_adaptive_grid.csv")
    print("  Per-yield frontier -> " + os.path.basename(yield_path))
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.smoke:
        print("=== SMOKE MODE ===")
    print("EXPERIMENT 3: adaptive adversary vs T_meanvar_z")
    build_adaptive_grid(
        os.path.join(here, "exp3_adaptive_grid.csv"),
        os.path.join(here, "exp3_adaptive_frontier.csv"),
        n_seeds=args.seeds, smoke=args.smoke,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
