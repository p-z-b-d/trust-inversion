"""D5 Stochastic investigation: signal-inversion mechanism under active AP6c.

Under forced-clique active manipulation, an attacker's mean signal is pulled
below the benign mean for any defense satisfying
    (s_a - s_b)/s_b < k_in/k_out
where k_in is the forced-clique in-degree (~5) and k_out is the ER
out-of-clique degree (~p_edge*(N - N_atk) ~ 4.2). Only D5 meets this criterion.

This script:
  1. Sweeps the inclique parameter from 0 to s_b on D5 and on D3 (control).
  2. Records attacker mean, benign mean, and TPR@FPR1% for T1, T_max, T_meanvar_z.
  3. Computes the theoretical crossover inclique value analytically.
  4. Generates fig_D5_inversion.png.
"""
import sys, os, csv
sys.path.insert(0, "/mnt/user-data/outputs")
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phase4_sim import (gen_er_digraph, assign_roles, assign_edge_signals,
                        compute_metrics, midpoint, s_b as get_s_b, s_a as get_s_a)
from trust_schemes import T1_mean, T_max, make_T_meanvar_z

OUT_DIR = "/mnt/user-data/outputs"


def run_clique_with_raw_scores(defense, scheme, inclique, N=20, p_edge=0.3,
                               rho_atk=0.3, sigma=0.15, n_seeds=200):
    """Returns mean attacker score, mean benign score, TPR@FPR1%."""
    T_det = midpoint(defense)
    atk_scores, ben_scores, tprs = [], [], []
    metrics_acc = defaultdict(list)
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        edges = gen_er_digraph(N, p_edge, rng)
        roles = assign_roles(N, rho_atk, rng)
        attackers = [i for i in range(N) if roles[i] == 1]
        existing = set(edges)
        for i in attackers:
            for j in attackers:
                if i != j and (i, j) not in existing:
                    edges.append((i, j))
        sig = assign_edge_signals(edges, roles, defense, "AP6", rng,
                                  sigma=sigma, inclique=inclique)
        scores = scheme(edges, sig, N)
        scored = ~np.isnan(scores)
        if scored.any():
            attacker_mask = (roles == 1) & scored
            benign_mask   = (roles == 0) & scored
            if attacker_mask.any():
                atk_scores.append(np.mean(scores[attacker_mask]))
            if benign_mask.any():
                ben_scores.append(np.mean(scores[benign_mask]))
        m = compute_metrics(scores, roles, T_det)
        for k, v in m.items():
            if not (isinstance(v, float) and np.isnan(v)):
                metrics_acc[k].append(v)
    return {
        "atk_mean": float(np.mean(atk_scores)) if atk_scores else np.nan,
        "ben_mean": float(np.mean(ben_scores)) if ben_scores else np.nan,
        "tpr":      float(np.mean(metrics_acc.get("tpr_at_fpr01", [np.nan]))),
    }


def theoretical_crossover(defense, k_in=5, k_out=4.2):
    """Inclique value at which attacker T1 mean == benign mean."""
    s_a, s_b = get_s_a(defense), get_s_b(defense)
    # Solve: (k_in*inclique + k_out*s_a) / (k_in+k_out) = s_b
    # => inclique = (s_b*(k_in+k_out) - k_out*s_a) / k_in
    return (s_b * (k_in + k_out) - k_out * s_a) / k_in


def main():
    SIGMA = 0.15
    inclique_values = np.linspace(0.0, 1.0, 21)

    rows = []
    for defense in ["D5_Stochastic", "D3_RHMD"]:
        sbv, sav = get_s_b(defense), get_s_a(defense)
        Tmv = make_T_meanvar_z(sbv, SIGMA)
        schemes = [("T1_mean", T1_mean), ("T_max", T_max), ("T_meanvar_z", Tmv)]
        x_cross = theoretical_crossover(defense)
        print(f"\n{defense}: s_a={sav:.3f} s_b={sbv:.3f}  "
              f"theoretical T1 crossover inclique = {x_cross:.3f}")
        for inclique in inclique_values:
            for sname, sch in schemes:
                r = run_clique_with_raw_scores(defense, sch, float(inclique),
                                                sigma=SIGMA, n_seeds=150)
                rows.append({"defense": defense, "scheme": sname,
                             "inclique": float(inclique),
                             "atk_mean": r["atk_mean"], "ben_mean": r["ben_mean"],
                             "tpr": r["tpr"], "x_cross_theory": x_cross,
                             "s_a": sav, "s_b": sbv})

    csv_path = f"{OUT_DIR}/phase4_D5_inversion.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nSaved -> {csv_path}")

    # ---- Figure: 2 rows (D5, D3) x 2 cols (mean scores, TPR) ----------------
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for r, defense in enumerate(["D5_Stochastic", "D3_RHMD"]):
        sub = df[df["defense"] == defense]
        x_cross = sub["x_cross_theory"].iloc[0]
        sb = sub["s_b"].iloc[0]

        # Left: mean scores
        ax = axes[r, 0]
        t1sub = sub[sub["scheme"] == "T1_mean"].sort_values("inclique")
        ax.plot(t1sub["inclique"], t1sub["atk_mean"], marker="o",
                color="#d62728", lw=1.8, label="attacker T1 mean")
        ax.plot(t1sub["inclique"], t1sub["ben_mean"], marker="o",
                color="#1f77b4", lw=1.8, label="benign T1 mean")
        ax.axhline(sb, color="gray", lw=0.8, ls="--", alpha=0.6,
                   label=f"s_b={sb:.3f}")
        if 0 <= x_cross <= 1:
            ax.axvline(x_cross, color="red", lw=1.2, ls=":", alpha=0.8,
                       label=f"predicted crossover\n@inclique={x_cross:.3f}")
        ax.set_xlabel("inclique (signal between colluders)")
        ax.set_ylabel("T1 mean score")
        ax.set_title(f"{defense}: signal inversion under active manipulation")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.0)

        # Right: TPR for the three schemes
        ax = axes[r, 1]
        for sname, color, marker in [("T1_mean", "#1f77b4", "o"),
                                       ("T_max", "#ff7f0e", "s"),
                                       ("T_meanvar_z", "#2ca02c", "D")]:
            ss = sub[sub["scheme"] == sname].sort_values("inclique")
            ax.plot(ss["inclique"], ss["tpr"], marker=marker, color=color,
                    lw=1.8, ms=6, label=sname)
        if 0 <= x_cross <= 1:
            ax.axvline(x_cross, color="red", lw=1.2, ls=":", alpha=0.8,
                       label=f"T1 inversion @inclique={x_cross:.3f}")
        ax.set_xlabel("inclique (signal between colluders)")
        ax.set_ylabel("TPR @ FPR=1%")
        ax.set_title(f"{defense}: detection across the inclique sweep")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("D5 Stochastic signal inversion: high-FPR detectors are "
                 "uniquely vulnerable to active forced-clique AP6\n"
                 "T_meanvar_z's variance channel is invariant to the inversion",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig_D5_inversion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {OUT_DIR}/fig_D5_inversion.png")


if __name__ == "__main__":
    main()
