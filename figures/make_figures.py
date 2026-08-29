"""
Phase 4 figure generator. Reads grid_driver CSVs and produces paper figures.

Outputs (to /mnt/user-data/outputs/):
  composition_heatmap.png/pdf  - headline figure: D x T x AP heatmap
  sigma_sweep.png/pdf          - Theorem 2 validation: TPR vs sigma on AP5
  N_invariance.png/pdf         - Corollary 1.1 validation: TPR vs N

Color conventions:
  Green (high TPR)  = scheme detects attackers
  Red (low TPR)     = scheme broken / unable to detect
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Explicit ordering for figures (matches paper narrative)
DEFENSE_ORDER = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
SCHEME_ORDER = ["T1_mean", "T_median", "T_max", "T_meanvar_z",
                "T4_EigenTrust", "T5_TFLDT"]
ATTACK_ORDER = ["AP1", "AP4_phi30", "AP5_full", "AP6c_active"]
ATTACK_DISPLAY = {
    "AP1":         "AP1 (full attack)",
    "AP4_phi30":   "AP4 dilution (φ=0.30)",
    "AP5_full":    "AP5 mimicry (φ=1, p=10%)",
    "AP6c_active": "AP6 collusion (clique, active)",
}
SCHEME_DISPLAY = {
    "T1_mean":       "T1 mean",
    "T_median":      "T_median",
    "T_max":         "T_max",
    "T_meanvar_z":   "T_meanvar_z*",
    "T4_EigenTrust": "T4 EigenTrust",
    "T5_TFLDT":      "T5 TFL-DT",
}

# Use Arial-ish stack and slightly larger DPI default
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 110,
})


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _ordered_present(values, order):
    """Subset of `order` that actually appears in `values`, preserving order."""
    present = set(values)
    return [x for x in order if x in present]


# =============================================================================
# Figure 1 — Composition heatmap (headline)
# =============================================================================

def fig_composition_heatmap(rows, out_stem):
    defenses = _ordered_present({r['defense'] for r in rows}, DEFENSE_ORDER)
    schemes  = _ordered_present({r['scheme']  for r in rows}, SCHEME_ORDER)
    attacks  = _ordered_present({r['attack']  for r in rows}, ATTACK_ORDER)

    fig, axes = plt.subplots(1, len(attacks),
                              figsize=(2.8 * len(attacks) + 0.7, 0.7 * len(defenses) + 1.6),
                              sharey=True)
    if len(attacks) == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn  # red for low TPR (broken), green for high (working)
    im = None
    for ax, ap in zip(axes, attacks):
        mat = np.full((len(defenses), len(schemes)), np.nan)
        for r in rows:
            if r['attack'] == ap:
                d = defenses.index(r['defense'])
                s = schemes.index(r['scheme'])
                v = r['TPR_at_FPR1']
                if v not in (None, "", "nan"):
                    try:
                        mat[d, s] = float(v)
                    except (TypeError, ValueError):
                        pass
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels([SCHEME_DISPLAY.get(s, s) for s in schemes],
                            rotation=40, ha='right')
        if ax is axes[0]:
            ax.set_yticks(range(len(defenses)))
            ax.set_yticklabels(defenses)
        ax.set_title(ATTACK_DISPLAY.get(ap, ap), pad=6)
        # cell annotations
        for i in range(len(defenses)):
            for j in range(len(schemes)):
                if not np.isnan(mat[i, j]):
                    color = 'white' if mat[i, j] < 0.45 else 'black'
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                            fontsize=7, color=color)
        # thin grid
        ax.set_xticks(np.arange(-0.5, len(schemes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(defenses), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.8)
        ax.tick_params(which='minor', length=0)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, label='TPR @ FPR=1%',
                         pad=0.02, fraction=0.04)
    fig.suptitle("Composition heatmap: detection vs attack profile, by (defense, trust scheme).\n"
                 "T_meanvar_z* is the proposed two-statistic defense.",
                 fontsize=11, y=1.02)
    plt.savefig(f"{out_stem}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{out_stem}.pdf", bbox_inches='tight')
    print(f"Saved {out_stem}.png/.pdf")
    plt.close()


# =============================================================================
# Figure 2 — sigma sweep (Theorem 2 validation)
# =============================================================================

def fig_sigma_sweep(rows, out_stem):
    by_d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_d[r['defense']][r['scheme']].append((float(r['sigma']),
                                                  float(r['TPR_at_FPR1'])))
    defenses = _ordered_present(by_d.keys(), DEFENSE_ORDER)
    schemes = _ordered_present(
        {s for d in defenses for s in by_d[d]}, SCHEME_ORDER)

    fig, axes = plt.subplots(1, len(defenses),
                              figsize=(3.4 * len(defenses), 3.3),
                              sharey=True)
    if len(defenses) == 1:
        axes = [axes]

    color_map = {"T1_mean": "#1f77b4", "T_max": "#ff7f0e", "T_meanvar_z": "#2ca02c"}
    marker_map = {"T1_mean": "o", "T_max": "s", "T_meanvar_z": "^"}

    # Theorem-2 predicted sigma* from PHASE3 signals
    # sigma_star = (s_m10 - s_b) * sqrt(deg) / z_FPR ; deg=5.7, z=2.33
    sigma_star = {
        "D3_RHMD":      (0.3881 - 0.1980) * np.sqrt(5.7) / 2.33,  # ≈ 0.195
        "D6_DRL":       (0.5397 - 0.0833) * np.sqrt(5.7) / 2.33,  # ≈ 0.467
        "D2_CNN":       (0.8362 - 0.0278) * np.sqrt(5.7) / 2.33,  # ≈ 0.827
    }

    for ax, defense in zip(axes, defenses):
        for scheme in schemes:
            pts = sorted(by_d[defense][scheme])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, label=SCHEME_DISPLAY.get(scheme, scheme),
                    marker=marker_map.get(scheme, "o"),
                    color=color_map.get(scheme, None), linewidth=1.4)
        # Theorem 2 sigma* prediction
        if defense in sigma_star:
            ax.axvline(sigma_star[defense], linestyle='--', color='gray',
                        linewidth=1, alpha=0.7,
                        label=f"$\\sigma^*_{{Thm.2}}$={sigma_star[defense]:.2f}")
        ax.set_title(defense, pad=4)
        ax.set_xlabel(r"per-edge noise $\sigma$")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("TPR @ FPR=1%")
        ax.legend(fontsize=7, loc='lower left')

    fig.suptitle("Theorem 2 validation: AP5 mimicry detection vs per-edge noise. "
                 "Vertical dashed lines mark predicted $\\sigma^*$.",
                 fontsize=10, y=1.04)
    plt.tight_layout()
    plt.savefig(f"{out_stem}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{out_stem}.pdf", bbox_inches='tight')
    print(f"Saved {out_stem}.png/.pdf")
    plt.close()


# =============================================================================
# Figure 3 — N invariance (Corollary 1.1)
# =============================================================================

def fig_N_invariance(rows, out_stem):
    by_attack = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_attack[r['attack']][r['scheme']].append((int(r['N']),
                                                     float(r['TPR_at_FPR1'])))
    attacks = _ordered_present(by_attack.keys(), ATTACK_ORDER)
    schemes = _ordered_present(
        {s for a in attacks for s in by_attack[a]}, SCHEME_ORDER)

    fig, axes = plt.subplots(1, len(attacks),
                              figsize=(2.7 * len(attacks) + 0.5, 3.3),
                              sharey=True)
    if len(attacks) == 1:
        axes = [axes]

    color_map = {"T1_mean": "#1f77b4", "T_meanvar_z": "#2ca02c",
                 "T4_EigenTrust": "#d62728"}

    for ax, ap in zip(axes, attacks):
        for scheme in schemes:
            pts = sorted(by_attack[ap][scheme])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, label=SCHEME_DISPLAY.get(scheme, scheme),
                    marker='o', color=color_map.get(scheme, None),
                    linewidth=1.4)
        ax.set_title(ATTACK_DISPLAY.get(ap, ap), pad=4)
        ax.set_xlabel("N (network size)")
        ax.set_xscale('log')
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3, which='both')
        if ax is axes[0]:
            ax.set_ylabel("TPR @ FPR=1%")
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle("N-invariance check on D3 RHMD. T1/T_meanvar_z stable across N; "
                 "T4 EigenTrust degrades structurally on AP4/AP5.",
                 fontsize=10, y=1.04)
    plt.tight_layout()
    plt.savefig(f"{out_stem}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{out_stem}.pdf", bbox_inches='tight')
    print(f"Saved {out_stem}.png/.pdf")
    plt.close()


if __name__ == "__main__":
    results_dir = Path("/tmp/grid_results")
    out_dir = Path("/mnt/user-data/outputs")

    main_csv = results_dir / "main_grid.csv"
    sigma_csv = results_dir / "sigma_sweep.csv"
    N_csv = results_dir / "N_sweep.csv"

    if main_csv.exists():
        fig_composition_heatmap(load_csv(main_csv),
                                 str(out_dir / "composition_heatmap"))
    if sigma_csv.exists():
        fig_sigma_sweep(load_csv(sigma_csv),
                         str(out_dir / "sigma_sweep"))
    if N_csv.exists():
        fig_N_invariance(load_csv(N_csv),
                          str(out_dir / "N_invariance"))
