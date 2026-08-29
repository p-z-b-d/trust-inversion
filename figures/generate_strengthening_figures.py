"""
generate_strengthening_figures.py
==================================
Two figures for the USENIX strengthening pass, built from the real experiment
CSVs (exp1_N_sweep.csv, exp2_dual_sigma.csv). Style matches
generate_figures_v2.py (same palette family, fonts, dpi).

FIG A — N-persistence (fig_N_persistence.png)
  TPR@FPR=1% vs network size N, one line per attack regime, with 95% CI bands.
  The story the figure tells at a glance:
    * inversion (D5+T1) sits at ~0 and STAYS there as N grows (deepens) —
      the vulnerability scale does NOT cure;
    * dilution & mimicry CLIMB toward 1.0 as N grows — scale DOES cure them
      (more edges tighten the detection margin);
    * the defended inversion cell (D5+T_meanvar_z) stays pinned at ~1.0.
  => The honest, stronger claim: inversion is the serious attack precisely
     because it is the one network scale cannot fix.

FIG B — Noise-independence (fig_noise_independence.png)
  Grouped bars, measured-sigma vs stress-sigma, for the D5 inversion story
  plus the easy baselines, showing:
    * AP1 / AP5 detection is trivial at BOTH sigmas (so single-edge detection
      is not the problem);
    * AP6c inversion on T1 fails at BOTH sigmas (0.22 measured, 0.03 stress) —
      the vulnerability is not manufactured by inflating sigma;
    * T_meanvar_z recovers to ~0.99 at BOTH sigmas.
  A small second panel generalizes across all six defenses for the inversion
  cell, so it's not a D5-only artifact.

Run:
    python generate_strengthening_figures.py
Outputs (to the script's own directory):
    fig_N_persistence.png
    fig_noise_independence.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe on headless / Windows without display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 200

# Palette — consistent, colorblind-friendly, matches the paper's tone.
C_INVERSION = "#a50026"   # deep red — the dangerous, uncured attack
C_DILUTION  = "#4575b4"   # blue
C_MIMICRY   = "#f46d43"   # orange
C_DEFENDED  = "#1a9850"   # green — the fix
C_MEASURED  = "#4575b4"   # blue bars
C_STRESS    = "#d73027"   # red bars

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
})


# ============================================================================
# FIG A — N-persistence
# ============================================================================
def fig_N_persistence(csv_path, out_path):
    df = pd.read_csv(csv_path)

    # Each cell label -> display name + color + marker
    series = [
        ("AP6c_inversion__D5_T1",   "Inversion (D5 + T1 mean)",            C_INVERSION, "o", "-"),
        ("AP4_dilution__D3_T1",     "Dilution (D3 + T1 mean)",             C_DILUTION,  "s", "--"),
        ("AP5_mimicry__D3_T1",      "Mimicry (D3 + T1 mean)",              C_MIMICRY,   "^", "--"),
        ("AP6c_inversion__D5_Tmvz", "Inversion, defended (D5 + T_meanvar_z)", C_DEFENDED, "D", "-"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for cell, label, color, marker, ls in series:
        sub = df[df["cell"] == cell].sort_values("N")
        if sub.empty:
            continue
        N = sub["N"].values
        tpr = sub["tpr_mean"].values
        ci = sub["tpr_ci95"].values
        ax.plot(N, tpr, marker=marker, ls=ls, color=color, lw=2,
                ms=7, label=label, zorder=3)
        ax.fill_between(N, tpr - ci, tpr + ci, color=color, alpha=0.15, zorder=2)

    # Shade the "detection floor" region to make the inversion story pop
    ax.axhspan(0, 0.10, color="#a50026", alpha=0.05, zorder=1)
    ax.text(205, 0.05, "detection\nfloor", fontsize=8.5, color="#a50026",
            va="center", ha="left", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200])
    ax.set_xticklabels([20, 50, 100, 200])
    ax.set_xlim(18, 260)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Network size $N$ (nodes, log scale)")
    ax.set_ylabel("Detection TPR @ FPR = 1%")
    ax.set_title("Attack persistence across network scale\n"
                 "(σ = 0.15, 100 seeds/point, 95% CI bands)", pad=12)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(loc="center right", framealpha=0.95, edgecolor="0.8")

    # Annotation calling out the key contrast
    ax.annotate("scale CURES these\n(margin tightens)",
                xy=(100, 1.0), xytext=(38, 0.62),
                fontsize=9, color="0.35", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.6", lw=1.2))
    ax.annotate("scale does NOT cure\ninversion — it deepens",
                xy=(200, 0.0), xytext=(60, 0.20),
                fontsize=9, color=C_INVERSION, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_INVERSION, lw=1.2))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ============================================================================
# FIG B — Noise-independence
# ============================================================================
def fig_noise_independence(csv_path, out_path):
    df = pd.read_csv(csv_path)

    # ---- Panel 1: D5 focus, the four telling profiles ----
    prof_order = ["AP1", "AP5_p10", "AP6c_active", "AP6c_active_Tmvz"]
    prof_labels = ["AP1\n(baseline)", "AP5\n(mimicry)",
                   "AP6c inversion\n(T1 mean)", "AP6c inversion\n(T_meanvar_z)"]

    d5 = df[df["defense"] == "D5_Stochastic"]
    meas = [d5[(d5["profile"] == p) & (d5["regime"] == "measured")]["tpr_mean"].values for p in prof_order]
    strs = [d5[(d5["profile"] == p) & (d5["regime"] == "stress")]["tpr_mean"].values for p in prof_order]
    meas = [float(x[0]) if len(x) else np.nan for x in meas]
    strs = [float(x[0]) if len(x) else np.nan for x in strs]
    meas_ci = [float(d5[(d5["profile"] == p) & (d5["regime"] == "measured")]["tpr_ci95"].values[0]) for p in prof_order]
    strs_ci = [float(d5[(d5["profile"] == p) & (d5["regime"] == "stress")]["tpr_ci95"].values[0]) for p in prof_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.arange(len(prof_order))
    w = 0.38
    ax1.bar(x - w/2, meas, w, yerr=meas_ci, capsize=3, color=C_MEASURED,
            label="measured σ ≈ 0.0095 (realistic)", edgecolor="white", lw=0.5)
    ax1.bar(x + w/2, strs, w, yerr=strs_ci, capsize=3, color=C_STRESS,
            label="stress σ = 0.15", edgecolor="white", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(prof_labels)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Detection TPR @ FPR = 1%")
    ax1.set_title("D5 Stochastic: the inversion survives at realistic noise", pad=10)
    ax1.axhspan(0, 0.10, color="#a50026", alpha=0.05)
    ax1.legend(loc="lower left", framealpha=0.95, edgecolor="0.8")
    ax1.grid(True, axis="y", alpha=0.25)

    # value labels
    for xi, m, s in zip(x, meas, strs):
        ax1.text(xi - w/2, m + 0.03, f"{m:.2f}", ha="center", fontsize=8.5, color=C_MEASURED)
        ax1.text(xi + w/2, s + 0.03, f"{s:.2f}", ha="center", fontsize=8.5, color=C_STRESS)

    # ---- Panel 2: generalization across all 6 defenses (inversion cell) ----
    defenses = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
    dlabels = ["D1", "D2", "D3", "D4", "D5", "D6"]
    inv_meas, inv_strs = [], []
    for d in defenses:
        sub = df[(df["defense"] == d) & (df["profile"] == "AP6c_active")]
        inv_meas.append(float(sub[sub["regime"] == "measured"]["tpr_mean"].values[0]))
        inv_strs.append(float(sub[sub["regime"] == "stress"]["tpr_mean"].values[0]))

    xd = np.arange(len(defenses))
    ax2.bar(xd - w/2, inv_meas, w, color=C_MEASURED, edgecolor="white", lw=0.5,
            label="measured σ")
    ax2.bar(xd + w/2, inv_strs, w, color=C_STRESS, edgecolor="white", lw=0.5,
            label="stress σ = 0.15")
    ax2.set_xticks(xd)
    ax2.set_xticklabels(dlabels)
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("Defense")
    ax2.set_title("Inversion (AP6c) on T1 mean:\nonly D5 is in the inversion regime ($R_D < k_{in}/k_{out}$)",
                  pad=10, fontsize=11)
    ax2.axhspan(0, 0.10, color="#a50026", alpha=0.05)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="lower right", framealpha=0.95, edgecolor="0.8")

    # highlight D5 as the one that drops
    ax2.annotate("only D5 inverts", xy=(4, inv_strs[4]), xytext=(1.4, 0.42),
                 fontsize=9, color=C_INVERSION,
                 arrowprops=dict(arrowstyle="->", color=C_INVERSION, lw=1.2))

    fig.suptitle("Noise-independence of the composition inversion "
                 "(measured per-defense σ vs stress-test σ)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    exp1 = os.path.join(HERE, "exp1_N_sweep.csv")
    exp2 = os.path.join(HERE, "exp2_dual_sigma.csv")

    print("Building strengthening figures...")
    if os.path.exists(exp1):
        fig_N_persistence(exp1, os.path.join(HERE, "fig_N_persistence.png"))
    else:
        print(f"  ! missing {exp1}")
    if os.path.exists(exp2):
        fig_noise_independence(exp2, os.path.join(HERE, "fig_noise_independence.png"))
    else:
        print(f"  ! missing {exp2}")
    print("Done.")


if __name__ == "__main__":
    main()
