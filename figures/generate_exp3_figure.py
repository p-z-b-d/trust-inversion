"""
generate_exp3_figure.py
=======================
Adaptive-adversary figure for the T_meanvar_z defense section (§7).

Reads:
    exp3_adaptive_yield_frontier.csv   (per-(defense,q) attacker-optimal evasion)
    exp3_adaptive_grid.csv             (full q x forge plane, z-channels)

Produces:
    fig_adaptive_adversary.png

Two panels:
  LEFT — The stealth/rate tradeoff frontier.
    x = attack yield q (fraction of edges carrying real payload)
    y = attacker-optimal detection TPR at that yield (the BEST the attacker can do)
    one line per defense. The shaded band q<0.2 marks the "throttled attack"
    regime where evasion is only possible by nearly not attacking. The message:
    to push TPR down, the attacker must move LEFT (attack less). At meaningful
    yield (q>=0.2) every defense holds high.

  RIGHT — Why: the binding channel across the (q, forge) plane for the
    inversion-regime defense (D5). Cells colored by which z-channel catches the
    attacker (variance vs mean). Variance dominates almost everywhere; the
    attacker only escapes it by driving q so low the attack is negligible.

Run:  python generate_exp3_figure.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 200

DEF_COLORS = {
    "D1_RF": "#4575b4", "D2_CNN": "#762a83", "D3_RHMD": "#f46d43",
    "D4_MTD": "#5aae61", "D5_Stochastic": "#a50026", "D6_DRL": "#d6a000",
}
DEF_LABELS = {
    "D1_RF": "D1 RF", "D2_CNN": "D2 CNN", "D3_RHMD": "D3 RHMD",
    "D4_MTD": "D4 MTD", "D5_Stochastic": "D5 Stochastic", "D6_DRL": "D6 DRL",
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": DPI,
})


def fig_adaptive(frontier_csv, grid_csv, out_path, focus_defense="D5_Stochastic"):
    fr = pd.read_csv(frontier_csv)
    grid = pd.read_csv(grid_csv)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- LEFT: stealth/rate tradeoff frontier ----
    defenses = [d for d in DEF_COLORS if d in fr["defense"].unique()]
    for d in defenses:
        sub = fr[fr["defense"] == d].sort_values("q_attack_frac")
        q = sub["q_attack_frac"].values
        tpr = sub["best_evasion_tpr"].values
        ci = sub["best_evasion_tpr_ci95"].values if "best_evasion_tpr_ci95" in sub else np.zeros_like(tpr)
        axL.plot(q, tpr, marker="o", ms=6, lw=2, color=DEF_COLORS[d],
                 label=DEF_LABELS[d], zorder=3)
        axL.fill_between(q, tpr - ci, tpr + ci, color=DEF_COLORS[d],
                         alpha=0.12, zorder=2)

    # shade the throttled-attack regime q < 0.2
    axL.axvspan(0.0, 0.2, color="0.5", alpha=0.12, zorder=1)
    axL.text(0.1, 0.06, "throttled attack\n(evasion only by\nnearly not attacking)",
             fontsize=8.3, ha="center", va="bottom", color="0.35")
    # detection floor band
    axL.axhspan(0, 0.5, color="#a50026", alpha=0.04, zorder=0)

    axL.set_xlim(0.05, 1.02)
    axL.set_ylim(0, 1.05)
    axL.set_xlabel("Attack yield $q$ (fraction of edges carrying real payload)")
    axL.set_ylabel("Attacker-optimal detection TPR @ FPR = 1%")
    axL.set_title("Stealth / rate tradeoff: to evade, the attacker must attack less",
                  pad=10)
    axL.grid(True, axis="y", alpha=0.25)
    axL.legend(loc="lower right", framealpha=0.95, edgecolor="0.8", ncol=2)

    axL.annotate("meaningful-attack regime:\nall defenses hold high",
                 xy=(0.6, 0.99), xytext=(0.42, 0.62), fontsize=9, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))

    # ---- RIGHT: binding-channel map for the focus (inversion-regime) defense ----
    g = grid[grid["defense"] == focus_defense].copy()
    q_vals = sorted(g["q_attack_frac"].unique())
    f_vals = sorted(g["forge_target"].unique())
    # matrix: 1 = variance-bound (caught by var channel), 0 = mean-bound
    M = np.zeros((len(f_vals), len(q_vals)))
    T = np.zeros((len(f_vals), len(q_vals)))
    for iy, fv in enumerate(f_vals):
        for ix, qv in enumerate(q_vals):
            cell = g[(g["q_attack_frac"] == qv) & (g["forge_target"] == fv)]
            if not cell.empty:
                M[iy, ix] = 1.0 if cell["binding_channel"].iloc[0] == "variance" else 0.0
                T[iy, ix] = cell["tpr_mean"].iloc[0]

    cmap = ListedColormap(["#fee08b", "#4575b4"])  # mean=yellow, variance=blue
    axR.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
    # overlay TPR as text
    for iy in range(len(f_vals)):
        for ix in range(len(q_vals)):
            axR.text(ix, iy, f"{T[iy, ix]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="white" if M[iy, ix] == 1 else "0.2")
    axR.set_xticks(range(len(q_vals)))
    axR.set_xticklabels([f"{q:.1f}" for q in q_vals])
    axR.set_yticks(range(len(f_vals)))
    axR.set_yticklabels([f"{f:.2f}" for f in f_vals])
    axR.set_xlabel("Attack yield $q$")
    axR.set_ylabel("Forge target (fabricated low value)")
    axR.set_title(f"{DEF_LABELS[focus_defense]}: binding channel + TPR per strategy\n"
                  "(blue = variance channel catches it, yellow = mean channel)",
                  pad=10, fontsize=11)

    fig.suptitle("Adaptive white-box adversary vs T_meanvar_z: "
                 "the variance channel forces a stealth/rate tradeoff",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    fr = os.path.join(HERE, "exp3_adaptive_yield_frontier.csv")
    grid = os.path.join(HERE, "exp3_adaptive_grid.csv")
    if not (os.path.exists(fr) and os.path.exists(grid)):
        print("  ! run exp3_adaptive_adversary.py first (need the CSVs)")
        return
    print("Building adaptive-adversary figure...")
    fig_adaptive(fr, grid, os.path.join(HERE, "fig_adaptive_adversary.png"))
    print("Done.")


if __name__ == "__main__":
    main()
