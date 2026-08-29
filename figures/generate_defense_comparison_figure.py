"""
generate_defense_comparison_figure.py
=====================================
The §7 headline figure: worst-case detection per trust scheme, each attacked by
the adversary tuned+verified-optimal FOR THAT SCHEME. Shows every baseline has a
catastrophic failure mode while T_meanvar_z degrades gracefully, at O(k) cost.

Reads: exp3_crosscheck.csv
Produces: fig_defense_comparison.png

Two panels:
  LEFT  — worst-case detection (bars), sorted, colored by whether the scheme
          collapses (<0.2, red) or holds (green). Cost multiple annotated on each
          bar; failure mode labeled. T_meanvar_z stands alone above the cliff line.
  RIGHT — the FPR-honesty scatter: benign_mean (x) vs thr_in_tail (y). Schemes
          jammed against a compressed range / deep tail are flagged. Contextualizes
          WHY the collapsing schemes are fragile.

Run: python generate_defense_comparison_figure.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 200

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": DPI,
})

# scheme display names + known failure modes (for annotation)
DISPLAY = {
    "T1_mean": "T1 mean", "T2_SL": "T2 SL", "T3_Beta": "T3 Beta",
    "T5_TFLDT": "T5 TFL-DT", "T_max": "T_max", "T_median": "T_median",
    "T4_EigenTrust": "T4 EigenTrust", "T_meanvar_z": "T_meanvar_z (ours)",
}
FAILURE = {
    "T1_mean": "dilution", "T2_SL": "dilution", "T3_Beta": "dilution",
    "T5_TFLDT": "dilution", "T_max": "single-edge forgery",
    "T_median": "largest-minority", "T4_EigenTrust": "collapse + 1629× cost",
    "T_meanvar_z": "none — graceful",
}
CLIFF = 0.20  # below this = catastrophic collapse


def build(csv_path, out_path):
    df = pd.read_csv(csv_path)
    defenses = sorted(df["defense"].unique())
    schemes = ["T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT", "T_max",
               "T_median", "T4_EigenTrust", "T_meanvar_z"]

    # worst-case = min tuned_attacker_tpr over defenses (worst defense), per scheme
    wc, cost, benign_mu, thr_tail, bkdn = {}, {}, {}, {}, {}
    for s in schemes:
        sub = df[df["scheme"] == s]
        if sub.empty:
            continue
        per_def_min = [sub[sub["defense"] == d]["tuned_attacker_tpr"].min()
                       for d in defenses]
        wc[s] = float(np.min(per_def_min))
        cost[s] = float(sub["cost_vs_T1"].iloc[0])
        benign_mu[s] = float(sub["benign_mean"].iloc[0])
        thr_tail[s] = float(sub["thr_in_tail_std"].iloc[0])
        # T_median breakdown (its most-favorable evading number), if present
        if "breakdown_tpr" in sub.columns:
            bd = pd.to_numeric(sub["breakdown_tpr"], errors="coerce").dropna()
            bkdn[s] = float(bd.min()) if len(bd) else None
        else:
            bkdn[s] = None

    order = sorted(wc, key=lambda s: wc[s])  # worst first
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- LEFT: worst-case detection bars ----
    x = np.arange(len(order))
    vals = [wc[s] for s in order]
    colors = ["#c0392b" if v < CLIFF else "#27ae60" for v in vals]
    bars = axL.bar(x, vals, color=colors, edgecolor="white", width=0.68, zorder=3)

    # cliff line
    axL.axhline(CLIFF, ls="--", color="0.4", lw=1.2, zorder=2)
    axL.text(len(order)-0.4, CLIFF+0.02, "collapse line (0.20)",
             fontsize=8.5, color="0.4", ha="right")

    # annotate cost + failure mode on/under each bar
    for i, s in enumerate(order):
        v = wc[s]
        axL.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        axL.text(i, v + 0.075, f"{cost[s]:.1f}×", ha="center", fontsize=8, color="0.35")
        # failure-mode label rotated under axis
        axL.text(i, -0.13, FAILURE[s], ha="center", va="top", fontsize=7.6,
                 color="0.3", rotation=25)
        # T_median breakdown marker
        if s == "T_median" and bkdn.get(s):
            axL.plot(i, bkdn[s], marker="D", ms=8, color="#e67e22", zorder=4)
            axL.text(i+0.30, bkdn[s], f"breakdown\n{bkdn[s]:.2f}", fontsize=7.5,
                     color="#e67e22", va="center")

    axL.set_xticks(x)
    axL.set_xticklabels([DISPLAY[s] for s in order], rotation=20, ha="right")
    axL.set_ylim(-0.02, 1.05)
    axL.set_ylabel("Worst-case detection TPR @ FPR=1%\n(under scheme-specific optimal attacker)")
    axL.set_title("Every baseline collapses; only the two-statistic test holds", pad=12)
    axL.grid(True, axis="y", alpha=0.25, zorder=0)

    # highlight ours
    for i, s in enumerate(order):
        if s == "T_meanvar_z":
            bars[i].set_edgecolor("#145a32")
            bars[i].set_linewidth(2.5)
            axL.text(i, wc[s]+0.14, "OURS", ha="center", fontsize=9,
                     fontweight="bold", color="#145a32")

    # ---- RIGHT: FPR-honesty context scatter ----
    for s in order:
        c = "#27ae60" if wc[s] >= CLIFF else "#c0392b"
        mk = "*" if s == "T_meanvar_z" else "o"
        sz = 320 if s == "T_meanvar_z" else 130
        axR.scatter(benign_mu[s], thr_tail[s], s=sz, marker=mk, color=c,
                    edgecolor="0.2", linewidth=0.8, zorder=3)
        axR.annotate(DISPLAY[s].replace(" (ours)", ""),
                     (benign_mu[s], thr_tail[s]),
                     textcoords="offset points", xytext=(7, 5), fontsize=8)
    axR.set_xlabel("Benign score mean\n(high = compressed dynamic range)")
    axR.set_ylabel("Threshold depth in tail (std above benign mean)\n(high = brittle calibration)")
    axR.set_title("FPR-honesty: all compared at genuine 1% FPR\n(realized FPR ≈ 0.010 for every scheme)", pad=12, fontsize=11)
    axR.grid(True, alpha=0.25)

    fig.suptitle("Defense comparison under scheme-specific optimal adversaries "
                 "(6 defenses, 100 seeds, verified attackers)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    csv = os.path.join(HERE, "exp3_crosscheck.csv")
    if not os.path.exists(csv):
        print(f"  ! missing {csv}")
        return
    print("Building §7 defense-comparison figure...")
    build(csv, os.path.join(HERE, "fig_defense_comparison.png"))
    print("Done.")


if __name__ == "__main__":
    main()
