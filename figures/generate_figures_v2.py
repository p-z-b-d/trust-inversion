"""Generate paper figures from grid CSV outputs.

Three figures:
  - fig_composition_heatmap.png : (6 defenses × 5 schemes) × 4 attack profiles,
                                  color = TPR@FPR=1%. The headline figure.
  - fig_T4_diagnostic.png       : T4 EigenTrust performance vs network size N
                                  across attack profiles (D3 RHMD).
  - fig_sigma_sweep.png         : T_meanvar_z robustness — TPR vs sigma for
                                  T1/T_max/T_meanvar_z across (D3, D6) × 3 APs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = "/mnt/user-data/outputs"

# Custom diverging colormap: red (broken) -> yellow (weak) -> green (robust).
CMAP = LinearSegmentedColormap.from_list(
    "paper_rdyg",
    ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b",
     "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"],
    N=256,
)


# ============================================================================
# Figure 1: Composition heatmap (HEADLINE)
# ============================================================================
def fig_composition_heatmap(csv_path, out_path):
    df = pd.read_csv(csv_path)

    scheme_order = ["T1_mean", "T_max", "T_meanvar_z", "T4_EigenTrust", "T5_TFLDT"]
    scheme_labels = ["T1\n(mean)", "T_max", "T_meanvar_z\n(C3)",
                     "T4\nEigenTrust", "T5\nTFL-DT"]
    defense_order = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD",
                     "D5_Stochastic", "D6_DRL"]
    defense_labels = ["D1 RF", "D2 CNN", "D3 RHMD", "D4 MTD",
                      "D5 Stoch", "D6 DRL"]
    profile_order = ["AP1", "AP4_phi030", "AP5_phi100_p10", "AP6c_active"]
    profile_titles = ["AP1 (baseline)",
                      "AP4 (dilution, φ=0.30)",
                      "AP5 (mimicry, p=10%)",
                      "AP6c+active (clique collusion)"]

    fig, axes = plt.subplots(1, len(profile_order), figsize=(18, 5.5), sharey=True)
    for i, (prof, title) in enumerate(zip(profile_order, profile_titles)):
        ax = axes[i]
        sub = df[df["profile"] == prof]
        mat = sub.pivot(index="defense", columns="scheme",
                        values="tpr_at_fpr01")
        mat = mat.reindex(defense_order)[scheme_order]
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap=CMAP, aspect="auto")
        ax.set_xticks(range(len(scheme_order)))
        ax.set_xticklabels(scheme_labels, fontsize=9.5)
        if i == 0:
            ax.set_yticks(range(len(defense_order)))
            ax.set_yticklabels(defense_labels, fontsize=10.5)
        ax.set_title(title, fontsize=11.5, pad=10)
        # Annotate cells
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat.values[r, c]
                if not np.isnan(val):
                    # Use white text on dark cells, black on light
                    color = "white" if val < 0.30 or val > 0.85 else "black"
                    weight = "bold" if val < 0.5 else "normal"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=9.5, fontweight=weight)

    fig.subplots_adjust(right=0.92, wspace=0.10)
    cbar_ax = fig.add_axes([0.94, 0.18, 0.012, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("TPR @ FPR = 1%", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Composition Vulnerability Heatmap\n"
                 "(σ=0.15, N=20, ρ_atk=0.30, p_edge=0.30, 100 seeds per cell)",
                 fontsize=12.5, y=1.04)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# ============================================================================
# Figure 2: T4 EigenTrust diagnostic — N sweep
# ============================================================================
def fig_T4_diagnostic(csv_path, out_path):
    df = pd.read_csv(csv_path)
    profile_order = ["AP1", "AP4_phi030", "AP5_phi100_p10", "AP6"]
    profile_labels = {"AP1": "AP1 (baseline)",
                      "AP4_phi030": "AP4 dilution (φ=0.30)",
                      "AP5_phi100_p10": "AP5 mimicry (p=10%)",
                      "AP6": "AP6 collusion (passive)"}

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for i, prof in enumerate(profile_order):
        ax = axes[i]
        for scheme, color, marker in [("T1_mean", "#1f77b4", "o"),
                                       ("T4_EigenTrust", "#d62728", "s")]:
            sub = df[(df["profile"] == prof)
                     & (df["scheme"] == scheme)].sort_values("N")
            ax.plot(sub["N"], sub["tpr_at_fpr01"],
                    marker=marker, color=color, ms=8, lw=1.8,
                    label=scheme.replace("_", " "))
        ax.set_xticks([20, 50, 100])
        ax.set_xlabel("Network size N", fontsize=10)
        if i == 0:
            ax.set_ylabel("TPR @ FPR = 1%", fontsize=10)
        ax.set_title(profile_labels[prof], fontsize=10.5)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
    axes[-1].legend(loc="center right", fontsize=9, framealpha=0.95)

    fig.suptitle("T4 EigenTrust has two distinct failure modes  (defense: D3 RHMD)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# ============================================================================
# Figure 3: T_meanvar_z robustness across sigma
# ============================================================================
def fig_sigma_sweep(csv_path, out_path):
    df = pd.read_csv(csv_path)
    defenses = ["D3_RHMD", "D6_DRL"]
    profile_order = ["AP4_phi030", "AP5_phi100_p10", "AP6c_active"]
    profile_labels = {"AP4_phi030": "AP4 dilution",
                      "AP5_phi100_p10": "AP5 mimicry",
                      "AP6c_active": "AP6c + active collusion"}
    scheme_style = {
        "T1_mean":     ("#1f77b4", "o", "T1 (mean)"),
        "T_max":       ("#ff7f0e", "s", "T_max"),
        "T_meanvar_z": ("#2ca02c", "D", "T_meanvar_z (C3)"),
    }

    fig, axes = plt.subplots(len(defenses), len(profile_order),
                              figsize=(13, 7), sharex=True, sharey=True)
    for r, d in enumerate(defenses):
        for c, p in enumerate(profile_order):
            ax = axes[r, c]
            for scheme, (color, marker, label) in scheme_style.items():
                sub = df[(df["defense"] == d)
                         & (df["profile"] == p)
                         & (df["scheme"] == scheme)].sort_values("sigma")
                ax.plot(sub["sigma"], sub["tpr_at_fpr01"],
                        color=color, marker=marker, ms=7, lw=1.8, label=label)
            if r == 0:
                ax.set_title(profile_labels[p], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{d.replace('_', ' ')}\nTPR @ FPR = 1%", fontsize=10)
            if r == len(defenses) - 1:
                ax.set_xlabel("σ (per-edge noise)", fontsize=10)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            ax.axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.4)
    axes[0, -1].legend(loc="lower left", fontsize=9, framealpha=0.95)

    fig.suptitle("T_meanvar_z robustness across noise levels  (N=20, 100 seeds)",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


if __name__ == "__main__":
    print("Generating composition heatmap...")
    fig_composition_heatmap(f"{OUT_DIR}/phase4_grid.csv",
                            f"{OUT_DIR}/fig_composition_heatmap.png")
    print("Generating T4 diagnostic figure...")
    fig_T4_diagnostic(f"{OUT_DIR}/phase4_T4_diag.csv",
                      f"{OUT_DIR}/fig_T4_diagnostic.png")
    print("Generating sigma sweep figure...")
    fig_sigma_sweep(f"{OUT_DIR}/phase4_sigma_sweep.csv",
                    f"{OUT_DIR}/fig_sigma_sweep.png")
    print("\nAll figures generated.")
