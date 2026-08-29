"""
rerun_grid_with_measured_sigma.py — Re-run the headline composition grid using
per-defense σ measured from real Pi data (not the placeholder σ=0.15).

INPUT:
  sigma_per_defense.json   produced by compute_sigma_per_defense.py
  sm_results.json          (existing) per-defense operating points
  phase4_sim.py            (existing) the simulator
  trust_schemes.py         (existing) the aggregation rules
  grid_driver.py           (existing) the grid runner — used as reference

OUTPUT:
  phase4_grid_v2.csv             new grid with per-defense σ
  fig_composition_heatmap_v2.png regenerated headline figure
  sigma_comparison_table.txt     side-by-side: σ=0.15 vs measured σ TPRs

Usage:
  python3 rerun_grid_with_measured_sigma.py
  # or with custom paths:
  python3 rerun_grid_with_measured_sigma.py --sigma-json sigma_per_defense.json \\
                                            --out-csv phase4_grid_v2.csv

Runtime: ~95 seconds on a laptop (matches the original headline-grid driver).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product

import numpy as np

# Make the simulator + schemes importable. If you keep this script in a
# different directory from phase4_sim.py / trust_schemes.py / grid_driver.py,
# adjust this path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_sim import (
    run_cell, gen_er_digraph, assign_roles, assign_edge_signals,
    s_b,
)
from trust_schemes import (
    T1_mean, T_max, T_median, make_T_meanvar_z,
    T2_SL, T3_Beta, T4_EigenTrust, T5_TFLDT,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-json", default="sigma_per_defense.json")
    ap.add_argument("--out-csv", default="phase4_grid_v2.csv")
    ap.add_argument("--out-figure", default="fig_composition_heatmap_v2.png")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--p-edge", type=float, default=0.30)
    ap.add_argument("--rho-atk", type=float, default=0.30)
    ap.add_argument("--reference-csv", default="phase4_grid.csv",
                    help="Original grid CSV for side-by-side comparison")
    ap.add_argument("--comparison-txt", default="sigma_comparison_table.txt")
    return ap.parse_args()


# ============================================================================
# Per-defense σ lookup
# ============================================================================

def load_per_defense_sigma(path: str) -> dict:
    """Map defense_key -> sigma_edge to use in the simulator."""
    with open(path) as f:
        data = json.load(f)
    sigmas = {}
    for d, entry in data.items():
        # Use the recommended (benign-derived) σ_edge.
        sigmas[d] = float(entry["sigma_edge_recommended"])
    return sigmas


# ============================================================================
# Grid runner — same structure as grid_driver.py but with per-defense σ
# ============================================================================

DEFENSES = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]

# Match grid_driver.py's headline grid: 5 schemes × 4 attack profiles
# T1, T2/SL, T3/Beta, T4 (EigenTrust), T5 (TFL-DT) are first-moment family;
# T_meanvar_z is the C3 defense.
SCHEMES = ["T1_mean", "T4_EigenTrust", "T5_TFLDT", "T_max", "T_meanvar_z"]

# Headline four-attack-profile suite
ATTACK_PROFILES = [
    ("AP1",            {}),
    ("AP4_phi030",     {"phi": 0.30}),
    ("AP5_phi100_p10", {"phi": 1.00, "p": 10}),
    ("AP6c_active",    {"inclique": 0.0, "clique_forced": True}),
]


def get_scheme(name, defense, sigma):
    """Build a scheme callable, matching grid_driver.py's get_scheme."""
    if name == "T_meanvar_z":
        return make_T_meanvar_z(s_b(defense), sigma)
    if name == "T1_mean":
        return T1_mean
    if name == "T_max":
        return T_max
    if name == "T_median":
        return T_median
    if name == "T2_SL":
        return T2_SL
    if name == "T3_Beta":
        return T3_Beta
    if name == "T4_EigenTrust":
        return T4_EigenTrust
    if name == "T5_TFLDT":
        return T5_TFLDT
    raise ValueError(f"unknown scheme: {name}")


def run_headline_grid(out_csv: str, n_seeds: int, N: int,
                      sigmas: dict) -> list[dict]:
    """Execute the full grid and write the CSV."""
    rows = []
    print(f"Running {len(DEFENSES)} defenses × {len(SCHEMES)} schemes × "
          f"{len(ATTACK_PROFILES)} attacks × {n_seeds} seeds = "
          f"{len(DEFENSES)*len(SCHEMES)*len(ATTACK_PROFILES)*n_seeds} simulations.")

    for defense in DEFENSES:
        sigma = sigmas.get(defense)
        if sigma is None:
            print(f"  WARNING: no σ for {defense}; falling back to 0.15")
            sigma = 0.15
        for scheme_name in SCHEMES:
            scheme = get_scheme(scheme_name, defense, sigma)
            for prof_name, kwargs in ATTACK_PROFILES:
                # run_cell handles AP1/AP4/AP5; AP6 uses run_clique_cell in
                # grid_driver, but we route the inclique kwarg through run_cell
                # which already handles it (see phase4_sim.py).
                r = run_cell(defense, prof_name.split("_")[0], scheme=scheme,
                             N=N, sigma=sigma, n_seeds=n_seeds, **kwargs)
                row = {
                    "defense": defense, "scheme": scheme_name,
                    "profile": prof_name, "N": N, "sigma": sigma,
                    "phi": kwargs.get("phi"), "p": kwargs.get("p"),
                    "inclique": kwargs.get("inclique"),
                    "clique_forced": bool(kwargs.get("clique_forced", False)),
                    "n_seeds": n_seeds,
                    "AUC": r.get("AUC"),
                    "tpr_at_fpr01": r["tpr_at_fpr01"],
                    "gap": r["gap"],
                    "stealth": r.get("stealth"),
                    "worst_vis": r.get("worst_vis"),
                }
                rows.append(row)
            print(f"  {defense:<14s} {scheme_name:<14s}  σ={sigma:.4f}  done")

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWrote: {out_csv}")
    return rows


# ============================================================================
# Side-by-side comparison vs. the original σ=0.15 grid
# ============================================================================

def write_comparison(rows_new: list[dict], ref_csv: str, out_txt: str) -> None:
    if not os.path.exists(ref_csv):
        print(f"No reference grid at {ref_csv}; skipping comparison.")
        return

    import pandas as pd
    df_new = pd.DataFrame(rows_new)
    df_ref = pd.read_csv(ref_csv)

    def keyof(row):
        return (row["defense"], row["scheme"], row["profile"])

    new_idx = {keyof(r): r for r in rows_new}
    lines = []
    lines.append("=" * 100)
    lines.append("Headline grid: σ=0.15 (old)  vs.  σ_per_defense (new)")
    lines.append("=" * 100)
    lines.append(
        f"{'defense':<14s} {'scheme':<16s} {'profile':<18s} "
        f"{'σ_old':>6s} {'σ_new':>7s}  "
        f"{'TPR_old':>8s} {'TPR_new':>8s}  {'Δ':>7s}"
    )
    lines.append("-" * 100)
    deltas = []
    for _, ref_row in df_ref.iterrows():
        k = (ref_row["defense"], ref_row["scheme"], ref_row["profile"])
        if k not in new_idx:
            continue
        new_row = new_idx[k]
        tpr_old = float(ref_row["tpr_at_fpr01"])
        tpr_new = float(new_row["tpr_at_fpr01"])
        delta = tpr_new - tpr_old
        deltas.append(delta)
        lines.append(
            f"{ref_row['defense']:<14s} {ref_row['scheme']:<16s} "
            f"{ref_row['profile']:<18s} "
            f"{float(ref_row['sigma']):>6.3f} {float(new_row['sigma']):>7.4f}  "
            f"{tpr_old:>8.3f} {tpr_new:>8.3f}  {delta:>+7.3f}"
        )

    if deltas:
        arr = np.array(deltas)
        lines.append("")
        lines.append("ΔTPR summary across all cells:")
        lines.append(f"  mean Δ  = {arr.mean():+.3f}")
        lines.append(f"  median  = {np.median(arr):+.3f}")
        lines.append(f"  max |Δ| = {np.max(np.abs(arr)):.3f}")
        lines.append(f"  cells with |Δ| > 0.05: {(np.abs(arr) > 0.05).sum()} of {arr.size}")
        lines.append(f"  cells with |Δ| > 0.10: {(np.abs(arr) > 0.10).sum()} of {arr.size}")
        lines.append("")
        lines.append("INTERPRETATION:")
        lines.append("  If max |Δ| < 0.05 across all cells, the per-defense σ change")
        lines.append("  is cosmetic and the original headline figure can stand.")
        lines.append("  If max |Δ| > 0.10 in any cell, regenerate fig_composition_heatmap")
        lines.append("  from phase4_grid_v2.csv and update the paper.")

    txt = "\n".join(lines)
    print()
    print(txt)
    with open(out_txt, "w") as f:
        f.write(txt + "\n")
    print(f"\nWrote: {out_txt}")


# ============================================================================
# Figure regeneration
# ============================================================================

def regenerate_heatmap(grid_csv: str, out_path: str) -> None:
    """Regenerate the composition heatmap from the new grid CSV.

    Reuses generate_figures_v2.fig_composition_heatmap if present; otherwise
    prints instructions for manual regeneration.
    """
    try:
        from generate_figures_v2 import fig_composition_heatmap
    except ImportError:
        print(f"\nNote: generate_figures_v2.py not importable in this env.")
        print(f"      Regenerate the heatmap manually with:")
        print(f"        python3 generate_figures_v2.py {grid_csv}")
        return
    print(f"\nRegenerating heatmap from {grid_csv} -> {out_path}")
    fig_composition_heatmap(grid_csv, out_path)
    print(f"Wrote: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if not os.path.exists(args.sigma_json):
        print(f"ERROR: {args.sigma_json} not found.")
        print("Run compute_sigma_per_defense.py first.")
        sys.exit(1)

    sigmas = load_per_defense_sigma(args.sigma_json)
    print(f"Loaded per-defense σ values from {args.sigma_json}:")
    for d, s in sorted(sigmas.items()):
        print(f"  {d:<14s} σ_edge = {s:.5f}")

    rows = run_headline_grid(args.out_csv, args.n_seeds, args.N, sigmas)
    write_comparison(rows, args.reference_csv, args.comparison_txt)
    regenerate_heatmap(args.out_csv, args.out_figure)


if __name__ == "__main__":
    main()
