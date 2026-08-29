"""
compute_sigma_per_defense.py — Off-Pi post-processing of raw per-window arrays.

INPUT (after transfer + unpacking from the Pi):
  raw_arrays/{defense}_fold{k}.npz   raw probability arrays per (defense, fold)
                                     with arrays: probs, is_attack, class_name
  sm_results_cv.json                 per-(defense, fold) summary stats with
                                     CV aggregates

OUTPUT:
  sigma_per_defense.json             one entry per defense, with:
                                       - mean σ_per_window across folds + 95%CI
                                       - σ stratified by benign / attack / per-class
                                       - implied σ_edge given n_windows_per_edge
  per_defense_distribution.txt       human-readable table of all σ statistics

WHY THIS MATTERS:
The simulator (phase4_sim.py line 52) currently uses SIGMA_EDGE_DEFAULT = 0.03
(or 0.15 in the headline grid) as a PLACEHOLDER. The code itself says:
  "calibrate from the real window-level variance (dump raw per-window
   probability arrays in one more measure_sm pass)."
This script does the calibration. The Pi data is dumped by pi_measure_full.py.

Use the OUTPUTS as follows:
  - sigma_per_defense.json['DEFENSE']['sigma_edge_recommended'] is the value
    to pass to the simulator for that defense.
  - 'sigma_edge_benign' is the most defensible single number (the FPR-controlled
    threshold is calibrated against the benign distribution variance).
  - 'sigma_edge_pooled' is more conservative (uses all windows).

Usage:
  python3 compute_sigma_per_defense.py [--raw-dir raw_arrays]
                                       [--cv-json sm_results_cv.json]
                                       [--n-windows-per-edge 570]
                                       [--out-json sigma_per_defense.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np


# Default n_windows_per_edge — matches the simulator's edge=mean(~570 windows)
# assumption. Override with --n-windows-per-edge if your edge-aggregation
# strategy changed.
DEFAULT_N_WINDOWS_PER_EDGE = 570


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw_arrays")
    ap.add_argument("--cv-json", default="sm_results_cv.json")
    ap.add_argument("--n-windows-per-edge", type=int,
                    default=DEFAULT_N_WINDOWS_PER_EDGE)
    ap.add_argument("--out-json", default="sigma_per_defense.json")
    ap.add_argument("--out-txt", default="per_defense_distribution.txt")
    return ap.parse_args()


def load_fold_arrays(raw_dir: str) -> Dict[str, Dict[int, dict]]:
    """Return {defense_name: {fold_idx: {probs, is_attack, class_name}}}."""
    pat = re.compile(r"^(?P<defense>D\d_\w+)_fold(?P<fold>\d+)\.npz$")
    by_defense: Dict[str, Dict[int, dict]] = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {raw_dir!r}")
    for path in files:
        m = pat.match(os.path.basename(path))
        if not m:
            print(f"  skip (no match): {path}")
            continue
        with np.load(path, allow_pickle=False) as z:
            by_defense[m.group("defense")][int(m.group("fold"))] = {
                "probs": z["probs"].astype(np.float64),
                "is_attack": z["is_attack"].astype(bool),
                "class_name": z["class_name"].astype(str),
            }
    return by_defense


def compute_sigma_for_fold(probs: np.ndarray, is_attack: np.ndarray,
                           class_name: np.ndarray) -> dict:
    """Compute multiple σ_per_window variants for one (defense, fold) array set."""
    out: dict = {}
    if probs.size > 1:
        out["sigma_all"] = float(probs.std(ddof=1))
    else:
        out["sigma_all"] = float("nan")
    ben = probs[~is_attack]
    atk = probs[is_attack]
    out["sigma_benign"] = float(ben.std(ddof=1)) if ben.size > 1 else float("nan")
    out["sigma_attack"] = float(atk.std(ddof=1)) if atk.size > 1 else float("nan")
    out["n_benign"] = int(ben.size)
    out["n_attack"] = int(atk.size)

    per_class = {}
    for cls in np.unique(class_name):
        mask = class_name == cls
        if mask.sum() > 1:
            per_class[str(cls)] = float(probs[mask].std(ddof=1))
    out["sigma_per_class"] = per_class
    return out


def aggregate(values: List[float]) -> dict:
    """Mean + 95% CI from a list of fold-level scalars."""
    vals = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    if vals.size == 1:
        return {"mean": float(vals[0]), "ci95": float("nan"), "n": 1}
    sem = vals.std(ddof=1) / np.sqrt(vals.size)
    return {"mean": float(vals.mean()), "ci95": float(1.96 * sem),
            "n": int(vals.size)}


def main():
    args = parse_args()

    print(f"Loading raw arrays from: {args.raw_dir}")
    by_defense = load_fold_arrays(args.raw_dir)
    print(f"Found {len(by_defense)} defenses with raw arrays.")
    for d, fmap in by_defense.items():
        print(f"  {d}: {len(fmap)} folds")

    # Per-fold σ measurements
    per_defense_folds: Dict[str, List[dict]] = {}
    for d, fmap in by_defense.items():
        per_defense_folds[d] = []
        for fold_idx in sorted(fmap.keys()):
            arr = fmap[fold_idx]
            stats = compute_sigma_for_fold(
                arr["probs"], arr["is_attack"], arr["class_name"]
            )
            stats["fold"] = fold_idx
            per_defense_folds[d].append(stats)

    # Aggregate across folds for each defense
    n_edge = args.n_windows_per_edge
    sqrt_n = float(np.sqrt(n_edge))
    sigma_per_defense = {}
    for d, fold_list in per_defense_folds.items():
        agg_all = aggregate([f["sigma_all"] for f in fold_list])
        agg_ben = aggregate([f["sigma_benign"] for f in fold_list])
        agg_atk = aggregate([f["sigma_attack"] for f in fold_list])

        # Per-class aggregation — only classes present in every fold
        common_classes = set(fold_list[0]["sigma_per_class"].keys())
        for f in fold_list[1:]:
            common_classes &= set(f["sigma_per_class"].keys())
        per_class_agg = {
            cls: aggregate([f["sigma_per_class"][cls] for f in fold_list])
            for cls in sorted(common_classes)
        }

        # Recommended sigma_edge = benign σ_per_window / √n_windows_per_edge.
        # This is the threshold-relevant noise (FPR-controlled threshold is
        # set against the benign distribution).
        sigma_edge_benign = agg_ben["mean"] / sqrt_n
        sigma_edge_all = agg_all["mean"] / sqrt_n
        sigma_edge_attack = agg_atk["mean"] / sqrt_n

        sigma_per_defense[d] = {
            "n_folds": len(fold_list),
            "n_windows_per_edge_assumed": n_edge,
            "sigma_per_window": {
                "all":    agg_all,
                "benign": agg_ben,
                "attack": agg_atk,
                "per_class": per_class_agg,
            },
            # The values to feed to the simulator
            "sigma_edge_benign":      float(sigma_edge_benign),  # recommended
            "sigma_edge_pooled":      float(sigma_edge_all),
            "sigma_edge_attack":      float(sigma_edge_attack),
            "sigma_edge_recommended": float(sigma_edge_benign),  # alias
        }

    # Save JSON
    with open(args.out_json, "w") as f:
        json.dump(sigma_per_defense, f, indent=2)
    print(f"\nWrote: {args.out_json}")

    # Human-readable summary
    lines = []
    lines.append("=" * 100)
    lines.append("Per-defense σ measurements (from raw per-window arrays, "
                 f"n_windows_per_edge={n_edge})")
    lines.append("=" * 100)
    hdr = (f"{'Defense':<22s} {'folds':>5s}  "
           f"{'σ_pw[benign]':>20s}  {'σ_pw[attack]':>20s}  "
           f"{'σ_edge[benign]':>16s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for d in sorted(sigma_per_defense.keys()):
        s = sigma_per_defense[d]
        ben = s["sigma_per_window"]["benign"]
        atk = s["sigma_per_window"]["attack"]
        lines.append(
            f"{d:<22s} {s['n_folds']:>5d}  "
            f"{ben['mean']:.4f} ± {ben['ci95']:.4f}    "
            f"{atk['mean']:.4f} ± {atk['ci95']:.4f}    "
            f"{s['sigma_edge_benign']:>16.5f}"
        )
    lines.append("")
    lines.append("Compare against the simulator's current SIGMA_EDGE_DEFAULT = 0.03")
    lines.append("(phase4_sim.py line 52) and the headline-grid value SIGMA = 0.15")
    lines.append("(grid_driver.py).  If sigma_edge_benign differs >2x from 0.15 for")
    lines.append("any defense, the headline grid should be re-run with per-defense σ.")

    txt = "\n".join(lines)
    print()
    print(txt)
    with open(args.out_txt, "w") as f:
        f.write(txt + "\n")
    print(f"\nWrote: {args.out_txt}")


if __name__ == "__main__":
    main()
