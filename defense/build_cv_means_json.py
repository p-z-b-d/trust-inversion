"""
build_cv_means_json.py
======================
Collapse the 5-fold sm_results_cv.json (30 entries: 6 defenses x 5 folds)
into a per-defense CV-means file in the EXACT format phase4_sim.py expects,
so the simulator runs on the locked 5-fold means instead of a single fold.

Why this exists
---------------
phase4_sim.py line 38 does `open("sm_results.json")` and reads per-defense
{s_a, s_b, s_m{...}} — but the original sm_results.json holds SINGLE-FOLD
values. The locked paper values are the 5-fold CV means. This script rebuilds
sm_results.json (CV-means version) without touching phase4_sim.py.

It also solves the two-folder problem: generate this file into whatever
directory you launch the simulator from, and the simulator's relative
open() finds it.

Output format (matches what phase4_sim.py reads):
{
  "D1_RF": {
     "s_a": <float>, "s_b": <float>,
     "s_m": {"5": <float>, "10": <float>, "25": <float>, "50": <float>},
     "per_class": {"A1_spectre_v1": <float>, ...},
     "n_attack_windows": <int>, "n_benign_windows": <int>,
     "_provenance": "5-fold CV mean"
  }, ...
}

Usage:
    python3 build_cv_means_json.py \
        --cv sm_results_cv.json \
        --out sm_results.json

    # defaults: --cv sm_results_cv.json  --out sm_results_cv_means.json
    # (writes to a distinct name by default so you don't clobber the
    #  single-fold file until you've inspected the means)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def _scalar(v):
    """Return a scalar from a field that may be a bare float or a
    {mean, ci95, n} summary dict."""
    if isinstance(v, dict):
        return v["mean"]
    return v


def collapse_folds(cv: dict) -> dict:
    """Produce per-defense CV-mean signals.

    The cv file contains, for each defense:
      - five per-fold entries keyed {defense}_fold0 .. _fold4
      - one pre-computed summary entry keyed {defense}_cv, whose scalar
        fields are {mean, ci95, n} dicts.

    We PREFER the pre-computed _cv summary (it uses the locked aggregation).
    If a defense has no _cv entry, we fall back to averaging its folds.
    """
    # Separate _cv summaries from fold entries
    cv_summaries = {}          # defense -> entry
    folds_by_def = defaultdict(list)
    for key, entry in cv.items():
        defense = entry.get("defense") or key.rsplit("_", 1)[0]
        if key.endswith("_cv") or entry.get("fold") in (None, "cv"):
            # treat as summary only if it actually carries _cv-style fields
            if key.endswith("_cv"):
                cv_summaries[defense] = entry
                continue
        folds_by_def[defense].append(entry)

    defenses = sorted(set(list(cv_summaries.keys()) + list(folds_by_def.keys())))
    out = {}
    for defense in defenses:
        if defense in cv_summaries:
            e = cv_summaries[defense]
            src = "pre-computed _cv summary"
        else:
            # fallback: average folds
            folds = folds_by_def[defense]
            n = len(folds)
            e = {
                "s_a": sum(_scalar(f["s_a"]) for f in folds) / n,
                "s_b": sum(_scalar(f["s_b"]) for f in folds) / n,
                "s_m": {k: sum(f["s_m"][k] for f in folds) / n
                        for k in folds[0]["s_m"]},
                "per_class": {k: sum(_scalar(f["per_class"][k]) for f in folds) / n
                              for k in folds[0]["per_class"]},
                "n_attack_windows": folds[0].get("n_attack_windows", 0),
                "n_benign_windows": folds[0].get("n_benign_windows", 0),
            }
            src = f"{n}-fold average (no _cv summary present)"

        # s_m may itself hold {mean,...} dicts per percentile in the summary
        s_m_raw = e["s_m"]
        s_m = {k: _scalar(v) for k, v in s_m_raw.items()}
        per_class = {k: _scalar(v) for k, v in e["per_class"].items()}

        out[defense] = {
            "s_a": _scalar(e["s_a"]),
            "s_b": _scalar(e["s_b"]),
            "s_m": s_m,
            "per_class": per_class,
            "n_attack_windows": e.get("n_attack_windows", 0),
            "n_benign_windows": e.get("n_benign_windows", 0),
            "_provenance": src,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="sm_results_cv.json")
    ap.add_argument("--out", default="sm_results_cv_means.json")
    args = ap.parse_args()

    with open(args.cv) as f:
        cv = json.load(f)

    means = collapse_folds(cv)

    with open(args.out, "w") as f:
        json.dump(means, f, indent=2)

    # Verification table against the locked Table 2
    expected = {
        "D1_RF": (0.998, 0.018), "D2_CNN": (0.980, 0.027),
        "D3_RHMD": (0.871, 0.199), "D4_MTD": (0.967, 0.066),
        "D5_Stochastic": (0.918, 0.479), "D6_DRL": (0.944, 0.091),
    }
    print(f"Wrote {args.out}  ({len(means)} defenses)")
    print(f"\n{'Defense':<16}{'s_a':>9}{'s_b':>9}{'s_m(10%)':>11}   locked-check")
    all_ok = True
    for d, e in means.items():
        ea, eb = expected.get(d, (None, None))
        sm10 = e["s_m"].get("10", float("nan"))
        ok = (ea is not None and abs(e["s_a"] - ea) < 0.006
              and abs(e["s_b"] - eb) < 0.006)
        all_ok = all_ok and ok
        flag = "OK" if ok else "CHECK"
        print(f"{d:<16}{e['s_a']:>9.4f}{e['s_b']:>9.4f}{sm10:>11.4f}"
              f"   (s_a~{ea}, s_b~{eb}) {flag}")
    print(f"\n{'ALL VALUES MATCH LOCKED TABLE 2' if all_ok else 'SOME VALUES DIFFER — inspect'}")


if __name__ == "__main__":
    main()
