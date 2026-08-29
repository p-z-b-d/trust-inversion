"""
scheme_cost.py
==============
Cost accounting for each trust-aggregation scheme, so the paper can answer the
USENIX reviewer's inevitable "your defense is effective, but at what cost?"

Two kinds of cost, both HARDWARE-INDEPENDENT (the honest kind, given no Pi):

  1. Analytic per-node aggregation cost as a function of in-degree k:
       - operation count (a concrete integer for a representative k)
       - complexity class (O(k), O(k log k), O(k * iters), ...)
     These are architecture-independent and reviewer-proof.

  2. Empirical wall-clock RATIO vs T1 mean (measured on whatever machine runs
     this). Absolute ms is meaningless off the Pi, but the RATIO
     (scheme_time / T1_time) is a fair indicative cost multiplier. Clearly
     labeled "indicative, relative" in any output.

The analytic model (per node with k outgoing-edge signals):
  T1_mean        : one pass sum + divide                 -> ~k ops,   O(k)
  T2_SL          : one pass (evidence transform) + mean   -> ~3k ops,  O(k)
  T3_Beta        : one pass (alpha/beta accumulate)       -> ~3k ops,  O(k)
  T5_TFLDT       : direct trust + 1-hop peer avg          -> ~2k + peers, O(k + deg)
  T_median       : sort of k values                       -> ~k log k, O(k log k)
  T_max          : one pass max                           -> ~k ops,   O(k)
  T_meanvar_z    : mean pass + variance pass + 2 z-calcs  -> ~2k + c,  O(k)  [2x T1]
  T4_EigenTrust  : power iteration, n_iter passes over    -> ~iters*E, O(iters * E)
                   the whole N-node graph (not per-node k)   [network-global]

The key headline the paper needs:
  T_meanvar_z is O(k), same class as T1 — it costs ~2x the baseline mean, NOT a
  new complexity class. That is the affordability argument: two-statistic
  robustness is a constant-factor cost, not an asymptotic one. Contrast with
  T_median (O(k log k)) and T4 EigenTrust (global power iteration).
"""
from __future__ import annotations

import time
import numpy as np


# Analytic per-node operation model. Returns an integer op-count estimate for a
# representative in-degree k, plus the complexity-class string.
def analytic_cost(scheme_name, k, N=20, E=None, n_iter=100):
    """k = per-node in-degree; E = total edges (for global schemes); n_iter =
    EigenTrust iterations."""
    if E is None:
        E = int(0.30 * N * (N - 1))  # ~p_edge * directed edges
    table = {
        "T1_mean":     (k + 1,               "O(k)"),
        "T2_SL":       (3 * k + 2,           "O(k)"),
        "T3_Beta":     (3 * k + 2,           "O(k)"),
        "T5_TFLDT":    (2 * k + k + 2,       "O(k + deg)"),
        "T_median":    (int(k * max(1, np.log2(max(2, k)))) + 1, "O(k log k)"),
        "T_max":       (k,                   "O(k)"),
        "T_meanvar_z": (2 * k + 6,           "O(k)"),   # mean pass + var pass + z-calcs
        "T4_EigenTrust": (n_iter * E,        "O(iters * E)"),
    }
    ops, cls = table.get(scheme_name, (k, "O(k)"))
    return {"op_count": int(ops), "complexity": cls}


def cost_relative_to_T1(scheme_name, k, **kw):
    """Analytic op-count as a multiple of T1's."""
    t1 = analytic_cost("T1_mean", k, **kw)["op_count"]
    sc = analytic_cost(scheme_name, k, **kw)["op_count"]
    return sc / t1 if t1 else float("nan")


def empirical_walltime_ratio(scheme_fns, edges, sig, N, repeats=200):
    """Measure wall-clock per aggregation for each scheme, return ratio vs T1.
    INDICATIVE ONLY (machine-relative). scheme_fns: dict name->callable."""
    times = {}
    for name, fn in scheme_fns.items():
        # warm up
        try:
            fn(edges, sig, N)
        except Exception:
            times[name] = float("nan")
            continue
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(edges, sig, N)
        times[name] = (time.perf_counter() - t0) / repeats
    t1 = times.get("T1_mean", float("nan"))
    return {name: (t / t1 if t1 and not np.isnan(t) else float("nan"))
            for name, t in times.items()}, times


if __name__ == "__main__":
    print("Analytic per-node cost model (representative k=6, N=20):")
    print(f"  {'scheme':<16}{'op_count':>10}{'vs T1':>8}   complexity")
    for s in ["T1_mean", "T2_SL", "T3_Beta", "T5_TFLDT", "T_max",
              "T_median", "T_meanvar_z", "T4_EigenTrust"]:
        c = analytic_cost(s, k=6)
        rel = cost_relative_to_T1(s, k=6)
        print(f"  {s:<16}{c['op_count']:>10}{rel:>8.1f}x   {c['complexity']}")
