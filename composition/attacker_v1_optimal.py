"""
attacker_v1_optimal.py
======================
V1 — the continuous-OPTIMAL adaptive attacker against T_meanvar_z.

V0 (already built) searched a discrete (q, forge) grid of TWO-POINT emission
distributions. V1 solves for the attacker's best response over a richer family
and via continuous optimization, so a reviewer cannot say "you only tried a few
strategies."

--------------------------------------------------------------------------
The attacker's problem (white-box on the scheme, threshold-BLIND variant)
--------------------------------------------------------------------------
The defense scores a node from its k out-edge signals x_1..x_k via
    z_mu  = (mean(x) - s_b) / (sigma / sqrt(k))
    z_var = (var(x)  - sigma^2) / sqrt(2 sigma^4 / (k-1))
    score = max(z_mu, z_var)
and flags the node if score exceeds an FPR-calibrated cutoff.

Each emitted per-edge value is  base_e + N(0, sigma)  where base_e is the
attacker-chosen "true" value on that edge. Over many edges, the observed
per-edge distribution has
    observed_mean = mean(base)
    observed_var  = var(base) + sigma^2        (emission spread + obs noise)
So the attacker directly controls mean(base) and var(base). The defense's
z_var measures (observed_var - sigma^2) = var(base): the attacker's own
emission spread is exactly what the variance channel sees. This is the crux —
the attacker cannot hide emission spread behind observation noise.

Attacker emission family (rich but tractable, strictly generalizes V0):
  a 3-point distribution on [0,1]:
      value s_a  with weight w_a   (real attack payload — fixed at the defense's s_a)
      value m    with weight w_m   (a tunable "camouflage" mass, m free in [0,1])
      value 0    with weight w_0   (hard forged-benign)
  with w_a + w_m + w_0 = 1.
  * w_a is pinned to the attack yield q (the attacker MUST deliver payload on a
    q-fraction of edges — this is the rate/stealth axis we sweep).
  * (w_m, m, w_0) are FREE and optimized to minimize max(z_mu, z_var).
  V0's two-point family is the special case w_m = 0, so V1 >= V0 always.

We minimize the NODE score max(z_mu, z_var) analytically in the large-k limit
(mean/var of the mixture are closed-form), which is what the optimizer targets;
the simulator then MEASURES the realized TPR at finite k for the optimized
strategy. Reporting the measured TPR (not the analytic score) keeps us honest.

--------------------------------------------------------------------------
Threshold-AWARE variant (the +T axis, factor 2 of the 2x2)
--------------------------------------------------------------------------
threshold_z is passed in. A threshold-aware attacker doesn't minimize the score
outright — it tries to sit JUST BELOW threshold_z on BOTH channels
simultaneously (maximizing attack yield / camouflage freedom while staying
under the bar). We implement this by minimizing a hinge objective:
    max(0, z_mu - (threshold_z - eps)) + max(0, z_var - (threshold_z - eps))
so once both channels are under the cutoff the attacker stops paying to lower
them further and can instead push more payload. When threshold_z is None the
attacker just minimizes max(z_mu, z_var) (threshold-blind).

This module exposes ONE function, best_response(...), returning the optimized
emission parameters + the analytic z-scores. The simulator harness
(exp3_compare.py) turns those parameters into per-edge signals and measures TPR.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _mixture_moments(q, w_m, m, s_a, s_b_unused=None):
    """Mean and variance of the 3-point emission base distribution.
        s_a with weight q, m with weight w_m, 0 with weight w_0=1-q-w_m.
    Returns (mean_base, var_base). var_base is what the variance channel sees
    (observation noise sigma^2 adds on top inside the defense, and the z_var
    formula subtracts sigma^2 back off, so var_base is the exposed quantity).
    """
    w_0 = 1.0 - q - w_m
    mean = q * s_a + w_m * m + w_0 * 0.0
    ex2 = q * s_a ** 2 + w_m * m ** 2 + w_0 * 0.0
    var = ex2 - mean ** 2
    return mean, var


def _z_scores(mean_base, var_base, k, s_b, sigma):
    """Analytic z_mu, z_var for a node with k edges whose emission base has the
    given mean/variance. Observed mean = mean_base; observed var = var_base +
    sigma^2 (obs noise), and z_var subtracts sigma^2, exposing var_base."""
    z_mu = (mean_base - s_b) / (sigma / np.sqrt(k))
    # observed sample variance ~ var_base + sigma^2; z_var uses (obs_var - sigma^2)
    z_var = (var_base) / np.sqrt(2 * sigma ** 4 / (k - 1))
    return z_mu, z_var


def best_response(q, s_a, s_b, sigma, k, threshold_z=None, n_restarts=12,
                  seed=0):
    """Compute the attacker's optimal (w_m, m) for a given attack yield q.

    Parameters
    ----------
    q : float           attack yield (weight on the real payload s_a). Fixed.
    s_a, s_b : float    defense operating point.
    sigma : float       per-edge noise (calibration sigma).
    k : int             typical out-degree (for z-score scaling). Use E[k].
    threshold_z : float or None
        None  -> threshold-BLIND: minimize max(z_mu, z_var).
        value -> threshold-AWARE: minimize hinge over (threshold_z - eps).
    Returns dict with optimized params + analytic z-scores + objective.
    """
    eps = 0.05
    rng = np.random.default_rng(seed)

    def objective(x):
        w_m, m = x
        # feasibility: weights in [0,1], w_0 = 1-q-w_m >= 0
        if w_m < 0 or w_m > (1.0 - q) or m < 0 or m > 1:
            return 1e6
        mean_base, var_base = _mixture_moments(q, w_m, m, s_a)
        z_mu, z_var = _z_scores(mean_base, var_base, k, s_b, sigma)
        if threshold_z is None:
            # threshold-blind: push the max z-score as low as possible
            return max(z_mu, z_var)
        # threshold-aware: only pay to get each channel below (thr - eps)
        thr = threshold_z - eps
        return max(0.0, z_mu - thr) + max(0.0, z_var - thr)

    best = None
    # multi-start to avoid local minima in the (w_m, m) plane
    for _ in range(n_restarts):
        w_m0 = rng.uniform(0, max(1e-3, 1.0 - q))
        m0 = rng.uniform(0, 1)
        res = minimize(objective, x0=[w_m0, m0], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        if res.success or res.fun < 1e5:
            if best is None or res.fun < best.fun:
                best = res

    w_m, m = best.x
    w_m = float(np.clip(w_m, 0, 1.0 - q))
    m = float(np.clip(m, 0, 1))
    mean_base, var_base = _mixture_moments(q, w_m, m, s_a)
    z_mu, z_var = _z_scores(mean_base, var_base, k, s_b, sigma)
    return {
        "q": q, "w_m": w_m, "m": m, "w_0": 1.0 - q - w_m,
        "mean_base": mean_base, "var_base": var_base,
        "z_mu": float(z_mu), "z_var": float(z_var),
        "score": float(max(z_mu, z_var)),
        "objective": float(best.fun),
        "threshold_aware": threshold_z is not None,
    }


def emit_signal(q, w_m, m, s_a, E, rng):
    """Turn optimized emission params into E per-edge BASE values (pre-noise).
    Returns an array of length E drawn from the 3-point mixture with the given
    weights. The harness adds observation noise and clips."""
    w_0 = max(0.0, 1.0 - q - w_m)
    # normalize defensively
    tot = q + w_m + w_0
    probs = [q / tot, w_m / tot, w_0 / tot]
    choices = rng.choice([0, 1, 2], size=E, p=probs)
    vals = np.where(choices == 0, s_a, np.where(choices == 1, m, 0.0))
    return vals


if __name__ == "__main__":
    # quick self-check: V1 best response should do no worse than V0's best
    # two-point strategy (w_m=0) at the same q.
    s_a, s_b, sigma, k = 0.918, 0.479, 0.15, 6  # D5-like
    print("V1 optimal best-response self-check (D5-like operating point):")
    print(f"{'q':>5}{'w_m':>7}{'m':>7}{'z_mu':>8}{'z_var':>8}{'score':>8}")
    for q in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        br = best_response(q, s_a, s_b, sigma, k)
        print(f"{q:>5.1f}{br['w_m']:>7.3f}{br['m']:>7.3f}"
              f"{br['z_mu']:>8.2f}{br['z_var']:>8.2f}{br['score']:>8.2f}")
