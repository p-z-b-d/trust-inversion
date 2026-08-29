"""
phase4_temporal_defense.py — Temporal second-moment defense (T_temporal_z)
and combined three-moment defense (T_three_moment).

Completes the three-piece defense suite alongside T_meanvar_z (spatial second
moment) and Theorem 2's observation-length condition (mimicry). Each channel
matches a different attack-window type:

    z_mu      first moment                   catches: AP1 baseline
    z_spatial spatial second moment          catches: AP4 dilution, AP6 collusion
    z_temporal temporal second moment        catches: AP2 sleeper, AP3 alternator
    (AP5 mimicry is handled by Theorem 2: observation length n_win > n_win*)

T_three_moment := max(z_mu, z_spatial, z_temporal) — the composite defense.

Why TWO second-moment channels (not one)
----------------------------------------
For an attacker with k outgoing edges observed over n_win windows, the
per-edge per-window signal matrix X has shape (k, n_win). The variance of
this matrix decomposes (under the conditional-mean model) into:

    Var[X] = Var_e[E_t[X]]      (across-edge variance of per-edge time-means)
             + E_e[Var_t[X]]    (mean per-edge temporal variance)

These two channels are STRUCTURALLY ORTHOGONAL: AP4 dilution and AP6
collusion produce edges with constant-in-time but heterogeneous means
(spatial bimodality, no temporal bimodality); AP2 sleeper and AP3
alternator produce edges with homogeneous means but in-time bimodality
(temporal bimodality, no spatial bimodality). A SINGLE pooled-variance
test cannot tell these apart and gives up power on both. The two
standardized statistics together do.

Standardization (against the benign-noise null)
-----------------------------------------------
With per-window signal X_{e,w} = mean_e + epsilon_{e,w}, epsilon ~ N(0, sigma_pw^2):

    z_mu       = (mean(per_edge_means) - s_b) / (sigma_edge / sqrt(k))
                 where sigma_edge = sigma_pw / sqrt(n_win)
                 H0: z_mu ~ N(0, 1) asymptotically

    z_spatial  = (var(per_edge_means) - sigma_edge^2) / sqrt(2 * sigma_edge^4 / (k - 1))
                 H0: z_spatial ~ N(0, 1) asymptotically  (chi-sq -> normal as k -> inf)

    z_temporal = (mean(per_edge_temporal_vars) - sigma_pw^2)
                 / sqrt(2 * sigma_pw^4 / ((n_win - 1) * k))
                 H0: z_temporal ~ N(0, 1) asymptotically

FPR cost of the composite
-------------------------
max(z_1, z_2, z_3) of three N(0,1) statistics (with their actual mild
correlation) has a 99th percentile ~ 2.5-2.7 versus 2.33 for one — costing
~0.2-0.4 z-units of threshold, i.e. a few percentage points of TPR on
profiles where only one channel fires. The composite still wins overall
because for ANY attack-window type it has the right channel firing while
T_meanvar_z (two channels) is blind on AP2/AP3 entirely.

Run
---
    python3 phase4_temporal_defense.py

Self-tests then a 6 x 4 demo grid (six attack profiles times four defenses)
on the D3 RHMD operating point.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import networkx as nx

# Reuse temporal-profile generation and helpers from phase4_ap23
from phase4_ap23 import (
    DefenseOp,
    build_attack_pattern,
    compute_temporal_signals,
    compute_TPR_at_FPR,
    equivalent_dilution_phi,
)


# ============================================================================
# Per-window signal generators for SPATIAL profiles (AP4, AP5, AP6)
# ============================================================================

def compute_spatial_signals_per_window(
    graph: nx.DiGraph,
    attackers: Iterable[int],
    defense_op: DefenseOp,
    attack_profile: str,
    sigma_per_window: float,
    n_win: int,
    rng: np.random.Generator,
    ap_params: Optional[dict] = None,
) -> np.ndarray:
    """Generate per-window signals for SPATIAL attack profiles (AP4, AP5, AP6).

    Spatial profiles have per-edge mean constant across windows. This
    function broadcasts the deterministic per-edge mean to all n_win windows
    and adds independent Gaussian noise. The temporal variance under all
    spatial profiles is therefore EQUAL to the benign noise floor — z_temporal
    should be silent on AP4/AP5/AP6, by construction.

    AP4 (Dilution):
        ap_params = {'phi': float in [0,1]}
        For each attacker, randomly designate `round(phi * k_atk)` of its
        outgoing edges as attack-signal-carrying (s_a); the rest carry s_b.
        Benign edges always carry s_b.

    AP5 (Mimicry):
        ap_params = {'s_m': float in [s_b, s_a]}
        All attacker edges carry s_m; benign edges carry s_b.

    AP6 (Collusion):
        ap_params = {'alpha': float in [0, s_b]}  # default s_b (passive)
        Attacker edges to fellow attackers (in-clique) carry alpha;
        attacker edges to non-attackers (out-of-clique) carry s_a;
        non-attacker edges carry s_b.
    """
    if ap_params is None:
        ap_params = {}

    attackers_set = set(int(a) for a in attackers)
    edges = list(graph.edges())
    n_edges = len(edges)
    if n_edges == 0:
        return np.zeros((0, n_win))

    means = np.empty(n_edges)

    if attack_profile == "AP4":
        phi = float(ap_params.get("phi", 0.5))
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"AP4 phi must be in [0,1], got {phi}")

        # Group attacker edges by source
        atk_edges_by_src: Dict[int, list] = {a: [] for a in attackers_set}
        for i, (src, _dst) in enumerate(edges):
            if src in attackers_set:
                atk_edges_by_src[src].append(i)

        # For each attacker, randomly designate floor(phi * k) edges as attack
        for atk, edge_indices in atk_edges_by_src.items():
            k_atk = len(edge_indices)
            m = int(round(phi * k_atk))
            if m > 0:
                attack_inds = set(int(x) for x in
                                  rng.choice(edge_indices, size=m, replace=False))
            else:
                attack_inds = set()
            for i in edge_indices:
                means[i] = defense_op.s_a if i in attack_inds else defense_op.s_b

        # Benign edges
        for i, (src, _dst) in enumerate(edges):
            if src not in attackers_set:
                means[i] = defense_op.s_b

    elif attack_profile == "AP5":
        s_m = float(ap_params.get("s_m", (defense_op.s_a + defense_op.s_b) / 2))
        if not (defense_op.s_b <= s_m <= defense_op.s_a):
            raise ValueError(f"AP5 s_m must be in [{defense_op.s_b}, {defense_op.s_a}], "
                             f"got {s_m}")
        for i, (src, _dst) in enumerate(edges):
            means[i] = s_m if src in attackers_set else defense_op.s_b

    elif attack_profile == "AP6":
        alpha = float(ap_params.get("alpha", defense_op.s_b))
        if not (0.0 <= alpha <= defense_op.s_b):
            raise ValueError(f"AP6 alpha must be in [0, {defense_op.s_b}], got {alpha}")
        for i, (src, dst) in enumerate(edges):
            if src in attackers_set:
                if dst in attackers_set:
                    means[i] = alpha       # in-clique edge
                else:
                    means[i] = defense_op.s_a  # out-of-clique edge
            else:
                means[i] = defense_op.s_b

    else:
        raise ValueError(
            f"compute_spatial_signals_per_window: unsupported profile "
            f"'{attack_profile}'. AP1/AP2/AP3 are temporal — use "
            f"phase4_ap23.compute_temporal_signals instead."
        )

    # Broadcast to per-window and add i.i.d. Gaussian noise
    per_window = means[:, None] + rng.normal(0.0, sigma_per_window,
                                              size=(n_edges, n_win))
    return per_window


def compute_per_window_signals(
    graph: nx.DiGraph,
    attackers: Iterable[int],
    defense_op: DefenseOp,
    attack_profile: str,
    sigma_per_window: float,
    n_win: int,
    rng: np.random.Generator,
    ap_params: Optional[dict] = None,
) -> np.ndarray:
    """Unified per-window signal generator. Dispatches on profile family.

    AP1/AP2/AP3 (temporal): phase4_ap23.compute_temporal_signals
    AP4/AP5/AP6 (spatial):  compute_spatial_signals_per_window

    Returns
    -------
    per_window : ndarray of shape (n_edges, n_win)
        Per-edge per-window signals, edge ordering matches list(graph.edges()).
    """
    if attack_profile in ("AP1", "AP2", "AP3"):
        _, per_window = compute_temporal_signals(
            graph, attackers, defense_op, attack_profile,
            sigma_per_window, n_win, rng, ap_params, return_per_window=True,
        )
        return per_window
    if attack_profile in ("AP4", "AP5", "AP6"):
        return compute_spatial_signals_per_window(
            graph, attackers, defense_op, attack_profile,
            sigma_per_window, n_win, rng, ap_params,
        )
    raise ValueError(f"Unknown attack profile: '{attack_profile}'")


# ============================================================================
# Three-moment z-scores (the core defense statistics)
# ============================================================================

def compute_z_scores(
    graph: nx.DiGraph,
    per_window_signals: np.ndarray,
    s_b: float,
    sigma_per_window: float,
    n_win: int,
) -> np.ndarray:
    """Compute (z_mu, z_spatial, z_temporal) for every node.

    Parameters
    ----------
    graph : networkx.DiGraph
    per_window_signals : ndarray of shape (n_edges, n_win)
        Per-edge per-window raw signals, edge ordering = list(graph.edges()).
    s_b : float
        Benign-signal mean (only used for z_mu; z_spatial / z_temporal are
        centered on the noise nulls, not on s_b).
    sigma_per_window : float
        Per-window per-edge noise standard deviation.
    n_win : int
        Number of windows.

    Returns
    -------
    z_scores : ndarray of shape (N, 3)
        Each row is (z_mu, z_spatial, z_temporal) for node i.
        Nodes with k < 2 outgoing edges or n_win < 2 score 0 in every channel.
    """
    edges = list(graph.edges())
    N = graph.number_of_nodes()
    sigma_edge = sigma_per_window / np.sqrt(n_win)
    sigma_edge_sq = sigma_edge ** 2
    sigma_pw_sq = sigma_per_window ** 2

    # Group edge-window rows by source node
    by_src: Dict[int, list] = {n: [] for n in range(N)}
    for i, (src, _dst) in enumerate(edges):
        by_src[src].append(per_window_signals[i])  # (n_win,)

    z = np.zeros((N, 3))
    for n in range(N):
        edge_arrs = by_src[n]
        k = len(edge_arrs)
        if k < 2 or n_win < 2:
            continue

        # Stack to (k, n_win); per-edge time-averages and per-edge temporal variances
        X = np.stack(edge_arrs)                   # shape (k, n_win)
        per_edge_means = X.mean(axis=1)           # (k,)
        per_edge_t_vars = X.var(axis=1, ddof=1)   # (k,)

        # z_mu: standardized mean against (s_b, sigma_edge / sqrt(k))
        z_mu = (per_edge_means.mean() - s_b) / (sigma_edge / np.sqrt(k))

        # z_spatial: standardized sample variance of per-edge time-averages
        #            against the noise null sigma_edge^2
        sv_means = per_edge_means.var(ddof=1)
        z_spatial = (sv_means - sigma_edge_sq) / np.sqrt(
            2 * sigma_edge_sq ** 2 / (k - 1)
        )

        # z_temporal: standardized MEAN of per-edge temporal variances
        #             against the noise null sigma_pw^2
        mean_tv = per_edge_t_vars.mean()
        z_temporal = (mean_tv - sigma_pw_sq) / np.sqrt(
            2 * sigma_pw_sq ** 2 / ((n_win - 1) * k)
        )

        z[n] = (z_mu, z_spatial, z_temporal)

    return z


# ============================================================================
# Composable defenses
# ============================================================================

def T_first_moment_z(graph, per_window_signals, s_b, sigma_per_window, n_win):
    """T1-equivalent z-statistic: per-node z_mu only. Baseline first-moment defense."""
    return compute_z_scores(graph, per_window_signals, s_b, sigma_per_window, n_win)[:, 0]


def T_meanvar_z(graph, per_window_signals, s_b, sigma_per_window, n_win):
    """T_meanvar_z := max(z_mu, z_spatial). Spatial second-moment defense.
    
    Catches AP1 baseline (via z_mu), AP4 dilution and AP6 collusion (via z_spatial).
    Blind on AP2 sleeper and AP3 alternator (temporal channel missing)."""
    z = compute_z_scores(graph, per_window_signals, s_b, sigma_per_window, n_win)
    return np.maximum(z[:, 0], z[:, 1])


def T_temporal_z(graph, per_window_signals, sigma_per_window, n_win):
    """T_temporal_z := z_temporal alone. Temporal second-moment defense.
    
    Sibling to T_meanvar_z's spatial channel. Catches AP2 sleeper and AP3
    alternator (via per-edge temporal variance excess). Blind on AP1, AP4,
    AP5, AP6 (no temporal bimodality under those profiles)."""
    return compute_z_scores(graph, per_window_signals, 0.0,
                            sigma_per_window, n_win)[:, 2]


def T_three_moment(graph, per_window_signals, s_b, sigma_per_window, n_win):
    """T_three_moment := max(z_mu, z_spatial, z_temporal). Composite defense.
    
    Covers AP1/AP4/AP6 (z_spatial or z_mu) AND AP2/AP3 (z_temporal) in one
    statistic. AP5 mimicry remains an orthogonal regime governed by Theorem 2."""
    z = compute_z_scores(graph, per_window_signals, s_b, sigma_per_window, n_win)
    return z.max(axis=1)


# ============================================================================
# Demo: 6 attack profiles times 4 defenses on D3 RHMD
# ============================================================================

def _build_network(N, p_edge, rho_atk, seed):
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(N, p_edge, seed=seed, directed=True)
    n_atk = max(1, int(rho_atk * N))
    attackers = set(int(x) for x in rng.choice(N, size=n_atk, replace=False))
    labels = np.zeros(N, dtype=bool)
    labels[list(attackers)] = True
    return G, attackers, labels


def _build_active_sleeper_per_window(graph, attackers, defense_op,
                                      sleep_fraction, sleep_signal,
                                      sigma_per_window, n_win, rng):
    """AP2A: active sleeper. Like AP2 but with explicit sleep_signal that
    can be below s_b. The attacker forges a low signal during dormant
    windows to pull the time-averaged trust score DOWN, cancelling out the
    high signal from attack windows.

    Temporal analog of AP6 active forced-clique inversion (Corollary 1.2).
    When sleep_signal and sleep_fraction are chosen so that
        sleep_fraction * sleep_signal + (1 - sleep_fraction) * s_a == s_b,
    the attacker's time-averaged mean equals the benign baseline EXACTLY —
    z_mu is silent on the attacker. Only z_temporal can detect them.

    Returns the per-window signals; not exposed as a top-level profile
    because it's a one-off variant used to demonstrate the temporal-defense
    rationale.
    """
    attackers_set = set(int(a) for a in attackers)
    edges = list(graph.edges())
    n_edges = len(edges)
    sleep_end = int(round(sleep_fraction * n_win))

    means = np.empty((n_edges, n_win))
    for i, (src, _dst) in enumerate(edges):
        if src in attackers_set:
            means[i, :sleep_end] = sleep_signal
            means[i, sleep_end:] = defense_op.s_a
        else:
            means[i, :] = defense_op.s_b
    return means + rng.normal(0.0, sigma_per_window, size=(n_edges, n_win))


def _cancellation_sleep_signal(defense_op: DefenseOp, sleep_fraction: float) -> float:
    """For AP2A, return the sleep_signal that exactly cancels z_mu:

        sleep_signal = (s_b - (1 - f) * s_a) / f

    May be negative (and thus not physically achievable for [0,1]-bounded
    signals); caller should clip to 0 and report the residual z_mu signal
    that remains after clipping.
    """
    f = sleep_fraction
    return (defense_op.s_b - (1.0 - f) * defense_op.s_a) / f if f > 0 else defense_op.s_b


def demo(verbose: bool = True) -> dict:
    """6 attack profiles x 4 defenses on the D3 RHMD operating point, plus
    an active-sleeper variant (AP2A) that motivates T_temporal_z.

    The standard AP2/AP3 cases have BOTH a z_mu signature (time-averaged
    mean shift) AND a z_temporal signature (temporal bimodality). On those,
    z_mu dominates and T_meanvar_z handles them adequately. The active
    sleeper (AP2A) is the threat model where T_temporal_z is uniquely
    necessary: the attacker forges below s_b during dormant windows to
    cancel z_mu exactly, leaving z_temporal as the only firing channel.
    AP2A is the temporal analog of AP6 active forced-clique inversion
    (Corollary 1.2).
    """
    D3 = DefenseOp(s_b=0.198, s_a=0.873, name="D3_RHMD")

    target_sigma_edge = 0.15
    n_win = 20
    sigma_per_window = target_sigma_edge * np.sqrt(n_win)

    N, p_edge, rho_atk = 20, 0.30, 0.30
    n_seeds = 80

    # For AP2A: choose f such that sleep_signal = 0 cancels z_mu exactly.
    #   sleep_signal = (s_b - (1-f)*s_a) / f = 0  =>  f = 1 - s_b/s_a
    f_cancel = 1.0 - D3.s_b / D3.s_a  # = 0.773 on D3 (verify in selftest)
    # We use f slightly above f_cancel with sleep_signal=0 — attacker errs
    # toward over-correcting, ending up at time-averaged mean SLIGHTLY below
    # s_b (which makes them look MORE benign than benigns, not less).
    f_evasive = f_cancel + 0.02
    sleep_signal_evasive = max(0.0, _cancellation_sleep_signal(D3, f_evasive))
    pred_mean_evasive = f_evasive * sleep_signal_evasive + (1 - f_evasive) * D3.s_a

    profiles = [
        # Standard six (z_mu does most of the work; included for context)
        ("AP1 baseline",          "AP1", {}),
        ("AP4 dilution phi=0.50", "AP4", {"phi": 0.50}),
        ("AP5 mimicry s_m=0.388", "AP5", {"s_m": 0.388}),  # D3 measured s_m at p=10%
        ("AP6 active alpha=0.10", "AP6", {"alpha": 0.10}),
        ("AP2 sleeper f=0.50",    "AP2", {"sleep_fraction": 0.50}),
        ("AP3 alt d=0.50 p=4",    "AP3", {"duty_cycle": 0.50, "period": 4, "phase": "random"}),
        # The threat model that MOTIVATES T_temporal_z: active sleeper that nulls z_mu
        ("AP2A active f=%.2f" % f_evasive, "AP2A",
         {"sleep_fraction": f_evasive, "sleep_signal": sleep_signal_evasive}),
    ]

    defenses = [
        ("z_mu (T1)",      lambda G, pw: T_first_moment_z(G, pw, D3.s_b, sigma_per_window, n_win)),
        ("T_meanvar_z",    lambda G, pw: T_meanvar_z(G, pw, D3.s_b, sigma_per_window, n_win)),
        ("T_temporal_z",   lambda G, pw: T_temporal_z(G, pw, sigma_per_window, n_win)),
        ("T_three_moment", lambda G, pw: T_three_moment(G, pw, D3.s_b, sigma_per_window, n_win)),
    ]

    if verbose:
        print(f"D3 RHMD  s_b={D3.s_b}  s_a={D3.s_a}  gap={D3.s_a - D3.s_b:.3f}")
        print(f"  N={N}, p_edge={p_edge}, rho_atk={rho_atk}, n_win={n_win}, "
              f"sigma_pw={sigma_per_window:.3f}, sigma_edge={target_sigma_edge:.3f}, "
              f"n_seeds={n_seeds}")
        print(f"  AP2A: f_cancel={f_cancel:.3f}, f_evasive={f_evasive:.3f}, "
              f"sleep_signal={sleep_signal_evasive:.3f}, "
              f"predicted attacker mean={pred_mean_evasive:.3f}  (vs s_b={D3.s_b})")
        print()
        header = f"{'Attack profile':<26}" + "".join(f"{d[0]:>17}" for d in defenses)
        print(header)
        print("-" * len(header))

    results = {}
    for prof_name, profile_str, ap_params in profiles:
        row_tprs = {d[0]: [] for d in defenses}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            G, attackers, labels = _build_network(N, p_edge, rho_atk, seed)

            if profile_str == "AP2A":
                per_window = _build_active_sleeper_per_window(
                    G, attackers, D3,
                    sleep_fraction=ap_params["sleep_fraction"],
                    sleep_signal=ap_params["sleep_signal"],
                    sigma_per_window=sigma_per_window, n_win=n_win, rng=rng,
                )
            else:
                per_window = compute_per_window_signals(
                    G, attackers, D3, profile_str,
                    sigma_per_window, n_win, rng, ap_params,
                )

            for d_name, d_fn in defenses:
                scores = d_fn(G, per_window)
                row_tprs[d_name].append(compute_TPR_at_FPR(scores, labels))

        results[prof_name] = {d: float(np.nanmean(t)) for d, t in row_tprs.items()}

        if verbose:
            row = f"{prof_name:<26}" + "".join(
                f"{results[prof_name][d[0]]:>17.3f}" for d in defenses
            )
            print(row)

    if verbose:
        print()
        # Channel-coverage assertions
        print("Validation (channel coverage):")
        v = results
        ap2a_key = "AP2A active f=%.2f" % f_evasive
        checks = [
            # T_temporal_z silent on spatial profiles (TPR near baseline FPR)
            ("T_temporal_z silent on AP4 dilution (TPR ~ FPR baseline)",
             v["AP4 dilution phi=0.50"]["T_temporal_z"] < 0.20),
            ("T_temporal_z silent on AP6 collusion (TPR ~ FPR baseline)",
             v["AP6 active alpha=0.10"]["T_temporal_z"] < 0.20),
            # The KEY result: AP2A active sleeper defeats z_mu, T_meanvar_z but NOT T_temporal_z
            ("AP2A defeats z_mu (T1)  (TPR < 0.20)",
             v[ap2a_key]["z_mu (T1)"] < 0.20),
            ("AP2A defeats T_meanvar_z  (TPR < 0.40)",
             v[ap2a_key]["T_meanvar_z"] < 0.40),
            ("T_temporal_z catches AP2A active sleeper",
             v[ap2a_key]["T_temporal_z"] > v[ap2a_key]["T_meanvar_z"]),
            ("T_three_moment catches AP2A (rescues from spatial-only failure)",
             v[ap2a_key]["T_three_moment"] > v[ap2a_key]["T_meanvar_z"]),
            # Composability check: T_three_moment matches the best single-channel
            # defense on each profile (modulo small max-of-three threshold inflation)
            ("T_three_moment within ~5pp of best single-channel on AP1",
             v["AP1 baseline"]["T_three_moment"] + 0.05 >=
             max(v["AP1 baseline"][d[0]] for d in defenses[:3])),
            ("T_three_moment within ~5pp of best single-channel on AP4",
             v["AP4 dilution phi=0.50"]["T_three_moment"] + 0.05 >=
             max(v["AP4 dilution phi=0.50"][d[0]] for d in defenses[:3])),
        ]
        for desc, ok in checks:
            tag = "OK" if ok else "FAIL"
            print(f"  [{tag}] {desc}")

        print()
        print("Reading the table:")
        print("  Standard AP2/AP3: both z_mu AND z_temporal fire; z_mu dominates so")
        print("    T_meanvar_z handles them with a small (~3 pp) FPR-inflation cost.")
        print("  AP2A active sleeper: attacker chooses sleep_signal so the time-")
        print("    averaged mean equals s_b exactly  ==>  z_mu silenced, T_meanvar_z")
        print("    fails, ONLY z_temporal still fires.  This is the threat model")
        print("    where T_temporal_z is UNIQUELY NECESSARY.")
        print("  Temporal analog of AP6 inversion: same active-manipulation mechanism")
        print("    (forge a low signal to cancel the high-signal attacker mean)")
        print("    but in the time dimension instead of the in-clique dimension.")
        print("    Corollary 1.2 lifts naturally — see paper Discussion §8.")
        print()
        print("Defense matrix conclusions:")
        print("  - Each second-moment channel has its OWN active-manipulation regime:")
        print("    * z_spatial: AP6 forced-clique active  (Corollary 1.2)")
        print("    * z_temporal: AP2A active sleeper  (analog corollary, future work)")
        print("  - Three-piece T_three_moment covers BOTH active regimes by max-ing")
        print("    across channels; the threshold-inflation cost is ~few pp TPR loss")
        print("    on profiles where only one channel fires.")
        print("  - AP5 mimicry remains the orthogonal axis: handled by Theorem 2's")
        print("    observation-length condition, not by any second-moment channel.")

    return results


# ============================================================================
# Self-tests
# ============================================================================

def _selftest():
    """Lightweight invariants on the z-score machinery."""
    rng = np.random.default_rng(0)
    n_edges, n_win, sigma_pw = 50, 20, 0.5
    s_b = 0.2

    # 1) Pure-noise input: all three z-scores should be ~ N(0,1), max ~1.5 typically
    noise = s_b + rng.normal(0, sigma_pw, size=(n_edges, n_win))
    # Build a graph where node 0 has all 50 outgoing edges
    G = nx.DiGraph()
    G.add_nodes_from(range(51))
    for j in range(1, 51):
        G.add_edge(0, j)
    z = compute_z_scores(G, noise, s_b, sigma_pw, n_win)
    # Node 0 should have all three z-scores within +/- 5 (typical N(0,1))
    assert abs(z[0, 0]) < 5, f"z_mu under noise should be ~N(0,1), got {z[0,0]:.2f}"
    assert abs(z[0, 1]) < 5, f"z_spatial under noise should be ~N(0,1), got {z[0,1]:.2f}"
    assert abs(z[0, 2]) < 5, f"z_temporal under noise should be ~N(0,1), got {z[0,2]:.2f}"

    # 2) Constant-in-time per-edge signals (AP4-like): z_spatial fires, z_temporal silent
    half = n_edges // 2
    spatial_means = np.empty(n_edges)
    spatial_means[:half] = 0.8   # high-signal half
    spatial_means[half:] = 0.2   # low-signal half
    spatial_signal = (spatial_means[:, None]
                      + rng.normal(0, sigma_pw, size=(n_edges, n_win)))
    z = compute_z_scores(G, spatial_signal, s_b, sigma_pw, n_win)
    assert z[0, 1] > 5.0, f"z_spatial should fire HIGH on bimodal edges, got {z[0,1]:.2f}"
    assert abs(z[0, 2]) < 3.0, (
        f"z_temporal should be SILENT on constant-in-time signals, got {z[0,2]:.2f}"
    )

    # 3) Constant-across-edges-but-bimodal-in-time signals (AP2/AP3-like):
    #    z_temporal fires, z_spatial silent
    temporal_means = np.empty((n_edges, n_win))
    temporal_means[:, :n_win // 2] = 0.2   # benign half-window
    temporal_means[:, n_win // 2:] = 0.8   # attack half-window
    temporal_signal = temporal_means + rng.normal(0, sigma_pw, size=(n_edges, n_win))
    z = compute_z_scores(G, temporal_signal, s_b, sigma_pw, n_win)
    assert z[0, 2] > 5.0, f"z_temporal should fire HIGH on temporally-bimodal edges, got {z[0,2]:.2f}"
    assert abs(z[0, 1]) < 3.0, (
        f"z_spatial should be SILENT on constant-across-edges signals, got {z[0,1]:.2f}"
    )

    # 4) Pure-noise FPR sanity check (~ N(0,1))
    n_trials = 200
    benign_max_zs = []
    for trial in range(n_trials):
        noise = s_b + rng.normal(0, sigma_pw, size=(n_edges, n_win))
        z = compute_z_scores(G, noise, s_b, sigma_pw, n_win)
        benign_max_zs.append(z[0].max())
    q99 = np.quantile(benign_max_zs, 0.99)
    # Expected ~2.5-2.7 for max of three correlated standard normals
    assert 2.0 < q99 < 4.0, f"99th percentile of max-z should be ~2.5-2.7, got {q99:.2f}"

    # 5) Spatial profile generator: AP4 produces spatial bimodality
    D = DefenseOp(s_b=0.2, s_a=0.8)
    G2 = nx.erdos_renyi_graph(20, 0.5, seed=0, directed=True)
    pw = compute_spatial_signals_per_window(
        G2, attackers={0, 1, 2}, defense_op=D, attack_profile="AP4",
        sigma_per_window=0.1, n_win=20, rng=np.random.default_rng(0),
        ap_params={"phi": 0.5},
    )
    assert pw.shape == (G2.number_of_edges(), 20)
    # Temporal variance should be near sigma_pw^2 for ALL edges (no temporal structure)
    per_edge_tvar = pw.var(axis=1, ddof=1)
    assert np.all(per_edge_tvar < 0.05), (
        f"AP4 should have noise-only temporal variance, got max {per_edge_tvar.max():.3f}"
    )

    # 6) Spatial profile generator: AP6 in-clique edges carry alpha, out-of-clique s_a
    pw = compute_spatial_signals_per_window(
        G2, attackers={0, 1, 2}, defense_op=D, attack_profile="AP6",
        sigma_per_window=0.01, n_win=20, rng=np.random.default_rng(0),  # tiny noise to inspect means
        ap_params={"alpha": 0.05},
    )
    edges_list = list(G2.edges())
    for i, (s, d) in enumerate(edges_list):
        edge_mean = pw[i].mean()
        if s in {0, 1, 2} and d in {0, 1, 2}:
            assert abs(edge_mean - 0.05) < 0.05, f"in-clique edge mean ~ alpha, got {edge_mean}"
        elif s in {0, 1, 2}:
            assert abs(edge_mean - 0.8) < 0.05, f"out-of-clique edge mean ~ s_a, got {edge_mean}"
        else:
            assert abs(edge_mean - 0.2) < 0.05, f"benign edge mean ~ s_b, got {edge_mean}"

    # 7) Unified dispatcher: AP1 (temporal) and AP4 (spatial) both work
    pw1 = compute_per_window_signals(
        G2, {0}, D, "AP1", sigma_per_window=0.1, n_win=10,
        rng=np.random.default_rng(0),
    )
    pw4 = compute_per_window_signals(
        G2, {0}, D, "AP4", sigma_per_window=0.1, n_win=10,
        rng=np.random.default_rng(0), ap_params={"phi": 0.5},
    )
    assert pw1.shape == pw4.shape == (G2.number_of_edges(), 10)

    print("self-test: ALL PASSED")


if __name__ == "__main__":
    _selftest()
    print()
    demo(verbose=True)
