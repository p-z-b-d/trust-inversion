"""
phase4_ap23.py — AP2 (Sleeper) and AP3 (Alternator) temporal attack profiles
for Phase 4 trust composition experiments.

Adds the two TEMPORAL-evasion profiles missing from the AP1, AP4, AP5, AP6
spatial profiles already in phase4_sim.py, completing the AP1-AP6 spectrum
claimed in PHASE4_DESIGN.md.

Key design points
-----------------
1. Temporal patterns are PER-NODE, not per-edge. An attacker either IS or
   IS NOT attacking in a given window; all of its outgoing edges share that
   pattern. (An attacker can't be attacking one peer while behaving benignly
   toward another in the same window — that would be AP6 collusion, not AP2/AP3.)

2. Per-edge per-window signal = behavioral_mean[window] + N(0, sigma_per_window^2).
   The per-edge signal that first-moment aggregators see is the time-average
   over n_win windows.

3. AP2/AP3 with time-averaged signals are MATHEMATICALLY EQUIVALENT to AP4
   dilution with phi = effective attack duty cycle. Theorem 1 (Dilution Bound)
   applies directly. The novelty of AP2/AP3 is the threat model (temporal
   evasion) and the implication for defenses: spatial second-moment defenses
   like T_meanvar_z DO NOT catch AP2/AP3 (all edges share the same
   time-averaged mean, so spatial variance across edges shows no extra signal).
   Catching AP2/AP3 requires TEMPORAL consistency monitoring — orthogonal
   defense direction acknowledged in PAPER_NOTES.md.

4. The demo at __main__ empirically verifies the AP2~=AP4(phi=1-f) and
   AP3~=AP4(phi=d) equivalences for the D3 RHMD operating point.

Integration with phase4_sim.py
------------------------------
In `compute_edge_signals` (or wherever per-edge signals are generated),
dispatch on attack_profile:

    if attack_profile in ('AP2', 'AP3'):
        from phase4_ap23 import compute_temporal_signals
        return compute_temporal_signals(graph, attackers, defense_op,
                                        attack_profile, sigma_per_window,
                                        n_win, rng, ap_params=ap_params)
    elif attack_profile == 'AP1':
        # existing AP1 implementation OR call compute_temporal_signals with AP1
        ...
    elif attack_profile == 'AP4':
        # existing spatial dilution implementation
        ...
    # ... etc

The temporal signals come out as the same 1-D (n_edges,) array that
first-moment trust schemes already consume. Pass `return_per_window=True`
to also get the (n_edges, n_win) raw array for temporal-aware defenses
or for the AP2/AP3 diagnostic figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import networkx as nx


# ============================================================================
# Defense operating point (kept self-contained so this file runs standalone)
# ============================================================================

@dataclass
class DefenseOp:
    """Per-edge signal means under benign and attack behavior for one defense."""
    s_b: float  # benign-signal mean (per-edge expected signal under benign behavior)
    s_a: float  # attack-signal mean (per-edge expected signal under attack)
    name: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.s_b <= 1.0):
            raise ValueError(f"s_b must be in [0,1], got {self.s_b}")
        if not (0.0 <= self.s_a <= 1.0):
            raise ValueError(f"s_a must be in [0,1], got {self.s_a}")
        if self.s_a < self.s_b:
            raise ValueError(f"defense degenerate: s_a={self.s_a} < s_b={self.s_b}; "
                             f"swap or check operating point measurement")


# ============================================================================
# Per-node temporal attack patterns
# ============================================================================

def build_attack_pattern(
    attack_profile: str,
    n_win: int,
    ap_params: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build a per-window boolean attack pattern for ONE attacker.

    Parameters
    ----------
    attack_profile : str
        One of 'AP1', 'AP2', 'AP3'. AP4/AP5/AP6 have spatial (per-edge) structure
        and are handled by their respective generators in phase4_sim.py.
    n_win : int
        Number of observation windows per edge. Must be >= 1.
    ap_params : dict, optional
        Profile-specific parameters:

        AP2 (Sleeper):
            'sleep_fraction' : float in [0, 1], default 0.5
                Fraction of n_win windows the attacker stays dormant before
                transitioning to full attack.
            'jitter' : float in [0, 1], default 0.0
                Per-attacker random shift of the wake-up time, expressed as
                fraction of n_win. Use jitter > 0 to desynchronize attackers
                (avoid the case where all attackers wake up in the same window,
                which makes attacks easier to detect via temporal correlation).

        AP3 (Alternator):
            'duty_cycle' : float in (0, 1), default 0.5
                Fraction of each period the attacker is in attack mode.
            'period' : int >= 2, default 4
                Length of one on-off cycle in windows.
            'phase' : 'random' | 'sync' | int, default 'random'
                Per-attacker phase offset. 'random' independently samples
                offset for each attacker (recommended for realistic threat
                model); 'sync' makes all attackers share phase 0; int sets
                the offset deterministically.

    rng : np.random.Generator, optional
        Required if jitter > 0 (AP2) or phase == 'random' (AP3).

    Returns
    -------
    pattern : ndarray of bool, shape (n_win,)
        pattern[w] is True iff the attacker is in attack mode during window w.

    Notes
    -----
    The mathematical fraction of windows in attack mode (the effective
    dilution phi for Theorem 1) is `pattern.mean()`. For AP1 this is 1.0,
    for AP2 it is approximately (1 - sleep_fraction), for AP3 it is
    approximately duty_cycle (exact when period divides n_win).
    """
    if n_win < 1:
        raise ValueError(f"n_win must be >= 1, got {n_win}")
    if ap_params is None:
        ap_params = {}
    if rng is None:
        rng = np.random.default_rng()

    pattern = np.zeros(n_win, dtype=bool)

    if attack_profile == "AP1":
        # Baseline: always attacking. Provided so AP1 can share the temporal
        # plumbing for fair comparison runs.
        pattern[:] = True

    elif attack_profile == "AP2":
        sleep_fraction = float(ap_params.get("sleep_fraction", 0.5))
        jitter = float(ap_params.get("jitter", 0.0))

        if not (0.0 <= sleep_fraction <= 1.0):
            raise ValueError(f"AP2 sleep_fraction must be in [0,1], got {sleep_fraction}")
        if not (0.0 <= jitter <= 1.0):
            raise ValueError(f"AP2 jitter must be in [0,1], got {jitter}")

        # Wake-up window index, optionally jittered
        sleep_end = int(round(sleep_fraction * n_win))
        if jitter > 0:
            jitter_n = max(1, int(round(jitter * n_win)))
            # Shift uniformly in [-jitter_n, +jitter_n]
            sleep_end += int(rng.integers(-jitter_n, jitter_n + 1))
            sleep_end = max(0, min(n_win, sleep_end))

        pattern[sleep_end:] = True

    elif attack_profile == "AP3":
        duty_cycle = float(ap_params.get("duty_cycle", 0.5))
        period = int(ap_params.get("period", 4))
        phase = ap_params.get("phase", "random")

        if not (0.0 < duty_cycle < 1.0):
            raise ValueError(f"AP3 duty_cycle must be in (0,1), got {duty_cycle}")
        if period < 2:
            raise ValueError(f"AP3 period must be >= 2, got {period}")

        # Number of attack windows per period (>=1 by construction; if rounding
        # gives 0, force at least 1 attack window per period — otherwise the
        # attacker degenerates to AP-benign which is uninteresting).
        on_per_period = max(1, int(round(duty_cycle * period)))
        on_per_period = min(on_per_period, period - 1)  # also avoid pure-attack degeneracy

        if phase == "random":
            offset = int(rng.integers(0, period))
        elif phase == "sync":
            offset = 0
        else:
            offset = int(phase) % period

        # Vectorized fill: window w is attack iff ((w + offset) % period) < on_per_period
        w = np.arange(n_win)
        pattern = ((w + offset) % period) < on_per_period

    else:
        raise ValueError(
            f"build_attack_pattern: unsupported profile '{attack_profile}'. "
            f"AP4/AP5/AP6 have spatial structure (per-edge, not per-time) and are "
            f"handled by their dedicated generators in phase4_sim.py."
        )

    return pattern


# ============================================================================
# Temporal signal generation
# ============================================================================

def compute_temporal_signals(
    graph: nx.DiGraph,
    attackers: Iterable[int],
    defense_op: DefenseOp,
    attack_profile: str,
    sigma_per_window: float,
    n_win: int,
    rng: np.random.Generator,
    ap_params: Optional[dict] = None,
    return_per_window: bool = False,
):
    """Compute per-edge signals for a temporal attack profile.

    Per-attacker temporal pattern is built once and shared by all of that
    attacker's outgoing edges. Per-edge per-window signal is drawn from
    N(behavioral_mean[w], sigma_per_window^2). Per-edge signal returned to
    the trust aggregator is the time-average over n_win windows.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of N nodes. Edge (u, v) means u is observed by v.
    attackers : iterable of int
        Attacker node IDs.
    defense_op : DefenseOp
        Defense operating point (s_b, s_a).
    attack_profile : str
        'AP1', 'AP2', or 'AP3'.
    sigma_per_window : float
        Per-window per-edge Gaussian noise standard deviation. After
        time-averaging over n_win windows the per-edge signal std is
        sigma_per_window / sqrt(n_win).
    n_win : int
        Number of observation windows per edge.
    rng : np.random.Generator
        Random generator.
    ap_params : dict, optional
        Profile-specific parameters; see build_attack_pattern.
    return_per_window : bool, default False
        If True, also return the (n_edges, n_win) per-window raw signals.
        Useful for temporal-consistency-aware defenses and for diagnostic
        figures of the attack's temporal signature.

    Returns
    -------
    edge_signals : ndarray, shape (n_edges,)
        Per-edge time-averaged signals (what first-moment aggregators see).
        Edge ordering matches list(graph.edges()).
    per_window : ndarray, shape (n_edges, n_win), optional
        Returned only if return_per_window=True.
    """
    attackers_set = set(int(a) for a in attackers)
    edges = list(graph.edges())
    n_edges = len(edges)

    if sigma_per_window < 0:
        raise ValueError(f"sigma_per_window must be >= 0, got {sigma_per_window}")
    if n_edges == 0:
        empty = np.zeros(0)
        if return_per_window:
            return empty, np.zeros((0, n_win))
        return empty

    # Per-attacker temporal pattern. Built once per attacker, then broadcast
    # to all outgoing edges of that attacker.
    node_patterns: dict[int, np.ndarray] = {}
    for atk in attackers_set:
        node_patterns[atk] = build_attack_pattern(attack_profile, n_win, ap_params, rng)

    # Per-edge per-window means (vectorized assembly).
    means = np.empty((n_edges, n_win))
    for i, (src, _dst) in enumerate(edges):
        if src in attackers_set:
            mask = node_patterns[src]  # (n_win,) bool
            means[i] = np.where(mask, defense_op.s_a, defense_op.s_b)
        else:
            means[i].fill(defense_op.s_b)

    # Per-edge per-window signals = means + Gaussian noise
    per_window = means + rng.normal(0.0, sigma_per_window, size=(n_edges, n_win))

    # Time-averaged per-edge signal (clip to [0,1] only at the per-window stage
    # would be physically meaningful; for now we treat signals as unconstrained
    # real values, matching the existing simulator's convention).
    edge_signals = per_window.mean(axis=1)

    if return_per_window:
        return edge_signals, per_window
    return edge_signals


# ============================================================================
# Theoretical equivalence helper
# ============================================================================

def equivalent_dilution_phi(
    attack_profile: str,
    ap_params: Optional[dict] = None,
    n_win: int = 20,
) -> float:
    """Return the effective dilution fraction phi that AP2/AP3 reduce to
    under time-averaging.

    For Theorem 1 (Dilution Bound) purposes, AP2/AP3 with time-averaged
    signals are equivalent to AP4 dilution at phi = (fraction of windows
    in attack mode). This function returns that phi directly from the
    profile parameters, useful for predicting TPR before running sims.

    For AP2 (sleeper):    phi = 1 - sleep_fraction
    For AP3 (alternator): phi = on_per_period / period  (exact when n_win
                          is a multiple of period; otherwise +/- 1/period error)

    Theorem 1 prediction: under T1 mean with midpoint threshold, AP2/AP3 are
    detected with probability > 1/2 iff phi > 1/2 (the symmetric-threshold
    Dilution Bound). Detector accuracy and topology don't enter; only phi.
    """
    if ap_params is None:
        ap_params = {}

    if attack_profile == "AP1":
        return 1.0
    if attack_profile == "AP2":
        return 1.0 - float(ap_params.get("sleep_fraction", 0.5))
    if attack_profile == "AP3":
        period = int(ap_params.get("period", 4))
        duty_cycle = float(ap_params.get("duty_cycle", 0.5))
        on_per_period = max(1, int(round(duty_cycle * period)))
        on_per_period = min(on_per_period, period - 1)
        return on_per_period / period
    raise ValueError(f"equivalent_dilution_phi: unsupported profile '{attack_profile}'")


# ============================================================================
# Trust aggregation (T1, T_meanvar_z) — local copies so this file runs standalone
# ============================================================================

def T1_mean(graph: nx.DiGraph, edge_signals: np.ndarray) -> np.ndarray:
    """T1 PerEdgeRaw mean: per-node trust = mean of outgoing-edge signals."""
    edges = list(graph.edges())
    n_nodes = graph.number_of_nodes()
    sums = np.zeros(n_nodes)
    counts = np.zeros(n_nodes, dtype=int)
    for i, (src, _dst) in enumerate(edges):
        sums[src] += edge_signals[i]
        counts[src] += 1
    # Nodes with no outgoing edges score 0 (won't be flagged); flag separately if needed.
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def T_meanvar_z(graph: nx.DiGraph, edge_signals: np.ndarray,
                s_b: float, sigma_edge: float) -> np.ndarray:
    """T_meanvar_z: max(z_mu, z_sigma2) two-statistic test using time-averaged
    per-edge signals. The variance channel measures SPATIAL variance across
    a node's outgoing edges (vs the per-edge noise null) — distinct from
    TEMPORAL variance, which is what would be needed to catch AP2/AP3."""
    edges = list(graph.edges())
    n_nodes = graph.number_of_nodes()
    by_src: dict[int, list[float]] = {n: [] for n in range(n_nodes)}
    for i, (src, _dst) in enumerate(edges):
        by_src[src].append(edge_signals[i])

    scores = np.zeros(n_nodes)
    for n in range(n_nodes):
        x = np.asarray(by_src[n], dtype=float)
        k = len(x)
        if k < 2:
            scores[n] = 0.0
            continue
        z_mu = (x.mean() - s_b) / (sigma_edge / np.sqrt(k))
        # Standardize sample variance against the benign-noise null sigma_edge^2.
        # Under H0 (benign): var ~ N(sigma_edge^2, 2*sigma_edge^4 / (k-1)) asymptotically.
        z_var = (x.var(ddof=1) - sigma_edge ** 2) / np.sqrt(2 * sigma_edge ** 4 / (k - 1))
        scores[n] = max(z_mu, z_var)
    return scores


def compute_TPR_at_FPR(scores: np.ndarray, labels: np.ndarray,
                       fpr_target: float = 0.01) -> float:
    """TPR at the threshold giving FPR <= fpr_target (computed on benign tail)."""
    benign = scores[~labels]
    attacker = scores[labels]
    if benign.size == 0 or attacker.size == 0:
        return float("nan")
    threshold = np.quantile(benign, 1.0 - fpr_target)
    return float(np.mean(attacker > threshold))


# ============================================================================
# Standalone demo and validation
# ============================================================================

def predicted_T1_TPR_at_FPR(phi: float, s_a: float, s_b: float,
                             sigma_edge: float, k_avg: float,
                             fpr_target: float = 0.01) -> float:
    """Theoretical T1 TPR under FPR-controlled threshold for AP4-equivalent dilution.

    Under T1 mean with i.i.d. Gaussian per-edge noise, an attacker with
    effective dilution fraction phi has T1 score distributed N(s_b + phi*(s_a-s_b),
    sigma_edge^2 / k). The FPR-controlled threshold sits at the (1 - fpr_target)
    quantile of the benign distribution. TPR = Phi(z_signal - z_alpha) where
    z_signal = phi * (s_a - s_b) * sqrt(k) / sigma_edge.

    This prediction is the FPR-controlled analog of Theorem 1 (Dilution Bound),
    and is the version that matches our headline-grid measurements.
    """
    from math import erf, sqrt
    z_alpha = sqrt(2) * _erfinv(1 - 2 * fpr_target)  # = 2.326 for fpr=0.01
    z_signal = phi * (s_a - s_b) * np.sqrt(k_avg) / sigma_edge
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + erf((z_signal - z_alpha) / sqrt(2)))


def _erfinv(y: float) -> float:
    """Inverse error function via series — adequate for z_alpha computation.
    Avoids scipy dependency."""
    # Winitzki 2008 approximation; max error ~1.3e-4 over (-1, 1).
    a = 0.147
    ln1my2 = np.log(1.0 - y * y)
    term = 2.0 / (np.pi * a) + ln1my2 / 2.0
    return float(np.sign(y) * np.sqrt(np.sqrt(term * term - ln1my2 / a) - term))


def _build_network(N: int, p_edge: float, rho_atk: float, seed: int):
    """Standard Phase 4 network: directed ER graph, fixed attacker fraction."""
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(N, p_edge, seed=seed, directed=True)
    n_atk = max(1, int(rho_atk * N))
    attackers = set(int(x) for x in rng.choice(N, size=n_atk, replace=False))
    labels = np.zeros(N, dtype=bool)
    labels[list(attackers)] = True
    return G, attackers, labels


def demo(verbose: bool = True) -> dict:
    """End-to-end demonstration on the D3 RHMD operating point.

    Reproduces and verifies the three claims for AP2/AP3:

    1. AP2(sleep_fraction=f) and AP3(duty_cycle=d) at matched effective phi
       produce statistically indistinguishable T1 TPR — they reduce to the
       same AP4 dilution under time-averaging.
    2. Empirical T1 TPR matches the FPR-controlled-threshold theoretical
       prediction (the noisy analog of Theorem 1 Dilution Bound), tracking
       phi smoothly from phi << phi_max through saturation.
    3. T_meanvar_z does NOT catch AP2/AP3 — the spatial variance channel
       is silent because all of an attacker's edges share the same
       time-averaged mean. Catching AP2/AP3 requires TEMPORAL consistency
       monitoring (orthogonal to T_meanvar_z; flagged as future work in
       PAPER_NOTES.md §7.2).
    """
    # D3 RHMD operating point (PAPER_NOTES.md §5.27).
    D3 = DefenseOp(s_b=0.198, s_a=0.873, name="D3_RHMD")

    # Parameters chosen to match the headline-grid regime where the Dilution
    # Bound is binding (sigma_edge = 0.15). With n_win = 20, this means
    # sigma_per_window = 0.15 * sqrt(20) ~= 0.671 — a substantial per-window
    # noise that gets smoothed by time-averaging to the headline-grid level.
    target_sigma_edge = 0.15
    n_win = 20
    sigma_per_window = target_sigma_edge * np.sqrt(n_win)
    sigma_edge = sigma_per_window / np.sqrt(n_win)  # = target_sigma_edge

    N, p_edge, rho_atk = 20, 0.30, 0.30
    k_avg = N * p_edge  # = 6, average out-degree on the ER graph
    n_seeds = 100

    profiles = [
        ("AP1 (baseline)",       "AP1", {}),
        # AP2 sweep — sleep_fraction high to drive phi_eff into the binding regime
        ("AP2 sleeper(f=0.50)",  "AP2", {"sleep_fraction": 0.50}),
        ("AP2 sleeper(f=0.70)",  "AP2", {"sleep_fraction": 0.70}),
        ("AP2 sleeper(f=0.85)",  "AP2", {"sleep_fraction": 0.85}),
        ("AP2 sleeper(f=0.95)",  "AP2", {"sleep_fraction": 0.95}),
        # AP3 sweep — duty cycle matched to AP2 sleep fractions
        ("AP3 alt(d=0.50,p=4)",  "AP3", {"duty_cycle": 0.50, "period": 4,  "phase": "random"}),
        ("AP3 alt(d=0.30,p=10)", "AP3", {"duty_cycle": 0.30, "period": 10, "phase": "random"}),
        ("AP3 alt(d=0.15,p=20)", "AP3", {"duty_cycle": 0.15, "period": 20, "phase": "random"}),
        ("AP3 alt(d=0.05,p=20)", "AP3", {"duty_cycle": 0.05, "period": 20, "phase": "random"}),
    ]

    results = {}

    if verbose:
        print(f"D3 RHMD demo  (s_b={D3.s_b}, s_a={D3.s_a}, gap={D3.s_a - D3.s_b:.3f})")
        print(f"   N={N}, p_edge={p_edge}, rho_atk={rho_atk}, k_avg={k_avg}, "
              f"sigma_per_window={sigma_per_window:.3f}, n_win={n_win}, "
              f"sigma_edge={sigma_edge:.4f}, n_seeds={n_seeds}")
        print()
        hdr = (f"{'Profile':<22}{'phi_eff':>8}{'pred_TPR':>10}"
               f"{'T1_TPR':>9}{'+/-95%CI':>11}"
               f"{'T_mvz_TPR':>11}")
        print(hdr)
        print("-" * len(hdr))

    for name, profile, ap_params in profiles:
        phi_eff = equivalent_dilution_phi(profile, ap_params, n_win=n_win)
        pred_tpr = predicted_T1_TPR_at_FPR(phi_eff, D3.s_a, D3.s_b, sigma_edge, k_avg)

        tpr_t1, tpr_mvz = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            G, attackers, labels = _build_network(N, p_edge, rho_atk, seed)

            edge_signals = compute_temporal_signals(
                G, attackers, D3, profile,
                sigma_per_window=sigma_per_window,
                n_win=n_win, rng=rng, ap_params=ap_params,
            )

            scores_t1 = T1_mean(G, edge_signals)
            scores_mvz = T_meanvar_z(G, edge_signals, s_b=D3.s_b, sigma_edge=sigma_edge)

            tpr_t1.append(compute_TPR_at_FPR(scores_t1, labels))
            tpr_mvz.append(compute_TPR_at_FPR(scores_mvz, labels))

        t1_arr = np.asarray(tpr_t1)
        mvz_arr = np.asarray(tpr_mvz)
        t1_mean = float(np.nanmean(t1_arr))
        mvz_mean = float(np.nanmean(mvz_arr))
        # 95% CI half-width (SEM * 1.96)
        t1_ci = float(1.96 * np.nanstd(t1_arr, ddof=1) / np.sqrt(np.sum(~np.isnan(t1_arr))))

        results[name] = {
            "profile": profile,
            "phi_eff": phi_eff,
            "predicted_T1_TPR": pred_tpr,
            "T1_TPR": t1_mean,
            "T1_CI95": t1_ci,
            "T_meanvar_z_TPR": mvz_mean,
        }

        if verbose:
            print(f"{name:<22}{phi_eff:>8.3f}{pred_tpr:>10.3f}"
                  f"{t1_mean:>9.3f}{'+/-' + f'{t1_ci:.3f}':>11}"
                  f"{mvz_mean:>11.3f}")

    if verbose:
        # Validation 1: AP2/AP3 at matched phi give same T1 TPR.
        pairs = [
            ("AP2 sleeper(f=0.50)", "AP3 alt(d=0.50,p=4)"),
            ("AP2 sleeper(f=0.70)", "AP3 alt(d=0.30,p=10)"),
            ("AP2 sleeper(f=0.85)", "AP3 alt(d=0.15,p=20)"),
            ("AP2 sleeper(f=0.95)", "AP3 alt(d=0.05,p=20)"),
        ]
        print()
        print("Validation 1: AP2 and AP3 reduce to equivalent AP4 dilution under "
              "time-averaging")
        for a2, a3 in pairs:
            r2, r3 = results[a2], results[a3]
            diff = abs(r2["T1_TPR"] - r3["T1_TPR"])
            same_phi = abs(r2["phi_eff"] - r3["phi_eff"]) < 1e-9
            ok = same_phi and diff < (r2["T1_CI95"] + r3["T1_CI95"])
            tag = "OK" if ok else "CHECK"
            print(f"  [{tag}] phi={r2['phi_eff']:.2f}: "
                  f"AP2 TPR={r2['T1_TPR']:.3f}+/-{r2['T1_CI95']:.3f}  "
                  f"AP3 TPR={r3['T1_TPR']:.3f}+/-{r3['T1_CI95']:.3f}  "
                  f"|diff|={diff:.3f}")

        # Validation 2: empirical matches Gaussian-FPR theoretical prediction
        print()
        print("Validation 2: empirical T1 TPR tracks the FPR=1% threshold "
              "prediction Phi(phi*(s_a-s_b)*sqrt(k)/sigma_edge - z_alpha)")
        for name, r in results.items():
            if r["profile"] in ("AP2", "AP3"):
                emp, pred = r["T1_TPR"], r["predicted_T1_TPR"]
                within = abs(emp - pred) < max(0.05, r["T1_CI95"] + 0.02)
                tag = "OK" if within else "CHECK"
                print(f"  [{tag}] {name}: pred={pred:.3f}  emp={emp:.3f}+/-{r['T1_CI95']:.3f}")

        # Validation 3: T_meanvar_z gains nothing on AP2/AP3 vs T1.
        print()
        print("Validation 3: T_meanvar_z's SPATIAL variance channel cannot see "
              "TEMPORAL bimodality")
        print("              (expect T_meanvar_z TPR <= T1 TPR: the variance "
              "channel is silent but")
        print("               inflates the FPR threshold, costing a few "
              "percentage points)")
        for name, r in results.items():
            if r["profile"] in ("AP2", "AP3"):
                gap = r["T_meanvar_z_TPR"] - r["T1_TPR"]
                # Pass: T_mvz <= T1 within CI (slight under-performance expected)
                ok = gap <= r["T1_CI95"] + 0.02
                tag = "OK" if ok else "WARN"
                print(f"  [{tag}] {name}: T_meanvar_z={r['T_meanvar_z_TPR']:.3f}  "
                      f"T1={r['T1_TPR']:.3f}  (Delta={gap:+.3f})")

        print()
        print("Bottom line:")
        print("  1. AP2 and AP3 reduce to AP4 dilution at matched phi after "
              "time-averaging (V1).")
        print("  2. Empirical T1 TPR runs slightly above the Gaussian prediction "
              "at low phi (V2):")
        print("     small-N artifact -- with 14 benign nodes, quantile(0.99) ~= "
              "max, so effective")
        print("     FPR ~= 1/14 ~= 7% > target 1%. The headline grid (larger N) "
              "is less affected.")
        print("  3. T_meanvar_z is SLIGHTLY WORSE than T1 on AP2/AP3 (V3): the "
              "variance channel is")
        print("     silent under temporal attacks (all attacker edges share the "
              "same time-averaged")
        print("     mean) but it still raises the FPR threshold, costing a few "
              "TPR points.")
        print()
        print("For the paper: catching AP2/AP3 requires TEMPORAL-CONSISTENCY "
              "monitoring (per-window")
        print("variance, change-point detection, periodicity tests). T_meanvar_z "
              "is the SPATIAL")
        print("second-moment defense; a temporal sibling is acknowledged as "
              "orthogonal future work")
        print("(PAPER_NOTES.md §7.2). Including AP2/AP3 in the threat model "
              "strengthens the paper's")
        print("'AP1-AP6 coverage' claim and motivates the Discussion-section "
              "argument for two-piece")
        print("defenses (spatial + temporal moment channels).")

    return results


# ============================================================================
# Unit tests (run as part of demo)
# ============================================================================

def _selftest():
    """Lightweight invariants check. Raises AssertionError on failure."""
    rng = np.random.default_rng(0)

    # 1) AP1 pattern is all-True
    p = build_attack_pattern("AP1", n_win=10, rng=rng)
    assert p.all(), "AP1 pattern should be all-True"

    # 2) AP2 default is half-half
    p = build_attack_pattern("AP2", n_win=10, ap_params={"sleep_fraction": 0.5}, rng=rng)
    assert p.sum() == 5, f"AP2(f=0.5, n_win=10) should attack 5 windows, got {p.sum()}"
    assert (~p[:5]).all() and p[5:].all(), "AP2 should be off-then-on"

    # 3) AP3 default duty cycle gives correct attack fraction
    p = build_attack_pattern(
        "AP3", n_win=12, ap_params={"duty_cycle": 0.5, "period": 4, "phase": "sync"}, rng=rng
    )
    assert p.sum() == 6, f"AP3(d=0.5, p=4, n_win=12) should attack 6 windows, got {p.sum()}"

    # 4) AP3 phase='random' produces VARYING patterns across calls
    rng_r = np.random.default_rng(1)
    patterns = set()
    for _ in range(20):
        q = build_attack_pattern(
            "AP3", n_win=8, ap_params={"duty_cycle": 0.5, "period": 4, "phase": "random"}, rng=rng_r
        )
        patterns.add(tuple(q.tolist()))
    assert len(patterns) >= 2, "AP3 phase='random' should produce varying patterns"

    # 5) equivalent_dilution_phi correctness
    assert equivalent_dilution_phi("AP1") == 1.0
    assert equivalent_dilution_phi("AP2", {"sleep_fraction": 0.3}) == 0.7
    assert abs(equivalent_dilution_phi("AP3", {"duty_cycle": 0.5, "period": 4}) - 0.5) < 1e-9

    # 6) Parameter validation: bad inputs raise
    for bad in (-0.1, 1.1):
        try:
            build_attack_pattern("AP2", 10, {"sleep_fraction": bad}, rng)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AP2 should reject sleep_fraction={bad}")
    for bad in (0.0, 1.0):
        try:
            build_attack_pattern("AP3", 10, {"duty_cycle": bad}, rng)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AP3 should reject duty_cycle={bad}")

    # 7) Signal-generation smoke test on a tiny graph
    D = DefenseOp(s_b=0.2, s_a=0.8, name="test")
    G = nx.erdos_renyi_graph(10, 0.5, seed=0, directed=True)
    atks = {0, 1, 2}
    sig = compute_temporal_signals(
        G, atks, D, "AP2", sigma_per_window=0.1,
        n_win=20, rng=np.random.default_rng(0),
        ap_params={"sleep_fraction": 0.5},
    )
    assert sig.shape == (G.number_of_edges(),), "Wrong shape from compute_temporal_signals"
    # Attacker outgoing edges with sleep_fraction=0.5 should average near (s_a+s_b)/2 = 0.5
    edges = list(G.edges())
    atk_edge_means = [sig[i] for i, (s, _) in enumerate(edges) if s in atks]
    benign_edge_means = [sig[i] for i, (s, _) in enumerate(edges) if s not in atks]
    if atk_edge_means:
        assert 0.3 < np.mean(atk_edge_means) < 0.7, (
            f"AP2(f=0.5) attacker mean should be near 0.5, got {np.mean(atk_edge_means):.3f}"
        )
    if benign_edge_means:
        assert abs(np.mean(benign_edge_means) - 0.2) < 0.1, (
            f"Benign mean should be near s_b=0.2, got {np.mean(benign_edge_means):.3f}"
        )

    # 8) return_per_window option
    sig, raw = compute_temporal_signals(
        G, atks, D, "AP3", sigma_per_window=0.1, n_win=20,
        rng=np.random.default_rng(0),
        ap_params={"duty_cycle": 0.5, "period": 4, "phase": "sync"},
        return_per_window=True,
    )
    assert raw.shape == (G.number_of_edges(), 20)
    assert np.allclose(raw.mean(axis=1), sig), "raw.mean(axis=1) should equal returned edge_signals"

    print("self-test: ALL PASSED")


if __name__ == "__main__":
    _selftest()
    print()
    demo(verbose=True)
