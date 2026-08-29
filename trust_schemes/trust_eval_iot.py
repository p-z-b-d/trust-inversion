"""
IoT Network Trust Evaluation — Python Implementation
Adapted from: TFL-DT (Guo et al., IEEE JSAC 2023)
Trust evaluation for IoT networks.

Equation reference numbers match the ACM draft.
Bug fixes applied:
  - Eq 8: sigmoid denominator uses + (not -)
  - Eqs 9/10: anomaly factor is Eq9, delay factor is Eq10 (corrected ordering)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BehaviorRecord:
    """Single interaction record for node n_i with neighbor n_j.
    Corresponds to bh_{i,k} in Eq 3.

    IMPORTANT — metric conventions (both in [0, 1], higher = better):
      abb:       anomaly/abnormal degree. 1 = fully normal, 0 = fully anomalous.
                 In the paper: acc_{i,k}. Can incorporate microarchitectural health score.
      timeliness: how promptly the node delivered data.  1 = on time, 0 = maximum delay.
                 In the paper: delay_{i,k} is raw latency, so:
                   timeliness = 1 - clamp(latency / max_latency, 0, 1)
    """
    node_id: int          # i
    abb: float            # anomaly degree in [0, 1], higher = MORE normal
    timeliness: float     # delivery timeliness in [0, 1], higher = faster/better
    timestamp: float      # time of interaction (e.g. seconds or iteration number)


@dataclass
class RecommendedTrustRecord:
    """Recommended trust from a third-party node n_k about node n_i.
    Corresponds to rl^i_k in Eq 5.
    """
    recommender_id: int   # k (the recommending node)
    rt: float             # recommended trust value for n_i, from n_k
    h_ik: int             # number of interactions n_k has had with n_i
    timestamp: float      # time the recommendation was sent


# ---------------------------------------------------------------------------
# Core equations — reliability and stability
# ---------------------------------------------------------------------------

def anomaly_factor(
    records: list[BehaviorRecord],
    k: int,
    theta: float = 0.95,
    current_time: float = None,
) -> float:
    """
    Eq 9 (CORRECTED): Anomaly factor c^i_{a,k}
    Accumulates abnormality evidence up to interaction k,
    weighted by time-forgetting factor theta^(delta_t).

    Args:
        records: ordered list of behavior records for n_i
        k:       index up to which to accumulate (1-based in paper, 0-based here)
        theta:   time-forgetting factor in [0, 1]
        current_time: reference time for delta_t; defaults to last record's time
    Returns:
        c^i_{a,k} in [0, 1)
    """
    if current_time is None:
        current_time = records[k].timestamp

    weighted_sum = 0.0
    for q in range(k + 1):
        delta_t = current_time - records[q].timestamp
        weight = theta ** delta_t
        weighted_sum += records[q].abb * weight

    # 1 - e^{-sum} maps [0, inf) -> [0, 1)
    return 1.0 - math.exp(-weighted_sum)


def delay_factor(
    records: list[BehaviorRecord],
    k: int,
    theta: float = 0.95,
    current_time: float = None,
) -> float:
    """
    Eq 10 (CORRECTED): Delay factor c^i_{d,k}
    Same exponential accumulation but using delay values.

    Returns:
        c^i_{d,k} in [0, 1)
    """
    if current_time is None:
        current_time = records[k].timestamp

    weighted_sum = 0.0
    for q in range(k + 1):
        delta_t = current_time - records[q].timestamp
        weight = theta ** delta_t
        weighted_sum += records[q].timeliness * weight

    return 1.0 - math.exp(-weighted_sum)


def behavioral_reliability(
    records: list[BehaviorRecord],
    k: int,
    theta: float = 0.95,
    current_time: float = None,
) -> float:
    """
    Eq 8 (BUG FIXED): r_{ik} = sigmoid(c^i_{a,k}) * c^i_{d,k}
    Sigmoid denominator MUST be (1 + e^{-x}), NOT (1 - e^{-x}).

    Higher reliability requires BOTH:
      - High abb (normal behavior)     → large c_af → sigmoid approaches 1
      - High timeliness (fast delivery) → large c_df → scaling term approaches 1

    Both c_af and c_df are bounded in [0, 1), so r_ik ∈ [0, 1).
    """
    c_a = anomaly_factor(records, k, theta, current_time)
    c_d = delay_factor(records, k, theta, current_time)

    # Sigmoid of anomaly factor (CORRECTED: + in denominator)
    sig = 1.0 / (1.0 + math.exp(-c_a))

    return sig * c_d


def behavioral_stability(reliability_series: list[float]) -> float:
    """
    Eq 11: S_i = 1 - sum(|r_{i,k+1} - r_{i,k}|) / (l_i - 1)
    Measures how consistent (stable) n_i's reliability has been.
    Returns 1.0 if fewer than 2 records (perfectly stable by default).
    """
    if len(reliability_series) < 2:
        return 1.0

    total_variation = sum(
        abs(reliability_series[k + 1] - reliability_series[k])
        for k in range(len(reliability_series) - 1)
    )
    return 1.0 - total_variation / (len(reliability_series) - 1)


# ---------------------------------------------------------------------------
# Local trust calculation
# ---------------------------------------------------------------------------

def familiarity(h_ij: int, phi: float = 0.5) -> float:
    """
    Eq 28: Omega_{i,j} — familiarity of n_j with n_i.
    Grows toward 1 as interaction count h_{i,j} increases.
    phi controls how quickly familiarity accumulates.

    Args:
        h_ij: number of direct interactions between n_i and n_j
        phi:  growth rate parameter (0 < phi <= 1)
    Returns:
        Omega in [0, 1)
    """
    if h_ij < 1:
        return 0.0
    return 1.0 - 1.0 / (phi * math.sqrt(math.e ** h_ij - 1 + 1))


def g_function(h: float, lam: float = 0.1) -> float:
    """
    Eq 24: g(h) — confidence scaling function.
    Grows slowly to 1 to enforce "trust is hard to earn, easy to lose."

    Args:
        h:   interaction count h_{i,j}
        lam: growth rate lambda
    Returns:
        g(h) in [0, 1]
    """
    threshold = 1.0 / (math.sqrt(2) * 2 * lam)
    if 0 <= h <= threshold:
        return lam * h ** 2 + 0.5
    return 1.0


def current_trust(
    reliability_series: list[float],
    h_ij: int,
    lam: float = 0.1,
) -> float:
    """
    Eq 23: curr_T_j(i) = (0.5 + g(h) * (r_{i,l_i} - 0.5)) * epsilon
    Uses only the MOST RECENT reliability observation.

    Eq 25: epsilon = 1 if r_{i,l_i} > 0.5, else 0
    (Prevents a bad current interaction from artificially raising trust)
    """
    if not reliability_series:
        return 0.5

    r_latest = reliability_series[-1]

    # Eq 25
    epsilon = 1.0 if r_latest > 0.5 else 0.0

    # Eq 23
    g = g_function(h_ij, lam)
    return (0.5 + g * (r_latest - 0.5)) * epsilon


def time_weight(t_current: float, ct_iv: float) -> float:
    """
    Eq 22: phi_v = 2^{-(t - ct_{i,v})}
    Exponential time-decay weight for historical records.
    """
    return 2.0 ** (-(t_current - ct_iv))


def historical_trust(
    reliability_series: list[float],
    record_times: list[float],
    t_current: float,
) -> float:
    """
    Eq 21: hist_T_j(i) = sum_v( r_{iv} * phi_v / sum_r(phi_r) )
    Weighted average of historical reliability, with time decay.
    """
    if not reliability_series:
        return 0.5

    phi_weights = [time_weight(t_current, ct) for ct in record_times]
    phi_sum = sum(phi_weights)

    if phi_sum == 0:
        return 0.5

    return sum(
        r * (phi / phi_sum)
        for r, phi in zip(reliability_series, phi_weights)
    )


def local_trust(
    reliability_series: list[float],
    record_times: list[float],
    t_current: float,
    h_ij: int,
    stability: float,
    lam: float = 0.1,
    phi_param: float = 0.5,
) -> float:
    """
    Eq 20: local_T_j(i) = omega_curr * curr_T + omega_hist * hist_T

    Eq 26: omega_curr = 1 - omega_hist
    Eq 27: omega_hist = Omega_{i,j} * S_i

    Returns local trust value in [0, 1].
    """
    omega_ij = familiarity(h_ij, phi_param)
    omega_hist = omega_ij * stability
    omega_curr = 1.0 - omega_hist

    curr_t = current_trust(reliability_series, h_ij, lam)
    hist_t = historical_trust(reliability_series, record_times, t_current)

    return omega_curr * curr_t + omega_hist * hist_t


# ---------------------------------------------------------------------------
# Weight calculation (local vs recommended)
# ---------------------------------------------------------------------------

def f_omega(sigma: float, xi: float = 2.0) -> float:
    """
    Eq 19: f_omega(sigma) = xi if sigma >= xi, else sigma
    Caps the ratio at xi to prevent runaway dominance of local trust.
    """
    return xi if sigma >= xi else sigma


def compute_weights(
    h_ij: int,
    H_ij: int,
    neighbor_h_values: list[int],
    neighbor_H_values: list[int],
    xi: float = 2.0,
) -> tuple[float, float]:
    """
    Eqs 14–19: compute omega_local and omega_recom for node n_i.

    Args:
        h_ij:              direct interactions between n_i and evaluating node n_j
        H_ij:              total indirect interactions of n_i with other nodes (except n_j)
        neighbor_h_values: h_{k,j} for all n_k in N_j (direct interactions of peers with n_j)
        neighbor_H_values: H_{k,j} for all n_k in N_j (indirect interactions of peers)
        xi:                cap threshold for f_omega
    Returns:
        (omega_local, omega_recom) summing to 1
    """
    s_i = len(neighbor_h_values)  # |N_j|

    # Paper edge cases (stated explicitly):
    # "if no direct interaction, alpha_i=0 and omega_local=0"
    # "if no indirect interaction, beta_i=0 and omega_local=1"
    if h_ij == 0:
        return 0.0, 1.0      # no local evidence — rely entirely on recommendations
    if H_ij == 0:
        return 1.0, 0.0      # no indirect evidence — rely entirely on local trust

    # Eq 16: alpha_i
    avg_h = sum(neighbor_h_values) / s_i if s_i > 0 else 1
    alpha_i = h_ij / avg_h if avg_h > 0 else 0.0

    # Eq 17: beta_i
    avg_H = sum(neighbor_H_values) / s_i if s_i > 0 else 1
    beta_i = H_ij / avg_H if avg_H > 0 else 0.0

    # Eq 18: sigma_i
    sigma_i = h_ij / H_ij if H_ij > 0 else 0.0

    # Eq 19
    fw = f_omega(sigma_i, xi)

    # Eq 14: omega_local
    numerator = alpha_i * (fw / (fw + 1))
    denominator = numerator + beta_i * (1.0 / (fw + 1))

    if denominator == 0:
        return 0.5, 0.5

    omega_local = numerator / denominator

    # Eq 15
    omega_recom = 1.0 - omega_local

    return omega_local, omega_recom


# ---------------------------------------------------------------------------
# Recommended trust
# ---------------------------------------------------------------------------

def recommended_trust(
    rl_records: list[RecommendedTrustRecord],
    H_ij: int,
) -> float:
    """
    Eq 29: recom_T_j(i) = sum_k( h_{i,k}/H_{i,j} * rt_{i,k} )
    Weighted average of peer recommendations, weighted by each
    peer's interaction count with n_i.

    Args:
        rl_records: list of recommendation records for n_i
        H_ij:       total indirect interactions (normalization)
    Returns:
        recom_T in [0, 1]
    """
    if not rl_records or H_ij == 0:
        return 0.5  # uncertain default

    return sum(
        (rec.h_ik / H_ij) * rec.rt
        for rec in rl_records
    )


# ---------------------------------------------------------------------------
# Global trust
# ---------------------------------------------------------------------------

def global_trust(
    local_t: float,
    recom_t: float,
    omega_local: float,
    omega_recom: float,
) -> float:
    """
    Eq 13: T_j(i) = omega_local * local_T + omega_recom * recom_T
    """
    assert abs(omega_local + omega_recom - 1.0) < 1e-6, \
        "Weights must sum to 1"
    return omega_local * local_t + omega_recom * recom_t


# ---------------------------------------------------------------------------
# High-level node evaluator
# ---------------------------------------------------------------------------

class NodeTrustEvaluator:
    """
    Encapsulates the full trust evaluation pipeline for a single node n_i
    as observed by evaluating node n_j.

    Usage:
        evaluator = NodeTrustEvaluator(theta=0.95, lam=0.1, phi=0.5, xi=2.0)
        trust = evaluator.evaluate(
            behavior_records=...,
            rl_records=...,
            neighbor_h_values=...,
            neighbor_H_values=...,
            t_current=100.0,
        )
    """

    def __init__(
        self,
        theta: float = 0.95,   # time-forgetting factor
        lam: float = 0.1,      # confidence growth rate
        phi: float = 0.5,      # familiarity growth rate
        xi: float = 2.0,       # f_omega cap
    ):
        self.theta = theta
        self.lam = lam
        self.phi = phi
        self.xi = xi

    def compute_reliability_series(
        self,
        records: list[BehaviorRecord],
        t_current: float,
    ) -> list[float]:
        """Compute r_{ik} for all k up to the latest interaction."""
        return [
            behavioral_reliability(records, k, self.theta, t_current)
            for k in range(len(records))
        ]

    def evaluate(
        self,
        behavior_records: list[BehaviorRecord],
        rl_records: list[RecommendedTrustRecord],
        neighbor_h_values: list[int],
        neighbor_H_values: list[int],
        t_current: float,
    ) -> dict:
        """
        Full pipeline evaluation. Returns a dict with all intermediate
        values for inspection / debugging.
        """
        n = len(behavior_records)
        if n == 0:
            return {
                "global_trust": 0.5,
                "local_trust": 0.5,
                "recom_trust": 0.5,
                "omega_local": 0.5,
                "omega_recom": 0.5,
                "stability": 1.0,
                "familiarity": 0.0,
                "reliability_series": [],
            }

        # Step 1: reliability series
        r_series = self.compute_reliability_series(behavior_records, t_current)

        # Step 2: stability (Eq 11)
        S_i = behavioral_stability(r_series)

        # Step 3: interaction counts
        h_ij = n
        H_ij = sum(rec.h_ik for rec in rl_records) if rl_records else 0

        # Step 4: local trust (Eqs 20-28)
        record_times = [r.timestamp for r in behavior_records]
        loc_t = local_trust(
            r_series, record_times, t_current,
            h_ij, S_i, self.lam, self.phi,
        )

        # Step 5: recommended trust (Eq 29)
        rec_t = recommended_trust(rl_records, H_ij) if H_ij > 0 else 0.5

        # Step 6: weights (Eqs 14-19)
        omega_l, omega_r = compute_weights(
            h_ij, H_ij,
            neighbor_h_values, neighbor_H_values,
            self.xi,
        )

        # Step 7: global trust (Eq 13)
        g_t = global_trust(loc_t, rec_t, omega_l, omega_r)

        return {
            "global_trust": round(g_t, 4),
            "local_trust": round(loc_t, 4),
            "recom_trust": round(rec_t, 4),
            "omega_local": round(omega_l, 4),
            "omega_recom": round(omega_r, 4),
            "stability": round(S_i, 4),
            "familiarity": round(familiarity(h_ij, self.phi), 4),
            "reliability_series": [round(r, 4) for r in r_series],
            "curr_trust": round(current_trust(r_series, h_ij, self.lam), 4),
            "hist_trust": round(
                historical_trust(r_series, record_times, t_current), 4
            ),
        }


# ---------------------------------------------------------------------------
# Trust classification
# ---------------------------------------------------------------------------

def classify_trust(trust_value: float) -> str:
    """
    Classify a trust value into benign / uncertain / malicious.
    Thresholds from the paper's experiment setup (Table II).
    """
    if trust_value >= 0.8:
        return "benign"
    elif trust_value >= 0.4:
        return "uncertain"
    else:
        return "malicious"


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("IoT Trust Evaluation — Demo Run")
    print("=" * 60)

    evaluator = NodeTrustEvaluator(theta=0.95, lam=0.1, phi=0.5, xi=2.0)

    # --- Benign node: always delivers high-quality data on time ---
    benign_records = [
        BehaviorRecord(node_id=1, abb=0.92, timeliness=0.95, timestamp=float(t))
        for t in range(1, 21)
    ]
    benign_peers = [
        RecommendedTrustRecord(
            recommender_id=k, rt=0.85, h_ik=10, timestamp=19.0
        )
        for k in range(2, 6)
    ]

    result_benign = evaluator.evaluate(
        benign_records, benign_peers,
        neighbor_h_values=[15, 12, 18, 10, 14],
        neighbor_H_values=[40, 35, 45, 30, 38],
        t_current=20.0,
    )
    print(f"\nBenign node:")
    print(f"  Global trust : {result_benign['global_trust']}  → {classify_trust(result_benign['global_trust'])}")
    print(f"  Local trust  : {result_benign['local_trust']}")
    print(f"  Recom trust  : {result_benign['recom_trust']}")
    print(f"  Stability    : {result_benign['stability']}")
    print(f"  Familiarity  : {result_benign['familiarity']}")
    print(f"  ω_local      : {result_benign['omega_local']}")

    # --- Malicious node: consistently abnormal, high delay ---
    malicious_records = [
        BehaviorRecord(node_id=2, abb=0.08, timeliness=0.15, timestamp=float(t))
        for t in range(1, 21)
    ]
    malicious_peers = [
        RecommendedTrustRecord(
            recommender_id=k, rt=0.15, h_ik=10, timestamp=19.0
        )
        for k in range(2, 6)
    ]

    result_mal = evaluator.evaluate(
        malicious_records, malicious_peers,
        neighbor_h_values=[15, 12, 18, 10, 14],
        neighbor_H_values=[40, 35, 45, 30, 38],
        t_current=20.0,
    )
    print(f"\nMalicious node:")
    print(f"  Global trust : {result_mal['global_trust']}  → {classify_trust(result_mal['global_trust'])}")
    print(f"  Local trust  : {result_mal['local_trust']}")
    print(f"  Stability    : {result_mal['stability']}")

    # --- Alternating node (on/off attack, pattern 3 from the paper) ---
    alternating_records = [
        BehaviorRecord(
            node_id=3,
            abb=0.92 if t % 2 == 0 else 0.08,
            timeliness=0.95 if t % 2 == 0 else 0.15,
            timestamp=float(t),
        )
        for t in range(1, 21)
    ]

    result_alt = evaluator.evaluate(
        alternating_records, [],  # no recommendations
        neighbor_h_values=[15, 12, 18, 10, 14],
        neighbor_H_values=[0, 0, 0, 0, 0],
        t_current=20.0,
    )
    print(f"\nAlternating (on/off) node:")
    print(f"  Global trust : {result_alt['global_trust']}  → {classify_trust(result_alt['global_trust'])}")
    print(f"  Stability    : {result_alt['stability']}")
    print(f"  Reliability  : {result_alt['reliability_series']}")

    # --- Trust evolution over 30 iterations for a benign node ---
    print(f"\nTrust evolution — benign node (20 iterations, no peer recommendations):")
    records_so_far = []
    for t in range(1, 21):
        records_so_far.append(
            BehaviorRecord(node_id=4, abb=0.92, timeliness=0.95, timestamp=float(t))
        )
        result = evaluator.evaluate(
            records_so_far, [],
            neighbor_h_values=[10] * 5,
            neighbor_H_values=[0] * 5,
            t_current=float(t),
        )
        label = classify_trust(result["global_trust"])
        bar = "█" * int(result["global_trust"] * 20)
        print(f"  iter {t:2d}: {result['global_trust']:.3f} [{bar:<20}] {label}")

    print("\nDone.")
