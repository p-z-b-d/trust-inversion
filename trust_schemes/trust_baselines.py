"""
Baseline Trust Evaluation Methods for Comparison
=================================================
Three baselines implemented against the proposed method:

  1. Subjective Logic (SL) — Kang et al. (2019)
     The primary baseline from the TFL-DT paper.
     Opinion = (belief, disbelief, uncertainty, base_rate)
     Supports recency-weighted interaction timeline.

  2. Beta Reputation System (Beta) — Gholami et al. (2021)
     Beta(alpha, beta) distribution over positive/negative counts.
     Optional time-decay on past observations.

  3. Exponential Moving Average (EMA)
     Lightweight baseline. Simple decaying average of
     binary interaction outcomes. Included as the "naive" floor.

All baselines share the same input interface:
    update(abb, timeliness) -> current trust value

Binary classification of interactions (positive/negative):
    positive if abb >= POS_THRESHOLD AND timeliness >= POS_THRESHOLD
    negative otherwise
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# Threshold for converting continuous behavior metrics to binary outcome.
# An interaction is "positive" (well-behaved) only if BOTH metrics meet it.
POS_THRESHOLD = 0.5


def is_positive(abb: float, timeliness: float) -> bool:
    """Convert continuous (abb, timeliness) to a binary good/bad outcome."""
    return abb >= POS_THRESHOLD and timeliness >= POS_THRESHOLD


# ---------------------------------------------------------------------------
# Baseline 1 — Subjective Logic (Kang et al. style)
# ---------------------------------------------------------------------------

class SubjectiveLogicTrust:
    """
    Subjective Logic based trust evaluation.
    
    Reference: Kang et al., "Incentive Mechanism for Reliable Federated 
    Learning", IEEE IoT Journal 2019.
    
    Each interaction produces a binary outcome (positive/negative).
    The opinion (b, d, u) is updated accordingly. Trust is the 
    projected probability: T = b + a * u.
    
    Recency weighting: recent interactions contribute more via a
    decay factor rho applied to the accumulated counts.

    Parameters
    ----------
    base_rate : float
        Prior probability a (base rate), default 0.5.
    rho : float
        Recency decay applied to historical counts each round (0,1].
        rho=1.0 → no decay (pure count accumulation).
        rho<1 → older evidence down-weighted.
    init_trust : float
        Starting trust (corresponds to initial u=1, b=d=0).
    """

    def __init__(
        self,
        base_rate: float = 0.5,
        rho: float = 0.9,
        init_trust: float = 0.5,
    ):
        self.a = base_rate
        self.rho = rho
        # Weighted positive and negative counts
        self.r: float = 0.0   # positive evidence
        self.s: float = 0.0   # negative evidence
        self._history: list[float] = []

    def update(self, abb: float, timeliness: float) -> float:
        """
        Process one interaction and return updated trust.
        """
        # Apply recency decay to existing counts
        self.r *= self.rho
        self.s *= self.rho

        if is_positive(abb, timeliness):
            self.r += 1.0
        else:
            self.s += 1.0

        trust = self._projected_probability()
        self._history.append(trust)
        return trust

    def _projected_probability(self) -> float:
        """
        Subjective logic projected probability:
            b = r / (r + s + 2)
            d = s / (r + s + 2)
            u = 2 / (r + s + 2)
            T = b + a * u
        """
        denom = self.r + self.s + 2.0
        b = self.r / denom
        u = 2.0 / denom
        return b + self.a * u

    @property
    def trust(self) -> float:
        return self._history[-1] if self._history else 0.5 + self.a * 0.5

    @property
    def history(self) -> list[float]:
        return self._history

    def reset(self):
        self.r = 0.0
        self.s = 0.0
        self._history = []


# ---------------------------------------------------------------------------
# Baseline 2 — Beta Reputation System (Gholami et al. style)
# ---------------------------------------------------------------------------

class BetaReputationTrust:
    """
    Beta distribution based trust / reputation.
    
    Reference: Gholami et al., "A Trust Evaluation Scheme for Users
    Involved in Federated Learning", 2021.
    
    Trust = E[Beta(alpha, beta)] = alpha / (alpha + beta)
    where alpha = 1 + weighted_positive_count
          beta  = 1 + weighted_negative_count

    The Beta(1,1) prior (uniform) gives initial trust = 0.5.

    Parameters
    ----------
    decay : float
        Time-decay multiplier applied to counts each round.
        decay=1.0 → no decay (standard Beta reputation).
        decay<1   → older evidence down-weighted.
    """

    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.pos: float = 0.0   # accumulated positive evidence
        self.neg: float = 0.0   # accumulated negative evidence
        self._history: list[float] = []

    def update(self, abb: float, timeliness: float) -> float:
        # Decay existing evidence
        self.pos *= self.decay
        self.neg *= self.decay

        if is_positive(abb, timeliness):
            self.pos += 1.0
        else:
            self.neg += 1.0

        trust = self._expected_value()
        self._history.append(trust)
        return trust

    def _expected_value(self) -> float:
        """E[Beta(alpha, beta)] = alpha / (alpha + beta)"""
        alpha = 1.0 + self.pos
        beta  = 1.0 + self.neg
        return alpha / (alpha + beta)

    @property
    def trust(self) -> float:
        return self._history[-1] if self._history else 0.5

    @property
    def history(self) -> list[float]:
        return self._history

    def reset(self):
        self.pos = 0.0
        self.neg = 0.0
        self._history = []


# ---------------------------------------------------------------------------
# Baseline 3 — Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------

class EMATrust:
    """
    Exponential Moving Average trust — the "naive" baseline.
    
    T_new = (1 - alpha) * T_old + alpha * outcome
    where outcome = 1.0 (positive) or 0.0 (negative)

    Parameters
    ----------
    alpha : float
        Learning rate / smoothing factor in (0, 1].
        Higher alpha = more reactive to recent behavior.
    init_trust : float
        Starting trust value.
    """

    def __init__(self, alpha: float = 0.2, init_trust: float = 0.5):
        self.alpha = alpha
        self._trust = init_trust
        self._history: list[float] = []

    def update(self, abb: float, timeliness: float) -> float:
        outcome = 1.0 if is_positive(abb, timeliness) else 0.0
        self._trust = (1.0 - self.alpha) * self._trust + self.alpha * outcome
        self._history.append(self._trust)
        return self._trust

    @property
    def trust(self) -> float:
        return self._trust

    @property
    def history(self) -> list[float]:
        return self._history

    def reset(self):
        self._trust = 0.5
        self._history = []


# ---------------------------------------------------------------------------
# Convergence speed metric
# ---------------------------------------------------------------------------

def iterations_to_threshold(
    series: list[float],
    threshold: float,
    direction: str = "above",
    sustained: int = 3,
) -> Optional[int]:
    """
    Return the first iteration at which the trust value crosses
    `threshold` and stays there for `sustained` consecutive iterations.
    Returns None if never reached.
    
    direction: "above" (benign detection) or "below" (malicious detection)
    """
    count = 0
    for i, v in enumerate(series):
        if direction == "above":
            crossed = v >= threshold
        else:
            crossed = v <= threshold

        if crossed:
            count += 1
            if count >= sustained:
                return i - sustained + 2  # 1-based iteration number
        else:
            count = 0
    return None


def detection_delay(series: list[float], threshold: float = 0.4) -> Optional[int]:
    """
    For a malicious node: how many iterations until trust drops below
    `threshold` and stays there (sustained=3)?
    """
    return iterations_to_threshold(series, threshold, direction="below", sustained=3)


def convergence_speed(series: list[float], threshold: float = 0.8) -> Optional[int]:
    """
    For a benign node: how many iterations to reach `threshold`?
    """
    return iterations_to_threshold(series, threshold, direction="above", sustained=3)


# ---------------------------------------------------------------------------
# Quantitative comparison table
# ---------------------------------------------------------------------------

def compare_methods(
    patterns: dict,
    n_iter: int = 60,
    proposed_series: dict = None,
) -> dict:
    """
    Run all three baselines + proposed method across behavior patterns.
    
    Args:
        patterns : dict mapping name → callable(t) → (abb, timeliness)
        n_iter   : number of iterations to simulate
        proposed_series : pre-computed proposed method series dict
        
    Returns:
        results dict: {pattern_name: {method_name: series}}
    """
    results = {}

    for pat_name, pat_fn in patterns.items():
        results[pat_name] = {}

        # Baselines
        sl   = SubjectiveLogicTrust(rho=0.9)
        beta = BetaReputationTrust(decay=0.95)
        ema  = EMATrust(alpha=0.2)

        sl_series, beta_series, ema_series = [], [], []

        for t in range(1, n_iter + 1):
            abb, tim = pat_fn(t)
            sl_series.append(sl.update(abb, tim))
            beta_series.append(beta.update(abb, tim))
            ema_series.append(ema.update(abb, tim))

        results[pat_name]["Subjective Logic"] = sl_series
        results[pat_name]["Beta Reputation"]  = beta_series
        results[pat_name]["EMA"]              = ema_series

        if proposed_series and pat_name in proposed_series:
            results[pat_name]["Proposed"] = proposed_series[pat_name]

    return results


if __name__ == "__main__":
    # Quick smoke test
    print("Subjective Logic (benign node, 20 iters):")
    sl = SubjectiveLogicTrust(rho=0.9)
    for t in range(1, 21):
        v = sl.update(abb=0.92, timeliness=0.95)
        bar = "█" * int(v * 20)
        print(f"  iter {t:2d}: {v:.3f} [{bar:<20}]")

    print("\nBeta Reputation (alternating node, 20 iters):")
    beta = BetaReputationTrust(decay=0.95)
    for t in range(1, 21):
        abb = 0.92 if t % 2 == 0 else 0.08
        tim = 0.95 if t % 2 == 0 else 0.15
        v = beta.update(abb, tim)
        print(f"  iter {t:2d}: {v:.3f}")
