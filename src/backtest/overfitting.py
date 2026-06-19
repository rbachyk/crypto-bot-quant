"""Anti-overfitting controls (AGENTS.md Section 16).

Deterministic, dependency-free implementations of the controls the spec requires beyond
walk-forward + locked hold-out:

* **Deflated Sharpe ratio** (Bailey & López de Prado) — the probability the strategy's true
  Sharpe is positive AFTER deflating for the number of trials tried (multiple-testing
  correction). The more configurations searched, the higher the bar.
* **Probabilistic Sharpe ratio** — the same machinery against an explicit benchmark.
* **Effective sample size** — discounts autocorrelated returns so a long but serially
  dependent series is not mistaken for many independent observations.
* **Purged + embargoed K-fold** — CV splits that purge train/test overlap and embargo the
  bars adjacent to each test fold (no leakage across the boundary).
* **Sample adequacy** — the spec's 300+/100+/30 "robust/limited/inconclusive" thresholds.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

_EULER = 0.5772156649015329  # Euler–Mascheroni constant
_SQRT2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _phi_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's algorithm; good to ~1e-9)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > benchmark) given the sample's higher moments (Bailey & LdP)."""
    if n_obs < 2:
        return 0.0
    den = math.sqrt(
        max(1e-12, 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2)
    )
    return _phi((observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / den)


def deflated_sharpe_ratio(
    trial_sharpes: list[float], n_obs: int, *, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Deflated Sharpe ratio: PSR of the BEST trial against the expected max of N trials.

    Corrects for selection across the ``len(trial_sharpes)`` configurations tried — the more
    trials, the higher the benchmark the winner must beat to be credible (Section 16)."""
    n_trials = len(trial_sharpes)
    if n_trials == 0:
        return 0.0
    observed = max(trial_sharpes)
    sr_var = statistics.variance(trial_sharpes) if n_trials > 1 else 0.0
    if sr_var <= 0:
        # No spread across trials → benchmark is 0 (single effective trial).
        return probabilistic_sharpe_ratio(observed, 0.0, n_obs, skew, kurtosis)
    # Expected maximum of N standard-normal trial Sharpes (Bailey & LdP).
    sr_star = math.sqrt(sr_var) * (
        (1 - _EULER) * _phi_inv(1 - 1.0 / n_trials)
        + _EULER * _phi_inv(1 - 1.0 / (n_trials * math.e))
    )
    return probabilistic_sharpe_ratio(observed, sr_star, n_obs, skew, kurtosis)


def effective_sample_size(returns: list[float]) -> float:
    """N discounted for lag-1 autocorrelation: N_eff = N·(1−ρ₁)/(1+ρ₁), clamped to [1, N]."""
    n = len(returns)
    if n < 3:
        return float(n)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns)
    if var <= 0:
        return float(n)
    cov1 = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n))
    rho1 = cov1 / var
    n_eff = n * (1.0 - rho1) / (1.0 + rho1) if (1.0 + rho1) > 1e-9 else float(n)
    return float(min(n, max(1.0, n_eff)))


def purged_kfold_indices(
    n: int, k: int, *, embargo_frac: float = 0.01
) -> list[tuple[list[int], list[int]]]:
    """Contiguous K-fold splits with purge + embargo around each test fold (no leakage)."""
    if k < 2 or n < k:
        raise ValueError("need k >= 2 and n >= k")
    embargo = int(round(n * embargo_frac))
    fold = n // k
    out: list[tuple[list[int], list[int]]] = []
    for i in range(k):
        start = i * fold
        end = n if i == k - 1 else (i + 1) * fold
        test = list(range(start, end))
        # Purge the test span and embargo the bars on either side from train.
        lo = max(0, start - embargo)
        hi = min(n, end + embargo)
        train = [j for j in range(n) if j < lo or j >= hi]
        out.append((train, test))
    return out


def sample_adequacy(n_trades: int) -> str:
    """Section-16 sample-size verdict for a result."""
    if n_trades >= 300:
        return "robust"
    if n_trades >= 100:
        return "limited"
    if n_trades >= 30:
        return "minimal"
    return "inconclusive"


@dataclass(slots=True)
class OverfittingSummary:
    deflated_sharpe: float
    effective_sample_size: float
    sample_adequacy: str
    n_trials: int

    def to_dict(self) -> dict:
        return {
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "effective_sample_size": round(self.effective_sample_size, 2),
            "sample_adequacy": self.sample_adequacy,
            "n_trials": self.n_trials,
        }


def overfitting_summary(
    trial_sharpes: list[float], returns: list[float], n_trades: int
) -> OverfittingSummary:
    """Bundle the Section-16 controls for a report block."""
    n_obs = max(len(returns), 2)
    return OverfittingSummary(
        deflated_sharpe=deflated_sharpe_ratio(trial_sharpes or [0.0], n_obs),
        effective_sample_size=effective_sample_size(returns),
        sample_adequacy=sample_adequacy(n_trades),
        n_trials=len(trial_sharpes),
    )
