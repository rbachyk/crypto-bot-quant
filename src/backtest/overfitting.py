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


def expected_max_sharpe(trial_sharpes: list[float]) -> float:
    """Expected maximum of N trial Sharpes under the no-edge null (Bailey & LdP) — the benchmark
    a selected winner must beat. 0.0 when there was no genuine selection (fewer than two trials,
    or no spread across the trials → single effective trial)."""
    n_trials = len(trial_sharpes)
    if n_trials < 2:
        return 0.0
    sr_var = statistics.variance(trial_sharpes)
    if sr_var <= 0:
        return 0.0
    return math.sqrt(sr_var) * (
        (1 - _EULER) * _phi_inv(1 - 1.0 / n_trials)
        + _EULER * _phi_inv(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    trial_sharpes: list[float], n_obs: int, *, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Deflated Sharpe ratio: PSR of the BEST trial against the expected max of N trials.

    Corrects for selection across the ``len(trial_sharpes)`` configurations tried — the more
    trials, the higher the benchmark the winner must beat to be credible (Section 16)."""
    if not trial_sharpes:
        return 0.0
    observed = max(trial_sharpes)
    return probabilistic_sharpe_ratio(
        observed, expected_max_sharpe(trial_sharpes), n_obs, skew, kurtosis
    )


def trade_sharpe(returns: list[float]) -> float:
    """Per-observation Sharpe (mean / sample std) of a per-trade return / R-multiple series."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return mean / math.sqrt(var) if var > 0 else 0.0


def deflated_sharpe_from_returns(
    returns: list[float], trial_sharpes: list[float] | None = None
) -> float:
    """Deflated Sharpe computed from the POOLED per-trade return sample (the statistically
    meaningful form — H3 fix): PSR of the sample's own per-trade Sharpe, with

    * ``n_obs`` = the EFFECTIVE sample size (trade count discounted for autocorrelation), so a
      500-trade edge and a 25-trade fluke with the same mean are no longer indistinguishable;
    * skew/kurtosis estimated from the same sample (fat-tailed R-multiples widen the PSR band);
    * benchmark = the expected max of the ``trial_sharpes`` genuinely COMPARED during selection
      (same per-trade-Sharpe units). With fewer than two trials the deflation term collapses to
      plain PSR against 0 — a single-config harness gets no artificial deflation and no credit.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 <= 0:
        # Degenerate sample (every trade identical) — nothing statistical to say.
        return 1.0 if mean > 0 else 0.0
    sd = math.sqrt(m2)
    skew = sum((r - mean) ** 3 for r in returns) / n / sd**3
    kurt = sum((r - mean) ** 4 for r in returns) / n / sd**4
    n_eff = int(round(effective_sample_size(returns)))
    return probabilistic_sharpe_ratio(
        trade_sharpe(returns), expected_max_sharpe(list(trial_sharpes or [])), n_eff, skew, kurt
    )


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
    """Bundle the Section-16 controls for a report block.

    ``returns`` is the POOLED per-trade OOS return sample (R-multiples across all folds) that the
    significance is computed on; ``trial_sharpes`` are the per-trade Sharpe proxies of the
    configurations genuinely COMPARED during selection (selection breadth — e.g. the long/short
    side variants the side decision chose between). Zero or one trial ⇒ plain PSR, no deflation."""
    return OverfittingSummary(
        deflated_sharpe=deflated_sharpe_from_returns(returns, trial_sharpes),
        effective_sample_size=effective_sample_size(returns),
        sample_adequacy=sample_adequacy(n_trades),
        n_trials=max(1, len(trial_sharpes)),
    )
