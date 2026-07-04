"""Shared bracket-exit GEOMETRY — the single source of truth for the stop / trailing-stop /
take-profit decision.

Three engines used to carry their own hand-maintained copy of this decision: the per-trade
``BacktestEngine._maybe_exit``, the offline lake-replay walk ``_simulate_replay_exit``, and the
paper/realtime ``PaperTradingEngine.simulate_paper_exits``. They encoded the *same* geometry
(stop-before-take-profit, a trailing stop ratcheted off the pre-bar peak, no intrabar look-ahead)
but as separate code that had to be kept in lock-step by hand — exactly the drift risk the parity
audit flagged. This module holds that geometry ONCE; each engine keeps only its own FILL accounting
(adverse slippage, maker fills, fees, funding, time-stop, end-of-data) around this call.

Pure and dependency-free by design. It answers one question — "did this bar's wick breach a
protective level, and at what price would it fill?" — and nothing about fees, slippage, funding,
time-stops, end-of-data, or ratcheting the peak, all of which are the caller's concern and
legitimately differ per engine.
"""

from __future__ import annotations


def effective_stop(side: int, stop: float, peak: float, trail_dist: float) -> float:
    """The stop ratcheted by the trailing stop: raised toward ``peak`` for a long, lowered for a
    short, by ``trail_dist``. ``peak`` is the best favorable price seen BEFORE this bar (callers
    ratchet it AFTER the checks) so a fresh extreme can't tighten the stop it is then tested against
    — no intrabar look-ahead. Returns the plain ``stop`` when no trail is set (or no stop)."""
    if trail_dist <= 0 or stop <= 0:
        return stop
    return max(stop, peak - trail_dist) if side > 0 else min(stop, peak + trail_dist)


def resolve_bracket_exit(
    side: int,
    high: float,
    low: float,
    *,
    stop: float,
    tp: float,
    peak: float,
    trail_dist: float,
) -> tuple[str | None, float]:
    """Resolve whether this bar breaches the position's protective bracket.

    Returns ``(reason, exit_level)`` — one of ``"stop"`` / ``"trailing_stop"`` / ``"take_profit"``
    with the exact price to fill at — or ``(None, 0.0)`` if the bar survives.

    The STOP is checked BEFORE the take-profit (conservative: when a bar's range spans both, assume
    the stop filled first). ``high``/``low`` are the bar's extremes; pass ``high == low == close``
    for a CLOSE-ONLY check (no intrabar wick — realtime paper before its bar completes). ``tp <= 0``
    means "no fixed take-profit" (the momentum sentinel convention); a positive ``tp`` is a real
    target. The reason is ``"trailing_stop"`` (vs ``"stop"``) whenever the trail has ratcheted the
    effective stop past the fixed stop. Callers own the FILL: an intrabar engine fills at
    ``exit_level``; a close-only engine fills at the bar close.
    """
    eff = effective_stop(side, stop, peak, trail_dist)
    if side > 0:
        if eff > 0 and low <= eff:
            return ("trailing_stop" if eff > stop else "stop"), eff
        if tp > 0 and high >= tp:
            return "take_profit", tp
    else:
        if eff > 0 and high >= eff:
            return ("trailing_stop" if eff < stop else "stop"), eff
        if tp > 0 and low <= tp:
            return "take_profit", tp
    return None, 0.0
