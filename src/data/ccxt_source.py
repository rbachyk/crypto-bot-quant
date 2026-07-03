"""Real ccxt-backed market-data source (AGENTS.md Section 5/6, Appendix C).

Implements the :class:`~src.data.source.DataSource` interface against a live exchange (default
Bybit, USDT-margined linear perps) using public REST — no API keys required for market data.
It returns rows in the exact canonical schema (``src.data.schema.COLUMNS``) on the timeframe
grid, so the existing ingest → backfill → validate → DATA_VERSION-snapshot pipeline runs over
real data with no other changes. Pagination + rate-limiting are handled here; downstream code
never sees ccxt.

Coverage note: Bybit public REST serves OHLCV, mark/index klines and funding-rate history over
long windows. Two series are constrained:

* ``spread`` — Bybit serves no historical L1 (bid/ask), so this is a **conservative estimate**
  derived from each candle close (real backtests must not understate costs — the FEE/SLIP stress
  gates add further margin).
* ``open_interest`` — Bybit's OI-history endpoint ignores ``since`` and returns only the most
  recent ~200 records, so lookback is **bounded by the sampling interval**: ~16h at ``5m``,
  ~8d at ``1h``, ~33d at ``4h``, ~199d at ``1d``. Historical windows older than that simply have
  no OI; the validator (correctly) flags it. Sample OI coarsely, or treat it as best-effort.

Kline timing rules (safety-critical for the append-only store):

* **Only CLOSED klines are emitted.** The venue returns the still-forming bar; the store's
  keep-first-write dedup would freeze that partial OHLCV forever (incremental update never
  re-reads it, repair sees no gap). Any kline whose close time (``open + interval``) has not
  passed yet is dropped at this fetch layer, for every kline-derived series (ohlcv, mark,
  index, spread) using that series' own interval.
* **Kline-derived point samples are stamped at the kline CLOSE time.** A mark/index close (and
  the spread estimated from it) is observable exactly when the kline closes, so the row is
  stamped ``open + interval`` — never at the open, which would hand decision-time consumers a
  value realized up to one interval in their future (see ``src.data.schema`` module docs).
  Funding rows are settlement events and open interest is a snapshot; both are realized AT
  their own timestamp and keep it.

Order-book and liquidation history are not collected (AGENTS.md "if available").
"""

from __future__ import annotations

import time
from typing import Any

from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    SeriesKey,
    timeframe_ms,
)
from src.data.source import DataSource

# Bybit v5 kline endpoints cap a page at 1000 rows.
_PAGE = 1000
# Conservative default L1 spread estimate (bps) when historical bid/ask is unavailable.
_DEFAULT_SPREAD_BPS = 5.0
# ccxt timeframe strings match our schema labels (1m/5m/1h/...), so no translation needed.


class CcxtDataSource(DataSource):
    """A real exchange data source behind the DataSource interface (default: Bybit swaps)."""

    def __init__(
        self,
        exchange_id: str = "bybit",
        *,
        estimated_spread_bps: float = _DEFAULT_SPREAD_BPS,
        enable_rate_limit: bool = True,
        client: Any | None = None,
        rate_limit_ms: int = 300,
        max_retries: int = 8,
        retry_base_sec: float = 1.0,
        retry_max_sec: float = 60.0,
        exchange_env: str = "live",
    ) -> None:
        self.exchange_id = exchange_id
        self._estimated_spread_bps = estimated_spread_bps
        # Rate-limit resilience: ccxt's enableRateLimit spaces requests (rate_limit_ms apart), and
        # on top of that every network call is retried with exponential backoff when Bybit returns
        # "Too many visits" (retCode 10006) or a transient network error — so a long multi-year
        # download throttles itself and resumes instead of failing.
        self._max_retries = max_retries
        self._retry_base_sec = retry_base_sec
        self._retry_max_sec = retry_max_sec
        self._ex: Any
        if client is not None:
            self._ex = client  # injected (tests)
        else:
            import ccxt

            klass = getattr(ccxt, exchange_id)
            self._ex = klass(
                {
                    "enableRateLimit": enable_rate_limit,
                    "rateLimit": max(rate_limit_ms, 1),  # ms between requests (conservative)
                    "options": {"defaultType": "swap"},
                }
            )
            # Route market data to the right environment for TESTNET (its klines must match the
            # testnet venue the orders hit). demo + live deliberately read MAINNET data — Bybit's
            # demo endpoint serves no public klines (the silent tick-0 bug), and demo trading is
            # mainnet-data by design; live is mainnet anyway.
            if exchange_env == "testnet":
                from src.execution.live_venue import apply_exchange_env

                apply_exchange_env(self._ex, "testnet")
        self._markets: dict | None = None

    # -- rate-limit-resilient call wrapper ------------------------------- #
    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke a ccxt method, backing off + retrying on rate-limit / transient network errors.

        Bybit replies retCode 10006 ("Too many visits") under burst load; ccxt raises
        ``RateLimitExceeded``. We sleep ``retry_base_sec`` and double up to ``retry_max_sec`` per
        retry, so the download self-throttles to the exchange's pace rather than crashing."""
        delay = self._retry_base_sec
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - classify by type/message, re-raise if final
                names = {c.__name__ for c in type(exc).__mro__}
                msg = str(exc).lower()
                # ccxt error TYPES that are always transient.
                by_type = bool(
                    names
                    & {
                        "RateLimitExceeded", "DDoSProtection", "NetworkError",
                        "ExchangeNotAvailable", "RequestTimeout", "OnMaintenance",
                    }
                )
                # Bybit retCodes / HTTP throttle + 5xx signatures that ccxt may wrap as a generic
                # ExchangeError (10006 too-many-visits, 10016 system busy, 10018 IP rate limit,
                # 10002 request-expired). Matched in the message so we self-throttle instead of
                # crashing a long download on a transient throttle/outage.
                by_msg = any(
                    s in msg
                    for s in (
                        "10006", "10016", "10018", "10002", "too many visits", "rate limit",
                        "too many requests", "system busy", "service unavailable",
                        "http 429", "429 ", "http 503", " 503", "bad gateway", "http 502",
                    )
                )
                if not (by_type or by_msg) or attempt >= self._max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, self._retry_max_sec)
        raise RuntimeError("unreachable")  # pragma: no cover

    # -- symbols --------------------------------------------------------- #
    def _markets_loaded(self) -> dict:
        if self._markets is None:
            self._markets = self._call(self._ex.load_markets)
        return self._markets

    def has_symbol(self, symbol: str) -> bool:
        try:
            return symbol in self._markets_loaded()
        except Exception:  # noqa: BLE001 - reachability failure ⇒ treat as no history
            return False

    # -- fetch ----------------------------------------------------------- #
    def fetch(self, key: SeriesKey, start_ms: int, end_ms: int) -> list[dict]:
        iv = key.interval_ms
        if key.data_type == OHLCV:
            rows = self._ohlcv_rows(key.symbol, key.timeframe, start_ms, end_ms)
        elif key.data_type == MARK:
            rows = self._kline_value(
                key.symbol, key.timeframe, start_ms, end_ms, "mark", "mark_price"
            )
        elif key.data_type == INDEX:
            rows = self._kline_value(
                key.symbol, key.timeframe, start_ms, end_ms, "index", "index_price"
            )
        elif key.data_type == FUNDING:
            rows = self._funding_rows(key, start_ms, end_ms)
        elif key.data_type == OPEN_INTEREST:
            rows = self._open_interest_rows(key.symbol, key.timeframe, start_ms, end_ms)
        elif key.data_type == SPREAD:
            rows = self._spread_rows(key.symbol, key.timeframe, start_ms, end_ms)
        else:  # pragma: no cover - guarded by schema
            raise ValueError(f"unsupported data_type: {key.data_type}")
        # Keep only canonical, grid-aligned rows in range; dedup keeping first.
        seen: set[int] = set()
        out: list[dict] = []
        for r in sorted(rows, key=lambda x: x["ts"]):
            ts = r["ts"]
            if ts < start_ms or ts >= end_ms or ts % iv != 0 or ts in seen:
                continue
            seen.add(ts)
            out.append(r)
        return out

    # -- per-series fetchers (with pagination) --------------------------- #
    def _paginate_ohlcv(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int, params: dict
    ):
        """Yield CLOSED klines with open time in ``[start_ms, end_ms)``, oldest first.

        Closed-bar guard: a kline is immutable only once its close time (``open + interval``)
        has passed — the venue serves the in-progress bar too, and storing it would freeze a
        partial candle forever (append-only keep-first dedup). The guard applies regardless of
        the caller's window math, so every kline-derived series (ohlcv, mark, index, spread)
        is safe even when ``end_ms`` reaches into the still-forming bar."""
        iv_ms = timeframe_ms(timeframe)  # our own grid math (Appendix C), never the library's
        since = max(start_ms, 0)
        last_seen = -1
        while since < end_ms:
            batch = self._call(
                self._ex.fetch_ohlcv, symbol, timeframe, since=since, limit=_PAGE, params=params
            )
            if not batch:
                break
            now_ms = int(time.time() * 1000)  # re-read per page (long downloads span hours)
            for candle in batch:
                if candle[0] >= end_ms or candle[0] + iv_ms > now_ms:
                    return  # out of window, or still forming (batches ascend ⇒ so is the rest)
                yield candle
            new_since = batch[-1][0]
            if new_since <= last_seen:
                break  # no progress ⇒ stop (guards against repeated last page)
            last_seen = new_since
            since = new_since + iv_ms

    def _ohlcv_rows(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> list[dict]:
        rows: list[dict] = []
        for c in self._paginate_ohlcv(symbol, timeframe, start_ms, end_ms, {}):
            rows.append(
                {
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]) if c[5] is not None else 0.0,
                }
            )
        return rows

    def _kline_value(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int, price: str, col: str
    ) -> list[dict]:
        """Mark/index samples from klines, stamped at the kline CLOSE time.

        The close value of the kline opening at ``o`` is observable exactly at ``o + iv``, so
        the row is ``{"ts": o + iv, col: close}`` — stamping it at ``o`` would let decision-time
        as-of joins read a value realized up to one interval in their future (look-ahead). The
        kline window is therefore shifted back one interval so the STAMPED rows cover
        ``[start_ms, end_ms)``; stamps stay on the grid (``o % iv == 0 ⇒ (o + iv) % iv == 0``)."""
        iv = timeframe_ms(timeframe)
        rows: list[dict] = []
        params = {"price": price}
        for c in self._paginate_ohlcv(symbol, timeframe, start_ms - iv, end_ms - iv, params):
            rows.append({"ts": int(c[0]) + iv, col: float(c[4])})
        return rows

    def _funding_rows(self, key: SeriesKey, start_ms: int, end_ms: int) -> list[dict]:
        interval_hours = key.interval_ms // 3_600_000
        rows: list[dict] = []
        # Bybit returns settlements strictly AFTER ``since``, so query one interval
        # early to include the settlement that lands exactly on the window start;
        # the caller's range filter drops anything before ``start_ms``.
        since = max(0, start_ms - key.interval_ms)
        last_seen = -1
        while since < end_ms:
            batch = self._call(
                self._ex.fetch_funding_rate_history, key.symbol, since=since, limit=200
            )
            if not batch:
                break
            for f in batch:
                ts = int(f["timestamp"])
                if ts >= end_ms:
                    break
                rate = f.get("fundingRate")
                if rate is not None:
                    rows.append(
                        {
                            "ts": ts,
                            "funding_rate": float(rate),
                            "funding_interval_hours": interval_hours,
                        }
                    )
            new_since = int(batch[-1]["timestamp"])
            if new_since <= last_seen:
                break
            last_seen = new_since
            since = new_since + key.interval_ms
        return rows

    def _open_interest_rows(self, symbol: str, timeframe: str, start_ms: int, end_ms: int):
        # NB: Bybit ignores ``since`` here and serves only the most recent block (see module
        # docstring). The range filter in fetch() keeps only in-window rows; for older windows
        # this legitimately yields nothing and coverage flags the gap.
        rows: list[dict] = []
        since = start_ms
        last_seen = -1
        while since < end_ms:
            batch = self._call(
                self._ex.fetch_open_interest_history, symbol, timeframe, since=since, limit=200
            )
            if not batch:
                break
            for oi in batch:
                ts = int(oi["timestamp"])
                if ts >= end_ms:
                    break
                amount = oi.get("openInterestAmount")
                if amount is not None:
                    rows.append({"ts": ts, "open_interest": float(amount)})
            new_since = int(batch[-1]["timestamp"])
            if new_since <= last_seen:
                break
            last_seen = new_since
            since = new_since + 1
        return rows

    def _spread_rows(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> list[dict]:
        # Estimated from each candle close (no public historical L1 spread on Bybit). The close
        # is observable at the candle CLOSE, so — like mark/index — the sample is stamped
        # ``open + iv`` and the kline window is shifted back one interval (see _kline_value).
        frac = self._estimated_spread_bps / 10_000.0
        iv = timeframe_ms(timeframe)
        rows: list[dict] = []
        for c in self._paginate_ohlcv(symbol, timeframe, start_ms - iv, end_ms - iv, {}):
            mid = float(c[4])
            bid = mid * (1.0 - frac / 2.0)
            ask = mid * (1.0 + frac / 2.0)
            rows.append(
                {
                    "ts": int(c[0]) + iv,
                    "bid": round(bid, 8),
                    "ask": round(ask, 8),
                    "spread": round(ask - bid, 8),
                    "spread_bps": round(self._estimated_spread_bps, 4),
                }
            )
        return rows

    def ping(self) -> bool:
        try:
            self._markets_loaded()
            return True
        except Exception:  # noqa: BLE001
            return False
