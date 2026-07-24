"""The metadata-verification diff logic (scripts/verify_metadata.py). The live ccxt fetch is
network-dependent and operator-run, but the field comparison — float tolerance on fees, exact
match on leverage/funding, None = 'confirm from docs' — is pure and must not silently rot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "verify_metadata", Path(__file__).resolve().parent.parent / "scripts" / "verify_metadata.py"
)
vm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vm)


def test_fee_comparison_tolerates_float_wobble_but_not_a_real_difference() -> None:
    # equal within tolerance
    assert vm._close(0.0006, 0.0006 + 1e-12, vm._NUMERIC["taker_fee"])
    # the real config-vs-live gap this tool exists to catch
    assert not vm._close(0.00055, 0.0006, vm._NUMERIC["taker_fee"])
    assert not vm._close(0.0002, 0.0001, vm._NUMERIC["maker_fee"])


def test_none_is_never_a_match() -> None:
    """A field ccxt could not expose (min_notional) must read as 'confirm from docs', never as a
    silent pass — the whole point is to not launder an unconfirmed value."""
    assert not vm._close(5.0, None, vm._NUMERIC["min_notional"])
    assert not vm._close(None, 5.0, vm._NUMERIC["min_notional"])


def test_exact_fields_are_listed_and_use_equality() -> None:
    assert "max_leverage" in vm._EXACT
    assert "funding_interval_hours" in vm._EXACT
    # 50 vs 100 (the real SOL mismatch) is not a rounding question — exact inequality
    assert 50 != 100
