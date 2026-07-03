"""Order ownership (AGENTS.md Section 7, ORDER-OWN gate).

The bot may only manage orders it created. Every bot order carries a client order
id beginning with ``ORDER_CLIENT_ID_PREFIX`` (which embeds ``BOT_INSTANCE_ID``);
the bot may only cancel/replace/close orders with its own prefix and must halt +
alert on any unknown order or position (Section 7). Emergency-close mode (touching
foreign orders) requires explicit confirmation and is fully audited.
"""

from __future__ import annotations

import itertools
import secrets

from src.config import Settings

# Bybit rejects an orderLinkId longer than 36 characters — enforce at mint time so a
# too-long id fails loudly here, not as a runtime order rejection on the exchange.
MAX_CLIENT_ID_LEN = 36


class OwnershipPolicy:
    """Mints prefixed client order ids and decides what the bot owns (Section 7)."""

    def __init__(self, settings: Settings) -> None:
        self.prefix = settings.order_client_id_prefix
        self.bot_instance_id = settings.bot_instance_id
        self.strategy_version = settings.strategy_version
        self.config_version = settings.config_version
        # Per-process session token: ids must stay unique ACROSS RESTARTS. A bare
        # counter resets every process start, so a restarted bot would re-mint
        # ``…entry_00000001`` while the previous session's identically-named orders may
        # still rest on the exchange (Bybit rejects a duplicate orderLinkId among active
        # orders → a re-minted stop gets REJECTED and the position runs unprotected).
        # The token lives in the SEQUENCE part only — ownership recognition/adoption
        # keys off the stable prefix (``is_own``), so prior-session orders still match.
        self._session_token = secrets.token_hex(3)
        self._counter = itertools.count(1)

    def configured(self) -> bool:
        """True only when both ownership identifiers are set (Section 7)."""
        return bool(self.prefix) and bool(self.bot_instance_id)

    def new_client_id(self, role: str) -> str:
        """Mint a unique, prefixed client order id for an order ``role``.

        ``role`` is e.g. ``entry`` / ``stop`` / ``tp`` / ``trail`` so a bracket's
        legs are distinguishable while all sharing the ownership prefix. Format is
        ``{prefix}{role}_{token}{seq:06d}`` — the 6-char session token makes ids
        collision-free across process restarts (with the default 14-char prefix and
        the longest role this is 32 chars, inside Bybit's 36-char orderLinkId limit).
        """
        seq = next(self._counter)
        cid = f"{self.prefix}{role}_{self._session_token}{seq:06d}"
        if len(cid) > MAX_CLIENT_ID_LEN:
            raise ValueError(
                f"client order id exceeds the exchange's {MAX_CLIENT_ID_LEN}-char "
                f"orderLinkId limit ({len(cid)}): {cid!r} — shorten ORDER_CLIENT_ID_PREFIX"
            )
        return cid

    def is_own(self, client_id: str | None) -> bool:
        """True iff ``client_id`` carries this bot's ownership prefix."""
        if not client_id or not self.prefix:
            return False
        return client_id.startswith(self.prefix)

    def tags(self, parent_id: str | None = None) -> dict[str, str]:
        """Provenance every bot order carries (Section 17 reconciliation)."""
        tags = {
            "bot_instance_id": self.bot_instance_id,
            "strategy_version": self.strategy_version,
            "config_version": self.config_version,
        }
        if parent_id is not None:
            tags["parent_id"] = parent_id
        return tags
