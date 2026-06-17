"""Order ownership (AGENTS.md Section 7, ORDER-OWN gate).

The bot may only manage orders it created. Every bot order carries a client order
id beginning with ``ORDER_CLIENT_ID_PREFIX`` (which embeds ``BOT_INSTANCE_ID``);
the bot may only cancel/replace/close orders with its own prefix and must halt +
alert on any unknown order or position (Section 7). Emergency-close mode (touching
foreign orders) requires explicit confirmation and is fully audited.
"""

from __future__ import annotations

import itertools

from src.config import Settings


class OwnershipPolicy:
    """Mints prefixed client order ids and decides what the bot owns (Section 7)."""

    def __init__(self, settings: Settings) -> None:
        self.prefix = settings.order_client_id_prefix
        self.bot_instance_id = settings.bot_instance_id
        self.strategy_version = settings.strategy_version
        self.config_version = settings.config_version
        self._counter = itertools.count(1)

    def configured(self) -> bool:
        """True only when both ownership identifiers are set (Section 7)."""
        return bool(self.prefix) and bool(self.bot_instance_id)

    def new_client_id(self, role: str) -> str:
        """Mint a unique, prefixed client order id for an order ``role``.

        ``role`` is e.g. ``entry`` / ``stop`` / ``tp`` / ``trail`` so a bracket's
        legs are distinguishable while all sharing the ownership prefix.
        """
        seq = next(self._counter)
        return f"{self.prefix}{role}_{seq:08d}"

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
