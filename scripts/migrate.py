"""Bring the database to this image's migration head — the compose `migrate` service.

Plain ``alembic upgrade head`` is almost right. The one case it gets operationally wrong is a
database that is AHEAD of the migrations in this image: alembic exits 255 with "Can't locate
revision identified by '00NN'". Since every app service waits on this one completing, that would
turn two ordinary situations into a total outage:

* ``docker compose up -d`` without ``--build`` after someone migrated the database by hand;
* a deliberate ROLLBACK to an older image — exactly when you least want the stack refusing to boot.

A database ahead of the code is not a schema the code can repair by migrating, and for the additive
migrations this project writes, older code reading a newer schema is fine. So that case WARNS and
proceeds, naming both revisions. Everything else — a database behind the head, an unreachable
database, a broken migration — fails loudly and holds the stack, which is the point of the gate.
"""

from __future__ import annotations

import sys

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from src.config.settings import REPO_ROOT
from src.db.base import get_engine


def _config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return cfg


def _db_revisions() -> set[str]:
    with get_engine().connect() as conn:
        return {
            row[0]
            for row in conn.execute(text("SELECT version_num FROM alembic_version"))
            if row[0]
        }


def main() -> int:
    cfg = _config()
    try:
        command.upgrade(cfg, "head")
    except Exception as exc:  # noqa: BLE001 - classified below; re-raised unless DB-is-ahead
        try:
            known = {r.revision for r in ScriptDirectory.from_config(cfg).walk_revisions()}
            current = _db_revisions()
        except Exception:  # noqa: BLE001 - can't classify → the original failure stands
            print(f"MIGRATION FAILED: {exc}", file=sys.stderr)
            return 1
        unknown = current - known
        if unknown:
            print(
                f"WARNING: the database is at {sorted(current)}, which this image does not know "
                f"(its migrations end at {sorted(ScriptDirectory.from_config(cfg).get_heads())}). "
                "That means the database is AHEAD of this code — a hand-run migration, or a "
                "rollback to an older image. Nothing to upgrade; starting anyway. Deploy the "
                "matching code, or roll the schema back deliberately.",
                file=sys.stderr,
            )
            return 0
        print(f"MIGRATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"schema at head {sorted(ScriptDirectory.from_config(cfg).get_heads())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
