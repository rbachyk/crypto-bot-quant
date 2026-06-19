"""Job-queue routing + Redis key layout (AGENTS.md Appendix B.6/B.13).

Two properties this module encodes:

* **Isolation / performance** — jobs are routed to a *per-class* Redis queue
  (``qbot:queue:<class>``) so heavy ML/RL/backtest work runs on its own dedicated
  worker(s) and can never starve light data/gate jobs behind a single shared FIFO.
* **Reliability** — workers consume via a per-worker *processing* list
  (``qbot:processing:<worker_id>``) using ``RPOPLPUSH``, and publish a TTL liveness
  beacon (``qbot:worker:<worker_id>``). If a worker dies mid-job the job id survives
  in its processing list and the reaper re-queues it once the beacon expires — a job
  is never silently lost (the old ``BRPOP`` removed the id before the DB row moved).
"""

from __future__ import annotations

QUEUE_PREFIX = "qbot:queue"
PROCESSING_PREFIX = "qbot:processing"
WORKER_PREFIX = "qbot:worker"
REAPER_LOCK_KEY = "qbot:reaper:lock"

DEFAULT_CLASS = "default"

# Queue classes a worker can serve, in the priority order a "serve everything" worker
# (the default, and any single dev worker) drains them — heavy classes first so they are
# never starved, ``default`` last.
ALL_CLASSES: tuple[str, ...] = ("ml", "rl", "backtest", "live", "data", "gates", DEFAULT_CLASS)

# job_type -> class. Each entry is an exact name or a ``prefix_`` (trailing underscore =
# startswith match). Evaluated in order; first match wins, else DEFAULT_CLASS.
_CLASS_MEMBERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ml", ("build_ml_", "train_ml_", "run_ml_", "run_lake_ml_", "evaluate_ml_models")),
    ("rl", ("train_rl_", "run_rl_", "simulate_rl_", "rl_")),
    # A live/demo run is a long-lived loop; give it a dedicated worker so it never blocks
    # (or is blocked by) backtest/data jobs. reset_env_stats rides the same class (light).
    ("live", ("run_live_session", "reset_env_stats")),
    (
        "backtest",
        (
            "run_backtest",
            "run_lake_backtest",
            "run_lake_paper_session",
            "run_walk_forward",
            "build_dataset_version",
            "run_strategy_validation",
            "run_lake_strategy_validation",
        ),
    ),
    (
        "data",
        (
            "download_",
            "repair_",
            "validate_data_quality",
            "sync_exchange_metadata",
            "verify_exchange_metadata",
            "build_symbol_universe",
            "build_feature_store",
            "run_feature_leakage_test",
        ),
    ),
    ("gates", ("run_gate", "run_all_gates", "run_backup_check", "run_restore_test_check")),
)


def queue_class(job_type: str) -> str:
    """Route a job_type to its queue class (Appendix B.13)."""
    for cls, members in _CLASS_MEMBERS:
        for m in members:
            if job_type == m or (m.endswith("_") and job_type.startswith(m)):
                return cls
    return DEFAULT_CLASS


def queue_key(cls: str) -> str:
    return f"{QUEUE_PREFIX}:{cls}"


def processing_key(worker_id: str) -> str:
    return f"{PROCESSING_PREFIX}:{worker_id}"


def worker_key(worker_id: str) -> str:
    return f"{WORKER_PREFIX}:{worker_id}"


def parse_queue_classes(spec: str | None) -> list[str]:
    """Parse a comma-separated queue spec into a validated, de-duplicated class list.

    Empty/None → every class (a worker that serves everything, preserving the old
    single-queue behaviour for a lone/dev worker). Unknown names are ignored; if nothing
    valid remains we fall back to all classes so a worker is never left serving nothing."""
    if not spec or not spec.strip():
        return list(ALL_CLASSES)
    out: list[str] = []
    for raw in spec.split(","):
        c = raw.strip().lower()
        if c in ALL_CLASSES and c not in out:
            out.append(c)
    return out or list(ALL_CLASSES)
