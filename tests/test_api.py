"""API & dashboard-auth tests (AGENTS.md Appendix B.8, B.17).

Includes the dashboard permission tests required by Section 31: the dashboard
shell must reject unauthenticated requests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings

from tests.conftest import requires_db, requires_redis

# Force basic-auth even though tests may run in local env.
_settings = Settings(
    _env_file=None,
    app_env="paper",
    dashboard_auth_mode="basic",
    dashboard_username="admin",
    dashboard_password="secret",
)
client = TestClient(create_app(_settings))
AUTH = ("admin", "secret")


def test_livez_ok() -> None:
    assert client.get("/livez").json() == {"status": "ok"}


def test_dashboard_requires_auth() -> None:
    assert client.get("/").status_code == 401


def test_dashboard_rejects_bad_credentials() -> None:
    assert client.get("/", auth=("admin", "wrong")).status_code == 401


def test_dashboard_renders_with_auth() -> None:
    resp = client.get("/", auth=AUTH)
    assert resp.status_code == 200
    assert "Control Center" in resp.text


def test_path_params_are_escaped_no_reflected_xss() -> None:
    """A path param that reaches the rendered HTML (symbol stats form action, gate id) must be
    HTML-escaped — no reflected XSS into the authenticated control plane."""
    payload = '"><script>alert(1)</script>'
    from urllib.parse import quote

    r = client.get(f"/dashboard/stats/{quote(payload, safe='')}", auth=AUTH)
    assert r.status_code == 200
    assert "<script>alert(1)" not in r.text  # escaped, not reflected raw
    g = client.get(f"/dashboard/gates/{quote(payload, safe='')}", auth=AUTH)
    assert "<script>alert(1)" not in g.text


def test_csrf_blocks_cross_site_post() -> None:
    """A browser-marked cross-site POST (Fetch-Metadata) to a state-changing endpoint is
    rejected even with valid credentials — defends the Basic-auth control plane from CSRF."""
    r = client.post(
        "/api/killswitch/engage",
        auth=AUTH,
        headers={"sec-fetch-site": "cross-site"},
    )
    assert r.status_code == 403


def test_csrf_blocks_foreign_origin_post() -> None:
    r = client.post(
        "/api/killswitch/engage",
        auth=AUTH,
        headers={"origin": "https://evil.example.com"},
    )
    assert r.status_code == 403


def test_csrf_allows_same_origin_and_non_browser_post() -> None:
    """Same-origin (sec-fetch-site=same-origin) and non-browser callers (no fetch-metadata,
    no origin — e.g. the test client / CLI) are allowed through the CSRF guard. A non-raising
    client is used so the assertion isolates the MIDDLEWARE result from whatever the endpoint
    does (it must not be a 403), independent of redis/db availability."""
    from fastapi.testclient import TestClient

    nr = TestClient(create_app(_settings), raise_server_exceptions=False)
    same = nr.post("/api/scheduler/resume", auth=AUTH, headers={"sec-fetch-site": "same-origin"})
    assert same.status_code != 403  # same-origin allowed through the guard
    plain = nr.post("/api/scheduler/resume", auth=AUTH)
    assert plain.status_code != 403  # non-browser (no fetch-metadata/origin) allowed


def test_api_me_requires_auth() -> None:
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", auth=AUTH).json()["user"] == "admin"


@requires_redis
def test_enqueue_unknown_job_rejected() -> None:
    resp = client.post("/api/jobs/not_a_real_job", auth=AUTH)
    assert resp.status_code == 400


def test_run_basket_rejects_non_cross_sectional_strategy() -> None:
    """The basket-paper start endpoint must reject a strategy that isn't cross-sectional BEFORE
    enqueueing (no redis hit) — only funding_carry / residual_momentum-style baskets run here."""
    resp = client.post(
        "/api/paper/run-basket",
        params={"strategy": "lead_lag", "timeframe": ""},
        auth=AUTH,
        headers={"sec-fetch-site": "same-origin"},
    )
    assert resp.status_code == 400


@requires_db
def test_live_page_offers_offline_paper_start() -> None:
    """The Live control offers an offline-PAPER start (mode=paper → SimulatedVenue: continuous, no
    real orders, no account reconciliation to halt on) distinct from the real-venue start — so the
    per-symbol ensemble can paper-trade on demo data instead of placing demo orders."""
    resp = client.get("/dashboard/live", auth=AUTH)
    assert resp.status_code == 200
    assert "Start paper session" in resp.text
    assert "mode=paper&timeframe=" in resp.text       # paper button passes the offline override
    assert "session (real orders)" in resp.text       # the real-venue start remains available


@requires_db
def test_open_positions_persist_and_render() -> None:
    """The basket persist helper writes/clears live open positions and the dashboard renders them
    with unrealized P&L (the feature: see held legs before they close)."""
    from src.db.base import session_scope
    from src.db.models import OpenPosition
    from src.live.basket import _persist_open_positions

    sid = "paper:basket:test_open:v:1h"
    try:
        _persist_open_positions(sid, [{
            "symbol": "ETH/USDT:USDT", "strategy": "funding_carry", "side": 1, "qty": 1.0,
            "entry_price": 100.0, "mark_price": 110.0, "notional": 100.0,
            "unrealized_pnl": 10.0, "entry_ts": 1,
        }])
        with session_scope() as s:
            rows = s.query(OpenPosition).filter_by(session_id=sid).all()
            assert len(rows) == 1 and rows[0].unrealized_pnl == 10.0
        resp = client.get("/dashboard/paper", auth=AUTH)
        assert "Open positions" in resp.text and "ETH/USDT:USDT" in resp.text
        # the auto-refresh fragment shows the position with a per-strategy subtotal + total
        frag = client.get("/api/open-positions", auth=AUTH)
        assert frag.status_code == 200
        assert "ETH/USDT:USDT" in frag.text and "funding_carry subtotal" in frag.text
        assert "Total unrealized" in frag.text
        _persist_open_positions(sid, [])  # empty snapshot clears the panel
        with session_scope() as s:
            assert s.query(OpenPosition).filter_by(session_id=sid).count() == 0
    finally:
        with session_scope() as s:
            s.query(OpenPosition).filter_by(session_id=sid).delete()


@requires_db
def test_api_stats_respects_env_selector() -> None:
    """The /api/stats JSON endpoint must honour the env selector (query param / qbot_env cookie)
    like the HTML pages — otherwise it blends paper/demo/testnet/live into one number that
    disagrees with the dashboard's per-env view."""
    from src.db.base import session_scope
    from src.db.models import PaperTradeRecord

    strat = "apienv_strat"
    with session_scope() as s:
        s.query(PaperTradeRecord).filter_by(strategy=strat).delete()
        s.add(PaperTradeRecord(
            session_id="paper_apienv", trade_id="p0", symbol="BTC/USDT:USDT", strategy=strat,
            side=1, pnl=1.0, pnl_r=0.1, fee=0.0, slippage_cost=0.0, regime="r",
            exit_reason="take_profit",
        ))
        s.add(PaperTradeRecord(
            session_id="demo:apienv", trade_id="d0", symbol="BTC/USDT:USDT", strategy=strat,
            side=1, pnl=2.0, pnl_r=0.1, fee=0.0, slippage_cost=0.0, regime="r",
            exit_reason="take_profit",
        ))
    try:
        paper = client.get(f"/api/stats?env=paper&strategy={strat}", auth=AUTH).json()["trading"]
        demo = client.get(f"/api/stats?env=demo&strategy={strat}", auth=AUTH).json()["trading"]
        assert paper["total_trades"] == 1 and paper["realized_pnl"] == 1.0  # paper only
        assert demo["total_trades"] == 1 and demo["realized_pnl"] == 2.0  # demo only, not blended
    finally:
        with session_scope() as s:
            s.query(PaperTradeRecord).filter_by(strategy=strat).delete()


@requires_db
def test_clear_orphan_open_positions_drops_prior_run_same_stream_only() -> None:
    """REGRESSION (R7): unique-per-run session ids mean a hard-killed run leaves orphan
    open-position rows; on start the new run clears prior runs of the SAME stream (id minus the run
    stamp) without touching a concurrent OTHER stream."""
    from src.db.base import session_scope
    from src.db.models import OpenPosition
    from src.live.basket import _clear_orphan_open_positions, _persist_open_positions

    old = "demo:bybit_0002:20260629T100000-aaaa"  # prior (crashed) run, same stream
    new = "demo:bybit_0002:20260629T120000-bbbb"  # this run (same stream → prefix demo:bybit_0002:)
    other = "paper:basket:funding_carry:bybit_0002:1h:20260629T120000-cccc"  # a different stream
    pos = {"symbol": "ETH/USDT:USDT", "strategy": "lead_lag_xasset", "side": 1, "qty": 1.0,
           "entry_price": 100.0, "mark_price": 100.0, "notional": 100.0, "unrealized_pnl": 0.0,
           "entry_ts": 1}
    try:
        for sid in (old, other):
            _persist_open_positions(sid, [pos])
        _clear_orphan_open_positions(new)  # what the new run does at start
        with session_scope() as s:
            assert s.query(OpenPosition).filter_by(session_id=old).count() == 0  # prior run cleared
            assert s.query(OpenPosition).filter_by(session_id=other).count() == 1  # other kept
    finally:
        with session_scope() as s:
            for sid in (old, new, other):
                s.query(OpenPosition).filter_by(session_id=sid).delete()


@requires_db
def test_open_positions_panel_excludes_real_venue_sessions() -> None:
    """The Paper page Open-positions panel must NOT show demo/testnet/live real-venue legs — the
    per-symbol live loop persists OpenPosition rows for those too, and mixing them into the paper
    panel (and its unrealized total) misreports paper exposure."""
    from src.db.base import session_scope
    from src.db.models import OpenPosition
    from src.live.basket import _persist_open_positions

    demo_sid = "demo:bybit_0002:20260629T120000-aa11"
    try:
        _persist_open_positions(demo_sid, [{
            "symbol": "ZECDEMO/USDT:USDT", "strategy": "lead_lag_xasset", "side": 1, "qty": 1.0,
            "entry_price": 100.0, "mark_price": 110.0, "notional": 100.0,
            "unrealized_pnl": 10.0, "entry_ts": 1,
        }])
        frag = client.get("/api/open-positions", auth=AUTH)
        assert frag.status_code == 200
        assert "ZECDEMO/USDT:USDT" not in frag.text  # real-venue leg excluded from the paper panel
    finally:
        with session_scope() as s:
            s.query(OpenPosition).filter_by(session_id=demo_sid).delete()


@requires_db
@requires_redis
def test_run_basket_rejects_duplicate_session() -> None:
    """A second Start for a strategy that already has a GENUINELY-ALIVE basket session (its job id
    in a live worker's processing list) is refused (409) — no double-booking once sessions run."""
    import uuid

    from src.config import get_settings
    from src.db.base import session_scope
    from src.db.models import Job, JobStatus
    from src.jobs import JobQueue
    from src.jobs.routing import processing_key, worker_key

    rc = JobQueue(get_settings()).redis
    jid = "test-dup-" + uuid.uuid4().hex[:8]
    wid = "test-worker-" + uuid.uuid4().hex[:8]
    with session_scope() as s:
        s.add(Job(
            job_id=jid, job_type="run_basket_paper_session", status=JobStatus.RUNNING,
            input_params={"strategy": "funding_carry"},
        ))
    rc.lpush(processing_key(wid), jid)  # claimed by a worker...
    rc.set(worker_key(wid), wid, ex=60)  # ...whose liveness beacon is fresh ⇒ genuinely alive
    try:
        r = client.post(
            "/api/paper/run-basket",
            params={"strategy": "funding_carry", "timeframe": ""},
            auth=AUTH, headers={"sec-fetch-site": "same-origin"},
        )
        assert r.status_code == 409
    finally:
        rc.delete(processing_key(wid))
        rc.delete(worker_key(wid))
        with session_scope() as s:
            obj = s.get(Job, jid)
            if obj is not None:
                s.delete(obj)


@requires_redis
def test_orphaned_session_is_auto_cleared_and_does_not_block() -> None:
    """A RUNNING session row with NO live owner (worker killed before the terminal write) is a
    ghost: the Start guard self-heals it to FAILED and does NOT block — so a killed session can't
    silently wedge every future Start."""
    import uuid

    from src.config import get_settings
    from src.db.base import session_scope
    from src.db.models import Job, JobStatus
    from src.jobs import JobQueue

    q = JobQueue(get_settings())
    q.redis.delete("qbot:queue:live")  # ensure the ghost isn't sitting in the pending queue
    jid = "test-ghost-" + uuid.uuid4().hex[:8]
    with session_scope() as s:
        s.add(Job(
            job_id=jid, job_type="run_basket_paper_session", status=JobStatus.RUNNING,
            input_params={"strategy": "funding_carry"},
        ))
    new_id = None
    try:
        r = client.post(
            "/api/paper/run-basket",
            params={"strategy": "funding_carry", "timeframe": ""},
            auth=AUTH, headers={"sec-fetch-site": "same-origin"},
        )
        assert r.status_code != 409  # the ghost did NOT block the new session
        with session_scope() as s:
            assert s.get(Job, jid).status == JobStatus.FAILED  # ghost auto-cleared
    finally:
        with session_scope() as s:
            for _id in (jid, new_id):
                obj = s.get(Job, _id) if _id else None
                if obj is not None:
                    s.delete(obj)


@requires_db
def test_dashboard_paper_offers_basket_start_form() -> None:
    """The Paper page exposes the basket launch control with the cross-sectional candidates."""
    resp = client.get("/dashboard/paper", auth=AUTH)
    assert resp.status_code == 200
    assert "Start basket paper session" in resp.text
    assert "funding_carry" in resp.text  # a cross-sectional candidate in the dropdown
    # the form must append the selects to the URL as QUERY params (the app's convention — the
    # endpoint reads query params, not the POST body), or the strategy arrives empty.
    assert "/api/paper/run-basket?strategy=" in resp.text
    assert "basket-strat" in resp.text and "basket-tf" in resp.text


def test_dashboard_killswitch_engage_and_recovery(tmp_path) -> None:
    # Isolated kill switch (own data lake, unreachable redis ⇒ file backend) so the
    # test never touches shared state (AGENTS.md Section 2.2, KILL gate).
    iso = Settings(
        _env_file=None,
        app_env="paper",
        dashboard_auth_mode="basic",
        dashboard_username="admin",
        dashboard_password="secret",
        data_lake_path=tmp_path / "dl",
        redis_url="redis://127.0.0.1:1/0",
    )
    c = TestClient(create_app(iso))

    assert c.post("/api/killswitch/engage").status_code == 401  # auth required
    engaged = c.post("/api/killswitch/engage", auth=AUTH)
    assert engaged.status_code == 200 and engaged.json()["engaged"] is True

    # Recovery requires an explicit manual confirmation (Section 35).
    assert c.post("/api/killswitch/disengage", auth=AUTH).status_code == 400
    assert c.get("/api/killswitch", auth=AUTH).json()["engaged"] is True
    cleared = c.post("/api/killswitch/disengage?confirm=true", auth=AUTH)
    assert cleared.status_code == 200 and cleared.json()["engaged"] is False


@requires_db
def test_approvals_create_list_decide_loop(monkeypatch) -> None:
    # The approvals surface is fully wired: an operator can REQUEST an approval, see it
    # pending, and approve it (previously the table was read by the UI but never written).
    import uuid

    from src.api.stats import GateStats

    # Make the activation precondition deterministic regardless of test order: force gates green
    # and persist a real-lake promotion (live activation requires both).
    monkeypatch.setattr(
        "src.api.stats.compute_gate_stats",
        lambda *_a, **_k: GateStats(
            total_critical_gates=20, critical_gates_passed=20, live_readiness_score=100.0
        ),
    )

    # Live activation now requires an active strategy validated on REAL lake data (Section 13);
    # persist one so the activation request can be built.
    from src.strategies.promotion import persist_validations
    from src.strategies.research import CandidateValidation, SideDecision

    sd = SideDecision(
        allow_long=True, allow_short=False, long_expectancy_r=0.2, short_expectancy_r=-0.1,
        long_trades=30, short_trades=5, disabled=["short"],
    )
    persist_validations(
        [
            CandidateValidation(
                candidate_id="basis_reversion", family="B",
                strategy_version=_settings.strategy_version, promoted=True, status="promoted",
                shelved_reasons=[], side_decision=sd, hypothesis={}, report={"expectancy_r": 0.2},
                walk_forward={}, fee_stress={}, slippage_stress={}, noise_control={},
            )
        ],
        data_source="lake",
    )

    sid = f"LIVE-{uuid.uuid4().hex[:8]}"
    created = client.post(
        f"/api/approvals?subject_type=live_activation&subject_id={sid}", auth=AUTH
    )
    assert created.status_code == 200
    aid = created.json()["id"]
    assert created.json()["status"] == "pending"

    # Idempotent per pending subject: a second request returns the same id.
    again = client.post(f"/api/approvals?subject_type=live_activation&subject_id={sid}", auth=AUTH)
    assert again.json()["id"] == aid

    listing = client.get("/api/approvals", auth=AUTH).json()
    assert any(a["id"] == aid and a["status"] == "pending" for a in listing)

    # L39: the requester may NOT approve their own live_activation — four-eyes needs a
    # SECOND operator identity (with one shared credential, activation stays impossible).
    selfie = client.post(f"/api/approvals/{aid}/approve", auth=AUTH)
    assert selfie.status_code == 403
    assert "second" in selfie.json()["detail"].lower()

    # Simulate the second operator (the request came from someone else) → decidable.
    from src.db.base import session_scope
    from src.db.models import Approval

    with session_scope() as s:
        s.get(Approval, aid).requested_by = "op2"
    approved = client.post(f"/api/approvals/{aid}/approve", auth=AUTH)
    assert approved.status_code == 200 and approved.json()["status"] == "approved"
    # Re-deciding a non-pending approval is rejected.
    assert client.post(f"/api/approvals/{aid}/approve", auth=AUTH).status_code == 400


# --- /readyz (L38) -----------------------------------------------------------


@requires_db
@requires_redis
def test_readyz_reports_ready() -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert {c["name"] for c in body["components"]} == {"database", "redis"}


def test_readyz_503_when_redis_unreachable() -> None:
    iso = Settings(
        _env_file=None,
        app_env="paper",
        dashboard_auth_mode="basic",
        dashboard_username="admin",
        dashboard_password="secret",
        redis_url="redis://127.0.0.1:1/0",
    )
    r = TestClient(create_app(iso)).get("/readyz")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"


# --- live reset requires a second-operator approval (M28) ---------------------


@requires_db
def test_live_reset_is_approval_gated_archives_then_deletes(tmp_path) -> None:
    """M28: env=live reset is never a one-click delete. The endpoint only RAISES a pending
    live_reset approval; the requester cannot self-approve (four-eyes); a second operator's
    approval executes archive-then-delete (compliance record exported before removal)."""
    import json
    import uuid
    from pathlib import Path

    from src.db.base import session_scope
    from src.db.models import Approval, ApprovalStatus, PaperRun, PaperTradeRecord

    iso = Settings(
        _env_file=None,
        app_env="paper",
        dashboard_auth_mode="basic",
        dashboard_username="admin",
        dashboard_password="secret",
        backup_path=tmp_path / "backups",
    )
    c = TestClient(create_app(iso))

    sid = f"live:m28_{uuid.uuid4().hex[:8]}"
    with session_scope() as s:
        s.add(PaperRun(session_id=sid))
        s.add(
            PaperTradeRecord(
                session_id=sid, trade_id=f"t_{sid}", symbol="BTC/USDT:USDT",
                strategy="basis_reversion", side=1, pnl=1.0, pnl_r=0.1,
            )
        )
    try:
        # 1. The reset endpoint deletes NOTHING for live — it raises a pending approval.
        r = c.post("/api/live/reset?env=live&confirm=true", auth=AUTH, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/dashboard/approvals"
        with session_scope() as s:
            assert s.query(PaperRun).filter_by(session_id=sid).count() == 1  # still there
            approval = (
                s.query(Approval)
                .filter_by(subject_type="live_reset", status=ApprovalStatus.PENDING)
                .order_by(Approval.id.desc())
                .first()
            )
            assert approval is not None and approval.requested_by == "admin"
            aid = approval.id

        # 2. Self-approval of the destructive request is rejected (L39 four-eyes).
        selfie = c.post(f"/api/approvals/{aid}/approve", auth=AUTH)
        assert selfie.status_code == 403
        with session_scope() as s:
            assert s.query(PaperRun).filter_by(session_id=sid).count() == 1

        # 3. A second operator's approval executes: archive written, THEN rows deleted.
        with session_scope() as s:
            s.get(Approval, aid).requested_by = "other-operator"
        done = c.post(f"/api/approvals/{aid}/approve", auth=AUTH)
        assert done.status_code == 200
        body = done.json()
        assert body["status"] == "approved" and body.get("archive")
        archive = Path(body["archive"])
        assert archive.exists() and str(archive).startswith(str(tmp_path / "backups"))
        dumped = json.loads(archive.read_text())
        assert any(row["session_id"] == sid for row in dumped["tables"]["paper_runs"])
        assert any(row["session_id"] == sid for row in dumped["tables"]["paper_trades"])
        with session_scope() as s:
            assert s.query(PaperRun).filter_by(session_id=sid).count() == 0  # now deleted
    finally:
        with session_scope() as s:
            s.query(PaperTradeRecord).filter_by(session_id=sid).delete()
            s.query(PaperRun).filter_by(session_id=sid).delete()
            s.query(Approval).filter_by(subject_type="live_reset").delete()


@requires_db
def test_demo_reset_stays_one_click() -> None:
    """M28 keeps demo/testnet/paper resets one-click — only live is approval-gated."""
    import uuid

    from src.db.base import session_scope
    from src.db.models import PaperRun

    sid = f"demo:m28_oneclick_{uuid.uuid4().hex[:8]}"
    with session_scope() as s:
        s.add(PaperRun(session_id=sid))
    try:
        r = client.post("/api/live/reset?env=demo&confirm=true", auth=AUTH,
                        follow_redirects=False)
        assert r.status_code == 303
        with session_scope() as s:
            assert s.query(PaperRun).filter_by(session_id=sid).count() == 0  # deleted directly
    finally:
        with session_scope() as s:
            s.query(PaperRun).filter_by(session_id=sid).delete()


# --- atomic check+enqueue (L40) ------------------------------------------------


@requires_db
@requires_redis
def test_enqueue_exclusive_is_atomic_under_concurrent_start() -> None:
    """L40: the check-then-enqueue critical section is serialized by a Redis SETNX lock —
    a concurrent Start holding the lock gets 409 and enqueues NOTHING; after a successful
    enqueue the lock is released and the duplicate is caught by the active-job check."""
    import uuid

    from fastapi import HTTPException
    from src.api.app import _enqueue_exclusive
    from src.db.base import session_scope
    from src.db.models import Job
    from src.jobs import JobQueue

    q = JobQueue(_settings)
    job_type = f"m40_{uuid.uuid4().hex[:8]}"
    lock_key = f"qbot:enqueue-lock:{job_type}"
    try:
        # A concurrent Start holds the lock → this one loses with 409, enqueues nothing.
        assert q.redis.set(lock_key, "other-click", nx=True, ex=15)
        with pytest.raises(HTTPException) as exc:
            _enqueue_exclusive(q, job_type, {}, requested_by="admin", conflict_detail="dup")
        assert exc.value.status_code == 409
        with session_scope() as s:
            assert s.query(Job).filter_by(job_type=job_type).count() == 0

        # Lock free → enqueue succeeds and RELEASES the lock.
        q.redis.delete(lock_key)
        job_id = _enqueue_exclusive(
            q, job_type, {}, requested_by="admin", conflict_detail="dup"
        )
        assert job_id and q.redis.get(lock_key) is None

        # A later duplicate is still refused: the QUEUED job trips the active-job check.
        with pytest.raises(HTTPException) as exc2:
            _enqueue_exclusive(q, job_type, {}, requested_by="admin", conflict_detail="dup")
        assert exc2.value.status_code == 409
        with session_scope() as s:
            assert s.query(Job).filter_by(job_type=job_type).count() == 1  # no double-book
    finally:
        q.redis.delete(lock_key)
        with session_scope() as s:
            s.query(Job).filter_by(job_type=job_type).delete()


@requires_db
@requires_redis
def test_a_basket_demo_is_refused_while_a_live_session_owns_the_account() -> None:
    """Per-symbol and basket sessions are DIFFERENT job types, so the per-type exclusivity guard
    cannot see each other — but they share one exchange account, which holds one net position per
    symbol. With no live_symbols partition declared (the shipped default) both scope to the whole
    universe, so each would mirror a book they share: a basket rebalance resizes or flattens the
    per-symbol strategy's stop-managed position, and vice versa."""
    import uuid

    from src.config import get_settings
    from src.db.base import session_scope
    from src.db.models import Job, JobStatus
    from src.jobs import JobQueue
    from src.jobs.routing import processing_key, worker_key
    from src.strategies.config import load_strategies_config

    if load_strategies_config().reserved_symbols():
        pytest.skip("a live_symbols partition is declared — coexistence is the supported setup")

    rc = JobQueue(get_settings()).redis
    jid = "test-live-" + uuid.uuid4().hex[:8]
    wid = "test-worker-" + uuid.uuid4().hex[:8]
    with session_scope() as s:
        s.add(Job(job_id=jid, job_type="run_live_session", status=JobStatus.RUNNING,
                  input_params={}))
    rc.lpush(processing_key(wid), jid)
    rc.set(worker_key(wid), wid, ex=60)  # a genuinely-alive live session owns the account
    try:
        r = client.post(
            "/api/paper/run-basket-demo",
            params={"strategies": "funding_carry", "timeframe": ""},
            auth=AUTH, headers={"sec-fetch-site": "same-origin"},
        )
        assert r.status_code == 409
        assert "live_symbols" in r.json()["detail"]  # …and says how to make them coexist
        assert jid in r.json()["detail"], "…and names the session holding the account"
    finally:
        rc.delete(processing_key(wid))
        rc.delete(worker_key(wid))
        with session_scope() as s:
            obj = s.get(Job, jid)
            if obj is not None:
                s.delete(obj)


@requires_db
@requires_redis
def test_a_paper_mode_live_session_does_not_block_a_basket_demo() -> None:
    """A live session started in PAPER mode drives the offline SimulatedVenue: no exchange orders,
    no shared account, nothing to collide with. The dashboard offers exactly that override, so
    refusing the basket demo for it was a false block on a session that cannot conflict."""
    import uuid

    from src.config import get_settings
    from src.db.base import session_scope
    from src.db.models import Job, JobStatus
    from src.jobs import JobQueue
    from src.jobs.routing import processing_key, worker_key
    from src.strategies.config import load_strategies_config

    if load_strategies_config().reserved_symbols():
        pytest.skip("a live_symbols partition is declared — the coexistence check stands aside")

    rc = JobQueue(get_settings()).redis
    jid = "test-paper-live-" + uuid.uuid4().hex[:8]
    wid = "test-worker-" + uuid.uuid4().hex[:8]
    with session_scope() as s:
        s.add(Job(job_id=jid, job_type="run_live_session", status=JobStatus.RUNNING,
                  input_params={"mode": "paper"}))
    rc.lpush(processing_key(wid), jid)
    rc.set(worker_key(wid), wid, ex=60)
    try:
        r = client.post(
            "/api/paper/run-basket-demo",
            params={"strategies": "funding_carry", "timeframe": ""},
            auth=AUTH, headers={"sec-fetch-site": "same-origin"},
        )
        assert r.status_code != 409, r.text
    finally:
        rc.delete(processing_key(wid))
        rc.delete(worker_key(wid))
        with session_scope() as s:
            # the seeded live job, plus the basket job this test really enqueued
            s.query(Job).filter(Job.job_id == jid).delete()
            for j in s.query(Job).filter(
                Job.job_type == "run_basket_demo_session",
                Job.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            ).all():
                j.status = JobStatus.CANCELLED
                j.failure_reason = "cancelled: enqueued by a test, never an intended session"
