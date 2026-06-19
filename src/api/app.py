"""FastAPI application: health endpoints, dashboard, jobs, gates, stats, reports.

Phase 7 expands the Phase 1 skeleton into a full dashboard control centre
(AGENTS.md Section 25, Appendix B.8):
  - Aggregate + per-symbol statistics with time-period selectors (Section 25)
  - Background Gate Runner UI with live progress and remediation panels (B.9)
  - "Road to Live" view (Section 25)
  - Approvals + audit log endpoints (B.4)
  - Reports list (Section 34)

No heavy work runs inside a request handler (B.17). Dangerous actions
require explicit confirmation and are logged to audit_log.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import desc, select

from src.api.auth import require_dashboard_auth
from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import (
    Approval,
    ApprovalStatus,
    AuditLog,
    GateResult,
    GateStatus,
    Job,
    RemediationAction,
    RemediationStatus,
)
from src.monitoring import Alert, AlertSeverity, check_health, get_alert_sink
from src.observability import configure_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DASHBOARD_PAGES = [
    "Overview",
    "Data Coverage",
    "Universe",
    "Jobs",
    "Gates",
    "Remediation Actions",
    "Backtests",
    "Paper Trading",
    "Live Trading",
    "General Statistics",
    "Per-Symbol Statistics",
    "Strategy Analytics",
    "Regime Analytics",
    "Session Analytics",
    "Execution Quality",
    "Risk",
    "ML Shadow",
    "Online Learning",
    "RL",
    "Reports",
    "Approvals",
    "System Health",
    "Settings",
]

TIME_PERIODS = [
    "today",
    "yesterday",
    "last_7d",
    "last_30d",
    "current_month",
    "prev_month",
    "custom",
    "all",
]

_CSS = """
<style>
:root{
  --bg:#0a0d14;--surface:#121724;--surface-2:#171d2c;--elev:#1b2233;
  --border:#222b3d;--border-soft:#1a2233;--text:#e8ecf5;--text-dim:#aab3c5;
  --muted:#79849a;--accent:#6c8cff;--accent-2:#8a6cff;--accent-soft:rgba(108,140,255,.14);
  --green:#3ddc97;--green-bg:rgba(61,220,151,.13);--red:#ff6b6b;--red-bg:rgba(255,107,107,.13);
  --amber:#f5c451;--amber-bg:rgba(245,196,81,.13);--blue:#5cc3ff;--blue-bg:rgba(92,195,255,.13);
  --sbw:248px;--radius:13px;
  --ui:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  --mono:"SF Mono",SFMono-Regular,ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{scrollbar-color:#2a3447 transparent}
body{font-family:var(--ui);margin:0;background:var(--bg);color:var(--text);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
a{color:var(--accent);text-decoration:none}
a:hover{color:#a6bbff}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#2a3447;border-radius:8px;border:2px solid var(--bg)}
::-webkit-scrollbar-thumb:hover{background:#36425c}

/* ---- layout: fixed left sidebar + scrolling main ---- */
.app{display:flex;min-height:100vh}
.sidebar{position:fixed;top:0;left:0;width:var(--sbw);height:100vh;overflow-y:auto;
  background:linear-gradient(180deg,#10141f 0%,#0d111a 100%);border-right:1px solid var(--border);
  padding:0 12px 28px;z-index:20}
.brand{display:flex;align-items:center;gap:11px;padding:18px 10px 16px;margin-bottom:6px;
  position:sticky;top:0;background:linear-gradient(180deg,#10141f 80%,rgba(16,20,31,0));z-index:2}
.brand .mark{width:34px;height:34px;border-radius:9px;flex:0 0 auto;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  display:flex;align-items:center;justify-content:center;box-shadow:0 4px 14px rgba(108,140,255,.4)}
.brand .mark svg{width:19px;height:19px;color:#fff}
.brand .name{font-weight:700;font-size:15px;letter-spacing:.2px;color:#fff;line-height:1.1}
.brand .sub{font-size:10.5px;color:var(--muted);letter-spacing:.4px;text-transform:uppercase}
.navgroup{font-size:10.5px;font-weight:600;letter-spacing:.9px;text-transform:uppercase;
  color:var(--muted);padding:16px 11px 7px}
.navlink{display:flex;align-items:center;gap:10px;padding:8px 11px;border-radius:9px;
  color:var(--text-dim);font-size:13.5px;font-weight:500;margin:1px 0;position:relative;transition:.12s}
.navlink svg{width:17px;height:17px;flex:0 0 auto;opacity:.78}
.navlink:hover{background:var(--surface-2);color:var(--text)}
.navlink.active{background:var(--accent-soft);color:#fff}
.navlink.active svg{opacity:1;color:var(--accent)}
.navlink.active::before{content:"";position:absolute;left:-12px;top:7px;bottom:7px;width:3px;
  border-radius:0 3px 3px 0;background:linear-gradient(180deg,var(--accent),var(--accent-2))}

.main{flex:1;margin-left:var(--sbw);min-width:0;display:flex;flex-direction:column}
.topbar{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;
  gap:16px;padding:0 26px;height:62px;background:rgba(10,13,20,.82);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border)}
.topbar h1{font-size:18px;font-weight:650;margin:0;color:#fff;letter-spacing:-.2px}
.topbar .crumb{font-size:12px;color:var(--muted);margin-bottom:1px}
.envchip{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--text-dim);
  background:var(--surface);border:1px solid var(--border);padding:6px 12px;border-radius:999px}
.envchip .dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}
.container{padding:24px 26px 56px;max-width:1320px;width:100%}

h2{color:#fff;font-size:15px;font-weight:620;margin:0 0 14px;letter-spacing:-.1px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 2px rgba(0,0,0,.25)}
.card h2{border-bottom:1px solid var(--border-soft);padding-bottom:11px}

/* ---- badges / status ---- */
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;
  font-size:11.5px;font-weight:600;letter-spacing:.2px;line-height:1.4}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.pass{background:var(--green-bg);color:var(--green)}
.fail{background:var(--red-bg);color:var(--red)}
.blocked{background:var(--amber-bg);color:var(--amber)}
.not_run{background:rgba(121,132,154,.14);color:var(--muted)}
.running{background:var(--blue-bg);color:var(--blue)}

/* ---- tables ---- */
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:var(--muted);text-align:left;padding:9px 12px;font-weight:600;font-size:11px;
  text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border)}
td{padding:10px 12px;border-bottom:1px solid var(--border-soft);vertical-align:top}
tbody tr:last-child td,table tr:last-child td{border-bottom:none}
tr:hover td{background:var(--surface-2)}
code{font-family:var(--mono);font-size:12px;background:var(--surface-2);padding:1px 6px;border-radius:5px;color:#cdd6e6}

.meta{color:var(--muted);font-size:12.5px}
.score{font-size:30px;font-weight:750;color:var(--green);letter-spacing:-1px}
.score-low{color:var(--red)}
.score-mid{color:var(--amber)}

/* ---- custom controls (no native look) ---- */
select,input,textarea{appearance:none;-webkit-appearance:none;background:var(--surface-2);
  color:var(--text);border:1px solid var(--border);padding:8px 12px;border-radius:9px;
  font-family:var(--ui);font-size:13px;transition:.12s}
select{padding-right:34px;cursor:pointer;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2379849a' stroke-width='2.4' stroke-linecap='round'><path d='M6 9l6 6 6-6'/></svg>");
  background-repeat:no-repeat;background-position:right 12px center}
select:hover,input:hover{border-color:#33405a}
select:focus,input:focus,textarea:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-soft)}
label{font-size:12px;color:var(--muted);font-weight:500}

button,.btn{background:linear-gradient(180deg,#7491ff,#6c8cff);color:#fff;border:none;
  padding:8px 16px;border-radius:9px;cursor:pointer;font-family:var(--ui);font-weight:600;
  font-size:13px;text-decoration:none;display:inline-flex;align-items:center;gap:7px;
  box-shadow:0 1px 2px rgba(0,0,0,.3);transition:.13s}
button:hover,.btn:hover{filter:brightness(1.08);transform:translateY(-1px);color:#fff}
button:active,.btn:active{transform:translateY(0)}
button:disabled,.btn:disabled{opacity:.45;cursor:not-allowed;transform:none;filter:none}
.btn-danger{background:linear-gradient(180deg,#ff7a7a,#f4595a)}
.btn-neutral{background:var(--surface-2);color:var(--text-dim);border:1px solid var(--border);box-shadow:none}
.btn-neutral:hover{background:var(--elev);color:#fff;filter:none}

pre{background:#0b0e16;padding:13px;border-radius:10px;overflow-x:auto;font-size:12px;
  border:1px solid var(--border);font-family:var(--mono);color:#cdd6e6}
.remediation-step{padding:10px 13px;margin:6px 0;border-left:3px solid var(--accent);
  background:var(--surface-2);border-radius:0 8px 8px 0;font-size:13px}
.form-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}

/* ---- segmented pill control (period selector) ---- */
.segment{display:inline-flex;background:var(--surface-2);border:1px solid var(--border);
  border-radius:10px;padding:3px;gap:2px}
.segment a{padding:6px 13px;border-radius:7px;font-size:12.5px;font-weight:550;color:var(--text-dim);
  white-space:nowrap;transition:.12s}
.segment a:hover{color:#fff;background:var(--elev)}
.segment a.on{background:linear-gradient(180deg,#7491ff,#6c8cff);color:#fff;
  box-shadow:0 1px 4px rgba(108,140,255,.4)}

/* ---- KPI cards ---- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px;margin-bottom:18px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:16px 17px;position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,var(--accent),var(--accent-2));opacity:.85}
.kpi .v{font-size:23px;font-weight:700;color:#fff;letter-spacing:-.5px}
.kpi .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px;
  margin-top:5px;font-weight:600}
.pos{color:var(--green)}
.neg{color:var(--red)}
.chart{background:#0b0e16;border:1px solid var(--border);border-radius:11px;padding:10px}

@media(max-width:820px){
  .sidebar{position:static;width:100%;height:auto;border-right:none;border-bottom:1px solid var(--border)}
  .main{margin-left:0}
  .brand{position:static}
}
</style>
"""


def _icon(key: str) -> str:
    """A small inline (Feather-style) SVG glyph; inherits color via ``currentColor``."""
    p = {
        "overview": "<rect x='3' y='3' width='7' height='9' rx='1'/><rect x='14' y='3' width='7' height='5' rx='1'/><rect x='14' y='12' width='7' height='9' rx='1'/><rect x='3' y='16' width='7' height='5' rx='1'/>",
        "stats": "<path d='M3 3v18h18'/><rect x='7' y='11' width='3' height='6'/><rect x='12' y='7' width='3' height='10'/><rect x='17' y='13' width='3' height='4'/>",
        "strategy": "<circle cx='12' cy='12' r='8'/><circle cx='12' cy='12' r='3.2'/>",
        "regime": "<path d='M12 3l9 5-9 5-9-5 9-5z'/><path d='M3 13l9 5 9-5'/>",
        "session": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>",
        "execution": "<path d='M13 2L4 14h7l-1 8 9-12h-7l1-8z'/>",
        "risk": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z'/>",
        "backtest": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/><path d='M3 5l2.5 2.5'/>",
        "leaderboard": "<path d='M8 21h8'/><path d='M12 17v4'/><path d='M7 4h10v4a5 5 0 01-10 0V4z'/><path d='M17 5h3v2a3 3 0 01-3 3'/><path d='M7 5H4v2a3 3 0 003 3'/>",
        "paper": "<path d='M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z'/><path d='M14 3v6h6'/><path d='M8 13h8M8 17h6'/>",
        "reports": "<path d='M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z'/><path d='M14 3v6h6'/>",
        "live": "<path d='M3 12h4l2 7 4-14 2 7h6'/>",
        "shadow": "<path d='M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z'/><circle cx='12' cy='12' r='3'/>",
        "learning": "<rect x='5' y='8' width='14' height='11' rx='2'/><path d='M12 8V4'/><circle cx='12' cy='3' r='1.4'/><path d='M9 13h.01M15 13h.01'/>",
        "rl": "<circle cx='6' cy='6' r='2.4'/><circle cx='6' cy='18' r='2.4'/><circle cx='18' cy='12' r='2.4'/><path d='M8.2 7.4L16 11M8 16.5L16 13'/>",
        "control": "<path d='M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3'/><circle cx='4' cy='12' r='2'/><circle cx='12' cy='6' r='2'/><circle cx='20' cy='14' r='2'/>",
        "data": "<ellipse cx='12' cy='5' rx='8' ry='3'/><path d='M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5'/><path d='M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6'/>",
        "universe": "<circle cx='12' cy='12' r='9'/><path d='M3 12h18'/><path d='M12 3c2.5 2.5 2.5 15 0 18-2.5-3-2.5-15 0-18z'/>",
        "jobs": "<path d='M8 6h13M8 12h13M8 18h13'/><path d='M3 6h.01M3 12h.01M3 18h.01'/>",
        "gates": "<circle cx='12' cy='12' r='9'/><path d='M8.5 12.5l2.5 2.5 4.5-5'/>",
        "road": "<path d='M5 21V4a2 2 0 012-2h11l-2.5 4L18 8H7'/>",
        "remediation": "<path d='M14.7 6.3a4 4 0 00-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 005.4-5.4l-2.7 2.7-2.4-2.4 2.7-2.7z'/>",
        "approvals": "<path d='M9 12l2 2 4-4'/><path d='M12 3l7 3v6c0 4-3 7-7 8-4-1-7-4-7-8V6l7-3z'/>",
        "audit": "<path d='M4 4h11l5 5v11a1 1 0 01-1 1H5a1 1 0 01-1-1V4z'/><path d='M14 4v5h5M8 13h8M8 17h5'/>",
        "health": "<path d='M3 12h4l2-5 3 10 2.5-7 1.5 2H21'/>",
        "settings": "<circle cx='12' cy='12' r='3'/><path d='M19 12a7 7 0 00-.1-1.3l2-1.6-2-3.4-2.4 1a7 7 0 00-2.2-1.3L14 2h-4l-.3 2.1a7 7 0 00-2.2 1.3l-2.4-1-2 3.4 2 1.6A7 7 0 005 12c0 .4 0 .9.1 1.3l-2 1.6 2 3.4 2.4-1a7 7 0 002.2 1.3L10 22h4l.3-2.1a7 7 0 002.2-1.3l2.4 1 2-3.4-2-1.6c.1-.4.1-.9.1-1.3z'/>",
    }.get(key, "<circle cx='12' cy='12' r='8'/>")
    return (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.7' "
        f"stroke-linecap='round' stroke-linejoin='round'>{p}</svg>"
    )


# Sidebar information architecture: (group, [(label, href, icon, [title-prefixes that mark active])]).
# Covers all 23 pages required by AGENTS.md §25 / Appendix B.8, logically grouped.
_NAV_GROUPS: list[tuple[str, list[tuple[str, str, str, list[str]]]]] = [
    (
        "Performance",
        [
            ("Overview", "/", "overview", ["Performance Overview", "Overview"]),
            ("Statistics", "/dashboard/stats", "stats", ["General Statistics", "Per-Symbol"]),
            ("Strategy", "/dashboard/strategy", "strategy", ["Strategy Analytics"]),
            ("Regime", "/dashboard/regime", "regime", ["Regime Analytics"]),
            ("Session", "/dashboard/session-analytics", "session", ["Session Analytics"]),
            ("Execution", "/dashboard/execution", "execution", ["Execution Quality"]),
            ("Risk", "/dashboard/risk", "risk", ["Risk"]),
            ("Analytics", "/dashboard/analytics", "stats", ["Analytics"]),
        ],
    ),
    (
        "Research & Testing",
        [
            ("Strategies", "/dashboard/strategies", "strategy", ["Strategies"]),
            ("Backtests", "/dashboard/backtests", "backtest", ["Backtests"]),
            ("Leaderboard", "/dashboard/leaderboard", "leaderboard", ["Leaderboard"]),
            ("Paper Trading", "/dashboard/paper", "paper", ["Paper Trading"]),
            ("Reports", "/dashboard/reports", "reports", ["Reports"]),
        ],
    ),
    (
        "Live & Learning",
        [
            ("Live Trading", "/dashboard/live", "live", ["Live Trading"]),
            ("ML Shadow", "/dashboard/shadow", "shadow", ["ML Shadow"]),
            ("Online Learning", "/dashboard/learning", "learning", ["Online learning"]),
            ("RL", "/dashboard/rl", "rl", ["RL"]),
        ],
    ),
    (
        "Operations",
        [
            ("Control Center", "/dashboard/system", "control", ["Control Center"]),
            ("Data Coverage", "/dashboard/data-coverage", "data", ["Data Coverage"]),
            ("Universe", "/dashboard/universe", "universe", ["Universe"]),
            ("Jobs", "/dashboard/jobs", "jobs", ["Jobs", "Job"]),
            ("Gates", "/dashboard/gates", "gates", ["Gates", "Gate:"]),
            ("Road to Live", "/dashboard/road-to-live", "road", ["Road to Live"]),
            ("Remediation", "/dashboard/remediation", "remediation", ["Remediation"]),
            ("Approvals", "/dashboard/approvals", "approvals", ["Approvals"]),
            ("Audit Logs", "/dashboard/audit-logs", "audit", ["Audit Log"]),
            ("System Health", "/dashboard/health", "health", ["System Health"]),
            ("Settings", "/dashboard/settings", "settings", ["Settings"]),
        ],
    ),
]

_BRAND_MARK = _icon("live")  # the equity-curve glyph as the product mark


def _render_sidebar(title: str) -> str:
    out = [
        '<aside class="sidebar">',
        '<div class="brand">'
        f'<span class="mark">{_BRAND_MARK}</span>'
        '<span><span class="name">Quant Bot</span><br>'
        '<span class="sub">Control Center</span></span></div>',
    ]
    for group, items in _NAV_GROUPS:
        out.append(f'<div class="navgroup">{group}</div>')
        for label, href, icon, prefixes in items:
            active = any(title == p or title.startswith(p) for p in prefixes)
            cls = "navlink active" if active else "navlink"
            out.append(f'<a class="{cls}" href="{href}">{_icon(icon)}<span>{label}</span></a>')
    out.append("</aside>")
    return "".join(out)


def _env_chip() -> str:
    """Topbar environment chip: trading mode · exchange env, coloured by live-risk."""
    try:
        s = get_settings()
        live = s.live_trading_allowed
        dot = "var(--red)" if live else "var(--green)"
        return (
            f'<span class="dot" style="background:{dot};box-shadow:0 0 8px {dot}"></span>'
            f"{_esc(s.trading_mode.value)} · {_esc(s.exchange_id)}/{_esc(s.exchange_env)}"
        )
    except Exception:  # noqa: BLE001 - chrome must render even if settings are unavailable
        return '<span class="dot"></span>dashboard'


def _page(title: str, body: str, *, env_chip: str = "") -> str:
    chip = env_chip or _env_chip()
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)} — Quant Bot</title>"
        f"{_CSS}"
        "</head><body>"
        '<div class="app">'
        f"{_render_sidebar(title)}"
        '<div class="main">'
        '<header class="topbar">'
        f"<div><div class='crumb'>Quant Trading Bot</div><h1>{_esc(title)}</h1></div>"
        f'<div class="envchip">{chip}</div>'
        "</header>"
        f'<div class="container">{body}</div>'
        "</div></div></body></html>"
    )


def _esc(value: object) -> str:
    """HTML-escape any value before interpolating it into dashboard markup. Defensive
    against stored-XSS if a rendered field (job params, failure text, log message, audit
    actor/action) ever carries attacker-influenced content."""
    return html.escape("" if value is None else str(value))


def _status_badge(status: str) -> str:
    cls = {
        "passed": "pass",
        "failed": "fail",
        "blocked": "blocked",
        "not_run": "not_run",
        "running": "running",
    }.get(status.lower(), "not_run")
    return f'<span class="badge {cls}">{status.upper()}</span>'


_PERIODS = [
    ("all", "All time"),
    ("today", "Today"),
    ("yesterday", "Yesterday"),
    ("last_7d", "Last 7 days"),
    ("last_30d", "Last 30 days"),
    ("current_month", "This month"),
    ("prev_month", "Last month"),
]


def _period_selector(action: str, period: str) -> str:
    """A custom segmented pill control that re-renders the page per time period (Section 25).

    Rendered as styled links (a non-standard control, works without JS); the active period
    is highlighted. Pages using this carry only the ``period`` query param, so a plain link
    that sets ``?period=`` is sufficient."""
    pills = "".join(
        f'<a href="{action}?period={value}" class="{"on" if value == period else ""}">{label}</a>'
        for value, label in _PERIODS
    )
    return f'<div class="form-row"><label>Period</label><div class="segment">{pills}</div></div>'


def _scope_selector(action: str, period: str, strategy: str, session: str) -> str:
    """Period + entity-scope (strategy / paper-or-live session) selector (Section 25)."""
    from src.api.stats import get_trade_scopes

    try:
        scopes = get_trade_scopes()
    except Exception:  # noqa: BLE001 - selector must render even if the DB is unavailable
        scopes = {"strategies": [], "sessions": []}

    def _opts(values: list[str], selected: str, all_label: str) -> str:
        out = f'<option value=""{" selected" if not selected else ""}>{all_label}</option>'
        for v in values:
            out += f'<option value="{_esc(v)}"{" selected" if v == selected else ""}>{_esc(v)}</option>'
        return out

    pers = "".join(
        f'<option value="{value}"{" selected" if value == period else ""}>{label}</option>'
        for value, label in _PERIODS
    )
    return (
        f'<form method="get" action="{action}" class="form-row">'
        f'<label class="meta">Period</label><select name="period" onchange="this.form.submit()">{pers}</select>'
        f'<label class="meta">Strategy</label><select name="strategy" onchange="this.form.submit()">'
        f"{_opts(scopes['strategies'], strategy, 'All strategies')}</select>"
        f'<label class="meta">Session</label><select name="session" onchange="this.form.submit()">'
        f"{_opts(scopes['sessions'], session, 'All sessions')}</select>"
        '<noscript><button class="btn" type="submit">Apply</button></noscript></form>'
    )


def _money(value: float) -> str:
    cls = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f'<span class="{cls}">{value:+,.2f}</span>'


def _kpi(label: str, value_html: str) -> str:
    return f'<div class="kpi"><div class="v">{value_html}</div><div class="l">{_esc(label)}</div></div>'


def _kpi_row(t: Any) -> str:
    """KPI cards for a TradingStats-like object."""
    pf = "∞" if t.gross_loss == 0 and t.gross_win > 0 else f"{t.profit_factor:.2f}"
    cards = [
        _kpi("Net P&L", _money(t.realized_pnl)),
        _kpi("Win rate", f"{t.win_rate * 100:.1f}%"),
        _kpi(
            "Expectancy R",
            f'<span class="{"pos" if t.expectancy_r > 0 else "neg" if t.expectancy_r < 0 else ""}">{t.expectancy_r:+.3f}</span>',
        ),
        _kpi("Profit factor", pf),
        _kpi("Max drawdown", f'<span class="neg">{t.max_drawdown_pct * 100:.2f}%</span>'),
        _kpi("Trades", f"{t.total_trades}"),
        _kpi(
            "Avg win / loss",
            f'<span class="pos">{t.avg_win:+.1f}</span> / <span class="neg">{t.avg_loss:+.1f}</span>',
        ),
        _kpi("Fees", f"{t.total_fees_paid:,.2f}"),
    ]
    return f'<div class="kpis">{"".join(cards)}</div>'


def _equity_svg(curve: list[float], width: int = 1120, height: int = 180) -> str:
    """Inline SVG equity curve (no JS / external deps)."""
    if len(curve) < 2:
        return '<p class="meta">No trades in this period — run a paper session.</p>'
    lo, hi = min(curve), max(curve)
    span = (hi - lo) or 1.0
    n = len(curve)
    pts = " ".join(
        f"{i / (n - 1) * (width - 8) + 4:.1f},{height - 4 - (v - lo) / span * (height - 8):.1f}"
        for i, v in enumerate(curve)
    )
    base = curve[0]
    end = curve[-1]
    color = "#3fb950" if end >= base else "#f85149"
    return (
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/>'
        f"</svg></div>"
        f'<p class="meta">Equity {base:,.0f} → {end:,.0f} over {n - 1} trades '
        f"(base {base:,.0f}).</p>"
    )


def _breakdown_table(title: str, rows: list[dict], group_header: str) -> str:
    body = "".join(
        f"<tr><td>{_esc(r['group'])}</td><td>{r['trades']}</td>"
        f"<td>{_money(r['pnl'])}</td><td>{r['win_rate'] * 100:.1f}%</td>"
        f"<td>{r['expectancy_r']:+.3f}</td></tr>"
        for r in rows
    )
    return (
        f'<div class="card"><h2>{_esc(title)}</h2><table>'
        f"<tr><th>{_esc(group_header)}</th><th>Trades</th><th>P&L</th>"
        f"<th>Win rate</th><th>Expectancy R</th></tr>"
        f"{body or '<tr><td colspan=5 class=meta>No trades.</td></tr>'}</table></div>"
    )


def _gate_status_line(g: Any) -> str:
    """Compact persistent gate widget (Section 25 'Gate Status Widget')."""
    score = g.live_readiness_score
    cls = "score" if score >= 80 else ("score score-mid" if score >= 50 else "score score-low")
    nxt = (
        f" · next: {_esc(g.next_critical_action)}" if getattr(g, "next_critical_action", "") else ""
    )
    return (
        '<div class="card"><div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">'
        f'<span class="{cls}">{score:.0f}%</span>'
        f'<span class="meta">Live readiness ({g.critical_gates_passed}/{g.total_critical_gates} critical) · '
        f"{_status_badge('passed')} {g.passed} {_status_badge('failed')} {g.failed} "
        f"{_status_badge('blocked')} {g.blocked} {_status_badge('not_run')} {g.not_run}{nxt}</span>"
        '<a href="/dashboard/road-to-live" class="btn btn-neutral">Road to Live →</a>'
        "</div></div>"
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    app = FastAPI(title="Quant Trading Bot — Control Center", version="0.7.0")

    app.dependency_overrides[get_settings] = lambda: settings

    # ----- health (unauthenticated; for orchestration/monitoring) ---------- #
    @app.get("/health")
    def health() -> dict[str, Any]:
        return check_health(settings=settings).to_dict()

    @app.get("/health/{service}")
    def health_service(service: str) -> dict[str, Any]:
        return check_health(service=service, settings=settings).to_dict()

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "ok"}

    # ----- dashboard overview (authenticated) ------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    def dashboard(
        period: str = "all",
        strategy: str = "",
        session: str = "",
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        """Performance overview (TradeZella-style) over the chosen period + entity scope.

        Sourced from real ``paper_trades`` (shadow-only). The operational control
        center (gates, jobs, universe, kill switch) lives under System → Control Center.
        """
        from src.api.stats import get_aggregate_stats

        env_info = (
            f"Signed in as <b>{_esc(user)}</b> · env={settings.app_env.value} · "
            f"mode={settings.trading_mode.value} · live_allowed={settings.live_trading_allowed}"
        )
        try:
            agg = get_aggregate_stats(period, strategy=strategy or None, session_id=session or None)
            t = agg.trading
            body = (
                f"<p class='meta'>{env_info}</p>"
                + _gate_status_line(agg.gates)
                + _scope_selector("/", period, strategy, session)
                + _kpi_row(t)
                + f'<div class="card"><h2>Equity Curve</h2>{_equity_svg(t.equity_curve)}</div>'
                + _breakdown_table("By Strategy", t.by_strategy, "Strategy")
                + _breakdown_table("By Symbol", t.by_symbol, "Symbol")
                + '<p class="meta">Realized performance from <code>paper_trades</code> '
                "(shadow-only; live still gated). Run sessions via Paper or "
                "<code>qbot paper-lake</code>. "
                f"config_version={settings.config_version} · data_version={settings.data_version}</p>"
            )
        except Exception as exc:  # noqa: BLE001 - dashboard must render even if stats fail
            body = f"<p class='meta'>{env_info}</p><div class='card'><p class='meta'>Stats unavailable: {_esc(exc)}</p></div>"
        return _page("Performance Overview", body)

    @app.get("/dashboard/analytics", response_class=HTMLResponse)
    def dashboard_analytics(
        period: str = "all",
        strategy: str = "",
        session: str = "",
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        """Performance broken down by strategy / regime / session / symbol (Section 25)."""
        from src.api.stats import get_aggregate_stats

        agg = get_aggregate_stats(period, strategy=strategy or None, session_id=session or None)
        t = agg.trading
        body = (
            _scope_selector("/dashboard/analytics", period, strategy, session)
            + _kpi_row(t)
            + _breakdown_table("By Strategy", t.by_strategy, "Strategy")
            + _breakdown_table("By Regime", t.by_regime, "Regime")
            + _breakdown_table("By Session", t.by_session, "Session (UTC)")
            + _breakdown_table("By Symbol", t.by_symbol, "Symbol")
        )
        return _page("Analytics", body)

    @app.get("/dashboard/system", response_class=HTMLResponse)
    def dashboard_system(user: str = Depends(require_dashboard_auth)) -> str:
        """Operational control center: gates, jobs, universe, kill switch (Section 25)."""
        from src.api.stats import get_aggregate_stats
        from src.killswitch import KillSwitch

        env_info = (
            f"Signed in as <b>{_esc(user)}</b> · env={settings.app_env.value} · "
            f"mode={settings.trading_mode.value} · live_allowed={settings.live_trading_allowed}"
        )
        try:
            agg = get_aggregate_stats("all")
            g = agg.gates
            gate_widget = _gate_status_line(g) + (
                '<p><a href="/dashboard/gates" class="btn btn-neutral">All Gates →</a></p>'
            )
            jobs_widget = f"""
<div class="card"><h2>Jobs (all-time)</h2>
  <p>Total {agg.jobs.total} &nbsp;|&nbsp; ✓ {agg.jobs.succeeded} &nbsp;|&nbsp;
     ✗ {agg.jobs.failed} &nbsp;|&nbsp; ↻ {agg.jobs.running} &nbsp;|&nbsp; ⏳ {agg.jobs.queued}</p>
  <p><a href="/dashboard/jobs" class="btn btn-neutral">View Jobs →</a></p></div>"""
            universe_widget = f"""
<div class="card"><h2>Universe</h2>
  <p>{agg.universe.active_symbols} active / {agg.universe.total_symbols} total symbols
     {f"(v: {agg.universe.universe_version})" if agg.universe.universe_version else ""}</p>
  <p>{agg.open_remediation_items} open remediation item(s)
     {'<a href="/dashboard/remediation" class="btn btn-neutral">View →</a>' if agg.open_remediation_items else ""}</p></div>"""
        except Exception as exc:  # noqa: BLE001
            gate_widget = (
                f'<div class="card"><p class="meta">Stats unavailable: {_esc(exc)}</p></div>'
            )
            jobs_widget = universe_widget = ""

        ks_engaged = KillSwitch(settings).engaged()
        ks_control = (
            '<form method="post" action="/api/killswitch/disengage?confirm=true" '
            'style="display:inline"><button class="btn" type="submit">'
            "Disengage (manual reset)</button></form>"
            if ks_engaged
            else '<form method="post" action="/api/killswitch/engage?reason=dashboard+manual+kill" '
            'style="display:inline"><button class="btn btn-danger" type="submit">'
            "ENGAGE KILL SWITCH</button></form>"
        )
        ks_widget = (
            '<div class="card"><h2>Kill Switch</h2><p>Status: '
            + (
                '<span class="badge fail">ENGAGED</span>'
                if ks_engaged
                else '<span class="badge pass">CLEAR</span>'
            )
            + f"</p>{ks_control}</div>"
        )
        return _page(
            "Control Center",
            f"<p class='meta'>{env_info}</p>"
            + gate_widget
            + ks_widget
            + jobs_widget
            + universe_widget
            + f'<p class="meta">config_version={settings.config_version} · '
            f"data_version={settings.data_version}</p>",
        )

    # ----- System Health (#22) -------------------------------------------- #
    @app.get("/dashboard/health", response_class=HTMLResponse)
    def dashboard_health(user: str = Depends(require_dashboard_auth)) -> str:
        """Per-service / per-component health (Section 25 page #22), rendered from the same
        probes the JSON ``/health`` endpoint and the Monitoring gate use."""
        report = check_health(settings=settings)
        overall = _status_badge("passed" if report.healthy else "failed")
        comp_rows = [
            [
                _esc(c.name),
                _status_badge("passed" if c.healthy else "failed"),
                _esc(c.detail or ""),
            ]
            for c in report.components
        ]
        body = (
            '<div class="card"><h2>Overall status</h2>'
            f"<p>Service <code>{_esc(report.service)}</code> &nbsp; {overall}</p>"
            '<p class="meta">Each dependency is probed independently; a failed probe is reported '
            "as an unhealthy component (it never crashes the dashboard). The JSON form is at "
            "<code>/health</code> and per-service at <code>/health/{service}</code>.</p></div>"
            '<div class="card"><h2>Components</h2>'
            + _rows_table(["Component", "Status", "Detail"], comp_rows, "No components probed.")
            + "</div>"
        )
        return _page("System Health", body)

    def _rows_table(headers: list[str], rows: list[list], empty: str = "No data.") -> str:
        head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        n = len(headers)
        return (
            f"<table><tr>{head}</tr>"
            f"{body or f'<tr><td colspan={n} class=meta>{_esc(empty)}</td></tr>'}</table>"
        )

    def _kv_card(title: str, pairs: list[tuple[str, object]]) -> str:
        rows = "".join(
            f'<tr><td class="meta">{_esc(k)}</td><td>{_esc(v)}</td></tr>' for k, v in pairs
        )
        return f'<div class="card"><h2>{_esc(title)}</h2><table>{rows}</table></div>'

    # ----- Data Coverage (#2) --------------------------------------------- #
    @app.get("/dashboard/data-coverage", response_class=HTMLResponse)
    def dashboard_data_coverage(user: str = Depends(require_dashboard_auth)) -> str:
        from src.db.models import DatasetVersion

        with session_scope() as s:
            rows = list(
                s.execute(select(DatasetVersion).order_by(desc(DatasetVersion.created_at)))
                .scalars()
                .all()
            )[:50]
            data = [
                [
                    f"<code>{_esc(r.version)}</code>",
                    _esc(r.data_version),
                    _esc(r.exchange_id),
                    _status_badge("passed" if r.validation_status == "valid" else "failed"),
                    _esc(", ".join(r.symbols or [])),
                    sum((r.row_counts or {}).values()),
                ]
                for r in rows
            ]
        body = (
            f'<div class="card"><h2>Data Coverage — DATA_VERSION snapshots ({len(data)})</h2>'
            + _rows_table(
                ["Snapshot", "Data Version", "Exchange", "Valid", "Symbols", "Rows"],
                data,
                "No snapshots — run `qbot download`.",
            )
            + "</div>"
        )
        return _page("Data Coverage", body)

    # ----- Universe (#3) -------------------------------------------------- #
    @app.get("/dashboard/universe", response_class=HTMLResponse)
    def dashboard_universe(user: str = Depends(require_dashboard_auth)) -> str:
        from src.db.models import UniverseMember, UniverseVersion

        with session_scope() as s:
            latest = s.execute(
                select(UniverseVersion).order_by(desc(UniverseVersion.created_at)).limit(1)
            ).scalar_one_or_none()
            members = (
                list(
                    s.execute(
                        select(UniverseMember).where(
                            UniverseMember.universe_version == latest.version
                        )
                    )
                    .scalars()
                    .all()
                )
                if latest
                else []
            )
            rows = [
                [
                    _esc(m.symbol),
                    _status_badge("passed" if m.status.value == "active" else "not_run"),
                    _esc(m.reason or ""),
                ]
                for m in members
            ]
        ver = latest.version if latest else "—"
        body = (
            f'<div class="card"><h2>Universe ({_esc(ver)}) — {len(rows)} symbols</h2>'
            + _rows_table(["Symbol", "Status", "Reason"], rows, "No universe built yet.")
            + "</div>"
        )
        return _page("Universe", body)

    # ----- Live Trading (#9) ---------------------------------------------- #
    @app.get("/dashboard/live", response_class=HTMLResponse)
    def dashboard_live(user: str = Depends(require_dashboard_auth)) -> str:
        from src.db.models import JobStatus, PaperRun
        from src.live.admin import summarize_env_stats

        env = settings.exchange_env
        is_demo = env == "demo"
        with session_scope() as s:
            runs = list(
                s.execute(select(PaperRun).order_by(desc(PaperRun.created_at))).scalars().all()
            )
            live_runs = [
                r
                for r in runs
                if str(r.session_id).startswith(("live:", "testnet:", "demo:"))
            ][:50]
            rows = [
                [
                    f"<code>{_esc(r.session_id)}</code>",
                    r.created_at.strftime("%Y-%m-%d %H:%M"),
                    r.executed_count,
                    f"{r.net_pnl:+.2f}",
                    f"{r.win_rate * 100:.1f}%",
                ]
                for r in live_runs
            ]
            # Live-session jobs (running first), so the operator sees progress + can Stop.
            jobs = list(
                s.execute(
                    select(Job)
                    .where(Job.job_type == "run_live_session")
                    .order_by(desc(Job.created_at))
                    .limit(20)
                )
                .scalars()
                .all()
            )
            job_rows = []
            active_ids = []
            for j in jobs:
                active = j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
                if active:
                    active_ids.append(j.job_id)
                prog = f"{j.progress_current}/{j.progress_total}" if j.progress_total else "-"
                stop = (
                    f"<form method='post' action='/api/jobs/{j.job_id}/cancel' "
                    f"style='display:inline'><button class='btn btn-danger' type='submit'>"
                    f"&#9632; Stop</button></form>"
                    if active
                    else "-"
                )
                job_rows.append(
                    [
                        f"<a href='/dashboard/jobs/{j.job_id}'>{j.job_id[:12]}…</a>",
                        _status_badge(j.status.value),
                        prog,
                        _esc(j.progress_message or ""),
                        stop,
                    ]
                )

        demo_stats = summarize_env_stats("demo")
        status = _kv_card(
            "Live status",
            [
                ("trading_mode", settings.trading_mode.value),
                ("app_env", settings.app_env.value),
                ("live_trading_allowed", settings.live_trading_allowed),
                ("exchange / env", f"{settings.exchange_id} / {env}"),
            ],
        )

        # --- Demo / testnet control panel (dashboard-only operation) ------------ #
        env_note = {
            "demo": "Bybit <b>demo</b> trading (mainnet market data + virtual funds, "
            "api-demo.bybit.com). No real money; safe for testing.",
            "testnet": "Bybit <b>testnet</b> (separate test network, virtual funds).",
            "live": "Bybit <b>live</b> (REAL MONEY) — every order is gated by the activation "
            "guard (gates green + sign-off + caps).",
        }.get(env, f"environment {_esc(env)}")
        running = bool(active_ids)
        start_disabled = " disabled" if running else ""
        controls = (
            '<div class="card"><h2>Demo / live control</h2>'
            f'<p class="meta">Current environment: <b>{_esc(env)}</b> — {env_note}</p>'
            '<form method="post" action="/api/live/start" style="display:inline;margin-right:8px">'
            f'<button class="btn" type="submit"{start_disabled}>&#9654; Start '
            f"{_esc(env)} session</button></form>"
            + (
                '<form method="post" action="/api/live/reset" style="display:inline" '
                "onsubmit=\"return confirm('Zero ALL demo statistics? This deletes every "
                "demo: run, trade, decision log and explainability row. Paper/testnet/live "
                "data is untouched.');\">"
                '<input type="hidden" name="confirm" value="true">'
                '<button class="btn btn-danger" type="submit">&#10227; Reset demo statistics</button>'
                "</form>"
                if is_demo
                else ""
            )
            + (
                '<p class="meta" style="margin-top:8px">A session runs in the background on the '
                "<code>live</code> worker; watch its progress below and press Stop to halt it "
                "cleanly (whatever executed is still saved). Restart any time.</p>"
            )
            + (
                f'<p class="meta">Demo statistics currently stored: {demo_stats.runs} runs, '
                f"{demo_stats.trades} trades, {demo_stats.decision_logs} decision logs, "
                f"{demo_stats.explainability} explainability rows. Reset to zero before a fresh "
                "demo-testing run so its statistics start clean and separated.</p>"
                if is_demo
                else ""
            )
            + (
                '<p class="meta" style="color:#b45309">A session is already running — Stop it '
                "before starting another.</p>"
                if running
                else ""
            )
            + "</div>"
        )

        jobs_card = (
            f'<div class="card"><h2>Live sessions — jobs ({len(job_rows)})</h2>'
            + _rows_table(
                ["Job", "Status", "Progress", "Message", ""],
                job_rows,
                "No live sessions started yet — click Start above.",
            )
            + "</div>"
        )

        body = (
            status
            + controls
            + jobs_card
            + '<p class="meta">Live (real-money) trading is hard-gated: TRADING_MODE=LIVE + '
            "APP_ENV=production + ENABLE_LIVE_TRADING=true, all blocks_live gates PASS, an "
            "approved live_activation sign-off, and bounded-live caps (configs/live.yaml). "
            "Demo and testnet use virtual funds and need no activation guard.</p>"
            + f'<div class="card"><h2>Demo / testnet / live sessions ({len(rows)})</h2>'
            + _rows_table(
                ["Session", "Created", "Executed", "Net P&L", "Win rate"],
                rows,
                "No demo/testnet/live runs yet.",
            )
            + "</div>"
        )
        return _page("Live Trading", body)

    # ----- live/demo session controls (dashboard-only operation) ---------- #
    @app.post("/api/live/start")
    def live_start(user: str = Depends(require_dashboard_auth)) -> RedirectResponse:
        """Start a dashboard-driven live/demo/testnet session on the dedicated live worker."""
        from src.jobs import JobQueue

        JobQueue(settings).enqueue("run_live_session", {"requested_by": user}, requested_by=user)
        _audit("run_live_session", target=settings.exchange_env, actor=user, detail={})
        return RedirectResponse(url="/dashboard/live", status_code=303)

    @app.post("/api/live/reset")
    def live_reset(
        confirm: bool = False, user: str = Depends(require_dashboard_auth)
    ) -> RedirectResponse:
        """Zero the demo environment's statistics (runs/trades/logs/explainability)."""
        if not confirm:
            raise HTTPException(status_code=400, detail="reset requires confirm=true")
        from src.live.admin import reset_env_stats

        removed = reset_env_stats("demo")
        _audit(
            "reset_env_stats", target="demo", actor=user, detail={"removed": removed.to_dict()}
        )
        return RedirectResponse(url="/dashboard/live", status_code=303)

    # ----- Execution Quality (#15) ---------------------------------------- #
    @app.get("/dashboard/execution", response_class=HTMLResponse)
    def dashboard_execution(
        period: str = "all", user: str = Depends(require_dashboard_auth)
    ) -> str:
        from src.api.stats import compute_trading_stats, resolve_window
        from src.db.models import PaperTradeRecord

        window = resolve_window(period, None, None)
        with session_scope() as s:
            q = select(PaperTradeRecord)
            if window.start:
                q = q.where(PaperTradeRecord.created_at >= window.start)
            if window.end:
                q = q.where(PaperTradeRecord.created_at <= window.end)
            trades = list(s.execute(q).scalars().all())
        n = len(trades)
        maker = sum(1 for t in trades if t.execution_route == "maker")
        with_stop = sum(1 for t in trades if t.has_exchange_side_stop)
        avg_slip = (sum(t.slippage_cost for t in trades) / n) if n else 0.0
        body = (
            _period_selector("/dashboard/execution", period)
            + _kpi_row(compute_trading_stats(window))
            + _kv_card(
                "Execution quality",
                [
                    ("trades", n),
                    ("maker fill %", f"{(maker / n * 100) if n else 0:.1f}%"),
                    ("exchange-side stop %", f"{(with_stop / n * 100) if n else 0:.1f}%"),
                    ("avg slippage cost", f"{avg_slip:.4f}"),
                    ("total fees", f"{sum(t.fee for t in trades):.2f}"),
                ],
            )
        )
        return _page("Execution Quality", body)

    # ----- Risk (#16) ----------------------------------------------------- #
    @app.get("/dashboard/risk", response_class=HTMLResponse)
    def dashboard_risk(user: str = Depends(require_dashboard_auth)) -> str:
        from src.risk import load_risk_config

        rc = load_risk_config()
        env = rc.envelope
        body = _kv_card(
            "Immutable risk envelope (Section 17 — hard ceilings)",
            [
                ("max risk % / trade", env.max_risk_pct_per_trade),
                ("max leverage", env.max_leverage),
                ("portfolio heat cap", env.portfolio_heat_cap),
                ("net beta (BTC) cap", env.net_beta_btc_cap),
                ("daily loss limit", env.daily_loss_limit),
                ("max drawdown limit", env.max_drawdown_limit),
            ],
        ) + (
            '<p class="meta">The envelope is immutable: values may only tighten via a new '
            "config version + approval; the live activation guard re-checks all gates + caps.</p>"
        )
        return _page("Risk", body)

    # ----- Online Learning (#18) + RL (#19) ------------------------------- #
    def _learner_page(title: str, modes: tuple[str, ...]) -> str:
        from src.db.models import LearnerLog

        with session_scope() as s:
            rows = list(
                s.execute(
                    select(LearnerLog)
                    .where(LearnerLog.mode.in_(modes))
                    .order_by(desc(LearnerLog.ts))
                )
                .scalars()
                .all()
            )[:50]
            data = [
                [
                    _esc(r.learner_id),
                    _esc(r.mode),
                    _status_badge("passed" if not r.applied else "failed"),
                    _esc(r.rollback_event or ""),
                ]
                for r in rows
            ]
        return (
            '<p class="meta">Shadow-only / gated: learner actions are never applied to live '
            "trading (applied=False) until promoted through the gates + sign-off (Section 21).</p>"
            + f'<div class="card"><h2>{_esc(title)} log ({len(data)})</h2>'
            + _rows_table(
                ["Learner", "Mode", "Applied=False", "Rollback"], data, "No learner activity yet."
            )
            + "</div>"
        )

    @app.get("/dashboard/learning", response_class=HTMLResponse)
    def dashboard_learning(user: str = Depends(require_dashboard_auth)) -> str:
        return _page(
            "Online Learning",
            _learner_page("Online learning", ("SHADOW", "RECOMMEND", "LIVE_BOUNDED", "FROZEN")),
        )

    @app.get("/dashboard/rl", response_class=HTMLResponse)
    def dashboard_rl(user: str = Depends(require_dashboard_auth)) -> str:
        return _page("RL", _learner_page("RL", ("SHADOW", "RECOMMEND", "LIVE_BOUNDED", "FROZEN")))

    # ----- Settings (#23) ------------------------------------------------- #
    @app.get("/dashboard/settings", response_class=HTMLResponse)
    def dashboard_settings(user: str = Depends(require_dashboard_auth)) -> str:
        versions = settings.versions()
        body = (
            _kv_card(
                "Environment & mode",
                [
                    ("app_env", settings.app_env.value),
                    ("trading_mode", settings.trading_mode.value),
                    ("live_trading_allowed", settings.live_trading_allowed),
                    ("enable_live_trading", settings.enable_live_trading),
                    ("exchange / env", f"{settings.exchange_id} / {settings.exchange_env}"),
                    ("order_client_id_prefix", settings.order_client_id_prefix),
                ],
            )
            + _kv_card("Versions", list(versions.items()))
            + '<p class="meta">Read-only. Settings are env-validated (src/config/settings.py); '
            "changing them is a new config version + freeze + approval (CONFIG-FREEZE / LIVE).</p>"
        )
        return _page("Settings", body)

    # ----- dedicated Strategy / Regime / Session analytics (#12–14) ------- #
    def _one_breakdown(title: str, attr: str, group_header: str, period: str) -> str:
        from src.api.stats import get_aggregate_stats

        t = get_aggregate_stats(period).trading
        return (
            _period_selector(f"/dashboard/{attr}", period)
            + _kpi_row(t)
            + _breakdown_table(title, getattr(t, f"by_{attr}"), group_header)
        )

    @app.get("/dashboard/strategy", response_class=HTMLResponse)
    def dashboard_strategy(period: str = "all", user: str = Depends(require_dashboard_auth)) -> str:
        return _page(
            "Strategy Analytics", _one_breakdown("By Strategy", "strategy", "Strategy", period)
        )

    @app.get("/dashboard/regime", response_class=HTMLResponse)
    def dashboard_regime(period: str = "all", user: str = Depends(require_dashboard_auth)) -> str:
        return _page("Regime Analytics", _one_breakdown("By Regime", "regime", "Regime", period))

    @app.get("/dashboard/session-analytics", response_class=HTMLResponse)
    def dashboard_session_analytics(
        period: str = "all", user: str = Depends(require_dashboard_auth)
    ) -> str:
        return _page(
            "Session Analytics", _one_breakdown("By Session", "session", "Session", period)
        )

    @app.get("/api/me")
    def me(user: str = Depends(require_dashboard_auth)) -> dict[str, str]:
        return {"user": user, "env": settings.app_env.value}

    # ----- stats ----------------------------------------------------------- #
    @app.get("/api/stats")
    def aggregate_stats(
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        strategy: str | None = None,
        session: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> dict[str, Any]:
        from src.api.stats import get_aggregate_stats

        return get_aggregate_stats(
            period, from_ts, to_ts, strategy=strategy, session_id=session
        ).to_dict()

    @app.get("/api/stats/scopes")
    def stats_scopes(user: str = Depends(require_dashboard_auth)) -> dict[str, list[str]]:
        from src.api.stats import get_trade_scopes

        return get_trade_scopes()

    @app.get("/api/stats/symbols")
    def stats_symbols(user: str = Depends(require_dashboard_auth)) -> list[str]:
        from src.api.stats import get_symbols_list

        return get_symbols_list()

    @app.get("/api/stats/{symbol}")
    def per_symbol_stats(
        symbol: str,
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> dict[str, Any]:
        from src.api.stats import get_per_symbol_stats

        return get_per_symbol_stats(symbol, period, from_ts, to_ts)

    # ----- stats dashboard pages ------------------------------------------- #
    @app.get("/dashboard/stats", response_class=HTMLResponse)
    def dashboard_stats(
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        from src.api.stats import get_aggregate_stats, get_symbols_list

        try:
            agg = get_aggregate_stats(period, from_ts, to_ts)
            symbols = get_symbols_list()
            symbol_links = (
                " · ".join(f'<a href="/dashboard/stats/{_esc(s)}">{_esc(s)}</a>' for s in symbols)
                if symbols
                else '<span class="meta">no symbols yet</span>'
            )
            g = agg.gates
            j = agg.jobs
            score_cls = (
                "score"
                if g.live_readiness_score >= 80
                else ("score score-mid" if g.live_readiness_score >= 50 else "score score-low")
            )
            body = f"""
<div class="form-row">
  <label>Period:</label>
  <form method="get">
    <select name="period" onchange="this.form.submit()">
      {"".join(f'<option value="{p}"{" selected" if p == period else ""}>{p}</option>' for p in TIME_PERIODS)}
    </select>
    {f'<input type="text" name="from_ts" value="{from_ts or ""}" placeholder="from (ISO)" style="width:200px">' if period == "custom" else ""}
    {f'<input type="text" name="to_ts" value="{to_ts or ""}" placeholder="to (ISO)" style="width:200px">' if period == "custom" else ""}
    <button type="submit">Apply</button>
  </form>
</div>
<div class="card">
  <h2>Aggregate Statistics — {period}</h2>
  {f'<p class="meta">Window: {agg.window_start} → {agg.window_end}</p>' if agg.window_start else ""}
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Live Readiness Score</td><td><span class="{score_cls}">{g.live_readiness_score:.1f}%</span></td></tr>
    <tr><td>Gates Passed</td><td>{g.passed}</td></tr>
    <tr><td>Gates Failed</td><td>{g.failed}</td></tr>
    <tr><td>Gates Blocked</td><td>{g.blocked}</td></tr>
    <tr><td>Gates Not Run</td><td>{g.not_run}</td></tr>
    <tr><td>Jobs (total)</td><td>{j.total}</td></tr>
    <tr><td>Jobs Succeeded</td><td>{j.succeeded}</td></tr>
    <tr><td>Jobs Failed</td><td>{j.failed}</td></tr>
    <tr><td>Active Symbols</td><td>{agg.universe.active_symbols}</td></tr>
    <tr><td>Open Remediation Items</td><td>{agg.open_remediation_items}</td></tr>
    <tr><td>Total Trades (Phase 8)</td><td>{agg.trading.total_trades}</td></tr>
    <tr><td>Win Rate (Phase 8)</td><td>{agg.trading.win_rate:.1%}</td></tr>
    <tr><td>Expectancy R (Phase 8)</td><td>{agg.trading.expectancy_r:.4f}</td></tr>
    <tr><td>Max Drawdown % (Phase 8)</td><td>{agg.trading.max_drawdown_pct:.1%}</td></tr>
  </table>
</div>
<p class="meta">Trading metrics (PnL, win-rate, drawdown, fees, slippage, funding) populate in Phase 8.</p>
<p><b>Per-Symbol Stats →</b> {symbol_links}</p>"""
        except Exception as exc:
            body = f'<div class="card"><p class="meta">Error: {exc}</p></div>'
        return _page("General Statistics", body)

    @app.get("/dashboard/stats/{symbol}", response_class=HTMLResponse)
    def dashboard_per_symbol_stats(
        symbol: str,
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        from src.api.stats import get_per_symbol_stats, get_symbols_list

        symbols = get_symbols_list()
        try:
            stats = get_per_symbol_stats(symbol, period, from_ts, to_ts)
            t = stats["trading"]
            body = f"""
<div class="form-row">
  <label>Symbol:</label>
  <form method="get">
    <select name="..." onchange="window.location='/dashboard/stats/'+this.value">
      {"".join(f"<option{'  selected' if s == symbol else ''}>{s}</option>" for s in (symbols or [symbol]))}
    </select>
    <label style="margin-left:8px">Period:</label>
    <select name="period" onchange="this.form.submit()">
      {"".join(f'<option value="{p}"{" selected" if p == period else ""}>{p}</option>' for p in TIME_PERIODS)}
    </select>
    <button type="submit">Apply</button>
  </form>
</div>
<div class="card">
  <h2>Per-Symbol Statistics — {symbol} — {period}</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Trades</td><td>{t["total_trades"]}</td></tr>
    <tr><td>Win Rate</td><td>{t["win_rate"]:.1%}</td></tr>
    <tr><td>Expectancy R</td><td>{t["expectancy_r"]:.4f}</td></tr>
    <tr><td>Realized PnL</td><td>{t["realized_pnl"]:.4f}</td></tr>
    <tr><td>Total Fees</td><td>{t["total_fees_paid"]:.4f}</td></tr>
    <tr><td>Total Slippage</td><td>{t["total_slippage"]:.4f}</td></tr>
    <tr><td>Funding Paid</td><td>{t["total_funding_paid"]:.4f}</td></tr>
    <tr><td>Max Drawdown %</td><td>{t["max_drawdown_pct"]:.1%}</td></tr>
  </table>
  <p class="meta">{stats.get("note", "")}</p>
</div>"""
        except Exception as exc:
            body = f'<div class="card"><p class="meta">Error: {exc}</p></div>'
        return _page(f"Per-Symbol Statistics: {symbol}", body)

    # ----- jobs ------------------------------------------------------------ #
    @app.get("/api/jobs")
    def list_jobs(
        limit: int = 50,
        gate_id: str | None = None,
        status: str | None = None,
        job_type: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(Job).order_by(desc(Job.created_at)).limit(limit)
            if gate_id:
                q = q.where(Job.related_gate_id == gate_id)
            if status:
                q = q.where(Job.status == status)
            if job_type:
                q = q.where(Job.job_type == job_type)
            rows = session.execute(q).scalars().all()
            return [
                {
                    "job_id": j.job_id,
                    "job_type": j.job_type,
                    "status": j.status.value,
                    "created_at": j.created_at.isoformat(),
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "finished_at": j.finished_at.isoformat() if j.finished_at else None,
                    "progress": f"{j.progress_current}/{j.progress_total}",
                    "progress_message": j.progress_message,
                    "failure_reason": j.failure_reason,
                    "next_action_hint": j.next_action_hint,
                    "related_gate_id": j.related_gate_id,
                }
                for j in rows
            ]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="job not found")
            logs = [
                {"ts": lg.ts.isoformat(), "level": lg.level, "message": lg.message}
                for lg in job.logs
            ]
            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "input_params": job.input_params,
                "progress_current": job.progress_current,
                "progress_total": job.progress_total,
                "progress_message": job.progress_message,
                "failure_reason": job.failure_reason,
                "next_action_hint": job.next_action_hint,
                "related_gate_id": job.related_gate_id,
                "logs": logs,
            }

    @app.get("/api/jobs/{job_id}/logs")
    def get_job_logs(job_id: str, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="job not found")
            return [
                {"ts": lg.ts.isoformat(), "level": lg.level, "message": lg.message}
                for lg in job.logs
            ]

    @app.post("/api/jobs/{job_type}")
    def enqueue_job(
        job_type: str,
        params: dict | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> dict[str, str]:
        from src.jobs import JobQueue
        from src.jobs.handlers import ensure_handlers_registered
        from src.jobs.registry import registry

        ensure_handlers_registered()
        if not registry.has(job_type):
            raise HTTPException(status_code=400, detail=f"unknown job_type {job_type}")
        job_id = JobQueue(settings).enqueue(job_type, params or {}, requested_by=user)
        _audit("enqueue_job", target=job_type, actor=user, detail={"job_type": job_type})
        return {"job_id": job_id}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict[str, bool]:
        from src.jobs import JobQueue

        _audit("cancel_job", target=job_id, actor=user, detail={})
        return {"cancelled": JobQueue(settings).cancel(job_id)}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict[str, bool]:
        from src.jobs import JobQueue

        _audit("retry_job", target=job_id, actor=user, detail={})
        return {"requeued": JobQueue(settings).retry(job_id)}

    # ----- jobs dashboard page ---------------------------------------------- #
    @app.get("/dashboard/jobs", response_class=HTMLResponse)
    def dashboard_jobs(
        limit: int = 50,
        status: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        with session_scope() as session:
            q = select(Job).order_by(desc(Job.created_at)).limit(limit)
            if status:
                q = q.where(Job.status == status)
            jobs = session.execute(q).scalars().all()

        rows = ""
        for j in jobs:
            prog = f"{j.progress_current}/{j.progress_total}" if j.progress_total else "-"
            rows += (
                f"<tr>"
                f"<td><a href='/dashboard/jobs/{j.job_id}'>{j.job_id[:12]}…</a></td>"
                f"<td>{j.job_type}</td>"
                f"<td>{_status_badge(j.status.value)}</td>"
                f"<td>{j.created_at.strftime('%Y-%m-%d %H:%M')}</td>"
                f"<td>{prog}</td>"
                f"<td>{j.related_gate_id or '-'}</td>"
                f"<td>{_esc((j.failure_reason or '')[:60])}</td>"
                f"</tr>"
            )
        filter_form = f"""
<form method="get" class="form-row">
  <label>Status:</label>
  <select name="status" onchange="this.form.submit()">
    <option value="">all</option>
    {"".join(f'<option value="{s}"{" selected" if s == status else ""}>{s}</option>' for s in ["queued", "running", "succeeded", "failed", "cancelled"])}
  </select>
  <button type="submit">Filter</button>
</form>"""
        body = (
            filter_form
            + f"""
<div class="card">
  <h2>Background Jobs ({len(jobs)} shown)</h2>
  <table>
    <tr><th>ID</th><th>Type</th><th>Status</th><th>Created</th><th>Progress</th><th>Gate</th><th>Failure</th></tr>
    {rows or '<tr><td colspan="7" class="meta">No jobs found.</td></tr>'}
  </table>
</div>"""
        )
        return _page("Jobs", body)

    @app.get("/dashboard/jobs/{job_id}", response_class=HTMLResponse)
    def dashboard_job_detail(job_id: str, user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                return _page("Job Not Found", '<p class="meta">Job not found.</p>')
            logs_html = (
                "".join(
                    f"<tr><td>{lg.ts.strftime('%H:%M:%S')}</td><td>{_esc(lg.level)}</td><td>{_esc(lg.message)}</td></tr>"
                    for lg in job.logs
                )
                or '<tr><td colspan="3" class="meta">No logs.</td></tr>'
            )
            jid = job.job_id
            st = job.status.value
            actions = ""
            if st in ("queued", "running"):
                actions += (
                    f'<form method="post" action="/api/jobs/{jid}/cancel" style="display:inline">'
                    f'<button class="btn btn-danger" type="submit">Cancel</button></form> '
                )
            if st in ("failed", "cancelled", "expired"):
                actions += (
                    f'<form method="post" action="/api/jobs/{jid}/retry" style="display:inline">'
                    f'<button class="btn" type="submit">Retry</button></form> '
                )
            actions_card = f'<div class="card"><h2>Actions</h2>{actions}</div>' if actions else ""
            body = f"""
<div class="card">
  <h2>Job: {job.job_id}</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Type</td><td>{job.job_type}</td></tr>
    <tr><td>Status</td><td>{_status_badge(job.status.value)}</td></tr>
    <tr><td>Created</td><td>{job.created_at.isoformat()}</td></tr>
    <tr><td>Progress</td><td>{job.progress_current}/{job.progress_total} — {_esc(job.progress_message)}</td></tr>
    <tr><td>Related Gate</td><td>{job.related_gate_id or "-"}</td></tr>
    <tr><td>Failure Reason</td><td>{_esc(job.failure_reason) or "-"}</td></tr>
    <tr><td>Next Action</td><td>{_esc(job.next_action_hint) or "-"}</td></tr>
  </table>
</div>
{actions_card}
<div class="card">
  <h2>Logs</h2>
  <table>
    <tr><th>Time</th><th>Level</th><th>Message</th></tr>
    {logs_html}
  </table>
</div>"""
        return _page(f"Job {job_id[:16]}", body)

    # ----- gates ----------------------------------------------------------- #
    @app.get("/api/gates")
    def list_gates(user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            # Only latest result per gate.
            seen: set[str] = set()
            out = []
            for r in rows:
                if r.gate_id in seen:
                    continue
                seen.add(r.gate_id)
                out.append(
                    {
                        "gate_id": r.gate_id,
                        "status": r.status.value,
                        "failure_reason": r.failure_reason,
                        "report_path": r.report_path,
                        "criteria": r.criteria,
                        "started_at": r.started_at.isoformat() if r.started_at else None,
                    }
                )
            return out

    @app.get("/api/gates/road-to-live")
    def road_to_live(user: str = Depends(require_dashboard_auth)) -> dict[str, Any]:
        """Road to Live: all gates with status, blocking criteria, and next action."""
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            latest_by_gate: dict[str, GateResult] = {}
            for r in rows:
                if r.gate_id not in latest_by_gate:
                    latest_by_gate[r.gate_id] = r

            gates_out = []
            critical_total = 0
            critical_passed = 0
            for gate_id, spec in catalog.items():
                result = latest_by_gate.get(gate_id)
                status = result.status.value if result else "not_run"
                is_critical = spec.blocks_live == "true"
                if is_critical:
                    critical_total += 1
                    if status == "passed":
                        critical_passed += 1

                blocking = [
                    dep
                    for dep in spec.depends_on
                    if latest_by_gate.get(dep) is None
                    or latest_by_gate[dep].status is not GateStatus.PASSED
                ]

                next_action = ""
                if status == "passed":
                    next_action = "DONE"
                elif blocking:
                    next_action = f"Fix upstream first: {', '.join(blocking)}"
                elif spec.remediation_steps:
                    next_action = spec.remediation_steps[0]
                else:
                    next_action = f"Re-run gate {gate_id}"

                gates_out.append(
                    {
                        "gate_id": gate_id,
                        "name": spec.name,
                        "phase": spec.phase,
                        "status": status,
                        "blocks_live": spec.blocks_live == "true",
                        "depends_on": spec.depends_on,
                        "blocking_dependencies": blocking,
                        "pass_condition": spec.pass_condition,
                        "next_action": next_action,
                        "remediation_steps": spec.remediation_steps,
                        "failure_reason": result.failure_reason if result else None,
                        "last_run": result.started_at.isoformat()
                        if result and result.started_at
                        else None,
                        "report_path": result.report_path if result else None,
                        "rerun_job": spec.rerun_job,
                    }
                )

            score = (critical_passed / critical_total * 100.0) if critical_total else 0.0
            return {
                "live_readiness_score": round(score, 1),
                "critical_gates_passed": critical_passed,
                "total_critical_gates": critical_total,
                "gates": gates_out,
            }

    @app.get("/api/gates/{gate_id}")
    def get_gate(gate_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        spec = catalog.get(gate_id)
        with session_scope() as session:
            result = session.execute(
                select(GateResult)
                .where(GateResult.gate_id == gate_id)
                .order_by(desc(GateResult.id))
                .limit(1)
            ).scalar_one_or_none()
            remediation = (
                session.execute(
                    select(RemediationAction)
                    .where(RemediationAction.gate_id == gate_id)
                    .order_by(RemediationAction.id.desc())
                    .limit(20)
                )
                .scalars()
                .all()
            )

        return {
            "gate_id": gate_id,
            "spec": {
                "name": spec.name if spec else gate_id,
                "phase": spec.phase if spec else "",
                "pass_condition": spec.pass_condition if spec else "",
                "remediation_steps": spec.remediation_steps if spec else [],
                "depends_on": spec.depends_on if spec else [],
                "blocks_live": (spec.blocks_live == "true") if spec else True,
                "rerun_job": spec.rerun_job if spec else "",
            },
            "latest_result": {
                "status": result.status.value if result else "not_run",
                "failure_reason": result.failure_reason if result else None,
                "criteria": result.criteria if result else [],
                "report_path": result.report_path if result else None,
                "started_at": result.started_at.isoformat()
                if result and result.started_at
                else None,
            },
            "remediation_actions": [
                {
                    "id": r.id,
                    "step_index": r.step_index,
                    "description": r.description,
                    "status": r.status.value,
                    "recommended_job": r.recommended_job,
                }
                for r in remediation
            ],
        }

    @app.get("/api/gates/{gate_id}/remediation")
    def gate_remediation(gate_id: str, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            actions = (
                session.execute(
                    select(RemediationAction)
                    .where(RemediationAction.gate_id == gate_id)
                    .order_by(RemediationAction.step_index)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id": a.id,
                    "step_index": a.step_index,
                    "description": a.description,
                    "status": a.status.value,
                    "recommended_job": a.recommended_job,
                    "owner_role": a.owner_role,
                    "created_at": a.created_at.isoformat(),
                }
                for a in actions
            ]

    @app.post("/api/gates/{gate_id}/run")
    def run_gate(gate_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue(
            "run_gate", {"gate_id": gate_id}, requested_by=user, related_gate_id=gate_id
        )
        _audit("run_gate", target=gate_id, actor=user, detail={"gate_id": gate_id})
        return {"job_id": job_id, "gate_id": gate_id}

    @app.post("/api/gates/run-all")
    def run_all_gates(user: str = Depends(require_dashboard_auth)) -> dict:
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue("run_all_gates", {}, requested_by=user)
        _audit("run_all_gates", target="gates", actor=user, detail={})
        return {"job_id": job_id}

    # ----- gates dashboard pages ------------------------------------------ #
    @app.get("/dashboard/gates", response_class=HTMLResponse)
    def dashboard_gates(user: str = Depends(require_dashboard_auth)) -> str:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            latest: dict[str, GateResult | None] = {}
            for r in rows:
                if r.gate_id not in latest:
                    latest[r.gate_id] = r

        table_rows = ""
        for gate_id, spec in catalog.items():
            gr = latest.get(gate_id)
            status = gr.status.value if gr else "not_run"
            last_run = gr.started_at.strftime("%Y-%m-%d %H:%M") if gr and gr.started_at else "never"
            table_rows += (
                f"<tr>"
                f"<td><a href='/dashboard/gates/{gate_id}'>{gate_id}</a></td>"
                f"<td>{spec.name}</td>"
                f"<td>{spec.phase}</td>"
                f"<td>{_status_badge(status)}</td>"
                f"<td>{last_run}</td>"
                f"<td><a href='/dashboard/gates/{gate_id}' class='btn btn-neutral' style='padding:2px 8px;font-size:12px'>Detail</a> "
                f"<form method='post' action='/api/gates/{gate_id}/run' style='display:inline'>"
                f"<button type='submit' class='btn' style='padding:2px 8px;font-size:12px'>▶ Run</button></form></td>"
                f"</tr>"
            )
        body = f"""
<div class="form-row">
  <form method="post" action="/api/gates/run-all">
    <button type="submit" class="btn">▶ Run All Gates</button>
  </form>
  <a href="/dashboard/road-to-live" class="btn btn-neutral">Road to Live →</a>
</div>
<div class="card">
  <h2>Gate Catalog ({len(catalog)} gates)</h2>
  <table>
    <tr><th>Gate ID</th><th>Name</th><th>Phase</th><th>Status</th><th>Last Run</th><th>Actions</th></tr>
    {table_rows or '<tr><td colspan="6" class="meta">No gates found.</td></tr>'}
  </table>
</div>"""
        return _page("Gates", body)

    @app.get("/dashboard/gates/{gate_id}", response_class=HTMLResponse)
    def dashboard_gate_detail(gate_id: str, user: str = Depends(require_dashboard_auth)) -> str:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        spec = catalog.get(gate_id)
        with session_scope() as session:
            result = session.execute(
                select(GateResult)
                .where(GateResult.gate_id == gate_id)
                .order_by(desc(GateResult.id))
                .limit(1)
            ).scalar_one_or_none()
            actions = (
                session.execute(
                    select(RemediationAction)
                    .where(RemediationAction.gate_id == gate_id)
                    .order_by(RemediationAction.step_index)
                )
                .scalars()
                .all()
            )

        status = result.status.value if result else "not_run"
        criteria_rows = ""
        if result and result.criteria:
            for c in result.criteria:
                c_status = c.get("status", "FAIL")
                c_badge = _status_badge(c_status.lower())
                detail = c.get("detail") or c.get("failure_reason") or ""
                criteria_rows += (
                    f"<tr><td>{c.get('id', '?')}</td><td>{c_badge}</td><td>{detail}</td></tr>"
                )
        remediation_html = ""
        if actions:
            remediation_html = "<h2>Remediation Steps</h2>"
            for a in actions:
                status_badge = _status_badge(a.status.value)
                # `recommended_job` (e.g. "gate:infra", "run_upstream_gates") is advisory
                # text, NOT an enqueuable job_type — POSTing it to /api/jobs/* returns 400.
                # The actionable remediation is to re-run THIS gate once the step is addressed,
                # so the button targets the working gate-rerun endpoint and shows the hint.
                run_btn = (
                    f'<form method="post" action="/api/gates/{gate_id}/run"'
                    f' style="display:inline" title="recommended: {a.recommended_job}">'
                    f'<button class="btn" type="submit"'
                    f' style="padding:2px 6px;font-size:11px">'
                    f"&#9654; re-run gate</button></form>"
                    if a.recommended_job
                    else ""
                )
                remediation_html += (
                    f'<div class="remediation-step">'
                    f"<b>Step {a.step_index + 1}:</b> {_esc(a.description)} "
                    f"{status_badge} {run_btn}"
                    f"</div>"
                )
        body = f"""
<p><a href="/dashboard/gates" class="btn btn-neutral">← All Gates</a>
   <form method="post" action="/api/gates/{gate_id}/run" style="display:inline;margin-left:8px">
     <button class="btn" type="submit">▶ Re-run Gate</button>
   </form></p>
<div class="card">
  <h2>Gate: {gate_id} — {spec.name if spec else gate_id}</h2>
  <p>Status: {_status_badge(status)}</p>
  <p class="meta">Phase: {spec.phase if spec else "?"} · Blocks live: {(spec.blocks_live if spec else "?")}</p>
  <p><b>Pass condition:</b> {spec.pass_condition if spec else "N/A"}</p>
  {f'<p class="meta"><b>Failure reason:</b> {_esc(result.failure_reason)}</p>' if result and result.failure_reason else ""}
  {f'<p class="meta">Last run: {result.started_at.isoformat() if result else "never"}</p>'}
  {f'<p class="meta">Report: <code>{result.report_path}</code></p>' if result and result.report_path else ""}
</div>
{(f'<div class="card"><h2>Criteria</h2><table><tr><th>ID</th><th>Status</th><th>Detail</th></tr>{criteria_rows}</table></div>') if criteria_rows else ""}
<div class="card">{remediation_html or '<p class="meta">No remediation actions.</p>'}</div>"""
        return _page(f"Gate: {gate_id}", body)

    @app.get("/dashboard/road-to-live", response_class=HTMLResponse)
    def dashboard_road_to_live(user: str = Depends(require_dashboard_auth)) -> str:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            latest: dict[str, GateResult | None] = {}
            for r in rows:
                if r.gate_id not in latest:
                    latest[r.gate_id] = r

        critical_total = sum(1 for s in catalog.values() if s.blocks_live == "true")
        critical_passed = sum(
            1
            for gid, s in catalog.items()
            if s.blocks_live == "true"
            and (gr2 := latest.get(gid)) is not None
            and gr2.status is GateStatus.PASSED
        )
        score = (critical_passed / critical_total * 100.0) if critical_total else 0.0
        score_cls = (
            "score" if score >= 80 else ("score score-mid" if score >= 50 else "score score-low")
        )

        table_rows = ""
        for gate_id, spec in catalog.items():
            gr = latest.get(gate_id)
            status = gr.status.value if gr else "not_run"
            blocking = [
                dep
                for dep in spec.depends_on
                if (dep_r := latest.get(dep)) is None or dep_r.status is not GateStatus.PASSED
            ]
            if status == "passed":
                next_action = "✓ Done"
            elif blocking:
                next_action = f"Fix upstream: {', '.join(blocking)}"
            elif spec.remediation_steps:
                next_action = spec.remediation_steps[0][:80]
            else:
                next_action = f"Re-run gate {gate_id}"

            table_rows += (
                f"<tr>"
                f"<td><a href='/dashboard/gates/{gate_id}'>{gate_id}</a></td>"
                f"<td>{spec.name}</td>"
                f"<td>{'✓' if spec.blocks_live == 'true' else ''}</td>"
                f"<td>{_status_badge(status)}</td>"
                f"<td style='max-width:300px'>{next_action}</td>"
                f"<td><form method='post' action='/api/gates/{gate_id}/run' style='display:inline'>"
                f"<button type='submit' class='btn' style='padding:2px 8px;font-size:12px'>▶ Run</button></form></td>"
                f"</tr>"
            )

        body = f"""
<div class="card">
  <h2>Live Readiness</h2>
  <div class="{score_cls}">{score:.0f}%</div>
  <p class="meta">{critical_passed} of {critical_total} critical gates passed</p>
  {'<p style="color:#3fb950">All critical gates PASS — system is ready for live activation (still requires manual approval).</p>' if score >= 100 else '<p class="meta">Fix all failed/blocked gates below to reach 100%.</p>'}
  {'<form method="post" action="/api/approvals?subject_type=live_activation&subject_id=LIVE" style="margin-top:8px"><button class="btn" type="submit">Request live-activation approval</button></form>' if score >= 100 else ""}
</div>
<div class="card">
  <h2>Gate Checklist</h2>
  <table>
    <tr><th>Gate</th><th>Name</th><th>Blocks Live</th><th>Status</th><th>Next Action</th><th>Run</th></tr>
    {table_rows}
  </table>
</div>"""
        return _page("Road to Live", body)

    # ----- remediation ----------------------------------------------------- #
    @app.get("/api/remediation")
    def list_remediation(
        gate_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(RemediationAction).order_by(desc(RemediationAction.id)).limit(limit)
            if gate_id:
                q = q.where(RemediationAction.gate_id == gate_id)
            if status:
                q = q.where(RemediationAction.status == status)
            actions = session.execute(q).scalars().all()
            return [
                {
                    "id": a.id,
                    "gate_id": a.gate_id,
                    "step_index": a.step_index,
                    "description": a.description,
                    "status": a.status.value,
                    "recommended_job": a.recommended_job,
                    "owner_role": a.owner_role,
                    "created_at": a.created_at.isoformat(),
                }
                for a in actions
            ]

    @app.post("/api/remediation/{action_id}/complete")
    def mark_remediation_complete(
        action_id: int, user: str = Depends(require_dashboard_auth)
    ) -> dict:
        with session_scope() as session:
            action = session.get(RemediationAction, action_id)
            if not action:
                raise HTTPException(status_code=404, detail="remediation action not found")
            action.status = RemediationStatus.DONE
        _audit(
            "remediation_complete",
            target=str(action_id),
            actor=user,
            detail={"action_id": action_id},
        )
        return {"id": action_id, "status": "done"}

    @app.get("/dashboard/remediation", response_class=HTMLResponse)
    def dashboard_remediation(user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            actions = (
                session.execute(
                    select(RemediationAction).order_by(desc(RemediationAction.id)).limit(200)
                )
                .scalars()
                .all()
            )

        rows = ""
        for a in actions:
            rows += (
                f"<tr>"
                f"<td>{a.gate_id}</td>"
                f"<td>Step {a.step_index + 1}</td>"
                f"<td>{_esc(a.description[:100])}</td>"
                f"<td>{_status_badge(a.status.value)}</td>"
                f"<td>{a.recommended_job or '-'}</td>"
                f"<td>"
                f'<a href="/dashboard/gates/{a.gate_id}" class="btn btn-neutral" style="padding:2px 6px;font-size:11px">Gate</a>'
                f"</td>"
                f"</tr>"
            )
        body = f"""
<div class="card">
  <h2>Remediation Actions ({len(actions)} shown)</h2>
  <p class="meta">These are ordered action items for non-PASS gates. A failed gate is never a dead end.</p>
  <table>
    <tr><th>Gate</th><th>Step</th><th>Description</th><th>Status</th><th>Job</th><th>Link</th></tr>
    {rows or '<tr><td colspan="6" class="meta">No remediation actions found.</td></tr>'}
  </table>
</div>"""
        return _page("Remediation Actions", body)

    # ----- approvals ------------------------------------------------------- #
    @app.post("/api/approvals")
    def create_approval(
        subject_type: str,
        subject_id: str,
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        """Raise a PENDING approval request (e.g. live activation) for an operator to decide.

        For ``live_activation`` the request carries a typed LiveActivationRequest (gate
        results + every version) as evidence, and is REFUSED unless the gates are green."""
        from src.approvals import request_approval

        evidence: dict[str, Any] = {}
        if subject_type == "live_activation":
            from src.live.activation import LiveActivationError, build_live_activation_request

            try:
                evidence = build_live_activation_request(
                    requested_by=user, settings=settings
                ).to_dict()
            except LiveActivationError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        approval_id = request_approval(
            subject_type, subject_id, requested_by=user, evidence=evidence
        )
        _audit(
            "approval_requested",
            target=f"{subject_type}:{subject_id}",
            actor=user,
            detail={"subject_type": subject_type, "subject_id": subject_id},
        )
        return {"id": approval_id, "status": "pending"}

    @app.get("/api/approvals")
    def list_approvals(
        limit: int = 50,
        status: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(Approval).order_by(desc(Approval.id)).limit(limit)
            if status:
                q = q.where(Approval.status == status)
            approvals = session.execute(q).scalars().all()
            return [
                {
                    "id": a.id,
                    "subject_type": a.subject_type,
                    "subject_id": a.subject_id,
                    "status": a.status.value,
                    "requested_by": a.requested_by,
                    "approver": a.approver,
                    "created_at": a.created_at.isoformat(),
                    "decided_at": a.decided_at.isoformat() if a.decided_at else None,
                    "evidence": a.evidence,
                }
                for a in approvals
            ]

    @app.post("/api/approvals/{approval_id}/approve")
    def approve(approval_id: int, user: str = Depends(require_dashboard_auth)) -> dict:
        from datetime import UTC, datetime

        with session_scope() as session:
            approval = session.get(Approval, approval_id)
            if not approval:
                raise HTTPException(status_code=404, detail="approval not found")
            if approval.status is not ApprovalStatus.PENDING:
                raise HTTPException(status_code=400, detail="approval is not pending")
            approval.status = ApprovalStatus.APPROVED
            approval.approver = user
            approval.decided_at = datetime.now(UTC)
        _audit("approval_approved", target=str(approval_id), actor=user, detail={})
        return {"id": approval_id, "status": "approved"}

    @app.post("/api/approvals/{approval_id}/reject")
    def reject(
        approval_id: int,
        reason: str = "rejected by operator",
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        from datetime import UTC, datetime

        with session_scope() as session:
            approval = session.get(Approval, approval_id)
            if not approval:
                raise HTTPException(status_code=404, detail="approval not found")
            if approval.status is not ApprovalStatus.PENDING:
                raise HTTPException(status_code=400, detail="approval is not pending")
            approval.status = ApprovalStatus.REJECTED
            approval.approver = user
            approval.decided_at = datetime.now(UTC)
            approval.evidence = {**approval.evidence, "rejection_reason": reason}
        _audit("approval_rejected", target=str(approval_id), actor=user, detail={"reason": reason})
        return {"id": approval_id, "status": "rejected"}

    @app.get("/dashboard/approvals", response_class=HTMLResponse)
    def dashboard_approvals(user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            approvals = (
                session.execute(select(Approval).order_by(desc(Approval.id)).limit(100))
                .scalars()
                .all()
            )

        rows = ""
        for a in approvals:
            if a.status is ApprovalStatus.PENDING:
                decide = (
                    f'<form method="post" action="/api/approvals/{a.id}/approve" '
                    f'style="display:inline"><button class="btn" type="submit" '
                    f'style="padding:2px 8px;font-size:11px">Approve</button></form> '
                    f'<form method="post" action="/api/approvals/{a.id}/reject" '
                    f'style="display:inline"><button class="btn btn-danger" type="submit" '
                    f'style="padding:2px 8px;font-size:11px">Reject</button></form>'
                )
            else:
                decide = '<span class="meta">—</span>'
            rows += (
                f"<tr>"
                f"<td>{_esc(a.subject_type)}</td>"
                f"<td>{_esc(a.subject_id[:32])}</td>"
                f"<td>{_status_badge(a.status.value)}</td>"
                f"<td>{_esc(a.requested_by)}</td>"
                f"<td>{_esc(a.approver or '-')}</td>"
                f"<td>{a.created_at.strftime('%Y-%m-%d %H:%M')}</td>"
                f"<td>{decide}</td>"
                f"</tr>"
            )
        body = f"""
<div class="card">
  <h2>Approvals ({len(approvals)} shown)</h2>
  <table>
    <tr><th>Type</th><th>Subject</th><th>Status</th><th>Requested By</th><th>Approver</th>
        <th>Created</th><th>Decide</th></tr>
    {rows or '<tr><td colspan="7" class="meta">No approvals found.</td></tr>'}
  </table>
</div>"""
        return _page("Approvals", body)

    # ----- audit logs ------------------------------------------------------ #
    @app.get("/api/audit-logs")
    def list_audit_logs(
        limit: int = 100,
        action: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(AuditLog).order_by(desc(AuditLog.id)).limit(limit)
            if action:
                q = q.where(AuditLog.action == action)
            logs = session.execute(q).scalars().all()
            return [
                {
                    "id": lg.id,
                    "ts": lg.ts.isoformat(),
                    "actor": lg.actor,
                    "action": lg.action,
                    "target": lg.target,
                    "environment": lg.environment,
                    "detail": lg.detail,
                }
                for lg in logs
            ]

    @app.get("/dashboard/audit-logs", response_class=HTMLResponse)
    def dashboard_audit_logs(user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            logs = (
                session.execute(select(AuditLog).order_by(desc(AuditLog.id)).limit(200))
                .scalars()
                .all()
            )

        rows = ""
        for lg in logs:
            rows += (
                f"<tr>"
                f"<td>{lg.ts.strftime('%Y-%m-%d %H:%M:%S')}</td>"
                f"<td>{_esc(lg.actor)}</td>"
                f"<td>{_esc(lg.action)}</td>"
                f"<td>{lg.target or '-'}</td>"
                f"<td>{lg.environment}</td>"
                f"</tr>"
            )
        body = f"""
<div class="card">
  <h2>Audit Log ({len(logs)} entries shown)</h2>
  <p class="meta">Immutable record of all system actions, approvals, config changes, gate runs, manual overrides.</p>
  <table>
    <tr><th>Timestamp</th><th>Actor</th><th>Action</th><th>Target</th><th>Env</th></tr>
    {rows or '<tr><td colspan="5" class="meta">No audit log entries found.</td></tr>'}
  </table>
</div>"""
        return _page("Audit Log", body)

    # ----- reports --------------------------------------------------------- #
    @app.get("/api/reports")
    def list_reports(user: str = Depends(require_dashboard_auth)) -> list[dict]:
        reports_path: Path = settings.reports_path
        if not reports_path.exists():
            return []
        out = []
        for p in sorted(
            reports_path.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
        )[:100]:
            rel = str(p.relative_to(reports_path))
            stat = p.stat()
            out.append(
                {
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                }
            )
        return out

    @app.get("/dashboard/reports", response_class=HTMLResponse)
    def dashboard_reports(user: str = Depends(require_dashboard_auth)) -> str:
        reports_path: Path = settings.reports_path
        reports = []
        if reports_path.exists():
            for p in sorted(
                reports_path.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
            )[:100]:
                rel = str(p.relative_to(reports_path))
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
                reports.append((rel, mtime.strftime("%Y-%m-%d %H:%M")))

        rows = "".join(
            f"<tr><td><code>{rel}</code></td><td>{mtime}</td></tr>" for rel, mtime in reports
        )
        body = f"""
<div class="card">
  <h2>Reports ({len(reports)} found)</h2>
  <p class="meta">Reports are stored under <code>{reports_path}</code></p>
  <table>
    <tr><th>Path</th><th>Modified</th></tr>
    {rows or '<tr><td colspan="2" class="meta">No reports found.</td></tr>'}
  </table>
</div>"""
        return _page("Reports", body)

    # ----- backtests ------------------------------------------------------- #
    @app.post("/api/backtests/run")
    def run_backtest(label: str = "", user: str = Depends(require_dashboard_auth)) -> dict:
        """Enqueue a background backtest (consumed by the dedicated `backtest` worker)."""
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue(
            "run_backtest",
            {"label": label or "dashboard_backtest", "requested_by": user},
            requested_by=user,
        )
        _audit("run_backtest", target=label or "dashboard_backtest", actor=user, detail={})
        return {"job_id": job_id}

    @app.get("/api/backtests")
    def list_backtests(limit: int = 50, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        from src.db.models import BacktestRun

        with session_scope() as session:
            rows = (
                session.execute(select(BacktestRun).order_by(desc(BacktestRun.created_at)))
                .scalars()
                .all()
            )[:limit]
            return [
                {
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "created_at": r.created_at.isoformat(),
                    "strategy_id": r.strategy_id,
                    "passed": r.passed,
                    "trade_count": r.trade_count,
                    "expectancy_r": r.expectancy_r,
                    "profit_factor": r.profit_factor,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "report_path": r.report_path,
                }
                for r in rows
            ]

    @app.get("/dashboard/backtests", response_class=HTMLResponse)
    def dashboard_backtests(user: str = Depends(require_dashboard_auth)) -> str:
        from src.backtest.config import load_backtest_config
        from src.db.models import BacktestRun

        try:
            acct = load_backtest_config().account
            init_eq, risk_pct = acct.initial_equity, acct.risk_pct
        except Exception:  # noqa: BLE001
            init_eq, risk_pct = 0.0, 0.0

        with session_scope() as session:
            rows = (
                session.execute(select(BacktestRun).order_by(desc(BacktestRun.created_at)))
                .scalars()
                .all()
            )[:100]
            runs = [
                (
                    r.run_id,
                    r.kind,
                    r.created_at.strftime("%Y-%m-%d %H:%M"),
                    bool(r.passed),
                    r.trade_count,
                    r.expectancy_r,
                    r.profit_factor,
                    r.total_return,
                    r.max_drawdown,
                )
                for r in rows
            ]

        body_rows = "".join(
            f"<tr><td><code>{_esc(rid)}</code></td><td>{_esc(kind)}</td><td>{created}</td>"
            f"<td>{_status_badge('passed' if passed else 'failed')}</td>"
            f"<td>{tc}</td><td>{er:+.4f}</td><td>{pf:.2f}</td>"
            f"<td>{ret:+.2%}</td><td>{dd:.2%}</td></tr>"
            for (rid, kind, created, passed, tc, er, pf, ret, dd) in runs
        )
        equity_card = f"""
<div class="card">
  <h2>How runs are compared</h2>
  <p>Every backtest starts from the <b>same fixed initial equity</b> of
     <b>{init_eq:,.0f}</b> (config <code>configs/backtest.yaml → account.initial_equity</code>),
     risking <b>{risk_pct * 100:.2f}%</b> per trade. Each run is <b>independent</b> — equity is
     reset to this value at the start and compounds <i>only within</i> that run; it is
     <b>never carried over</b> from a previous run. So all runs are directly comparable.</p>
  <p class="meta">The ranking metrics (<b>Expectancy R</b>, <b>Return %</b>) are normalised and
     equity-independent anyway, so the absolute equity is only a numeraire — change it in the
     config and every run still moves together.</p>
</div>"""
        body = equity_card + f"""
<div class="card">
  <h2>Backtests ({len(runs)})</h2>
  <form method="post" action="/api/backtests/run" style="margin-bottom:12px">
    <input type="text" name="label" placeholder="label (optional)" style="width:220px">
    <button class="btn" type="submit">&#9654; Run backtest</button>
  </form>
  <p class="meta">Runs execute in the background on the <code>backtest</code> worker; this
     page reads the <code>backtest_runs</code> index. Each run starts from the same fixed
     initial equity ({init_eq:,.0f}). Profitability is judged authoritatively
     by the BT / WF / FEE / SLIP gates.</p>
  <table>
    <tr><th>Run</th><th>Kind</th><th>Created</th><th>Net&gt;0</th><th>Trades</th>
        <th>Expectancy R</th><th>Profit Factor</th><th>Return</th><th>Max DD</th></tr>
    {body_rows or '<tr><td colspan="9" class="meta">No backtests yet — click Run backtest.</td></tr>'}
  </table>
</div>"""
        return _page("Backtests", body)

    # ----- leaderboard (M3 iteration comparison) --------------------------- #
    @app.get("/api/backtests/leaderboard")
    def api_leaderboard(
        kind: str = "backtest",
        dataset_version: str = "",
        strategy: str = "",
        limit: int = 50,
        best_per_iteration: bool = True,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        from src.backtest.leaderboard import build_leaderboard

        entries = build_leaderboard(
            kind=None if kind == "all" else kind,
            dataset_version=dataset_version or None,
            strategy_id=strategy or None,
            limit=limit,
            best_per_iteration=best_per_iteration,
        )
        return [e.to_dict() for e in entries]

    @app.get("/dashboard/leaderboard", response_class=HTMLResponse)
    def dashboard_leaderboard(user: str = Depends(require_dashboard_auth)) -> str:
        from src.backtest.leaderboard import build_leaderboard

        entries = build_leaderboard(limit=100, best_per_iteration=True)
        body_rows = "".join(
            f"<tr><td>{e.rank}</td><td><code>{_esc(e.run_id)}</code></td>"
            f"<td>{_esc(e.dataset_version or '—')}</td><td>{_esc(e.strategy_id)}</td>"
            f"<td>{_esc(e.timeframe or '—')}</td>"
            f"<td>{_status_badge('passed' if e.meets_bar else 'failed')}</td>"
            f"<td>{e.trade_count}</td><td>{e.expectancy_r:+.4f}</td>"
            f"<td>{e.profit_factor:.2f}</td><td>{e.total_return:+.2%}</td>"
            f"<td>{e.max_drawdown:.2%}</td></tr>"
            for e in entries
        )
        body = f"""
<div class="card">
  <h2>Iteration Leaderboard ({len(entries)})</h2>
  <p class="meta">Best run per (strategy, DATA_VERSION snapshot, timeframe), ranked by the
     profitability bar (expectancy &ge; 0.03R, PF &ge; 1.10, max-DD &le; 0.25, enough trades).
     Runs are immutable, so every research iteration is retained and comparable. <b>Meets bar</b>
     is a display flag — the BT / WF / FEE / SLIP gates remain the binding judgement before live.
     Add iterations with <code>qbot backtest-lake --config configs/data.bybit.yaml</code>.</p>
  <table>
    <tr><th>#</th><th>Run</th><th>Data Version</th><th>Strategy</th><th>TF</th><th>Meets bar</th>
        <th>Trades</th><th>Expectancy R</th><th>Profit Factor</th><th>Return</th><th>Max DD</th></tr>
    {body_rows or '<tr><td colspan="11" class="meta">No runs yet — run qbot backtest-lake.</td></tr>'}
  </table>
</div>"""
        return _page("Leaderboard", body)

    # ----- paper trading --------------------------------------------------- #
    @app.post("/api/paper/run")
    def run_paper(user: str = Depends(require_dashboard_auth)) -> dict:
        """Enqueue a background paper session over the promoted strategies."""
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue(
            "run_paper_session", {"session_name": "dashboard_paper"}, requested_by=user
        )
        _audit("run_paper_session", target="paper", actor=user, detail={})
        return {"job_id": job_id}

    @app.get("/dashboard/paper", response_class=HTMLResponse)
    def dashboard_paper(user: str = Depends(require_dashboard_auth)) -> str:
        from src.db.models import PaperRun

        with session_scope() as session:
            runs = (
                session.execute(select(PaperRun).order_by(desc(PaperRun.created_at)).limit(100))
                .scalars()
                .all()
            )
            rows = [
                (
                    r.session_id,
                    r.created_at.strftime("%Y-%m-%d %H:%M"),
                    r.executed_count,
                    r.rejected_count,
                    r.net_pnl,
                    r.expectancy_r,
                    r.win_rate,
                    ", ".join(r.strategies or []),
                )
                for r in runs
            ]

        body_rows = "".join(
            f"<tr><td><code>{_esc(sid)}</code></td><td>{created}</td><td>{ex}</td><td>{rej}</td>"
            f"<td>{net:+.2f}</td><td>{er:+.4f}</td><td>{wr:.0%}</td><td>{_esc(strats)}</td></tr>"
            for (sid, created, ex, rej, net, er, wr, strats) in rows
        )
        body = f"""
<div class="card">
  <h2>Paper Trading ({len(rows)})</h2>
  <form method="post" action="/api/paper/run" style="margin-bottom:12px">
    <button class="btn" type="submit">&#9654; Run paper session</button>
  </form>
  <p class="meta">Runs execute in the background and source candidates only from
     <b>promoted</b> strategies (the research promotion registry); trades persist to
     <code>paper_trades</code>.</p>
  <table>
    <tr><th>Session</th><th>Created</th><th>Executed</th><th>Rejected</th><th>Net PnL</th>
        <th>Expectancy R</th><th>Win Rate</th><th>Strategies</th></tr>
    {body_rows or '<tr><td colspan="8" class="meta">No paper sessions yet — click Run paper session.</td></tr>'}
  </table>
</div>"""
        return _page("Paper Trading", body)

    # ----- Strategies (sourcing, validation, active promoted set) ---------- #
    @app.post("/api/strategies/validate")
    def validate_strategies(user: str = Depends(require_dashboard_auth)) -> RedirectResponse:
        """Source + validate the candidate pool: run every enabled candidate through the
        research gate loop (backtest + walk-forward + fee/slippage stress + noise control) and
        persist promote/shelve verdicts. The live engine then runs the top-N promoted."""
        from src.jobs import JobQueue

        JobQueue(settings).enqueue("run_strategy_validation", {}, requested_by=user)
        _audit("run_strategy_validation", target="strategies", actor=user, detail={})
        return RedirectResponse(url="/dashboard/strategies", status_code=303)

    @app.get("/dashboard/strategies", response_class=HTMLResponse)
    def dashboard_strategies(user: str = Depends(require_dashboard_auth)) -> str:
        from src.strategies.config import load_strategies_config
        from src.strategies.promotion import promoted_strategy_details

        scfg = load_strategies_config()
        cap = scfg.max_active_strategies
        details = promoted_strategy_details(scfg.strategy_version)
        active = [d for d in details if d.active]
        enabled_pool = scfg.enabled_candidates()

        def _sides(d) -> str:
            s = []
            if d.allow_long:
                s.append("long")
            if d.allow_short:
                s.append("short")
            return "/".join(s) or "—"

        promoted_rows = [
            [
                f"<code>{_esc(d.candidate_id)}</code>",
                _esc(d.family),
                f"{d.expectancy_r:+.4f}",
                _sides(d),
                (
                    '<span class="badge pass">ACTIVE</span>'
                    if d.active
                    else '<span class="badge not_run">BENCHED</span>'
                ),
            ]
            for d in details
        ]
        pool_rows = [
            [f"<code>{_esc(c.id)}</code>", _esc(c.family), _esc(c.exit_profile)]
            for c in enabled_pool
        ]

        status = _kv_card(
            "Live strategy set",
            [
                ("candidate pool (enabled in config)", len(enabled_pool)),
                ("promoted (passed gates)", len(details)),
                ("active cap (max_active_strategies)", cap),
                ("running now (top-N by expectancy)", len(active)),
            ],
        )
        body = (
            status
            + '<div class="card"><h2>Source &amp; validate strategies</h2>'
            '<form method="post" action="/api/strategies/validate" style="margin-bottom:10px">'
            '<button class="btn" type="submit">&#9654; Source &amp; validate strategies</button>'
            "</form>"
            '<p class="meta">Runs every <b>enabled</b> candidate from '
            "<code>configs/strategies.yaml</code> through the research gate loop (backtest + "
            "walk-forward + fee/slippage stress + noise control) and writes promote/shelve "
            "verdicts. The live/demo engine then runs the <b>top "
            f"{cap}</b> promoted strategies by validated expectancy. Adding genuinely new "
            "strategy ideas means adding candidates to that config (a human hypothesis — there "
            "is no random strategy search, by design); this button re-sources and re-ranks the "
            "existing pool.</p></div>"
            '<div class="card"><h2>Promoted strategies (ranked)</h2>'
            + _rows_table(
                ["Candidate", "Family", "Expectancy R", "Sides", "State"],
                promoted_rows,
                "Nothing promoted yet — click Source & validate, then promote passing candidates.",
            )
            + f'<p class="meta">The top {cap} (ACTIVE) trade in live/demo; the rest stay '
            "promoted-but-benched until they rank into the top set.</p></div>"
            '<div class="card"><h2>Candidate pool (enabled in config)</h2>'
            + _rows_table(
                ["Candidate", "Family", "Exit profile"], pool_rows, "No enabled candidates."
            )
            + "</div>"
        )
        return _page("Strategies", body)

    # ----- ML shadow ------------------------------------------------------- #
    @app.post("/api/shadow/run")
    def run_shadow(user: str = Depends(require_dashboard_auth)) -> dict:
        """Enqueue a background ML shadow pass (predictions logged, never applied)."""
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue("run_ml_shadow_pass", {}, requested_by=user)
        _audit("run_ml_shadow_pass", target="ml_shadow", actor=user, detail={})
        return {"job_id": job_id}

    @app.get("/dashboard/shadow", response_class=HTMLResponse)
    def dashboard_shadow(user: str = Depends(require_dashboard_auth)) -> str:
        from src.db.models import ShadowLog

        with session_scope() as session:
            logs = (
                session.execute(select(ShadowLog).order_by(desc(ShadowLog.ts)).limit(200))
                .scalars()
                .all()
            )
            total = len(logs)
            applied = sum(1 for lg in logs if lg.applied)
            by_type: dict[str, int] = {}
            by_mode: dict[str, int] = {}
            for lg in logs:
                by_type[lg.model_type] = by_type.get(lg.model_type, 0) + 1
                by_mode[lg.mode] = by_mode.get(lg.mode, 0) + 1
            recent = [
                (
                    lg.ts.strftime("%Y-%m-%d %H:%M"),
                    lg.model_type,
                    lg.mode,
                    lg.symbol or "-",
                    f"{lg.confidence:.3f}" if lg.confidence is not None else "-",
                    lg.applied,
                )
                for lg in logs[:50]
            ]

        rows = "".join(
            f"<tr><td>{ts}</td><td>{_esc(mt)}</td><td>{_esc(mode)}</td><td>{_esc(sym)}</td>"
            f"<td>{conf}</td><td>{_status_badge('passed' if not ap else 'failed')}</td></tr>"
            for (ts, mt, mode, sym, conf, ap) in recent
        )
        applied_badge = _status_badge("passed") if applied == 0 else _status_badge("failed")
        body = f"""
<div class="card">
  <h2>ML Shadow ({total} recent predictions)</h2>
  <form method="post" action="/api/shadow/run" style="margin-bottom:12px">
    <button class="btn" type="submit">&#9654; Run ML shadow pass</button>
  </form>
  <p>Shadow-only enforcement — applied predictions: <b>{applied}</b> {applied_badge}
     (must be 0; ML can never affect a live decision until promoted).</p>
  <p class="meta">By model: {_esc(by_type)} · By mode: {_esc(by_mode)}</p>
  <table>
    <tr><th>Time</th><th>Model</th><th>Mode</th><th>Symbol</th><th>Confidence</th><th>Shadow-only</th></tr>
    {rows or '<tr><td colspan="6" class="meta">No shadow predictions yet — click Run ML shadow pass.</td></tr>'}
  </table>
</div>"""
        return _page("ML Shadow", body)

    # ----- alerts ---------------------------------------------------------- #
    @app.get("/api/alerts")
    def alerts(limit: int = 50, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        return [a.to_dict() for a in get_alert_sink().recent(limit=limit)]

    # ----- kill switch ----------------------------------------------------- #
    @app.get("/api/killswitch")
    def killswitch_status(user: str = Depends(require_dashboard_auth)) -> dict:
        from src.killswitch import KillSwitch

        return KillSwitch(settings).status()

    @app.post("/api/killswitch/engage")
    def killswitch_engage(
        reason: str = "dashboard manual kill",
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        from src.killswitch import KillSwitch

        KillSwitch(settings).engage(reason=reason, actor=f"dashboard:{user}")
        get_alert_sink().send(
            Alert(
                title="kill switch engaged",
                severity=AlertSeverity.CRITICAL,
                component="safety",
                environment=settings.app_env.value,
                recommended_action="Trading halted. Resume requires manual review (Section 35).",
            )
        )
        _audit("killswitch_engage", target="killswitch", actor=user, detail={"reason": reason})
        return KillSwitch(settings).status()

    @app.post("/api/killswitch/disengage")
    def killswitch_disengage(
        confirm: bool = False,
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        from src.killswitch import KillSwitch

        if not confirm:
            raise HTTPException(
                status_code=400, detail="disengage requires confirm=true (manual review)"
            )
        KillSwitch(settings).disengage(actor=f"dashboard:{user}")
        _audit("killswitch_disengage", target="killswitch", actor=user, detail={})
        return KillSwitch(settings).status()

    # ----- helpers --------------------------------------------------------- #
    def _audit(action: str, *, target: str, actor: str, detail: dict) -> None:
        from src.db.models import AuditLog

        try:
            with session_scope() as session:
                session.add(
                    AuditLog(
                        actor=actor,
                        action=action,
                        target=target,
                        environment=settings.app_env.value,
                        detail=detail,
                    )
                )
        except Exception:  # noqa: BLE001
            pass

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn / gunicorn)
# ---------------------------------------------------------------------------
from datetime import UTC, datetime  # noqa: E402 — import at bottom to avoid circular

app = create_app()
