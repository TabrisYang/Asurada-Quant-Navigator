"""阿斯拉量化系統 — v103 6A：經濟日曆事件注入器（不付費版）。

從 backend/data/calendar/events.json 讀取手動維護的高影響事件，
注入到 chart_state，給 LLM prompt 警示用。

使用方式：
    from app.core.event_injector import get_upcoming_events
    events = get_upcoming_events(within_hours=72)
    chart_state["upcoming_events"] = events
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

_CALENDAR_PATH = Path(__file__).resolve().parents[2] / "data" / "calendar" / "events.json"


def _load_events() -> dict:
    if not _CALENDAR_PATH.exists():
        return {"events": [], "_meta": {}}
    try:
        return json.loads(_CALENDAR_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"無法讀取 events.json: {e}")
        return {"events": [], "_meta": {}}


def _parse_event_dt(event: dict) -> Optional[datetime]:
    """解析 event 的 UTC datetime。"""
    date = event.get("date")
    time = event.get("time_utc", "12:00")
    if not date:
        return None
    try:
        return datetime.fromisoformat(f"{date}T{time}").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def get_upcoming_events(
    within_hours: int = 72,
    min_severity: str = "medium",
    scope_match: Optional[str] = None,
) -> list[dict]:
    """取得未來 N 小時內、嚴重度 >= min_severity 的事件。

    Args:
        within_hours: 看未來幾小時內的事件（預設 72h = 3 天）
        min_severity: low / medium / high（過濾掉低於此級的）
        scope_match: 若給，只回傳 scope 包含此關鍵字的事件
                     （例：scope_match='crypto' → 過濾掉純 equities 事件）

    Returns:
        list of dict，含 name / severity / event_dt_utc / hours_until / note。
        若無事件回 []。
    """
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = severity_rank.get(min_severity, 1)

    raw = _load_events().get("events", [])
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(hours=within_hours)

    out: list[dict] = []
    for ev in raw:
        ev_dt = _parse_event_dt(ev)
        if not ev_dt:
            continue
        if ev_dt < now_utc or ev_dt > cutoff:
            continue
        if severity_rank.get(ev.get("severity", "low"), 0) < min_rank:
            continue
        if scope_match and scope_match.lower() not in (ev.get("scope", "").lower()):
            continue

        hours_until = (ev_dt - now_utc).total_seconds() / 3600
        out.append({
            "name": ev.get("name"),
            "severity": ev.get("severity"),
            "category": ev.get("category"),
            "scope": ev.get("scope"),
            "event_dt_utc": ev_dt.isoformat(),
            "hours_until": round(hours_until, 1),
            "note": ev.get("note", ""),
        })

    out.sort(key=lambda e: e["hours_until"])
    return out


def calendar_age_days() -> Optional[float]:
    """經濟日曆「最後更新」距今天數，給 UI 顯示「過期警示」。"""
    meta = _load_events().get("_meta", {})
    last = meta.get("last_updated")
    if not last:
        return None
    try:
        last_dt = datetime.fromisoformat(last)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - last_dt).total_seconds() / 86400, 1)
    except Exception:
        return None


def format_events_warning(events: list[dict]) -> str:
    """把事件 list 格式化成 LLM prompt 用的警示文字。"""
    if not events:
        return ""
    lines = ["⚠️ 未來 72 小時內事件警示："]
    for e in events:
        sev_emoji = "🔴" if e["severity"] == "high" else "🟡"
        lines.append(
            f"  {sev_emoji} {e['hours_until']:.0f}h 後｜{e['name']}（{e.get('category','?')}）"
            + (f"｜{e['note']}" if e.get('note') else "")
        )
    lines.append("  → 建議：高嚴重度事件前 24h 慎用槓桿、縮小倉位、避免新進場。")
    return "\n".join(lines)
