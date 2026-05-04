"""阿斯拉量化系統 — v106 C4：Observability metrics 端點。

GET /api/system/metrics → 回傳系統健康關鍵指標：
- predictions_today / 7d
- 7-day rolling win rate（含 Wilson CI 下界）
- avg latency / token trend（近 7 天）
- LLM 工具失敗率（從 chat_history 抓 tool_failure pattern）
- prompt cache hit rate（若 provider 回報）

讓使用者（也讓你開發時）能在一個畫面看到健康狀態，提早發現異常。
所有資料都從現有 DB 聚合，不新增表。
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter

from app.core.config.settings import settings


router = APIRouter()
_PRED_DB = settings.db_path / "predictions.db"
_USAGE_DB = settings.db_path / "usage.db"
_CHAT_DB = settings.db_path / "chat_history.db"


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    pm = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - pm) / denom)


def _query_predictions_metrics() -> dict:
    """聚合 predictions.db 指標。"""
    if not _PRED_DB.exists():
        return {"error": "predictions.db 不存在"}

    out: dict = {}
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    month_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

    with sqlite3.connect(_PRED_DB) as conn:
        cur = conn.cursor()

        # 今日 / 本週 / 本月新增
        for label, since in [("today", today_iso), ("last_7d", week_ago), ("last_30d", month_ago)]:
            cur.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                (since,),
            )
            out[f"new_predictions_{label}"] = cur.fetchone()[0]

        # 7 日 / 30 日已驗證樣本勝率（hit_target 才算 win）
        for label, since in [("last_7d", week_ago), ("last_30d", month_ago)]:
            cur.execute(
                """SELECT
                     SUM(CASE WHEN status='hit_target' THEN 1 ELSE 0 END) AS wins,
                     COUNT(*) AS n
                   FROM predictions
                   WHERE status IN ('hit_target','hit_stop')
                     AND validated_at IS NOT NULL
                     AND validated_at >= ?""",
                (since,),
            )
            row = cur.fetchone()
            wins = row[0] or 0
            n = row[1] or 0
            out[f"validated_{label}"] = n
            out[f"winrate_{label}"] = round(wins / n, 3) if n > 0 else None
            out[f"winrate_ci_lower_{label}"] = round(_wilson_lower(wins, n), 3)

        # active 數量
        cur.execute("SELECT COUNT(*) FROM predictions WHERE status='active'")
        out["active_predictions"] = cur.fetchone()[0]

        # By regime breakdown（近 30 天）
        cur.execute(
            """SELECT regime,
                      SUM(CASE WHEN status='hit_target' THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN status IN ('hit_target','hit_stop') THEN 1 ELSE 0 END) AS validated
               FROM predictions
               WHERE created_at >= ?
               GROUP BY regime""",
            (month_ago,),
        )
        by_regime = []
        for regime, wins, n in cur.fetchall():
            wins = wins or 0
            n = n or 0
            by_regime.append({
                "regime": regime or "unknown",
                "validated": n,
                "wins": wins,
                "winrate": round(wins / n, 3) if n > 0 else None,
                "ci_lower": round(_wilson_lower(wins, n), 3),
            })
        out["by_regime_30d"] = sorted(by_regime, key=lambda r: r["validated"], reverse=True)

    return out


def _query_token_metrics() -> dict:
    """聚合 usage.db 的 token / cost 指標（近 7 天）。"""
    if not _USAGE_DB.exists():
        return {"error": "usage.db 不存在"}

    out: dict = {}
    week_ago_iso = (datetime.utcnow() - timedelta(days=7)).isoformat()

    with sqlite3.connect(_USAGE_DB) as conn:
        cur = conn.cursor()

        # 7 日總計
        cur.execute(
            """SELECT
                 COUNT(*) AS calls,
                 COALESCE(SUM(prompt_tokens), 0) AS p_tok,
                 COALESCE(SUM(completion_tokens), 0) AS c_tok,
                 COALESCE(SUM(total_tokens), 0) AS t_tok,
                 COALESCE(SUM(estimated_cost_usd), 0) AS cost
               FROM token_usage
               WHERE timestamp >= ?""",
            (week_ago_iso,),
        )
        calls, p_tok, c_tok, t_tok, cost = cur.fetchone()
        out["calls_last_7d"] = calls
        out["prompt_tokens_last_7d"] = p_tok
        out["completion_tokens_last_7d"] = c_tok
        out["total_tokens_last_7d"] = t_tok
        out["cost_usd_last_7d"] = round(cost or 0, 4)

        # By day（近 7 天 trend）
        cur.execute(
            """SELECT date(timestamp) AS d,
                      COUNT(*) AS calls,
                      SUM(total_tokens) AS toks,
                      SUM(estimated_cost_usd) AS cost
               FROM token_usage
               WHERE timestamp >= ?
               GROUP BY d
               ORDER BY d""",
            (week_ago_iso,),
        )
        out["daily_trend"] = [
            {
                "date": r[0],
                "calls": r[1],
                "tokens": r[2] or 0,
                "cost_usd": round(r[3] or 0, 4),
            }
            for r in cur.fetchall()
        ]

        # By provider（近 7 天）
        cur.execute(
            """SELECT provider, COUNT(*), SUM(total_tokens), SUM(estimated_cost_usd)
               FROM token_usage
               WHERE timestamp >= ?
               GROUP BY provider
               ORDER BY SUM(total_tokens) DESC""",
            (week_ago_iso,),
        )
        out["by_provider_7d"] = [
            {"provider": r[0], "calls": r[1], "tokens": r[2] or 0, "cost_usd": round(r[3] or 0, 4)}
            for r in cur.fetchall()
        ]

    return out


def _query_chat_metrics() -> dict:
    """聚合對話量 / tool failure 比率（從 chat_history.db）。"""
    if not _CHAT_DB.exists():
        return {"error": "chat_history.db 不存在"}

    out: dict = {}
    week_ago_iso = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with sqlite3.connect(_CHAT_DB) as conn:
        cur = conn.cursor()
        # tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]

        if "messages" in tables:
            cur.execute("SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (today_iso,))
            out["messages_today"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (week_ago_iso,))
            out["messages_last_7d"] = cur.fetchone()[0]

            # tool_failure pattern（"未回傳有效數據" / "tool failed"）
            cur.execute(
                """SELECT COUNT(*) FROM messages
                   WHERE timestamp >= ?
                   AND (content LIKE '%未回傳有效數據%' OR content LIKE '%工具失敗%' OR content LIKE '%tool failed%')""",
                (week_ago_iso,),
            )
            out["tool_failure_mentions_last_7d"] = cur.fetchone()[0]

    return out


@router.get("/metrics")
async def get_metrics() -> dict:
    """v106 C4：聚合系統關鍵 metrics（給 PredictionDashboard 加 metrics tab 用）。"""
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "predictions": _query_predictions_metrics(),
        "tokens": _query_token_metrics(),
        "chat": _query_chat_metrics(),
    }
