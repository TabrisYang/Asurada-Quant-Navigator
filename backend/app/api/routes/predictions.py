"""阿斯拉量化系統 — 預測追蹤 API

端點：
  GET   /api/predictions/stats        — 預測績效統計
  GET   /api/predictions/active       — 進行中的預測
  GET   /api/predictions/history      — 歷史預測
  PUT   /api/predictions/{id}/note    — 更新預測筆記
  POST  /api/predictions/review       — AI 生成覆盤報告
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query
from loguru import logger
from pydantic import BaseModel

from app.core.prediction_tracker import prediction_tracker

router = APIRouter()


@router.get("/stats")
async def get_prediction_stats(
    symbol: Optional[str] = Query(None),
    regime: Optional[str] = Query(None),
    days: int = Query(90, ge=7, le=365),
):
    """取得預測績效統計（命中率、指標勝率、多空對比、連勝連敗）"""
    stats = prediction_tracker.get_stats(symbol=symbol, regime=regime, days=days)
    direction = prediction_tracker.get_direction_stats(symbol=symbol, days=days)
    streak = prediction_tracker.get_recent_streak(symbol=symbol)
    return {
        "status": "success",
        "stats": stats,
        "direction_stats": direction,
        "streak": streak,
    }


@router.get("/active")
async def get_active_predictions(
    symbol: Optional[str] = Query(None),
):
    """取得進行中的預測"""
    active = prediction_tracker.get_active(symbol=symbol)
    return {"status": "success", "predictions": active, "count": len(active)}


@router.get("/history")
async def get_prediction_history(
    symbol: Optional[str] = Query(None),
    regime: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """取得歷史已驗證預測"""
    history = prediction_tracker.get_validated(symbol=symbol, regime=regime, limit=limit)
    return {"status": "success", "predictions": history, "count": len(history)}


# ─── v100：Calibration 面板專用 endpoint ──────────────────────
@router.get("/calibration")
async def get_calibration(
    symbol: Optional[str] = Query(None),
    days: int = Query(90, ge=7, le=365),
):
    """暴露 Brier / ECE / 信心桶 / Bayesian / regime / indicator 拆分數據（前端 Calibration 面板用）。"""
    stats = prediction_tracker.get_stats(symbol=symbol, days=days)
    return {
        "status": "success",
        "calibration": stats.get("calibration", {}),                      # Brier / ECE / reliability_curve
        "confidence_calibration": stats.get("confidence_calibration", {}),  # 信心 vs 命中率
        "bayesian": stats.get("bayesian", {}),                            # credible interval
        "by_regime": stats.get("by_regime", {}),                          # per-regime 拆分
        "by_indicator": stats.get("indicator_stats", {}),                 # per-indicator 拆分
        "win_rate_weighted": stats.get("win_rate_weighted", 0),
        "total": stats.get("total", 0),
    }


# ─── v101：模仿學習狀態 + Champion-Challenger 控制 ───────────
@router.get("/imitation/status")
async def get_imitation_status():
    """取目前模型版本 + AUC + Quality Gate + Canary 狀態（前端 PredictionDashboard 用）。"""
    from app.core.champion_challenger import get_status
    return get_status()


@router.post("/imitation/retrain")
async def retrain_imitation_endpoint(force: bool = Query(False)):
    """手動觸發重訓。force=True 跳過 Champion-Challenger 切換閾值。"""
    from app.core.imitation_trainer import train_imitation_model
    return train_imitation_model(min_samples=50, force=force)


@router.post("/imitation/disable")
async def disable_imitation_endpoint():
    """1-click 停用 v101 — 所有使用者立刻退回 v100 體驗。"""
    from app.core.champion_challenger import disable_v101
    return disable_v101()


@router.post("/imitation/rollback")
async def rollback_imitation_endpoint():
    """1-click 回退到 stable_fallback 模型。"""
    from app.core.champion_challenger import manual_rollback_to_stable
    return manual_rollback_to_stable()


@router.get("/imitation/quality_gate")
async def evaluate_quality_gate_endpoint():
    """手動觸發 Quality Gate 評估（不更動 canary）。"""
    from app.core.v101_self_validator import evaluate_v101_readiness
    return evaluate_v101_readiness()


@router.get("/imitation/last_run")
async def get_last_run_status():
    """讀最近一次 launchd 重訓的執行紀錄（給前端顯示「上次自動重訓何時、結果如何」）。"""
    from pathlib import Path
    import json
    status_path = Path("data/db/imitation_status.json")
    if not status_path.exists():
        return {"status": "no_run_yet", "message": "尚未執行過自動重訓"}
    try:
        return json.loads(status_path.read_text())
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── v100：圖表標註過往預測用 endpoint ────────────────────────
@router.get("/by_symbol")
async def get_predictions_for_chart(
    symbol: str = Query(..., description="必填：幣種如 BTC/USDT"),
    timeframe: Optional[str] = Query(None, description="若指定，僅回傳該時間框架的預測"),
    days: int = Query(90, ge=7, le=365),
):
    """給圖表用 — 拉某 symbol 過去 N 天的所有預測（含 active + validated）。

    用於 ChartView 自動標註過往預測（箭頭 + SL/TP 線 + 結果 ✓/✗）。
    """
    prediction_tracker._ensure_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    query = "SELECT * FROM predictions WHERE symbol = ? AND created_at >= ?"
    params: list = [symbol, cutoff]
    if timeframe:
        query += " AND timeframe = ?"
        params.append(timeframe)
    query += " ORDER BY created_at ASC"

    rows = prediction_tracker._conn.execute(query, params).fetchall()
    predictions = []
    for r in rows:
        d = dict(r)
        predictions.append({
            "id": d.get("id"),
            "created_at": d.get("created_at"),
            "validated_at": d.get("validated_at"),
            "expires_at": d.get("expires_at"),
            "status": d.get("status"),
            "direction": d.get("direction"),
            "entry_price": d.get("entry_price"),
            "target_price": d.get("target_price"),
            "stop_price": d.get("stop_price"),
            "timeframe_hours": d.get("timeframe_hours"),
            "confidence": d.get("confidence"),
            "regime": d.get("regime"),
            "indicators": d.get("indicators"),
            "actual_outcome_pct": d.get("actual_outcome_pct"),
            "max_favorable_pct": d.get("max_favorable_pct"),
            "max_adverse_pct": d.get("max_adverse_pct"),
            "invalidation": d.get("invalidation"),
        })
    return {"status": "success", "symbol": symbol, "count": len(predictions), "predictions": predictions}


class NoteUpdate(BaseModel):
    note: str


class ReviewRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbol: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None


@router.put("/{pred_id}/note")
async def update_prediction_note(pred_id: int, body: NoteUpdate):
    """更新預測的使用者筆記"""
    prediction_tracker.update_note(pred_id, body.note)
    return {"status": "success", "id": pred_id}


@router.delete("/clear")
async def clear_predictions(
    symbol: Optional[str] = Query(None, description="按幣種篩選，不填則清除全部"),
):
    """一鍵清除預測紀錄"""
    count = prediction_tracker.clear_all(symbol=symbol)
    return {
        "status": "success",
        "deleted": count,
        "message": f"已清除 {count} 筆預測紀錄",
    }


@router.get("/adjustments")
async def get_adjustments(
    symbol: Optional[str] = Query(None),
):
    """取得自動調整規則"""
    from app.core.auto_adjuster import get_active_adjustments
    adjustments = get_active_adjustments(symbol)
    return {"status": "success", "adjustments": adjustments, "count": len(adjustments)}


class AdjustmentOverride(BaseModel):
    value: Optional[float] = None


@router.put("/adjustments/{adj_id}/override")
async def override_adjustment(adj_id: int, body: AdjustmentOverride):
    """使用者手動覆蓋自動調整規則"""
    from app.core.auto_adjuster import auto_adjuster
    result = auto_adjuster.set_override(adj_id, body.value)
    return {"status": "success", "adjustment": result}


@router.post("/adjustments/recalculate")
async def recalculate_adjustments(
    symbol: Optional[str] = Query(None),
):
    """強制重新計算自動調整規則"""
    from app.core.auto_adjuster import run_adjustment_cycle
    result = run_adjustment_cycle(symbol)
    return {"status": "success", "result": result}


@router.get("/reviews")
async def get_review_history(limit: int = Query(10, ge=1, le=50)):
    """取得歷史覆盤報告列表"""
    reviews = prediction_tracker.get_reviews(limit=limit)
    return {"status": "success", "reviews": reviews, "count": len(reviews)}


@router.post("/review")
async def generate_review(body: ReviewRequest):
    """AI 生成覆盤報告

    取得指定期間的預測資料，用 LLM 分析成功/失敗模式，
    輸出覆盤報告並自動將關鍵教訓存入知識庫。
    """
    # 預設：過去 7 天
    if not body.start_date:
        body.start_date = (datetime.now() - timedelta(days=7)).isoformat()

    data = prediction_tracker.get_review_data(
        start_date=body.start_date,
        end_date=body.end_date,
        symbol=body.symbol,
    )

    if data["summary"]["total"] == 0:
        return {
            "status": "success",
            "report": "該期間內沒有已驗證的預測記錄，無法生成覆盤報告。",
            "summary": data["summary"],
        }

    # 構建覆盤 prompt
    preds_text = []
    for p in data["predictions"]:
        outcome = f"{p['actual_outcome_pct']:.2f}%" if p.get("actual_outcome_pct") is not None else "N/A"
        note = f" | 筆記: {p['notes']}" if p.get("notes") else ""
        preds_text.append(
            f"- {p['symbol']} {p['direction']} | 入場={p['entry_price']} "
            f"目標={p['target_price']} 止損={p['stop_price']} | "
            f"結果: {p['status']} ({outcome}) | "
            f"信心: {p['confidence']} | 指標: {p['indicators']} | "
            f"regime: {p['regime']}{note}"
        )

    summary = data["summary"]
    prompt = f"""你是一位量化交易覆盤教練。請根據以下預測記錄，生成一份覆盤報告。

## 統計摘要
- 總預測數: {summary['total']}
- 命中目標: {summary['wins']} ({summary['win_rate']}%)
- 觸及止損: {summary['losses']}
- 到期未觸發: {summary['expired']}

## 預測明細
{chr(10).join(preds_text)}

## 請分析以下幾點：
1. **成功模式** — 命中目標的預測有什麼共同特徵？（方向、指標組合、信心水準、regime）
2. **失敗模式** — 觸及止損的預測有什麼共同特徵？有哪些可以避免的錯誤？
3. **關鍵教訓** — 列出 3-5 條具體、可執行的改進建議
4. **下週建議** — 根據近期表現，建議調整什麼策略參數？

請用繁體中文回覆，格式清晰易讀。"""

    try:
        from app.core.llm.adapter import create_adapter
        from app.core.security.key_manager import key_manager

        provider = body.provider or "openai"
        api_key = body.api_key
        model_name = body.model_name

        if body.session_id:
            session_info = key_manager.get_session_info(body.session_id)
            if session_info:
                provider = session_info["provider"]
                model_name = model_name or session_info.get("model_name")
                api_key = api_key or key_manager.get_key(body.session_id)

        adapter = create_adapter(
            provider=provider, api_key=api_key, model_name=model_name,
        )
        llm_response = await adapter.chat(
            messages=[{"role": "user", "content": prompt}],
        )
        report = llm_response.message

        # 嘗試將關鍵教訓存入知識庫
        try:
            _save_lessons_from_review(report, body.symbol)
        except Exception as e:
            logger.warning(f"覆盤教訓存入知識庫失敗: {e}")

        # 覆盤後觸發自動調整
        try:
            from app.core.auto_adjuster import run_adjustment_cycle
            run_adjustment_cycle(symbol=body.symbol)
        except Exception as e:
            logger.warning(f"覆盤後自動調整失敗: {e}")

        # 儲存報告到 review_log
        prediction_tracker.save_review(
            report=report, symbol=body.symbol, summary=data["summary"], is_auto=False,
        )

        return {
            "status": "success",
            "report": report,
            "summary": data["summary"],
        }
    except Exception as e:
        logger.error(f"覆盤報告生成失敗: {e}")
        # Fallback：生成模板報告（無需 LLM）
        report = _generate_template_report(data)
        prediction_tracker.save_review(
            report=report, symbol=body.symbol, summary=data["summary"], is_auto=False,
        )
        return {
            "status": "success",
            "report": report,
            "summary": data["summary"],
            "fallback": True,
        }


def _generate_template_report(data: dict) -> str:
    """無 LLM 時的 fallback 模板覆盤報告。"""
    s = data["summary"]
    preds = data["predictions"]

    lines = [
        "# 覆盤報告（自動模板）\n",
        "## 統計摘要",
        f"- 總預測數: {s['total']}",
        f"- 命中目標: {s['wins']} ({s['win_rate']}%)",
        f"- 觸及止損: {s['losses']}",
        f"- 到期未觸發: {s['expired']}\n",
    ]

    # 成功 / 失敗分類
    hits = [p for p in preds if p["status"] == "hit_target"]
    stops = [p for p in preds if p["status"] == "hit_stop"]

    if hits:
        lines.append("## 命中目標的預測")
        for p in hits[:10]:
            outcome = f"{p['actual_outcome_pct']:+.2f}%" if p.get("actual_outcome_pct") is not None else "N/A"
            lines.append(
                f"- {p['symbol']} {p['direction']} | "
                f"信心:{p['confidence']} | 指標:{p['indicators']} | "
                f"regime:{p['regime']} | 結果:{outcome}"
            )

    if stops:
        lines.append("\n## 觸及止損的預測")
        for p in stops[:10]:
            outcome = f"{p['actual_outcome_pct']:+.2f}%" if p.get("actual_outcome_pct") is not None else "N/A"
            mfe = f"{p['max_favorable_pct']:+.2f}%" if p.get("max_favorable_pct") is not None else "N/A"
            lines.append(
                f"- {p['symbol']} {p['direction']} | "
                f"信心:{p['confidence']} | 指標:{p['indicators']} | "
                f"regime:{p['regime']} | 結果:{outcome} | 最大有利:{mfe}"
            )

    # 指標統計
    ind_count: dict[str, list[int, int]] = {}
    for p in preds:
        is_win = p["status"] == "hit_target"
        for ind in (p.get("indicators") or "").split(","):
            ind = ind.strip()
            if not ind:
                continue
            if ind not in ind_count:
                ind_count[ind] = [0, 0]
            ind_count[ind][1] += 1
            if is_win:
                ind_count[ind][0] += 1

    if ind_count:
        lines.append("\n## 指標勝率統計")
        sorted_inds = sorted(ind_count.items(), key=lambda x: x[1][0] / max(x[1][1], 1), reverse=True)
        for ind, (wins, total) in sorted_inds:
            wr = round(wins / total * 100, 1) if total > 0 else 0
            tier = "★★★" if wr >= 55 else ("★★" if wr >= 40 else "★")
            lines.append(f"- {tier} {ind}: {wr}% ({wins}/{total})")

    lines.append("\n> 此報告為自動模板生成。如需更深入的 AI 分析，請先設定 LLM 金鑰。")
    return "\n".join(lines)


def _save_lessons_from_review(report: str, symbol: Optional[str] = None):
    """從覆盤報告中提取教訓，存入知識碎片庫"""
    # 提取「關鍵教訓」區塊中的條目
    lesson_pattern = re.compile(r"(?:^|\n)\s*\d+[\.\)]\s*\*{0,2}(.+?)(?:\n|$)")
    # 尋找教訓區段
    section_match = re.search(r"關鍵教訓.*?\n((?:.*?\n)*?)(?:\n##|\n\d+\.\s*\*{0,2}下|$)", report)
    if not section_match:
        return

    section = section_match.group(1)
    lessons = lesson_pattern.findall(section)

    if not lessons:
        return

    from app.core.knowledge_fragments import fragment_store

    for lesson_text in lessons[:5]:  # 最多存 5 條
        lesson_text = lesson_text.strip().strip("*").strip()
        if len(lesson_text) < 10:
            continue
        fragment_store.store_fragment(
            content=lesson_text,
            fragment_type="lesson",
            symbol=symbol or "",
            source_question="prediction_review",
        )
        logger.info(f"覆盤教訓已存入知識庫: {lesson_text[:50]}...")
