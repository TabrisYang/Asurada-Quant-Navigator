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
        from app.core.auth.key_manager import key_manager

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

        return {
            "status": "success",
            "report": report,
            "summary": data["summary"],
        }
    except Exception as e:
        logger.error(f"覆盤報告生成失敗: {e}")
        return {
            "status": "error",
            "message": f"LLM 生成覆盤報告失敗: {e}",
            "summary": data["summary"],
        }


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
