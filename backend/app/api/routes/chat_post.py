"""chat 路由的回應後處理層（v157 拆分：從 chat.py 機械搬移，邏輯零改動）。

職責：串流結束後的落地工作 — 命中率佔位行替換、對話/快取/知識碎片/預測寫入
DB、串流用的文字分塊。

⚠️ _post_process_chat_message 全是 sync SQLite 操作，caller 必須用
asyncio.to_thread 丟到 thread pool（v117 修 backend hang 的關鍵，見 docstring）。
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.chat_history import chat_history
from app.core.analysis_cache import analysis_cache
from app.core.semantic_cache import semantic_cache
from app.core.knowledge_fragments import (
    fragment_store, parse_key_insights, strip_key_insights,
    parse_system_distill, strip_system_distill,
)
from app.core.prediction_tracker import (
    prediction_tracker, parse_predictions, strip_predictions,
)
from app.core.prediction_validator import validate_all_active
from app.core.symbol_extractor import extract_symbol_from_text
from app.utils.timezone import taipei_now


# ─── v100/v103 1B：結論卡「📈 系統參考」自動注入歷史命中率 ────────
# v103 1B 寬鬆化：容忍 LLM 寫不同字眼（將/由/會/即將...）
_PLACEHOLDER_PATTERN = re.compile(
    r"📈\s*系統參考[：:][^\n]*",
)


def _inject_recent_accuracy(final_text: str, symbol: str, regime: str) -> str:
    """v100：替換結論卡「📈 系統參考：」佔位行為實際歷史命中率。

    若 final_text 不含此佔位行（不是結論卡格式），原樣返回。
    若樣本不足（< 3 筆驗證），顯示「樣本不足」訊息。
    """
    matched = _PLACEHOLDER_PATTERN.search(final_text)
    if not matched:
        logger.debug("[v103 1B] _inject_recent_accuracy: 沒找到 placeholder，可能 LLM 沒產出結論卡")
        return final_text
    logger.info(f"[v103 1B] 命中 placeholder: '{matched.group(0)[:80]}'")
    try:
        stats = prediction_tracker.get_stats(symbol=symbol, regime=regime, days=30)
    except Exception as e:
        logger.warning(f"_inject_recent_accuracy 取統計失敗：{e}")
        return final_text

    total = stats.get("total", 0)

    # v108 Phase 3：另查 invalidated 筆數，附加到 hit-rate 後讓使用者知道
    invalidated_n = 0
    try:
        prediction_tracker._ensure_db()
        with prediction_tracker._lock:
            from datetime import timedelta as _td
            cutoff = (taipei_now() - _td(days=30)).isoformat()
            _q = "SELECT COUNT(*) FROM predictions WHERE status='invalidated' AND created_at > ?"
            _params: list = [cutoff]
            if symbol:
                _q += " AND symbol = ?"
                _params.append(symbol)
            if regime:
                _q += " AND regime = ?"
                _params.append(regime)
            row = prediction_tracker._conn.execute(_q, _params).fetchone()
            if row:
                invalidated_n = int(row[0] or 0)
    except Exception as _inv_err:
        logger.debug(f"_inject_recent_accuracy 取 invalidated 數失敗：{_inv_err}")

    if total < 3:
        replacement = f"📈 系統參考：樣本不足（n={total}，需 ≥ 3 筆已驗證），命中率尚不可估"
    else:
        wr = stats.get("win_rate_weighted", 0)
        bayesian = stats.get("bayesian", {})
        ci = bayesian.get("credible_interval_95", [None, None])
        ci_str = f"，CI {ci[0]}-{ci[1]}%" if ci[0] is not None else ""
        replacement = (
            f"📈 系統參考：你近 30 天類似條件（{regime}）命中率 {wr}%（n={total}{ci_str}）"
        )

    # 若有 invalidated，附加說明
    if invalidated_n > 0:
        replacement += f"｜📛 另 {invalidated_n} 筆因失效條件觸發已排除（不計入命中率）"

    return _PLACEHOLDER_PATTERN.sub(replacement, final_text)


def _post_process_chat_message(
    *,
    final_text: str,
    request_message: str,
    chart_state: Optional[dict],
    chart_symbol_for_save: Optional[str],
    conversation_id: str,
    total_usage,  # TokenUsage | None — 不嚴格型別避免循環 import
) -> None:
    """串流結束後的所有 DB 寫入和知識提取（v117：純 sync 跑在 thread pool）。

    ★ v117 改動：本函式內部全是 sync 操作（chat_history / analysis_cache /
       semantic_cache / fragment_store / prediction_tracker 都是 sync SQLite），
       原本 `async def` 但內部沒 `await` → 仍跑在主 event loop 上。當 SQLite
       fsync / embedding encode 慢時，會阻塞所有其他 endpoints，造成 backend
       hang（已實測 reproduce）。改成 sync function 由 caller 用
       `asyncio.create_task(asyncio.to_thread(...))` 丟到 thread pool 跑。

    ★ 設計理由：原本這段同步跑在 stream_gen 內，會造成 30-300 秒沉默
       （embedding 計算、predictions 驗證、OHLCV 重抓），導致前端誤判
       「分析回應超時」。改成 background task 後 SSE 可以立刻 yield done，
       使用者體驗即時、後處理悄悄完成。

    所有錯誤只 log 不 raise（不影響使用者已收到的回應）。
    """
    try:
        if not final_text.strip():
            return

        insights = parse_key_insights(final_text)
        distill_fragments = parse_system_distill(final_text)
        clean_text = strip_system_distill(strip_predictions(strip_key_insights(final_text)))

        usage_dict_for_save = total_usage.to_dict() if total_usage else None
        chat_history.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=clean_text,
            token_usage=usage_dict_for_save,
        )

        analysis_cache.store(
            question=request_message,
            answer=clean_text,
            chart_state=chart_state,
        )

        semantic_cache.store(
            question=request_message,
            answer=clean_text,
            chart_state=chart_state,
        )

        frag_symbol = extract_symbol_from_text(request_message) or chart_symbol_for_save or ""
        if insights:
            stored = fragment_store.store_batch(
                fragments=insights,
                symbol=frag_symbol,
                source_question=request_message,
            )
            if stored:
                logger.info(f"[bg] 自動提取 {stored} 筆知識碎片（{frag_symbol}）")

        if distill_fragments:
            stored_d = fragment_store.store_batch(
                fragments=distill_fragments,
                symbol=frag_symbol,
                source_question=request_message,
            )
            if stored_d:
                logger.info(f"[bg] 自動提取 {stored_d} 筆蒸餾碎片（{frag_symbol}）")

        # 提取並存儲預測（v100：低信心過濾，避免汙染命中率統計）
        predictions = parse_predictions(final_text)
        # v100：過濾掉信心="low" 的預測（雙保險，prompt 已要求低信心改用「建議觀望」格式）
        before_filter = len(predictions)
        predictions = [p for p in predictions if p.get("confidence") != "low"]
        if before_filter > len(predictions):
            logger.info(f"[bg] 過濾 {before_filter - len(predictions)} 筆低信心預測（不追蹤）")

        if predictions:
            pred_symbol = extract_symbol_from_text(request_message) or chart_symbol_for_save or ""
            pred_tf = (chart_state or {}).get("timeframe", "4h")
            for pred in predictions:
                try:
                    pred_id = prediction_tracker.store(
                        symbol=pred_symbol,
                        timeframe=pred_tf,
                        prediction=pred,
                        source_question=request_message,
                        chart_state=chart_state,  # v120.3: capture external_signals
                    )
                    # ★ Phase 2.0c：即時記錄 39 個特徵快照（feature flag 保護）
                    if pred_id and settings.feature_recording_enabled:
                        try:
                            from app.core.feature_extractor import record_features
                            from app.data.fetchers.crypto_engine import crypto_engine
                            df = crypto_engine.load_local_data(pred_symbol, pred_tf)
                            if df is not None and not df.empty:
                                pred_for_features = {**pred, "symbol": pred_symbol}
                                record_features(pred_id, df, chart_state, pred_for_features)
                        except Exception as fe:
                            logger.debug(f"[bg] 特徵記錄失敗（不影響預測）: {fe}")
                except Exception as pe:
                    logger.warning(f"[bg] 儲存預測失敗: {pe}")
            logger.info(f"[bg] 自動提取 {len(predictions)} 筆預測（{pred_symbol}）")

        # 追蹤 ML 預測（若存在）
        ml_pred = (chart_state or {}).get("mlPrediction")
        if ml_pred and isinstance(ml_pred, dict) and ml_pred.get("probability"):
            try:
                _ml_symbol = chart_symbol_for_save or ""
                _ml_tf = (chart_state or {}).get("timeframe", "4h")
                _ml_direction = ml_pred.get("direction", "long")
                _ml_prob = ml_pred.get("probability", 0.5)
                _ml_entry = (chart_state or {}).get("current_price", 0)
                if _ml_direction == "long":
                    _ml_target = _ml_entry * 1.03
                    _ml_stop = _ml_entry * 0.97
                else:
                    _ml_target = _ml_entry * 0.97
                    _ml_stop = _ml_entry * 1.03
                _ml_conf = "high" if _ml_prob >= 0.7 else ("medium" if _ml_prob >= 0.5 else "low")
                ml_pred_data = {
                    "direction": _ml_direction,
                    "entry_price": _ml_entry,
                    "target_price": round(_ml_target, 2),
                    "stop_price": round(_ml_stop, 2),
                    "timeframe_hours": 24,
                    "confidence": _ml_conf,
                    "regime": ml_pred.get("regime", "unknown"),
                    "indicators": "ml_model",
                }
                _ml_pid = prediction_tracker.store(
                    symbol=_ml_symbol, timeframe=_ml_tf,
                    prediction=ml_pred_data, source_question="ml_auto",
                    chart_state=chart_state,  # v120.3
                )
                prediction_tracker._ensure_db()
                prediction_tracker._conn.execute(
                    "UPDATE predictions SET ml_enhanced=1 WHERE id=?", (_ml_pid,)
                )
                prediction_tracker._conn.commit()
                logger.info(f"[bg] ML 預測已追蹤 #{_ml_pid}: {_ml_symbol} {_ml_direction}")
            except Exception as me:
                logger.warning(f"[bg] 儲存 ML 預測失敗: {me}")

        # 順便驗證已到期的預測
        try:
            val_result = validate_all_active()
            if val_result.get("validated", 0) > 0:
                logger.info(f"[bg] 自動驗證 {val_result['validated']} 筆預測")
        except Exception as ve:
            logger.warning(f"[bg] 自動驗證預測失敗: {ve}")

    except Exception as save_err:
        logger.error(f"[bg] post-processing 整體失敗（不影響使用者）: {save_err}")


def _split_text_for_streaming(text: str) -> list[str]:
    """
    將文字切割成適合串流的 chunks。

    策略：按標點/換行自然斷句，每 chunk 約 20-80 字元。
    這樣串流看起來像真人在打字。
    """
    if len(text) <= 30:
        return [text]

    chunks: list[str] = []
    current = ""

    for char in text:
        current += char
        # 在標點符號或換行處斷開
        if char in "。！？\n；：，、」）】" and len(current) >= 8:
            chunks.append(current)
            current = ""
        elif len(current) >= 60:
            # 太長了，強制斷開
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)

    return chunks
