"""阿斯拉量化系統 v108 Phase 3：失效條件 watcher

把報告中的「失效條件」（如「跌破 $X」「突破 $Y + 放量」）從純文字裝飾，
升級為系統可實際監控、自動標記預測狀態的機制。

設計原則（user explicit）：
- 系統不接交易所自動下單，watcher 只**標記預測狀態**為 invalidated
- 標記後從 hit-rate 分母排除（避免污染統計），且報告中可見「📛 此分析已失效」
- 風險控制：60 分鐘最小存活時間 + 2 根 K 線確認（防 wick 誤觸）
- shadow mode（settings.imitation_shadow_mode=True）：只記錄不改 status

觸發點：
- data_sync 完成某 symbol+tf → sweep_symbol(symbol, tf)
- 啟動時冷啟動 → startup_sweep()（用既有本地資料，不強制重抓）

範圍精簡（v108）：
- 條件：價格條件（price_above / price_below）+ 量過濾（volume_mult）
- 動作：mark invalidated（不嘗試方向切換 / 不嘗試平倉）
- 事件條件（FOMC 等）由 event_calendar_sync 預先處理，不在此範圍
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from app.core.config.settings import settings
from app.core.prediction_tracker import prediction_tracker
from app.utils.timezone import taipei_now

# ─── 風險控制常數 ─────────────────────────────────────
_MIN_ALIVE_MINUTES = 60       # 預測建立後至少存活 60 分鐘才考慮失效（防剛開單就被 wick）
_CONFIRMATION_BARS = 2        # 連續 N 根 K 線都滿足才觸發（防單根 wick）


def _check_one_condition(bar: pd.Series, cond: dict, prev_volume_avg: Optional[float]) -> bool:
    """單根 K 線是否滿足單一失效條件。

    bar: 含 high/low/close/volume 的 K 線資料
    cond: {"type": "price_above"|"price_below", "threshold": X, "volume_mult": optional}
    prev_volume_avg: 用於量過濾的均量（None 代表跳過量過濾）
    """
    cond_type = cond.get("type")
    threshold = cond.get("threshold")
    if threshold is None:
        return False

    # 用 high/low 判定穿越（一根 K 線只要 high > threshold 就算穿越上緣）
    price_hit = False
    if cond_type == "price_above":
        price_hit = float(bar.get("high", 0)) >= float(threshold)
    elif cond_type == "price_below":
        price_hit = float(bar.get("low", 0)) <= float(threshold)
    else:
        return False  # 未知條件型別

    if not price_hit:
        return False

    # 量過濾（若有指定）
    vol_mult = cond.get("volume_mult")
    if vol_mult is not None and prev_volume_avg is not None and prev_volume_avg > 0:
        if float(bar.get("volume", 0)) < float(vol_mult) * prev_volume_avg:
            return False  # 有條件但量不夠

    return True


def _check_prediction_against_df(pred: dict, df: pd.DataFrame, last_sweep_ts: Optional[datetime]) -> Optional[dict]:
    """檢查單筆預測在 df 範圍內是否觸發失效條件。

    Returns:
        - None：未觸發
        - dict：{"trigger_bar_ts": "...", "trigger_desc": "...", "matched_condition": {...}}
    """
    inv_json = pred.get("invalidation_json")
    if not inv_json:
        return None  # 無結構化條件 → 跳過（保留為純文字裝飾）

    try:
        inv_data = json.loads(inv_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug(f"預測 #{pred.get('id')} invalidation_json 格式錯誤，跳過")
        return None

    conditions = inv_data.get("conditions") or []
    if not conditions:
        return None

    # 預測建立時間（用於 60 分鐘最小存活時間）
    try:
        created_at = datetime.fromisoformat(pred["created_at"])
    except (ValueError, KeyError, TypeError):
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=taipei_now().tzinfo)

    # 篩 df 內「預測建立 + 60 分鐘」之後的 K 線
    min_check_time = created_at + timedelta(minutes=_MIN_ALIVE_MINUTES)
    if "timestamp" not in df.columns:
        return None

    df_ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df_filtered = df[df_ts >= pd.Timestamp(min_check_time)].copy()
    if df_filtered.empty:
        return None

    # 若有上次 sweep 時間，從那之後開始檢（避免重掃舊資料），但仍從 min_check_time 限制起
    if last_sweep_ts is not None:
        start_from = max(last_sweep_ts, min_check_time)
        df_filtered = df_filtered[pd.to_datetime(df_filtered["timestamp"], errors="coerce") >= pd.Timestamp(start_from)].copy()
        if df_filtered.empty:
            return None

    # 算每根 K 線前 20 根均量（給量過濾用）
    if "volume" in df.columns:
        full_vol_avg20 = df["volume"].rolling(window=20, min_periods=1).mean()
    else:
        full_vol_avg20 = None

    # 對每根 K 線逐一檢查、追蹤連續確認
    consecutive_hit_count = 0
    last_hit_cond: Optional[dict] = None
    last_hit_bar: Optional[pd.Series] = None

    for idx, bar in df_filtered.iterrows():
        prev_avg = float(full_vol_avg20.iloc[idx - 1]) if (full_vol_avg20 is not None and idx > 0) else None

        bar_hit_any = False
        for cond in conditions:
            if _check_one_condition(bar, cond, prev_avg):
                bar_hit_any = True
                last_hit_cond = cond
                last_hit_bar = bar
                break

        if bar_hit_any:
            consecutive_hit_count += 1
            if consecutive_hit_count >= _CONFIRMATION_BARS:
                # 確認觸發
                trigger_bar_ts = str(last_hit_bar.get("timestamp"))
                ct = last_hit_cond.get("type")
                th = last_hit_cond.get("threshold")
                desc_op = ">" if ct == "price_above" else "<"
                vol_part = (
                    f"+ vol > {last_hit_cond.get('volume_mult')}× avg20"
                    if last_hit_cond.get("volume_mult") is not None
                    else ""
                )
                trigger_desc = f"price {desc_op} {th} {vol_part}".strip()
                return {
                    "trigger_bar_ts": trigger_bar_ts,
                    "trigger_desc": trigger_desc,
                    "matched_condition": last_hit_cond,
                }
        else:
            consecutive_hit_count = 0  # 中斷連續

    return None


def _mark_invalidated(pred_id: int, trigger_desc: str, trigger_bar_ts: str) -> bool:
    """把預測標為 invalidated。回傳 True 代表成功。"""
    try:
        prediction_tracker._ensure_db()
        with prediction_tracker._lock:
            prediction_tracker._conn.execute(
                """UPDATE predictions
                   SET status = 'invalidated',
                       invalidated_at = ?,
                       invalidation_trigger = ?
                   WHERE id = ? AND status = 'active'""",
                (taipei_now().isoformat(), f"{trigger_desc} @ {trigger_bar_ts}", pred_id),
            )
            prediction_tracker._conn.commit()
        return True
    except Exception as e:
        logger.warning(f"watcher 標記 #{pred_id} invalidated 失敗：{e}")
        return False


def sweep_symbol(symbol: str, timeframe: Optional[str] = None) -> dict:
    """對單一 symbol（可選 tf）的 active 預測跑 sweep，檢查失效條件。

    Args:
        symbol: 例 "BTC/USDT"
        timeframe: 例 "4h"。若 None，掃該 symbol 的所有 tf

    Returns:
        {
            "symbol": str,
            "timeframe": str|None,
            "checked": int,                 # 實際檢查的預測數
            "invalidated": int,             # 標記 invalidated 的數量
            "shadow_only": bool,            # True = shadow mode，未實際改 status
            "details": [{"id": int, "desc": str, "ts": str}, ...],
        }
    """
    result: dict = {
        "symbol": symbol,
        "timeframe": timeframe,
        "checked": 0,
        "invalidated": 0,
        "shadow_only": bool(getattr(settings, "imitation_shadow_mode", True)),
        "details": [],
    }

    try:
        actives = prediction_tracker.get_active(symbol=symbol)
    except Exception as e:
        logger.warning(f"sweep_symbol 取 active 失敗：{e}")
        return result

    if not actives:
        return result

    # 過濾 timeframe（若指定）
    if timeframe:
        actives = [p for p in actives if str(p.get("timeframe")) == str(timeframe)]
    if not actives:
        return result

    # 載入該 symbol 的 K 線（按 tf 分組載）
    from app.data.fetchers.crypto_engine import crypto_engine
    df_cache: dict[str, pd.DataFrame] = {}

    for pred in actives:
        # 跳過無結構化條件的（純文字 fallback，watcher 不處理）
        if not pred.get("invalidation_json"):
            continue

        tf = pred.get("timeframe", "4h")
        if tf not in df_cache:
            try:
                df = crypto_engine.load_local_data(symbol, tf)
                if df is None or df.empty:
                    continue
                df_cache[tf] = df.reset_index(drop=True)
            except Exception as e:
                logger.debug(f"sweep load {symbol} {tf} 失敗：{e}")
                continue
        df = df_cache[tf]

        result["checked"] += 1
        triggered = _check_prediction_against_df(pred, df, last_sweep_ts=None)
        if not triggered:
            continue

        # 觸發了
        pid = pred.get("id")
        desc = triggered["trigger_desc"]
        ts = triggered["trigger_bar_ts"]

        if result["shadow_only"]:
            # shadow mode：只記錄到 log + 統計，不改 predictions 狀態
            logger.info(
                f"[watcher SHADOW] 預測 #{pid} {symbol} {tf} would be invalidated: {desc} @ {ts}"
            )
            result["invalidated"] += 1  # 計入「應該觸發」的數量
            result["details"].append({"id": pid, "desc": desc, "ts": ts, "shadow": True})
        else:
            # 實際標記
            ok = _mark_invalidated(pid, desc, ts)
            if ok:
                result["invalidated"] += 1
                result["details"].append({"id": pid, "desc": desc, "ts": ts, "shadow": False})
                logger.info(
                    f"[watcher] 預測 #{pid} {symbol} {tf} marked invalidated: {desc} @ {ts}"
                )

    if result["checked"] > 0:
        logger.info(
            f"[watcher] sweep {symbol} {timeframe or 'all-tf'}: "
            f"checked={result['checked']} invalidated={result['invalidated']} "
            f"shadow={result['shadow_only']}"
        )
    return result


def startup_sweep() -> dict:
    """啟動時冷啟動 sweep — 對所有 active 預測跑一輪。

    使用既有本地資料（不強制重抓），快速處理「離線期間是否有失效」。
    """
    summary: dict = {"total_symbols": 0, "total_checked": 0, "total_invalidated": 0, "shadow_only": True}
    try:
        actives = prediction_tracker.get_active(symbol=None)
    except Exception as e:
        logger.warning(f"startup_sweep 取 active 失敗：{e}")
        return summary

    if not actives:
        logger.info("[watcher startup_sweep] 無 active 預測")
        return summary

    summary["shadow_only"] = bool(getattr(settings, "imitation_shadow_mode", True))
    symbols = sorted({p.get("symbol") for p in actives if p.get("symbol")})
    summary["total_symbols"] = len(symbols)

    for sym in symbols:
        r = sweep_symbol(sym, timeframe=None)
        summary["total_checked"] += r.get("checked", 0)
        summary["total_invalidated"] += r.get("invalidated", 0)

    logger.info(
        f"[watcher startup_sweep] symbols={summary['total_symbols']} "
        f"checked={summary['total_checked']} invalidated={summary['total_invalidated']} "
        f"shadow={summary['shadow_only']}"
    )
    return summary
