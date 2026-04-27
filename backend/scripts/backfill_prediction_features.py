"""阿斯拉量化系統 — 回填 v101 prediction_features（Phase 2.0c）

對既有 verified predictions 從 OHLCV 重算 39 個特徵，補進 prediction_features 表。

執行：
  cd backend && .venv/bin/python3 scripts/backfill_prediction_features.py

特性：
  - 容忍部分樣本失敗（資料不足、symbol 對應不到 OHLCV → 跳過）
  - 已存在的 prediction_features row 會被 INSERT OR REPLACE 覆蓋
  - 純讀取既有 OHLCV，不下載新數據
  - 不改 v100 既有預測表（純加法保證）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 讓腳本能 import app 模組
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from datetime import datetime
from loguru import logger

from app.core.prediction_tracker import prediction_tracker
from app.core.feature_extractor import record_features
from app.data.fetchers.crypto_engine import crypto_engine


def main():
    print("═" * 60)
    print("  阿斯拉量化系統 — v101 prediction_features 回填")
    print(f"  時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    prediction_tracker._ensure_db()

    # 1. 找所有已驗證的預測
    rows = prediction_tracker._conn.execute(
        """SELECT id, symbol, timeframe, direction, entry_price, target_price,
                  stop_price, timeframe_hours, confidence, regime, created_at
           FROM predictions
           WHERE status IN ('hit_target', 'hit_stop', 'expired', 'active')
           ORDER BY id ASC"""
    ).fetchall()

    if not rows:
        print("⚠ predictions 表無資料")
        return

    print(f"📊 共 {len(rows)} 筆預測待處理")

    # 2. 找已有 features 的 ID（避免重複處理）
    existing = {
        r[0] for r in prediction_tracker._conn.execute(
            "SELECT prediction_id FROM prediction_features"
        ).fetchall()
    }
    print(f"   其中 {len(existing)} 筆已有 features（會覆蓋）")

    # 3. 逐筆處理
    success = 0
    no_data = 0
    failed = 0

    for row in rows:
        pred_id = row[0]
        pred = {
            "symbol": row[1],
            "direction": row[3],
            "entry_price": row[4],
            "target_price": row[5],
            "stop_price": row[6],
            "timeframe_hours": row[7],
            "confidence": row[8],
            "regime": row[9],
        }
        timeframe = row[2]
        created_at = row[10]

        # 載入 OHLCV — 只取預測時點以前的資料（防止 look-ahead bias）
        try:
            df = crypto_engine.load_local_data(pred["symbol"], timeframe)
            if df is None or df.empty:
                no_data += 1
                continue

            # 切到 created_at 之前
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)
                df = df[df["timestamp"] <= created_at].copy()

            if len(df) < 30:
                no_data += 1
                continue

            ok = record_features(pred_id, df, chart_state=None,
                               prediction=pred, source="backfill")
            if ok:
                success += 1
            else:
                failed += 1

        except Exception as e:
            logger.warning(f"prediction_id={pred_id} 失敗: {e}")
            failed += 1

    print()
    print(f"✅ 回填完成：成功 {success} ｜ 無資料 {no_data} ｜ 失敗 {failed}")

    # 4. 驗證
    total_features = prediction_tracker._conn.execute(
        "SELECT COUNT(*) FROM prediction_features"
    ).fetchone()[0]
    print(f"   prediction_features 表現有 {total_features} 筆")


if __name__ == "__main__":
    main()
