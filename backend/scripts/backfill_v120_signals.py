"""v120.4：對既有 predictions 回填 fear_greed + funding_rate（兩個有完整 free history 的訊號）。

Why：v120.5 get_signal_combo_stats 需要歷史 buckets 才能算命中率。245 筆既有
prediction 沒 capture 訊號，加上新欄位後是 NULL → 沒歷史命中率可用。

可回填的訊號（API 有完整歷史 + 免費）：
- fear_greed: alternative.me /fng/?limit=N（一次抓全部）
- funding_rate: Binance fapi/v1/fundingRate?symbol=X&startTime=N（每 8h 一筆）

無法回填（API 30 天限制 / 付費 / 已擋）：
- open_interest 24h 變化  → Binance 30 天限制
- long_short_ratio        → Binance 30 天限制
- coinbase_premium        → 要 Coinbase historical klines × 245 次 API call，工時大效益小
- etf_flow                → 公開 API 全擋（v119.5 跳過）
- ob_imbalance            → snapshot only，無 historical

跑一次即可（idempotent — 重跑會 overwrite 但結果不變）。

Usage:
    cd backend && .venv/bin/python3 scripts/backfill_v120_signals.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def _fetch_all_fear_greed(client: httpx.Client) -> dict[str, int]:
    """alternative.me /fng/?limit=2000 一次抓全部（支援限制 max=2000）。

    Returns {"YYYY-MM-DD": value}
    """
    out: dict[str, int] = {}
    try:
        r = client.get("https://api.alternative.me/fng/", params={"limit": 2000}, timeout=15.0)
        r.raise_for_status()
        data = r.json().get("data") or []
        for item in data:
            ts = int(item.get("timestamp") or 0)
            if ts > 0:
                date_key = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                out[date_key] = int(item.get("value") or 50)
    except Exception as e:
        print(f"  ✗ fear_greed 全歷史抓取失敗: {e}")
    return out


def _fetch_funding_rate_for(client: httpx.Client, sym: str, target_ts_ms: int) -> float | None:
    """抓最接近 target_ts 的 funding rate（往前 8h 內找）。"""
    try:
        # 抓 target_ts 前後各 8h 範圍的 funding（Binance 8h 一期）
        start = target_ts_ms - 8 * 3600 * 1000
        end = target_ts_ms + 4 * 3600 * 1000
        r = client.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": sym, "startTime": start, "endTime": end, "limit": 5},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json() or []
        if not data:
            return None
        # 找最接近 target_ts 的
        best = min(data, key=lambda d: abs(int(d["fundingTime"]) - target_ts_ms))
        return float(best.get("fundingRate") or 0)
    except Exception:
        return None


def main():
    from app.core.prediction_tracker import prediction_tracker
    from app.core.signal_buckets import classify_funding, classify_fear_greed

    prediction_tracker._ensure_db()
    conn = prediction_tracker._conn

    rows = conn.execute(
        "SELECT id, symbol, created_at, fear_greed_at_entry, funding_at_entry, "
        "buckets_json, signals_json "
        "FROM predictions ORDER BY id"
    ).fetchall()
    total = len(rows)
    print(f"總 predictions: {total}")

    if total == 0:
        print("沒資料可回填")
        return

    with httpx.Client() as client:
        # 1. 抓全部 fear_greed 歷史（一次 API call）
        print("抓 fear_greed 全歷史...")
        fg_history = _fetch_all_fear_greed(client)
        print(f"  fear_greed 歷史 {len(fg_history)} 天")

        # 2. 對每筆 prediction 回填
        updated_fg = 0
        updated_fr = 0
        skipped_fg = 0
        skipped_fr = 0
        for i, r in enumerate(rows, 1):
            pid = r["id"]
            symbol = r["symbol"]
            created_at = r["created_at"]
            try:
                created_ts_ms = int(datetime.fromisoformat(created_at).timestamp() * 1000)
            except Exception:
                skipped_fg += 1
                skipped_fr += 1
                continue

            # 解析現有 buckets_json（保留其他訊號 bucket）
            try:
                buckets = json.loads(r["buckets_json"]) if r["buckets_json"] else {}
            except Exception:
                buckets = {}
            try:
                signals = json.loads(r["signals_json"]) if r["signals_json"] else {
                    "derivatives": {}, "sentiment": {}
                }
            except Exception:
                signals = {"derivatives": {}, "sentiment": {}}

            # ─── fear_greed 回填 ─────────
            new_fg = None
            if r["fear_greed_at_entry"] is None:
                date_key = created_at[:10]  # YYYY-MM-DD
                new_fg = fg_history.get(date_key)
                # 找不到當天的 → 找最近 3 天
                if new_fg is None:
                    for delta in range(1, 4):
                        for sign in (-1, 1):
                            try:
                                d = datetime.fromisoformat(created_at) + __import__("datetime").timedelta(days=delta * sign)
                                k = d.strftime("%Y-%m-%d")
                                if k in fg_history:
                                    new_fg = fg_history[k]
                                    break
                            except Exception:
                                pass
                        if new_fg is not None:
                            break

            if new_fg is not None:
                buckets["fear_greed"] = classify_fear_greed(new_fg)
                signals.setdefault("sentiment", {})["fear_greed_value"] = new_fg
                updated_fg += 1
            else:
                skipped_fg += 1

            # ─── funding_rate 回填（只對含 USDT 的 symbol）─────
            new_fr = None
            if r["funding_at_entry"] is None and ("USDT" in symbol.upper() or "USD" in symbol.upper()):
                bn_sym = _to_binance_symbol(symbol)
                rate = _fetch_funding_rate_for(client, bn_sym, created_ts_ms)
                if rate is not None:
                    new_fr = round(rate * 100, 4)  # to %
                    buckets["funding"] = classify_funding(new_fr)
                    signals.setdefault("derivatives", {})["funding_rate_pct"] = new_fr
                    updated_fr += 1
                else:
                    skipped_fr += 1
                # 避免 rate limit
                time.sleep(0.05)
            else:
                skipped_fr += 1

            # 寫回 DB
            if new_fg is not None or new_fr is not None:
                conn.execute(
                    "UPDATE predictions SET "
                    "fear_greed_at_entry=COALESCE(?, fear_greed_at_entry), "
                    "funding_at_entry=COALESCE(?, funding_at_entry), "
                    "buckets_json=?, signals_json=? "
                    "WHERE id=?",
                    (new_fg, new_fr, json.dumps(buckets, ensure_ascii=False),
                     json.dumps(signals, ensure_ascii=False), pid),
                )

            if i % 20 == 0:
                conn.commit()
                print(f"  進度 {i}/{total} | fg+{updated_fg} fr+{updated_fr}")

        conn.commit()
        print()
        print(f"完成：fear_greed 回填 {updated_fg}/{total}（skip {skipped_fg}），"
              f"funding 回填 {updated_fr}/{total}（skip {skipped_fr}）")


if __name__ == "__main__":
    main()
