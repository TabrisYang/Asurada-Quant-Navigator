"""標註治理工具 — LLM 畫線的白名單 / 吸附 / 聚合 / 上限。

從 executor.py 抽出（repo health 行數護欄）。這裡集中處理「畫不準」與
「太多線」兩類問題：
- _exec_annotate / _exec_draw_pattern：把 LLM function call 轉為 annotation dict
- _snap_annotations：價格/時間吸附到實際 K 線（修 LLM 生成數字不精準）
- _dedupe_close_annotations：ATR 動態閾值聚合相近水平線
- _cap_horizontal_lines：水平線硬上限（prompt 口頭約束的程式強制版）
"""

from typing import Optional

import pandas as pd
from loguru import logger

from app.core.indicators import registry
from app.core.llm.data_access import _load_local_data

_ANNOTATE_ALLOWED_TYPES = {"horizontal_line", "text_label"}


def _get_current_price_from_chart_state(chart_state: Optional[dict]) -> Optional[float]:
    """v111 helper：從 chart_state 取最新 close（給 _dedupe_close_annotations 用）。"""
    if not chart_state:
        return None
    df = chart_state.get("_cached_df")
    if df is None or getattr(df, "empty", True):
        return None
    try:
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def _calc_dynamic_threshold(
    chart_state: Optional[dict], fallback: float = 0.01,
) -> float:
    """v111：依標的近期波動性（ATR-14）動態決定 horizontal_line 聚合閾值。

    threshold = ATR / current_price × 0.5（半個 ATR 內視為同一支撐區），
    再 clamp 到 [0.005, 0.025]（0.5%~2.5%）防極端值。

    為什麼用 ATR：
        固定 1% 對 BTC/ETH（4h ATR%≈1-3%）合理，但對極大波動標的（小幣
        ATR%>5%）太緊、極小波動標的（穩定幣 ATR%<0.1%）太鬆。ATR 反映標的
        近期慣性，動態 threshold 比寫死值更穩健。

    範例：
        - BTC 4h ATR%=2% → threshold=1%（與舊寫死值相當）
        - SHIB 4h ATR%=8% → threshold=2.5%（被 clamp 到上限）
        - USDT 穩定幣 ATR%=0.05% → threshold=0.5%（被 clamp 到下限）

    fallback 觸發條件：
        - chart_state 無 _cached_df（如 round 2 已精簡）
        - 數據不足（< 14 根 K 線）
        - 計算過程任何例外
    """
    if not chart_state:
        return fallback
    df = chart_state.get("_cached_df")
    if df is None or getattr(df, "empty", True) or len(df) < 14:
        return fallback
    try:
        # ATR-14（Wilder True Range 的 14 期 SMA — 簡化版，足以做 threshold 估算）
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = tr1.combine(tr2, max).combine(tr3, max)
        atr = float(tr.tail(14).mean())
        current_price = float(close.iloc[-1])
        if current_price <= 0 or atr <= 0:
            return fallback
        atr_pct = atr / current_price
        # 半個 ATR 內視為同一支撐區，clamp 防極端
        threshold = atr_pct * 0.5
        return max(0.005, min(0.025, threshold))
    except Exception:
        return fallback


def _dedupe_close_annotations(
    anns: list[dict],
    current_price: Optional[float],
    threshold_pct: Optional[float] = None,
) -> list[dict]:
    """v111：聚合相近價位的 horizontal_line，避免主圖上 5 個近似支撐擠成一團。

    規則：
    - 只處理 type=horizontal_line；text_label 不動
    - 按 price 排序，相鄰價位差 / current_price < threshold_pct 合併
    - 合併後 price=均值；text=「支撐區 X-Y（指標 A + B + C）」
    - 若 current_price 不可用（chart_state 沒 _cached_df）→ 直接回原 list 不動
    - threshold_pct=None 時走 fallback 1%（呼叫端應傳入 _calc_dynamic_threshold(chart_state)）

    範例：
        輸入：[
          {price: 2429, text: "壓力 DC上緣"},
          {price: 2325, text: "BB 中軌"},
          {price: 2309, text: "EMA20 支撐"},
          {price: 2223, text: "Supertrend 支撐"},
          {price: 2203, text: "SMC Fib 0.5 支撐"}
        ]
        當前價 2,800、threshold 1% = ~28：
        2325/2309 (差 16 < 28) → 合併
        2223/2203 (差 20 < 28) → 合併
        其餘獨立
        輸出：3 條 — DC上緣 / 支撐區 2309-2325 / 支撐區 2203-2223
    """
    if threshold_pct is None:
        threshold_pct = 0.01  # fallback：呼叫端沒傳就用 1%
    if not anns or current_price is None or current_price <= 0:
        return anns

    # 分離 horizontal_line 與其他類型
    horizontal_lines: list[dict] = []
    others: list[dict] = []
    for a in anns:
        if a.get("type") == "horizontal_line" and a.get("price") is not None:
            horizontal_lines.append(a)
        else:
            others.append(a)

    if len(horizontal_lines) <= 1:
        return anns  # 無需合併

    # 按 price 排序
    horizontal_lines.sort(key=lambda x: x.get("price", 0))

    threshold_abs = float(current_price) * threshold_pct

    # 分組：相鄰差 < threshold 合併
    groups: list[list[dict]] = []
    current_group: list[dict] = [horizontal_lines[0]]
    for ann in horizontal_lines[1:]:
        prev_price = current_group[-1].get("price", 0)
        if abs(ann.get("price", 0) - prev_price) < threshold_abs:
            current_group.append(ann)
        else:
            groups.append(current_group)
            current_group = [ann]
    groups.append(current_group)

    # 每組合併為 1 個 annotation
    merged: list[dict] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        prices = [g.get("price", 0) for g in group]
        texts = [str(g.get("text", "") or "").strip() for g in group if g.get("text")]
        # 簡化每個 text：移除價格數字（保留指標名稱）
        import re as _re
        clean_texts = []
        for t in texts:
            # 移除常見的 ": 1234" "= 1234" "（1234）" "$1234" 這類 — 留指標名
            cleaned = _re.sub(r"[:：=]?\s*\$?\s*\d+[\d,.\s]*$", "", t).strip()
            cleaned = _re.sub(r"[（(]\s*\d+[\d,.\s]*[)）]\s*$", "", cleaned).strip()
            if cleaned:
                clean_texts.append(cleaned)

        # 合併 representative annotation：用第一個的 color / lineWidth / groupId 等
        rep = dict(group[0])
        rep["price"] = sum(prices) / len(prices)
        # 文字格式：「[類別] X-Y｜A + B + C」
        price_min = min(prices)
        price_max = max(prices)
        merged_text = f"{price_min:.0f}-{price_max:.0f}"
        if clean_texts:
            merged_text += "｜" + " + ".join(clean_texts[:4])  # 最多顯示 4 個避免過長
            if len(clean_texts) > 4:
                merged_text += f" +{len(clean_texts) - 4}"
        rep["text"] = merged_text
        merged.append(rep)

    # 與其他類型合併回傳
    return merged + others


def _snap_annotations(anns: list[dict], chart_state: Optional[dict]) -> list[dict]:
    """把 LLM 生成的標註價格/時間吸附到實際 K 線資料。

    LLM 的 price/time 是自行生成的數字，常見兩類誤差：
    - 時間不落在實際 bar timestamp 上 → lightweight-charts 會把 trend_line 端點
      /marker 對齊到最近的 bar，造成視覺漂移
    - 價格記錯/四捨五入 → 水平線偏離真實支撐壓力

    處理：
    - startTime/endTime 吸附到最近的實際 bar timestamp（格式與 K 線資料完全一致）
    - price/endPrice 在動態閾值內（ATR 半閾值，約 0.25%~1.25%）吸附到最近的
      swing high/low 或最新 close；超出閾值視為刻意指定的指標價位（如 BB 中軌），不動
    - horizontal_line 價格超出全部資料範圍 ±10% → 視為幻覺，直接丟棄
    - chart_state 無 _cached_df（如 round 2 已精簡）→ 原樣返回
    """
    if not anns or not chart_state:
        return anns
    df = chart_state.get("_cached_df")
    if df is None or getattr(df, "empty", True) or "timestamp" not in df.columns:
        return anns

    import numpy as np

    try:
        ts = pd.to_datetime(df["timestamp"]).values.astype("datetime64[ns]").astype("int64")
        ts_strings = df["timestamp"].astype(str).tolist()
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        current_price = float(df["close"].iloc[-1])

        # swing high/low（前後各 2 根的 fractal）+ 最新 close 作為吸附候選價位
        window = 5
        pivot_high = highs[highs == highs.rolling(window, center=True).max()]
        pivot_low = lows[lows == lows.rolling(window, center=True).min()]
        levels = np.unique(
            np.concatenate([pivot_high.values, pivot_low.values, [current_price]])
        )
        snap_pct = _calc_dynamic_threshold(chart_state) * 0.5
        data_min = float(lows.min())
        data_max = float(highs.max())
        margin = (data_max - data_min) * 0.10
    except Exception:
        return anns

    def _snap_time(t):
        if not t:
            return t
        try:
            # 用 Timestamp.value（恆為奈秒）而非 to_datetime64()——
            # pandas 2.x 後者可能回傳微秒精度，與 ns 陣列比對會差 1000 倍
            target = int(pd.to_datetime(str(t)).value)
        except Exception:
            return t
        i = int(np.searchsorted(ts, target))
        if i <= 0:
            return ts_strings[0]
        if i >= len(ts):
            return ts_strings[-1]
        # 取前後兩根中較近者
        if abs(ts[i] - target) < abs(target - ts[i - 1]):
            return ts_strings[i]
        return ts_strings[i - 1]

    def _snap_price(p):
        if p is None or not levels.size or current_price <= 0:
            return p
        try:
            p = float(p)
        except (TypeError, ValueError):
            return p
        i = int(np.searchsorted(levels, p))
        candidates = levels[max(0, i - 1): i + 1]
        if not candidates.size:
            return p
        nearest = float(candidates[np.argmin(np.abs(candidates - p))])
        if abs(nearest - p) / current_price <= snap_pct:
            return nearest
        return p

    snapped: list[dict] = []
    for a in anns:
        price = a.get("price")
        if a.get("type") == "horizontal_line" and price is not None:
            try:
                if not (data_min - margin <= float(price) <= data_max + margin):
                    logger.warning(
                        f"標註價格 {price} 遠超資料範圍 [{data_min:.4g}, {data_max:.4g}]，已丟棄（疑似 LLM 幻覺）"
                    )
                    continue
            except (TypeError, ValueError):
                continue
        out = dict(a)
        out["startTime"] = _snap_time(a.get("startTime"))
        out["endTime"] = _snap_time(a.get("endTime"))
        out["price"] = _snap_price(a.get("price"))
        out["endPrice"] = _snap_price(a.get("endPrice"))
        snapped.append(out)
    return snapped


def _cap_horizontal_lines(
    anns: list[dict], current_price: Optional[float], max_lines: int = 8,
) -> list[dict]:
    """程式強制的水平線上限（prompt 的「最多 5 條」只是口頭要求，此處硬性攔截）。

    超過 max_lines 時保留離當前價最近的幾條 — 離價格越近的支撐壓力對決策越有意義。
    """
    lines: list[dict] = []
    others: list[dict] = []
    for a in anns:
        if a.get("type") == "horizontal_line" and a.get("price") is not None:
            lines.append(a)
        else:
            others.append(a)
    if len(lines) <= max_lines:
        return anns
    if current_price and current_price > 0:
        lines.sort(key=lambda a: abs(float(a["price"]) - current_price))
    dropped = len(lines) - max_lines
    logger.warning(f"水平線超過上限 {max_lines} 條，已丟棄離當前價最遠的 {dropped} 條")
    return lines[:max_lines] + others


def _exec_annotate(args: dict) -> list[dict]:
    """執行 annotate_chart — 支援批量繪圖，回傳 annotation 列表。
    白名單過濾：只允許 horizontal_line 和 text_label，
    trend_line / highlight_range / vertical_line 一律丟棄。
    """
    import uuid
    group_id = str(uuid.uuid4())[:8]
    group_name = args.get("group_name", "AI 標記")

    def _build_one(a: dict) -> dict | None:
        ann_type = a.get("annotation_type", "horizontal_line")
        if ann_type not in _ANNOTATE_ALLOWED_TYPES:
            return None
        return {
            "type": ann_type,
            "startTime": a.get("start_time"),
            "endTime": a.get("end_time"),
            "price": a.get("price"),
            "endPrice": a.get("end_price"),
            "text": a.get("text"),
            "color": a.get("color", "#58a6ff"),
            "lineWidth": a.get("line_width", 2),
            "lineStyle": a.get("line_style", 0),
            "groupId": group_id,
            "groupName": group_name,
        }

    batch = args.get("annotations")
    if batch and isinstance(batch, list):
        return [r for item in batch if (r := _build_one(item)) is not None]

    result = _build_one(args)
    return [result] if result is not None else []


def _exec_draw_pattern(args: dict) -> list[dict]:
    """draw_pattern — 根據關鍵點自動連線和標注，回傳 annotation 列表"""
    import uuid
    pattern_name = args.get("pattern_name", "Pattern")
    points = args.get("points", [])
    color = args.get("color", "#f0b90b")
    line_width = args.get("line_width", 2)
    bullish = args.get("bullish")

    if bullish is True:
        color = args.get("color", "#26a69a")
    elif bullish is False:
        color = args.get("color", "#ef5350")

    group_id = str(uuid.uuid4())[:8]
    group_name = f"{pattern_name}"
    annotations: list[dict] = []

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        annotations.append({
            "type": "trend_line",
            "startTime": p1.get("time"),
            "endTime": p2.get("time"),
            "price": p1.get("price"),
            "endPrice": p2.get("price"),
            "color": color,
            "lineWidth": line_width,
            "lineStyle": 0,
            "groupId": group_id,
            "groupName": group_name,
        })

    for pt in points:
        label = pt.get("label", "")
        annotations.append({
            "type": "text_label",
            "startTime": pt.get("time"),
            "price": pt.get("price"),
            "text": label,
            "color": color,
            "groupId": group_id,
            "groupName": group_name,
        })

    return annotations


async def _exec_find_conditions(args: dict, default_symbol: str, default_tf: str) -> dict:
    """執行 find_conditions"""
    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    conditions = args.get("conditions", [])
    start = args.get("start_date")
    end = args.get("end_date")

    df = _load_local_data(symbol, timeframe, start, end)
    if df.empty:
        return {"matched_periods": [], "summary": "找不到數據", "annotations": []}

    import pandas as pd
    condition_masks = []
    for cond in conditions:
        indicator_id = cond.get("indicator", "").lower()
        calc_result = registry.calculate(indicator_id, df, cond.get("parameters"))
        if not calc_result:
            continue

        series_name = list(calc_result.keys())[0]
        values = pd.Series(calc_result[series_name])
        op = cond.get("operator", ">")
        val = cond.get("value", 0)

        if op == ">":
            mask = values > val
        elif op == "<":
            mask = values < val
        elif op == ">=":
            mask = values >= val
        elif op == "<=":
            mask = values <= val
        elif op == "==":
            mask = values == val
        elif op == "cross_above":
            mask = (values > val) & (values.shift(1) <= val)
        elif op == "cross_below":
            mask = (values < val) & (values.shift(1) >= val)
        elif op == "between":
            mask = (values >= val) & (values <= cond.get("value2", val))
        else:
            continue
        condition_masks.append(mask)

    if not condition_masks:
        return {"matched_periods": [], "summary": "無有效條件", "annotations": []}

    logical = args.get("logical_operator", "AND")
    combined = condition_masks[0]
    for m in condition_masks[1:]:
        combined = combined & m if logical == "AND" else combined | m

    matched = df[combined]
    annotations = []
    if not matched.empty:
        timestamps = matched["timestamp"].tolist()
        for ts in timestamps:
            annotations.append({
                "type": "vertical_line",
                "time": str(ts),
                "color": "#f85149",
            })

    return {
        "matched_periods": len(matched),
        "summary": f"找到 {len(matched)} 個匹配點",
        "annotations": annotations,
    }
