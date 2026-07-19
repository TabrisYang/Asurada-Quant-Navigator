"""chat 路由的純文字處理層（v157 拆分：從 chat.py 機械搬移，邏輯零改動）。

職責：LLM 輸出文字的解析與偵測 — 誤輸出的 JSON function call 救援、
段落編號偵測、文字中提及但未經 function call 的指標偵測。
全部是無副作用的純函式，可獨立測試。
"""

from __future__ import annotations

import json
import re

from loguru import logger


# 指標名稱到 indicator_id 的映射（用於自動偵測 LLM 文字中提到的指標）
_INDICATOR_TEXT_MAP: dict[str, tuple[str, str]] = {
    # (文字關鍵字, (indicator_id, display_mode))
    "rsi": ("rsi", "sub_chart"),
    "相對強弱": ("rsi", "sub_chart"),
    "macd": ("macd", "sub_chart"),
    "布林": ("bb", "overlay"),
    "bollinger": ("bb", "overlay"),
    "sma": ("sma", "overlay"),
    "簡單移動平均": ("sma", "overlay"),
    "ema": ("ema", "overlay"),
    "指數移動平均": ("ema", "overlay"),
    "adx": ("adx", "sub_chart"),
    "平均趨向": ("adx", "sub_chart"),
    "atr": ("atr", "sub_chart"),
    "真實波動": ("atr", "sub_chart"),
    "obv": ("obv", "sub_chart"),
    "能量潮": ("obv", "sub_chart"),
    "rel_vol": ("rel_vol", "sub_chart"),
    "相對量能": ("rel_vol", "sub_chart"),
    "relative volume": ("rel_vol", "sub_chart"),
    "stochrsi": ("stochrsi", "sub_chart"),
    "隨機相對強弱": ("stochrsi", "sub_chart"),
    "supertrend": ("supertrend", "overlay"),
    "超級趨勢": ("supertrend", "overlay"),
    "ichimoku": ("ichimoku", "overlay"),
    "一目均衡": ("ichimoku", "overlay"),
    "vwap": ("vwap", "overlay"),
    "psar": ("psar", "overlay"),
    "拋物線": ("psar", "overlay"),
    "donchian": ("donchian", "overlay"),
    "唐奇安": ("donchian", "overlay"),
    "keltner": ("keltner", "overlay"),
    "肯特納": ("keltner", "overlay"),
    "roc": ("roc", "sub_chart"),
    "變動率": ("roc", "sub_chart"),
    "bias": ("bias", "sub_chart"),
    "乖離率": ("bias", "sub_chart"),
    "vol_switch": ("vol_switch", "sub_chart"),
    "量價背離": ("vol_switch", "sub_chart"),
    "trailing_stop": ("trailing_stop", "overlay"),
    "追蹤止損": ("trailing_stop", "overlay"),
    "kelly": ("kelly", "sub_chart"),
    "凱利": ("kelly", "sub_chart"),
    "max_drawdown": ("max_drawdown", "sub_chart"),
    "最大回撤": ("max_drawdown", "sub_chart"),
    "fear_greed": ("fear_greed", "sub_chart"),
    "恐懼貪婪": ("fear_greed", "sub_chart"),
    "funding": ("funding", "sub_chart"),
    "資金費率": ("funding", "sub_chart"),
    "market_structure": ("market_structure", "overlay"),
    "市場結構": ("market_structure", "overlay"),
    "harmonic": ("harmonic", "overlay"),
    "諧波": ("harmonic", "overlay"),
    "cvd": ("cvd", "sub_chart"),
    "累計量差": ("cvd", "sub_chart"),
    "poc": ("poc", "overlay"),
    "成交量密集": ("poc", "overlay"),
    "hv": ("hv", "sub_chart"),
    "歷史波動率": ("hv", "sub_chart"),
    "vol_squeeze": ("vol_squeeze", "sub_chart"),
    "波動壓縮": ("vol_squeeze", "sub_chart"),
    "rsi_divergence": ("rsi_divergence", "sub_chart"),
    "rsi背離": ("rsi_divergence", "sub_chart"),
    "rsi 背離": ("rsi_divergence", "sub_chart"),
    "macd_divergence": ("macd_divergence", "sub_chart"),
    "macd背離": ("macd_divergence", "sub_chart"),
    "macd 背離": ("macd_divergence", "sub_chart"),
    "vol_divergence": ("vol_divergence", "sub_chart"),
    "成交量背離": ("vol_divergence", "sub_chart"),
    "leading_composite": ("leading_composite", "sub_chart"),
    "綜合先行": ("leading_composite", "sub_chart"),
    "先行訊號": ("leading_composite", "sub_chart"),
    "mtf_mss": ("mtf_mss", "sub_chart"),
    "mss": ("mtf_mss", "sub_chart"),
    "結構轉變": ("mtf_mss", "sub_chart"),
    "多時間框架": ("mtf_mss", "sub_chart"),
}


# ─── JSON function call 過濾（方案 A 安全網）───────────

_KNOWN_FC_KEYS = {"group_name", "annotations", "annotation_type", "pattern_name", "points"}

# fenced code blocks: ```json ... ``` or ``` ... ```
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```")
# inline JSON with known FC keys
_INLINE_JSON_RE = re.compile(
    r'(\{[^{}]*"(?:group_name|annotation_type|pattern_name|annotations)"[\s\S]*?\}(?:\s*\})?)'
)


def _extract_json_function_calls(text: str) -> tuple[str, list[dict]]:
    """偵測並提取 LLM 文字中誤輸出的 JSON function call。

    Returns:
        (cleaned_text, extracted_function_calls)
        cleaned_text: 移除 JSON 區塊後的文字
        extracted_function_calls: 可執行的 function call 列表
    """
    if not text:
        return text, []

    extracted: list[dict] = []
    spans_to_remove: list[tuple[int, int]] = []

    # 先嘗試 fenced code blocks，再嘗試 inline JSON
    all_matches = list(_FENCED_JSON_RE.finditer(text)) + list(_INLINE_JSON_RE.finditer(text))
    # 按起始位置排序，並去除重疊
    all_matches.sort(key=lambda m: m.start())
    seen_spans: set[tuple[int, int]] = set()
    for match in all_matches:
        span = (match.start(), match.end())
        if any(s[0] <= span[0] < s[1] for s in seen_spans):
            continue
        seen_spans.add(span)
        raw = match.group(1)
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # 嘗試修復常見的 JSON 問題（尾逗號等）
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        if not isinstance(obj, dict):
            continue

        # 判斷是否為已知的 function call 格式
        fc = _try_parse_as_function_call(obj)
        if fc:
            extracted.append(fc)
            spans_to_remove.append((match.start(), match.end()))

    if not spans_to_remove:
        return text, []

    # 從後往前移除，避免 offset 錯位
    cleaned = text
    for start, end in reversed(spans_to_remove):
        cleaned = cleaned[:start] + cleaned[end:]

    # 清理多餘空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    logger.info(f"從 LLM 文字中提取 {len(extracted)} 個 JSON function call 並移除原始 JSON")
    return cleaned, extracted


def _try_parse_as_function_call(obj: dict) -> dict | None:
    """嘗試將 JSON 物件解析為可執行的 function call。"""
    # annotate_chart 格式
    if "group_name" in obj and ("annotations" in obj or "annotation_type" in obj):
        return {"name": "annotate_chart", "arguments": obj}

    # draw_pattern 格式
    if "pattern_name" in obj and "points" in obj:
        return {"name": "draw_pattern", "arguments": obj}

    # manage_indicator 格式
    if "action" in obj and "indicator_id" in obj:
        return {"name": "manage_indicator", "arguments": obj}

    return None


_V138_SEGMENT_PATTERNS = [
    # # 1 / ## 1 / ### 1 / #### 1（含 .5 副段）
    re.compile(r'(?:^|\n)\s*#{1,4}\s*(\d+(?:\.5)?)\b'),
    # ## 第一部分 / ## 第二、 / ### 三、 / ### 第三項（「第」可選）
    re.compile(r'(?:^|\n)\s*#{1,4}\s*第?\s*([一二三四五六七八九十])\s*[、.部項段章]'),
    # **1.** / **1)** / 1. / 1) — 行首加粗或裸數字編號
    re.compile(r'(?:^|\n)\s*(?:\*{1,2})?\s*(\d+)[.\)、]\s*'),
    # 段落 #1 / 段落 1 / 段落-1
    re.compile(r'(?:^|\n)\s*段落\s*[#-]?\s*(\d+)\b'),
]

_CN_TO_NUM = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}


def _detect_segments_v138(text: str) -> set[str]:
    r"""v138：用多 pattern 偵測段落編號，支援中英文混合多種結構。

    取代原本 `r'(?:^|\n)\s*#\s*(\d+(?:\.5)?)\b'` 單一 pattern（誤判率高）。
    支援：# N / ## N / ## 第一 / **1. / 段落 #N 等常見格式。
    """
    if not text:
        return set()
    found: set[str] = set()
    for pat in _V138_SEGMENT_PATTERNS:
        for m in pat.finditer(text):
            num = m.group(1)
            if num in _CN_TO_NUM:
                num = _CN_TO_NUM[num]
            found.add(num)
    return found


def _detect_mentioned_indicators(text: str, existing_ids: set[str]) -> list[dict]:
    """從 LLM 文字中偵測提到但未透過 function call 添加的指標"""
    if not text:
        return []
    text_lower = text.lower()
    detected: dict[str, str] = {}  # indicator_id → display_mode
    for keyword, (ind_id, display_mode) in _INDICATOR_TEXT_MAP.items():
        if keyword in text_lower and ind_id not in existing_ids and ind_id not in detected:
            detected[ind_id] = display_mode
    return [
        {"action": "add", "indicator_id": ind_id, "display_mode": dm}
        for ind_id, dm in detected.items()
    ]
