"""阿斯拉量化系統 — v105 C1：regime 字串標準化 mapping。

predictions.db 的 199 筆樣本 regime 是 v100 早期自由文字（80+ 種變體），
無法直接用於 v103+ 的 per-regime 子模型。這裡建 mapping 表把所有 regime
字串映射到 6 種標準 label：
  trending_up / trending_down / ranging / high_vol / low_vol / unknown

策略：先用關鍵字優先序匹配，再 fallback unknown。
"""

from __future__ import annotations

from typing import Optional

# 關鍵字優先序：從最具體到最籠統
# 上面命中就 return，避免「趨勢上行_低波盤整」被「盤整」搶走
_RULES: list[tuple[str, list[str]]] = [
    # high_vol（最具體：高波動相關）
    ("high_vol", [
        "高波動", "高 vol", "high_vol", "強趨勢高波動", "高波動結構",
        "高波動回調", "高波動盤整", "高波動展開",
    ]),
    # low_vol（低波動）
    ("low_vol", [
        "低波動", "低波盤整", "low_vol", "低波", "波動壓縮", "波動收斂",
        "盤整壓縮", "盤整/波動壓縮", "盤整+波動壓縮",
    ]),
    # trending_up（趨勢向上 / 多頭結構）
    ("trending_up", [
        "趨勢上行", "趨勢上升", "上升趨勢", "uptrend", "trending_up",
        "強趨勢上行", "趨勢上行_突破", "趨勢上行_高波動", "盤整→趨勢初期",
        "盤整轉趨勢", "趨勢形成中", "初期趨勢", "趨勢初期", "趨勢啟動",
        "趨勢擴張", "突破上行", "強勢上漲",
    ]),
    # trending_down（趨勢向下 / 空頭結構）
    ("trending_down", [
        "趨勢下行", "趨勢下降", "下降趨勢", "downtrend", "trending_down",
        "強趨勢下行", "空頭趨勢", "下行趨勢", "趨勢下行初期", "趨勢下行_逆勢",
        "趨勢下行_超賣反彈", "趨勢下行_動能衰減", "趨勢轉弱_空頭結構",
        "盤整轉空", "弱趨勢偏空", "趨勢衰退_空頭",
    ]),
    # ranging（盤整、震盪、區間）
    ("ranging", [
        "盤整", "震盪", "ranging", "consolidation", "區間",
        "盤整偏空", "盤整偏多", "盤整偏弱", "弱趨勢偏空盤整",
        "盤整轉折", "盤整過渡", "趨勢轉盤整", "趨勢衰竭轉盤整",
        "下行趨勢轉盤整", "趨勢衰減", "趨勢衰退", "趨勢衰退期",
        "趨勢衰退過渡期", "趨勢轉折", "趨勢→盤整過渡", "結構轉折期",
        "均值回歸", "均值回歸_支撐帶", "超賣反彈", "急漲回撤",
        "修正", "回調", "趨勢回調", "過渡期", "趨勢邊界態",
        "趨勢成熟期", "趨勢末段", "趨勢減速", "趨勢反轉",
        "時框衝突_均值回歸",
    ]),
    # 未知（剩下的）— 留 unknown
]


def standardize_regime(raw: Optional[str]) -> str:
    """把任意 regime 字串標準化到 6 種 label。

    匹配順序：high_vol → low_vol → trending_up/down → ranging → unknown。
    用「關鍵字 substring 匹配」避免完全字串比對失敗（regime 字串常含底線後綴）。
    """
    if not raw:
        return "unknown"
    s = str(raw).lower().strip()

    # 直接命中標準 label
    if s in {"trending_up", "trending_down", "ranging", "high_vol", "low_vol", "unknown"}:
        return s

    for std_label, keywords in _RULES:
        for kw in keywords:
            if kw.lower() in s:
                return std_label

    return "unknown"


