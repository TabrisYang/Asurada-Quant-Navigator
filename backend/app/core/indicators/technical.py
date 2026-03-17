"""阿斯拉量化系統 — A 類技術指標計算（18 項 OHLCV 指標）

所有指標都從 OHLCV 數據即時計算，不需要外部 API。
每個指標用 @registry.register 自動註冊。
"""

import numpy as np
import pandas as pd

from app.core.indicators.registry import registry


# =============================================
# 輔助函式
# =============================================

def _ema(series: pd.Series, period: int) -> pd.Series:
    """指數移動平均"""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """簡單移動平均"""
    return series.rolling(window=period).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """真實波幅"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _safe_list(series: pd.Series) -> list:
    """將 Series 轉為可 JSON 序列化的 list"""
    return [None if pd.isna(v) else round(float(v), 6) for v in series]


def _vectorized_divergence(price: pd.Series, indicator: pd.Series, lookback: int) -> pd.Series:
    """向量化背離偵測（共用於 RSI 背離、MACD 背離等）。

    底背離(+1)：價格在近期低位，但指標值高於近期最低值（動能未跟隨下探）。
    頂背離(-1)：價格在近期高位，但指標值低於近期最高值（動能未跟隨上衝）。
    """
    roll_price_min = price.rolling(window=lookback, min_periods=lookback).min()
    roll_price_max = price.rolling(window=lookback, min_periods=lookback).max()
    roll_ind_min = indicator.rolling(window=lookback, min_periods=lookback).min()
    roll_ind_max = indicator.rolling(window=lookback, min_periods=lookback).max()

    price_range = roll_price_max - roll_price_min
    safe_range = price_range.replace(0, np.nan)

    # 價格在近期低位（低於 15 分位）
    price_low_pctile = (price - roll_price_min) / safe_range
    # 價格在近期高位（高於 85 分位）
    price_high_pctile = price_low_pctile

    # 指標沒有跟隨創新低/新高（5% 容差）
    ind_above_min = indicator > roll_ind_min * 1.05
    ind_below_max = indicator < roll_ind_max * 0.95

    valid = indicator.notna() & roll_ind_min.notna()

    divergence = pd.Series(0.0, index=price.index)
    divergence[valid & (price_low_pctile <= 0.15) & ind_above_min] = 1.0
    divergence[valid & (price_high_pctile >= 0.85) & ind_below_max] = -1.0

    return divergence


# =============================================
# 1. SMA 移動平均
# =============================================
@registry.register(
    id="sma",
    name="SMA 簡單移動平均",
    category="動能與趨勢",
    description="判斷市場長短期的平均成本方向",
    parameters={
        "period": {"default": 20, "min": 5, "max": 200, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: p.get("period", 20),
    pro_tip="均線黃金交叉常有延遲，建議搭配斜率使用",
)
def calc_sma(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    sma = _sma(df["close"], period)
    return {f"SMA({period})": _safe_list(sma)}


# =============================================
# 2. EMA 指數移動平均
# =============================================
@registry.register(
    id="ema",
    name="EMA 指數移動平均",
    category="動能與趨勢",
    description="對近期價格賦予更高權重的移動平均",
    parameters={
        "period": {"default": 20, "min": 5, "max": 200, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: p.get("period", 20),
    pro_tip="EMA 比 SMA 更靈敏，適合追蹤快速趨勢",
)
def calc_ema(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    ema = _ema(df["close"], period)
    return {f"EMA({period})": _safe_list(ema)}


# =============================================
# 3. ADX 趨勢強度
# =============================================
@registry.register(
    id="adx",
    name="ADX 趨勢強度",
    category="動能與趨勢",
    description="衡量趨勢的強度，而非方向。ADX > 25 代表趨勢啟動",
    parameters={
        "period": {"default": 14, "min": 5, "max": 50, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 14) * 2,
    pro_tip="避免在 ADX 低時使用趨勢策略，否則會被雙向洗盤",
)
def calc_adx(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # 當 +DM < -DM 時 +DM = 0，反之亦然
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)

    tr = _true_range(df)
    atr = _ema(tr, period)

    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * _ema(plus_dm, period) / atr_safe
    minus_di = 100 * _ema(minus_dm, period) / atr_safe

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = _ema(dx, period)

    return {
        "ADX": _safe_list(adx),
        "+DI": _safe_list(plus_di),
        "-DI": _safe_list(minus_di),
    }


# =============================================
# 4. RSI 相對強弱指標
# =============================================
@registry.register(
    id="rsi",
    name="RSI 相對強弱指標",
    category="均值回歸",
    description="衡量價格漲跌速度，判斷超買超賣。0-100 範圍",
    parameters={
        "period": {"default": 14, "min": 2, "max": 100, "type": "int"},
        "overbought": {"default": 70, "min": 50, "max": 95, "type": "float"},
        "oversold": {"default": 30, "min": 5, "max": 50, "type": "float"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 14),
    pro_tip="加密貨幣波動大，RSI 常會鈍化（在 80 以上待很久）",
)
def calc_rsi(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss 全為 0 → rs=inf → RSI 應為 100（純漲）；avg_gain 全為 0 → RSI 應為 0（純跌）
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(100.0)

    return {"RSI": _safe_list(rsi)}


# =============================================
# 5. 乖離率 (Bias)
# =============================================
@registry.register(
    id="bias",
    name="乖離率",
    category="均值回歸",
    description="衡量價格偏離移動平均線的程度（百分比）",
    parameters={
        "period": {"default": 20, "min": 5, "max": 120, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 20),
    pro_tip="用來抓 BTC 的短線暴跌反彈（抄底）非常有效",
)
def calc_bias(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    ma = _sma(df["close"], period)
    bias = ((df["close"] - ma) / ma.replace(0, np.nan) * 100)
    return {"Bias": _safe_list(bias)}


# =============================================
# 6. 布林帶 (Bollinger Bands)
# =============================================
@registry.register(
    id="bb",
    name="布林帶",
    category="波動率",
    description="根據標準差定義價格的波動範圍",
    parameters={
        "period": {"default": 20, "min": 5, "max": 200, "type": "int"},
        "std_dev": {"default": 2.0, "min": 0.5, "max": 4.0, "type": "float"},
    },
    display_mode="overlay",
    warmup_func=lambda p: p.get("period", 20),
    pro_tip="Squeeze（擠壓）狀態是量化交易員最愛的爆發訊號",
)
def calc_bollinger(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    std_dev = float(params["std_dev"])

    middle = _sma(df["close"], period)
    std = df["close"].rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = ((upper - lower) / middle.replace(0, np.nan) * 100)

    return {
        "BB_Upper": _safe_list(upper),
        "BB_Middle": _safe_list(middle),
        "BB_Lower": _safe_list(lower),
        "BB_Width": _safe_list(bandwidth),
    }


# =============================================
# 7. ATR 真實波幅
# =============================================
@registry.register(
    id="atr",
    name="ATR 真實波幅",
    category="波動率",
    description="衡量市場波動的絕對水平",
    parameters={
        "period": {"default": 14, "min": 5, "max": 50, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 14),
    pro_tip="主要用於設定止損（如止損設在 2 倍 ATR 處）",
)
def calc_atr(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    tr = _true_range(df)
    atr = _ema(tr, period)
    return {"ATR": _safe_list(atr)}


# =============================================
# 8. 爆量突破 (Relative Volume)
# =============================================
@registry.register(
    id="rel_vol",
    name="爆量突破",
    category="成交量",
    description="驗證價格變動是否有真實資金支持",
    parameters={
        "period": {"default": 20, "min": 5, "max": 60, "type": "int"},
        "threshold": {"default": 2.0, "min": 1.5, "max": 5.0, "type": "float"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 20),
    pro_tip="價格突破但沒爆量，通常是假突破（誘多/誘空）",
)
def calc_relative_volume(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    avg_vol = _sma(df["volume"], period)
    rel_vol = df["volume"] / avg_vol.replace(0, np.nan)
    return {"RelVol": _safe_list(rel_vol)}


# =============================================
# 9. OBV 能量潮
# =============================================
@registry.register(
    id="obv",
    name="OBV 能量潮",
    category="成交量",
    description="將成交量數量化，觀察資金流向",
    parameters={},
    display_mode="sub_chart",
    warmup_func=lambda p: 1,
    pro_tip="OBV 先創新高而價格沒動，通常代表即將補漲",
)
def calc_obv(df: pd.DataFrame, params: dict) -> dict[str, list]:
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    return {"OBV": _safe_list(obv)}


# =============================================
# 10. MACD
# =============================================
@registry.register(
    id="macd",
    name="MACD",
    category="動能",
    description="指數平滑異同移動平均線，含柱狀體斜率分析",
    parameters={
        "fast": {"default": 12, "min": 5, "max": 30, "type": "int"},
        "slow": {"default": 26, "min": 15, "max": 60, "type": "int"},
        "signal": {"default": 9, "min": 5, "max": 20, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("slow", 26) + p.get("signal", 9),
    pro_tip="觀察柱狀體是否由負轉正且連續三根放大",
)
def calc_macd(df: pd.DataFrame, params: dict) -> dict[str, list]:
    fast = int(params["fast"])
    slow = int(params["slow"])
    signal = int(params["signal"])

    ema_fast = _ema(df["close"], fast)
    ema_slow = _ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        "MACD": _safe_list(macd_line),
        "Signal": _safe_list(signal_line),
        "Histogram": _safe_list(histogram),
    }


# =============================================
# 11. ROC 變動率
# =============================================
@registry.register(
    id="roc",
    name="ROC 變動率",
    category="動能",
    description="衡量價格在 N 天內的純粹漲跌幅",
    parameters={
        "period": {"default": 14, "min": 5, "max": 60, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 14),
    pro_tip="尋找 ROC 曲線斜率陡峭上升的區段",
)
def calc_roc(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    shifted = df["close"].shift(period).replace(0, np.nan)
    roc = ((df["close"] / shifted) - 1) * 100
    return {"ROC": _safe_list(roc)}


# =============================================
# 12. 唐奇安通道 (Donchian Channel)
# =============================================
@registry.register(
    id="donchian",
    name="唐奇安通道",
    category="動能",
    description="經典趨勢突破指標，過去 N 天的最高/最低價",
    parameters={
        "period": {"default": 20, "min": 10, "max": 55, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: p.get("period", 20),
    pro_tip="突破上軌買入，跌破下軌賣出（海龜交易法則）",
)
def calc_donchian(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params["period"])
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    middle = (upper + lower) / 2
    return {
        "DC_Upper": _safe_list(upper),
        "DC_Middle": _safe_list(middle),
        "DC_Lower": _safe_list(lower),
    }


# =============================================
# 13. Keltner Channels
# =============================================
@registry.register(
    id="keltner",
    name="Keltner 通道",
    category="波動率",
    description="結合均線與 ATR，比布林帶更貼合趨勢",
    parameters={
        "ema_period": {"default": 20, "min": 10, "max": 50, "type": "int"},
        "atr_multiplier": {"default": 2.0, "min": 1.0, "max": 3.0, "type": "float"},
        "atr_period": {"default": 14, "min": 5, "max": 50, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: max(p.get("ema_period", 20), p.get("atr_period", 14)),
    pro_tip="突破通道通常是真突破，比布林帶更貼合趨勢",
)
def calc_keltner(df: pd.DataFrame, params: dict) -> dict[str, list]:
    ema_period = int(params["ema_period"])
    atr_mult = float(params["atr_multiplier"])
    atr_period = int(params["atr_period"])

    middle = _ema(df["close"], ema_period)
    tr = _true_range(df)
    atr = _ema(tr, atr_period)

    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr

    return {
        "KC_Upper": _safe_list(upper),
        "KC_Middle": _safe_list(middle),
        "KC_Lower": _safe_list(lower),
    }


# =============================================
# 14. 波動性切換 (Volatility Switch)
# =============================================
@registry.register(
    id="vol_switch",
    name="波動性切換",
    category="波動率",
    description="區分市場處於低波動蓄勢還是高波動釋放",
    parameters={
        "short_period": {"default": 10, "min": 5, "max": 30, "type": "int"},
        "long_period": {"default": 50, "min": 20, "max": 200, "type": "int"},
        "threshold": {"default": 0.5, "min": 0.1, "max": 1.0, "type": "float"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("long_period", 50),
    pro_tip="比率低於閾值代表波動性噴發即將來臨",
)
def calc_vol_switch(df: pd.DataFrame, params: dict) -> dict[str, list]:
    short = int(params["short_period"])
    long = int(params["long_period"])

    std_short = df["close"].rolling(window=short).std()
    std_long = df["close"].rolling(window=long).std()
    ratio = std_short / std_long.replace(0, np.nan)

    return {"VolRatio": _safe_list(ratio)}


# =============================================
# 15. 追蹤止損 (Trailing Stop)
# =============================================
@registry.register(
    id="trailing_stop",
    name="追蹤止損",
    category="風險控管",
    description="隨價格上漲自動上移的止損點",
    parameters={
        "atr_multiplier": {"default": 3.0, "min": 1.0, "max": 5.0, "type": "float"},
        "atr_period": {"default": 14, "min": 5, "max": 50, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: p.get("atr_period", 14),
    pro_tip="能保住獲利，讓利潤在趨勢中奔跑",
)
def calc_trailing_stop(df: pd.DataFrame, params: dict) -> dict[str, list]:
    atr_mult = float(params["atr_multiplier"])
    atr_period = int(params["atr_period"])

    tr = _true_range(df)
    atr = _ema(tr, atr_period)

    # Chandelier Exit 方式
    long_stop = df["high"].cummax() - atr_mult * atr
    short_stop = df["low"].cummin() + atr_mult * atr

    return {
        "Long_Stop": _safe_list(long_stop),
        "Short_Stop": _safe_list(short_stop),
    }


# =============================================
# 16. 日內時段效應
# =============================================
@registry.register(
    id="session",
    name="日內時段效應",
    category="時間週期",
    description="標記亞洲/歐洲/美國交易時段",
    parameters={},
    display_mode="overlay",
    warmup_func=lambda p: 0,
    pro_tip="BTC 常在美股開盤前後半小時出現當日最大波動",
)
def calc_session(df: pd.DataFrame, params: dict) -> dict[str, list]:
    sessions = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        hour = ts.hour
        if 0 <= hour < 8:
            sessions.append("asia")
        elif 8 <= hour < 16:
            sessions.append("europe")
        else:
            sessions.append("us")
    return {"Session": sessions}


# =============================================
# 17. 凱利公式
# =============================================
@registry.register(
    id="kelly",
    name="凱利公式",
    category="部位管理",
    description="根據勝率與盈虧比計算最優投注比例",
    parameters={
        "win_rate": {"default": 0.55, "min": 0.1, "max": 0.9, "type": "float"},
        "payoff_ratio": {"default": 2.0, "min": 0.5, "max": 10.0, "type": "float"},
        "fraction": {"default": 0.5, "min": 0.1, "max": 1.0, "type": "float"},
    },
    display_mode="info_panel",
    warmup_func=lambda p: 0,
    pro_tip="實戰通常使用半凱利或更保守比例",
)
def calc_kelly(df: pd.DataFrame, params: dict) -> dict[str, list]:
    p = float(params["win_rate"])
    b = float(params["payoff_ratio"])
    frac = float(params["fraction"])
    q = 1 - p

    full_kelly = (b * p - q) / b
    adjusted_kelly = full_kelly * frac

    return {
        "Full_Kelly": [round(full_kelly, 4)] * len(df),
        "Adjusted_Kelly": [round(max(0, adjusted_kelly), 4)] * len(df),
    }


# =============================================
# 18. 最大回撤 (Max Drawdown)
# =============================================
@registry.register(
    id="max_drawdown",
    name="最大回撤",
    category="風險控管",
    description="從最高點回落到最低點的最大幅度",
    parameters={},
    display_mode="sub_chart",
    warmup_func=lambda p: 0,
    pro_tip="一旦 MDD 超過預期，代表市場環境已改變",
)
def calc_max_drawdown(df: pd.DataFrame, params: dict) -> dict[str, list]:
    cummax = df["close"].cummax()
    drawdown = (df["close"] - cummax) / cummax.replace(0, np.nan) * 100
    return {"Drawdown": _safe_list(drawdown)}


# =============================================
# 19. 恐懼貪婪指數 (Fear & Greed Index)
# =============================================
@registry.register(
    id="fear_greed",
    name="恐懼貪婪指數",
    category="市場情緒",
    description="綜合市場情緒的 0-100 指數",
    parameters={},
    display_mode="sub_chart",
    data_source="external",
    warmup_func=lambda p: 0,
    pro_tip="極端恐懼時買入，極端貪婪時賣出",
)
def calc_fear_greed(df: pd.DataFrame, params: dict) -> dict[str, list]:
    """
    Fear & Greed 指標 — 需要外部數據。
    此函式在 data_source='external' 時由路由層特殊處理。
    如果直接呼叫（fallback），回傳空值。
    """
    return {"Fear_Greed": [None] * len(df)}


# =============================================
# 20. 資金費率 (Funding Rate)
# =============================================
@registry.register(
    id="funding",
    name="資金費率",
    category="市場情緒",
    description="永續合約多空力量的均衡費率",
    parameters={
        "alert_threshold": {"default": 0.05, "min": 0.01, "max": 0.2, "type": "float"},
    },
    display_mode="sub_chart",
    data_source="external",
    warmup_func=lambda p: 0,
    pro_tip="費率極端時反向操作勝率較高",
)
def calc_funding(df: pd.DataFrame, params: dict) -> dict[str, list]:
    """
    Funding Rate 指標 — 需要外部數據。
    此函式在 data_source='external' 時由路由層特殊處理。
    如果直接呼叫（fallback），回傳空值。
    """
    return {"Funding_Rate": [None] * len(df)}


# =============================================
# 21. Stochastic RSI (StochRSI)
# =============================================
@registry.register(
    id="stochrsi",
    name="StochRSI 隨機相對強弱",
    category="均值回歸",
    description="在 RSI 上再套用隨機指標，比 RSI 更敏感的超買超賣判斷",
    parameters={
        "rsi_period": {"default": 14, "min": 5, "max": 50, "type": "int"},
        "stoch_period": {"default": 14, "min": 5, "max": 50, "type": "int"},
        "k_smooth": {"default": 3, "min": 1, "max": 10, "type": "int"},
        "d_smooth": {"default": 3, "min": 1, "max": 10, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: int(p.get("rsi_period", 14)) + int(p.get("stoch_period", 14)) + 10,
    pro_tip="StochRSI 在 0.2 以下是超賣、0.8 以上是超買，比 RSI 反應更快",
)
def calc_stochrsi(df: pd.DataFrame, params: dict) -> dict[str, list]:
    close = df["close"]
    rsi_period = int(params["rsi_period"])
    stoch_period = int(params["stoch_period"])
    k_smooth = int(params["k_smooth"])
    d_smooth = int(params["d_smooth"])

    # 計算 RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 在 RSI 上計算 Stochastic
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
    stoch_rsi = (rsi - rsi_min) / rsi_range

    # %K 和 %D
    k_line = stoch_rsi.rolling(window=k_smooth).mean()
    d_line = k_line.rolling(window=d_smooth).mean()

    return {
        "%K": _safe_list(k_line),
        "%D": _safe_list(d_line),
    }


# =============================================
# 22. VWAP (成交量加權平均價格)
# =============================================
@registry.register(
    id="vwap",
    name="VWAP 成交量加權均價",
    category="動能與趨勢",
    description="以成交量為權重的平均價格，機構交易者的核心基準線",
    parameters={
        "anchor": {"default": 20, "min": 1, "max": 200, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: int(p.get("anchor", 20)),
    pro_tip="價格在 VWAP 上方代表多頭佔優，下方代表空頭佔優",
)
def calc_vwap(df: pd.DataFrame, params: dict) -> dict[str, list]:
    anchor = int(params["anchor"])
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]

    # 滾動 VWAP（使用 anchor 週期內累積）
    tp_vol = typical_price * volume
    cum_tp_vol = tp_vol.rolling(window=anchor, min_periods=1).sum()
    cum_vol = volume.rolling(window=anchor, min_periods=1).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    return {"VWAP": _safe_list(vwap)}


# =============================================
# 23. Ichimoku Cloud (一目均衡表)
# =============================================
@registry.register(
    id="ichimoku",
    name="Ichimoku 一目均衡表",
    category="動能與趨勢",
    description="日本經典多功能趨勢系統，同時提供支撐/阻力/趨勢方向",
    parameters={
        "tenkan": {"default": 9, "min": 5, "max": 30, "type": "int"},
        "kijun": {"default": 26, "min": 10, "max": 60, "type": "int"},
        "senkou_b": {"default": 52, "min": 20, "max": 120, "type": "int"},
    },
    display_mode="overlay",
    warmup_func=lambda p: int(p.get("senkou_b", 52)) + int(p.get("kijun", 26)),
    pro_tip="價格在雲層上方做多、下方做空；雲層厚度代表支撐/阻力強度",
)
def calc_ichimoku(df: pd.DataFrame, params: dict) -> dict[str, list]:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan_p = int(params["tenkan"])
    kijun_p = int(params["kijun"])
    senkou_b_p = int(params["senkou_b"])

    # 轉換線 (Tenkan-sen)
    tenkan_sen = (high.rolling(window=tenkan_p).max() + low.rolling(window=tenkan_p).min()) / 2

    # 基準線 (Kijun-sen)
    kijun_sen = (high.rolling(window=kijun_p).max() + low.rolling(window=kijun_p).min()) / 2

    # 先行帶 A (Senkou Span A) — 移動 kijun_p 期
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_p)

    # 先行帶 B (Senkou Span B) — 移動 kijun_p 期
    senkou_b = ((high.rolling(window=senkou_b_p).max() + low.rolling(window=senkou_b_p).min()) / 2).shift(kijun_p)

    # 遲行帶 (Chikou Span) — 收盤價往回移
    chikou = close.shift(-kijun_p)

    return {
        "Tenkan": _safe_list(tenkan_sen),
        "Kijun": _safe_list(kijun_sen),
        "Senkou_A": _safe_list(senkou_a),
        "Senkou_B": _safe_list(senkou_b),
        "Chikou": _safe_list(chikou),
    }


# =============================================
# 24. Parabolic SAR (拋物線止損轉向)
# =============================================
@registry.register(
    id="psar",
    name="Parabolic SAR 拋物線轉向",
    category="動能與趨勢",
    description="隨趨勢自動調整止損點，反轉時發出明確的進出場訊號",
    parameters={
        "af_start": {"default": 0.02, "min": 0.005, "max": 0.1, "type": "float"},
        "af_max": {"default": 0.2, "min": 0.05, "max": 0.5, "type": "float"},
    },
    display_mode="overlay",
    warmup_func=lambda p: 5,
    pro_tip="SAR 點從下方翻到上方 = 空頭訊號，從上方翻到下方 = 多頭訊號",
)
def calc_psar(df: pd.DataFrame, params: dict) -> dict[str, list]:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    af_start = float(params["af_start"])
    af_max = float(params["af_max"])

    psar = np.full(n, np.nan)
    af = af_start
    is_long = True

    if n < 2:
        return {"PSAR": [None] * n}

    # 初始化
    psar[0] = low[0]
    ep = high[0]  # Extreme Point

    for i in range(1, n):
        prev_psar = psar[i - 1]

        if is_long:
            psar[i] = prev_psar + af * (ep - prev_psar)
            # SAR 不能高於前兩根 K 線的低點
            psar[i] = min(psar[i], low[i - 1])
            if i >= 2:
                psar[i] = min(psar[i], low[i - 2])

            if low[i] < psar[i]:
                # 反轉為空頭
                is_long = False
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            # SAR 不能低於前兩根 K 線的高點
            psar[i] = max(psar[i], high[i - 1])
            if i >= 2:
                psar[i] = max(psar[i], high[i - 2])

            if high[i] > psar[i]:
                # 反轉為多頭
                is_long = True
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)

    return {"PSAR": _safe_list(pd.Series(psar))}


# =============================================
# 25. Supertrend (超級趨勢)
# =============================================
@registry.register(
    id="supertrend",
    name="Supertrend 超級趨勢",
    category="動能與趨勢",
    description="基於 ATR 的趨勢追蹤指標，提供明確的多空方向",
    parameters={
        "period": {"default": 10, "min": 5, "max": 50, "type": "int"},
        "multiplier": {"default": 3.0, "min": 1.0, "max": 6.0, "type": "float"},
    },
    display_mode="overlay",
    warmup_func=lambda p: int(p.get("period", 10)) + 5,
    pro_tip="綠色（多頭）上穿時買入，紅色（空頭）下穿時賣出；非常適合趨勢交易",
)
def calc_supertrend(df: pd.DataFrame, params: dict) -> dict[str, list]:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    period = int(params["period"])
    multiplier = float(params["multiplier"])
    n = len(close)

    # ATR 計算
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    hl2 = (high + low) / 2

    # 基本上下軌
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = pd.Series(np.nan, index=df.index)
    lower_band = pd.Series(np.nan, index=df.index)
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 = 多頭, -1 = 空頭

    for i in range(period, n):
        # Upper band
        if pd.isna(upper_basic.iloc[i]):
            upper_band.iloc[i] = np.nan
        elif pd.isna(upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = (
                min(upper_basic.iloc[i], upper_band.iloc[i - 1])
                if close.iloc[i - 1] <= upper_band.iloc[i - 1]
                else upper_basic.iloc[i]
            )

        # Lower band
        if pd.isna(lower_basic.iloc[i]):
            lower_band.iloc[i] = np.nan
        elif pd.isna(lower_band.iloc[i - 1]):
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = (
                max(lower_basic.iloc[i], lower_band.iloc[i - 1])
                if close.iloc[i - 1] >= lower_band.iloc[i - 1]
                else lower_basic.iloc[i]
            )

        # Direction & Supertrend
        if i == period:
            direction.iloc[i] = 1 if close.iloc[i] > upper_band.iloc[i] else -1
        else:
            prev_dir = direction.iloc[i - 1]
            if prev_dir == -1 and close.iloc[i] > upper_band.iloc[i]:
                direction.iloc[i] = 1
            elif prev_dir == 1 and close.iloc[i] < lower_band.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = prev_dir

        supertrend.iloc[i] = (
            lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        )

    return {"Supertrend": _safe_list(supertrend)}


# =============================================
# Market Structure（市場結構 — HH/HL/LH/LL）
# =============================================
@registry.register(
    id="market_structure",
    name="Market Structure 市場結構",
    category="動能與趨勢",
    description="辨識 Swing High/Low 並標記 HH（更高高點）、HL（更高低點）、LH（更低高點）、LL（更低低點），畫出結構線以判斷趨勢方向。BOS = 結構突破，CHoCH = 趨勢轉變。",
    parameters={
        "swing_length": {"default": 5, "min": 2, "max": 20, "type": "int", "step": 1},
    },
    display_mode="overlay",
    warmup_func=lambda p: p.get("swing_length", 5) * 2 + 1,
    pro_tip="swing_length 越大識別的結構越宏觀；搭配 RSI 背離使用效果更佳",
)
def calc_market_structure(df: pd.DataFrame, params: dict) -> dict[str, list]:
    n = len(df)
    length = int(params["swing_length"])
    highs = df["high"].values
    lows = df["low"].values

    # 1) 辨識 Swing High / Swing Low
    swing_high_idx: list[int] = []
    swing_low_idx: list[int] = []

    for i in range(length, n - length):
        is_sh = True
        is_sl = True
        for j in range(1, length + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_sh = False
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_sl = False
            if not is_sh and not is_sl:
                break
        if is_sh:
            swing_high_idx.append(i)
        if is_sl:
            swing_low_idx.append(i)

    # 2) 分類 HH/LH 和 HL/LL
    sh_values: list[float | None] = [None] * n
    sl_values: list[float | None] = [None] * n
    hh_values: list[float | None] = [None] * n
    lh_values: list[float | None] = [None] * n
    hl_values: list[float | None] = [None] * n
    ll_values: list[float | None] = [None] * n

    prev_sh: float | None = None
    for idx in swing_high_idx:
        val = float(highs[idx])
        sh_values[idx] = val
        if prev_sh is not None:
            if val > prev_sh:
                hh_values[idx] = val
            else:
                lh_values[idx] = val
        prev_sh = val

    prev_sl: float | None = None
    for idx in swing_low_idx:
        val = float(lows[idx])
        sl_values[idx] = val
        if prev_sl is not None:
            if val > prev_sl:
                hl_values[idx] = val
            else:
                ll_values[idx] = val
        prev_sl = val

    return {
        "Swing_High": _safe_list(pd.Series(sh_values)),
        "Swing_Low": _safe_list(pd.Series(sl_values)),
        "HH": _safe_list(pd.Series(hh_values)),
        "LH": _safe_list(pd.Series(lh_values)),
        "HL": _safe_list(pd.Series(hl_values)),
        "LL": _safe_list(pd.Series(ll_values)),
    }


# =============================================
# 諧波型態偵測（Harmonic Patterns）
# =============================================

import math as _math

# Fibonacci 比例容差 ±tolerance
_HARMONIC_PATTERNS = {
    "Gartley": {
        "AB_XA": (0.618, 0.618),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.272, 1.618),
        "AD_XA": (0.786, 0.786),
    },
    "Bat": {
        "AB_XA": (0.382, 0.500),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "AD_XA": (0.886, 0.886),
    },
    "Butterfly": {
        "AB_XA": (0.786, 0.786),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "AD_XA": (1.272, 1.618),
    },
    "Crab": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (2.240, 3.618),
        "AD_XA": (1.618, 1.618),
    },
    "Shark": {
        "AB_XA": (0.446, 0.618),
        "BC_AB": (1.130, 1.618),
        "CD_BC": (1.618, 2.236),
        "AD_XA": (0.886, 1.130),
    },
}


def _find_zigzag_pivots(highs, lows, closes, deviation_pct: float = 5.0) -> list[tuple[int, float, str]]:
    """
    Zigzag 偵測：找到價格的主要轉折點。
    Returns: [(idx, price, 'H'|'L'), ...]
    """
    n = len(closes)
    if n < 5:
        return []

    deviation = deviation_pct / 100.0
    pivots: list[tuple[int, float, str]] = []

    last_pivot_type = ""
    last_pivot_price = closes[0]
    last_pivot_idx = 0

    hi_since = float(highs[0])
    hi_idx = 0
    lo_since = float(lows[0])
    lo_idx = 0

    for i in range(1, n):
        h = float(highs[i])
        l = float(lows[i])

        if h > hi_since:
            hi_since = h
            hi_idx = i
        if l < lo_since:
            lo_since = l
            lo_idx = i

        if last_pivot_type != "H":
            if hi_since > 0 and (hi_since - l) / hi_since >= deviation:
                pivots.append((hi_idx, hi_since, "H"))
                last_pivot_type = "H"
                last_pivot_price = hi_since
                last_pivot_idx = hi_idx
                lo_since = l
                lo_idx = i
                hi_since = h
                hi_idx = i

        if last_pivot_type != "L":
            if lo_since > 0 and (h - lo_since) / lo_since >= deviation:
                pivots.append((lo_idx, lo_since, "L"))
                last_pivot_type = "L"
                last_pivot_price = lo_since
                last_pivot_idx = lo_idx
                hi_since = h
                hi_idx = i
                lo_since = l
                lo_idx = i

    return pivots


def _check_ratio(actual: float, expected_range: tuple[float, float], tolerance: float = 0.10) -> bool:
    lo = expected_range[0] * (1 - tolerance)
    hi = expected_range[1] * (1 + tolerance)
    return lo <= actual <= hi


def _detect_harmonics(pivots: list[tuple[int, float, str]], tolerance: float = 0.10) -> list[dict]:
    """從 Zigzag 轉折點中偵測所有可能的諧波型態"""
    results: list[dict] = []
    n = len(pivots)
    if n < 5:
        return results

    for i in range(n - 4):
        x_idx, x_price, x_type = pivots[i]
        a_idx, a_price, a_type = pivots[i + 1]
        b_idx, b_price, b_type = pivots[i + 2]
        c_idx, c_price, c_type = pivots[i + 3]
        d_idx, d_price, d_type = pivots[i + 4]

        xa = abs(a_price - x_price)
        if xa < 1e-8:
            continue
        ab = abs(b_price - a_price)
        bc = abs(c_price - b_price)
        cd = abs(d_price - c_price)
        ad = abs(d_price - a_price)

        ab_xa = ab / xa
        bc_ab = bc / ab if ab > 1e-8 else 999
        cd_bc = cd / bc if bc > 1e-8 else 999
        ad_xa = ad / xa

        for name, ratios in _HARMONIC_PATTERNS.items():
            if (
                _check_ratio(ab_xa, ratios["AB_XA"], tolerance)
                and _check_ratio(bc_ab, ratios["BC_AB"], tolerance)
                and _check_ratio(cd_bc, ratios["CD_BC"], tolerance)
                and _check_ratio(ad_xa, ratios["AD_XA"], tolerance)
            ):
                is_bullish = d_type == "L"
                results.append({
                    "pattern": name,
                    "direction": "bullish" if is_bullish else "bearish",
                    "points": {
                        "X": {"idx": x_idx, "price": x_price},
                        "A": {"idx": a_idx, "price": a_price},
                        "B": {"idx": b_idx, "price": b_price},
                        "C": {"idx": c_idx, "price": c_price},
                        "D": {"idx": d_idx, "price": d_price},
                    },
                    "ratios": {
                        "AB/XA": round(ab_xa, 3),
                        "BC/AB": round(bc_ab, 3),
                        "CD/BC": round(cd_bc, 3),
                        "AD/XA": round(ad_xa, 3),
                    },
                    "score": round(1.0 - (
                        abs(ab_xa - sum(ratios["AB_XA"]) / 2)
                        + abs(ad_xa - sum(ratios["AD_XA"]) / 2)
                    ) / 2, 3),
                })

    # 按分數排序，取最佳結果
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:5]


@registry.register(
    id="harmonic",
    name="Harmonic Patterns 諧波型態",
    category="型態辨識",
    description="自動偵測諧波型態（Gartley、Bat、Butterfly、Crab、Shark），在圖表上標記 X-A-B-C-D 五個轉折點。基於 Zigzag 偵測 + Fibonacci 比例驗證。",
    parameters={
        "zigzag_deviation": {"default": 5.0, "min": 1.0, "max": 15.0, "type": "float", "step": 0.5},
        "tolerance": {"default": 0.10, "min": 0.05, "max": 0.25, "type": "float", "step": 0.01},
    },
    display_mode="overlay",
    warmup_func=lambda p: 30,
    pro_tip="zigzag_deviation 控制最小波動幅度(%)，越小偵測越敏感但誤報越多；tolerance 控制 Fibonacci 比例容差",
)
def calc_harmonic_patterns(df: pd.DataFrame, params: dict) -> dict[str, list]:
    n = len(df)
    deviation = float(params.get("zigzag_deviation", 5.0))
    tolerance = float(params.get("tolerance", 0.10))

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    pivots = _find_zigzag_pivots(highs, lows, closes, deviation)
    patterns = _detect_harmonics(pivots, tolerance)

    # 輸出格式：Zigzag 線 + 各轉折點標記
    zigzag_line: list[float | None] = [None] * n
    for idx, price, _ in pivots:
        zigzag_line[idx] = price

    # 型態標記：每個偵測到的型態的 D 點
    pattern_x: list[float | None] = [None] * n
    pattern_a: list[float | None] = [None] * n
    pattern_b: list[float | None] = [None] * n
    pattern_c: list[float | None] = [None] * n
    pattern_d: list[float | None] = [None] * n
    classification: list[str | None] = [None] * n

    for pat in patterns[:3]:
        pts = pat["points"]
        label = f"{pat['pattern']}({'看漲' if pat['direction'] == 'bullish' else '看跌'})"
        pattern_x[pts["X"]["idx"]] = pts["X"]["price"]
        pattern_a[pts["A"]["idx"]] = pts["A"]["price"]
        pattern_b[pts["B"]["idx"]] = pts["B"]["price"]
        pattern_c[pts["C"]["idx"]] = pts["C"]["price"]
        pattern_d[pts["D"]["idx"]] = pts["D"]["price"]
        classification[pts["D"]["idx"]] = label

    return {
        "Zigzag": _safe_list(pd.Series(zigzag_line)),
        "Pattern_X": _safe_list(pd.Series(pattern_x)),
        "Pattern_A": _safe_list(pd.Series(pattern_a)),
        "Pattern_B": _safe_list(pd.Series(pattern_b)),
        "Pattern_C": _safe_list(pd.Series(pattern_c)),
        "Pattern_D": _safe_list(pd.Series(pattern_d)),
        "Classification": [v for v in classification],
    }


# =============================================
# 28. CVD 累計量差 (Cumulative Volume Delta)
# =============================================
@registry.register(
    id="cvd",
    name="CVD 累計量差",
    category="量能分析",
    description="用 K 線近似累計量差（上漲 K 線量 - 下跌 K 線量）。CVD 上升 + 價格上漲 = 趨勢健康；CVD 背離 = 趨勢可能反轉",
    parameters={
        "period": {"default": 1, "min": 1, "max": 1, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: 2,
    pro_tip="CVD 與價格的背離是強力反轉信號：價格新高但 CVD 未新高 = 賣壓累積",
)
def calc_cvd(df: pd.DataFrame, params: dict) -> dict[str, list]:
    close = df["close"]
    volume = df["volume"]
    delta = close.diff().fillna(0)
    signed_vol = volume.copy()
    signed_vol[delta > 0] = volume[delta > 0]
    signed_vol[delta < 0] = -volume[delta < 0]
    signed_vol[delta == 0] = 0
    cvd = signed_vol.cumsum()
    return {"CVD": _safe_list(cvd)}


# =============================================
# 29. POC 成交量密集區 (Volume Profile POC)
# =============================================
@registry.register(
    id="poc",
    name="POC 成交量密集區",
    category="量能分析",
    description="以 N 個價位區間統計成交量分布，找出最大成交量的價位（POC），以及高量區上下緣（VAH/VAL，覆蓋 70% 成交量）",
    parameters={
        "bins": {"default": 50, "min": 10, "max": 200, "type": "int"},
        "value_area_pct": {"default": 0.7, "min": 0.5, "max": 0.9, "type": "float"},
    },
    display_mode="overlay",
    warmup_func=lambda p: 20,
    pro_tip="POC 是強力支撐/阻力位；價格在 VAH-VAL 之間波動時為盤整，突破則可能啟動趨勢",
)
def calc_poc(df: pd.DataFrame, params: dict) -> dict[str, list]:
    n = len(df)
    bins = int(params.get("bins", 50))
    va_pct = float(params.get("value_area_pct", 0.7))

    close = df["close"].values
    volume = df["volume"].values
    price_min = float(np.nanmin(close))
    price_max = float(np.nanmax(close))

    if price_max - price_min < 1e-8:
        return {"POC": [None] * n, "VAH": [None] * n, "VAL": [None] * n}

    edges = np.linspace(price_min, price_max, bins + 1)
    bin_indices = np.digitize(close, edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    vol_profile = np.zeros(bins)
    for i in range(n):
        vol_profile[bin_indices[i]] += volume[i]

    poc_bin = int(np.argmax(vol_profile))
    poc_price = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0

    total_vol = np.sum(vol_profile)
    target_vol = total_vol * va_pct
    sorted_bins = np.argsort(vol_profile)[::-1]
    cum_vol = 0.0
    va_bins = set()
    for b in sorted_bins:
        va_bins.add(b)
        cum_vol += vol_profile[b]
        if cum_vol >= target_vol:
            break

    va_bin_list = sorted(va_bins)
    vah_price = edges[max(va_bin_list) + 1] if va_bin_list else price_max
    val_price = edges[min(va_bin_list)] if va_bin_list else price_min

    return {
        "POC": [round(poc_price, 6)] * n,
        "VAH": [round(float(vah_price), 6)] * n,
        "VAL": [round(float(val_price), 6)] * n,
    }


# =============================================
# 30. HV 歷史波動率 (Historical Volatility)
# =============================================
@registry.register(
    id="hv",
    name="HV 歷史波動率",
    category="波動率",
    description="以對數收益率的滾動標準差衡量歷史波動率（年化）。可替代隱含波動率 (IV) 評估市場風險",
    parameters={
        "period": {"default": 20, "min": 5, "max": 100, "type": "int"},
        "annualize": {"default": 365, "min": 252, "max": 365, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: int(p.get("period", 20)) + 1,
    pro_tip="HV 高 = 市場波動大，期權賣方風險高；HV 低谷後常伴隨大行情",
)
def calc_hv(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params.get("period", 20))
    annualize = int(params.get("annualize", 365))

    close = df["close"]
    shifted = close.shift(1).replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(close / shifted)
    rolling_std = log_returns.rolling(window=period).std()
    hv = rolling_std * np.sqrt(annualize) * 100

    return {"HV": _safe_list(hv)}


# =============================================
# 價格偽指標（供回測引擎使用，不在 UI 顯示）
# =============================================

@registry.register(
    id="close",
    name="收盤價",
    category="價格",
    description="收盤價原始數值，供回測引擎以固定價位作為條件使用",
    parameters={},
    display_mode="overlay",
    warmup_func=lambda p: 0,
)
def calc_close(df: pd.DataFrame, params: dict) -> dict[str, list]:
    return {"Close": _safe_list(df["close"])}


@registry.register(
    id="high",
    name="最高價",
    category="價格",
    description="最高價原始數值",
    parameters={},
    display_mode="overlay",
    warmup_func=lambda p: 0,
)
def calc_high(df: pd.DataFrame, params: dict) -> dict[str, list]:
    return {"High": _safe_list(df["high"])}


@registry.register(
    id="low",
    name="最低價",
    category="價格",
    description="最低價原始數值",
    parameters={},
    display_mode="overlay",
    warmup_func=lambda p: 0,
)
def calc_low(df: pd.DataFrame, params: dict) -> dict[str, list]:
    return {"Low": _safe_list(df["low"])}


# =============================================
# 先行訊號偵測指標
# =============================================

@registry.register(
    id="vol_squeeze",
    name="波動壓縮",
    category="先行訊號",
    description="布林帶寬度 (BB Width) 相對近 N 期的分位，低值代表波動壓縮，預示即將突破",
    parameters={
        "bb_period": {"default": 20, "min": 10, "max": 50, "type": "int"},
        "lookback": {"default": 120, "min": 40, "max": 300, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: max(p.get("bb_period", 20), p.get("lookback", 120)),
    pro_tip="分位低於 10% 為強壓縮，通常預示大幅波動即將來臨",
)
def calc_vol_squeeze(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params.get("bb_period", 20))
    lookback = int(params.get("lookback", 120))
    close = df["close"]

    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    safe_sma = sma.replace(0, np.nan)
    bb_width = (2 * std / safe_sma) * 100

    # 計算滾動分位：目前 BB Width 在近 lookback 期中的百分位
    squeeze_pctile = bb_width.rolling(window=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        if len(x) >= lookback else np.nan,
        raw=False,
    )

    # 壓縮信號：0=無, 1=壓縮中(分位<20%), 2=強壓縮(分位<10%)
    signal = pd.Series(0.0, index=df.index)
    signal[squeeze_pctile < 20] = 1.0
    signal[squeeze_pctile < 10] = 2.0

    return {
        "BB_Width_Pctile": _safe_list(squeeze_pctile),
        "Squeeze_Signal": _safe_list(signal),
    }


@registry.register(
    id="rsi_divergence",
    name="RSI 背離",
    category="先行訊號",
    description="偵測價格與 RSI 的底背離（看多）和頂背離（看空），是趨勢反轉的早期信號",
    parameters={
        "rsi_period": {"default": 14, "min": 7, "max": 30, "type": "int"},
        "lookback": {"default": 30, "min": 10, "max": 60, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("rsi_period", 14) + p.get("lookback", 30),
    pro_tip="+1=底背離(看多信號), -1=頂背離(看空信號), 0=無背離",
)
def calc_rsi_divergence(df: pd.DataFrame, params: dict) -> dict[str, list]:
    rsi_period = int(params.get("rsi_period", 14))
    lookback = int(params.get("lookback", 30))
    close = df["close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period).mean()
    safe_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / safe_loss
    rsi = 100 - (100 / (1 + rs))

    divergence = _vectorized_divergence(close, rsi, lookback)
    return {"RSI_Div": _safe_list(divergence)}


@registry.register(
    id="vol_divergence",
    name="成交量背離",
    category="先行訊號",
    description="偵測價格方向與 OBV 方向不一致的情況，預示趨勢可能反轉",
    parameters={
        "period": {"default": 20, "min": 10, "max": 50, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("period", 20) + 5,
    pro_tip="+1=量價背離看多(價格跌但 OBV 升), -1=量價背離看空(價格漲但 OBV 跌), 0=正常",
)
def calc_vol_divergence(df: pd.DataFrame, params: dict) -> dict[str, list]:
    period = int(params.get("period", 20))
    close = df["close"]
    volume = df["volume"]

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    price_slope = close.rolling(window=period).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= period else np.nan,
        raw=False,
    )
    obv_slope = obv.rolling(window=period).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= period else np.nan,
        raw=False,
    )

    # 標準化斜率方向
    price_dir = np.sign(price_slope)
    obv_dir = np.sign(obv_slope)

    divergence = pd.Series(0.0, index=df.index)
    # 價格下跌但 OBV 上升 → 底部背離（看多）
    divergence[(price_dir < 0) & (obv_dir > 0)] = 1.0
    # 價格上漲但 OBV 下跌 → 頂部背離（看空）
    divergence[(price_dir > 0) & (obv_dir < 0)] = -1.0

    return {
        "Vol_Div": _safe_list(divergence),
    }


@registry.register(
    id="macd_divergence",
    name="MACD 背離",
    category="先行訊號",
    description="偵測價格與 MACD 柱狀圖的底背離（看多）和頂背離（看空），與 RSI 背離形成雙重確認",
    parameters={
        "fast": {"default": 12, "min": 5, "max": 20, "type": "int"},
        "slow": {"default": 26, "min": 15, "max": 40, "type": "int"},
        "signal": {"default": 9, "min": 5, "max": 15, "type": "int"},
        "lookback": {"default": 30, "min": 10, "max": 60, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: p.get("slow", 26) + p.get("lookback", 30),
    pro_tip="+1=底背離(看多信號), -1=頂背離(看空信號), 0=無背離",
)
def calc_macd_divergence(df: pd.DataFrame, params: dict) -> dict[str, list]:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    sig = int(params.get("signal", 9))
    lookback = int(params.get("lookback", 30))
    close = df["close"]

    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, sig)
    histogram = macd_line - signal_line

    divergence = _vectorized_divergence(close, histogram, lookback)
    return {"MACD_Div": _safe_list(divergence)}


@registry.register(
    id="leading_composite",
    name="綜合先行訊號",
    category="先行訊號",
    description="合併波動壓縮、RSI 背離、MACD 背離、成交量背離四種先行訊號，產生三級警示",
    parameters={
        "bb_period": {"default": 20, "min": 10, "max": 50, "type": "int"},
        "rsi_period": {"default": 14, "min": 7, "max": 30, "type": "int"},
        "lookback": {"default": 30, "min": 10, "max": 60, "type": "int"},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: max(p.get("bb_period", 20), p.get("rsi_period", 14)) + p.get("lookback", 30) + 120,
    pro_tip="1=注意(單一訊號), 2=警示(雙重確認), 3=強烈警示(三重以上)。正值看多，負值看空。",
)
def calc_leading_composite(df: pd.DataFrame, params: dict) -> dict[str, list]:
    bb_period = int(params.get("bb_period", 20))
    rsi_period = int(params.get("rsi_period", 14))
    lookback = int(params.get("lookback", 30))
    close = df["close"]
    n = len(df)

    # 1. 波動壓縮（布林帶寬度分位 <20%）
    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    safe_sma = sma.replace(0, np.nan)
    bb_width = (2 * std / safe_sma) * 100
    squeeze_pctile = bb_width.rolling(window=120, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    is_squeeze = (squeeze_pctile < 0.20).astype(float)

    # 2. RSI 背離
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period).mean()
    safe_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / safe_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_div = _vectorized_divergence(close, rsi, lookback)

    # 3. MACD 背離
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_hist = ema12 - ema26 - _ema(ema12 - ema26, 9)
    macd_div = _vectorized_divergence(close, macd_hist, lookback)

    # 4. 成交量背離（簡化版：OBV slope vs price slope）
    volume = df["volume"]
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    price_slope = close.rolling(window=lookback).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= lookback else np.nan, raw=False,
    )
    obv_slope = obv.rolling(window=lookback).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= lookback else np.nan, raw=False,
    )
    vol_div = pd.Series(0.0, index=df.index)
    vol_div[(np.sign(price_slope) < 0) & (np.sign(obv_slope) > 0)] = 1.0
    vol_div[(np.sign(price_slope) > 0) & (np.sign(obv_slope) < 0)] = -1.0

    # 合併：計算看多/看空分別有幾個訊號觸發
    bullish_count = (
        is_squeeze
        + (rsi_div == 1).astype(float)
        + (macd_div == 1).astype(float)
        + (vol_div == 1).astype(float)
    )
    bearish_count = (
        is_squeeze
        + (rsi_div == -1).astype(float)
        + (macd_div == -1).astype(float)
        + (vol_div == -1).astype(float)
    )

    # 產生綜合信號：正值看多，負值看空，絕對值=警示級別
    composite = pd.Series(0.0, index=df.index)
    composite[bullish_count >= 1] = bullish_count[bullish_count >= 1]
    composite[bearish_count >= 1] = -bearish_count[bearish_count >= 1]
    # 看多看空同時觸發時取較強的
    both = (bullish_count >= 1) & (bearish_count >= 1)
    composite[both] = np.where(
        bullish_count[both] >= bearish_count[both],
        bullish_count[both], -bearish_count[both],
    )

    # 上限 ±3
    composite = composite.clip(-3, 3)

    return {
        "Leading_Signal": _safe_list(composite),
        "Bullish_Count": _safe_list(bullish_count),
        "Bearish_Count": _safe_list(bearish_count),
    }


# =============================================
# 多時間框架結構轉變 (MSS) 偵測
# =============================================

_ALL_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]
_TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}


def _detect_structure(close: pd.Series, high: pd.Series, low: pd.Series,
                      pct_threshold: float = 0.03) -> pd.Series:
    """用 Zigzag 邏輯偵測市場結構（HH/HL/LH/LL）。

    回傳值：+1=多頭結構(HH+HL), -1=空頭結構(LH+LL), 0=不確定。
    使用百分比閾值過濾小波動。
    """
    n = len(close)
    structure = pd.Series(0.0, index=close.index)
    if n < 10:
        return structure

    # Zigzag 轉折點偵測
    pivots_high = []
    pivots_low = []

    window = max(5, int(n * 0.02))  # 動態窗口
    window = min(window, 30)

    roll_max = high.rolling(window=window, center=True, min_periods=3).max()
    roll_min = low.rolling(window=window, center=True, min_periods=3).min()

    is_pivot_high = (high == roll_max) & (high > close.shift(1) * (1 + pct_threshold * 0.3))
    is_pivot_low = (low == roll_min) & (low < close.shift(1) * (1 - pct_threshold * 0.3))

    for i in range(window, n - window // 2):
        if is_pivot_high.iloc[i]:
            pivots_high.append((i, float(high.iloc[i])))
        if is_pivot_low.iloc[i]:
            pivots_low.append((i, float(low.iloc[i])))

    if len(pivots_high) < 2 or len(pivots_low) < 2:
        return structure

    # 取最近的轉折點判斷結構
    last_highs = pivots_high[-3:] if len(pivots_high) >= 3 else pivots_high[-2:]
    last_lows = pivots_low[-3:] if len(pivots_low) >= 3 else pivots_low[-2:]

    hh = last_highs[-1][1] > last_highs[-2][1]  # Higher High
    ll = last_lows[-1][1] < last_lows[-2][1]     # Lower Low
    hl = last_lows[-1][1] > last_lows[-2][1]     # Higher Low
    lh = last_highs[-1][1] < last_highs[-2][1]   # Lower High

    # 填入最近 N 根的結構判斷
    fill_start = max(last_highs[-1][0], last_lows[-1][0])
    if hh and hl:
        structure.iloc[fill_start:] = 1.0   # 多頭
    elif ll and lh:
        structure.iloc[fill_start:] = -1.0  # 空頭
    elif hh and not hl:
        structure.iloc[fill_start:] = 0.5   # 偏多但不確定
    elif ll and not lh:
        structure.iloc[fill_start:] = -0.5  # 偏空但不確定

    return structure


@registry.register(
    id="mtf_mss",
    name="多時間框架結構 (MSS)",
    category="先行訊號",
    description="跨所有時間級別（15m/1h/4h/1d/1w）偵測市場結構並交叉比對。當小級別結構反轉但大級別未反轉時，產生先行警示。",
    parameters={
        "pct_threshold": {"default": 3.0, "min": 1.0, "max": 10.0, "type": "float", "step": 0.5},
    },
    display_mode="sub_chart",
    warmup_func=lambda p: 50,
    pro_tip="Current_TF=當前級別結構。Alignment=多時間框架一致性(-5~+5)。MSS_Signal=當小級別反轉但大級別未反轉時觸發。",
)
def calc_mtf_mss(df: pd.DataFrame, params: dict) -> dict[str, list]:
    pct_threshold = float(params.get("pct_threshold", 3.0)) / 100.0
    n = len(df)

    # 當前時間級別的結構
    current_structure = _detect_structure(df["close"], df["high"], df["low"], pct_threshold)

    # 嘗試載入其他時間級別的數據做交叉比對
    try:
        from app.data.fetchers.crypto_engine import crypto_engine

        symbol = params.get("_symbol", "BTC/USDT")
        current_tf = params.get("_timeframe", "4h")

        # fallback: 從 df 的時間間隔推測
        if current_tf not in _TF_MINUTES and n >= 2:
            timestamps = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df.index)
            if len(timestamps) >= 2:
                diff_minutes = (timestamps.iloc[-1] - timestamps.iloc[-2]).total_seconds() / 60
                for tf, mins in sorted(_TF_MINUTES.items(), key=lambda x: x[1]):
                    if abs(diff_minutes - mins) < mins * 0.3:
                        current_tf = tf
                        break

        # 載入其他級別數據並計算結構
        tf_structures: dict[str, float] = {}
        for tf in _ALL_TIMEFRAMES:
            if tf == current_tf:
                val = float(current_structure.iloc[-1]) if not pd.isna(current_structure.iloc[-1]) else 0.0
                tf_structures[tf] = val
                continue
            try:
                tf_df = crypto_engine.load_local_data(symbol, tf)
                if tf_df.empty or len(tf_df) < 30:
                    continue
                tf_df = tf_df.tail(300)
                s = _detect_structure(tf_df["close"], tf_df["high"], tf_df["low"], pct_threshold)
                val = float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 0.0
                tf_structures[tf] = val
            except Exception:
                pass

        # 計算多時間框架一致性分數 (-5 ~ +5)
        alignment = sum(tf_structures.values())

        # MSS 信號：小級別反轉但大級別未反轉
        mss_signal = pd.Series(0.0, index=df.index)
        current_tf_idx = _ALL_TIMEFRAMES.index(current_tf) if current_tf in _ALL_TIMEFRAMES else 2
        current_val = tf_structures.get(current_tf, 0)

        # 找大級別的結構共識
        larger_tfs = [tf for tf in _ALL_TIMEFRAMES[current_tf_idx + 1:] if tf in tf_structures]
        if larger_tfs:
            larger_consensus = sum(tf_structures[tf] for tf in larger_tfs) / len(larger_tfs)

            # 小級別轉空但大級別仍多 → 先行看空警示
            if current_val < -0.3 and larger_consensus > 0.3:
                mss_signal.iloc[-1] = -1.0
            # 小級別轉多但大級別仍空 → 先行看多警示
            elif current_val > 0.3 and larger_consensus < -0.3:
                mss_signal.iloc[-1] = 1.0

        # 向前填充 MSS 信號（最近 5 根都標記）
        if float(mss_signal.iloc[-1]) != 0.0:
            fill_count = min(5, n)
            mss_signal.iloc[-fill_count:] = mss_signal.iloc[-1]

        # 將一致性分數填充到整個序列（給 LLM 參考）
        alignment_series = pd.Series(alignment, index=df.index)

        # 構建各級別結構標籤
        tf_label_map = {}
        for tf, val in tf_structures.items():
            if val > 0.5:
                tf_label_map[tf] = "bullish"
            elif val < -0.5:
                tf_label_map[tf] = "bearish"
            elif val > 0:
                tf_label_map[tf] = "lean_bullish"
            elif val < 0:
                tf_label_map[tf] = "lean_bearish"
            else:
                tf_label_map[tf] = "neutral"

        return {
            "Current_TF_Structure": _safe_list(current_structure),
            "MTF_Alignment": _safe_list(alignment_series),
            "MSS_Signal": _safe_list(mss_signal),
        }

    except Exception:
        # 如果無法載入其他級別，只返回當前級別結構
        return {
            "Current_TF_Structure": _safe_list(current_structure),
            "MTF_Alignment": _safe_list(pd.Series(0.0, index=df.index)),
            "MSS_Signal": _safe_list(pd.Series(0.0, index=df.index)),
        }
