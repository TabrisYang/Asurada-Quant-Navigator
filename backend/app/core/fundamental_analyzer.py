"""阿斯拉量化系統 — 台股基本面分析引擎

整合月營收、法人買賣超、外資持股、財報摘要，
產出綜合基本面評分和分析報告。
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

from app.data.fetchers.tw_fundamental import tw_fundamental
from app.data.tw_sectors import get_stock_name


async def analyze_fundamentals(symbol: str) -> dict:
    """台股基本面分析。

    Args:
        symbol: 如 "2330/TWD"

    Returns:
        dict: 營收趨勢、法人動向、財報摘要、綜合評分
    """
    code = symbol.split("/")[0]
    name = get_stock_name(code)

    logger.info(f"基本面分析 [{code} {name}]: 開始抓取數據...")

    # 平行抓取四種數據
    import asyncio
    revenue_task = tw_fundamental.fetch_revenue(code, months=12)
    institutional_task = tw_fundamental.fetch_institutional(code, days=30)
    holding_task = tw_fundamental.fetch_foreign_holding(code)
    financials_task = tw_fundamental.fetch_financials(code)

    results = await asyncio.gather(
        revenue_task, institutional_task, holding_task, financials_task,
        return_exceptions=True,
    )

    revenue_df = results[0] if not isinstance(results[0], Exception) else pd.DataFrame()
    institutional_df = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
    holding_df = results[2] if not isinstance(results[2], Exception) else pd.DataFrame()
    financials = results[3] if not isinstance(results[3], Exception) else {}

    # 分析各維度
    revenue_analysis = _analyze_revenue(revenue_df)
    institutional_analysis = _analyze_institutional(institutional_df)
    holding_analysis = _analyze_holding(holding_df)
    financial_analysis = _analyze_financials(financials)

    # 綜合評分
    score = _compute_fundamental_score(
        revenue_analysis, institutional_analysis, holding_analysis, financial_analysis,
    )

    return {
        "status": "success",
        "symbol": symbol,
        "code": code,
        "name": name,
        "revenue": revenue_analysis,
        "institutional": institutional_analysis,
        "foreign_holding": holding_analysis,
        "financials": financial_analysis,
        "fundamental_score": score,
    }


def _analyze_revenue(df: pd.DataFrame) -> dict:
    """分析月營收趨勢。"""
    if df.empty or "revenue" not in df.columns:
        return {"available": False}

    latest = df.iloc[-1]
    result = {
        "available": True,
        "latest_month": latest.get("year_month", "?"),
        "latest_revenue": round(float(latest["revenue"]), 0) if pd.notna(latest["revenue"]) else None,
        "months_data": len(df),
    }

    if "mom_pct" in df.columns and pd.notna(df["mom_pct"].iloc[-1]):
        result["mom_pct"] = round(float(df["mom_pct"].iloc[-1]), 2)

    if "yoy_pct" in df.columns and pd.notna(df["yoy_pct"].iloc[-1]):
        result["yoy_pct"] = round(float(df["yoy_pct"].iloc[-1]), 2)

    # 連續成長月數
    if "mom_pct" in df.columns:
        streak = 0
        for i in range(len(df) - 1, -1, -1):
            if pd.notna(df["mom_pct"].iloc[i]) and df["mom_pct"].iloc[i] > 0:
                streak += 1
            else:
                break
        result["consecutive_growth_months"] = streak

    # 趨勢判定
    yoy = result.get("yoy_pct", 0)
    if yoy and yoy > 20:
        result["trend"] = "高速成長"
    elif yoy and yoy > 5:
        result["trend"] = "穩定成長"
    elif yoy and yoy > -5:
        result["trend"] = "持平"
    elif yoy and yoy > -20:
        result["trend"] = "衰退"
    else:
        result["trend"] = "大幅衰退" if yoy else "無法判定"

    return result


def _analyze_institutional(df: pd.DataFrame) -> dict:
    """分析三大法人買賣超。"""
    if df.empty:
        return {"available": False}

    result = {"available": True, "days_data": len(df)}

    # 外資
    if "foreign_buy_sell" in df.columns:
        foreign = df["foreign_buy_sell"]
        result["foreign_net_20d"] = round(float(foreign.sum()), 0)
        result["foreign_consecutive_buy"] = _count_streak(foreign, positive=True)
        result["foreign_consecutive_sell"] = _count_streak(foreign, positive=False)

    # 投信
    if "trust_buy_sell" in df.columns:
        trust = df["trust_buy_sell"]
        result["trust_net_20d"] = round(float(trust.sum()), 0)
        result["trust_consecutive_buy"] = _count_streak(trust, positive=True)

    # 自營商
    if "dealer_buy_sell" in df.columns:
        dealer = df["dealer_buy_sell"]
        result["dealer_net_20d"] = round(float(dealer.sum()), 0)

    # 法人方向判定
    foreign_net = result.get("foreign_net_20d", 0)
    trust_net = result.get("trust_net_20d", 0)
    if foreign_net > 0 and trust_net > 0:
        result["direction"] = "外資+投信同步買超（強烈偏多）"
    elif foreign_net > 0:
        result["direction"] = "外資買超（偏多）"
    elif trust_net > 0:
        result["direction"] = "投信買超（偏多）"
    elif foreign_net < 0 and trust_net < 0:
        result["direction"] = "外資+投信同步賣超（偏空）"
    elif foreign_net < 0:
        result["direction"] = "外資賣超（偏空）"
    else:
        result["direction"] = "法人中性"

    return result


def _analyze_holding(df: pd.DataFrame) -> dict:
    """分析外資持股。"""
    if df.empty:
        return {"available": False}

    latest = df.iloc[-1]
    return {
        "available": True,
        "institution_pct": round(float(latest.get("institution_pct", 0)) * 100, 2) if latest.get("institution_pct") else None,
        "insider_pct": round(float(latest.get("insider_pct", 0)) * 100, 2) if latest.get("insider_pct") else None,
    }


def _analyze_financials(financials: dict) -> dict:
    """分析財報摘要。"""
    if not financials:
        return {"available": False}

    result = {"available": True}

    for key in ["pe_ratio", "forward_pe", "pb_ratio", "dividend_yield",
                "eps_trailing", "eps_forward", "market_cap", "52w_high",
                "52w_low", "avg_volume", "company_name"]:
        val = financials.get(key)
        if val is not None:
            result[key] = round(float(val), 2) if isinstance(val, (int, float)) else val

    # 本益比評估
    pe = financials.get("pe_ratio")
    if pe:
        if pe < 10:
            result["pe_assessment"] = "低估（PE < 10）"
        elif pe < 15:
            result["pe_assessment"] = "合理偏低"
        elif pe < 25:
            result["pe_assessment"] = "合理"
        elif pe < 40:
            result["pe_assessment"] = "偏高"
        else:
            result["pe_assessment"] = "高估（PE > 40）"

    # 殖利率評估
    dy = financials.get("dividend_yield")
    if dy:
        dy_pct = dy * 100 if dy < 1 else dy  # yfinance 可能回傳小數或百分比
        result["dividend_yield"] = round(dy_pct, 2)
        if dy_pct > 5:
            result["yield_assessment"] = "高殖利率（> 5%）"
        elif dy_pct > 3:
            result["yield_assessment"] = "中等殖利率"
        else:
            result["yield_assessment"] = "低殖利率"

    return result


def _compute_fundamental_score(revenue, institutional, holding, financials) -> dict:
    """綜合基本面評分 -10 ~ +10。"""
    score = 0

    # 營收趨勢
    if revenue.get("available"):
        yoy = revenue.get("yoy_pct", 0)
        if yoy and yoy > 20:
            score += 3
        elif yoy and yoy > 5:
            score += 1
        elif yoy and yoy < -20:
            score -= 3
        elif yoy and yoy < -5:
            score -= 1

        streak = revenue.get("consecutive_growth_months", 0)
        if streak >= 3:
            score += 1

    # 法人動向
    if institutional.get("available"):
        foreign_net = institutional.get("foreign_net_20d", 0)
        trust_net = institutional.get("trust_net_20d", 0)
        if foreign_net > 0 and trust_net > 0:
            score += 2
        elif foreign_net > 0 or trust_net > 0:
            score += 1
        elif foreign_net < 0 and trust_net < 0:
            score -= 2
        elif foreign_net < 0:
            score -= 1

    # 財報
    if financials.get("available"):
        pe = financials.get("pe_ratio")
        if pe:
            if pe < 12:
                score += 1
            elif pe > 40:
                score -= 1

        dy = financials.get("dividend_yield")
        if dy and dy > 4:
            score += 1

    score = max(-10, min(10, score))

    if score >= 5:
        direction = "基本面強勢"
    elif score >= 2:
        direction = "基本面偏多"
    elif score <= -5:
        direction = "基本面弱勢"
    elif score <= -2:
        direction = "基本面偏空"
    else:
        direction = "基本面中性"

    return {"score": score, "max_possible": 10, "direction": direction}


def _count_streak(series: pd.Series, positive: bool = True) -> int:
    """計算連續正/負值天數。"""
    streak = 0
    for val in reversed(series.values):
        if (positive and val > 0) or (not positive and val < 0):
            streak += 1
        else:
            break
    return streak
