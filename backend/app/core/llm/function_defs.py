"""阿斯拉量化系統 — LLM Function Calling 定義

定義 LLM 可呼叫的函式（Level 1 + Level 2）。
格式遵循 OpenAI 規範，其他供應商的適配器負責轉換。

SYSTEM_PROMPT v2.0 — 融入專業量化決策引擎定位
拆分為核心 + 按需模組，依使用者意圖動態組裝，減少 token 消耗並提升 LLM 專注度。
新增模組：regime_v2, analysis_v2, factor_validation, risk_checklist, alpha_monitor,
          output_lite, output_full
"""

# ═══════════════════════════════════════════════════════
#  意圖偵測
# ═══════════════════════════════════════════════════════

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "simple_query": [
        "多少", "幾點", "現在", "目前", "查一下", "數值",
        "顯示", "看一下", "切換", "換成", "打開", "關閉",
    ],
    "drawing": [
        "畫", "繪", "標記", "線型", "趨勢線", "支撐", "壓力", "通道",
        "諧波", "型態", "pattern", "旗型", "三角", "頭肩", "楔形",
        "draw", "annotate", "mark", "PRZ",
    ],
    "analysis": [
        "分析", "趨勢", "方向", "進場", "出場", "預測", "建議",
        "走勢", "看多", "看空", "漲", "跌", "做多", "做空",
        "多頭", "空頭", "突破", "反轉", "目標", "止損", "止盈",
    ],
    "backtest": [
        "回測", "策略", "backtest", "勝率", "開倉", "合約", "槓桿",
        "盈虧比", "sharpe", "報酬", "比較策略",
    ],
    "quant_research": [
        "量化研究", "alpha", "因子", "ic分析", "monte carlo",
        "walk forward", "穩定性", "因子分析", "策略可行", "穩不穩",
        "decay", "衰退", "失效", "預測力", "哪個指標最準",
        "什麼指標最有用", "最適合的因子", "有效因子", "最準確",
    ],
    "calibrate": [
        "校準", "calibrat", "最佳參數", "適合的參數", "閾值",
        "超買超賣門檻",
    ],
    "event_analysis": [
        "大漲", "大跌", "爆量", "共通", "特徵", "事件",
        "之前都", "通常", "暴跌前", "暴漲前", "規律",
    ],
    "conditional_prob": [
        "機率", "概率", "多少的時候", "在什麼值", "什麼範圍", "幾的時候",
        "條件機率", "conditional", "勝率最高",
        "什麼數值", "多少時", "在哪個區間", "最佳區間",
    ],
    "scenario": [
        "情境", "預測情境", "三種可能", "三個可能", "可能性", "情境分析",
        "接下來", "未來走勢", "後續走勢", "scenario", "會怎麼走",
        "預測", "三大", "最有可能",
    ],
    "smc": [
        "訂單流", "SMC", "聰明錢", "smart money",
        "BOS", "CHoCH", "流動性", "sweep",
        "FVG", "失衡區", "order block",
        "機構", "結構破壞", "結構轉折",
    ],
    "deep_analysis": [
        "完整分析", "全面分析", "詳細分析", "深度分析",
        "full analysis", "deep analysis",
    ],
    "deep_phase1": [
        "完整分析一", "完整分析1", "全面分析一", "深度分析一",
    ],
    "deep_phase2": [
        "完整分析二", "完整分析2", "全面分析二", "深度分析二",
    ],
    "deep_phase3": [
        "完整分析三", "完整分析3", "全面分析三", "深度分析三",
    ],
    "comprehensive_analysis": [
        "全部分析", "完整全面分析", "一次完整", "comprehensive",
    ],
    "factor_validation": [
        "因子驗證", "因子排名", "ic排名", "哪個因子", "哪些因子有效",
        "因子分析", "factor validation", "因子評分",
    ],
    "strategy_backtest": [
        "策略回測", "回測驗證", "backtest", "策略驗證", "mc驗證",
        "walk forward", "cpcv", "過擬合",
    ],
    "regime_analysis": [
        "市場體制", "regime", "gmm", "garch", "hmm", "波動率預測",
        "市場環境", "市場狀態", "體制分析",
    ],
    "momentum_analysis": [
        "動能", "動量", "momentum", "追強", "動能分析",
        "相對強弱", "roc", "加速", "減速", "動量反轉",
    ],
    "data_sync": [
        "下載", "同步", "抓取", "下載數據", "sync", "download",
        "沒有數據", "下載K線", "抓取K線", "抓取數據",
    ],
    "fundamental_analysis": [
        "基本面", "營收", "法人", "外資", "投信", "本益比", "eps",
        "殖利率", "股利", "財報", "買賣超", "持股",
        "fundamental", "revenue", "pe ratio",
    ],
    "crypto_fundamental": [
        "代幣經濟", "tokenomics", "協議基本面", "生態活躍", "tvl",
        "流通量", "流通供給", "fdv", "鏈上", "代幣供給", "代幣解鎖",
    ],
    "sector_analysis": [
        "族群", "概念股", "板塊", "產業分析", "sector",
        "半導體族群", "金融股", "航運股", "哪些族群",
        "整個產業", "產業趨勢", "族群分析", "類股",
        "哪些產業", "什麼產業", "什麼族群", "列出族群", "列出產業",
        "可以分析哪些",
        # 主要族群名稱
        "半導體", "晶圓代工", "ic設計", "封測", "矽晶圓",
        "金融", "電子代工", "ai概念", "ai伺服器", "ai晶片",
        "航運", "鋼鐵", "生技", "綠能",
        "汽車電子", "電動車", "車用零件",
        "被動元件", "mlcc", "電源散熱", "網通",
        "電信", "食品", "營建", "觀光", "面板", "pcb",
        "記憶體", "蘋果供應鏈", "蘋概", "伺服器",
    ],
}


def detect_intents(message: str, mode: str | None = None) -> set[str]:
    """根據使用者訊息和 mode 偵測意圖，決定載入哪些 SYSTEM_PROMPT 模組。"""
    intents: set[str] = set()
    msg = message.lower()

    if mode == "quant_research":
        intents.update({"backtest", "quant_research", "analysis"})
    elif mode == "calibrate":
        intents.add("calibrate")
    elif mode == "crypto_fundamental":
        intents.add("crypto_fundamental")

    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            intents.add(intent)

    # 加密基本面優先於台股基本面：「加密基本面 vs 技術面」prompt 含「基本面」
    # 會誤命中 fundamental_analysis（台股模組）→ 強制移除，只留 crypto 模組。
    if "crypto_fundamental" in intents:
        intents.discard("fundamental_analysis")

    # ── 深度分析互斥：phase1/2/3 命中時，移除泛用的 deep_analysis 和 analysis
    _deep_phases = {"deep_phase1", "deep_phase2", "deep_phase3"}
    if intents & _deep_phases:
        intents.discard("deep_analysis")
        intents.discard("analysis")  # 各階段有專屬模組，不需一般分析模組
    elif "deep_analysis" in intents:
        intents.discard("analysis")  # 完整分析有專屬模組

    # ── comprehensive_analysis（全部分析）支配：吃掉所有其他分析意圖
    if "comprehensive_analysis" in intents:
        for sub in (
            "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3",
            "analysis", "factor_validation", "momentum_analysis",
            "regime_analysis", "strategy_backtest",
        ):
            intents.discard(sub)

    # 降級邏輯：同時命中 simple_query + analysis 但沒有明確分析詞 → 視為簡單查詢
    if "simple_query" in intents and "analysis" in intents:
        _strong_analysis = {"分析", "預測", "建議", "進場", "出場", "做多", "做空"}
        has_strong = any(kw in msg for kw in _strong_analysis)
        deep = {"backtest", "quant_research", "calibrate", "event_analysis",
                "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3"}
        if not has_strong and not (intents & deep):
            intents.discard("analysis")

    if not intents:
        intents.add("general")

    return intents


# prompt 內容常數已抽至 prompt_modules.py（repo health 行數護欄）；
# 此處 import 回原名稱，對外介面（assemble_* / SYSTEM_PROMPT 等）完全不變
from app.core.llm.prompt_modules import (  # noqa: F401
    _DIMENSION_TAIL_SPEC,
    _PROMPT_CORE,
    _PROMPT_MODULES,
)



# ═══════════════════════════════════════════════════════
#  動態組裝
# ═══════════════════════════════════════════════════════

_INTENT_TO_MODULES: dict[str, list[str]] = {
    "general": [],
    "simple_query": [],
    "drawing": ["drawing"],
    "analysis": [
        "regime_v2", "analysis_v2", "factor_validation",
        "scenario", "smc",  # ★ 日常分析自動含情境預測 + SMC
        "output_lite", "drawing", "backtest", "risk_checklist", "auto_backtest",
    ],
    "backtest": ["regime_v2", "backtest", "risk_checklist"],
    "quant_research": [
        "regime_v2", "quant_research", "backtest",
        "factor_validation", "risk_checklist", "alpha_monitor", "output_full",
        "conditional_prob",
    ],
    "calibrate": ["calibrate"],
    "event_analysis": ["event_analysis"],
    "conditional_prob": ["conditional_prob"],
    "scenario": ["scenario", "regime_v2", "analysis_v2"],
    "smc": ["smc", "regime_v2", "drawing"],
    # ── 三階段完整分析 ──
    "deep_analysis": [  # 不帶數字 → 等同第一階段
        "regime_v2", "analysis_v2", "factor_validation",
        "scenario", "smc", "drawing", "risk_checklist", "output_deep_phase1",
    ],
    "deep_phase1": [
        "regime_v2", "analysis_v2", "factor_validation",
        "scenario", "smc", "drawing", "risk_checklist", "output_deep_phase1",
    ],
    "deep_phase2": [
        "smc", "backtest", "risk_checklist", "auto_backtest",
        "event_analysis", "conditional_prob", "drawing", "output_deep_phase2",
    ],
    "deep_phase3": [
        "quant_research", "calibrate", "alpha_monitor",
        "risk_checklist", "output_full", "output_deep_phase3",
    ],
    "factor_validation": [
        "regime_v2", "quant_research", "factor_validation",
        "conditional_prob", "output_lite", "factor_validation_mode",
    ],
    "strategy_backtest": [
        "regime_v2", "backtest", "quant_research",
        "risk_checklist", "output_lite", "strategy_backtest_mode",
    ],
    "regime_analysis": [
        "regime_v2", "analysis_v2", "output_lite", "regime_analysis_mode",
    ],
    "momentum_analysis": [
        "regime_v2", "output_lite", "risk_checklist", "momentum_analysis_mode",
    ],
    "data_sync": [
        "data_sync_mode",
    ],
    "fundamental_analysis": [
        "output_lite", "fundamental_analysis_mode",
    ],
    "crypto_fundamental": [
        "output_lite", "crypto_fundamental_mode",
    ],
    "sector_analysis": [
        "regime_v2", "sector_analysis", "output_lite", "risk_checklist",
    ],
    # ── 全部分析（去重統合版）──
    # 載入所有相關模組，後續由 _DOMINANCE 統一移除被支配的低階模組
    "comprehensive_analysis": [
        "regime_v2", "analysis_v2", "factor_validation",
        "scenario", "smc", "drawing", "risk_checklist",
        "backtest", "auto_backtest", "conditional_prob",
        "quant_research", "calibrate", "alpha_monitor",
        "comprehensive_analysis",  # 統合報告 prompt（最重要，定義報告結構）
    ],
}

# ═══════════════════════════════════════════════════════
# 模組支配關係（B3）— 高階模組支配低階，避免重複載入
# ═══════════════════════════════════════════════════════
# 例：comprehensive_analysis 已包含三階段 + 因子驗證 + 動能 + 量化研究的內容，
#     若同時被偵測到（理論上 detect_intents 已排除），仍要保險把低階模組移除
_DOMINANCE: dict[str, list[str]] = {
    "quant_research": ["factor_validation"],
    "output_deep_phase3": ["output_deep_phase1", "output_deep_phase2"],
    "comprehensive_analysis": [
        "factor_validation_mode", "momentum_analysis_mode",
        "regime_analysis_mode", "strategy_backtest_mode",
        "output_deep_phase1", "output_deep_phase2", "output_deep_phase3",
        "output_lite", "output_full",  # 統合 prompt 自帶輸出格式
    ],
}


# ═══════════════════════════════════════════════════════
# 編排管線（map-reduce）— 全部分析 seg2 拆成 5 個維度 focused call
# ═══════════════════════════════════════════════════════
# 每個維度載入自己的專用 _mode 模組（品質對齊「單獨問該維度」），
# 由 assemble_dimension_prompt 組裝。reduce 階段用 comprehensive_synthesis。
_PIPELINE_DIMENSIONS: dict[str, list[str]] = {
    "regime":    ["regime_v2", "analysis_v2", "regime_analysis_mode"],
    "structure": ["smc", "scenario"],
    "momentum":  ["momentum_analysis_mode", "conditional_prob"],
    "backtest":  ["strategy_backtest_mode", "backtest", "auto_backtest", "risk_checklist"],
    "quant":     ["factor_validation_mode", "quant_research", "factor_validation", "alpha_monitor"],
}

# 維度執行順序（決定 bubble 顯示順序 / segment 編號）
_PIPELINE_DIMENSION_ORDER = ("regime", "structure", "momentum", "backtest", "quant")


def assemble_system_prompt(
    intents: set[str], teaching_mode: bool = False, segment: int = 0,
) -> str:
    """根據偵測到的意圖集合，組裝最終的 SYSTEM_PROMPT。

    Args:
        intents: 偵測到的使用者意圖集合
        teaching_mode: 啟用教學模式（解釋指標意義、信號邏輯、策略風險）
        segment: 分段輸出指定（S2 新增）
            0 = 預設單段（向後相容）
            1 = 全部分析第 1 段（30 秒結論卡 + 倉位）
            2 = 全部分析第 2 段（完整詳細分析）
            僅在 intents 含 "comprehensive_analysis" 時生效
    """
    modules_needed: set[str] = set()
    for intent in intents:
        for mod in _INTENT_TO_MODULES.get(intent, []):
            modules_needed.add(mod)

    # 永遠載入數據下載指引（讓 LLM 在任何場景都知道可以下載數據）
    modules_needed.add("data_sync_mode")

    # 自動注入用戶設定的策略模組
    try:
        from app.core.user_strategies import get_auto_inject_modules
        for mod in get_auto_inject_modules(intents):
            if mod in _PROMPT_MODULES:
                modules_needed.add(mod)
    except Exception:
        pass  # 策略庫尚未初始化時靜默忽略

    if teaching_mode:
        modules_needed.add("teaching")

    # S2: 分段輸出 — 替換 comprehensive_analysis 為 seg1 / seg2
    if segment in (1, 2) and "comprehensive_analysis" in modules_needed:
        modules_needed.discard("comprehensive_analysis")
        modules_needed.add(f"comprehensive_analysis_seg{segment}")

    static, dynamic = assemble_system_prompt_split(modules_needed)
    return static + dynamic


def assemble_dimension_prompt(dimension: str, teaching_mode: bool = False) -> str:
    """組裝編排管線「單一維度」focused call 的 system prompt。

    載入該維度的專用模組（含 _mode 模組）+ 共用維度尾規格 _DIMENSION_TAIL_SPEC。
    複用 assemble_system_prompt_split 以保留 _DOMINANCE 處理與 prompt caching 結構。

    Args:
        dimension: _PIPELINE_DIMENSIONS 的 key（regime / structure / momentum / backtest / quant）
        teaching_mode: 啟用教學模式
    """
    modules = _PIPELINE_DIMENSIONS.get(dimension)
    if not modules:
        raise ValueError(f"unknown pipeline dimension: {dimension!r}")
    modules_needed = set(modules)
    modules_needed.add("data_sync_mode")
    if teaching_mode:
        modules_needed.add("teaching")

    static, dynamic = assemble_system_prompt_split(modules_needed)
    # 維度尾規格附在 static 段（內容穩定、可 cache），時間戳仍置於最後
    return static + "\n" + _DIMENSION_TAIL_SPEC + dynamic


def assemble_synthesis_prompt(teaching_mode: bool = False) -> str:
    """組裝編排管線 reduce 階段（synthesis）的 system prompt。"""
    modules_needed = {"comprehensive_synthesis", "risk_checklist", "data_sync_mode"}
    if teaching_mode:
        modules_needed.add("teaching")
    static, dynamic = assemble_system_prompt_split(modules_needed)
    return static + dynamic


# 固定順序確保 prompt 結構一致（模組層常數，給 split 函式用）
_MODULE_ORDER = (
    "teaching",
    "regime_v2", "analysis_v2", "factor_validation",
    "auto_backtest", "risk_checklist", "alpha_monitor",
    "output_lite", "output_full",
    "output_deep_phase1", "output_deep_phase2", "output_deep_phase3",
    "comprehensive_analysis",  # ★ 全部分析統合 prompt（位置在三階段之後、其他細節之前）
    "comprehensive_analysis_seg1",  # S2: 全部分析第 1 段（30 秒結論卡 + 倉位）
    "comprehensive_analysis_seg2",  # S2: 全部分析第 2 段（完整詳細分析）
    "comprehensive_synthesis",  # 編排管線 reduce 階段（取代 seg2 monolith 的 #6-#11）
    "drawing", "event_analysis", "conditional_prob", "scenario", "smc",
    "quant_research", "calibrate", "backtest",
    "sector_analysis",
    "factor_validation_mode", "strategy_backtest_mode", "regime_analysis_mode",
    "momentum_analysis_mode", "fundamental_analysis_mode", "crypto_fundamental_mode",
    "data_sync_mode",  # 所有模式都載入，讓 LLM 隨時能建議下載
)


def assemble_system_prompt_split(modules_needed: set[str]) -> tuple[str, str]:
    """拆成 (static, dynamic) 兩段，供 prompt caching 用。

    static：CORE + 模組（永不變）→ 適合放進 cache_control 區塊
    dynamic：時間戳記（每分鐘變）→ 不能 cache，必須放後面

    Cache 可命中的關鍵：static 段內容 100% 穩定，跟掛時鐘無關。

    ★ B3：套用 _DOMINANCE 模組支配關係 — 高階模組存在時自動移除被支配的低階模組，
       避免重複內容浪費 token（例：comprehensive_analysis 會吃掉三階段 + 因子驗證等）。
    """
    from datetime import datetime

    # ★ 模組支配關係處理（避免高階+低階模組同時載入造成重複）
    modules_needed = set(modules_needed)  # 不污染呼叫端
    for dominant, dominated_list in _DOMINANCE.items():
        if dominant in modules_needed:
            for d in dominated_list:
                modules_needed.discard(d)

    parts = [_PROMPT_CORE]
    for mod_key in _MODULE_ORDER:
        if mod_key in modules_needed:
            parts.append(_PROMPT_MODULES[mod_key])
    static_part = "\n".join(parts)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    dynamic_part = f"\n【目前時間】{now_str}（台北時區 UTC+8）"

    return static_part, dynamic_part


# 保留完整版供向下相容（例如 non-streaming endpoint）
SYSTEM_PROMPT = assemble_system_prompt({
    "analysis", "drawing", "backtest", "quant_research",
    "calibrate", "event_analysis", "conditional_prob",
})

# 純靜態版（僅 CORE + 模組，無時間戳）— 給 prompt caching 用
SYSTEM_PROMPT_STATIC, _ = assemble_system_prompt_split({
    "analysis", "drawing", "backtest", "quant_research",
    "calibrate", "event_analysis", "conditional_prob",
})


# ═══════════════════════════════════════════════════════
#  Level 1 函式定義
# ═══════════════════════════════════════════════════════

FUNCTION_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_chart_data",
            "description": (
                "查詢歷史 K 線數據並取得壓縮價格摘要（含精確的期間最高/最低價及日期、月度/日度 OHLC）。"
                "用於回答使用者關於特定日期範圍的價格問題（如「某月低點是多少」「某段時間最高價」），"
                "也可用於切換幣種、時間週期、調整日期範圍。"
                "務必指定 start_date 和 end_date 以獲取精確的歷史價格數據。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "交易對，如 BTC/USDT、ETH/USDT",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["15m", "1h", "4h", "1d", "1w"],
                        "description": "時間週期",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "開始日期，格式 YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "結束日期，格式 YYYY-MM-DD",
                    },
                },
                "required": ["symbol", "timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_indicator",
            "description": "新增、移除或調整圖表上的技術指標。可調整所有指標的參數。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "update"],
                    },
                    "indicator_id": {
                        "type": "string",
                        "description": "指標 ID（共 28 個）：sma, ema, adx, vwap, ichimoku, psar, supertrend, market_structure, harmonic, rsi, bias, bb, stochrsi, atr, donchian, keltner, rel_vol, obv, vol_switch, macd, roc, trailing_stop, session, kelly, max_drawdown, fear_greed, funding, seasonal",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "指標參數，如 {\"period\": 21} 或 {\"std_dev\": 3}",
                    },
                },
                "required": ["action", "indicator_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_conditions",
            "description": "查詢滿足特定技術指標條件的時間點或時間段。例如 RSI < 30 的時間。",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string", "description": "指標 ID"},
                                "operator": {
                                    "type": "string",
                                    "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"],
                                },
                                "value": {"type": "number"},
                                "value2": {"type": "number", "description": "between 的上界"},
                                "parameters": {"type": "object", "description": "指標參數覆蓋"},
                            },
                            "required": ["indicator", "operator", "value"],
                        },
                    },
                    "logical_operator": {
                        "type": "string",
                        "enum": ["AND", "OR"],
                        "description": "多條件組合邏輯",
                    },
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["conditions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "annotate_chart",
            "description": "在圖表上添加標記。每次呼叫可添加一個或多個標記（用 annotations 陣列批量繪製）。務必設定 group_name 讓使用者能識別和管理這組標記。",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {
                        "type": "string",
                        "description": "這組標記的名稱（必填），如「上升通道」「看跌旗型」「支撐壓力位」「諧波 Gartley」。使用者可據此開關顯示。",
                    },
                    "annotations": {
                        "type": "array",
                        "description": "標記陣列，一次繪製多條線。每個元素定義一個標記。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "annotation_type": {
                                    "type": "string",
                                    "enum": ["highlight_range", "vertical_line", "horizontal_line", "text_label", "trend_line"],
                                },
                                "start_time": {"type": "string", "description": "起始時間 YYYY-MM-DD HH:MM:SS"},
                                "end_time": {"type": "string", "description": "結束時間"},
                                "price": {"type": "number"},
                                "end_price": {"type": "number", "description": "趨勢線終點價格"},
                                "text": {"type": "string"},
                                "color": {"type": "string"},
                                "line_width": {"type": "integer", "default": 2},
                                "line_style": {"type": "integer", "description": "0=實線 1=點線 2=虛線", "default": 0},
                            },
                            "required": ["annotation_type"],
                        },
                    },
                    "annotation_type": {
                        "type": "string",
                        "enum": ["highlight_range", "vertical_line", "horizontal_line", "text_label", "trend_line"],
                        "description": "單一標記模式（如果不用 annotations 陣列）",
                    },
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "price": {"type": "number"},
                    "end_price": {"type": "number"},
                    "text": {"type": "string"},
                    "color": {"type": "string", "default": "#58a6ff"},
                    "line_width": {"type": "integer", "default": 2},
                    "line_style": {"type": "integer", "default": 0},
                },
                "required": ["group_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_pattern",
            "description": "一次繪製完整技術形態（如諧波 Gartley/Bat/Butterfly/Crab、上升三角、下降楔形、頭肩頂等）。指定形態名稱和關鍵轉折點，系統自動連線和標注。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_name": {
                        "type": "string",
                        "description": "形態名稱，如 Gartley、Bat、Butterfly、Crab、Head and Shoulders、Ascending Triangle、Descending Wedge、Flag 等",
                    },
                    "points": {
                        "type": "array",
                        "description": "關鍵點位陣列（按連線順序排列），每個點包含 time 和 price。例如諧波: [X, A, B, C, D]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "點位標籤 (X, A, B, C, D 等)"},
                                "time": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
                                "price": {"type": "number"},
                            },
                            "required": ["label", "time", "price"],
                        },
                    },
                    "color": {"type": "string", "default": "#f0b90b", "description": "形態線條顏色"},
                    "line_width": {"type": "integer", "default": 2},
                    "bullish": {"type": "boolean", "description": "看漲=true, 看跌=false，影響顏色和標籤"},
                },
                "required": ["pattern_name", "points"],
            },
        },
    },
    # Level 2 函式
    {
        "type": "function",
        "function": {
            "name": "generate_analysis",
            "description": "根據當前圖表上的數據和指標，生成一份技術分析報告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "分析重點，如 趨勢、支撐壓力、波動率",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_indicators",
            "description": "根據使用者的分析目標，推薦合適的指標組合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_goal": {
                        "type": "string",
                        "description": "分析目標，如 趨勢判斷、超買超賣、波動率分析、量能確認",
                    },
                },
                "required": ["analysis_goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": (
                "用指定的技術指標進出場條件執行策略回測，返回完整績效統計（勝率、盈虧比、Sharpe、最大回撤等）。"
                "自動含滑點 0.05% + 手續費 0.1%，並自動分割樣本內/外數據檢測過擬合。\n"
                "★ 可回測維度 = 任何已註冊數值指標，涵蓋技術指標、價格(close/high/low)、成交量(volume/rel_vol)、"
                "漲跌率(roc)、日期與盤中時間(seasonal)。\n"
                "★ 指標 vs 指標：條件的右側除了填固定 value，也可用 compare_to 改成跟另一個指標逐根比較"
                "（黃金交叉、close 高於自己的均線、快慢線交叉等）。\n"
                "★ 季節性/日曆條件用 indicator='seasonal'，series 填 is_month_start/is_month_end/day_of_week/"
                "month/day_of_month/hour_of_day，operator='=='（如每月首根 K 線用 is_month_start==1、"
                "週一用 day_of_week==0、每月 15 日用 day_of_month==15）。\n"
                "【範圍確認】首次呼叫設 confirmed=false（或省略）取得本地可用範圍並向使用者確認要用全部或指定區間，"
                "確認後再以 confirmed=true 執行。start_date/end_date 僅在使用者明確指定日期時才帶入。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，如 BTC/USDT"},
                    "timeframe": {"type": "string", "description": "K 線週期"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD。僅在使用者明確指定時才帶入，否則留空使用全部數據"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD。僅在使用者明確指定時才帶入，否則留空使用全部數據"},
                    "direction": {
                        "type": "string",
                        "enum": ["long", "short"],
                        "description": "交易方向，預設 long",
                    },
                    "entry_conditions": {
                        "type": "array",
                        "description": "進場條件陣列",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string", "description": "指標 ID（如 rsi, macd, bb；季節性用 seasonal；原始價量用 close/high/low/volume）"},
                                "operator": {
                                    "type": "string",
                                    "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"],
                                },
                                "value": {"type": "number", "description": "比較值（固定數值）。若用 compare_to 改跟另一指標比，這裡填 0 即可"},
                                "value2": {"type": "number", "description": "between 運算的上界"},
                                "parameters": {"type": "object", "description": "指標參數覆蓋"},
                                "compare_to": {"type": "object", "description": "【指標 vs 指標】改成跟另一個指標逐根比較（而非固定 value）：{indicator, series?, parameters?, mult?, offset?}，右值=該指標*mult+offset。例：黃金交叉 → indicator='sma'(period20) + operator='cross_above' + compare_to={'indicator':'sma','parameters':{'period':60}}；收盤高於均線5%乖離 → indicator='close' + operator='>' + compare_to={'indicator':'sma','parameters':{'period':20},'mult':1.05}。between 不支援 compare_to。"},
                            },
                            "required": ["indicator", "operator", "value"],
                        },
                    },
                    "exit_conditions": {
                        "type": "array",
                        "description": "出場條件陣列",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string"},
                                "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"]},
                                "value": {"type": "number"},
                                "value2": {"type": "number"},
                                "parameters": {"type": "object"},
                                "compare_to": {"type": "object", "description": "改跟另一指標比較（而非固定 value）：{indicator, series?, parameters?, mult?, offset?}。見 entry_conditions 的 compare_to 說明。"},
                            },
                            "required": ["indicator", "operator", "value"],
                        },
                    },
                    "stop_loss_pct": {"type": "number", "description": "止損百分比（小數），如 0.02 = 2%"},
                    "take_profit_pct": {"type": "number", "description": "止盈百分比（小數），如 0.05 = 5%"},
                    "initial_capital": {"type": "number", "description": "初始資金 USDT，預設 10000"},
                    "leverage": {"type": "number", "description": "槓桿倍數（如 5 = 五倍合約），預設 1（無槓桿）。盈虧按槓桿放大，含爆倉模擬。"},
                    "confirmed": {"type": "boolean", "description": "回測範圍確認旗標。首次呼叫設 false（或省略）→ 回傳本地可用資料範圍(needs_confirmation)供你向使用者確認；使用者確認後再以 confirmed=true 重新呼叫才會真正執行回測。"},
                },
                "required": ["entry_conditions", "exit_conditions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_strategies",
            "description": (
                "同時執行 2-5 個不同策略的回測並比較績效（勝率、Sharpe、報酬、回撤）。結果按 Sharpe 排名。\n"
                "★ 條件的 series 名稱必須用以下精確值（大小寫需正確，寫錯=該條件永不成立=0 交易）：\n"
                "  - macd：series 用 'MACD' / 'Signal' / 'Histogram'\n"
                "  - adx：series 用 'ADX' / '+DI' / '-DI'\n"
                "  - stochrsi：series 用 '%K' / '%D'\n"
                "  - bb：series 用 'BB_Upper' / 'BB_Middle' / 'BB_Lower' / 'BB_Width'\n"
                "  - rsi/roc/atr/obv/bias：單一序列，可省略 series\n"
                "  - supertrend：**只有 'Supertrend' 價格線、無方向序列**。要判斷多空方向請用 "
                "close 與 Supertrend 比較（如 entry 用 indicator='close' 或改用 ADX +DI/-DI），"
                "**禁止使用不存在的 'Supertrend_Direction'**。\n"
                "★ 指標 vs 指標：條件右側可用 compare_to 跟另一指標比較（黃金交叉、close 高於均線等）。\n"
                "★ 進場條件勿過度嚴苛（多條件 AND 易 0 交易），優先用單一明確條件 + SL/TP。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，如 BTC/USDT"},
                    "timeframe": {"type": "string", "description": "K 線週期"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD。僅在使用者明確指定時才帶入，否則留空使用全部數據"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD。僅在使用者明確指定時才帶入，否則留空使用全部數據"},
                    "strategies": {
                        "type": "array",
                        "description": "策略陣列（2-5 個），每個策略包含名稱、進出場條件",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "策略名稱，如「RSI 超賣反彈」"},
                                "direction": {"type": "string", "enum": ["long", "short"]},
                                "entry_conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "indicator": {"type": "string"},
                                            "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"]},
                                            "value": {"type": "number"},
                                            "value2": {"type": "number"},
                                            "parameters": {"type": "object"},
                                            "compare_to": {"type": "object", "description": "改跟另一指標比較（而非固定 value）：{indicator, series?, parameters?, mult?, offset?}，右值=指標*mult+offset。例:黃金交叉 sma20 cross_above compare_to=sma60。between 不支援。"},
                                        },
                                        "required": ["indicator", "operator", "value"],
                                    },
                                },
                                "exit_conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "indicator": {"type": "string"},
                                            "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"]},
                                            "value": {"type": "number"},
                                            "value2": {"type": "number"},
                                            "parameters": {"type": "object"},
                                            "compare_to": {"type": "object", "description": "改跟另一指標比較（而非固定 value）：{indicator, series?, parameters?, mult?, offset?}，右值=指標*mult+offset。例:黃金交叉 sma20 cross_above compare_to=sma60。between 不支援。"},
                                        },
                                        "required": ["indicator", "operator", "value"],
                                    },
                                },
                                "stop_loss_pct": {"type": "number"},
                                "take_profit_pct": {"type": "number"},
                            },
                            "required": ["name", "entry_conditions", "exit_conditions"],
                        },
                    },
                    "confirmed": {"type": "boolean", "description": "回測範圍確認旗標。首次呼叫設 false（或省略）→ 回傳本地可用資料範圍(needs_confirmation)供你向使用者確認；確認後再以 confirmed=true 執行。"},
                },
                "required": ["strategies"],
            },
        },
    },
    # ── 事件回溯統計分析 ──
    {
        "type": "function",
        "function": {
            "name": "analyze_event_patterns",
            "description": (
                "事件驅動回溯統計：找出歷史上符合特定條件的事件（如大漲超過N%、爆量等），"
                "並統計事件發生前若干根 K 線的技術指標共通性。"
                "適用場景：「大漲10%之前通常有什麼指標特徵」「暴跌前RSI通常在什麼範圍」。"
                "後端使用 NumPy 完成計算，不消耗額外 token。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": ["price_surge", "price_drop", "volume_spike", "volatility_expansion"],
                        "description": "事件類型：price_surge=漲幅≥threshold%, price_drop=跌幅≥threshold%, volume_spike=量能≥threshold倍(vs 20MA), volatility_expansion=ATR≥threshold倍(vs 20MA)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "事件閾值。price_surge/drop=%（如10代表10%）; volume_spike/volatility_expansion=倍數（如3代表3倍）",
                        "default": 10,
                    },
                    "n_bars": {
                        "type": "integer",
                        "description": "計算漲跌幅時的 K 線數（如 n_bars=1 代表單根 K 線漲跌幅，n_bars=3 代表連續3根的累計漲跌幅）",
                        "default": 1,
                    },
                    "lookback_bars": {
                        "type": "integer",
                        "description": "事件發生前回看多少根 K 線來統計指標特徵",
                        "default": 5,
                    },
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要分析的指標 ID 列表，如 ['rsi','macd','adx','bb','rel_vol','atr','obv','stoch_rsi']",
                        "default": ["rsi", "macd", "adx", "rel_vol", "bb", "atr"],
                    },
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                },
                "required": ["event_type"],
            },
        },
    },
    # ── 布林壓縮 → 首次穿越軌道 time-to-event 統計（v153）──
    {
        "type": "function",
        "function": {
            "name": "analyze_squeeze_breakout",
            "description": (
                "布林壓縮 time-to-event 統計：找出歷史上「進入壓縮狀態」（%B < pctb_max 且 "
                "BB 帶寬百分位 < width_pctile_max）的每一次事件，統計進入後第幾根 K 線首次收盤"
                "突破上軌 / 跌破下軌，回傳方向占比與天數分布（median/p25/p75）。"
                "適用場景：「進入 %B<10 且帶寬百分位<10 後，通常幾天會突破或跌破布林帶」。"
                "口徑：BB(20, ±2σ)，帶寬百分位 = 帶寬的 rolling 120 根排名，與跨日追蹤/vol_squeeze 一致。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pctb_max": {
                        "type": "number",
                        "description": "%B 上限（%）。%B = (close-下軌)/(上軌-下軌)×100，<10 代表貼近/跌破下軌",
                        "default": 10,
                    },
                    "width_pctile_max": {
                        "type": "number",
                        "description": "BB 帶寬百分位上限（%），<10 代表處於歷史最壓縮的 10% 區間",
                        "default": 10,
                    },
                    "horizon_bars": {
                        "type": "integer",
                        "description": "進入後最多往前看幾根 K 線等待穿越（超過視為「未表態」）",
                        "default": 20,
                    },
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                },
                "required": [],
            },
        },
    },
    # ── 完整量化研究 ──
    {
        "type": "function",
        "function": {
            "name": "run_quant_research",
            "description": (
                "執行完整量化研究流程：因子 IC 分析 → 因子相關性 → 策略回測（含 Sortino/Expectancy）"
                "→ Monte Carlo 模擬 → Walk Forward 驗證 → 動態倉位建議。"
                "最終輸出策略是否具備 Alpha、穩定度評分、建議槓桿和倉位。"
                "適用場景：使用者要求「完整量化研究」「策略是否可行」「策略穩定性檢測」。"
                "【重要】start_date / end_date 僅在使用者明確指定日期時才帶入，"
                "否則一律留空以使用全部本地數據，避免數據不足導致分析失敗。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要進行因子分析的指標 ID，如 ['rsi','macd','adx','bb','obv','supertrend']",
                    },
                    "entry_conditions": {
                        "type": "array",
                        "description": "進場條件（同 run_backtest 格式）。如不提供則只做因子分析。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string"},
                                "operator": {"type": "string"},
                                "value": {"type": "number"},
                            },
                        },
                    },
                    "exit_conditions": {
                        "type": "array",
                        "description": "出場條件（同 run_backtest 格式）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string"},
                                "operator": {"type": "string"},
                                "value": {"type": "number"},
                            },
                        },
                    },
                    "direction": {"type": "string", "enum": ["long", "short"], "default": "long"},
                    "stop_loss_pct": {"type": "number"},
                    "take_profit_pct": {"type": "number"},
                    "leverage": {"type": "number", "description": "槓桿倍數（如 5 = 五倍合約），預設 1"},
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD。僅在使用者明確指定時才帶入，否則留空使用全部數據"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD。僅在使用者明確指定時才帶入，否則留空使用全部數據"},
                },
            },
        },
    },
    # ── 指標參數校準 ──
    {
        "type": "function",
        "function": {
            "name": "optimize_indicator_params",
            "description": (
                "掃描歷史數據，找出每個技術指標最適合此標的的閾值參數。"
                "輸出穩健區間（非單一值）+ 可信度等級（★~★★★）。"
                "結果自動儲存，之後分析時會自動參考。"
                "適用場景：使用者問「RSI 多少算超買」「幫我校準指標」「這個幣的 ADX 門檻是多少」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要校準的指標 ID（如 ['rsi','adx','macd','stochrsi']），留空校準全部可校準指標",
                    },
                    "forward_bars": {
                        "type": "integer",
                        "description": "預測未來幾根 K 線的報酬率（預設 5）",
                        "default": 5,
                    },
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                },
            },
        },
    },
    # ── 條件機率掃描 ──
    {
        "type": "function",
        "function": {
            "name": "scan_conditional_probability",
            "description": (
                "條件機率掃描 + 命中特徵分析：掃描指定技術指標的所有數值區間，統計每個區間在後續 N 根 K 線內"
                "最高漲幅/最大跌幅超過 X% 的條件機率，找出機率最高的區間，並分析命中 K 線前 N 根的共同特徵。"
                "還會計算當前走勢與歷史成功模式的多維度相似度（技術指標、趨勢方向、量能、波動率、價格位置）。"
                "適用場景：「RSI 在多少時後續上漲 3% 機率最高」「什麼條件下勝率最高」「往前看 10 根預測後 12 根漲 5%」。\n"
                "★ 日曆/季節性問題（『每月 1 日漲跌機率』『星期幾表現』『月份季節性』『幾點漲跌』）"
                "請把 indicators 設為 ['seasonal']，掃描會自動對離散日曆值逐值分組（非連續分箱），"
                "各組附 Wilson CI 與 baseline lift。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要掃描的指標 ID，如 ['rsi','macd','adx','bb','stochrsi']；日曆/季節性用 ['seasonal']",
                    },
                    "lookback_bars": {
                        "type": "integer",
                        "description": "回推分析前幾根 K 線的共同特徵（預設 7）。使用者說「往前看 10 根」時填入",
                        "default": 7,
                    },
                    "forward_bars": {
                        "type": "integer",
                        "description": "觀察後續幾根 K 線的最高漲/跌幅（預設 6）",
                        "default": 6,
                    },
                    "target_pct": {
                        "type": "number",
                        "description": "目標漲/跌幅百分比（如 3 代表 3%）",
                        "default": 3.0,
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "目標方向：up=上漲, down=下跌",
                        "default": "up",
                    },
                    "n_bins": {
                        "type": "integer",
                        "description": "將指標數值範圍分成幾個區間（預設 10）",
                        "default": 10,
                    },
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                    "confirmed": {"type": "boolean", "description": "範圍確認旗標。首次呼叫設 false（或省略）→ 回傳本地可用資料範圍(needs_confirmation)供你向使用者確認；確認後再以 confirmed=true 執行掃描。"},
                },
                "required": ["indicators"],
            },
        },
    },

    # ── 情境預測 ──
    {
        "type": "function",
        "function": {
            "name": "generate_scenarios",
            "description": (
                "三大情境預測：整合 ML 模型、技術指標、歷史相似度、市場結構，"
                "產出三個最有可能的價格情境（看漲/中性/看跌），每個附帶統計計算的機率百分比。"
                "適用場景：「未來走勢預測」「接下來會怎樣」「三種可能」「情境分析」。"
                "所有機率由後端統計計算，非 LLM 生成。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "forward_bars": {
                        "type": "integer",
                        "description": "預測有效期（K 線根數，預設 5）",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },

    # ── 分批進場價位計算（v99）──
    {
        "type": "function",
        "function": {
            "name": "compute_laddered_entries",
            "description": (
                "計算分批進場價位（後端從 BB / EMA / Donchian / ATR 取，禁止 LLM 自行推算）。"
                "依當前 regime 自動決定配比（trending → 50/30/20 金字塔加碼；ranging → 25/35/40 倒金字塔接刀；high_vol → 33/33/34 對稱）。"
                "輸出多空各 N 檔具體價位 + 加權均價 + SL/TP（基於 ATR），給「全部分析」結論段引用。"
                "regime confidence < 0.5 時自動跳過分批（用既有 regimeWarning 機制）。"
                "適用場景：「給我具體進場價」「分批進場建議」「ladder 進場」「scale-in 策略」「全部分析」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，留空使用當前圖表"},
                    "timeframe": {"type": "string", "description": "時間框架（如 1h, 4h, 1d）"},
                    "direction": {
                        "type": "string",
                        "enum": ["long", "short", "both"],
                        "description": "計算方向（預設 both — 多空都算）",
                        "default": "both",
                    },
                    "n_tranches": {
                        "type": "integer",
                        "description": "分檔數（預設 3，本版固定 3）",
                        "default": 3,
                    },
                },
                "required": [],
            },
        },
    },

    # ── SMC 訂單流結構分析 ──
    {
        "type": "function",
        "function": {
            "name": "detect_smc_structure",
            "description": (
                "SMC 訂單流結構分析：偵測 BOS/CHoCH、Fair Value Gap (fvg_zones)、"
                "流動性 Sweep (sweep_events，BSL/SSL)、Order Block (order_blocks，bullish/bearish 機構訂單塊)、"
                "Premium/Discount 區域 (premium_discount + fib_50 線)、多時區共振（MTF Alignment），"
                "計算交易建議（BUY/SELL/WAIT）和信心分數。"
                "回傳完整 SMC 四要素：BSL/SSL (sweep_events) + OB (order_blocks) + FVG (fvg_zones) + Premium-Discount (premium_discount, fib_50)。"
                "適用場景：「訂單流分析」「SMC 結構」「聰明錢」「機構行為」「BOS/CHoCH」「流動性」「訂單塊」「公允價值缺口」「溢價折價區」。"
                "所有結構判定由後端精確計算，非 LLM 推算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間框架，如 1h, 4h, 1d"},
                    "htf": {"type": "string", "description": "高時區框架（可選，預設自動推斷）"},
                    "lookback": {
                        "type": "integer",
                        "description": "回溯 K 線數量（預設 120）",
                        "default": 120,
                    },
                },
                "required": [],
            },
        },
    },
    # ─── 因子 IC 校準 ───
    {
        "type": "function",
        "function": {
            "name": "compute_factor_ic",
            "description": (
                "對 ADX/RSI/MACD/ATR 等技術因子計算 Spearman IC (Information Coefficient)，"
                "驗證因子對未來 N 期收益的預測力。回傳每個因子在 forward_periods (例如 1/5/10 K 線) "
                "的 ic、pval、sample_n、significant 標記、strength 分級（強/中/弱/可忽略）。"
                "適用場景：「因子預測力」「IC 校準」「指標有效性」「歷史 IC」「因子驗證」「方向修正」。"
                "用於分析報告的「量化因子校準」章節。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對（留空使用當前）"},
                    "timeframe": {"type": "string", "description": "時間框架"},
                    "factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "因子清單，預設 [adx, rsi, macd, atr]",
                    },
                    "forward_periods": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "未來 N 期，預設 [1, 5, 10]",
                    },
                    "lookback": {
                        "type": "integer",
                        "description": "回溯 K 線數量（預設 250）",
                        "default": 250,
                    },
                },
                "required": [],
            },
        },
    },
    # ─── 族群/概念股分析 ───
    {
        "type": "function",
        "function": {
            "name": "analyze_sector",
            "description": (
                "分析台股特定產業族群/概念股的整體趨勢。"
                "合成族群等權指數 → 技術分析（RSI/MACD/ADX/均線）→ Breadth（站上均線比例）"
                "→ 個股相對強弱排名。"
                "支援族群（含子族群）：半導體（晶圓代工/IC設計/封測/記憶體/矽晶圓）、"
                "AI概念股（AI伺服器/AI晶片）、汽車電子（電動車/車用零件）、"
                "被動元件、電源散熱、網通、金融、電子代工、航運、鋼鐵、生技醫療、綠能、"
                "電信、食品、營建、觀光餐飲、面板、PCB、蘋果供應鏈、伺服器概念。"
                "也支援別名（如「ai」「蘋概股」「mlcc」「ev」「封裝測試」）。"
                "適用場景：「半導體族群趨勢」「AI概念股分析」「封測族群表現」「哪個族群最強」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector_name": {
                        "type": "string",
                        "description": "族群名稱，如「半導體」「AI概念股」「封測」「被動元件」「電動車」",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1d", "1w"],
                        "description": "時間週期（預設 1d）",
                    },
                    "lookback_days": {
                        "type": "number",
                        "description": "分析回看天數（預設 120）",
                    },
                },
                "required": ["sector_name"],
            },
        },
    },
    # ─── 列出可用族群 ───
    {
        "type": "function",
        "function": {
            "name": "list_sectors",
            "description": (
                "列出所有可分析的台股產業族群/概念股清單。"
                "當使用者詢問「有哪些族群可以分析」「列出產業」「支援什麼概念股」時呼叫。"
                "回傳每個族群的名稱、成分股數量、來源（內建/自訂）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    # ─── 基本面分析 ───
    {
        "type": "function",
        "function": {
            "name": "analyze_fundamentals",
            "description": (
                "台股基本面分析：月營收趨勢（MoM/YoY/連續成長）、三大法人買賣超（外資/投信/自營商）、"
                "外資持股比例、財報摘要（本益比/EPS/殖利率）。僅限台股。"
                "適用場景：「台積電基本面」「國巨營收」「法人買賣超」「本益比多少」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "台股代碼，如 2330/TWD。留空使用當前標的",
                    },
                },
                "required": [],
            },
        },
    },
    # ─── 加密基本面分析 ───
    {
        "type": "function",
        "function": {
            "name": "analyze_crypto_fundamentals",
            "description": (
                "加密貨幣協議基本面（即時、免費 API、非台股）：代幣經濟學（流通/總/最大供給、市值、FDV、稀釋、ATH）、"
                "生態活躍度（DeFiLlama TVL、GitHub 開發活躍度 commit/stars/issues、社群數據）、官方連結。"
                "回傳含 data_status 與 fetched_at 時間戳；路線圖無即時 API 故附免責。"
                "適用場景：「BTC 代幣經濟學」「ETH 協議基本面」「ADA 生態活躍度」「加密基本面 vs 技術面」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "加密交易對，如 BTC/USDT。留空使用當前標的",
                    },
                },
                "required": [],
            },
        },
    },
    # ─── 數據下載 ───
    {
        "type": "function",
        "function": {
            "name": "sync_symbol_data",
            "description": (
                "下載指定標的的歷史 K 線數據到本地。"
                "在呼叫此函式前，必須先跟用戶確認：(1) 標的代碼 (2) 時間框架 (3) 起始日期。"
                "用戶明確確認後才呼叫。如果用戶已在對話中指定了日期，可以直接呼叫。"
                "適用場景：「下載國巨的數據」「同步 BTC 資料」「抓取 2330 從 2020 年開始」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "交易對，如 BTC/USDT、2330/TWD。台股用 {代號}/TWD 格式",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["15m", "1h", "4h", "1d", "1w"],
                        "description": "時間框架（台股僅支援 1d/1w），預設 1d",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "起始日期 YYYY-MM-DD。必須由用戶確認，不可自行決定",
                    },
                },
                "required": ["symbol", "start_date"],
            },
        },
    },
    # ─── 族群批次下載 ───
    {
        "type": "function",
        "function": {
            "name": "sync_sector_data",
            "description": (
                "批次下載整個台股族群/概念股的所有成分股 K 線數據。"
                "例如「下載所有被動元件的數據」→ 自動展開 2327/2492/2428/2456/3026 並逐一下載。"
                "在呼叫前必須先跟用戶確認起始日期。"
                "支援 29 個族群 + 別名（如「ai」「半導體」「蘋概股」）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector_name": {
                        "type": "string",
                        "description": "族群名稱，如「被動元件」「半導體」「AI概念股」",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1d", "1w"],
                        "description": "時間框架，預設 1d",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "起始日期 YYYY-MM-DD，必須由用戶確認",
                    },
                },
                "required": ["sector_name", "start_date"],
            },
        },
    },
    # ─── 動能分析 ───
    {
        "type": "function",
        "function": {
            "name": "analyze_momentum",
            "description": (
                "動能交易分析：多週期動量因子、動量加速/減速偵測、相對動量（vs BTC）、"
                "動量反轉偵測（RSI/MACD 背離）、動量策略回測（追強/反轉/趨勢三種）。"
                "適用場景：「動能分析」「動量排名」「追強勢」「動量反轉」「相對強弱」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間框架"},
                },
                "required": [],
            },
        },
    },
]
