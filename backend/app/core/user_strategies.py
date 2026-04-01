"""使用者自訂分析策略庫 — 持久化儲存與注入 LLM context

儲存路徑: data/db/user_strategies.json
結構: { strategies: [ {id, title, content, enabled, created_at, updated_at} ] }
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.core.config.settings import settings

_STRATEGIES_FILE = settings.db_path / "user_strategies.json"


def _load_all() -> list[dict]:
    if not _STRATEGIES_FILE.exists():
        return []
    try:
        data = json.loads(_STRATEGIES_FILE.read_text(encoding="utf-8"))
        return data.get("strategies", [])
    except Exception as e:
        logger.warning(f"讀取自訂策略檔案失敗: {e}")
        return []


def _save_all(strategies: list[dict]) -> None:
    _STRATEGIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STRATEGIES_FILE.write_text(
        json.dumps({"strategies": strategies}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_strategies() -> list[dict]:
    return _load_all()


def get_strategy(strategy_id: str) -> dict | None:
    for s in _load_all():
        if s["id"] == strategy_id:
            return s
    return None


def add_strategy(
    title: str,
    content: str,
    enabled: bool = True,
    auto_inject: bool = False,
    trigger: str = "manual",
    prompt_module: str | None = None,
) -> dict:
    strategies = _load_all()
    now = datetime.utcnow().isoformat()
    new = {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "content": content,
        "enabled": enabled,
        "auto_inject": auto_inject,
        "trigger": trigger,           # "analysis" / "all" / "manual"
        "prompt_module": prompt_module,  # 對應 _PROMPT_MODULES 的 key
        "created_at": now,
        "updated_at": now,
    }
    strategies.append(new)
    _save_all(strategies)
    logger.info(f"新增自訂策略: {title} ({new['id']})")
    return new


def update_strategy(
    strategy_id: str,
    title: str | None = None,
    content: str | None = None,
    enabled: bool | None = None,
    auto_inject: bool | None = None,
    trigger: str | None = None,
    prompt_module: str | None = None,
) -> dict | None:
    strategies = _load_all()
    for s in strategies:
        if s["id"] == strategy_id:
            if title is not None:
                s["title"] = title
            if content is not None:
                s["content"] = content
            if enabled is not None:
                s["enabled"] = enabled
            if auto_inject is not None:
                s["auto_inject"] = auto_inject
            if trigger is not None:
                s["trigger"] = trigger
            if prompt_module is not None:
                s["prompt_module"] = prompt_module
            s["updated_at"] = datetime.utcnow().isoformat()
            _save_all(strategies)
            logger.info(f"更新自訂策略: {s['title']} ({strategy_id})")
            return s
    return None


def delete_strategy(strategy_id: str) -> bool:
    strategies = _load_all()
    before = len(strategies)
    strategies = [s for s in strategies if s["id"] != strategy_id]
    if len(strategies) < before:
        _save_all(strategies)
        logger.info(f"刪除自訂策略: {strategy_id}")
        return True
    return False


def seed_default_strategies() -> None:
    """如果策略庫為空，預先載入預設的分析框架"""
    existing = _load_all()
    if existing:
        return

    logger.info("策略庫為空，載入預設的 30 層機構交易分析框架")
    add_strategy(
        title="30 層機構交易 + 量化交易分析模型",
        content=_DEFAULT_30_LAYER_FRAMEWORK,
        enabled=True,
    )


_DEFAULT_30_LAYER_FRAMEWORK = """\
一、宏觀市場層（Macro Layer）
1. Market Regime（市場環境）
   - 判斷市場屬於：趨勢市場(Trend) / 盤整市場(Range) / 高波動市場(Volatility Expansion)
   - 分析：ADX、ATR → 決定適合的交易策略
2. Macro Trend（宏觀趨勢）
   - 分析週線趨勢 + 日線趨勢 → 判斷長期市場方向
3. Multi Timeframe Trend（多時間框架趨勢）
   - 分析週線/日線/4H/1H → 判斷趨勢是否一致

二、市場結構層（Structure Layer）
4. Market Structure（市場結構）
   - 分析 Higher High / Higher Low / Lower High / Lower Low → 判斷多頭或空頭
5. Break of Structure (BOS)
   - 分析是否出現結構破壞
6. Market Structure Shift (MSS)
   - 分析是否出現趨勢轉折

三、流動性層（Liquidity Layer）
7. Liquidity Zones（流動性區）
   - 找出前高/前低/Equal High/Equal Low
8. Liquidity Sweep（流動性掃單）
   - 分析市場是否可能掃流動性、是否接近停損聚集區
9. Stop Cluster（停損聚集區）
   - 分析停損密集區

四、機構交易層（Institution Layer）
10. Order Block
    - 分析 Bullish Order Block / Bearish Order Block
11. Fair Value Gap
    - 找出不平衡區（FVG）
12. Supply & Demand Zone
    - 分析供給區 / 需求區
13. Institutional Cost（機構成本）
    - 分析 VWAP + 機構成本區

五、價格行為層（Price Action）
14. K線結構
    - 分析 Pin Bar / Engulfing / Breakout Candle
15. Breakout Analysis（突破分析）
    - 判斷真突破 vs 假突破
16. Trend Pattern（趨勢延續形態）
    - 分析 Bull Flag / Bear Flag / Triangle

六、成交量層（Volume）
17. Volume Trend（成交量趨勢）
18. Volume Spike（成交量異常）
19. OBV / Volume Ratio（成交量累積）

七、動能層（Momentum）
20. RSI Momentum
    - RSI > 50 / < 50、背離分析
21. MACD Momentum
    - 動能增強或減弱
22. Momentum Shift（動能變化）

八、波動層（Volatility）
23. ATR Volatility（平均波動）
24. Volatility Expansion（波動擴張）

九、衍生市場層（Derivatives）
25. Funding Rate（資金費率）
26. Liquidation Map（清算區）— 多空清算區分析
27. Open Interest（未平倉量）

十、量化層（Quant Layer）
28. Moving Average Trend（統計趨勢）
29. Market Sentiment（市場情緒）— 恐懼貪婪指數

十一、交易策略層（Trading Layer）
30. Trading Strategy — 綜合以上 29 層分析，提供：
    - 多單進場 / 空單進場
    - 止損 / 止盈
    - Risk Reward 風險報酬比
    - 勝率評估
"""


def get_enabled_strategies_prompt() -> str:
    """產生注入 LLM 的自訂策略 prompt（僅啟用的策略）"""
    strategies = [s for s in _load_all() if s.get("enabled", True)]
    if not strategies:
        return ""

    lines = [
        "【使用者自訂分析策略（僅供參考，不強制限制）】",
        "以下是使用者提供的分析方法論。分析時請優先參考這些方法，",
        "但不侷限於此 — 當你判斷有更適合當前情境的分析角度時，應自行擴展或結合其他方法。",
        "如果使用了其中某個策略，在回覆中簡要提及。如果你認為有更好的方法，也要一併提出。",
        "",
    ]

    for i, s in enumerate(strategies, 1):
        lines.append(f"═══ 策略 {i}：{s['title']} ═══")
        lines.append(s["content"])
        lines.append("")

    return "\n".join(lines)


def get_auto_inject_modules(intents: set[str]) -> list[str]:
    """根據當前意圖，回傳需要自動注入的 prompt 模組名

    策略需同時滿足 enabled=True 且 auto_inject=True。
    trigger 決定何時注入：
    - "all": 所有對話都注入
    - "analysis" / "smc" / ...: 僅當偵測到對應意圖時注入
    - "manual": 不自動注入（僅在用戶明確提到關鍵字時由意圖系統觸發）
    """
    strategies = [s for s in _load_all() if s.get("enabled") and s.get("auto_inject")]
    modules: list[str] = []
    for s in strategies:
        trigger = s.get("trigger", "manual")
        module = s.get("prompt_module")
        if not module:
            continue
        if trigger == "manual":
            continue
        if trigger == "all":
            modules.append(module)
        elif trigger in intents:
            modules.append(module)
    return modules
