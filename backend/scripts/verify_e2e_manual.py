"""LLM 覆核層手動 e2e 驗收（需 LLM，會實際打一次覆核模型）

用法（backend/ 下）：
    .venv/bin/python scripts/verify_e2e_manual.py --provider claude_subscription --model claude-opus-4-8
    .venv/bin/python scripts/verify_e2e_manual.py --provider gemini --model gemini-2.0-flash --api-key XXX

四個案例：
  (a) 數字錯 + 方向矛盾：digest RSI=42.1，回答寫「RSI 71 超買建議做空」
  (b) 編造：回答宣稱系統算出「鏈上大戶淨流入 3.2 億美元」，digest 無此類別
  (c) 邏輯矛盾：前段「勝率 62% 具優勢」後段「勝率不足五成故觀望」
  (d) 完全正確（誤報煙霧測試，應 pass）
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm.verifier import build_data_digest, pick_verifier_model, verify_answer  # noqa: E402

_CHART_STATE = {
    "symbol": "BTCUSDT", "timeframe": "4h", "currentPrice": 65000,
    "currentRegime": {"regime": "ranging"},
    "indicatorValues": {"RSI": 42.1, "MACD": -15.2, "ADX": 18.5},
    "recent_accuracy": {"win_rate_30d": 62.0},
}

_PAD = "本段為篇幅填充說明：以下分析基於系統注入之即時數據，區間結構觀察與風險控管原則詳見完整報告。" * 12

_CASES = {
    "(a) 數字錯+方向矛盾": (
        f"技術面分析：目前 RSI 為 71，已進入超買區，動能過熱，建議做空。{_PAD}",
        {"number", "direction"},
    ),
    "(b) 編造數據": (
        f"根據系統鏈上數據計算，大戶淨流入 3.2 億美元，籌碼面強勁支撐。RSI 42.1 中性偏弱。{_PAD}",
        {"fabricated"},
    ),
    "(c) 邏輯矛盾": (
        f"系統 30 日勝率 62%，具統計優勢，訊號可信度高。綜合結論：因勝率不足五成，建議觀望不進場。RSI 42.1。{_PAD}",
        {"logic"},
    ),
    "(d) 完全正確（應 pass）": (
        f"RSI 為 42.1，中性偏弱；ADX 18.5 顯示趨勢不明確，屬盤整市。系統 30 日勝率 62%。綜合建議：區間操作，等待突破確認。{_PAD}",
        set(),
    ),
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="claude_subscription")
    ap.add_argument("--model", default="claude-opus-4-8", help="主模型（覆核模型自動降階）")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    verifier_model = pick_verifier_model(args.provider, args.model)
    print(f"主模型 {args.model} → 覆核模型 {verifier_model}\n")
    print(f"digest:\n{build_data_digest(_CHART_STATE, None)}\n{'=' * 60}")

    passed = 0
    for name, (answer, expected_types) in _CASES.items():
        result = await verify_answer(
            answer, _CHART_STATE, None,
            args.provider, args.api_key, args.base_url, args.model,
        )
        if result is None:
            print(f"{name}: ❌ 覆核呼叫失敗（None）")
            continue
        got_types = {i["type"] for i in result["issues"]}
        if expected_types:
            ok = bool(got_types & expected_types)
            verdict = "✅" if ok else "❌"
            print(f"{name}: {verdict} 期望抓到 {expected_types}，實際 {got_types or '(無)'}")
        else:
            ok = result["status"] == "pass"
            verdict = "✅" if ok else f"❌（誤報：{got_types}）"
            print(f"{name}: {verdict} 期望 pass，實際 {result['status']}")
        for i in result["issues"]:
            print(f"    [{i['type']}/{i['severity']}] 「{i['quote']}」— {i['why']}")
        passed += ok

    print(f"\n{'=' * 60}\n{passed}/{len(_CASES)} 案例符合預期")


if __name__ == "__main__":
    asyncio.run(main())
