"""主入口：跑單次完整實驗 + Walk-forward OOS 評估。

執行：
    cd backend/research/lstm_sca_arima_garch
    python3 run_experiment.py

預設用 BTC/USDT 1d 資料。修改 SYMBOL / TIMEFRAME 換標的。
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from arima_trend import fit_predict_walk_forward
from data_loader import load
from decomposer import decompose
from evaluate import evaluate_all
from hybrid_pipeline import run_hybrid_forecast


# ─── 實驗設定 ───────────────────────────────────
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
N_FORECAST = 5
STL_PERIOD = 20
WALK_FORWARD_TRAIN_SIZE = 200  # 用前 200 根訓練 ARIMA 做 walk-forward baseline
# ────────────────────────────────────────────────


def main():
    print(f"\n{'═' * 60}")
    print(f"  LSTM-SCA-ARIMA-GARCH PoC 實驗")
    print(f"  Symbol: {SYMBOL}  Timeframe: {TIMEFRAME}")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 60}\n")

    # 建立結果資料夾
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / ts
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 結果輸出：{results_dir}\n")

    # ─── Step 1: 跑混合預測（PoC：固定超參，無 SCA）───
    print("[1/3] 執行混合預測...")
    hybrid = run_hybrid_forecast(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        n_forecast=N_FORECAST,
        stl_period=STL_PERIOD,
    )
    if "error" in hybrid:
        print(f"❌ 失敗：{hybrid['error']}")
        return
    print(f"  最後收盤：{hybrid['last_close']:.2f}")
    print(f"  混合預測（未來 {N_FORECAST} 步）：")
    for i, p in enumerate(hybrid["final_close_forecast"], 1):
        print(f"    +{i}: {p:.2f}")

    # ─── Step 2: ARIMA Walk-forward Baseline ───
    print(f"\n[2/3] ARIMA-only Baseline（walk-forward, train_size={WALK_FORWARD_TRAIN_SIZE}）...")
    df = load(SYMBOL, TIMEFRAME)
    if df is None or len(df) < WALK_FORWARD_TRAIN_SIZE + 50:
        print(f"  ⚠ 資料不足做 walk-forward，跳過 baseline")
        baseline_metrics = None
    else:
        decomp = decompose(df["close"].reset_index(drop=True), period=STL_PERIOD)
        wf_predictions = fit_predict_walk_forward(
            decomp.trend, train_size=WALK_FORWARD_TRAIN_SIZE,
        )
        actual_trend = decomp.trend.iloc[WALK_FORWARD_TRAIN_SIZE:]
        baseline_metrics = evaluate_all(actual_trend, wf_predictions)
        print(f"  RMSE: {baseline_metrics['rmse']}")
        print(f"  MAE: {baseline_metrics['mae']}")
        print(f"  方向命中率: {baseline_metrics['directional_accuracy_pct']}%")
        print(f"  樣本數: {baseline_metrics['n_samples']}")

    # ─── Step 3: 寫結果 ───
    print(f"\n[3/3] 寫入結果到 {results_dir}/")
    config = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "n_forecast": N_FORECAST,
        "stl_period": STL_PERIOD,
        "walk_forward_train_size": WALK_FORWARD_TRAIN_SIZE,
        "ran_at": datetime.now().isoformat(),
    }
    with (results_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "hybrid_forecast": {
            "last_close": hybrid["last_close"],
            "final_close_forecast": hybrid["final_close_forecast"],
            "trend_forecast": hybrid["trend_forecast"],
            "seasonal_forecast": hybrid["seasonal_forecast"],
            "residual_forecast": hybrid.get("residual_forecast"),
            "garch_cond_std_forecast": hybrid.get("garch_cond_std_forecast"),
        },
        "baseline_arima_walk_forward": baseline_metrics,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"  ✓ config.json")
    print(f"  ✓ metrics.json")

    print(f"\n{'═' * 60}")
    print(f"  實驗完成。結果：{results_dir}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
