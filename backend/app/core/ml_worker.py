"""阿斯拉量化系統 — v102 ML 推論子進程入口

每次預測時 spawn 獨立 Python process 執行此腳本：
  python -m app.core.ml_worker  < {"features": ..., "chart_state": ...}

主進程透過 ml_client.predict_via_subprocess() 呼叫此 worker。
這樣主進程**永不載 lightgbm/shap**，徹底避免 native lib 衝突造成 segfault。

設計原則：
  - 子進程 1-2 秒內完成（啟動 + 載 model + 推論 + 退出）
  - stdin: JSON 輸入；stdout: JSON 輸出；stderr: log
  - exit code: 0 成功 / 1 推論錯 / 2 輸入錯
  - 失敗永不污染主進程（subprocess 死了主進程繼續跑）
"""

from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    """從 stdin 讀 JSON，呼叫 imitation_predictor.predict，輸出 JSON 到 stdout。"""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(json.dumps({"error": "no_input"}), file=sys.stderr)
            return 2

        payload = json.loads(raw)
        features = payload.get("features") or {}
        chart_state = payload.get("chart_state") or {}
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid_json: {e}"}), file=sys.stderr)
        return 2
    except Exception as e:
        print(json.dumps({"error": f"input_error: {e}"}), file=sys.stderr)
        return 2

    try:
        # ★ 重點：lazy import — 只在這個子進程載 lightgbm/shap
        from app.core.imitation_predictor import imitation_predictor

        insight = imitation_predictor.predict(features, chart_state)
        print(json.dumps(insight, ensure_ascii=False, default=str))
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": tb[-500:]}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
