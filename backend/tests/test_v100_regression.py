"""v101 防護驗證 — 確保 v101 改動不破壞 v100 行為。

關鍵鐵律：
  1. 所有 flag OFF 時，行為等同 v100
  2. SHADOW MODE 期間 user 看不到任何 v101 輸出
  3. Quality Gate 未通過時 use_v101() 必須回 False
  4. v101 自動 rollback 不會破壞 v100 結論卡

每個 commit 都該跑這些測試，全通過才能 merge。
"""

import pytest

from app.core.canary import use_v101, v101_passes_quality_gate, get_canary_status
from app.core.config.settings import settings


@pytest.fixture(autouse=True)
def reset_settings():
    """每個測試開始前恢復預設值（避免測試污染）。"""
    saved = {
        "imitation_learning_enabled": settings.imitation_learning_enabled,
        "imitation_shadow_mode": settings.imitation_shadow_mode,
        "imitation_canary_pct": settings.imitation_canary_pct,
        "quality_gate_enabled": settings.quality_gate_enabled,
    }
    yield
    settings.imitation_learning_enabled = saved["imitation_learning_enabled"]
    settings.imitation_shadow_mode = saved["imitation_shadow_mode"]
    settings.imitation_canary_pct = saved["imitation_canary_pct"]
    settings.quality_gate_enabled = saved["quality_gate_enabled"]


def test_default_settings_user_sees_v100():
    """預設設定下 use_v101() 必須回 False（user 看到 v100）。"""
    # v102 預設：learning_enabled=False, shadow_mode=True (subprocess 安全), canary_pct=0
    assert settings.imitation_learning_enabled is False
    assert settings.imitation_shadow_mode is True
    assert settings.imitation_canary_pct == 0

    for _ in range(100):  # 多次採樣確保不是 random hit
        assert use_v101() is False


def test_learning_enabled_alone_not_enough():
    """只開 learning_enabled 但 shadow_mode 還在 → 仍 False。"""
    settings.imitation_learning_enabled = True
    settings.imitation_shadow_mode = True  # 還在 shadow
    settings.imitation_canary_pct = 100  # 即使 100% canary

    for _ in range(100):
        assert use_v101() is False, "SHADOW MODE 期間絕對不能暴露 v101"


def test_shadow_off_but_quality_gate_fail():
    """SHADOW 關了但 Quality Gate 沒通過 → 仍 False。"""
    settings.imitation_learning_enabled = True
    settings.imitation_shadow_mode = False
    settings.imitation_canary_pct = 100
    settings.quality_gate_enabled = True
    # quality_gate_log 表無紀錄 → v101_passes_quality_gate 回 False

    for _ in range(100):
        assert use_v101() is False, "Quality Gate 未通過時不能暴露 v101"


def test_canary_zero_blocks_user_exposure():
    """Canary 0% → 所有人看 v100，即使 quality gate 通過。"""
    settings.imitation_learning_enabled = True
    settings.imitation_shadow_mode = False
    settings.quality_gate_enabled = False  # 跳過 gate 檢查
    settings.imitation_canary_pct = 0

    for _ in range(100):
        assert use_v101() is False


def test_get_canary_status_returns_correct_active_for_users():
    """get_canary_status 的 active_for_users 計算正確。"""
    # 預設：應該不啟用
    s = get_canary_status()
    assert s["active_for_users"] is False
    assert s["shadow_mode"] is True  # v102 subprocess 安全，預設 True

    # 全開但 quality gate 沒過 → 仍不啟用
    settings.imitation_learning_enabled = True
    settings.imitation_shadow_mode = False
    settings.imitation_canary_pct = 100
    s = get_canary_status()
    assert s["active_for_users"] is False  # quality gate 沒過


def test_quality_gate_log_table_exists():
    """v101 schema 已加 quality_gate_log 表。"""
    from app.core.prediction_tracker import prediction_tracker
    prediction_tracker._ensure_db()
    tables = [
        t[0] for t in prediction_tracker._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    # v101 新增的 4 表
    for required in ["prediction_features", "shadow_predictions",
                     "imitation_model_metrics", "quality_gate_log"]:
        assert required in tables, f"v101 表 {required} 未建立"


def test_predictions_table_unchanged():
    """v100 既有 predictions 表結構不被破壞（純加法保證）。"""
    from app.core.prediction_tracker import prediction_tracker
    prediction_tracker._ensure_db()

    cols = [
        r[1] for r in prediction_tracker._conn.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    ]
    # v100 必要欄位仍在
    required_v100 = [
        "id", "symbol", "timeframe", "direction",
        "entry_price", "target_price", "stop_price",
        "status", "created_at",
    ]
    for col in required_v100:
        assert col in cols, f"v100 必要欄位 {col} 被移除！"


def test_shadow_runner_failure_does_not_propagate():
    """Shadow 失敗時不應該冒泡（鐵律：永不影響使用者）。"""
    from app.core.shadow_runner import maybe_run_shadow

    settings.imitation_shadow_mode = True

    def failing_fn():
        raise RuntimeError("simulated v101 failure")

    # 不應該 raise
    result = maybe_run_shadow("test_failure", failing_fn)
    assert result is None  # 失敗回 None，但沒拋例外
