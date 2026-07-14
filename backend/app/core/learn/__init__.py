"""互動式教學模組 — 課程資料與存取函式。

設計原則：純資料 + 讀取層，不碰核心 alpha 邏輯（regime / bias_score /
prediction_tracker）。回測透過 /api/learn/run 直呼 run_backtest 純函式，
不經 LLM function calling。新增課程只需往 lessons.py 的 LESSONS 加資料。
"""

from app.core.learn.lessons import get_lesson, get_lesson_summaries

__all__ = ["get_lesson", "get_lesson_summaries"]
