"""v138 段落偵測測試 — 支援 5 種結構（# N / ## N / 中文 / **N. / 段落 #N）。"""
from __future__ import annotations

import pytest


def test_detect_segments_v138_hash_format():
    """純 # N 結構（原 v130 唯一支援的）。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "# 1 開頭\n\n# 2 中段\n\n# 3 結尾"
    assert _detect_segments_v138(text) == {"1", "2", "3"}


def test_detect_segments_v138_hash_multilevel():
    """## N / ### N / #### N 多級 # 結構。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "## 1 第一\n\n### 2 第二\n\n#### 3 第三"
    assert _detect_segments_v138(text) == {"1", "2", "3"}


def test_detect_segments_v138_chinese_numbers():
    """## 第一部分 / ### 二、 / # 第三章 等中文數字結構。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "## 第一部分：A\n\n### 二、技術面\n\n# 第三章 風險"
    found = _detect_segments_v138(text)
    assert {"1", "2", "3"}.issubset(found)


def test_detect_segments_v138_bold_numbered():
    """**1.** / **2)** 行首加粗編號。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "**1. 第一段**\n\n內容\n\n**2. 第二段**\n\n更多內容"
    found = _detect_segments_v138(text)
    assert {"1", "2"}.issubset(found)


def test_detect_segments_v138_naked_numbered():
    """1. / 2) / 3、 裸數字編號。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "1. 首段\n2) 次段\n3、終段"
    found = _detect_segments_v138(text)
    assert {"1", "2", "3"}.issubset(found)


def test_detect_segments_v138_paragraph_label():
    """段落 #1 / 段落 2 / 段落-3 顯式段落標記。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "段落 #1\n內容 A\n\n段落 2\n內容 B\n\n段落-3\n內容 C"
    found = _detect_segments_v138(text)
    assert {"1", "2", "3"}.issubset(found)


def test_detect_segments_v138_mixed_realistic():
    """v138 的真實使用情境：LLM 可能混合多種結構。"""
    from app.api.routes.chat import _detect_segments_v138
    text = """
## 第一部分：市場結構分析
RSI 處於 65。

### 二、技術面分析
MACD 黃金交叉。

**3. 動能評估**
ADX = 28。

# 4. 風險評估
建議倉位 2%。

段落 #5
止損設置在...

#### 第六項
總結建議...
"""
    found = _detect_segments_v138(text)
    assert {"1", "2", "3", "4", "5", "6"}.issubset(found), (
        f"Expected 1-6, got: {sorted(found)}"
    )


def test_detect_segments_v138_empty_text():
    """空字串 / None 不應丟錯，回空集合。"""
    from app.api.routes.chat import _detect_segments_v138
    assert _detect_segments_v138("") == set()
    assert _detect_segments_v138(None) == set()


def test_detect_segments_v138_subsections():
    """`.5` 副段（如 #5.5 / #6.5）也應偵測到。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "# 5\n內容\n\n# 5.5\n副段\n\n# 6\n結尾"
    found = _detect_segments_v138(text)
    assert {"5", "5.5", "6"}.issubset(found)


def test_detect_segments_v138_no_segments():
    """純散文無段落結構應回空集合。"""
    from app.api.routes.chat import _detect_segments_v138
    text = "這是一段沒有任何段落編號的散文，討論市場結構，但完全沒有 #號。"
    assert _detect_segments_v138(text) == set()
