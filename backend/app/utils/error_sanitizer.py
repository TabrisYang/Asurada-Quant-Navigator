"""阿斯拉量化系統 — 錯誤訊息 sanitization

生產環境下隱藏內部路徑、版本等敏感資訊，
debug 模式下仍返回完整錯誤。
"""

import re

from app.core.config.settings import settings

# 常見的敏感路徑模式
_PATH_PATTERN = re.compile(r"(/[\w\-./]+\.py(?::\d+)?)")
_TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)

# 錯誤類型到用戶友好訊息的映射
_FRIENDLY_MESSAGES = {
    "ConnectionError": "無法連線到服務，請檢查網路設定",
    "TimeoutError": "請求逾時，請稍後再試",
    "AuthenticationError": "API 認證失敗，請檢查 API Key",
    "RateLimitError": "API 請求過於頻繁，請稍後再試",
    "ValueError": "輸入資料格式不正確",
    "FileNotFoundError": "找不到所需的資源",
}


def sanitize_error(error: Exception) -> str:
    """將例外轉為安全的錯誤訊息

    - debug=True: 返回完整錯誤
    - debug=False: 隱藏路徑和 traceback，返回友好訊息
    """
    if settings.debug:
        return str(error)

    error_type = type(error).__name__
    if error_type in _FRIENDLY_MESSAGES:
        return _FRIENDLY_MESSAGES[error_type]

    msg = str(error)
    # 移除檔案路徑
    msg = _PATH_PATTERN.sub("[internal]", msg)
    # 移除 traceback
    msg = _TRACEBACK_PATTERN.sub("", msg)
    # 截斷過長訊息
    if len(msg) > 200:
        msg = msg[:200] + "..."

    return msg.strip() or "發生內部錯誤，請稍後再試"
