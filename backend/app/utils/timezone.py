"""阿斯拉量化系統 — 時區處理工具"""

from datetime import datetime

import pytz

from app.core.config.settings import settings

# 台北時區
TAIPEI_TZ = pytz.timezone(settings.display_timezone)
UTC_TZ = pytz.UTC


def utc_now() -> datetime:
    """取得目前 UTC 時間"""
    return datetime.now(UTC_TZ)


def taipei_now() -> datetime:
    """取得目前台北時間"""
    return datetime.now(TAIPEI_TZ)


def utc_to_taipei(dt: datetime) -> datetime:
    """UTC 轉台北時區"""
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    return dt.astimezone(TAIPEI_TZ)


def taipei_to_utc(dt: datetime) -> datetime:
    """台北時區轉 UTC"""
    if dt.tzinfo is None:
        dt = TAIPEI_TZ.localize(dt)
    return dt.astimezone(UTC_TZ)


def parse_date_string(date_str: str, as_end_of_day: bool = False) -> datetime:
    """解析日期字串，支援多種格式（含 datetime-local 的 T 分隔符）

    Args:
        date_str: 日期字串
        as_end_of_day: 若為 True 且字串只有日期（無時分秒），自動補 23:59:59
    """
    if not date_str:
        raise ValueError("日期字串為空")

    # datetime-local 輸入使用 T 分隔符（如 2026-02-23T17:30）
    normalized = date_str.replace("T", " ")

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(normalized, fmt)
            # 只有日期（無時分秒）且要求當日結尾 → 補 23:59:59
            if as_end_of_day and fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            continue
    raise ValueError(f"無法解析日期字串: {date_str}")


def format_taipei_time(dt: datetime, include_time: bool = True) -> str:
    """格式化為台北時區顯示字串"""
    taipei_dt = utc_to_taipei(dt) if dt.tzinfo == UTC_TZ else dt
    if include_time:
        return taipei_dt.strftime("%Y-%m-%d %H:%M")
    return taipei_dt.strftime("%Y-%m-%d")
