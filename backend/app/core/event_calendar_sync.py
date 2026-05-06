"""阿斯拉量化系統 — v106 A1：經濟日曆自動同步（不付費版，有限）。

⚠️ 重要限制（實測 2026-05-04）：
免費官方來源大多有 anti-bot 防護或不公開未來日期：
- BLS（NFP / CPI 來源）：返回 403 Forbidden — anti-bot 擋住自動抓取
- Federal Reserve FOMC 頁面：URL pattern 只含已發生會議的 press conference 連結，
  未來會議不公開於可解析格式
- BEA GDP：未測試但結構類似 BLS

→ 自動同步在不付費前提下**不可靠**。框架保留供：
1. 未來如果某網站開放/換 user-agent 後可用
2. 手動觸發（cd backend && .venv/bin/python -m app.core.event_calendar_sync）
3. 將來若接 TradingEconomics 付費 API 可快速擴充

實際解決方式：保留既有 events.json 手動維護 + v104.3 unverified 旗標
+ v104.3 staleness warning（> 14 天自動警示）+ 提醒使用者每月 1 號核對。

不部署 launchd cron（會每天失敗灌爆 log）。

執行：
    cd backend && .venv/bin/python3 -m app.core.event_calendar_sync

日誌：寫到 data/db/calendar_sync.log
"""

from __future__ import annotations

import calendar as _cal
import json
import re
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from loguru import logger

_CALENDAR_PATH = Path(__file__).resolve().parents[2] / "data" / "calendar" / "events.json"
_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "calendar_sync.log"

# 共用 HTTP client config — User-Agent 偽裝避免被 anti-bot 擋
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
_HTTP_TIMEOUT = 10.0


def _log(msg: str) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass
    logger.info(f"[calendar_sync] {msg}")


# ─── NFP — 規則計算（v109，比 BLS 爬蟲更可靠）─────────────────────────

# 美國聯邦假日（影響 NFP 公布）— 必要時補齊
# 規則：若每月第一週五碰到聯邦假日，BLS 會推遲到下個工作日
_US_FEDERAL_HOLIDAYS_FIXED = {
    # (month, day) — 固定日期假日
    (1, 1): "New Year's Day",
    (7, 4): "Independence Day",
    (11, 11): "Veterans Day",
    (12, 25): "Christmas Day",
    # 註：MLK Day / Presidents Day / Memorial Day / Labor Day / Columbus Day /
    # Thanksgiving 是「第 N 個週幾」規則，但這些都不會落在月初第一週五，所以不處理
}


def _is_us_federal_holiday(d: _date) -> bool:
    """是否為美國聯邦假日（簡化版，只檢查可能影響月初第一週五的假日）。"""
    if (d.month, d.day) in _US_FEDERAL_HOLIDAYS_FIXED:
        return True
    # New Year's Day 若落週六 → 前一個週五補假；若落週日 → 隔週一補假
    # 月初第一週五最可能碰到的就是 1/1（如 2027 年 1/1 週五就是真假日）和 7/3（若 7/4 週六則 7/3 週五補假）
    # 7/4 週六 → 7/3 補假
    if d.month == 7 and d.day == 3:
        next_day = d + timedelta(days=1)
        if next_day.weekday() == 5:  # 7/4 是 Saturday
            return True
    return False


def compute_nfp_dates(months_ahead: int = 6) -> list[dict]:
    """v109：規則計算 NFP 公布日期，比 BLS 爬蟲（被 anti-bot 擋）更可靠。

    規則：
    - NFP = 每月第一個週五，8:30 ET 公布（12:30 UTC）
    - 例外：若第一週五碰到聯邦假日，BLS 推遲到下個工作日
    - 公布的是「上個月」資料（如 5/1 公布的是 April NFP）

    Args:
        months_ahead: 計算未來幾個月（預設 6）

    Returns:
        list of event dicts，按日期排序
    """
    out: list[dict] = []
    today_dt = datetime.now()
    today = _date(today_dt.year, today_dt.month, today_dt.day)

    for offset in range(months_ahead + 1):
        target_year = today_dt.year + (today_dt.month + offset - 1) // 12
        target_month = (today_dt.month + offset - 1) % 12 + 1

        # 找該月第一個週五
        first_day = _date(target_year, target_month, 1)
        # weekday(): Mon=0, Fri=4
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)

        # 假日推遲：若第一週五是假日，往後找下個工作日
        actual_release = first_friday
        max_iter = 5
        while max_iter > 0 and (_is_us_federal_holiday(actual_release) or actual_release.weekday() >= 5):
            actual_release = actual_release + timedelta(days=1)
            max_iter -= 1

        if actual_release < today:
            continue

        # NFP 名稱用「上一個月」
        report_month_dt = first_day - timedelta(days=1)  # 前一個月
        report_month_name = _cal.month_name[report_month_dt.month]

        # 推遲說明
        note_parts = [f"first Friday of {first_day.strftime('%B %Y')}"]
        if actual_release != first_friday:
            note_parts.append(f"shifted from {first_friday.isoformat()} due to US federal holiday")

        out.append({
            "date": actual_release.isoformat(),
            "time_utc": "12:30",  # 8:30 AM ET = 12:30 UTC（簡化忽略 DST）
            "name": f"Non-Farm Payrolls ({report_month_name})",
            "category": "macro",
            "severity": "high",
            "scope": "all_crypto,equities",
            "source_strategy": "rule_first_friday",
            "note": "; ".join(note_parts),
        })

    out.sort(key=lambda e: e["date"])
    _log(f"NFP rule_first_friday: 計算出 {len(out)} 筆未來日期")
    return out


# ─── NFP — BLS 爬蟲（legacy，anti-bot 擋住，留作 fallback）─────────────────────────


def fetch_nfp_dates() -> list[dict]:
    """從 BLS 官方頁面抓 NFP 未來 6 個月的公布日期。

    來源：https://www.bls.gov/schedule/news_release/empsit.htm
    回傳 list of {date, time_utc, name, source}。
    """
    url = "https://www.bls.gov/schedule/news_release/empsit.htm"
    try:
        with httpx.Client(headers=_HEADERS, timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        out: list[dict] = []
        # BLS 頁面 schedule 通常是 table 結構：reference month, release date, time
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue
                # 找像 "May 02, 2026" 或 "May 2, 2026" 的日期
                date_str = None
                month_label = None
                for cell in cells:
                    m = re.search(r"(January|February|March|April|May|June|July|August|"
                                  r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
                                  cell, re.IGNORECASE)
                    if m and not date_str:
                        try:
                            date_str = datetime.strptime(
                                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
                            ).strftime("%Y-%m-%d")
                        except ValueError:
                            try:
                                date_str = datetime.strptime(
                                    f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y"
                                ).strftime("%Y-%m-%d")
                            except Exception:
                                continue
                    elif not month_label:
                        month_label = cell

                if date_str:
                    # 只取未來日期
                    today = datetime.now().strftime("%Y-%m-%d")
                    if date_str < today:
                        continue
                    name = "Non-Farm Payrolls"
                    if month_label:
                        ml = re.search(
                            r"(January|February|March|April|May|June|July|August|"
                            r"September|October|November|December)",
                            month_label, re.IGNORECASE,
                        )
                        if ml:
                            name = f"Non-Farm Payrolls ({ml.group(1)})"
                    out.append({
                        "date": date_str,
                        "time_utc": "12:30",  # 8:30 AM ET = 12:30 UTC（含 DST 變動忽略）
                        "name": name,
                        "category": "macro",
                        "severity": "high",
                        "scope": "all_crypto,equities",
                        "source": "BLS official",
                    })

        _log(f"BLS NFP: 抓到 {len(out)} 筆未來日期")
        return out
    except Exception as e:
        _log(f"BLS NFP 抓取失敗: {e}")
        return []


# ─── FOMC（Federal Reserve）─────────────────────────


def fetch_fomc_dates() -> list[dict]:
    """v109 改寫：從 Fed 官方 calendar 用 HTML 結構抓 FOMC 會議日期。

    舊策略（fomcpresconf URL pattern）只能抓「已發生」會議（Fed 在會議結束後才上傳
    press conference URL）。新策略改解析 `class="fomc-meeting"` 結構，可預先取得未來
    全年會議日期。

    來源：https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

    HTML 結構：
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month ...">June</div>
          <div class="fomc-meeting__date ...">16-17*</div>
          ...
        </div>
    （* 標星 = 含 SEP/Dot Plot 經濟預測摘要）

    取會議結束日為利率公告日（FOMC 兩天會議，第 2 天 14:00 ET 公布利率 + Powell 演講）。
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        with httpx.Client(headers=_HEADERS, timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Fed 頁面結構：年份 panel 包多個 fomc-meeting，每筆含 month + date 兩個 div
        # 必須從 panel 找上下文判斷年份（month 只給「June」沒給年）
        out: list[dict] = []
        today = datetime.now().strftime("%Y-%m-%d")
        today_dt = datetime.now()
        seen: set[str] = set()

        # 找所有含「YYYY FOMC Meetings」標題的 panel
        panels = soup.find_all(class_=re.compile(r"panel.*default"))
        if not panels:
            # fallback：直接找所有 fomc-meeting + 從上下文找 year
            panels = [soup]

        for panel in panels:
            # 從 panel 文字找 year（如 "2026 FOMC Meetings"）
            panel_text = panel.get_text(" ", strip=True) if hasattr(panel, "get_text") else ""
            year_m = re.search(r"\b(20\d{2})\s+FOMC\s+Meeting", panel_text)
            if not year_m:
                continue
            year = year_m.group(1)

            # 找 panel 內所有 fomc-meeting（含標準 row + 含 SEP/Dot Plot 的 --shaded variant）
            # BeautifulSoup class_=string 匹配「class list 含此 class」
            # "fomc-meeting" 在標準會議 class="row fomc-meeting" 與 SEP class="fomc-meeting--shaded row fomc-meeting" 中都有
            meetings = panel.find_all(class_="fomc-meeting")
            for meeting in meetings:
                month_el = meeting.find(class_=re.compile(r"fomc-meeting__month"))
                date_el = meeting.find(class_=re.compile(r"fomc-meeting__date"))
                if not month_el or not date_el:
                    continue
                month_name = month_el.get_text(strip=True)
                date_range = date_el.get_text(strip=True)  # 例 "16-17*" / "27-28" / "8-9*" / "22 (notation vote)"

                # 跳過 notation vote 等非會議式條目（無利率決議）
                if "notation" in date_range.lower():
                    continue

                # 解析日期範圍：優先抓「結束日」（兩天會議的 day 2 = 利率公告日）
                # 例 "16-17*" → 17；"8-9*" → 9；"27-28" → 28
                has_sep = "*" in date_range
                clean = date_range.replace("*", "").strip()
                # 提取兩個數字
                nums = re.findall(r"\d+", clean)
                if not nums:
                    continue
                end_day = int(nums[-1]) if len(nums) >= 1 else None
                if end_day is None:
                    continue

                try:
                    meeting_dt = datetime.strptime(f"{year} {month_name} {end_day}", "%Y %B %d")
                except ValueError:
                    continue

                date_str = meeting_dt.strftime("%Y-%m-%d")
                if date_str < today:
                    continue
                if (meeting_dt - today_dt).days > 365:  # 12 個月內
                    continue
                if date_str in seen:
                    continue
                seen.add(date_str)

                name = f"FOMC Rate Decision ({month_name})"
                if has_sep:
                    name += " + Dot Plot"

                # 兩天會議命名加上完整日期範圍（避免維護者誤解）
                start_day = int(nums[0]) if len(nums) >= 2 else end_day
                day_range_str = f"{month_name} {start_day}-{end_day}" if start_day != end_day else f"{month_name} {end_day}"

                out.append({
                    "date": date_str,
                    "time_utc": "18:00",  # 14:00 ET = 18:00 UTC（夏令 EDT 改 17:00；本版簡化用 18:00）
                    "name": name,
                    "category": "macro",
                    "severity": "high",
                    "scope": "all_crypto,equities",
                    "source_strategy": "scraper_fed",
                    "note": (
                        f"meeting {day_range_str}, rate decision 14:00 ET on day {end_day}"
                        + (" + SEP/Dot Plot" if has_sep else "")
                    ),
                })

        out.sort(key=lambda e: e["date"])
        _log(f"Federal Reserve FOMC (HTML structure): 抓到 {len(out)} 筆未來會議")
        return out
    except Exception as e:
        _log(f"Federal Reserve FOMC 抓取失敗: {e}")
        return []


# ─── CPI（BLS）─────────────────────────


def fetch_cpi_dates() -> list[dict]:
    """從 BLS 官方頁面抓 CPI 公布日期。

    來源：https://www.bls.gov/schedule/news_release/cpi.htm
    """
    url = "https://www.bls.gov/schedule/news_release/cpi.htm"
    try:
        with httpx.Client(headers=_HEADERS, timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        out: list[dict] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue
                date_str = None
                month_label = None
                for cell in cells:
                    m = re.search(r"(January|February|March|April|May|June|July|August|"
                                  r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
                                  cell, re.IGNORECASE)
                    if m and not date_str:
                        try:
                            date_str = datetime.strptime(
                                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
                            ).strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                    elif not month_label:
                        month_label = cell

                if date_str:
                    today = datetime.now().strftime("%Y-%m-%d")
                    if date_str < today:
                        continue
                    name = "US CPI"
                    if month_label:
                        ml = re.search(
                            r"(January|February|March|April|May|June|July|August|"
                            r"September|October|November|December)",
                            month_label, re.IGNORECASE,
                        )
                        if ml:
                            name = f"US CPI ({ml.group(1)})"
                    out.append({
                        "date": date_str,
                        "time_utc": "12:30",
                        "name": name,
                        "category": "macro",
                        "severity": "high",
                        "scope": "all_crypto,equities",
                        "note": "core inflation print",
                        "source": "BLS official",
                    })

        _log(f"BLS CPI: 抓到 {len(out)} 筆未來日期")
        return out
    except Exception as e:
        _log(f"BLS CPI 抓取失敗: {e}")
        return []


# ─── v109 主同步函式：只收 A+B 類可驗證事件 ─────────────────────────


def sync_verifiable_events() -> dict:
    """v109：只同步可自動驗證的事件（FOMC + NFP），其他類別不收。

    策略：
    - A 類（FOMC）：用 fetch_fomc_dates()（HTML 結構爬蟲）
    - B 類（NFP）：用 compute_nfp_dates()（規則計算，比爬蟲可靠）
    - C 類（CPI/PPI/GDP/PCE）：完全不收，由 system prompt 通用提醒處理
    - D 類（OPEC/地緣）：不在 events.json 範疇

    合併規則：
    - source_strategy="manual" 的 entry 不被自動同步覆寫（保留使用者手動加的事件）
    - source_strategy="scraper_fed" / "rule_first_friday" 的舊 entry 會被新計算結果覆寫
    - 過期 entry（date < today - 1d）自動移除
    - 自動同步全部失敗 → 不動 events.json + log warning + 標 unverified

    Returns:
        summary dict
    """
    today_dt = datetime.now()
    today = _date(today_dt.year, today_dt.month, today_dt.day)
    today_str = today.isoformat()

    # 1) 抓 A 類（FOMC）
    fomc_status = "unknown"
    try:
        fomc_events = fetch_fomc_dates()
        fomc_status = f"fetched {len(fomc_events)}"
    except Exception as e:
        _log(f"sync_verifiable: FOMC fetch 例外: {e}")
        fomc_events = []
        fomc_status = f"error: {e}"

    # 2) 算 B 類（NFP）— 規則計算不會失敗
    try:
        nfp_events = compute_nfp_dates(months_ahead=6)
        nfp_status = f"computed {len(nfp_events)}"
    except Exception as e:
        _log(f"sync_verifiable: NFP compute 例外: {e}")
        nfp_events = []
        nfp_status = f"error: {e}"

    auto_events = list(fomc_events) + list(nfp_events)

    # 3) 讀現有 events.json
    if _CALENDAR_PATH.exists():
        try:
            existing = json.loads(_CALENDAR_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {"_meta": {}, "events": []}
    else:
        existing = {"_meta": {}, "events": []}

    existing_events: list[dict] = existing.get("events", [])

    # 4) 分類保留：manual entry 保留、自動 entry 全部丟掉用新算的
    manual_events = [
        ev for ev in existing_events
        if ev.get("source_strategy") == "manual"
    ]

    # 5) 合併 — 過期過濾 + 排序
    merged: list[dict] = []
    seen_keys: set[str] = set()

    def _key(ev: dict) -> str:
        # 用 date + name 前綴當去重 key（容忍 month 後綴變動）
        name_short = (ev.get("name") or "").split("(")[0].strip()[:40]
        return f"{ev.get('date')}|{name_short}"

    # auto events 優先（覆蓋 manual 同 key）
    for ev in auto_events:
        d = ev.get("date") or ""
        if d < today_str:
            continue
        k = _key(ev)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        ev["auto_synced_at"] = today_dt.isoformat()
        merged.append(ev)

    # manual 補進來（不覆蓋 auto）
    for ev in manual_events:
        d = ev.get("date") or ""
        if d < today_str:
            continue  # 過期跳過
        k = _key(ev)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        # manual entry 強制標 unverified（除非已標）
        if "unverified" not in ev:
            ev["unverified"] = True
        merged.append(ev)

    # 6) 全部失敗 fallback：保留現有非過期 entries 但標 unverified
    if not auto_events and not manual_events:
        _log("⚠️ 自動同步全部失敗且無 manual entries，events.json 保持空 events 狀態")
        for ev in existing_events:
            d = ev.get("date") or ""
            if d < today_str:
                continue
            ev["unverified"] = True
            ev.setdefault("source_strategy", "manual")
            merged.append(ev)

    merged.sort(key=lambda e: e.get("date", ""))

    # 7) 寫回 events.json
    new_meta = existing.get("_meta", {})
    new_meta.update({
        "version": 3,
        "last_updated": today_str,
        "last_auto_sync": today_dt.isoformat(),
        "auto_sync_status": {
            "fomc_scraper_fed": fomc_status,
            "nfp_rule_first_friday": nfp_status,
        },
        "maintainer": "auto-synced (FOMC scraper + NFP rule); manual entries kept if source_strategy=manual",
    })

    _CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CALENDAR_PATH.write_text(
        json.dumps({"_meta": new_meta, "events": merged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "ok" if (fomc_events or nfp_events) else "all_failed_kept_manual",
        "fomc_status": fomc_status,
        "nfp_status": nfp_status,
        "n_auto": len(auto_events),
        "n_manual_kept": len([e for e in merged if e.get("source_strategy") == "manual"]),
        "n_total": len(merged),
    }
    _log(f"sync_verifiable_events: {summary}")
    return summary


# ─── Legacy v106 主同步函式（保留向下相容）─────────────────────────


def sync_calendar() -> dict:
    """從所有來源抓取未來經濟事件，更新 events.json。

    策略：
    - 抓到的事件 → 加 source 欄位 + 移除 unverified flag
    - 既有 events.json 中沒被抓到的事件 → 保留（手動維護的不刪）
    - 重複事件（同 date + 同 name 開頭關鍵字）→ 用新抓的覆蓋
    - 全部失敗 → 不動 events.json（graceful fallback）

    回傳 summary dict。
    """
    fetchers = [
        ("NFP", fetch_nfp_dates),
        ("FOMC", fetch_fomc_dates),
        ("CPI", fetch_cpi_dates),
    ]

    fetched_events: list[dict] = []
    fetcher_status: dict[str, int] = {}
    for name, fn in fetchers:
        try:
            events = fn()
            fetcher_status[name] = len(events)
            fetched_events.extend(events)
        except Exception as e:
            _log(f"fetcher {name} 例外: {e}")
            fetcher_status[name] = -1

    if not fetched_events:
        _log("⚠️ 所有來源都失敗，events.json 保留不動")
        return {"status": "all_failed", "fetcher_status": fetcher_status}

    # 讀現有 events.json
    if _CALENDAR_PATH.exists():
        try:
            existing = json.loads(_CALENDAR_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {"_meta": {}, "events": []}
    else:
        existing = {"_meta": {}, "events": []}

    existing_events = existing.get("events", [])

    # 建立「key → existing event」index，用來保留手動維護的事件
    def _event_key(ev: dict) -> str:
        # 用「name 前 4 字 + date」當 key（容忍 month 後綴變動）
        name_short = (ev.get("name") or "").split("(")[0].strip()[:30]
        return f"{name_short}|{ev.get('date')}"

    merged: dict[str, dict] = {_event_key(ev): ev for ev in existing_events}

    # 用抓到的事件覆蓋同 key 條目（標 verified）
    n_added = 0
    n_updated = 0
    for ev in fetched_events:
        ev["unverified"] = False  # 從官方來源抓的，verified
        k = _event_key(ev)
        if k in merged:
            old = merged[k]
            # 只覆寫 date / time_utc / source / unverified，保留 note
            old.update({
                "date": ev["date"],
                "time_utc": ev["time_utc"],
                "name": ev["name"],
                "source": ev["source"],
                "unverified": False,
            })
            n_updated += 1
        else:
            merged[k] = ev
            n_added += 1

    # 移除已過期事件
    today = datetime.now().strftime("%Y-%m-%d")
    final_events = [
        ev for ev in merged.values()
        if (ev.get("date") or "9999-99-99") >= today
    ]

    # 按日期排序
    final_events.sort(key=lambda e: e.get("date", ""))

    # 更新 _meta
    new_meta = existing.get("_meta", {})
    new_meta.update({
        "version": (new_meta.get("version") or 1),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "last_synced_at": datetime.now(timezone.utc).isoformat(),
        "auto_sync_sources": list(fetcher_status.keys()),
        "fetcher_status": fetcher_status,
        "maintainer": "auto + manual (v106 A1)",
    })

    # 寫回
    _CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CALENDAR_PATH.write_text(
        json.dumps({"_meta": new_meta, "events": final_events}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "fetcher_status": fetcher_status,
        "n_added": n_added,
        "n_updated": n_updated,
        "total_events": len(final_events),
    }
    _log(f"✅ 同步完成: 新增 {n_added}、更新 {n_updated}、總計 {len(final_events)}")
    return summary


if __name__ == "__main__":
    import sys
    result = sync_calendar()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("status") == "ok" else 1)
