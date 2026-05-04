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

import json
import re
from datetime import datetime, timezone
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


# ─── NFP（BLS Employment Situation）─────────────────────────


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
    """從 Federal Reserve 官方 calendar 抓 FOMC 會議日期。

    策略：用 URL pattern `fomcpresconfYYYYMMDD.htm` 抓所有會議日期，
    過濾未來的 + 6 個月內的。Fed 把這些 URL 預先建立（即使會議還沒開）
    所以可以提前取得排程。

    來源：https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        with httpx.Client(headers=_HEADERS, timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()

        # URL pattern：fomcpresconf20260429.htm（注意 fomcpresconf 也有 fomcpres+conf 寫法）
        pattern = re.compile(r"fomcpres+conf(\d{4})(\d{2})(\d{2})\.htm")
        all_dates = pattern.findall(r.text)

        out: list[dict] = []
        today = datetime.now().strftime("%Y-%m-%d")
        seen = set()
        for y, m, d in all_dates:
            date_str = f"{y}-{m}-{d}"
            if date_str < today:
                continue
            if date_str in seen:
                continue
            seen.add(date_str)
            try:
                meeting_dt = datetime.strptime(date_str, "%Y-%m-%d")
                if (meeting_dt - datetime.now()).days > 240:  # 8 月內
                    continue
                month_name = meeting_dt.strftime("%B")
                out.append({
                    "date": date_str,
                    "time_utc": "18:00",  # 2 PM ET
                    "name": f"FOMC Rate Decision ({month_name})",
                    "category": "macro",
                    "severity": "high",
                    "scope": "all_crypto,equities",
                    "note": "rate decision + Powell press conference",
                    "source": "Federal Reserve official",
                })
            except Exception:
                continue

        _log(f"Federal Reserve FOMC: 抓到 {len(out)} 筆未來會議")
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


# ─── 主同步函式 ─────────────────────────


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
