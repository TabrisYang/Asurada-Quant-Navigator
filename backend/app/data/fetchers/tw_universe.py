"""阿斯拉量化系統 — 台股標的池載入器

從 TWSE ISIN 公開頁面抓取全部上市 + 上櫃股票清單，含代號、名稱、產業別。
本地 JSON 快取 24 小時，避免每次掃描都發遠端請求。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

from app.core.config.settings import settings


# TWSE ISIN 公告頁面（上市 strMode=2 / 上櫃 strMode=4）
_TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
_CACHE_TTL = timedelta(hours=24)
_CODE_PATTERN = re.compile(r"^\d{4}$")  # 一般股票 4 碼


@dataclass
class TwStock:
    code: str         # 如 "2330"
    name: str         # 如 "台積電"
    market: str       # "listed" / "otc"
    industry: str     # 如 "半導體業"；無對應時為 "其他"

    def to_dict(self) -> dict:
        return asdict(self)


class TwUniverse:
    """台股上市櫃完整標的池。"""

    def __init__(self, cache_path: Optional[Path] = None):
        if cache_path is None:
            cache_path = settings.ohlcv_path.parent / "cache" / "tw_universe.json"
        self._cache_path = cache_path
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    async def fetch_all(self, use_cache: bool = True) -> list[TwStock]:
        """取得全部上市櫃股票清單。

        use_cache=True 時，若本地 JSON 在 24 小時內則直接讀；過期則重抓。
        """
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                logger.info(f"TwUniverse: 使用本地快取 ({len(cached)} 檔)")
                return cached

        logger.info("TwUniverse: 從 TWSE ISIN 抓取最新清單...")
        listed, otc = await asyncio.gather(
            asyncio.to_thread(self._scrape, "2"),
            asyncio.to_thread(self._scrape, "4"),
        )
        for s in listed:
            s.market = "listed"
        for s in otc:
            s.market = "otc"
        merged = listed + otc
        logger.info(f"TwUniverse: 抓取完成（上市 {len(listed)} 檔 + 上櫃 {len(otc)} 檔）")

        self._save_cache(merged)
        return merged

    # ──────────────────────────────────────────────
    # HTML 解析
    # ──────────────────────────────────────────────

    @staticmethod
    def _scrape(mode: str) -> list[TwStock]:
        """解析 TWSE ISIN 頁面，只取「股票」section 的 4 碼一般股。"""
        url = _TWSE_ISIN_URL.format(mode=mode)
        r = requests.get(url, timeout=30)
        r.encoding = "big5"
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.find_all("tr")

        results: list[TwStock] = []
        in_stock_section = False

        for row in rows:
            cells = row.find_all("td")
            # Section 標頭：colspan=7 的單格列（如「股票」「臺灣存託憑證」）
            if len(cells) == 1 and cells[0].get("colspan") == "7":
                label = cells[0].get_text(strip=True)
                in_stock_section = label == "股票"
                continue

            if not in_stock_section or len(cells) < 5:
                continue

            # 欄位：代號+名稱 / ISIN / 上市日 / 市場別 / 產業別 / CFICode / 備註
            code_name = cells[0].get_text(strip=True).replace("　", " ")
            parts = code_name.split(" ", 1)
            if len(parts) != 2:
                continue
            code, name = parts[0].strip(), parts[1].strip()
            if not _CODE_PATTERN.match(code):
                continue  # 過濾 ETF（6 碼）、權證等

            industry = cells[4].get_text(strip=True) or "其他"
            results.append(TwStock(
                code=code, name=name, market="",  # market 由呼叫端填入
                industry=industry,
            ))

        return results

    # ──────────────────────────────────────────────
    # JSON 快取
    # ──────────────────────────────────────────────

    def _load_cache(self) -> Optional[list[TwStock]]:
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(data["saved_at"])
            if datetime.now() - saved_at > _CACHE_TTL:
                return None
            return [TwStock(**item) for item in data["stocks"]]
        except Exception as e:
            logger.warning(f"TwUniverse: 快取讀取失敗 ({e})，將重抓")
            return None

    def _save_cache(self, stocks: list[TwStock]) -> None:
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "stocks": [s.to_dict() for s in stocks],
        }
        self._cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"TwUniverse: 已快取 {len(stocks)} 檔至 {self._cache_path}")


# 單例
tw_universe = TwUniverse()
