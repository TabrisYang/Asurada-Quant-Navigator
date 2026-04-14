"""阿斯拉量化系統 — CryptoDataEngine 核心數據引擎 v2.0

完整實現：五源投票、斷點續傳、增量更新、重試機制、
各交易所收盤價保存、自動計算技術指標、CSV 完整性驗證、
sync_metadata.json 元數據管理。
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from app.core.config.settings import settings
from app.utils.symbol import get_coinbase_symbol, normalize_symbol, symbol_to_filename
from app.utils.timezone import TAIPEI_TZ, taipei_to_utc, parse_date_string


# 時間週期對應的毫秒數
TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}

# Kraken 預設不啟用
DEFAULT_ENABLED_EXCHANGES = ["binance", "bybit", "okx", "coinbase"]


class CryptoDataEngine:
    """加密貨幣數據引擎 v2.0

    完整功能：
    - 多交易所 OHLCV 抓取 + 五源投票
    - 重試機制 (max_retries=3, delays=[1,2,5])
    - 各交易所個別收盤價保存
    - 抓取後自動計算技術指標 (RSI, ATR, BB, MACD)
    - CSV 完整性驗證與修復
    - sync_metadata.json 元數據管理
    - 斷點續傳 + 增量更新
    """

    def __init__(self):
        self.data_dir = settings.ohlcv_path
        self.exchanges: dict[str, ccxt.Exchange] = {}
        self.anomaly_threshold = settings.anomaly_threshold
        self._metadata_path = self.data_dir / "sync_metadata.json"
        # 同步進度回呼（供 data_sync 路由使用）
        self._progress_callback: Optional[Callable] = None
        # 每交易所並行限制（避免觸發 rate limit）
        self._exchange_semaphores: dict[str, asyncio.Semaphore] = {}

    # ─────────────────────────────────────────────
    # 交易所管理
    # ─────────────────────────────────────────────

    async def _init_exchange(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """初始化交易所連線"""
        if exchange_id in self.exchanges:
            return self.exchanges[exchange_id]

        try:
            exchange_class = getattr(ccxt, exchange_id, None)
            if not exchange_class:
                logger.warning(f"不支援的交易所: {exchange_id}")
                return None

            exchange = exchange_class({
                "enableRateLimit": True,
                "timeout": settings.api_timeout * 1000,
            })
            self.exchanges[exchange_id] = exchange
            return exchange
        except Exception as e:
            logger.error(f"初始化交易所 {exchange_id} 失敗: {e}")
            return None

    async def close_all(self):
        """關閉所有交易所連線"""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception:
                pass
        self.exchanges.clear()

    def _get_semaphore(self, exchange_id: str) -> asyncio.Semaphore:
        """取得交易所專屬 semaphore（限制並行請求數）"""
        if exchange_id not in self._exchange_semaphores:
            # 根據 rate limit 設定並行度：delay 越大，並行度越低
            delay = settings.exchange_rate_limits.get(exchange_id, 0.3)
            max_concurrent = max(1, int(1.0 / delay))  # 0.2s → 5 並發, 0.3s → 3 並發
            self._exchange_semaphores[exchange_id] = asyncio.Semaphore(max_concurrent)
        return self._exchange_semaphores[exchange_id]

    # ─────────────────────────────────────────────
    # 數據抓取（含重試）
    # ─────────────────────────────────────────────

    async def _fetch_from_exchange(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """從單一交易所抓取數據（含重試機制）"""
        exchange = await self._init_exchange(exchange_id)
        if not exchange:
            return None

        # Coinbase 特殊處理：USDT → USD
        fetch_symbol = symbol
        if exchange_id == "coinbase":
            fetch_symbol = get_coinbase_symbol(symbol)

        # 重試邏輯
        for attempt in range(settings.max_retries):
            try:
                delay = settings.get_exchange_delay(exchange_id)
                await asyncio.sleep(delay)

                ohlcv = await exchange.fetch_ohlcv(
                    fetch_symbol, timeframe, since=since_ms, limit=limit
                )
                if not ohlcv:
                    return None

                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["exchange"] = exchange_id
                return df

            except Exception as e:
                if attempt < settings.max_retries - 1:
                    retry_delay = settings.retry_delays[attempt] if attempt < len(settings.retry_delays) else 5.0
                    logger.warning(
                        f"從 {exchange_id} 抓取 {symbol} {timeframe} 失敗 "
                        f"(第 {attempt + 1} 次重試，等待 {retry_delay}s): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"從 {exchange_id} 抓取 {symbol} {timeframe} 最終失敗 "
                        f"(已重試 {settings.max_retries} 次): {e}"
                    )
                    return None

    # ─────────────────────────────────────────────
    # 五源投票機制
    # ─────────────────────────────────────────────

    def _calculate_voting(self, group: pd.DataFrame) -> dict:
        """五源投票機制：計算中位數價格和異常檢測，保存各交易所收盤價"""
        closes = group["close"].dropna().values
        n = len(closes)

        # 各交易所個別收盤價
        exchange_closes = {}
        for _, row in group.iterrows():
            exchange_closes[f"{row['exchange']}_close"] = row["close"]

        if n == 0:
            return {
                "median_price": np.nan,
                "final_price": np.nan,
                "anomaly_detected": False,
                "anomaly_sources": "",
                "data_sources_count": 0,
                **exchange_closes,
            }

        # 中位數計算（規格定義）
        sorted_closes = np.sort(closes)
        if n >= 5:
            median = sorted_closes[2]
        elif n == 4:
            median = (sorted_closes[1] + sorted_closes[2]) / 2
        elif n == 3:
            median = sorted_closes[1]
        elif n == 2:
            median = np.mean(sorted_closes)
        else:
            median = sorted_closes[0]

        # 異常檢測
        anomaly_sources = []
        for _, row in group.iterrows():
            if median > 0 and abs(row["close"] - median) / median > self.anomaly_threshold:
                anomaly_sources.append(row["exchange"])

        # 最終價格：優先 Binance（如果正常）
        binance_row = group[group["exchange"] == "binance"]
        if not binance_row.empty and "binance" not in anomaly_sources:
            final_price = binance_row.iloc[0]["close"]
        else:
            final_price = median

        return {
            "median_price": median,
            "final_price": final_price,
            "anomaly_detected": len(anomaly_sources) > 0,
            "anomaly_sources": ",".join(anomaly_sources),
            "data_sources_count": n,
            **exchange_closes,
        }

    # ─────────────────────────────────────────────
    # 技術指標計算
    # ─────────────────────────────────────────────

    @staticmethod
    def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標：RSI, ATR, BB_width, MACD 系列"""
        if df.empty or len(df) < 2:
            return df

        try:
            # RSI (14)
            rsi = ta.rsi(df["close"], length=14)
            if rsi is not None:
                df["rsi"] = rsi

            # ATR (14)
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr is not None:
                df["atr"] = atr

            # Bollinger Bands width (20, 2)
            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None:
                # bb_width = (upper - lower) / middle
                upper_col = [c for c in bb.columns if "BBU" in c]
                lower_col = [c for c in bb.columns if "BBL" in c]
                mid_col = [c for c in bb.columns if "BBM" in c]
                if upper_col and lower_col and mid_col:
                    df["bb_width"] = (bb[upper_col[0]] - bb[lower_col[0]]) / bb[mid_col[0]]

            # MACD (12, 26, 9)
            macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd_result is not None:
                # pandas_ta 的 MACD 欄位命名: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                for col in macd_result.columns:
                    if col.startswith("MACD_"):
                        df["macd"] = macd_result[col]
                    elif col.startswith("MACDh_"):
                        df["macd_hist"] = macd_result[col]
                    elif col.startswith("MACDs_"):
                        df["macd_signal"] = macd_result[col]

        except Exception as e:
            logger.warning(f"計算技術指標失敗: {e}")

        return df

    # ─────────────────────────────────────────────
    # CSV 完整性驗證
    # ─────────────────────────────────────────────

    @staticmethod
    def _validate_csv(filepath: Path) -> pd.DataFrame:
        """驗證 CSV 完整性，修復不完整的行"""
        try:
            # 先讀取 header
            with open(filepath, "r") as f:
                header_line = f.readline().strip()
                if not header_line:
                    return pd.DataFrame()
                expected_cols = len(header_line.split(","))

            # 讀取所有行，過濾不完整的行
            valid_lines = [header_line]
            with open(filepath, "r") as f:
                next(f)  # 跳過 header
                for line_num, line in enumerate(f, start=2):
                    stripped = line.strip()
                    if stripped and len(stripped.split(",")) == expected_cols:
                        valid_lines.append(stripped)
                    else:
                        logger.warning(f"CSV 第 {line_num} 行欄位數不符，已移除")

            if len(valid_lines) <= 1:
                return pd.DataFrame()

            # 從有效行重建 DataFrame
            from io import StringIO
            csv_content = "\n".join(valid_lines)
            df = pd.read_csv(StringIO(csv_content))

            return df

        except Exception as e:
            logger.error(f"CSV 驗證失敗: {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────
    # sync_metadata.json 管理
    # ─────────────────────────────────────────────

    def _load_metadata(self) -> dict:
        """載入同步元數據"""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self, metadata: dict):
        """儲存同步元數據"""
        from app.utils.timezone import taipei_now
        metadata["updated_at"] = taipei_now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _update_metadata(self, symbol: str, timeframe: str, total_bars: int):
        """更新特定幣種/週期的元數據"""
        from app.utils.timezone import taipei_now
        metadata = self._load_metadata()
        key = symbol.replace("/", "_")

        if key not in metadata:
            metadata[key] = {
                "symbol": symbol,
                "timeframes": [],
                "last_sync": {},
                "total_bars": {},
            }

        entry = metadata[key]
        if timeframe not in entry["timeframes"]:
            entry["timeframes"].append(timeframe)
        entry["last_sync"][timeframe] = taipei_now().strftime("%Y-%m-%dT%H:%M:%S")
        entry["total_bars"][timeframe] = total_bars

        self._save_metadata(metadata)

    # ─────────────────────────────────────────────
    # 主要抓取方法
    # ─────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchanges: Optional[list[str]] = None,
        force_update: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        主要數據抓取方法：五源投票 + 斷點續傳 + 重試 + 指標計算

        Args:
            symbol: 交易對 (如 BTC/USDT)
            timeframe: 時間週期 (15m/1h/4h/1d/1w)
            start_date: 開始日期
            end_date: 結束日期
            exchanges: 交易所列表
            force_update: 是否強制重新抓取
            progress_callback: 進度回呼 (message: str)
        """
        symbol = normalize_symbol(symbol)
        exchanges = exchanges or settings.default_exchanges
        filename = symbol_to_filename(symbol, timeframe)
        filepath = self.data_dir / filename

        def _report(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # 驗證至少 2 家交易所
        if len(exchanges) < 2:
            raise ValueError("至少需要選擇 2 家交易所以啟用五源投票機制")

        # 設定日期範圍
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            # 有本地數據 → 增量更新不需要 start_date；沒有 → 預設從 2020
            start_date = end_date - timedelta(days=settings.default_fetch_days)

        # 斷點續傳：檢查本地數據（同時檢查前段和後段）
        existing_df = None
        need_fetch_before = False  # 是否需要往前補抓
        need_fetch_after = False   # 是否需要往後補抓
        fetch_before_end: Optional[datetime] = None
        fetch_after_start: Optional[datetime] = None

        if filepath.exists() and not force_update:
            try:
                existing_df = self._validate_csv(filepath)
                if not existing_df.empty:
                    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
                    first_ts = existing_df["timestamp"].min()
                    last_ts = existing_df["timestamp"].max()
                    tf_ms = TIMEFRAME_MS.get(timeframe, 86400000)

                    # 檢查是否需要往前補抓（使用者要求的 start_date 比現有數據更早）
                    first_ts_naive = first_ts.to_pydatetime().replace(tzinfo=None)
                    # CSV 存的是台北時間，轉成 UTC 來比較
                    first_ts_utc = taipei_to_utc(TAIPEI_TZ.localize(first_ts_naive))
                    first_ts_utc_naive = first_ts_utc.replace(tzinfo=None)

                    if start_date < first_ts_utc_naive - timedelta(milliseconds=tf_ms):
                        need_fetch_before = True
                        fetch_before_end = first_ts_utc_naive - timedelta(milliseconds=tf_ms)
                        _report(
                            f"需要往前補抓: {start_date.strftime('%Y-%m-%d')} ~ "
                            f"{fetch_before_end.strftime('%Y-%m-%d')}"
                        )

                    # 檢查是否需要往後補抓（現有數據的結尾還沒到 end_date）
                    last_ts_naive = last_ts.to_pydatetime().replace(tzinfo=None)
                    last_ts_utc = taipei_to_utc(TAIPEI_TZ.localize(last_ts_naive))
                    last_ts_utc_naive = last_ts_utc.replace(tzinfo=None)
                    since_after_ms = int(last_ts_utc_naive.timestamp() * 1000) + tf_ms

                    if since_after_ms < int(end_date.timestamp() * 1000):
                        need_fetch_after = True
                        fetch_after_start = datetime.utcfromtimestamp(since_after_ms / 1000)
                        _report(
                            f"需要往後補抓: {fetch_after_start.strftime('%Y-%m-%d')} ~ "
                            f"{end_date.strftime('%Y-%m-%d')}"
                        )

                    # 如果前後都不需要補抓 → 數據完整
                    if not need_fetch_before and not need_fetch_after:
                        _report(f"本地數據已是最新: {filename} ({len(existing_df)} 筆)")
                        return existing_df

            except Exception as e:
                logger.warning(f"讀取本地數據失敗，將重新抓取: {e}")
                existing_df = None
                need_fetch_before = False
                need_fetch_after = False

        # 決定要抓取的時間段（可能有多段）
        fetch_ranges: list[tuple[int, int]] = []

        if existing_df is not None and not existing_df.empty:
            # 有本地數據：只抓缺失的部分
            if need_fetch_before:
                fetch_ranges.append((
                    int(start_date.timestamp() * 1000),
                    int(fetch_before_end.timestamp() * 1000) if fetch_before_end else int(end_date.timestamp() * 1000),
                ))
            if need_fetch_after:
                fetch_ranges.append((
                    int(fetch_after_start.timestamp() * 1000) if fetch_after_start else int(start_date.timestamp() * 1000),
                    int(end_date.timestamp() * 1000),
                ))
        else:
            # 沒有本地數據：抓全部
            fetch_ranges.append((
                int(start_date.timestamp() * 1000),
                int(end_date.timestamp() * 1000),
            ))

        all_data: list[pd.DataFrame] = []

        _report(f"需要抓取 {len(fetch_ranges)} 個時間段")

        for range_idx, (range_since_ms, range_end_ms) in enumerate(fetch_ranges):
            range_start_str = datetime.utcfromtimestamp(range_since_ms / 1000).strftime('%Y-%m-%d')
            range_end_str = datetime.utcfromtimestamp(range_end_ms / 1000).strftime('%Y-%m-%d')
            _report(f"時間段 {range_idx+1}/{len(fetch_ranges)}: {range_start_str} ~ {range_end_str}")

            async def _fetch_exchange_all_pages(exchange_id: str) -> Optional[pd.DataFrame]:
                """從單一交易所抓取所有分頁數據（受 semaphore 保護）"""
                sem = self._get_semaphore(exchange_id)
                _report(f"  正在從 {exchange_id} 抓取 {symbol} {timeframe}...")
                current_since = range_since_ms
                exchange_data = []

                while current_since < range_end_ms:
                    async with sem:
                        df = await self._fetch_from_exchange(
                            exchange_id, symbol, timeframe, current_since,
                            limit=settings.max_bars_per_request
                        )
                    if df is None or df.empty:
                        break

                    exchange_data.append(df)
                    last_ts = df["timestamp"].max()
                    current_since = int(last_ts) + TIMEFRAME_MS.get(timeframe, 86400000)

                    if len(df) < 2:
                        break

                if exchange_data:
                    combined = pd.concat(exchange_data, ignore_index=True)
                    _report(f"    {exchange_id}: {len(combined)} 筆數據")
                    return combined
                return None

            # 所有交易所並行抓取
            results = await asyncio.gather(
                *[_fetch_exchange_all_pages(eid) for eid in exchanges],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, pd.DataFrame):
                    all_data.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"交易所抓取異常: {result}")

        if not all_data:
            if existing_df is not None and not existing_df.empty:
                return existing_df
            logger.warning(f"無法取得任何數據: {symbol} {timeframe}")
            return pd.DataFrame()

        # 合併所有交易所數據
        merged = pd.concat(all_data, ignore_index=True)

        # 五源投票
        _report("執行五源投票機制...")
        results = []
        for ts, group in merged.groupby("timestamp"):
            voting = self._calculate_voting(group)
            # 成交量：優先使用 Binance（最大交易所）的 volume 作為基準，
            # 避免跨所加總導致量能指標（OBV, RelVol）失真。
            # 若 Binance 不存在，退回到所有交易所的中位數。
            binance_rows = group[group["exchange"] == "binance"]
            if not binance_rows.empty:
                vol = float(binance_rows["volume"].iloc[0])
            else:
                vol = float(group["volume"].median())
            results.append({
                "timestamp": pd.Timestamp(ts, unit="ms"),
                "open": group["open"].median(),
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": voting["final_price"],
                "volume": vol,
                **voting,
            })

        new_df = pd.DataFrame(results)

        if new_df.empty:
            if existing_df is not None:
                return existing_df
            return pd.DataFrame()

        # 轉換時區到台北時間
        new_df["timestamp"] = (
            new_df["timestamp"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(TAIPEI_TZ)
            .dt.tz_localize(None)
        )

        # 合併舊數據
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = new_df.sort_values("timestamp").reset_index(drop=True)

        # 計算技術指標
        _report("計算技術指標...")
        combined = self._calculate_indicators(combined)

        # 儲存到 CSV
        combined.to_csv(filepath, index=False)
        _report(f"已儲存 {filename}: {len(combined)} 筆數據")

        # 更新元數據
        self._update_metadata(symbol, timeframe, len(combined))

        # ── 抓取衍生品數據（Funding Rate + OI + 多空比）──
        try:
            await self._fetch_and_save_derivatives(
                symbol, timeframe, start_date, end_date, progress_callback,
            )
        except Exception as e:
            logger.warning(f"衍生品數據抓取失敗（不影響主數據）: {e}")

        return combined

    # ─────────────────────────────────────────────
    # 本地數據讀取
    # ─────────────────────────────────────────────

    def load_local_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """讀取本地 CSV 數據（不走網路），含 CSV 完整性驗證"""
        symbol = normalize_symbol(symbol)
        filename = symbol_to_filename(symbol, timeframe)
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.info(f"本地數據不存在: {filename}")
            return pd.DataFrame()

        try:
            df = self._validate_csv(filepath)
            if df.empty:
                return df

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # 日期範圍篩選（end_date 若只有日期，自動補到當天 23:59:59）
            if start_date:
                try:
                    sd = parse_date_string(start_date)
                except ValueError:
                    sd = pd.to_datetime(start_date)
                df = df[df["timestamp"] >= pd.Timestamp(sd)]
            if end_date:
                try:
                    ed = parse_date_string(end_date, as_end_of_day=True)
                except ValueError:
                    ed = pd.to_datetime(end_date)
                df = df[df["timestamp"] <= pd.Timestamp(ed)]

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"讀取 {filename} 失敗: {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────
    # 查詢與工具
    # ─────────────────────────────────────────────

    def list_available_data(self) -> list[dict]:
        """列出所有本地可用的數據檔案"""
        result = []
        metadata = self._load_metadata()

        for f in self.data_dir.glob("*.csv"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                try:
                    with open(f) as fp:
                        total_rows = sum(1 for _ in fp) - 1  # 扣掉 header
                    sym = f"{parts[0]}/{parts[1]}"
                    tf = parts[2]
                    key = f"{parts[0]}_{parts[1]}"
                    last_sync = None
                    if key in metadata and tf in metadata[key].get("last_sync", {}):
                        last_sync = metadata[key]["last_sync"][tf]

                    result.append({
                        "symbol": sym,
                        "timeframe": tf,
                        "filename": f.name,
                        "records": total_rows,
                        "last_sync": last_sync,
                    })
                except Exception:
                    pass
        return result

    def get_metadata(self) -> dict:
        """取得完整同步元數據"""
        return self._load_metadata()

    # ─────────────────────────────────────────────
    # 衍生品數據（Funding Rate / OI / Long-Short Ratio）
    # ─────────────────────────────────────────────

    async def _fetch_and_save_derivatives(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        progress_callback: Optional[Callable] = None,
    ):
        """平行抓取 Binance Futures 衍生品數據並存 CSV。"""
        import aiohttp

        def _report(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # 只有主要幣種才有衍生品數據
        base = symbol.split("/")[0].upper()
        supported = {"BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "AVAX", "LINK", "DOT", "MATIC"}
        if base not in supported:
            return

        _report(f"抓取衍生品數據: {symbol}...")

        safe_name = symbol.replace("/", "_")
        deriv_file = self.data_dir / f"{safe_name}_derivatives_{timeframe}.csv"

        # Binance API 基礎 URL
        base_url = "https://fapi.binance.com"
        binance_symbol = symbol.replace("/", "").replace("USDT", "USDT")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_funding_rate(session, base_url, binance_symbol, start_date, end_date),
                self._fetch_open_interest_hist(session, base_url, binance_symbol, timeframe, start_date, end_date),
                self._fetch_long_short_ratio(session, base_url, binance_symbol, timeframe, start_date, end_date),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        funding = results[0] if not isinstance(results[0], Exception) else []
        oi = results[1] if not isinstance(results[1], Exception) else []
        ls = results[2] if not isinstance(results[2], Exception) else []

        # 合併成 DataFrame
        dfs = []
        if funding:
            df_f = pd.DataFrame(funding)
            df_f["timestamp"] = pd.to_datetime(df_f["fundingTime"], unit="ms")
            df_f = df_f.rename(columns={"fundingRate": "funding_rate"})
            df_f = df_f[["timestamp", "funding_rate"]].drop_duplicates("timestamp")
            df_f["funding_rate"] = df_f["funding_rate"].astype(float)
            dfs.append(("funding_rate", df_f))

        if oi:
            df_o = pd.DataFrame(oi)
            df_o["timestamp"] = pd.to_datetime(df_o["timestamp"], unit="ms")
            df_o = df_o.rename(columns={"sumOpenInterestValue": "open_interest"})
            df_o = df_o[["timestamp", "open_interest"]].drop_duplicates("timestamp")
            df_o["open_interest"] = df_o["open_interest"].astype(float)
            dfs.append(("open_interest", df_o))

        if ls:
            df_ls = pd.DataFrame(ls)
            df_ls["timestamp"] = pd.to_datetime(df_ls["timestamp"], unit="ms")
            df_ls = df_ls.rename(columns={"longShortRatio": "long_short_ratio"})
            df_ls = df_ls[["timestamp", "long_short_ratio"]].drop_duplicates("timestamp")
            df_ls["long_short_ratio"] = df_ls["long_short_ratio"].astype(float)
            dfs.append(("long_short_ratio", df_ls))

        if not dfs:
            _report("衍生品數據：所有 API 均未回傳數據")
            return

        # 合併（以時間為 key）
        merged = dfs[0][1]
        for _, df_extra in dfs[1:]:
            merged = pd.merge(merged, df_extra, on="timestamp", how="outer")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # 存檔
        merged.to_csv(deriv_file, index=False)
        _report(f"衍生品數據已儲存: {deriv_file.name}（{len(merged)} 筆）")

    @staticmethod
    async def _fetch_funding_rate(session, base_url, symbol, start_date, end_date):
        """Binance Futures Funding Rate"""
        params = {"symbol": symbol, "limit": 1000}
        if start_date:
            params["startTime"] = int(start_date.timestamp() * 1000)
        if end_date:
            params["endTime"] = int(end_date.timestamp() * 1000)
        async with session.get(f"{base_url}/fapi/v1/fundingRate", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
        return []

    @staticmethod
    async def _fetch_open_interest_hist(session, base_url, symbol, timeframe, start_date, end_date):
        """Binance Futures Open Interest History"""
        period_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        period = period_map.get(timeframe, "1d")
        params = {"symbol": symbol, "period": period, "limit": 500}
        if start_date:
            params["startTime"] = int(start_date.timestamp() * 1000)
        if end_date:
            params["endTime"] = int(end_date.timestamp() * 1000)
        async with session.get(f"{base_url}/futures/data/openInterestHist", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
        return []

    @staticmethod
    async def _fetch_long_short_ratio(session, base_url, symbol, timeframe, start_date, end_date):
        """Binance Futures Global Long/Short Account Ratio"""
        period_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        period = period_map.get(timeframe, "1d")
        params = {"symbol": symbol, "period": period, "limit": 500}
        if start_date:
            params["startTime"] = int(start_date.timestamp() * 1000)
        if end_date:
            params["endTime"] = int(end_date.timestamp() * 1000)
        async with session.get(f"{base_url}/futures/data/globalLongShortAccountRatio", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
        return []

    def load_derivatives(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """讀取衍生品數據（Funding Rate / OI / Long-Short Ratio）。"""
        safe_name = normalize_symbol(symbol).replace("/", "_")
        deriv_file = self.data_dir / f"{safe_name}_derivatives_{timeframe}.csv"
        if not deriv_file.exists():
            return None
        try:
            df = pd.read_csv(deriv_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception:
            return None


# 全域單例
crypto_engine = CryptoDataEngine()
