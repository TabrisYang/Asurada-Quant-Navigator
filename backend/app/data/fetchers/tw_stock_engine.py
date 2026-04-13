"""阿斯拉量化系統 — TwStockEngine 台股數據引擎

使用 yfinance 抓取台股 OHLCV 數據，輸出格式與 CryptoDataEngine 完全一致。
支援斷點續傳、增量更新、技術指標計算。
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Callable

import pandas as pd
import yfinance as yf
from loguru import logger

from app.core.config.settings import settings
from app.data.fetchers.crypto_engine import CryptoDataEngine
from app.utils.symbol import normalize_symbol, symbol_to_filename, symbol_to_yf_ticker
from app.utils.timezone import parse_date_string

# yfinance 時間框架映射
YF_TIMEFRAME_MAP = {
    "1d": "1d",
    "1w": "1wk",
}

# 台股支援的時間框架
SUPPORTED_TW_TIMEFRAMES = {"1d", "1w"}


class TwStockEngine:
    """台股數據引擎

    功能：
    - 透過 yfinance 抓取台股 OHLCV
    - 輸出 CSV 格式與加密貨幣完全一致
    - 斷點續傳 + 增量更新
    - 自動計算技術指標（RSI, ATR, BB, MACD）
    - sync_metadata.json 元數據管理
    """

    def __init__(self):
        self.data_dir = settings.ohlcv_path
        self._metadata_path = self.data_dir / "sync_metadata.json"

    # ─────────────────────────────────────────────
    # 主要抓取方法
    # ─────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_update: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        抓取台股 OHLCV 數據

        Args:
            symbol: 交易對（如 2330/TWD）
            timeframe: 時間週期（僅支援 1d / 1w）
            start_date: 開始日期
            end_date: 結束日期
            force_update: 是否強制重新抓取
            progress_callback: 進度回呼
        """
        symbol = normalize_symbol(symbol)
        if timeframe not in SUPPORTED_TW_TIMEFRAMES:
            raise ValueError(
                f"台股不支援 {timeframe} 時間框架，僅支援: {', '.join(SUPPORTED_TW_TIMEFRAMES)}"
            )

        filename = symbol_to_filename(symbol, timeframe)
        filepath = self.data_dir / filename
        yf_ticker = symbol_to_yf_ticker(symbol)
        yf_interval = YF_TIMEFRAME_MAP[timeframe]

        def _report(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # 設定日期範圍
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            # 檢查是否有本地數據：有 → 增量更新不需要 start_date
            # 沒有 → 預設從 2020-01-01 開始
            if not filepath.exists():
                start_date = datetime(2020, 1, 1)
                _report("未指定起始日期，預設從 2020-01-01 開始抓取")

        # 斷點續傳：檢查本地數據
        existing_df = None
        if filepath.exists() and not force_update:
            try:
                existing_df = CryptoDataEngine._validate_csv(filepath)
                if not existing_df.empty:
                    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
                    first_ts = existing_df["timestamp"].min()
                    first_date = first_ts.to_pydatetime().replace(tzinfo=None)
                    last_ts = existing_df["timestamp"].max()
                    last_date = last_ts.to_pydatetime().replace(tzinfo=None)

                    need_backfill = start_date is not None and first_date > start_date + timedelta(days=1)
                    need_forward = last_date < end_date - timedelta(days=1)

                    # 往前補抓歷史數據
                    if need_backfill:
                        _report(f"往前補抓: {start_date.strftime('%Y-%m-%d')} ~ {first_date.strftime('%Y-%m-%d')}")
                        earlier_df = await asyncio.to_thread(
                            self._download_from_yfinance,
                            yf_ticker, yf_interval, start_date, first_date,
                        )
                        if earlier_df is not None and not earlier_df.empty:
                            earlier_df = self._to_standard_format(earlier_df)
                            existing_df = pd.concat([earlier_df, existing_df], ignore_index=True)
                            existing_df = existing_df.drop_duplicates(subset=["timestamp"], keep="last")
                            existing_df = existing_df.sort_values("timestamp").reset_index(drop=True)

                    # 往後增量更新
                    if need_forward:
                        start_date = last_date + timedelta(days=1)
                        _report(f"增量更新: 從 {start_date.strftime('%Y-%m-%d')} 開始抓取")
                    elif not need_backfill:
                        _report(f"本地數據已是最新: {filename} ({len(existing_df)} 筆)")
                        return existing_df
                    else:
                        # 只有往前補抓，不需要往後抓
                        start_date = None
            except Exception as e:
                logger.warning(f"讀取本地數據失敗，將重新抓取: {e}")
                existing_df = None

        # 透過 yfinance 抓取（在 executor 中跑同步呼叫）
        # 如果 start_date=None，表示只做了往前補抓，不需要往後抓
        new_df = None
        if start_date is not None:
            _report(f"正在從 Yahoo Finance 抓取 {yf_ticker} {yf_interval}...")

            try:
                new_df = await asyncio.to_thread(
                    self._download_from_yfinance,
                    yf_ticker, yf_interval, start_date, end_date,
                )
            except Exception as e:
                _report(f"yfinance 抓取失敗: {e}")
                if existing_df is not None and not existing_df.empty:
                    new_df = None  # 繼續用現有數據
                else:
                    return pd.DataFrame()

        if new_df is None or new_df.empty:
            if existing_df is not None and not existing_df.empty:
                # 只有 backfill 或 fetch 失敗，用現有合併資料繼續
                combined = existing_df
            else:
                _report(f"yfinance 未回傳任何數據: {yf_ticker}")
                return pd.DataFrame()
        else:
            _report(f"Yahoo Finance 回傳 {len(new_df)} 筆數據")
            new_df = self._to_standard_format(new_df)

        # 合併舊數據 + 新數據
        if new_df is not None and not new_df.empty:
            if existing_df is not None and not existing_df.empty:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)
            else:
                combined = new_df.sort_values("timestamp").reset_index(drop=True)

        # 計算技術指標（復用 CryptoDataEngine 的靜態方法）
        _report("計算技術指標...")
        combined = CryptoDataEngine._calculate_indicators(combined)

        # 儲存到 CSV
        combined.to_csv(filepath, index=False)
        _report(f"已儲存 {filename}: {len(combined)} 筆數據")

        # 更新元數據
        self._update_metadata(symbol, timeframe, len(combined))

        return combined

    # ─────────────────────────────────────────────
    # yfinance 下載
    # ─────────────────────────────────────────────

    @staticmethod
    def _download_from_yfinance(
        ticker: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """從 yfinance 下載數據（同步方法，由 asyncio.to_thread 包裝）"""
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                return None

            # yfinance 新版回傳 MultiIndex columns，如 ('Close', '^TWII')
            # 壓平為單層：'Close', 'High', ...
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            return df
        except Exception as e:
            logger.error(f"yfinance 下載 {ticker} 失敗: {e}")
            return None

    # ─────────────────────────────────────────────
    # 格式轉換
    # ─────────────────────────────────────────────

    def _to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """將 yfinance 資料轉為與 CryptoDataEngine 一致的 CSV 格式"""
        result = pd.DataFrame()

        # yfinance 可能用 "Date" 或 "Datetime" 作為欄位名
        date_col = "Date" if "Date" in df.columns else "Datetime"
        result["timestamp"] = pd.to_datetime(df[date_col])

        # 處理 MultiIndex columns（yfinance 有時會回傳 MultiIndex）
        def _get_col(df, name):
            if isinstance(df.columns, pd.MultiIndex):
                for col in df.columns:
                    if col[0] == name:
                        return df[col]
            return df[name]

        result["open"] = _get_col(df, "Open").values
        result["high"] = _get_col(df, "High").values
        result["low"] = _get_col(df, "Low").values
        result["close"] = _get_col(df, "Close").values
        result["volume"] = _get_col(df, "Volume").values

        # 台股單一數據源：median_price = final_price = close
        result["median_price"] = result["close"]
        result["final_price"] = result["close"]
        result["anomaly_detected"] = False
        result["anomaly_sources"] = ""
        result["data_sources_count"] = 1

        # 轉為台北時間（台股本身就是台北時間，確保格式一致）
        result["timestamp"] = result["timestamp"].dt.tz_localize(None)

        return result

    # ─────────────────────────────────────────────
    # 本地數據讀取（與 CryptoDataEngine 一致）
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
            df = CryptoDataEngine._validate_csv(filepath)
            if df.empty:
                return df

            df["timestamp"] = pd.to_datetime(df["timestamp"])

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
        """列出所有本地可用的台股數據檔案"""
        result = []
        metadata = self._load_metadata()

        for f in self.data_dir.glob("*_TWD_*.csv"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                try:
                    with open(f) as fp:
                        total_rows = sum(1 for _ in fp) - 1
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

    # ─────────────────────────────────────────────
    # sync_metadata.json 管理（與 CryptoDataEngine 共用）
    # ─────────────────────────────────────────────

    def _load_metadata(self) -> dict:
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self, metadata: dict):
        from app.utils.timezone import taipei_now
        metadata["updated_at"] = taipei_now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _update_metadata(self, symbol: str, timeframe: str, total_bars: int):
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


# 全域單例
tw_stock_engine = TwStockEngine()
