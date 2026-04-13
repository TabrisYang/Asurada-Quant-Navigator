/** 阿斯拉量化系統 — 頂部工具列 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useChartStore } from '../stores/chartStore';
import type { Timeframe } from '../types';
import { fetchTwStockName, getTwStockNameSync, setTwStockNameCache } from '../services/api';
import FactorScanPanel from './FactorScanPanel/FactorScanPanel';

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
  { label: '15m', value: '15m' },
  { label: '1H', value: '1h' },
  { label: '4H', value: '4h' },
  { label: '1D', value: '1d' },
  { label: '1W', value: '1w' },
];

// 預設交易對
const DEFAULT_CRYPTO = [
  'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
  'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT',
];

const DEFAULT_TW_STOCK = [
  'TWII/TWD',  // 加權指數
  '2330/TWD', '2317/TWD', '2454/TWD', '2412/TWD', '3008/TWD',
  '2881/TWD', '2882/TWD', '1301/TWD', '2308/TWD', '2303/TWD',
];

// 台股代碼 ↔ 名稱對照表（常見 50 檔）
const TW_STOCK_DB: { code: string; name: string }[] = [
  { code: 'TWII', name: '加權指數' }, { code: 'TWOII', name: '櫃買指數' },
  { code: '2330', name: '台積電' }, { code: '2317', name: '鴻海' },
  { code: '2454', name: '聯發科' }, { code: '2412', name: '中華電' },
  { code: '3008', name: '大立光' }, { code: '2881', name: '富邦金' },
  { code: '2882', name: '國泰金' }, { code: '1301', name: '台塑' },
  { code: '2308', name: '台達電' }, { code: '2303', name: '聯電' },
  { code: '2886', name: '兆豐金' }, { code: '2891', name: '中信金' },
  { code: '2884', name: '玉山金' }, { code: '2885', name: '元大金' },
  { code: '2887', name: '台新金' }, { code: '2880', name: '華南金' },
  { code: '2892', name: '第一金' }, { code: '2883', name: '開發金' },
  { code: '1303', name: '南亞' },   { code: '1326', name: '台化' },
  { code: '2002', name: '中鋼' },   { code: '2207', name: '和泰車' },
  { code: '2301', name: '光寶科' }, { code: '2382', name: '廣達' },
  { code: '2395', name: '研華' },   { code: '2912', name: '統一超' },
  { code: '3711', name: '日月光投控' }, { code: '2327', name: '國巨' },
  { code: '2345', name: '智邦' },   { code: '2357', name: '華碩' },
  { code: '2379', name: '瑞昱' },   { code: '2474', name: '可成' },
  { code: '2603', name: '長榮' },   { code: '2609', name: '陽明' },
  { code: '2615', name: '萬海' },   { code: '2618', name: '長榮航' },
  { code: '3034', name: '聯詠' },   { code: '3037', name: '欣興' },
  { code: '3443', name: '創意' },   { code: '3661', name: '世芯-KY' },
  { code: '4904', name: '遠傳' },   { code: '5871', name: '中租-KY' },
  { code: '5880', name: '合庫金' }, { code: '6505', name: '台塑化' },
  { code: '6669', name: '緯穎' },   { code: '2105', name: '正新' },
  { code: '1216', name: '統一' },   { code: '2049', name: '上銀' },
  { code: '4938', name: '和碩' },   { code: '2347', name: '聯強' },
];

// 建立快速查找映射
const TW_CODE_TO_NAME: Record<string, string> = {};
const TW_NAME_TO_CODE: Record<string, string> = {};
for (const { code, name } of TW_STOCK_DB) {
  TW_CODE_TO_NAME[code] = name;
  TW_NAME_TO_CODE[name] = code;
  setTwStockNameCache(code, name); // 同步到共用快取
}

const STORAGE_KEY = 'asura_custom_symbols';

interface SymbolList {
  crypto: string[];
  tw_stock: string[];
}

export function loadSymbolList(): SymbolList {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { crypto: [...DEFAULT_CRYPTO], tw_stock: [...DEFAULT_TW_STOCK] };
}

export function saveSymbolList(list: SymbolList) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

/** 把新標的加入下拉選單（如果不存在） */
export function addSymbolsToList(symbols: string[]) {
  const list = loadSymbolList();
  let changed = false;
  for (const sym of symbols) {
    if (sym.endsWith('/TWD')) {
      if (!list.tw_stock.includes(sym)) { list.tw_stock.push(sym); changed = true; }
    } else {
      if (!list.crypto.includes(sym)) { list.crypto.push(sym); changed = true; }
    }
  }
  if (changed) saveSymbolList(list);
}

function getTwDisplayName(sym: string): string {
  const code = sym.split('/')[0];
  // 先查本地靜態表
  const localName = TW_CODE_TO_NAME[code];
  if (localName) return `${code} ${localName}`;
  // 再查共用快取（含動態查詢結果）
  const cachedName = getTwStockNameSync(code);
  if (cachedName) {
    TW_CODE_TO_NAME[code] = cachedName; // 回填本地表
    return `${code} ${cachedName}`;
  }
  return code;
}

function getDisplayName(sym: string): string {
  if (sym.endsWith('/TWD')) return getTwDisplayName(sym);
  return sym;
}

/** 搜尋台股：支援代碼或名稱模糊匹配 */
function searchTwStocks(query: string): { code: string; name: string }[] {
  if (!query) return [];
  const q = query.trim();
  return TW_STOCK_DB.filter(
    (s) => s.code.includes(q) || s.name.includes(q)
  ).slice(0, 5);
}

interface TopBarProps {
  onSettingsClick: () => void;
}

export default function TopBar({ onSettingsClick }: TopBarProps) {
  const symbol = useChartStore((s) => s.symbol);
  const setSymbol = useChartStore((s) => s.setSymbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const setTimeframe = useChartStore((s) => s.setTimeframe);
  const ohlcvData = useChartStore((s) => s.ohlcvData);
  const startDate = useChartStore((s) => s.startDate);
  const endDate = useChartStore((s) => s.endDate);
  const setDateRange = useChartStore((s) => s.setDateRange);
  const setShowSyncPanel = useChartStore((s) => s.setShowSyncPanel);

  const [symbolList, setSymbolList] = useState<SymbolList>(loadSymbolList);
  const [showEditor, setShowEditor] = useState(false);
  const [newCrypto, setNewCrypto] = useState('');
  const [newTwStock, setNewTwStock] = useState('');
  const [twSuggestions, setTwSuggestions] = useState<{ code: string; name: string }[]>([]);
  const editorRef = useRef<HTMLDivElement>(null);
  const [, forceUpdate] = useState(0);

  // 動態查詢不在靜態表中的台股名稱
  const resolveTwNames = useCallback(async (codes: string[]) => {
    const unknowns = codes
      .map((sym) => sym.split('/')[0])
      .filter((c) => !TW_CODE_TO_NAME[c] && !getTwStockNameSync(c));
    if (unknowns.length === 0) return;
    const results = await Promise.all(unknowns.map((c) => fetchTwStockName(c)));
    let changed = false;
    unknowns.forEach((c, i) => {
      if (results[i]) {
        TW_CODE_TO_NAME[c] = results[i];
        changed = true;
      }
    });
    if (changed) forceUpdate((n) => n + 1); // 觸發重新渲染以顯示名稱
  }, []);

  useEffect(() => {
    resolveTwNames(symbolList.tw_stock);
  }, [symbolList.tw_stock, resolveTwNames]);

  // 點擊外部關閉編輯面板
  useEffect(() => {
    if (!showEditor) return;
    const handler = (e: MouseEvent) => {
      if (editorRef.current && !editorRef.current.contains(e.target as Node)) {
        setShowEditor(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showEditor]);

  // 台股輸入即時搜尋建議
  useEffect(() => {
    setTwSuggestions(searchTwStocks(newTwStock));
  }, [newTwStock]);

  const updateList = (next: SymbolList) => {
    setSymbolList(next);
    saveSymbolList(next);
  };

  const removeSymbol = (sym: string, type: 'crypto' | 'tw_stock') => {
    const next = { ...symbolList, [type]: symbolList[type].filter((s) => s !== sym) };
    updateList(next);
  };

  const addCrypto = () => {
    const raw = newCrypto.trim().toUpperCase();
    if (!raw) return;
    let normalized = raw;
    if (!raw.includes('/')) {
      if (raw.endsWith('USDT')) normalized = raw.slice(0, -4) + '/USDT';
      else if (raw.endsWith('USD')) normalized = raw.slice(0, -3) + '/USD';
      else normalized = raw + '/USDT';
    }
    if (!symbolList.crypto.includes(normalized)) {
      updateList({ ...symbolList, crypto: [...symbolList.crypto, normalized] });
    }
    setNewCrypto('');
  };

  const addTwStock = (code?: string) => {
    let stockCode = code;
    if (!stockCode) {
      const raw = newTwStock.trim();
      if (!raw) return;
      // 純數字 → 直接當代碼
      if (/^\d{4,6}$/.test(raw)) {
        stockCode = raw;
      } else if (/^[A-Za-z]+$/i.test(raw)) {
        // 英文字母 → 可能是指數代碼如 TWII
        stockCode = raw.toUpperCase();
      } else {
        // 中文名稱 → 查本地對照表 + 後端搜尋
        if (TW_NAME_TO_CODE[raw]) {
          stockCode = TW_NAME_TO_CODE[raw];
        } else {
          const match = TW_STOCK_DB.find((s) => s.name.startsWith(raw));
          if (match) {
            stockCode = match.code;
          } else {
            // 本地找不到 → 呼叫後端搜尋 API
            import('../services/api').then(async ({ searchTwStock, setTwStockNameCache }) => {
              const results = await searchTwStock(raw);
              if (results.length > 0) {
                const best = results[0];
                setTwStockNameCache(best.code, best.name);
                const sym = best.symbol;
                if (!symbolList.tw_stock.includes(sym)) {
                  updateList({ ...symbolList, tw_stock: [...symbolList.tw_stock, sym] });
                }
                setNewTwStock('');
              }
            });
            return;
          }
        }
      }
    }
    const sym = `${stockCode}/TWD`;
    if (!symbolList.tw_stock.includes(sym)) {
      updateList({ ...symbolList, tw_stock: [...symbolList.tw_stock, sym] });
      // 不在靜態表中 → 動態查詢名稱
      if (!TW_CODE_TO_NAME[stockCode]) {
        fetchTwStockName(stockCode).then((name) => {
          if (name) {
            TW_CODE_TO_NAME[stockCode!] = name;
            forceUpdate((n) => n + 1);
          }
        });
      }
    }
    setNewTwStock('');
    setTwSuggestions([]);
  };

  const resetDefaults = () => {
    updateList({ crypto: [...DEFAULT_CRYPTO], tw_stock: [...DEFAULT_TW_STOCK] });
  };

  const sectionStyle = {
    borderTop: '1px solid var(--border-color)',
    paddingTop: '10px',
    marginTop: '10px',
  };

  return (
    <div
      className="flex items-center gap-4 px-4 py-2 border-b"
      style={{
        borderColor: 'var(--border-color)',
        background: 'var(--bg-secondary)',
      }}
    >
      {/* 系統標題 */}
      <h1
        className="text-lg font-bold mr-4"
        style={{ color: 'var(--accent-blue)' }}
      >
        阿斯拉量化系統
      </h1>

      {/* 幣種選擇 + 編輯按鈕 */}
      <div className="relative flex items-center gap-1">
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="px-3 py-1 rounded text-sm border-none outline-none cursor-pointer"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
          }}
        >
          {symbolList.crypto.length > 0 && (
            <optgroup label="加密貨幣">
              {symbolList.crypto.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </optgroup>
          )}
          {symbolList.tw_stock.length > 0 && (
            <optgroup label="台股">
              {symbolList.tw_stock.map((s) => (
                <option key={s} value={s}>{getDisplayName(s)}</option>
              ))}
            </optgroup>
          )}
        </select>

        {/* 編輯按鈕 */}
        <button
          onClick={() => setShowEditor(!showEditor)}
          className="px-1.5 py-1 rounded text-xs cursor-pointer transition-opacity hover:opacity-80"
          style={{
            background: showEditor ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
            color: showEditor ? '#fff' : 'var(--text-secondary)',
          }}
          title="自訂交易對清單"
        >
          ✎
        </button>

        {/* 編輯面板 */}
        {showEditor && (
          <div
            ref={editorRef}
            className="absolute top-full left-0 mt-2 rounded-lg shadow-xl z-50 p-4"
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              width: '360px',
              maxHeight: '520px',
              overflowY: 'auto',
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                自訂交易對
              </h3>
              <button
                onClick={resetDefaults}
                className="text-xs cursor-pointer px-2 py-0.5 rounded"
                style={{ color: 'var(--accent-blue)', background: 'var(--bg-tertiary)' }}
              >
                還原預設
              </button>
            </div>

            {/* ===== 加密貨幣區 ===== */}
            <div>
              <div className="text-xs font-semibold mb-1.5" style={{ color: 'var(--text-primary)' }}>
                加密貨幣
              </div>
              <div className="flex gap-1.5 mb-2">
                <input
                  type="text"
                  value={newCrypto}
                  onChange={(e) => setNewCrypto(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addCrypto()}
                  placeholder="輸入代碼，如 PEPE"
                  className="flex-1 px-2.5 py-1.5 rounded text-xs border-none outline-none"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                />
                <button
                  onClick={addCrypto}
                  className="px-3 py-1.5 rounded text-xs cursor-pointer"
                  style={{ background: 'var(--accent-blue)', color: '#fff' }}
                >
                  新增
                </button>
              </div>
              <div className="flex flex-wrap gap-1">
                {symbolList.crypto.map((s) => (
                  <span
                    key={s}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  >
                    {s}
                    <button
                      onClick={() => removeSymbol(s, 'crypto')}
                      className="cursor-pointer hover:opacity-60"
                      style={{ color: 'var(--text-secondary)', fontSize: '10px' }}
                    >
                      ✕
                    </button>
                  </span>
                ))}
              </div>
            </div>

            {/* ===== 台股區 ===== */}
            <div style={sectionStyle}>
              <div className="text-xs font-semibold mb-1.5" style={{ color: 'var(--text-primary)' }}>
                台股
              </div>
              <div className="relative">
                <div className="flex gap-1.5 mb-1">
                  <input
                    type="text"
                    value={newTwStock}
                    onChange={(e) => setNewTwStock(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addTwStock()}
                    placeholder="輸入代號或公司名，如 2330 或 台積電"
                    className="flex-1 px-2.5 py-1.5 rounded text-xs border-none outline-none"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  />
                  <button
                    onClick={() => addTwStock()}
                    className="px-3 py-1.5 rounded text-xs cursor-pointer"
                    style={{ background: 'var(--accent-blue)', color: '#fff' }}
                  >
                    新增
                  </button>
                </div>
                {/* 搜尋建議下拉 */}
                {twSuggestions.length > 0 && newTwStock.trim() && (
                  <div
                    className="rounded shadow-lg mb-2 overflow-hidden"
                    style={{
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-color)',
                    }}
                  >
                    {twSuggestions.map((s) => (
                      <button
                        key={s.code}
                        onClick={() => addTwStock(s.code)}
                        className="w-full text-left px-3 py-1.5 text-xs cursor-pointer hover:brightness-125 transition-all"
                        style={{
                          color: 'var(--text-primary)',
                          background: symbolList.tw_stock.includes(`${s.code}/TWD`)
                            ? 'rgba(88, 166, 255, 0.1)'
                            : 'transparent',
                        }}
                      >
                        <span style={{ color: 'var(--accent-blue)' }}>{s.code}</span>
                        <span className="ml-2">{s.name}</span>
                        {symbolList.tw_stock.includes(`${s.code}/TWD`) && (
                          <span className="ml-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
                            (已加入)
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-1">
                {symbolList.tw_stock.map((s) => (
                  <span
                    key={s}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  >
                    {getDisplayName(s)}
                    <button
                      onClick={() => removeSymbol(s, 'tw_stock')}
                      className="cursor-pointer hover:opacity-60"
                      style={{ color: 'var(--text-secondary)', fontSize: '10px' }}
                    >
                      ✕
                    </button>
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 時間週期選擇 */}
      <div className="flex items-center gap-1">
        {TIMEFRAMES.map((tf) => {
          const isTwStock = symbol.endsWith('/TWD');
          const disabled = isTwStock && ['15m', '1h', '4h'].includes(tf.value);
          return (
            <button
              key={tf.value}
              onClick={() => !disabled && setTimeframe(tf.value)}
              disabled={disabled}
              className="px-3 py-1 rounded text-sm transition-colors"
              style={{
                background:
                  timeframe === tf.value ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                color:
                  timeframe === tf.value ? '#fff' : 'var(--text-secondary)',
                opacity: disabled ? 0.35 : 1,
                cursor: disabled ? 'not-allowed' : 'pointer',
              }}
              title={disabled ? '台股僅支援日線和週線' : undefined}
            >
              {tf.label}
            </button>
          );
        })}
      </div>

      {/* 分隔線 */}
      <div
        className="h-6 w-px mx-2"
        style={{ background: 'var(--border-color)' }}
      />

      {/* 日期範圍 */}
      <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <input
          type="datetime-local"
          lang="zh-TW"
          className="px-2 py-1 rounded border-none outline-none text-xs"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            colorScheme: 'dark',
            maxWidth: '170px',
          }}
          value={startDate || ''}
          onChange={(e) => setDateRange(e.target.value || null, endDate)}
        />
        <span>~</span>
        <input
          type="datetime-local"
          lang="zh-TW"
          className="px-2 py-1 rounded border-none outline-none text-xs"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            colorScheme: 'dark',
            maxWidth: '170px',
          }}
          value={endDate || ''}
          onChange={(e) => setDateRange(startDate, e.target.value || null)}
        />
      </div>

      {/* 數據狀態 */}
      {ohlcvData.length > 0 && (
        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          {ohlcvData.length} bars
        </span>
      )}

      {/* 右側工具 */}
      <div className="ml-auto flex items-center gap-3">
        {/* 因子掃描 */}
        <FactorScanPanel />

        {/* 同步數據按鈕 */}
        <button
          onClick={() => setShowSyncPanel(true)}
          className="px-3 py-1 rounded text-sm cursor-pointer transition-opacity hover:opacity-80 font-medium"
          style={{
            background: 'var(--accent-green)',
            color: '#fff',
          }}
        >
          同步數據
        </button>

        <button
          onClick={onSettingsClick}
          className="px-3 py-1 rounded text-sm cursor-pointer transition-opacity hover:opacity-80"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-secondary)',
          }}
        >
          設定
        </button>
      </div>
    </div>
  );
}
