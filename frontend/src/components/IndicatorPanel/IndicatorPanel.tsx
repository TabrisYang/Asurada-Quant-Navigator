/** 阿斯拉量化系統 — 指標面板組件 */

import { useState, useRef, useEffect } from 'react';
import { useChartStore } from '../../stores/chartStore';
import type { ActiveIndicator } from '../../types';
import { calculateIndicator } from '../../services/api';

// ─── 指標參數定義型別 ────────────────────────
interface ParamConfig {
  default: number;
  min: number;
  max: number;
  label?: string; // 參數含義標籤
}

interface IndicatorDef {
  id: string;
  name: string;
  type: string;
  display: 'overlay' | 'sub_chart';
  desc: string;   // 指標定義
  tip: string;     // 實戰建議
  params: Record<string, ParamConfig>;
}

// ─── 指標分類與預設定義（含說明）────────────────
const INDICATOR_CATEGORIES: { name: string; indicators: IndicatorDef[] }[] = [
  {
    name: '動能與趨勢',
    indicators: [
      {
        id: 'sma', name: 'SMA 移動平均', type: 'SMA', display: 'overlay',
        desc: '判斷市場長短期的平均成本方向',
        tip: '均線黃金交叉常有延遲，建議搭配斜率使用',
        params: { period: { default: 20, min: 5, max: 200, label: '計算週期（天數）' } },
      },
      {
        id: 'ema', name: 'EMA 指數移動平均', type: 'EMA', display: 'overlay',
        desc: '對近期價格賦予更高權重的移動平均',
        tip: 'EMA 比 SMA 更靈敏，適合追蹤快速趨勢',
        params: { period: { default: 20, min: 5, max: 200, label: '計算週期（天數）' } },
      },
      {
        id: 'adx', name: 'ADX 趨勢強度', type: 'ADX', display: 'sub_chart',
        desc: '衡量趨勢的強度而非方向。ADX > 25 代表趨勢啟動',
        tip: '避免在 ADX 低時使用趨勢策略，否則會被雙向洗盤',
        params: { period: { default: 14, min: 5, max: 50, label: '計算週期' } },
      },
      {
        id: 'vwap', name: 'VWAP 成交量加權均價', type: 'VWAP', display: 'overlay',
        desc: '以成交量為權重的平均價格，機構交易者的核心基準線',
        tip: '價格在 VWAP 上方代表多頭佔優，下方代表空頭佔優',
        params: { anchor: { default: 20, min: 1, max: 200, label: '錨定週期（天數）' } },
      },
      {
        id: 'ichimoku', name: 'Ichimoku 一目均衡表', type: 'ICHIMOKU', display: 'overlay',
        desc: '日本經典多功能趨勢系統，同時提供支撐/阻力/趨勢方向',
        tip: '價格在雲層上方做多、下方做空；雲層厚度代表支撐/阻力強度',
        params: {
          tenkan: { default: 9, min: 5, max: 30, label: '轉換線週期' },
          kijun: { default: 26, min: 10, max: 60, label: '基準線週期' },
          senkou_b: { default: 52, min: 20, max: 120, label: '先行帶 B 週期' },
        },
      },
      {
        id: 'psar', name: 'Parabolic SAR 拋物線轉向', type: 'PSAR', display: 'overlay',
        desc: '隨趨勢自動調整止損點，反轉時發出明確的進出場訊號',
        tip: 'SAR 點從下方翻到上方 = 空頭訊號，反之 = 多頭訊號',
        params: {
          af_start: { default: 0.02, min: 0.005, max: 0.1, label: '加速因子初始值' },
          af_max: { default: 0.2, min: 0.05, max: 0.5, label: '加速因子上限' },
        },
      },
      {
        id: 'supertrend', name: 'Supertrend 超級趨勢', type: 'SUPERTREND', display: 'overlay',
        desc: '基於 ATR 的趨勢追蹤指標，提供明確的多空方向',
        tip: '趨勢線從下方翻到上方時買入，反之賣出；適合趨勢交易',
        params: {
          period: { default: 10, min: 5, max: 50, label: 'ATR 週期' },
          multiplier: { default: 3.0, min: 1.0, max: 6.0, label: 'ATR 倍數' },
        },
      },
    ],
  },
  {
    name: '均值回歸',
    indicators: [
      {
        id: 'rsi', name: 'RSI 相對強弱', type: 'RSI', display: 'sub_chart',
        desc: '衡量價格漲跌速度，判斷超買超賣。0-100 範圍',
        tip: '加密貨幣波動大，RSI 常會鈍化（在 80 以上待很久）',
        params: {
          period: { default: 14, min: 2, max: 100, label: '計算週期' },
          overbought: { default: 70, min: 50, max: 95, label: '超買線（高於此值=超買）' },
          oversold: { default: 30, min: 5, max: 50, label: '超賣線（低於此值=超賣）' },
        },
      },
      {
        id: 'bias', name: '乖離率', type: 'BIAS', display: 'sub_chart',
        desc: '衡量價格偏離移動平均線的程度（百分比）',
        tip: '用來抓 BTC 的短線暴跌反彈（抄底）非常有效',
        params: { period: { default: 20, min: 5, max: 120, label: '均線週期' } },
      },
      {
        id: 'stochrsi', name: 'StochRSI 隨機強弱', type: 'STOCHRSI', display: 'sub_chart',
        desc: '在 RSI 上再套用隨機指標，比 RSI 更敏感的超買超賣判斷',
        tip: 'StochRSI 在 0.2 以下是超賣、0.8 以上是超買，比 RSI 反應更快',
        params: {
          rsi_period: { default: 14, min: 5, max: 50, label: 'RSI 週期' },
          stoch_period: { default: 14, min: 5, max: 50, label: 'Stochastic 週期' },
          k_smooth: { default: 3, min: 1, max: 10, label: '%K 平滑' },
          d_smooth: { default: 3, min: 1, max: 10, label: '%D 平滑' },
        },
      },
    ],
  },
  {
    name: '波動率',
    indicators: [
      {
        id: 'bb', name: '布林帶', type: 'BB', display: 'overlay',
        desc: '根據標準差定義價格的波動範圍',
        tip: 'Squeeze（擠壓）狀態是量化交易員最愛的爆發訊號',
        params: {
          period: { default: 20, min: 5, max: 200, label: '均線週期' },
          std_dev: { default: 2, min: 0.5, max: 4, label: '標準差倍數（越大通道越寬）' },
        },
      },
      {
        id: 'atr', name: 'ATR 真實波幅', type: 'ATR', display: 'sub_chart',
        desc: '衡量市場波動的絕對水平',
        tip: '主要用於設定止損（如止損設在 2 倍 ATR 處）',
        params: { period: { default: 14, min: 5, max: 50, label: '計算週期' } },
      },
      {
        id: 'keltner', name: 'Keltner 通道', type: 'KELTNER', display: 'overlay',
        desc: '結合均線與 ATR，比布林帶更貼合趨勢',
        tip: '突破通道通常是真突破，比布林帶更貼合趨勢',
        params: {
          ema_period: { default: 20, min: 10, max: 50, label: 'EMA 週期' },
          atr_multiplier: { default: 2, min: 1, max: 3, label: 'ATR 倍數' },
        },
      },
      {
        id: 'donchian', name: '唐奇安通道', type: 'DONCHIAN', display: 'overlay',
        desc: '經典趨勢突破指標，過去 N 天的最高/最低價',
        tip: '突破上軌買入，跌破下軌賣出（海龜交易法則）',
        params: { period: { default: 20, min: 10, max: 55, label: '回溯天數' } },
      },
      {
        id: 'vol_switch', name: '波動性切換', type: 'VOL_SWITCH', display: 'sub_chart',
        desc: '區分市場處於低波動蓄勢還是高波動釋放',
        tip: '比率低於閾值代表波動性噴發即將來臨',
        params: {
          short_period: { default: 10, min: 5, max: 30, label: '短期窗口' },
          long_period: { default: 50, min: 20, max: 200, label: '長期窗口' },
        },
      },
    ],
  },
  {
    name: '動能',
    indicators: [
      {
        id: 'macd', name: 'MACD', type: 'MACD', display: 'sub_chart',
        desc: '指數平滑異同移動平均線，含柱狀體斜率分析',
        tip: '觀察柱狀體是否由負轉正且連續三根放大',
        params: {
          fast: { default: 12, min: 5, max: 30, label: '快線 EMA 週期' },
          slow: { default: 26, min: 15, max: 60, label: '慢線 EMA 週期' },
          signal: { default: 9, min: 5, max: 20, label: '訊號線週期' },
        },
      },
      {
        id: 'roc', name: 'ROC 變動率', type: 'ROC', display: 'sub_chart',
        desc: '衡量價格在 N 天內的純粹漲跌幅',
        tip: '尋找 ROC 曲線斜率陡峭上升的區段',
        params: { period: { default: 14, min: 5, max: 60, label: '回溯天數' } },
      },
    ],
  },
  {
    name: '成交量',
    indicators: [
      {
        id: 'rel_vol', name: '爆量突破', type: 'REL_VOL', display: 'sub_chart',
        desc: '驗證價格變動是否有真實資金支持',
        tip: '價格突破但沒爆量，通常是假突破（誘多/誘空）',
        params: {
          period: { default: 20, min: 5, max: 60, label: '均量週期' },
          threshold: { default: 2, min: 1.5, max: 5, label: '倍率閾值（>此值=爆量）' },
        },
      },
      {
        id: 'obv', name: 'OBV 能量潮', type: 'OBV', display: 'sub_chart',
        desc: '將成交量數量化，觀察資金流向',
        tip: 'OBV 先創新高而價格沒動，通常代表即將補漲',
        params: {},
      },
    ],
  },
  {
    name: '市場情緒',
    indicators: [
      {
        id: 'funding', name: '資金費率', type: 'FUNDING', display: 'sub_chart',
        desc: '永續合約多空力量的均衡費率',
        tip: '費率極端時反向操作勝率較高',
        params: { alert_threshold: { default: 0.05, min: 0.01, max: 0.2, label: '警報閾值（%）' } },
      },
      {
        id: 'fear_greed', name: '恐懼貪婪指數', type: 'FEAR_GREED', display: 'sub_chart',
        desc: '綜合市場情緒的 0-100 指數',
        tip: '極端恐懼時買入，極端貪婪時賣出',
        params: {},
      },
    ],
  },
];

export default function IndicatorPanel() {
  const [expandedCategory, setExpandedCategory] = useState<string | null>('動能與趨勢');
  const { activeIndicators, addIndicator, removeIndicator, updateIndicatorParams, toggleIndicatorVisibility, clearIndicators } = useChartStore();

  const ohlcvData = useChartStore((s) => s.ohlcvData);
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const startDate = useChartStore((s) => s.startDate);
  const endDate = useChartStore((s) => s.endDate);
  const updateIndicatorData = useChartStore((s) => s.updateIndicatorData);

  // 自動載入缺少數據的指標（核心機制：不依賴 handleToggle）
  const loadingRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (ohlcvData.length === 0) return;

    const indicatorsNeedingData = activeIndicators.filter(
      (ind) => !ind.data && !loadingRef.current.has(ind.id)
    );

    for (const ind of indicatorsNeedingData) {
      loadingRef.current.add(ind.id);

      calculateIndicator(ind.indicator_type, 'add', {
        ...ind.parameters,
        symbol,
        timeframe,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      })
        .then((res) => {
          if (res?.data) {
            updateIndicatorData(ind.id, res.data);
          }
        })
        .catch((err) => {
          console.error(`指標載入失敗: ${ind.indicator_type}`, err?.message);
        })
        .finally(() => {
          loadingRef.current.delete(ind.id);
        });
    }
  }, [activeIndicators, ohlcvData.length, symbol, timeframe, startDate, endDate, updateIndicatorData]);

  const isActive = (id: string) => activeIndicators.some((i) => i.id === id);

  const handleToggle = (indicator: IndicatorDef) => {
    if (isActive(indicator.id)) {
      removeIndicator(indicator.id);
    } else {
      const defaultParams: Record<string, number> = {};
      Object.entries(indicator.params).forEach(([key, val]) => {
        defaultParams[key] = val.default;
      });

      const newIndicator: ActiveIndicator = {
        id: indicator.id,
        indicator_type: indicator.type,
        parameters: defaultParams,
        display_mode: indicator.display,
        visible: true,
      };
      addIndicator(newIndicator);
      // 數據載入由上方的 useEffect 自動偵測並處理
    }
  };

  const paramTimerRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const handleParamChange = (indicatorId: string, paramKey: string, value: number) => {
    updateIndicatorParams(indicatorId, { [paramKey]: value });

    // 防抖：參數變更 500ms 後重新計算指標
    if (paramTimerRef.current[indicatorId]) {
      clearTimeout(paramTimerRef.current[indicatorId]);
    }
    paramTimerRef.current[indicatorId] = setTimeout(() => {
      const ind = useChartStore.getState().activeIndicators.find((i) => i.id === indicatorId);
      if (ind) {
        // 清除舊數據，觸發 useEffect 重新載入
        useChartStore.getState().updateIndicatorData(indicatorId, undefined as any);
      }
    }, 500);
  };

  return (
    <div className="flex flex-col h-full">
      {/* 標題 */}
      <div
        className="px-4 py-3 border-b"
        style={{ borderColor: 'var(--border-color)' }}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            技術指標
          </h2>
          {activeIndicators.length > 0 && (
            <button
              onClick={() => clearIndicators()}
              className="text-xs cursor-pointer px-2 py-0.5 rounded transition-opacity hover:opacity-80"
              style={{
                color: 'var(--accent-red, #f85149)',
                background: 'rgba(248, 81, 73, 0.1)',
              }}
            >
              清除全部
            </button>
          )}
        </div>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
          已啟用 {activeIndicators.length} 項
        </p>
      </div>

      {/* 指標分類列表 */}
      <div className="flex-1 overflow-y-auto">
        {INDICATOR_CATEGORIES.map((category) => (
          <div key={category.name}>
            {/* 分類標題 */}
            <button
              onClick={() =>
                setExpandedCategory(expandedCategory === category.name ? null : category.name)
              }
              className="w-full flex items-center justify-between px-4 py-2 text-xs font-medium cursor-pointer hover:opacity-80"
              style={{
                background: 'var(--bg-tertiary)',
                color: 'var(--text-secondary)',
              }}
            >
              <span>{category.name}</span>
              <span>{expandedCategory === category.name ? '▼' : '▶'}</span>
            </button>

            {/* 指標列表 */}
            {expandedCategory === category.name && (
              <div className="py-1">
                {category.indicators.map((indicator) => {
                  const active = isActive(indicator.id);
                  const activeInd = activeIndicators.find((i) => i.id === indicator.id);

                  return (
                    <div key={indicator.id} className="px-4 py-2">
                      {/* 指標名稱 + 資訊圖標 + 開關 */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 flex-1 min-w-0">
                          <button
                            onClick={() => handleToggle(indicator)}
                            className="text-xs cursor-pointer text-left truncate"
                            style={{
                              color: active ? 'var(--accent-blue)' : 'var(--text-primary)',
                            }}
                          >
                            {indicator.name}
                          </button>

                          {/* ⓘ Tooltip 資訊圖標 */}
                          <div className="relative group shrink-0">
                            <span
                              className="text-xs cursor-help opacity-40 hover:opacity-100 transition-opacity"
                              style={{ color: 'var(--accent-blue)' }}
                            >
                              ⓘ
                            </span>
                            {/* Tooltip 浮動卡片 */}
                            <div
                              className="absolute left-5 top-0 z-50 hidden group-hover:block w-56 p-3 rounded-lg shadow-xl text-xs leading-relaxed"
                              style={{
                                background: '#1c2129',
                                border: '1px solid var(--border-color)',
                                color: 'var(--text-primary)',
                              }}
                            >
                              <p className="font-semibold mb-1" style={{ color: 'var(--accent-blue)' }}>
                                {indicator.name}
                              </p>
                              <p style={{ color: 'var(--text-secondary)' }}>
                                {indicator.desc}
                              </p>
                              <div
                                className="mt-2 pt-2"
                                style={{ borderTop: '1px solid var(--border-color)' }}
                              >
                                <p className="font-medium mb-0.5" style={{ color: 'var(--accent-yellow, #d29922)' }}>
                                  實戰建議
                                </p>
                                <p style={{ color: 'var(--text-secondary)' }}>
                                  {indicator.tip}
                                </p>
                              </div>
                              {indicator.display === 'overlay' && (
                                <p className="mt-1.5 opacity-60">顯示方式：疊加在 K 線圖上</p>
                              )}
                              {indicator.display === 'sub_chart' && (
                                <p className="mt-1.5 opacity-60">顯示方式：獨立副圖</p>
                              )}
                            </div>
                          </div>
                        </div>

                        {active && (
                          <button
                            onClick={() => toggleIndicatorVisibility(indicator.id)}
                            className="text-xs cursor-pointer ml-2 shrink-0"
                            style={{
                              color: activeInd?.visible
                                ? 'var(--accent-green)'
                                : 'var(--text-secondary)',
                            }}
                          >
                            {activeInd?.visible ? '●' : '○'}
                          </button>
                        )}
                      </div>

                      {/* 參數控制：標籤 + 滑桿 + 手動輸入 */}
                      {active && Object.keys(indicator.params).length > 0 && (
                        <div className="mt-2 space-y-2">
                          {Object.entries(indicator.params).map(([key, config]) => {
                            const currentVal = activeInd?.parameters[key] ?? config.default;
                            const step = config.max <= 5 ? 0.1 : 1;
                            return (
                              <div key={key}>
                                {/* 參數名稱 + 含義標籤 */}
                                <div className="flex items-center justify-between mb-0.5">
                                  <span
                                    className="text-xs font-medium"
                                    style={{ color: 'var(--text-secondary)' }}
                                  >
                                    {key}
                                  </span>
                                  {config.label && (
                                    <span
                                      className="text-xs truncate ml-2"
                                      style={{ color: 'var(--text-secondary)', opacity: 0.6, fontSize: '10px' }}
                                    >
                                      {config.label}
                                    </span>
                                  )}
                                </div>
                                {/* 滑桿 + 數字輸入 */}
                                <div className="flex items-center gap-2">
                                  <span
                                    className="text-xs opacity-50 shrink-0"
                                    style={{ color: 'var(--text-secondary)', fontSize: '10px' }}
                                  >
                                    {config.min}
                                  </span>
                                  <input
                                    type="range"
                                    min={config.min}
                                    max={config.max}
                                    step={step}
                                    value={currentVal}
                                    onChange={(e) =>
                                      handleParamChange(indicator.id, key, Number(e.target.value))
                                    }
                                    className="flex-1 h-1 cursor-pointer accent-blue-500"
                                  />
                                  <span
                                    className="text-xs opacity-50 shrink-0"
                                    style={{ color: 'var(--text-secondary)', fontSize: '10px' }}
                                  >
                                    {config.max}
                                  </span>
                                  <input
                                    type="number"
                                    min={config.min}
                                    max={config.max}
                                    step={step}
                                    value={currentVal}
                                    onChange={(e) => {
                                      const v = Number(e.target.value);
                                      if (!isNaN(v) && v >= config.min && v <= config.max) {
                                        handleParamChange(indicator.id, key, v);
                                      }
                                    }}
                                    onBlur={(e) => {
                                      let v = Number(e.target.value);
                                      if (isNaN(v)) v = config.default;
                                      v = Math.max(config.min, Math.min(config.max, v));
                                      handleParamChange(indicator.id, key, v);
                                    }}
                                    className="w-14 text-xs text-right px-1 py-0.5 rounded border-none outline-none"
                                    style={{
                                      background: 'var(--bg-tertiary)',
                                      color: 'var(--text-primary)',
                                    }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
