/** 阿斯拉量化系統 — 互動教學面板
 *
 * 「理論 → 圖表 → 回測驗證 → 本質」四段式學習：
 *   - 理論卡逐段閱讀，卡上按鈕直接操作主圖表（套用指標 / 跑回測）
 *   - 回測走 /api/learn/run 瘦端點（全部本地歷史，與聊天回測同引擎）
 *   - 調參滑桿 + 引導實驗：改參數重跑，前後結果並排對比
 *   - 進出場點標回主 K 線圖（獨立 annotation group，重跑自動替換）
 *   - 「問 AI 導師」預填問題 + 回測上下文到聊天輸入框（配合 teaching_mode）
 */

import { useState, useEffect, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import {
  fetchLessons,
  fetchLessonDetail,
  runLessonBacktest,
  type LessonSummary,
  type LessonDetail,
  type LearnBacktestResult,
} from '../../services/api';
import { toast } from '../Toast';
import EquityCurveChart from './EquityCurveChart';
import type { ActiveIndicator, Annotation } from '../../types';

const ANNOTATION_GROUP_ID = 'learn-backtest';
const PROGRESS_STORAGE_KEY = 'asura_learn_progress';

interface LessonProgress {
  runs: number;
}

function loadProgress(): Record<string, LessonProgress> {
  try {
    return JSON.parse(localStorage.getItem(PROGRESS_STORAGE_KEY) || '{}');
  } catch {
    return {};
  }
}

function saveProgress(progress: Record<string, LessonProgress>) {
  localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(progress));
}

/** 帶著參數的回測結果（供前後對比） */
interface RunRecord {
  params: Record<string, number>;
  result: LearnBacktestResult;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function MetricCell({ label, value, unit, goodWhen }: {
  label: string;
  value: number | null;
  unit?: string;
  goodWhen?: (v: number) => boolean;
}) {
  const color =
    value === null || !goodWhen ? 'var(--text-primary)' : goodWhen(value) ? '#3fb950' : '#f85149';
  return (
    <div style={{ padding: '8px 10px', background: 'var(--bg-primary)', borderRadius: 6 }}>
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, color }}>
        {value === null ? '—' : `${value}${unit || ''}`}
      </div>
    </div>
  );
}

export default function LearnPanel() {
  const setShowLearnPanel = useChartStore((s) => s.setShowLearnPanel);
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);

  const [summaries, setSummaries] = useState<LessonSummary[] | null>(null);
  const [lesson, setLesson] = useState<LessonDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, number>>({});
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [current, setCurrent] = useState<RunRecord | null>(null);
  const [previous, setPrevious] = useState<RunRecord | null>(null);
  const [runs, setRuns] = useState(0);

  // 載入課程列表
  useEffect(() => {
    (async () => {
      try {
        const { lessons } = await fetchLessons();
        if (lessons.length === 0) {
          setLoadError('目前沒有可用課程');
          return;
        }
        setSummaries(lessons);
      } catch (err) {
        setLoadError((err as Error)?.message || '課程載入失敗');
      }
    })();
  }, []);

  /** 進入某一課（切課時重置參數與結果） */
  const selectLesson = useCallback(async (lessonId: string) => {
    setLoadError(null);
    try {
      const detail = await fetchLessonDetail(lessonId);
      setLesson(detail);
      const defaults: Record<string, number> = {};
      for (const p of detail.tunable_params) defaults[p.name] = p.default;
      setParams(defaults);
      setCurrent(null);
      setPrevious(null);
      setRunError(null);
      setRuns(loadProgress()[detail.id]?.runs || 0);
    } catch (err) {
      setLoadError((err as Error)?.message || '課程載入失敗');
    }
  }, []);

  /** 回到課程列表 */
  const backToList = useCallback(() => {
    setLesson(null);
    setCurrent(null);
    setPrevious(null);
    setRunError(null);
  }, []);

  /** 把課程指標（重新）套到主圖表 — 移除再新增，保證依目前參數重算 */
  const applyIndicator = useCallback(() => {
    if (!lesson) return;
    const store = useChartStore.getState();
    const id = lesson.indicator_id;
    if (store.activeIndicators.some((i) => i.id === id)) {
      store.removeIndicator(id);
    }
    // 只傳指標實際接受的參數（課程可調參數不一定與指標參數同名，如 squeeze 門檻）
    const paramSpecs = lesson.indicator_info?.parameters || {};
    const indicatorParams: Record<string, number> = {};
    for (const [key, spec] of Object.entries(paramSpecs)) {
      indicatorParams[key] = params[key] ?? spec.default;
    }
    const indicator: ActiveIndicator = {
      id,
      indicator_type: id.toUpperCase(),
      parameters: indicatorParams,
      display_mode: lesson.indicator_info?.display_mode || 'overlay',
      visible: true,
    };
    store.addIndicator(indicator);
    toast(`已套用${lesson.indicator_info?.name || lesson.indicator_id.toUpperCase()}到主圖表`, 'success');
  }, [lesson, params]);

  /** 把回測進出場點標回主 K 線圖（獨立 group，重跑自動替換舊標記） */
  const annotateTrades = useCallback((result: LearnBacktestResult) => {
    const store = useChartStore.getState();
    store.removeAnnotationGroup(ANNOTATION_GROUP_ID);
    for (const ann of result.trade_annotations) {
      const annotation: Annotation = {
        id: `learn-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        type: (ann.type || 'text_label') as Annotation['type'],
        startTime: ann.startTime || ann.time,
        price: ann.price,
        text: ann.text,
        color: ann.color || '#58a6ff',
        groupId: ANNOTATION_GROUP_ID,
        groupName: '教學回測進出場',
      };
      store.addAnnotation(annotation);
    }
  }, []);

  const runBacktest = useCallback(
    async (overrideParams?: Record<string, number>) => {
      if (!lesson || running) return;
      const effective = { ...params, ...(overrideParams || {}) };
      if (overrideParams) setParams(effective);
      setRunning(true);
      setRunError(null);
      try {
        const result = await runLessonBacktest(lesson.id, symbol, timeframe, effective);
        setCurrent((prev) => {
          if (prev) setPrevious(prev);
          return { params: effective, result };
        });
        annotateTrades(result);
        const progress = loadProgress();
        const next = (progress[lesson.id]?.runs || 0) + 1;
        progress[lesson.id] = { runs: next };
        saveProgress(progress);
        setRuns(next);
        toast(`回測完成：${result.total_trades} 筆交易，進出場點已標到 K 線圖`, 'success');
      } catch (err: unknown) {
        const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
        setRunError(detail || (err as Error)?.message || '回測失敗');
      } finally {
        setRunning(false);
      }
    },
    [lesson, running, params, symbol, timeframe, annotateTrades]
  );

  /** 預填問題（含回測上下文）到 AI 聊天輸入框 */
  const askAI = useCallback(
    (prompt: string) => {
      let context = '';
      if (current) {
        const m = current.result.metrics;
        context =
          `\n\n（我的回測上下文：${symbol} ${timeframe}，參數 ${JSON.stringify(current.params)}，` +
          `${current.result.total_trades} 筆交易、勝率 ${m.win_rate}%、獲利因子 ${m.profit_factor}、` +
          `總報酬 ${m.total_return_pct}%、最大回撤 ${m.max_drawdown_pct}%）`;
      }
      setPendingChatMessage(prompt + context);
      toast('問題已填入聊天輸入框，按送出即可', 'info');
    },
    [current, symbol, timeframe, setPendingChatMessage]
  );

  const close = () => setShowLearnPanel(false);

  const m = current?.result.metrics;
  const oos = current?.result.oos_metrics;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.55)' }}
      onClick={close}
    >
      <div
        className="flex flex-col rounded-lg overflow-hidden"
        style={{
          width: 'min(760px, 95vw)',
          height: '88vh',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 標題列 */}
        <div
          className="flex items-center justify-between px-4 py-3 border-b"
          style={{ borderColor: 'var(--border-color)' }}
        >
          <div className="flex items-center gap-2">
            {lesson && (
              <button
                onClick={backToList}
                className="cursor-pointer hover:opacity-70 px-1.5 py-0.5 rounded"
                style={{ color: 'var(--text-secondary)', fontSize: 12, border: '1px solid var(--border-color)' }}
              >
                ← 課程列表
              </button>
            )}
            <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
              📖 互動教學{lesson ? ` — ${lesson.title}` : ''}
            </span>
            {lesson && (
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', marginLeft: 4 }}>
                {lesson.difficulty} · 約 {lesson.estimated_minutes} 分鐘
                {runs > 0 && ` · 已回測 ${runs} 次`}
              </span>
            )}
          </div>
          <button
            onClick={close}
            className="cursor-pointer hover:opacity-70"
            style={{ color: 'var(--text-secondary)', fontSize: 18, lineHeight: 1 }}
          >
            ✕
          </button>
        </div>

        {/* 內容區 */}
        <div className="flex-1 overflow-y-auto px-4 py-3" style={{ color: 'var(--text-primary)' }}>
          {loadError && (
            <div style={{ color: '#f85149', fontSize: 13 }}>{loadError}</div>
          )}
          {!lesson && !loadError && !summaries && (
            <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>課程載入中...</div>
          )}

          {/* 課程列表 */}
          {!lesson && summaries && (
            <>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
                每一課都是「理論 → 疊到圖表 → 真實歷史回測 → 引導實驗」的完整循環。建議按順序學。
              </p>
              {summaries.map((s, i) => {
                const lessonRuns = loadProgress()[s.id]?.runs || 0;
                return (
                  <button
                    key={s.id}
                    onClick={() => selectLesson(s.id)}
                    className="block w-full text-left rounded-md p-3 mb-2 cursor-pointer hover:opacity-85"
                    style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
                  >
                    <div className="flex items-center justify-between">
                      <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>
                        {i + 1}. {s.title}
                      </span>
                      <span style={{ fontSize: 10.5, color: 'var(--text-secondary)', whiteSpace: 'nowrap', marginLeft: 8 }}>
                        {s.difficulty} · {s.estimated_minutes} 分鐘
                        {lessonRuns > 0 && (
                          <span style={{ color: '#3fb950' }}> · ✓ 已回測 {lessonRuns} 次</span>
                        )}
                      </span>
                    </div>
                    <div style={{ fontSize: 11.5, color: 'var(--text-secondary)', marginTop: 3 }}>{s.subtitle}</div>
                  </button>
                );
              })}
            </>
          )}

          {lesson && (
            <>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
                {lesson.subtitle}
              </p>

              {/* 理論卡 */}
              {lesson.theory_sections.map((sec) => (
                <div
                  key={sec.title}
                  className="rounded-md p-3 mb-3"
                  style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
                >
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>{sec.title}</div>
                  <div style={{ fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-line', color: 'var(--text-primary)' }}>
                    {sec.body}
                  </div>
                  {sec.key_points && sec.key_points.length > 0 && (
                    <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 11.5, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                      {sec.key_points.map((kp) => (
                        <li key={kp}>{kp}</li>
                      ))}
                    </ul>
                  )}
                  {sec.chart_action && (
                    <button
                      onClick={() => (sec.chart_action === 'apply_indicator' ? applyIndicator() : runBacktest())}
                      disabled={running}
                      className="mt-2 px-3 py-1 rounded text-xs cursor-pointer hover:opacity-85 disabled:opacity-50"
                      style={{ background: '#58a6ff', color: '#fff', fontWeight: 500 }}
                    >
                      {sec.chart_action === 'run_backtest' && running ? '回測中...' : sec.chart_action_label}
                    </button>
                  )}
                </div>
              ))}

              {/* 指標小抄（registry 元資料） */}
              {lesson.indicator_info?.pro_tip && (
                <div
                  className="rounded-md p-3 mb-3"
                  style={{ background: 'rgba(88, 166, 255, 0.08)', border: '1px solid rgba(88, 166, 255, 0.3)' }}
                >
                  <span style={{ fontSize: 11, fontWeight: 600, color: '#58a6ff' }}>💡 交易員筆記　</span>
                  <span style={{ fontSize: 11.5, color: 'var(--text-primary)' }}>{lesson.indicator_info.pro_tip}</span>
                </div>
              )}

              {/* 參數實驗室 */}
              <div
                className="rounded-md p-3 mb-3"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span style={{ fontSize: 13, fontWeight: 600 }}>🧪 參數實驗室</span>
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    回測標的：{symbol} · {timeframe}（跟隨主圖表，用全部本地歷史）
                  </span>
                </div>
                {lesson.tunable_params.map((p) => (
                  <div key={p.name} className="mb-2">
                    <div className="flex items-center justify-between" style={{ fontSize: 11.5 }}>
                      <span>
                        {p.label}：<b>{params[p.name] ?? p.default}</b>
                        {params[p.name] !== p.default && (
                          <span style={{ color: 'var(--text-secondary)' }}>（預設 {p.default}）</span>
                        )}
                      </span>
                      <span style={{ color: 'var(--text-secondary)', fontSize: 10.5 }}>{p.hint}</span>
                    </div>
                    <input
                      type="range"
                      min={p.min}
                      max={p.max}
                      step={p.step}
                      value={params[p.name] ?? p.default}
                      onChange={(e) => setParams((prev) => ({ ...prev, [p.name]: Number(e.target.value) }))}
                      style={{ width: '100%', accentColor: '#58a6ff' }}
                    />
                  </div>
                ))}
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={() => runBacktest()}
                    disabled={running}
                    className="px-3 py-1.5 rounded text-xs cursor-pointer hover:opacity-85 disabled:opacity-50"
                    style={{ background: '#3fb950', color: '#fff', fontWeight: 600 }}
                  >
                    {running ? '回測中...' : '▶ 用目前參數回測'}
                  </button>
                  <button
                    onClick={applyIndicator}
                    className="px-3 py-1.5 rounded text-xs cursor-pointer hover:opacity-85"
                    style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                  >
                    把目前參數套到圖表
                  </button>
                </div>
                {runError && (
                  <div style={{ color: '#f85149', fontSize: 11.5, marginTop: 8 }}>{runError}</div>
                )}
              </div>

              {/* 回測結果卡 */}
              {current && m && (
                <div
                  className="rounded-md p-3 mb-3"
                  style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span style={{ fontSize: 13, fontWeight: 600 }}>📊 回測結果</span>
                    <span style={{ fontSize: 10.5, color: 'var(--text-secondary)' }}>
                      {current.result.data_range.start} ~ {current.result.data_range.end}（
                      {current.result.data_range.bars} 根 · 參數 {JSON.stringify(current.params)}）
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mb-2" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                    <MetricCell label="交易次數" value={current.result.total_trades} />
                    <MetricCell label="勝率" value={num(m.win_rate)} unit="%" goodWhen={(v) => v >= 50} />
                    <MetricCell label="獲利因子 PF" value={num(m.profit_factor)} goodWhen={(v) => v >= 1} />
                    <MetricCell label="總報酬" value={num(m.total_return_pct)} unit="%" goodWhen={(v) => v > 0} />
                    <MetricCell label="最大回撤" value={num(m.max_drawdown_pct)} unit="%" goodWhen={(v) => v > -30} />
                    <MetricCell label="夏普比率" value={num(m.sharpe_ratio)} goodWhen={(v) => v > 0} />
                  </div>
                  {oos && (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
                      樣本外驗證：勝率 {oos.win_rate ?? '—'}%、PF {oos.profit_factor ?? '—'}
                      （後段未參與「調參」的數據——樣本內好看、樣本外崩壞就是過擬合的訊號）
                    </div>
                  )}
                  <EquityCurveChart points={current.result.equity_curve} />
                  {current.result.warnings.length > 0 && (
                    <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 11, color: '#d29922', lineHeight: 1.7 }}>
                      {current.result.warnings.map((w) => (
                        <li key={w}>{w}</li>
                      ))}
                    </ul>
                  )}

                  {/* 前後對比 */}
                  {previous && (
                    <div
                      className="rounded p-2 mt-2"
                      style={{ background: 'var(--bg-secondary)', fontSize: 11, color: 'var(--text-secondary)' }}
                    >
                      對比上一次（參數 {JSON.stringify(previous.params)}）：
                      交易 {previous.result.total_trades} → <b>{current.result.total_trades}</b> 筆、
                      勝率 {previous.result.metrics.win_rate}% → <b>{m.win_rate}%</b>、
                      PF {previous.result.metrics.profit_factor} → <b>{m.profit_factor}</b>、
                      總報酬 {previous.result.metrics.total_return_pct}% → <b>{m.total_return_pct}%</b>
                    </div>
                  )}

                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={close}
                      className="px-3 py-1 rounded text-xs cursor-pointer hover:opacity-85"
                      style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                    >
                      到 K 線圖上看進出場點 →
                    </button>
                  </div>
                </div>
              )}

              {/* 引導實驗 */}
              <div
                className="rounded-md p-3 mb-3"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
              >
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>🔬 引導實驗（學「本質」的關鍵）</div>
                {lesson.experiments.map((exp) => (
                  <div key={exp.id} className="mb-3">
                    <div style={{ fontSize: 12, lineHeight: 1.6 }}>{exp.question}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <button
                        onClick={() => runBacktest(exp.override)}
                        disabled={running}
                        className="px-2.5 py-0.5 rounded text-xs cursor-pointer hover:opacity-85 disabled:opacity-50"
                        style={{ background: '#a371f7', color: '#fff' }}
                      >
                        試試看（{Object.entries(exp.override).map(([k, v]) => `${k}=${v}`).join(', ')}）
                      </button>
                      {exp.insight_hint && (
                        <details style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                          <summary style={{ cursor: 'pointer' }}>提示</summary>
                          {exp.insight_hint}
                        </details>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* AI 導師 */}
              <div
                className="rounded-md p-3 mb-3"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
              >
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>🎓 問 AI 導師</div>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
                  點一下把問題（連同你剛跑的回測數據）填入右側聊天框。到「設定 → AI 教學模式」開啟後，AI 會用教學口吻解說。
                </div>
                {lesson.ask_ai_prompts.map((p) => (
                  <button
                    key={p}
                    onClick={() => askAI(p)}
                    className="block w-full text-left px-2.5 py-1.5 rounded text-xs cursor-pointer hover:opacity-80 mb-1.5"
                    style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                  >
                    💬 {p}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
