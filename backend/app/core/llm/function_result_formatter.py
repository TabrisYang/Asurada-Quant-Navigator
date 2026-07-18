"""阿斯拉量化系統 — function call 結果格式化

從 chat.py 抽出降低行數（重構，行為不變）。純函式：把 execute_function_calls
的結果格式化成 LLM round2 可讀的文字摘要。僅依賴 json / loguru / app.utils.price。
"""

import json
from loguru import logger


def _json_safe_default(o):
    """json.dumps 用的 default callback：把 numpy / pandas 型別轉成 Python 原生。

    numpy.bool_ 不是 Python bool 的子類別，會讓標準 json 拋 TypeError；
    numpy 數值 / ndarray / pandas Timestamp 同樣不被原生 json 認識。
    """
    # numpy 標量
    try:
        import numpy as _np
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            f = float(o)
            return f if f == f else None  # NaN → None
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    # pandas Timestamp / Timedelta
    try:
        import pandas as _pd
        if isinstance(o, (_pd.Timestamp, _pd.Timedelta)):
            return str(o)
        if hasattr(o, "to_dict"):  # DataFrame / Series
            return o.to_dict()
    except Exception:
        pass
    # set / frozenset
    if isinstance(o, (set, frozenset)):
        return list(o)
    # 最後手段：str()
    return str(o)




def format_function_results(function_calls: list[dict], exec_result: dict) -> str:
    """將 function call 執行結果格式化為 LLM 可讀的文字摘要"""
    parts = []

    results = exec_result.get("results", [])
    chart_updates = exec_result.get("chart_updates", {})

    for i, fc in enumerate(function_calls):
        fname = fc.get("name", "unknown")
        fargs = fc.get("arguments", {})
        result = results[i] if i < len(results) else {}

        # v110：顯示 result 內的 function name 而非 fc.name
        # 主因：executor v110 之前 results 順序跟 function_calls 不對應，會把 error 訊息掛到別的 function 上
        # v110 已改 by-index 但顯示層也用 result.function 雙重保險
        actual_name = result.get("function") if isinstance(result, dict) and result.get("function") else fname
        if actual_name != fname:
            logger.warning(f"[fc_results] 函式 {i+1} fc.name={fname!r} 但 result.function={actual_name!r} (順序對應錯)")

        parts.append(f"### 函式 {i+1}: {actual_name}")
        parts.append(f"參數: {json.dumps(fargs, ensure_ascii=False, default=_json_safe_default)}")

        if "error" in result:
            parts.append(f"錯誤: {result['error']}")
        elif "result" in result:
            r = result["result"]
            # 格式化不同類型的結果
            if fname == "query_chart_data":
                cu = r.get("chart_updates", {})
                sym = cu.get("symbol", "?")
                # 計算是否為多幣查詢（超過 1 個 query_chart_data）
                chart_call_count = sum(1 for fc2 in function_calls if fc2.get("name") == "query_chart_data")
                is_multi = chart_call_count > 1

                parts.append(f"========== {sym} 價格數據 ==========")
                parts.append(f"幣種: {sym}，時間框架: {cu.get('timeframe', '?')}，"
                             f"共 {cu.get('dataPoints', 0)} 根 K 線")
                # ★ 沒有數據時強制提示 LLM 呼叫下載
                if r.get("no_data") or cu.get("dataPoints", 0) == 0:
                    hint = r.get("hint", "")
                    if hint:
                        parts.append(f"⚠️ {hint}")
                    else:
                        parts.append(f"⚠️ 沒有本地數據。請呼叫 sync_symbol_data 下載 {sym} 的數據，不要只回覆沒有數據。")
                ps = r.get("price_summary")
                if ps:
                    parts.append(f"[{sym}] 期間最高: {ps.get('period_high')} ({ps.get('period_high_date')})")
                    parts.append(f"[{sym}] 期間最低: {ps.get('period_low')} ({ps.get('period_low_date')})")
                    parts.append(f"[{sym}] 首日開盤: {ps.get('first_open')} ({ps.get('first_date')})")
                    parts.append(f"[{sym}] 最後收盤: {ps.get('last_close')} ({ps.get('last_date')})")
                    for mo in (ps.get("monthly_ohlc") or []):
                        parts.append(f"  [{sym}] {mo['m']}: H={mo['h']} L={mo['l']} C={mo['c']}")
                    daily = ps.get("daily_ohlc") or []
                    if is_multi and len(daily) > 7:
                        daily = daily[-7:]  # 多幣查詢時只保留最近 7 天，避免上下文過長混淆
                        parts.append(f"  （{sym} 僅顯示最近 7 天）")
                    for day in daily:
                        parts.append(f"  [{sym}] {day['d']}: H={day['h']} L={day['l']} C={day['c']}")
                    candles = ps.get("candles") or []
                    if is_multi and len(candles) > 10:
                        candles = candles[-10:]
                        parts.append(f"  （{sym} 僅顯示最近 10 根 K 線）")
                    for c in candles:
                        parts.append(f"  [{sym}] {c['t']}: H={c['h']} L={c['l']} C={c['c']}")
            elif fname == "find_conditions":
                parts.append(f"結果: 找到 {r.get('matched_periods', 0)} 個匹配時間點")
                if r.get("summary"):
                    parts.append(f"摘要: {r['summary']}")
            elif fname == "manage_indicator":
                parts.append(f"結果: {r.get('action', '?')} 指標 {r.get('indicator_name', r.get('indicator_id', '?'))}")
            elif fname == "suggest_indicators":
                recs = r.get("recommended", [])
                names = ", ".join(rec.get("name", rec.get("id", "?")) for rec in recs)
                parts.append(f"推薦指標: {names}")
            elif fname == "generate_scenarios":
                parts.append(f"當前價格: {r.get('current_price', '?')}")
                parts.append(f"數據量: {r.get('data_points', 0)} 根 K 線")
                for sc in r.get("scenarios", []):
                    arrow = {"bullish": "▲", "neutral": "▬", "bearish": "▼"}.get(sc.get("direction", ""), "●")
                    parts.append(f"\n{arrow} {sc['label']} ({sc.get('probability_pct', '?')})")
                    pt = sc.get("price_target", {})
                    parts.append(f"  目標區間: {pt.get('low', '?')} ~ {pt.get('high', '?')}")
                    parts.append(f"  風險: {sc.get('risk_level', '?')}")
                    if sc.get("invalidation"):
                        parts.append(f"  失效條件: {sc['invalidation']}")
                    for sig in sc.get("supporting_signals", []):
                        parts.append(f"  ↳ {sig.get('name', '?')}: {sig.get('interpretation', '')}")
                sources = r.get("signal_sources", {})
                w = sources.get("weights", {})
                if w:
                    w_str = ", ".join(f"{k}({v*100:.0f}%)" for k, v in w.items())
                    parts.append(f"\n信心來源權重: {w_str}")
                # Bucket 因子群評分
                bucket = r.get("bucket_scores", {})
                if bucket:
                    parts.append(f"\n因子群 Bucket 評分（合計 {bucket.get('total', 0)}/{bucket.get('max_possible', 10)}）→ {bucket.get('direction', '?')}")
                    for group_name, group_score in bucket.get("scores", {}).items():
                        indicator = "▲" if group_score > 0 else ("▼" if group_score < 0 else "▬")
                        parts.append(f"  {group_name}: {indicator} {group_score:+d}")
                # 附加 GMM/GARCH/HMM（基礎分析也有）
                gmm_s = r.get("gmm_regime", {})
                if gmm_s and gmm_s.get("status") == "success":
                    parts.append(f"\nGMM 市場體制: {gmm_s.get('current_regime', '?')}")
                    for rn, rp in gmm_s.get("regime_probabilities", {}).items():
                        parts.append(f"  {rn}: {rp}%")
                garch_s = r.get("garch_volatility", {})
                if garch_s and garch_s.get("status") == "success":
                    parts.append(f"GARCH 波動率: {garch_s.get('vol_direction', '?')} | 止損倍率 {garch_s.get('suggested_sl_multiplier', '?')}x")
                hmm_s = r.get("hmm_regime", {})
                if hmm_s and hmm_s.get("status") == "success":
                    parts.append(f"HMM 狀態: {hmm_s.get('current_state', '?')} | {hmm_s.get('interpretation', '?')}")
                # 歷史準確率
                ha = r.get("historical_accuracy", {})
                if ha:
                    parts.append(f"情境預測歷史準確率: {ha.get('direction_accuracy_pct', '?')}%（{ha.get('n_evaluations', '?')} 次評估）")
            elif fname == "analyze_fundamentals":
                parts.append(f"📊 基本面分析 — {r.get('code', '?')} {r.get('name', '?')}")
                # 營收
                rev = r.get("revenue", {})
                if rev.get("available"):
                    parts.append(f"\n營收趨勢: {rev.get('trend', '?')}")
                    parts.append(f"  最新月: {rev.get('latest_month', '?')} 營收={rev.get('latest_revenue', '?')}")
                    if rev.get("mom_pct") is not None:
                        parts.append(f"  MoM: {rev['mom_pct']}% | YoY: {rev.get('yoy_pct', '?')}%")
                    if rev.get("consecutive_growth_months"):
                        parts.append(f"  連續成長: {rev['consecutive_growth_months']} 個月")
                # 法人
                inst = r.get("institutional", {})
                if inst.get("available"):
                    parts.append(f"\n法人動向: {inst.get('direction', '?')}")
                    parts.append(f"  外資近20日: {inst.get('foreign_net_20d', '?')} 張")
                    parts.append(f"  投信近20日: {inst.get('trust_net_20d', '?')} 張")
                    if inst.get("foreign_consecutive_buy"):
                        parts.append(f"  外資連買: {inst['foreign_consecutive_buy']} 天")
                # 財報
                fin = r.get("financials", {})
                if fin.get("available"):
                    parts.append(f"\n財報指標:")
                    if fin.get("pe_ratio"):
                        parts.append(f"  本益比: {fin['pe_ratio']} ({fin.get('pe_assessment', '')})")
                    if fin.get("eps_trailing"):
                        parts.append(f"  EPS: {fin['eps_trailing']}")
                    if fin.get("dividend_yield"):
                        parts.append(f"  殖利率: {fin['dividend_yield']}% ({fin.get('yield_assessment', '')})")
                # 評分
                score = r.get("fundamental_score", {})
                if score:
                    parts.append(f"\n綜合基本面評分: {score.get('score', '?')}/{score.get('max_possible', 10)} → {score.get('direction', '?')}")
            elif fname == "analyze_crypto_fundamentals":
                parts.append(f"🪙 加密基本面 — {r.get('symbol', '?')} (status={r.get('status', '?')}, data_status={r.get('data_status', {})})")
                t = r.get("tokenomics", {})
                if t.get("available"):
                    parts.append(f"  代幣經濟[{t.get('source')}]: 流通={t.get('circulating_supply')}/上限={t.get('max_supply')} (circ {t.get('circ_pct')}%) "
                                 f"市值=${t.get('market_cap_usd')} FDV=${t.get('fdv_usd')} mcap/FDV={t.get('mcap_to_fdv_pct')}% | "
                                 f"{t.get('dilution_assessment') or ''}{t.get('supply_maturity') or ''} | ATH=${t.get('ath_usd')}({t.get('ath_change_pct')}%)")
                else:
                    parts.append("  代幣經濟: ⚠️ 即時資料暫不可得（請勿編造）")
                eco = r.get("ecosystem", {})
                dev, tvl = eco.get("dev_community", {}), eco.get("tvl", {})
                if dev.get("available"):
                    parts.append(f"  開發: commit_4w={dev.get('commit_count_4_weeks')} stars={dev.get('stars')} issue關閉={dev.get('issue_close_ratio_pct')}% | {eco.get('dev_activity_assessment') or ''}")
                if tvl.get("available"):
                    parts.append(f"  TVL[{tvl.get('scope')}]: ${tvl.get('tvl_usd')} (30d {tvl.get('tvl_change_30d_pct')}%) | {eco.get('tvl_trend') or ''}")
                elif tvl.get("reason") == "not_a_defi_protocol":
                    parts.append("  TVL: 此資產非 DeFi 協議，無 TVL 指標")
                parts.append(f"  🗺️ 路線圖(非即時): {r.get('roadmap_disclaimer', '')} | links={r.get('links') or {}}")
            elif fname == "sync_symbol_data":
                if r.get("status") == "success":
                    parts.append(f"✅ 數據下載完成: {r.get('symbol', '?')} {r.get('timeframe', '?')}")
                    parts.append(f"  共 {r.get('bars', 0)} 根 K 線")
                    parts.append(f"  範圍: {r.get('range', '?')}")
                else:
                    parts.append(f"❌ 下載失敗: {r.get('message', '?')}")
            elif fname == "sync_sector_data":
                parts.append(f"📦 族群批次下載: {r.get('sector_name', '?')}（{r.get('success', 0)}/{r.get('total', 0)} 檔成功）")
                for d in r.get("details", []):
                    icon = "✅" if d.get("status") == "ok" and d.get("bars", 0) > 0 else "❌"
                    parts.append(f"  {icon} {d.get('name', '?')}（{d.get('symbol', '?')}）: {d.get('bars', 0)} 根")
            elif fname == "analyze_momentum":
                parts.append(f"📊 動能分析 — {r.get('symbol', '?')} {r.get('timeframe', '?')}（{r.get('total_bars', 0)} 根）")
                # 動量因子
                mf = r.get("momentum_factors", {})
                for k, v in mf.items():
                    if k.startswith("mom_"):
                        parts.append(f"  {v.get('label', k)}: {v.get('return_pct', '?')}%")
                cm = mf.get("classic_momentum", {})
                if cm:
                    parts.append(f"  經典動量: {cm.get('value', '?')} ({cm.get('interpretation', '')})")
                cons = mf.get("consistency", {})
                if cons:
                    parts.append(f"  方向一致性: {cons.get('interpretation', '?')}")
                # 動量加速
                ms = r.get("momentum_shift", {})
                if ms:
                    parts.append(f"\n動量狀態: {ms.get('state', '?')} | ROC5={ms.get('roc_5', '?')}% | 加速度={ms.get('acceleration', '?')}")
                    if ms.get("shift_detected"):
                        parts.append(f"  ⚡ 轉折: {ms.get('shift_type', '?')}")
                # 相對動量
                rm = r.get("relative_momentum", {})
                if rm and rm.get("available"):
                    parts.append(f"\n相對動量（vs BTC）: {rm.get('interpretation', '?')}")
                    for pk, pv in rm.get("periods", {}).items():
                        parts.append(f"  {pk}: 標的 {pv.get('target_return', '?')}% vs BTC {pv.get('benchmark_return', '?')}% → RS={pv.get('relative_strength', '?')}")
                # 反轉
                rev = r.get("reversal_detection", {})
                if rev:
                    parts.append(f"\n反轉信號: {rev.get('net_direction', '?')}（多 {rev.get('bullish_signals', 0)} / 空 {rev.get('bearish_signals', 0)}）")
                    for sig in rev.get("signals", []):
                        parts.append(f"  {'🟢' if sig.get('direction') == 'bullish' else '🔴'} {sig.get('type', '?')}")
                # 策略回測
                sb = r.get("strategy_backtest", {})
                if sb:
                    parts.append(f"\n動量策略回測: 最佳={sb.get('best_strategy_name', '?')}")
                    for sk, sv in sb.get("strategies", {}).items():
                        parts.append(f"  {sv.get('name', sk)}: 勝率={sv.get('win_rate', '?')}% Sharpe={sv.get('sharpe', '?')} 報酬={sv.get('return_pct', '?')}%")
                # 綜合評分
                mscore = r.get("momentum_score", {})
                if mscore:
                    parts.append(f"\n綜合動量評分: {mscore.get('score', '?')}/{mscore.get('max_possible', 10)} → {mscore.get('direction', '?')}")
            elif fname == "detect_smc_structure":
                parts.append(f"📊 SMC 訂單流分析 — {r.get('symbol', '?')} {r.get('timeframe', '?')}")
                parts.append(f"結構方向: HTF={r.get('trend_htf', '?')} / LTF={r.get('trend_ltf', '?')} / 共振={'✅' if r.get('mtf_aligned') else '❌'}")
                # v145：價位用 format_price 補滿小數位，與 K 線圖完全一致（0.251700 而非 0.2517）
                from app.utils.price import format_price as _fp
                parts.append(f"溢價/折價: {r.get('premium_discount', '?')} (Fib 0.5 = {_fp(r.get('fib_50'))})")
                # BOS/CHoCH 事件（price=突破收盤、level_price=被破結構位，皆精確值）
                for evt in r.get("bos_points", []) + r.get("choch_points", []):
                    if evt.get("filtered"):
                        continue
                    tag = "🔴" if evt.get("type") == "CHoCH" else "🔵"
                    parts.append(
                        f"  {tag} {evt['type']} 收盤 {_fp(evt.get('price'))} 突破結構位 {_fp(evt.get('level_price'))} (量比 {evt.get('vol_ratio', '?')}x)"
                    )
                # Sweep 事件（level_price=被掃結構位、price=影線極值，皆精確值）
                for sw in r.get("sweep_events", []):
                    if not sw.get("filtered"):
                        parts.append(
                            f"  💧 {sw['type']} Sweep 結構位 {_fp(sw.get('level_price'))} 影線觸及 {_fp(sw.get('price'))} (量比 {sw.get('vol_ratio', '?')}x)"
                        )
                # Order Block（機構訂單塊，精確 [low, high] 區間）
                for ob in r.get("order_blocks", []):
                    ob_tag = "🟩" if ob.get("type") == "bullish" else "🟥"
                    tested = "已測試" if ob.get("tested") else "未測試"
                    parts.append(
                        f"  {ob_tag} {ob.get('type')} OB 區間 [{_fp(ob.get('low'))}, {_fp(ob.get('high'))}] ({tested})"
                    )
                # FVG（列出未填補缺口的精確 [low, high] 區間，非只數量）
                unfilled_fvg = [f for f in r.get("fvg_zones", []) if not f.get("filled")]
                if unfilled_fvg:
                    parts.append(f"  未填補 FVG ({len(unfilled_fvg)} 個):")
                    for fvg in unfilled_fvg[:5]:
                        parts.append(
                            f"    ⚡ {fvg.get('type')} FVG [{_fp(fvg.get('low'))}, {_fp(fvg.get('high'))}]"
                        )
                # 交易建議
                bias = r.get("bias", "WAIT")
                parts.append(f"\n建議: {bias}")
                if r.get("entry"):
                    parts.append(f"  Entry: {_fp(r.get('entry'))} / SL: {_fp(r.get('stop_loss'))} / TP: {_fp(r.get('take_profit'))}")
                    parts.append(f"  RR: {r.get('rr_ratio', '?')} / 信心: {r.get('confidence', '?')}%")
                # 信心分項
                bd = r.get("confidence_breakdown", {})
                if bd:
                    bd_str = ", ".join(f"{k}({v:+d})" for k, v in bd.items())
                    parts.append(f"  評分明細: {bd_str}")
            elif fname == "scan_conditional_probability":
                parts.append(f"目標: {r.get('target', '?')}")
                parts.append(f"數據範圍: {r.get('data_range', '?')}，共 {r.get('total_bars', 0)} 根 K 線")
                if r.get("range_notice"):
                    parts.append(f"⚠️ {r['range_notice']}")
                if r.get("status") == "error" and r.get("suggestion"):
                    parts.append(f"執行失敗：{r.get('message', '?')}")
                    parts.append(f"💡 修正建議（必須轉告使用者或修正參數重試）: {r['suggestion']}")
                ob = r.get("overall_best", {})
                if ob:
                    parts.append(f"★ 最佳區間: {ob.get('indicator', '?')} = {ob.get('range', '?')} → "
                                 f"機率 {ob.get('prob_pct', 0)}%（基線 {ob.get('baseline_pct', 0)}%，提升 {ob.get('lift', 0)} 個百分點）")
                for key, ind_data in r.get("indicators", {}).items():
                    parts.append(f"\n指標 {key}（樣本 {ind_data.get('total_valid_samples', 0)}）:")
                    parts.append(f"  基線機率: {ind_data.get('baseline_prob_pct', 0)}%")
                    parts.append(f"  最佳區間: {ind_data.get('best_range', '?')} → {ind_data.get('best_prob_pct', 0)}%")
                    for b in ind_data.get("bins", []):
                        if b.get("prob_pct") is not None:
                            parts.append(f"  {b['range']}: {b['prob_pct']}% ({b['hit']}/{b['count']})")
                        elif b.get("note"):
                            parts.append(f"  {b['range']}: {b['note']} ({b['count']})")
            elif fname == "run_quant_research":
                parts.append(f"📊 量化研究報告 — {r.get('symbol', '?')} {r.get('timeframe', '?')}（{r.get('total_bars', 0)} 根 K 線）")
                if r.get("notice"):
                    parts.append(f"⚠️ {r['notice']}")
                # 因子 IC 排名
                ranking = r.get("factor_ic", {}).get("ranking", [])
                if ranking:
                    parts.append(f"\n因子預測力排名（共分析 {r.get('factor_ic', {}).get('total_analyzed', 0)} 個因子）:")
                    for f in ranking[:8]:
                        parts.append(f"  {f.get('factor', '?')}: IC={f.get('best_ic', '?')} [{f.get('power', '?')}] decay={f.get('decay_trend', '?')}")
                # 因子相關性
                corr = r.get("factor_correlation", {})
                if corr.get("recommendation"):
                    parts.append(f"因子相關性: {corr['recommendation']}")
                # 組合因子 IC（combo_top）
                fscan = r.get("factor_scan", {})
                combo_top = fscan.get("combo_top", [])
                if combo_top and isinstance(combo_top, list):
                    parts.append(f"\n雙因子組合 IC（top {min(5, len(combo_top))}）:")
                    for combo in combo_top[:5]:
                        if isinstance(combo, dict):
                            parts.append(f"  {combo.get('factor_a', '?')} + {combo.get('factor_b', '?')}: combo_IC={combo.get('combo_ic', '?')}")
                # Bucket 因子群評分
                bucket_qr = r.get("bucket_scores", {})
                if bucket_qr:
                    parts.append(f"\n因子群 Bucket 評分（合計 {bucket_qr.get('total', 0)}/{bucket_qr.get('max_possible', 10)}）→ {bucket_qr.get('direction', '?')}")
                    for grp_name, grp_score in bucket_qr.get("scores", {}).items():
                        ind = "▲" if grp_score > 0 else ("▼" if grp_score < 0 else "▬")
                        parts.append(f"  {grp_name}: {ind} {grp_score:+d}")
                # 系統候選策略（未提供策略時自動生成，in-sample，須由 WF/CPCV 裁判）
                ads = r.get("auto_derived_strategy")
                if ads:
                    parts.append(
                        f"\n🧪 系統候選策略（{ads.get('source')}，主因子 {ads.get('top_factor')}）："
                        f"進場區間 {ads.get('entry_range')}（歷史命中 {ads.get('hist_prob_pct')}% vs 基線 {ads.get('baseline_pct')}%，"
                        f"lift +{ads.get('lift_pp')}pp、n={ads.get('samples')}）方向={ads.get('direction')} 預設SL/TP={ads.get('default_sl_tp')}"
                    )
                    parts.append("  ⚠️ 此為系統自動建議、in-sample 回測 — 是否真有 alpha 以下方 Walk Forward / CPCV 樣本外結果為準（一致性低=過擬合、不可實盤）")
                # 回測
                bt = r.get("backtest", {})
                if bt and not bt.get("error"):
                    parts.append("\n回測績效:")
                    parts.append(f"  勝率: {bt.get('win_rate', '?')}% | PF: {bt.get('profit_factor', '?')} | 交易: {bt.get('total_trades', '?')} 筆")
                    parts.append(f"  Sharpe: {bt.get('sharpe_ratio', '?')} | Sortino: {bt.get('sortino_ratio', '?')}")
                    parts.append(f"  Expectancy: {bt.get('expectancy_pct', '?')}% | 最大回撤: {bt.get('max_drawdown_pct', '?')}%")
                    parts.append(f"  總報酬: {bt.get('total_return_pct', '?')}%")
                # Monte Carlo
                mc = r.get("monte_carlo", {})
                if mc and mc.get("status") == "success":
                    parts.append("\nMonte Carlo 模擬:")
                    parts.append(f"  獲利機率: {mc.get('profit_probability', '?')}% | 破產風險: {mc.get('ruin_probability', '?')}%")
                    parts.append(f"  報酬分布 p25={mc.get('p25_return', '?')}% / p50={mc.get('p50_return', '?')}% / p75={mc.get('p75_return', '?')}%")
                    if mc.get("reliability") == "low":
                        parts.append("  策略穩健: ⚠️ 樣本不足（<30 筆），破產率/穩健性僅供研究參考、不可作實盤定論")
                    else:
                        parts.append(f"  策略穩健: {'✅' if mc.get('strategy_robust') else '❌'}")
                # Walk Forward
                wf = r.get("walk_forward", {})
                if wf:
                    assessment = wf.get("assessment", {})
                    summary = wf.get("summary", {})
                    if assessment:
                        parts.append("\nWalk Forward 驗證:")
                        if assessment.get("reliability") == "low":
                            parts.append(f"  Alpha: ⚠️ 視窗數不足（{assessment.get('n_windows', '?')}<4），結論僅供研究參考 | 評分: {assessment.get('score', '?')}/100")
                        else:
                            parts.append(f"  Alpha: {'✅' if assessment.get('has_alpha') else '❌'} | 評分: {assessment.get('score', '?')}/100")
                        if summary:
                            parts.append(f"  視窗數: {summary.get('n_windows', '?')} | OOS 平均報酬: {summary.get('avg_oos_return', '?')}%")
                # GMM Regime
                gmm = r.get("gmm_regime", {})
                if gmm and gmm.get("status") == "success":
                    parts.append(f"\nGMM 市場體制分類:")
                    parts.append(f"  當前 Regime: {gmm.get('current_regime', '?')}")
                    for rname, prob in gmm.get("regime_probabilities", {}).items():
                        parts.append(f"  {rname}: {prob}%")
                    for rname, rstats in gmm.get("regime_stats", {}).items():
                        parts.append(f"  {rname}: 歷史佔比 {rstats.get('pct', '?')}%, 平均報酬 {rstats.get('avg_return', '?')}%")
                # GARCH 波動率
                garch = r.get("garch_volatility", {})
                if garch and garch.get("status") == "success":
                    parts.append(f"\nGARCH 波動率預測:")
                    parts.append(f"  當前波動率: {garch.get('current_volatility', '?')}")
                    parts.append(f"  方向: {garch.get('vol_direction', '?')} | 止損倍率建議: {garch.get('suggested_sl_multiplier', '?')}x")
                    parts.append(f"  解讀: {garch.get('interpretation', '?')}")
                # HMM 狀態轉移
                hmm = r.get("hmm_regime", {})
                if hmm and hmm.get("status") == "success":
                    parts.append(f"\nHMM 狀態轉移:")
                    parts.append(f"  當前狀態: {hmm.get('current_state', '?')}")
                    parts.append(f"  解讀: {hmm.get('interpretation', '?')}")
                    for state_name, dur in hmm.get("expected_duration_bars", {}).items():
                        parts.append(f"  {state_name}: 預期持續 {dur} 根 K 線")
                # CPCV
                cpcv_data = r.get("cpcv", {})
                if cpcv_data:
                    cpcv_assess = cpcv_data.get("assessment", {})
                    cpcv_met = cpcv_data.get("metrics_distribution", {})
                    if cpcv_assess:
                        parts.append(f"\nCPCV 組合淨化交叉驗證（{cpcv_data.get('n_combinations', '?')} 組合）:")
                        parts.append(f"  一致性: {cpcv_met.get('consistency_pct', '?')}%")
                        cpcv_ret = cpcv_met.get("return_pct", {})
                        parts.append(f"  報酬: 均值={cpcv_ret.get('mean', '?')}%, P25={cpcv_ret.get('p25', '?')}%, 中位數={cpcv_ret.get('median', '?')}%")
                        if cpcv_assess.get("reliability") == "low":
                            parts.append(f"  評分: {cpcv_assess.get('score', '?')}/100 | 有邊際: ⚠️ 組合數不足（{cpcv_assess.get('n_combos', '?')}<5），僅供研究參考")
                        else:
                            parts.append(f"  評分: {cpcv_assess.get('score', '?')}/100 | 有邊際: {'✅' if cpcv_assess.get('has_edge') else '❌'}")
                        parts.append(f"  結論: {cpcv_assess.get('verdict', '?')}")
                # OOS Monte Carlo
                mc_oos = r.get("monte_carlo_oos", {})
                if mc_oos and mc_oos.get("status") == "success":
                    parts.append(f"\nOOS Monte Carlo（Walk Forward OOS 交易）:")
                    parts.append(f"  獲利機率: {mc_oos.get('profit_probability', '?')}% | 破產風險: {mc_oos.get('ruin_probability', '?')}%")
                    parts.append(f"  策略穩健: {'✅' if mc_oos.get('strategy_robust') else '❌'}")
                # 倉位建議
                pos = r.get("position_sizing", {})
                if pos and not pos.get("error"):
                    parts.append(f"\n倉位建議: {pos.get('recommendation', pos.get('summary', '?'))}")
                    mc_adj = pos.get("mc_adjustment", {})
                    if mc_adj:
                        parts.append(f"  MC 調整: {mc_adj.get('reason', '?')}")
                # 結論
                conclusion = r.get("conclusion", {})
                if conclusion:
                    parts.append(f"\n結論（綜合評分: {conclusion.get('score', '?')}/100）:")
                    for finding in conclusion.get("findings", []):
                        parts.append(f"  {finding}")
                    if conclusion.get("recommendation"):
                        parts.append(f"  建議: {conclusion['recommendation']}")
            elif fname == "run_backtest":
                m = r.get("metrics", {})
                parts.append(f"回測結果（{r.get('total_trades', 0)} 筆交易）:")
                parts.append(f"  勝率: {m.get('win_rate', '?')}% | PF: {m.get('profit_factor', '?')} | Sharpe: {m.get('sharpe_ratio', '?')}")
                parts.append(f"  總報酬: {m.get('total_return_pct', '?')}% | 最大回撤: {m.get('max_drawdown_pct', '?')}%")
                parts.append(f"  Sortino: {m.get('sortino_ratio', '?')} | Expectancy: {m.get('expectancy_pct', '?')}%")
                if r.get("warnings"):
                    parts.append(f"  ⚠️ 警告: {'; '.join(r['warnings'][:3])}")
            elif fname == "compare_strategies":
                parts.append(f"策略比較（{r.get('symbol', '?')} {r.get('timeframe', '?')}，共 {r.get('total_strategies', 0)} 個策略）:")
                for c in r.get("comparison", []):
                    if c.get("status") == "success":
                        m = c.get("metrics", {})
                        rank = c.get("rank", "?")
                        parts.append(f"  #{rank} {c['name']}: 勝率={m.get('win_rate', '?')}% PF={m.get('profit_factor', '?')} "
                                     f"Sharpe={m.get('sharpe_ratio', '?')} 報酬={m.get('total_return_pct', '?')}%")
                    else:
                        parts.append(f"  ✗ {c.get('name', '?')}: {c.get('message', '錯誤')}")
            elif fname == "compute_laddered_entries":
                # v106 B1：分批進場專用 formatter（取代 raw JSON dump）
                if not r.get("enabled"):
                    parts.append(f"分批進場：disabled — {r.get('warning', '?')}")
                    if r.get("sl_mult_hint"):
                        parts.append(f"  建議 SL ≈ {r['sl_mult_hint']} × {r.get('timeframe_used','?')} ATR")
                        parts.append(f"  建議 TP ≈ {r.get('tp_mult_hint','?')} × {r.get('timeframe_used','?')} ATR")
                else:
                    parts.append(
                        f"分批進場（regime={r.get('regime_used','?')} {r.get('regime_confidence',0):.2f} | "
                        f"配比 {r.get('ratio_strategy','?')}）"
                    )
                    parts.append(f"  ATR: {r.get('atr_used','?')} | 當前: {r.get('current_price','?')}")
                    if r.get("long_entries"):
                        parts.append("  Long entries:")
                        for e in r["long_entries"]:
                            parts.append(f"    {e['size_pct']}% @ ${e['price']} ({e.get('source','')})")
                        parts.append(
                            f"    avg=${r.get('weighted_avg_entry_long','?')} | "
                            f"SL=${r.get('stop_loss_long','?')} (≈{r.get('sl_mult_used','?')}×ATR) | "
                            f"TP=${r.get('take_profit_long','?')} (≈{r.get('tp_mult_used','?')}×ATR) | "
                            f"RR={r.get('rr_long','?')}"
                        )
                    if r.get("short_entries"):
                        parts.append("  Short entries:")
                        for e in r["short_entries"]:
                            parts.append(f"    {e['size_pct']}% @ ${e['price']} ({e.get('source','')})")
                        parts.append(
                            f"    avg=${r.get('weighted_avg_entry_short','?')} | "
                            f"SL=${r.get('stop_loss_short','?')} | "
                            f"TP=${r.get('take_profit_short','?')} | "
                            f"RR={r.get('rr_short','?')}"
                        )
                    if r.get("missing_indicators"):
                        parts.append(f"  ⚠️ 缺失指標: {', '.join(r['missing_indicators'])}")
            elif fname == "analyze_sector":
                # v106 B1：族群分析專用 formatter
                parts.append(f"族群分析：{r.get('sector_name', '?')}")
                if r.get("status") == "error":
                    parts.append(f"  錯誤：{r.get('message', '?')}")
                else:
                    parts.append(f"  成員: {r.get('member_count', '?')} 檔")
                    parts.append(f"  廣度: {r.get('breadth_pct', '?')}% advancing")
                    if r.get("leader"):
                        parts.append(f"  龍頭: {r['leader']} (5K 線 {r.get('leader_change_5d','?')}%)")
                    if r.get("interpretation"):
                        parts.append(f"  解讀: {r['interpretation'][:300]}")
            else:
                # 通用格式化（截斷過長內容）
                # v105.7：limit 3000→8000，且改為「前 6000 + 後 2000」截斷
                result_str = json.dumps(r, ensure_ascii=False, default=_json_safe_default)
                if len(result_str) > 8000:
                    result_str = (
                        result_str[:6000]
                        + f"\n\n... [中段省略 {len(result_str) - 8000} 字以節省 token] ...\n\n"
                        + result_str[-2000:]
                    )
                parts.append(f"結果: {result_str}")
        parts.append("")

    if chart_updates:
        parts.append(f"圖表更新: {json.dumps(chart_updates, ensure_ascii=False, default=_json_safe_default)[:300]}")

    return "\n".join(parts)
