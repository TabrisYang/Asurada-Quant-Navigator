"""阿斯拉量化系統 — LLM 智能分析報告匯出

流程：收集知識碎片 + 蒸餾知識 + 近期對話 → 按幣種整理 → LLM 生成結構化報告 → fpdf2 渲染 PDF
報告結構：封面 → 執行摘要 → 各標的深度分析 → 關鍵價位表 → 附錄
"""

import asyncio
import io
import json
import os
import re
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from loguru import logger

from app.core.chat_history import chat_history
from app.core.knowledge_distiller import knowledge_distiller
from app.core.knowledge_fragments import fragment_store
from app.core.semantic_cache import semantic_cache
from app.core.usage_tracker import usage_tracker

router = APIRouter()

# ────────────────────────────────────────────
#  色彩常數 (R, G, B)
# ────────────────────────────────────────────

_C_PRIMARY = (15, 52, 96)
_C_DARK = (22, 33, 62)
_C_TEXT = (26, 26, 46)
_C_GRAY = (136, 136, 136)
_C_LIGHT_GRAY = (200, 200, 200)
_C_BG_LIGHT = (248, 249, 250)
_C_BG_BLUE = (232, 240, 254)
_C_BG_GRAY = (240, 240, 240)
_C_BG_BOX = (240, 244, 255)
_C_WHITE = (255, 255, 255)
_C_GREEN_BG = (230, 245, 230)
_C_RED_BG = (250, 230, 230)
_C_YELLOW_BG = (255, 248, 225)

_TYPE_LABELS = {
    "support_resistance": "支撐/壓力",
    "trend": "趨勢",
    "pattern": "型態",
    "indicator": "指標",
    "strategy": "策略",
    "volume": "量能",
    "sentiment": "情緒",
    "general": "通用",
}

# ────────────────────────────────────────────
#  CJK 字型
# ────────────────────────────────────────────

_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    # Windows
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    # Linux
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def _find_cjk_font() -> Optional[str]:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


# ────────────────────────────────────────────
#  資料收集
# ────────────────────────────────────────────

def _get_all_fragments() -> list[dict]:
    conn = fragment_store._conn
    if not conn:
        return []
    try:
        rows = conn.execute(
            """SELECT content, fragment_type, symbol, source_question,
                      hit_count, datetime(created_at, 'unixepoch', 'localtime') as created
               FROM fragments
               ORDER BY symbol, hit_count DESC, created_at DESC"""
        ).fetchall()
        return [
            {"content": r[0], "type": r[1], "symbol": r[2],
             "source_question": r[3], "hit_count": r[4], "created_at": r[5]}
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"讀取知識碎片失敗: {e}")
        return []


def _get_recent_analyses(limit: int = 30) -> list[dict]:
    conversations = chat_history.list_conversations(limit=50)
    records: list[dict] = []
    for conv in conversations:
        msgs = chat_history.get_conversation_messages(conv["id"], limit=50)
        pairs: list[dict] = []
        i = 0
        while i < len(msgs):
            if msgs[i]["role"] == "user":
                user_msg = msgs[i]
                assistant_msg = None
                if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                    assistant_msg = msgs[i + 1]
                    i += 2
                else:
                    i += 1
                if assistant_msg and len(assistant_msg.get("content", "")) > 100:
                    pairs.append({
                        "user": user_msg.get("content", ""),
                        "assistant": assistant_msg.get("content", ""),
                        "timestamp": assistant_msg.get("timestamp", ""),
                    })
            else:
                i += 1
        if pairs:
            records.append({
                "title": conv.get("title", "對話"),
                "symbol": conv.get("symbol", ""),
                "timeframe": conv.get("timeframe", ""),
                "pairs": pairs[-5:],
            })
        if len(records) >= limit:
            break
    return records


def _group_data_by_symbol(
    fragments: list[dict],
    all_knowledge: list[dict],
    analyses: list[dict],
) -> dict[str, dict]:
    """將所有資料按幣種分組，回傳 {symbol: {fragments, knowledge, analyses}}。"""
    grouped: dict[str, dict] = {}

    for f in fragments:
        sym = f.get("symbol") or "通用"
        grouped.setdefault(sym, {"fragments": [], "knowledge": [], "analyses": []})
        grouped[sym]["fragments"].append(f)

    for k in all_knowledge:
        sym = k.get("symbol", "通用")
        grouped.setdefault(sym, {"fragments": [], "knowledge": [], "analyses": []})
        grouped[sym]["knowledge"].append(k)

    for a in analyses:
        sym = a.get("symbol") or "通用"
        grouped.setdefault(sym, {"fragments": [], "knowledge": [], "analyses": []})
        grouped[sym]["analyses"].append(a)

    return grouped


# ────────────────────────────────────────────
#  LLM 報告生成
# ────────────────────────────────────────────

_REPORT_SYSTEM_PROMPT = """你是一位專業的加密貨幣量化分析師，正在撰寫一份正式的技術分析報告。

你會收到某個標的（如 BTC/USDT）的所有歷史分析資料，包括：
- 知識碎片（支撐壓力、趨勢、型態、指標、策略等）
- 蒸餾知識摘要
- 近期 AI 分析對話摘錄

請根據這些資料，撰寫一份結構化的分析報告。格式要求：

## [標的名稱] 技術分析報告

### 市場概況
（用 2-3 段話描述目前市場狀態、整體趨勢方向）

### 趨勢分析
（多級別趨勢判斷：日線/4H/1H，描述趨勢方向和關鍵轉折點）

### 關鍵價位
（列出重要的支撐位和壓力位，說明每個價位的重要性）
- 支撐位：$XX,XXX（原因）
- 壓力位：$XX,XXX（原因）

### 技術指標訊號
（彙整 RSI、MACD、布林通道等指標的最新訊號和解讀）

### 型態辨識
（如果有識別到的技術型態，描述型態和含義）

### 策略建議
（基於以上分析，給出具體的交易建議，包含進場點位、止損止盈）

### 風險提示
（提醒需要注意的風險因素）

規則：
1. 用繁體中文撰寫
2. 必須基於提供的資料，不要編造數據
3. 如果某個分類沒有資料就跳過該段落
4. 價格數字要精確到小數點後適當位數
5. 語氣專業但易懂
6. 每個段落都要有實質內容，不要空泛的描述"""

_EXEC_SUMMARY_PROMPT = """你是一位專業的加密貨幣量化分析師。
請根據以下各標的的分析報告，撰寫一份簡潔的「執行摘要」。

要求：
1. 用 3-5 段話概括所有標的的核心發現
2. 標示出最值得關注的交易機會
3. 指出目前最大的市場風險
4. 用繁體中文撰寫
5. 語氣專業、資訊密度高"""


def _build_symbol_prompt(symbol: str, data: dict) -> str:
    """將某幣種的所有資料組成 LLM prompt。"""
    parts = [f"以下是 {symbol} 的所有歷史分析資料：\n"]

    frags = data.get("fragments", [])
    if frags:
        parts.append("=== 知識碎片 ===")
        for f in frags[:40]:
            ftype = _TYPE_LABELS.get(f.get("type", ""), f.get("type", ""))
            parts.append(f"[{ftype}] (命中{f.get('hit_count', 0)}次) {f.get('content', '')}")
        parts.append("")

    knowledge = data.get("knowledge", [])
    if knowledge:
        parts.append("=== 蒸餾知識 ===")
        for k in knowledge:
            parts.append(f"期間 {k.get('period_start', '?')[:10]}~{k.get('period_end', '?')[:10]}:")
            parts.append(k.get("summary", ""))
        parts.append("")

    analyses = data.get("analyses", [])
    if analyses:
        parts.append("=== 近期分析對話摘錄 ===")
        for a in analyses:
            for pair in a.get("pairs", [])[-3:]:
                parts.append(f"Q: {pair.get('user', '')[:200]}")
                asst = pair.get("assistant", "")
                parts.append(f"A: {asst[:800]}")
        parts.append("")

    return "\n".join(parts)


async def _generate_report_with_llm(
    session_id: str,
    grouped_data: dict[str, dict],
) -> dict[str, str]:
    """呼叫 LLM 為每個幣種生成分析報告，回傳 {symbol: report_text}。"""
    from app.core.llm.adapter import create_adapter
    from app.core.security.key_manager import key_manager

    session_info = key_manager.get_session_info(session_id)
    if not session_info:
        raise ValueError("LLM session 已過期，請重新設定 API Key")
    api_key = key_manager.get_key(session_id)
    if not api_key:
        raise ValueError("無法取得 API Key")

    adapter = create_adapter(
        provider=session_info["provider"],
        api_key=api_key,
        model_name=session_info.get("model_name"),
        base_url=session_info.get("base_url"),
    )

    reports: dict[str, str] = {}

    symbols = [s for s in grouped_data if s != "通用"]
    if "通用" in grouped_data:
        symbols.append("通用")

    for symbol in symbols:
        data = grouped_data[symbol]
        total_items = len(data.get("fragments", [])) + len(data.get("knowledge", [])) + len(data.get("analyses", []))
        if total_items < 2:
            continue

        prompt = _build_symbol_prompt(symbol, data)
        messages = [
            {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info(f"正在為 {symbol} 生成分析報告...")
            response = await adapter.chat(messages, force_text=True)
            if response.message:
                reports[symbol] = response.message
                logger.info(f"{symbol} 報告生成完成 ({len(response.message)} 字)")
        except Exception as e:
            logger.error(f"{symbol} 報告生成失敗: {e}")
            reports[symbol] = f"（{symbol} 分析報告生成失敗：{e}）"

    if len(reports) > 1:
        try:
            combined = "\n\n---\n\n".join(
                f"[{s}]\n{r}" for s, r in reports.items()
            )
            messages = [
                {"role": "system", "content": _EXEC_SUMMARY_PROMPT},
                {"role": "user", "content": combined},
            ]
            resp = await adapter.chat(messages, force_text=True)
            if resp.message:
                reports["__executive_summary__"] = resp.message
        except Exception as e:
            logger.warning(f"執行摘要生成失敗: {e}")

    return reports


# ────────────────────────────────────────────
#  Markdown → PDF 解析
# ────────────────────────────────────────────

def _parse_markdown_sections(text: str) -> list[dict]:
    """將 Markdown 文字拆分為結構化段落列表。

    回傳 [{"level": 2|3|0, "title": str, "body": str}, ...]
    level 0 表示普通段落。
    """
    sections: list[dict] = []
    current: Optional[dict] = None

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            if current:
                sections.append(current)
            current = {"level": 3, "title": stripped[4:].strip(), "body": ""}
        elif stripped.startswith("## "):
            if current:
                sections.append(current)
            current = {"level": 2, "title": stripped[3:].strip(), "body": ""}
        else:
            if current is None:
                current = {"level": 0, "title": "", "body": ""}
            if current["body"]:
                current["body"] += "\n"
            current["body"] += line

    if current:
        sections.append(current)

    return sections


# ────────────────────────────────────────────
#  PDF 報告渲染 (fpdf2)
# ────────────────────────────────────────────

class _ReportPDF:
    def __init__(self):
        from fpdf import FPDF

        self.pdf = FPDF(format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self._w = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

        font_path = _find_cjk_font()
        if font_path:
            self.pdf.add_font("CJK", "", font_path)
            self.pdf.add_font("CJK", "B", font_path)
            self._font = "CJK"
        else:
            self._font = "Helvetica"
            logger.warning("未找到 CJK 字型")

    def _set_font(self, size: float = 10, bold: bool = False):
        self.pdf.set_font(self._font, "B" if bold else "", size)

    def _set_color(self, rgb: tuple):
        self.pdf.set_text_color(*rgb)

    def _set_fill(self, rgb: tuple):
        self.pdf.set_fill_color(*rgb)

    def _set_draw(self, rgb: tuple):
        self.pdf.set_draw_color(*rgb)

    def _safe(self, text: str) -> str:
        replacements = {
            "\u2713": "[v]", "\u2717": "[x]",
            "\U0001F464": "", "\U0001F916": "",
            "\u2022": "-", "\u25cf": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _check_page(self, needed: float = 30):
        if self.pdf.get_y() > 280 - needed:
            self.pdf.add_page()

    def _section_title(self, title: str, level: int = 2):
        self._check_page(20)
        self.pdf.ln(5)
        if level == 2:
            self._set_color(_C_PRIMARY)
            self._set_font(15, bold=True)
            self.pdf.cell(w=0, h=10, text=self._safe(title),
                          new_x="LMARGIN", new_y="NEXT")
            self._set_draw(_C_PRIMARY)
            self.pdf.line(self.pdf.l_margin, self.pdf.get_y(),
                          self.pdf.w - self.pdf.r_margin, self.pdf.get_y())
            self.pdf.ln(4)
        else:
            self._set_color(_C_DARK)
            self._set_font(12, bold=True)
            self.pdf.cell(w=0, h=8, text=self._safe(title),
                          new_x="LMARGIN", new_y="NEXT")
            self.pdf.ln(2)
        self._set_color(_C_TEXT)

    def _body_text(self, text: str, size: float = 10):
        self._set_font(size)
        self._set_color(_C_TEXT)
        lm = self.pdf.l_margin
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                self.pdf.ln(2)
                continue
            self.pdf.set_x(lm)
            if stripped.startswith("- ") or stripped.startswith("* "):
                bullet_text = stripped[2:].strip()
                if not bullet_text:
                    continue
                self._set_font(size)
                indent = 5
                self.pdf.set_x(lm + indent)
                self.pdf.cell(w=4, h=5.5, text="-", new_x="END", new_y="TOP")
                self.pdf.multi_cell(w=0, h=5.5, text=self._safe(bullet_text))
            elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                self._set_font(size, bold=True)
                self.pdf.multi_cell(w=0, h=5.5, text=self._safe(stripped.strip("* ")))
                self._set_font(size)
            else:
                cleaned = stripped.lstrip("#").strip()
                if cleaned:
                    self.pdf.multi_cell(w=0, h=5.5, text=self._safe(cleaned))
        self.pdf.ln(1)

    def _gray_text(self, text: str, size: float = 8):
        self._set_font(size)
        self._set_color(_C_GRAY)
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(w=0, h=4.5, text=self._safe(text))
        self._set_color(_C_TEXT)
        self.pdf.ln(1)

    def _box(self, text: str, bg: tuple = _C_BG_BOX):
        self._set_fill(bg)
        self._set_draw(_C_PRIMARY)
        x = self.pdf.l_margin
        self.pdf.set_x(x)
        y = self.pdf.get_y()
        self._set_font(10)
        safe = self._safe(text)
        inner_w = self._w - 8
        if inner_w < 20:
            inner_w = self._w - 2
        self.pdf.set_xy(x + 4, y)
        text_h = self.pdf.multi_cell(
            w=inner_w, h=5.5, text=safe,
            dry_run=True, output="HEIGHT"
        )
        box_h = text_h + 8
        if y + box_h > 270:
            self.pdf.add_page()
            y = self.pdf.get_y()
        self.pdf.rect(x, y, self._w, box_h, style="DF")
        self.pdf.line(x, y, x, y + box_h)
        self.pdf.set_xy(x + 4, y + 4)
        self.pdf.multi_cell(w=inner_w, h=5.5, text=safe)
        self.pdf.set_y(y + box_h + 3)

    # ─── 報告區段 ─────────────────

    def build_cover(self, chat_stats: dict, frag_stats: dict, symbol_count: int):
        self.pdf.add_page()
        self.pdf.ln(55)
        self._set_color(_C_DARK)
        self._set_font(28, bold=True)
        self.pdf.cell(w=0, h=15, text="阿斯拉量化系統", align="C",
                      new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(3)
        self._set_color(_C_PRIMARY)
        self._set_font(16)
        self.pdf.cell(w=0, h=10, text="AI 量化分析報告", align="C",
                      new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(25)
        self._set_color(_C_GRAY)
        self._set_font(10)
        now = datetime.now()
        lines = [
            f"報告日期：{now.strftime('%Y 年 %m 月 %d 日')}",
            f"分析標的：{symbol_count} 個",
            f"資料來源：{chat_stats.get('total_conversations', 0)} 筆對話 / {frag_stats.get('total_fragments', 0)} 筆知識碎片",
            "",
            "本報告由 AI 根據歷史分析資料自動生成",
            "投資有風險，報告僅供參考，不構成投資建議",
        ]
        for line in lines:
            self.pdf.cell(w=0, h=7, text=line, align="C",
                          new_x="LMARGIN", new_y="NEXT")

    def build_executive_summary(self, summary_text: str):
        self.pdf.add_page()
        self._section_title("執行摘要")
        sections = _parse_markdown_sections(summary_text)
        for sec in sections:
            if sec["level"] == 3:
                self._section_title(sec["title"], level=3)
            if sec["body"].strip():
                self._body_text(sec["body"])

    def build_symbol_report(self, symbol: str, report_text: str, section_num: int):
        self.pdf.add_page()
        self._section_title(f"{_num_to_chinese(section_num)}、{symbol} 技術分析")
        sections = _parse_markdown_sections(report_text)
        for sec in sections:
            if sec["level"] >= 2:
                self._section_title(sec["title"], level=sec["level"])
            if sec["body"].strip():
                self._body_text(sec["body"])

    def build_key_prices_table(self, fragments: list[dict]):
        sr = [f for f in fragments if f.get("type") == "support_resistance"]
        if not sr:
            return
        self.pdf.add_page()
        self._section_title("關鍵價位一覽表")

        headers = ["幣種", "分析結論", "命中", "日期"]
        col_w = [25, self._w - 65, 15, 25]
        self._set_fill(_C_DARK)
        self._set_color(_C_WHITE)
        self._set_font(9, bold=True)
        for i, h in enumerate(headers):
            self.pdf.cell(w=col_w[i], h=7, text=h, fill=True, border=1,
                          new_x="END", new_y="TOP")
        self.pdf.ln()

        self._set_color(_C_TEXT)
        self._set_font(9)
        for idx, f in enumerate(sr[:40]):
            bg = _C_BG_LIGHT if idx % 2 == 0 else _C_WHITE
            self._set_fill(bg)
            content = f.get("content", "")
            if len(content) > 80:
                content = content[:80] + "..."
            row = [
                f.get("symbol", "通用"),
                content,
                str(f.get("hit_count", 0)),
                f.get("created_at", "")[:10],
            ]
            self._check_page(8)
            for i, val in enumerate(row):
                self.pdf.cell(w=col_w[i], h=6.5, text=self._safe(val),
                              fill=True, new_x="END", new_y="TOP")
            self.pdf.ln()

    def build_appendix_distilled(self, all_knowledge: list, user_profile: str):
        if not all_knowledge and not user_profile:
            return
        self.pdf.add_page()
        self._section_title("附錄 A：蒸餾知識")
        if not all_knowledge:
            self._gray_text("尚未進行知識蒸餾。")
        else:
            for k in all_knowledge:
                sym = k.get("symbol", "通用")
                period = f"{k.get('period_start', '?')[:10]} ~ {k.get('period_end', '?')[:10]}"
                self._section_title(f"{sym}（{period}）", level=3)
                self._gray_text(
                    f"來源 {k.get('source_messages', 0)} 則 | "
                    f"壓縮率 {k.get('compression_ratio', 0)}% | "
                    f"v{k.get('version', 1)}"
                )
                self._box(k.get("summary", "（無）"))

        if user_profile:
            self._section_title("使用者分析風格", level=3)
            self._box(user_profile)

    def build_appendix_stats(self, chat_stats: dict, frag_stats: dict):
        self._check_page(40)
        self.pdf.ln(5)
        self._section_title("附錄 B：系統統計")

        cards = [
            (str(chat_stats.get('total_conversations', 0)), "對話次數"),
            (str(chat_stats.get('total_messages', 0)), "訊息總數"),
            (str(frag_stats.get('total_fragments', 0)), "知識碎片"),
            (str(frag_stats.get('total_hits', 0)), "碎片命中"),
        ]
        card_w = (self._w - 12) / 4
        x_start = self.pdf.l_margin
        y_start = self.pdf.get_y() + 2

        for idx, (value, label) in enumerate(cards):
            x = x_start + idx * (card_w + 4)
            self._set_fill(_C_BG_LIGHT)
            self._set_draw(_C_LIGHT_GRAY)
            self.pdf.rect(x, y_start, card_w, 22, style="DF")
            self.pdf.set_xy(x, y_start + 3)
            self._set_color(_C_PRIMARY)
            self._set_font(18, bold=True)
            self.pdf.cell(w=card_w, h=8, text=value, align="C",
                          new_x="LEFT", new_y="NEXT")
            self.pdf.set_x(x)
            self._set_color(_C_GRAY)
            self._set_font(8)
            self.pdf.cell(w=card_w, h=5, text=label, align="C")

        self.pdf.set_y(y_start + 28)
        self._set_color(_C_TEXT)

    def output(self) -> bytes:
        return self.pdf.output()


def _num_to_chinese(n: int) -> str:
    m = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
         6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    return m.get(n, str(n))


# ────────────────────────────────────────────
#  主流程
# ────────────────────────────────────────────

async def _build_report_pdf(session_id: Optional[str] = None) -> bytes:
    all_knowledge = knowledge_distiller.get_all_knowledge()
    user_profile = knowledge_distiller.get_user_profile()
    fragments = _get_all_fragments()
    analyses = _get_recent_analyses(limit=20)
    chat_stats = chat_history.get_stats()
    frag_stats = fragment_store.get_stats()

    grouped = _group_data_by_symbol(fragments, all_knowledge, analyses)
    real_symbols = [s for s in grouped if s != "通用"]

    llm_reports: dict[str, str] = {}

    if session_id:
        try:
            llm_reports = await _generate_report_with_llm(session_id, grouped)
            logger.info(f"LLM 報告生成完成：{len(llm_reports)} 個標的")
        except Exception as e:
            logger.error(f"LLM 報告生成失敗，改用純資料模式: {e}")

    report = _ReportPDF()

    report.build_cover(chat_stats, frag_stats, len(real_symbols))

    exec_summary = llm_reports.pop("__executive_summary__", "")
    if exec_summary:
        report.build_executive_summary(exec_summary)

    section_num = 1
    for symbol in real_symbols:
        if symbol in llm_reports:
            report.build_symbol_report(symbol, llm_reports[symbol], section_num)
            section_num += 1

    if "通用" in llm_reports:
        report.build_symbol_report("通用分析", llm_reports["通用"], section_num)
        section_num += 1

    report.build_key_prices_table(fragments)

    report.build_appendix_distilled(all_knowledge, user_profile)
    report.build_appendix_stats(chat_stats, frag_stats)

    return report.output()


# ────────────────────────────────────────────
#  API 端點
# ────────────────────────────────────────────

@router.get("/knowledge-pdf")
async def export_knowledge_pdf(
    session_id: Optional[str] = Query(None, description="LLM session ID，用於生成 AI 分析報告"),
):
    """匯出 AI 量化分析報告 PDF

    若提供 session_id，會呼叫 LLM 將所有知識整理成結構化報告；
    若未提供，僅輸出原始資料附錄。
    """
    try:
        pdf_bytes = await _build_report_pdf(session_id=session_id)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        ascii_name = f"asura_report_{ts}.pdf"
        utf8_name = f"阿斯拉分析報告_{ts}.pdf"
        disposition = (
            f"attachment; filename=\"{ascii_name}\"; "
            f"filename*=UTF-8''{quote(utf8_name)}"
        )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": disposition,
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except Exception as e:
        logger.error(f"PDF 匯出失敗: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"PDF 匯出失敗: {str(e)}")
