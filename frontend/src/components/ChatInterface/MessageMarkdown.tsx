/** 阿斯拉量化系統 — AI 訊息 Markdown 渲染
 *
 * 聊天訊息原本以純文字渲染，AI 輸出的 markdown 表格顯示成原始 | 字元、粗體顯示 **。
 * 此元件用 react-markdown + GFM 渲染表格/粗體/標題；remark-breaks 保留單行換行
 * （AI 輸出大量單換行文字，預設 markdown 會併段）。
 *
 * 效能：串流中仍用純文字（呼叫端控制），串流完成後才切到 markdown，
 * 避免長訊息逐 token 重新解析。表格樣式在 index.css 的 .chat-md（scoped，
 * 補回被全域 * reset 蓋掉的 padding）。
 */

import { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

export const MessageMarkdown = memo(function MessageMarkdown({ content }: { content: string }) {
  return (
    <div className="chat-md">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
        {content}
      </ReactMarkdown>
    </div>
  );
});
