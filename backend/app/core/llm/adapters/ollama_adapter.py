"""OllamaAdapter — v154 由 adapter.py 純搬家拆出（原 L1927-2009），邏輯零改動。"""

import json
from typing import Optional

from loguru import logger

from app.core.config.settings import settings

from app.core.llm.adapters.base import (
    BaseLLMAdapter,
    LLMResponse,
    TokenUsage,
    _LLM_STREAM_TIMEOUT,
    _LLM_TIMEOUT,
)

class OllamaAdapter(BaseLLMAdapter):
    """Ollama 本地模型適配器"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None, r2_mode: bool = False) -> LLMResponse:
        try:
            import httpx

            sys_msg = {"role": "system", "content": self._build_system_message(chart_state, system_prompt)}
            all_messages = [sys_msg] + messages

            # Ollama 多模態模型（llava 等）支援 images 欄位
            if chart_screenshot:
                b64_data, _ = self._extract_base64(chart_screenshot)
                for i in range(len(all_messages) - 1, -1, -1):
                    if all_messages[i]["role"] == "user":
                        all_messages[i] = {
                            **all_messages[i],
                            "images": [b64_data],
                        }
                        break

            async with httpx.AsyncClient(timeout=float(_LLM_TIMEOUT)) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": all_messages,
                        "stream": False,
                        "options": {"temperature": settings.llm_temperature},
                    },
                )
                data = response.json()
                # 提取 token 用量
                usage = TokenUsage(
                    prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                    completion_tokens=data.get("eval_count", 0) or 0,
                    total_tokens=(data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
                    model=self.model,
                    provider="ollama",
                )
                return LLMResponse(
                    message=data.get("message", {}).get("content", ""),
                    raw_response=data,
                    usage=usage,
                )
        except Exception as e:
            logger.error(f"Ollama 請求失敗: {e}")
            return LLMResponse(message=f"Ollama 請求失敗: {str(e)}")

    async def chat_stream(self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None):
        try:
            import httpx

            sys_msg = {"role": "system", "content": self._build_system_message(chart_state, system_prompt)}
            all_messages = [sys_msg] + messages

            async with httpx.AsyncClient(timeout=float(_LLM_STREAM_TIMEOUT)) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": all_messages,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"[錯誤] {str(e)}"


