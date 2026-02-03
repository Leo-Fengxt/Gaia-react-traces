"""
OpenRouter LLM Client for GAIA ReAct Runner.

Configured for reasoning mode (defaults, can be overridden):
- temperature: 1.0
- top_p: 0.95 (set to None to omit top_p entirely)
- max_tokens: 32768
- reasoning.effort: "medium"
- reasoning.exclude: false (return reasoning tokens separately when supported)
- timeout: 900s (15 minutes)

Does NOT use :online suffix - web search is handled by Exa tools.
"""

from __future__ import annotations

import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    # Provider-specific: optional hidden reasoning tokens (chain-of-thought).
    # When OpenRouter is configured with reasoning.exclude=false, some models return this separately.
    reasoning: Optional[str] = None
    cost: float = 0.0
    model: str = ""
    finish_reason: str = ""
    usage: Optional[Dict[str, int]] = None


class OpenRouterClient:
    """
    OpenRouter API client configured for reasoning mode.

    Settings:
    - temperature: 1.0 (recommended for some reasoning models)
    - top_p: 0.95 (optional; set to None to omit)
    - max_tokens: 32768
    - reasoning.effort: "medium"
    - reasoning.exclude: false
    - timeout: 900s

    Does NOT use web search (:online suffix) - web is handled by Exa.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
        temperature: float = 1.0,
        top_p: Optional[float] = 0.95,
        max_tokens: int = 32768,
        timeout: int = 900,
        reasoning_effort: str = "medium",
        reasoning_exclude: bool = True,
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use
            temperature: Sampling temperature (1.0 for reasoning)
            top_p: Top-p sampling (0.95 by default). Use None to omit top_p entirely.
            max_tokens: Maximum tokens to generate
            timeout: API timeout in seconds
            reasoning_effort: Reasoning effort ("none", "low", "medium", "high")
            reasoning_exclude: Whether to exclude reasoning from response
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.reasoning_exclude = reasoning_exclude

        self._client = None
        self._async_client = None
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def _get_client(self):
        """Get or create the sync OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.BASE_URL,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    def _get_async_client(self):
        """Get or create the async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.BASE_URL,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._async_client

    def _build_request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build the request kwargs for chat completion."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        # Some callers want "no top_p" semantics (don't send nucleus sampling at all).
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p

        # Stop sequences (crucial for ReAct: prevent runaway generations)
        if stop is not None:
            request_kwargs["stop"] = stop

        # Configure reasoning
        if self.reasoning_effort != "none":
            request_kwargs["extra_body"] = {
                "reasoning": {
                    "enabled": True,
                    "effort": self.reasoning_effort,
                    "exclude": self.reasoning_exclude,
                }
            }
        else:
            # Explicitly disable reasoning
            request_kwargs["extra_body"] = {
                "reasoning": {
                    "enabled": False,
                    "effort": "none",
                }
            }

        return request_kwargs

    def _parse_response(self, response) -> LLMResponse:
        """Parse the API response into LLMResponse."""
        message = response.choices[0].message
        content = message.content or ""

        # Best-effort extraction of OpenRouter "reasoning" field.
        # The OpenAI SDK may not expose unknown fields directly, so we try a few fallbacks.
        reasoning: Optional[str] = None
        try:
            r = getattr(message, "reasoning", None)
            if isinstance(r, str) and r.strip():
                reasoning = r
        except Exception:
            reasoning = None
        if reasoning is None:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                r = extra.get("reasoning")
                if isinstance(r, str) and r.strip():
                    reasoning = r
        if reasoning is None:
            # pydantic v2
            try:
                d = message.model_dump()  # type: ignore[attr-defined]
            except Exception:
                d = None
            if isinstance(d, dict):
                r = d.get("reasoning")
                if isinstance(r, str) and r.strip():
                    reasoning = r
        if reasoning is None:
            # pydantic v1
            try:
                d = message.dict()  # type: ignore[attr-defined]
            except Exception:
                d = None
            if isinstance(d, dict):
                r = d.get("reasoning")
                if isinstance(r, str) and r.strip():
                    reasoning = r

        # Calculate cost estimate and track tokens
        cost = 0.0
        usage = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            # Track cumulative tokens
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            # Rough placeholder cost estimate. Pricing varies by provider/model on OpenRouter.
            # If you need accurate cost reporting, update these rates per your model selection.
            cost = (prompt_tokens * 0.10 + completion_tokens * 0.40) / 1_000_000

        self._total_cost += cost

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            cost=cost,
            model=self.model,
            finish_reason=response.choices[0].finish_reason,
            usage=usage,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Any] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request (sync).

        Returns:
            LLMResponse with content
        """
        client = self._get_client()
        request_kwargs = self._build_request_kwargs(messages, temperature, max_tokens, stop=stop)
        if model is not None:
            request_kwargs["model"] = model
        request_kwargs.update(kwargs)
        used_model = str(request_kwargs.get("model") or self.model)

        try:
            response = client.chat.completions.create(**request_kwargs)
            parsed = self._parse_response(response)
            parsed.model = used_model
            return parsed
        except Exception as e:
            error_msg = str(e)
            # Some OpenRouter/Cloudflare failures include an entire HTML error page in the exception text.
            # Keep the error message small so it doesn't pollute prompts/logs.
            lower = error_msg.lower()
            if "<!doctype html" in lower or "<html" in lower or "cloudflare" in lower:
                error_msg = "OpenRouter temporarily unavailable (Cloudflare)."
            if len(error_msg) > 2000:
                error_msg = error_msg[:2000] + " ...(truncated)"
            return LLMResponse(
                content=f"Error: {error_msg}",
                cost=0.0,
                model=used_model,
                finish_reason="error",
            )

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Any] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request (async).
        """
        client = self._get_async_client()
        request_kwargs = self._build_request_kwargs(messages, temperature, max_tokens, stop=stop)
        if model is not None:
            request_kwargs["model"] = model
        request_kwargs.update(kwargs)
        used_model = str(request_kwargs.get("model") or self.model)

        try:
            response = await client.chat.completions.create(**request_kwargs)
            parsed = self._parse_response(response)
            parsed.model = used_model
            return parsed
        except Exception as e:
            error_msg = str(e)
            # Check for content too long error
            if "too long" in error_msg.lower() or "context" in error_msg.lower():
                return LLMResponse(
                    content=f"CONTEXT_TOO_LONG: {error_msg}",
                    cost=0.0,
                    model=used_model,
                    finish_reason="context_length_exceeded",
                )
            # Some OpenRouter/Cloudflare failures include an entire HTML error page in the exception text.
            # Keep the error message small so it doesn't pollute prompts/logs.
            lower = error_msg.lower()
            if "<!doctype html" in lower or "<html" in lower or "cloudflare" in lower:
                error_msg = "OpenRouter temporarily unavailable (Cloudflare)."
            if len(error_msg) > 2000:
                error_msg = error_msg[:2000] + " ...(truncated)"
            return LLMResponse(
                content=f"Error: {error_msg}",
                cost=0.0,
                model=used_model,
                finish_reason="error",
            )

    @property
    def total_cost(self) -> float:
        """Get total cost of all API calls."""
        return self._total_cost

    @property
    def total_prompt_tokens(self) -> int:
        """Get total prompt/input tokens across all API calls."""
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Get total completion/output tokens across all API calls."""
        return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        """Get total tokens (prompt + completion) across all API calls."""
        return self._total_prompt_tokens + self._total_completion_tokens

    def reset_cost(self) -> None:
        """Reset all counters (cost and tokens)."""
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0


class SummarizationClient(OpenRouterClient):
    """
    Specialized client for summarizing conversation history.

    NOTE: We often run summarization at a low temperature for stability, and callers may
    omit top_p entirely. The constructor supports top_p=None to avoid sending nucleus
    sampling to the provider.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
        temperature: float = 0.05,
        top_p: Optional[float] = None,
        max_tokens: int = 65_536,
        timeout: int = 900,
        reasoning_effort: str = "medium",
        reasoning_exclude: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
        )

    async def summarize_history(
        self,
        history_text: str,
        question: str,
    ) -> str:
        """
        Summarize a conversation history.
        """
        # Chunking guard: summarizing multi-million-char histories in one call is fragile
        # (may exceed model/provider context even for long-context models).
        max_chunk_chars = 200_000
        # When we must fall back, prefer middle-out over truncation.
        fallback_keep_head = 20_000
        fallback_keep_tail = 20_000
        retries = 3

        async def summarize_once(text: str, *, max_tokens: int) -> str:
            # Summarization policy: preserve high-signal structured artifacts first.
            summarizer_system = (
                "You are a summarization assistant for agent traces and web/tool outputs.\n\n"
                "PRIORITIES (in order):\n"
                "1) Preserve EXACT function/tool calls and their arguments (e.g., web_search(...), web_contents(...), execute_python(...), subgoal(...), done(...), finish(...)).\n"
                "2) Preserve URLs/links and identifiers EXACTLY (http(s)://..., vault://..., DOIs, arXiv IDs, repo names).\n"
                "3) Preserve high-level structured content: titles, names, headings, key highlights, key numbers/dates/units.\n"
                "4) Then summarize the remaining prose concisely.\n\n"
                "RULES:\n"
                "- Do NOT invent facts or links.\n"
                "- Keep the summary compact but information-dense.\n"
                "- If you drop low-signal details, say so briefly.\n"
            )
            messages = [
                {
                    "role": "system",
                    "content": summarizer_system,
                },
                {
                    "role": "user",
                    "content": f"""Summarize the following conversation history concisely while preserving all important information.

Original Question: {question}

Conversation History:
{text}

Provide a concise summary that captures:
1. Key facts and data discovered
2. Important observations
3. Any errors or dead ends encountered
4. Progress made toward answering the question

Summary:""",
                },
            ]

            last = ""
            for i in range(max(1, retries)):
                resp = await self.achat(messages, max_tokens=max_tokens)
                # Retry on hard errors / context overflow.
                if resp.finish_reason in ("error", "context_length_exceeded"):
                    await asyncio.sleep(0.5 * (2**i))
                    continue

                content = (resp.content or "").strip()
                if content:
                    return content
                last = content
                await asyncio.sleep(0.5 * (2**i))

            return last.strip()

        history_text = (history_text or "")
        if not history_text.strip():
            return ""

        # Fast path: small enough to summarize directly.
        if len(history_text) <= max_chunk_chars:
            out = await summarize_once(history_text, max_tokens=self.max_tokens)
            if out:
                return out
            head = history_text[:fallback_keep_head]
            tail = history_text[-fallback_keep_tail:] if len(history_text) > fallback_keep_tail else ""
            return (
                "(middle-out fallback: summarization failed)\n\n"
                f"EARLY:\n{head}\n\n"
                "[... middle omitted due to summarization error ...]\n\n"
                f"RECENT:\n{tail}"
            ).strip()

        # Chunk summarize, then summarize the chunk summaries.
        chunks = [history_text[i : i + max_chunk_chars] for i in range(0, len(history_text), max_chunk_chars)]
        partials = []
        for chunk in chunks:
            part = await summarize_once(chunk, max_tokens=4096)
            if not part:
                head = history_text[:fallback_keep_head]
                tail = history_text[-fallback_keep_tail:] if len(history_text) > fallback_keep_tail else ""
                return (
                    "(middle-out fallback: chunk summarization failed)\n\n"
                    f"EARLY:\n{head}\n\n"
                    "[... middle omitted due to summarization error ...]\n\n"
                    f"RECENT:\n{tail}"
                ).strip()
            partials.append(part)

        combined = "\n\n".join(partials)
        final = await summarize_once(combined, max_tokens=self.max_tokens)
        if final:
            return final
        head = combined[:fallback_keep_head]
        tail = combined[-fallback_keep_tail:] if len(combined) > fallback_keep_tail else ""
        return (
            "(middle-out fallback: final summarization failed)\n\n"
            f"EARLY:\n{head}\n\n"
            "[... middle omitted due to summarization error ...]\n\n"
            f"RECENT:\n{tail}"
        ).strip()

