"""
Classic ReAct Agent with regex-parsed tool calls.
"""

from __future__ import annotations

import re
import json
import asyncio
import difflib
from typing import Dict, List, Optional, Any, Tuple, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..clients.openrouter import OpenRouterClient, SummarizationClient, LLMResponse
from ..tools.web_search import WebSearchTool
from ..tools.web_contents import WebContentsTool
from ..tools.e2b_python import E2BPythonTool
from ..tools.browserbase_playwright import BrowserbasePlaywrightTool
from ..tools.video_understanding import VideoUnderstandingTool
from .prompt import (
    STOP_TOKEN,
    build_system_prompt,
    build_initial_prompt,
    build_continuation_prompt,
    build_summarized_prompt,
    build_reformat_prompt,
)


@dataclass
class ReActStep:
    """Represents one step in the ReAct loop."""

    step_num: int
    thought: str
    action: str
    action_input: str
    observation: str = ""
    raw_response: str = ""
    parse_error: Optional[str] = None
    llm_finish_reason: Optional[str] = None
    llm_usage: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReActTrace:
    """Complete trace of a ReAct agent execution."""

    task_id: str
    question: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    total_cost: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "question": self.question,
            "steps": [
                {
                    "step_num": s.step_num,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation[:5000] if len(s.observation) > 5000 else s.observation,
                    "raw_response": s.raw_response[:2000] if len(s.raw_response) > 2000 else s.raw_response,
                    "parse_error": s.parse_error,
                    "llm_finish_reason": s.llm_finish_reason,
                    "llm_usage": s.llm_usage,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
        }


class ReActAgent:
    """
    Classic ReAct agent with regex-parsed tool calls.
    """

    STEP_BLOCK_PATTERN = re.compile(
        r"Thought:\s*(?P<thought>.*?)\n" r"Action:\s*(?P<action>\w+)\s*\n" r"Action Input:\s*(?P<action_input>.*?)(?=\n\s*Thought:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    ACTION_LINE_PATTERN = re.compile(r"^\s*Action:\s*(?P<action>\w+)\s*$", re.IGNORECASE | re.MULTILINE)
    ACTION_INPUT_LINE_PATTERN = re.compile(
        r"^\s*Action Input:\s*(?P<action_input>.+?)\s*$",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    VALID_ACTIONS = {
        "web_search",
        "web_contents",
        "execute_python",
        "browser",
        "understand_video",
        "finish",
    }

    MAX_IMAGE_HISTORY = 10

    def __init__(
        self,
        llm_client: OpenRouterClient,
        web_search_tool: WebSearchTool,
        web_contents_tool: WebContentsTool,
        e2b_tool: E2BPythonTool,
        browser_tool: Optional[BrowserbasePlaywrightTool] = None,
        video_tool: Optional[VideoUnderstandingTool] = None,
        allowed_tools: Optional[List[str]] = None,
        max_steps: int = 30,
        max_context_chars: int = 600_000,
        max_context_tokens: int = 150_000,
        keep_start_steps: int = 3,
        keep_end_steps: int = 5,
        max_parse_retries: int = 1,
        verbose: bool = False,
        max_consecutive_web_search: int = 4,
        web_search_similarity_threshold: float = 0.9,
        max_loop_guard_hits: int = 6,
        initial_images: Optional[List[str]] = None,
    ):
        self.llm_client = llm_client
        self.web_search_tool = web_search_tool
        self.web_contents_tool = web_contents_tool
        self.e2b_tool = e2b_tool
        self.browser_tool = browser_tool
        self.video_tool = video_tool

        supported_tools = {"web_search", "web_contents", "execute_python", "browser", "understand_video"}
        if allowed_tools is None:
            allowed_tool_set = set(supported_tools)
        else:
            allowed_tool_set = {str(t).strip().lower() for t in allowed_tools if str(t).strip()}
        allowed_tool_set.discard("finish")
        unknown = sorted(list(allowed_tool_set - supported_tools))
        if unknown:
            raise ValueError(f"Unknown allowed_tools: {unknown}. Supported: {sorted(list(supported_tools))}")
        self.allowed_actions = set(allowed_tool_set) | {"finish"}
        self._system_prompt = build_system_prompt(list(self.allowed_actions))
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars
        self.max_context_tokens = int(max_context_tokens)
        self.keep_start_steps = keep_start_steps
        self.keep_end_steps = keep_end_steps
        self.max_parse_retries = max_parse_retries
        self.verbose = verbose
        self.max_consecutive_web_search = max_consecutive_web_search
        self.web_search_similarity_threshold = web_search_similarity_threshold
        self.max_loop_guard_hits = max_loop_guard_hits

        self._image_history: List[str] = []
        if initial_images:
            self.add_images(initial_images)

        self.summarizer = SummarizationClient(
            api_key=self.llm_client.api_key,
            model=self.llm_client.model,
            temperature=0.05,
            top_p=None,
            max_tokens=65_536,
            timeout=self.llm_client.timeout,
            reasoning_effort=self.llm_client.reasoning_effort,
            reasoning_exclude=self.llm_client.reasoning_exclude,
        )

        self._history_summary: Optional[str] = None
        self._early_history: Optional[str] = None

        self._last_web_search_query_norm: Optional[str] = None
        self._last_web_search_urls: List[str] = []
        self._consecutive_web_searches: int = 0
        self._loop_guard_hits: int = 0

        # Reasoning archive (per-task): store hidden reasoning tokens separately from traces.
        self._active_task_id: Optional[str] = None
        self._llm_call_index: int = 0
        self._reasoning_archive: List[Dict[str, Any]] = []

    def _archive_reasoning(self, resp: LLMResponse, *, step_num: Optional[int], phase: str) -> None:
        self._llm_call_index += 1
        reasoning = getattr(resp, "reasoning", None)
        if not isinstance(reasoning, str) or not reasoning.strip():
            return
        self._reasoning_archive.append(
            {
                "call_index": int(self._llm_call_index),
                "task_id": self._active_task_id,
                "step_num": int(step_num) if step_num is not None else None,
                "phase": str(phase),
                "model": str(getattr(resp, "model", "") or ""),
                "finish_reason": str(getattr(resp, "finish_reason", "") or ""),
                "usage": getattr(resp, "usage", None),
                "reasoning": reasoning,
                "archived_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            resp.reasoning = None  # type: ignore[attr-defined]
        except Exception:
            pass

    # -------------------------
    # Image handling
    # -------------------------
    @staticmethod
    def _middle_out_drop(items: List[str], max_items: int) -> List[str]:
        max_items = max(0, int(max_items))
        if max_items <= 0:
            return []
        if len(items) <= max_items:
            return list(items)
        kept = list(items)
        while len(kept) > max_items:
            kept.pop(len(kept) // 2)
        return kept

    def add_images(self, images: Iterable[str]) -> None:
        for img in images:
            u = str(img or "").strip()
            if not u:
                continue
            self._image_history.append(u)
        self._image_history = self._middle_out_drop(self._image_history, self.MAX_IMAGE_HISTORY)

    def _build_user_message_content(self, user_text: str) -> Any:
        if not self._image_history:
            return user_text
        parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        n = len(self._image_history)
        for i, url in enumerate(self._image_history):
            tag = "CURRENT" if i == n - 1 else "HISTORY"
            parts.append({"type": "text", "text": f"[SCREENSHOT {i+1}/{n} - {tag}]"})
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    async def _achat_with_image_fallback(
        self,
        *,
        system_prompt: str,
        user_text: str,
        stop: Optional[Any],
    ) -> LLMResponse:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_user_message_content(user_text)},
        ]
        resp = await self.llm_client.achat(messages, stop=stop)
        if resp.finish_reason != "context_length_exceeded":
            return resp

        # Drop to half (middle-out) and retry once.
        if self._image_history:
            self._image_history = self._middle_out_drop(self._image_history, max(0, len(self._image_history) // 2))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_user_message_content(user_text)},
            ]
            resp = await self.llm_client.achat(messages, stop=stop)
            if resp.finish_reason != "context_length_exceeded":
                return resp

        # Drop all images and retry once.
        if self._image_history:
            self._image_history = []
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            resp = await self.llm_client.achat(messages, stop=stop)
        return resp

    @staticmethod
    def _normalize_query(q: str) -> str:
        return " ".join((q or "").strip().lower().split())

    @staticmethod
    def _extract_urls_from_observation(observation: str, max_urls: int = 5) -> List[str]:
        if not observation:
            return []
        urls = re.findall(r"URL:\s*(\S+)", observation)
        return urls[:max_urls]

    def _similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a or "", b or "").ratio()

    def _parse_response(self, response: str) -> Tuple[str, str, str, Optional[str]]:
        thought = ""
        action = ""
        action_input = ""

        cleaned = response.replace(STOP_TOKEN, "").strip()

        matches = list(self.STEP_BLOCK_PATTERN.finditer(cleaned))
        if matches:
            last = matches[-1]
            thought = (last.group("thought") or "").strip()
            action = (last.group("action") or "").strip().lower()
            action_input = (last.group("action_input") or "").strip()
        else:
            action_match = self.ACTION_LINE_PATTERN.search(cleaned)
            input_match = self.ACTION_INPUT_LINE_PATTERN.search(cleaned)
            if not action_match or not input_match:
                return thought, action, action_input, "Could not parse Thought/Action/Action Input block"

            action = (action_match.group("action") or "").strip().lower()
            action_input = (input_match.group("action_input") or "").strip()
            thought_block = cleaned[: action_match.start()].strip()
            thought_block = re.sub(r"^\s*thought\s*:?\s*", "", thought_block, flags=re.IGNORECASE).strip()
            thought = thought_block

        if action not in self.allowed_actions:
            return thought, action, action_input, f"Invalid action '{action}'. Allowed actions: {sorted(list(self.allowed_actions))}"

        if not action_input:
            return thought, action, action_input, "Empty Action Input"

        return thought, action, action_input, None

    async def _finalize_after_max_steps(self, question: str, steps: List[ReActStep]) -> Optional[ReActStep]:
        history_or_prompt, was_summarized = await self._maybe_summarize_history(question, steps)
        if was_summarized:
            user_prompt = history_or_prompt
        else:
            user_prompt = build_continuation_prompt(question, history_or_prompt)

        final_system = build_system_prompt(["finish"]) + "\n\nFINALIZATION MODE: Tool use is disabled. You MUST answer now.\n" + "Only allowed Action is: finish\n"
        user_text = user_prompt + "\n\nYou have reached the step limit. Produce a final answer now using Action: finish and \\boxed{...}."
        response = await self._achat_with_image_fallback(
            system_prompt=final_system,
            user_text=user_text,
            stop=[STOP_TOKEN],
        )
        self._archive_reasoning(response, step_num=len(steps) + 1, phase="finalize")
        raw_response = response.content
        thought, action, action_input, parse_error = self._parse_response(raw_response)

        step = ReActStep(
            step_num=len(steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            raw_response=raw_response,
            parse_error=parse_error,
            llm_finish_reason=response.finish_reason,
            llm_usage=response.usage,
        )

        if parse_error:
            step.observation = f"Parse error during finalization: {parse_error}"
            return step

        if action != "finish":
            step.observation = f"Finalization did not finish (action={action})"
            return step

        step.observation = "FINISH"
        return step

    async def _execute_tool(self, action: str, action_input: str) -> str:
        try:
            if action == "web_search":
                if self.verbose:
                    print(f"  [tool] web_search input={action_input[:200]}{'...' if len(action_input) > 200 else ''}")
                try:
                    params = json.loads(action_input)
                except json.JSONDecodeError:
                    params = {"query": action_input.strip('"').strip("'")}

                query = params.get("query", "")
                num_results = params.get("num_results", 10)
                search_type = params.get("search_type", "auto")
                include_domains = params.get("include_domains")
                exclude_domains = params.get("exclude_domains")
                category = params.get("category")
                start_published_date = params.get("start_published_date") or params.get("startPublishedDate")
                end_published_date = params.get("end_published_date") or params.get("endPublishedDate")
                include_text = params.get("include_text") or params.get("includeText")
                exclude_text = params.get("exclude_text") or params.get("excludeText")
                include_contents = params.get("include_contents")
                max_characters = params.get("max_characters") or params.get("maxCharacters")
                context = params.get("context")
                context_max_characters = params.get("context_max_characters") or params.get("contextMaxCharacters")

                if not query:
                    return "Error: No query provided for web_search"

                query_norm = self._normalize_query(query)
                warn_msgs: List[str] = []
                if self._consecutive_web_searches >= self.max_consecutive_web_search:
                    warn_msgs.append(
                        "WARNING: Many consecutive web_search calls. Consider switching to web_contents on a promising URL."
                    )
                if self._last_web_search_query_norm and query_norm:
                    sim = self._similarity(query_norm, self._last_web_search_query_norm)
                    if sim >= self.web_search_similarity_threshold:
                        warn_msgs.append(
                            "WARNING: This web_search query is very similar to the previous one. Consider switching strategy (web_contents/execute_python)."
                        )

                observation = await self.web_search_tool.execute(
                    query,
                    num_results,
                    search_type=search_type,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    category=category,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                    include_text=include_text,
                    exclude_text=exclude_text,
                    include_contents=include_contents,
                    max_characters=max_characters,
                    context=context,
                    context_max_characters=context_max_characters,
                )
                self._consecutive_web_searches += 1
                self._last_web_search_query_norm = query_norm
                self._last_web_search_urls = self._extract_urls_from_observation(observation, max_urls=5)
                if warn_msgs:
                    hint = ""
                    if self._last_web_search_urls:
                        hint = f"\nHint: try web_contents with one of these URLs: {self._last_web_search_urls[:3]}"
                    warning_block = "\n".join(warn_msgs) + hint
                    return f"{warning_block}\n\n{observation}"
                return observation

            elif action == "web_contents":
                if self.verbose:
                    print(f"  [tool] web_contents input={action_input[:200]}{'...' if len(action_input) > 200 else ''}")
                self._consecutive_web_searches = 0
                self._loop_guard_hits = 0
                try:
                    params = json.loads(action_input)
                except json.JSONDecodeError:
                    url = action_input.strip('"').strip("'")
                    params = {"urls": [url]}

                urls = params.get("urls", [])
                max_characters = params.get("max_characters")
                if isinstance(urls, str):
                    urls = [urls]

                if not urls:
                    return "Error: No URLs provided for web_contents"

                return await self.web_contents_tool.execute(urls, max_characters=max_characters)

            elif action == "execute_python":
                if self.verbose:
                    print(f"  [tool] execute_python input={action_input[:200]}{'...' if len(action_input) > 200 else ''}")
                self._consecutive_web_searches = 0
                self._loop_guard_hits = 0
                try:
                    params = json.loads(action_input)
                except json.JSONDecodeError:
                    params = {"code": action_input}

                code = params.get("code", "")

                if not code:
                    return "Error: No code provided for execute_python"

                result = await self.e2b_tool.execute(code)

                lines = []
                if result.success:
                    lines.append("Code executed successfully.")
                else:
                    lines.append("Code execution failed.")
                lines.append(f"Execution time: {result.execution_time:.2f}s")
                lines.append("")
                if result.output:
                    lines.append("Output:")
                    lines.append(result.output)
                if result.error:
                    lines.append("")
                    lines.append("Error:")
                    lines.append(result.error)
                return "\n".join(lines)

            elif action == "browser":
                self._consecutive_web_searches = 0
                self._loop_guard_hits = 0
                if self.browser_tool is None:
                    return "Error: browser tool not configured (set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID)"
                try:
                    params = json.loads(action_input)
                except json.JSONDecodeError:
                    params = {"script": action_input}
                if not isinstance(params, dict):
                    params = {"script": str(params)}
                if "script" not in params and "op" not in params:
                    params = {"script": str(action_input)}
                out = await self.browser_tool.execute(**params)
                if getattr(out, "screenshot_data_url", None):
                    try:
                        self.add_images([out.screenshot_data_url])
                    except Exception:
                        pass
                return getattr(out, "observation", str(out))

            elif action == "understand_video":
                self._consecutive_web_searches = 0
                self._loop_guard_hits = 0
                if self.video_tool is None:
                    return "Error: video tool not configured"
                try:
                    params = json.loads(action_input)
                except json.JSONDecodeError:
                    params = {"prompt": action_input}
                if not isinstance(params, dict):
                    params = {"prompt": str(params)}
                return await self.video_tool.execute(**params)

            elif action == "finish":
                self._consecutive_web_searches = 0
                self._loop_guard_hits = 0
                return "FINISH"

            else:
                return f"Error: Unknown action '{action}'"

        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def _format_step(self, step: ReActStep) -> str:
        lines = [
            f"Thought: {step.thought}",
            f"Action: {step.action}",
            f"Action Input: {step.action_input}",
        ]
        if step.observation and step.action != "finish":
            lines.append(f"Observation: {step.observation}")
        return "\n".join(lines)

    def _build_history(self, steps: List[ReActStep]) -> str:
        if not steps:
            return ""

        parts = []
        for step in steps:
            parts.append(self._format_step(step))
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        ascii_len = len(text) - non_ascii
        return int(non_ascii + (ascii_len / 4.0))

    async def _maybe_summarize_history(self, question: str, steps: List[ReActStep]) -> Tuple[str, bool]:
        history = self._build_history(steps)
        token_est = self._estimate_tokens(history)

        if len(history) > self.max_context_chars or token_est > self.max_context_tokens:
            if self.verbose:
                print(f"  History too long (chars={len(history)} tokens~={token_est}), summarizing...")

            early_steps = steps[: self.keep_start_steps]
            early_history = self._build_history(early_steps)

            recent_steps = steps[-self.keep_end_steps :]
            recent_history = self._build_history(recent_steps)

            middle_steps = steps[self.keep_start_steps : -self.keep_end_steps]
            if middle_steps:
                middle_history = self._build_history(middle_steps)
                summary = await self.summarizer.summarize_history(
                    middle_history,
                    question,
                )

                self._history_summary = summary
                self._early_history = early_history

                prompt = build_summarized_prompt(
                    question,
                    early_history,
                    summary,
                    recent_history,
                )
                prompt_tokens = self._estimate_tokens(prompt)
                if prompt_tokens > self.max_context_tokens:
                    full_summary = await self.summarizer.summarize_history(history, question)
                    prompt = build_summarized_prompt(
                        question,
                        "(omitted: summarized for context budget)",
                        full_summary,
                        "(omitted: summarized for context budget)",
                    )
                return prompt, True

        return history, False

    async def run(self, task_id: str, question: str) -> ReActTrace:
        import time

        start_time = time.time()

        trace = ReActTrace(
            task_id=task_id,
            question=question,
        )

        steps: List[ReActStep] = []

        self._active_task_id = str(task_id)
        self._llm_call_index = 0
        self._reasoning_archive = []

        self._last_web_search_query_norm = None
        self._last_web_search_urls = []
        self._consecutive_web_searches = 0
        self._loop_guard_hits = 0

        for step_num in range(1, self.max_steps + 1):
            if step_num == 1:
                user_prompt = build_initial_prompt(question)
            else:
                history_or_prompt, was_summarized = await self._maybe_summarize_history(question, steps)
                if was_summarized:
                    user_prompt = history_or_prompt
                else:
                    user_prompt = build_continuation_prompt(question, history_or_prompt)

            try:
                response = await self._achat_with_image_fallback(
                    system_prompt=self._system_prompt,
                    user_text=user_prompt,
                    stop=[STOP_TOKEN],
                )
                self._archive_reasoning(response, step_num=step_num, phase="step")
            except Exception as e:
                trace.error = f"LLM error: {str(e)}"
                trace.success = False
                break

            if response.finish_reason == "context_length_exceeded":
                old_max = self.max_context_chars
                self.max_context_chars = 0
                history, _ = await self._maybe_summarize_history(question, steps)
                self.max_context_chars = old_max

                user_prompt = history
                response = await self._achat_with_image_fallback(
                    system_prompt=self._system_prompt,
                    user_text=user_prompt,
                    stop=[STOP_TOKEN],
                )
                self._archive_reasoning(response, step_num=step_num, phase="step_retry_context")

            raw_response = response.content

            thought, action, action_input, parse_error = self._parse_response(raw_response)

            if parse_error and self.max_parse_retries > 0:
                reformat_prompt = build_reformat_prompt(raw_response, allowed_actions=list(self.allowed_actions))
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": reformat_prompt},
                ]

                response = await self.llm_client.achat(messages, stop=[STOP_TOKEN])
                self._archive_reasoning(response, step_num=step_num, phase="reformat")
                raw_response = response.content
                thought, action, action_input, parse_error = self._parse_response(raw_response)

            step = ReActStep(
                step_num=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                raw_response=raw_response,
                parse_error=parse_error,
                llm_finish_reason=response.finish_reason,
                llm_usage=response.usage,
            )

            if parse_error:
                step.observation = f"Parse error: {parse_error}"
                steps.append(step)
                trace.error = parse_error
                break

            if action == "finish":
                step.observation = "FINISH"
                steps.append(step)
                trace.final_answer = action_input
                trace.success = True
                break

            observation = await self._execute_tool(action, action_input)
            step.observation = observation
            steps.append(step)

        else:
            try:
                final_step = await self._finalize_after_max_steps(question, steps)
            except Exception:
                final_step = None

            if final_step is not None:
                steps.append(final_step)
                if final_step.action == "finish" and not final_step.parse_error:
                    trace.final_answer = final_step.action_input
                    trace.success = True
                    trace.error = None
                else:
                    trace.error = f"Max steps ({self.max_steps}) reached without finish"
                    trace.success = False
            else:
                trace.error = f"Max steps ({self.max_steps}) reached without finish"
                trace.success = False

        trace.steps = steps
        exa_cost = 0.0
        try:
            exa_cost = float(getattr(self.web_search_tool.exa_client, "total_cost", 0.0) or 0.0)
        except Exception:
            exa_cost = 0.0
        trace.total_cost = self.llm_client.total_cost + exa_cost
        trace.total_time = time.time() - start_time

        await self.e2b_tool.close()

        try:
            if self.browser_tool is not None:
                await self.browser_tool.close()
        except Exception:
            pass

        if self._reasoning_archive:
            setattr(trace, "_reasoning_archive", list(self._reasoning_archive))
        self._active_task_id = None

        return trace


async def create_agent(
    model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
    max_steps: int = 100,
    verbose: bool = False,
    allowed_tools: Optional[List[str]] = None,
) -> ReActAgent:
    from ..clients.exa import ExaClient

    llm_client = OpenRouterClient(
        model=model,
        temperature=1.0,
        top_p=0.95,
        max_tokens=32768,
        timeout=900,
        reasoning_effort="medium",
        reasoning_exclude=False,
    )

    exa_client = ExaClient()
    web_search_tool = WebSearchTool(exa_client=exa_client)
    web_contents_tool = WebContentsTool(exa_client=exa_client)
    e2b_tool = E2BPythonTool()

    browser_tool = None
    try:
        import os as _os

        if _os.getenv("BROWSERBASE_API_KEY") and _os.getenv("BROWSERBASE_PROJECT_ID"):
            browser_tool = BrowserbasePlaywrightTool()
    except Exception:
        browser_tool = None

    video_tool = VideoUnderstandingTool()

    return ReActAgent(
        llm_client=llm_client,
        web_search_tool=web_search_tool,
        web_contents_tool=web_contents_tool,
        e2b_tool=e2b_tool,
        browser_tool=browser_tool,
        video_tool=video_tool,
        allowed_tools=allowed_tools,
        max_steps=max_steps,
        verbose=verbose,
    )

