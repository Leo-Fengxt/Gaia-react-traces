"""
Prompts for the GAIA ReAct Agent.
"""

from __future__ import annotations

from typing import List, Optional

# Sentinel token used with the OpenAI/OpenRouter `stop` parameter to prevent runaway generations.
STOP_TOKEN = "<END_OF_STEP>"

ACTION_ORDER = [
    "web_search",
    "web_contents",
    "execute_python",
    "browser",
    "understand_video",
    "finish",
]

TOOL_INPUT_DOCS = {
    "web_search": (
        '- web_search: {"query": string, "num_results": int (optional), '
        '"search_type": "auto|neural|fast|deep" (optional), "include_domains": [string] (optional), '
        '"exclude_domains": [string] (optional), "category": string (optional)}'
    ),
    "web_contents": '- web_contents: {"urls": [string], "max_characters": int (optional)}',
    "execute_python": '- execute_python: {"code": string}',
    "browser": (
        '- browser: {"script": string (Playwright code), "proxies": bool (optional, enable for CAPTCHAs)}\n'
        "  The script runs against a connected `page` object (Playwright async API).\n"
        "  Available: page, context, browser, screenshot(full_page=False), get_downloads(), extract_downloads()\n"
        '  Example: {"script": "await page.goto(\'https://example.com\')\\nawait page.click(\'text=Login\')"}'
    ),
    "understand_video": (
        '- understand_video: {"prompt": string, "youtube_url": string (optional), "video_data_url": string (optional), '
        '"browserbase_session_id": string (optional), "browserbase_filename": string (optional), '
        '"browserbase_retry_seconds": int (optional)}'
    ),
    "finish": r"- finish: \boxed{...}",
}


def build_system_prompt(allowed_actions: Optional[List[str]] = None) -> str:
    """
    Build the ReAct system prompt with an optional per-run allowlist.
    """
    allow = {a.strip().lower() for a in (allowed_actions or ACTION_ORDER) if str(a).strip()}
    allow.add("finish")
    ordered = [a for a in ACTION_ORDER if a in allow]

    allowed_bullets = "\n".join(f"- {a}" for a in ordered)
    tool_inputs = "\n".join(TOOL_INPUT_DOCS[a] for a in ordered if a in TOOL_INPUT_DOCS)
    action_enum = "|".join(ordered)

    return f"""You are a tool-using ReAct agent.

You must produce EXACTLY ONE step per message.

## Allowed Actions
{allowed_bullets}

## TOOL INPUTS (JSON for tools)
{tool_inputs}

## RESPONSE FORMAT (exactly 3 lines, no extra text)
Thought: <1-3 short sentences>
Action: <{action_enum}>
Action Input: <JSON for tools OR \\boxed{{...}} for finish>

## CRITICAL RULES
- Do NOT output any extra prose, explanations, markdown, code fences, or multiple steps.
- Do NOT output "Observation:". Observations are provided by the environment.
- For tool actions, Action Input MUST be valid JSON on a SINGLE LINE.
  - For execute_python, put multi-line code inside the JSON string using \\n escapes.
- For finish, Action Input MUST be exactly one \\boxed{{...}} (no surrounding text).
- After the Action Input line, output the token {STOP_TOKEN} on its own line.

## ANTI-LOOP GUIDELINES (general)
- If you are stuck/unsure, use web_search to find authoritative sources, then web_contents to read exact steps/details. Do not guess.
- If you keep issuing web_search with only minor rewording, you are stuck. Change strategy.
- After web_search returns promising URLs, prefer web_contents to read primary sources and extract exact words.
- If the task asks for exact wording from a paper/figure/table, do NOT keep searching; open the source with web_contents.
- If you receive a WARNING about repeated web_search, consider switching action (web_contents or execute_python).

## HOW TO USE web_search (important)
- web_search is for DISCOVERY only. Use short keyword queries to find candidate sources (paper title fragments, arXiv IDs, authors, year/month).
- Do NOT keep repeating long natural-language queries. After 1-3 searches you should have candidate URLs.
- Then switch to web_contents to read the sources and extract the exact words.

## PDF / FIGURE NOTE (important)
- The model cannot read binary PDFs directly. Use web_contents on the PDF/HTML URL; Exa will return extracted text.
- If the needed words are only inside an image/figure (not present in extractable text), use execute_python to download and do PDF parsing / OCR.
"""


SYSTEM_PROMPT = build_system_prompt()


def build_initial_prompt(question: str) -> str:
    return f"""Question:
{question}

Provide your next step in the required 3-line format, then output {STOP_TOKEN}."""


def build_continuation_prompt(question: str, history: str) -> str:
    return f"""Question:
{question}

Previous steps (your actions + observations):
{history}

Provide your next step in the required 3-line format, then output {STOP_TOKEN}."""


def build_summarized_prompt(question: str, early_history: str, summary: str, recent_history: str) -> str:
    return f"""Question:
{question}

Early steps:
{early_history}

Summary of middle steps:
{summary}

Recent steps:
{recent_history}

Provide your next step in the required 3-line format, then output {STOP_TOKEN}."""


def build_reformat_prompt(invalid_response: str, allowed_actions: Optional[List[str]] = None) -> str:
    allow = {a.strip().lower() for a in (allowed_actions or ACTION_ORDER) if str(a).strip()}
    allow.add("finish")
    ordered = [a for a in ACTION_ORDER if a in allow]
    action_enum = "|".join(ordered)
    return "\n".join(
        [
            "Your previous response did not follow the required format.",
            "",
            "Restate ONLY ONE step using EXACTLY this 3-line format:",
            "Thought: <one short sentence>",
            f"Action: <{action_enum}>",
            r"Action Input: <JSON for tools OR \boxed{...} for finish>",
            f"{STOP_TOKEN}",
            "",
            "Your previous response was:",
            str(invalid_response),
        ]
    )

