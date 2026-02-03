"""
Configuration for GAIA/BBH/ReasoningGym ReAct trace collection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    model: str = "google/gemini-2.5-flash-lite-preview-09-2025"
    temperature: float = 1.0
    # Use None to omit top_p entirely (provider/model default).
    top_p: Optional[float] = 0.95
    max_tokens: int = 32768

    # Reasoning configuration
    reasoning_effort: str = "medium"  # "none", "low", "medium", "high"
    # Include reasoning in provider response when supported; archived separately by collector.
    reasoning_exclude: bool = False

    timeout: int = 900  # seconds
    base_url: str = "https://openrouter.ai/api/v1"

    @property
    def api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return key


@dataclass
class ExaConfig:
    timeout: int = 60
    num_results: int = 25
    max_characters: int = 150000
    livecrawl: str = "fallback"

    @property
    def api_key(self) -> str:
        key = os.getenv("EXA_API_KEY")
        if not key:
            raise ValueError("EXA_API_KEY environment variable not set")
        return key


@dataclass
class E2BConfig:
    sandbox_timeout: int = 300
    execution_timeout: int = 120

    @property
    def api_key(self) -> str:
        key = os.getenv("E2B_API_KEY")
        if not key:
            raise ValueError("E2B_API_KEY environment variable not set")
        return key


@dataclass
class AgentConfig:
    max_steps: int = 100
    max_context_chars: int = 600_000
    keep_start_steps: int = 3
    keep_end_steps: int = 5
    max_parse_retries: int = 1
    allowed_tools: str = "all"


@dataclass
class RunConfig:
    concurrency: int = 8
    verbose: bool = False
    run_id: Optional[str] = None
    output_dir: str = "runs/collect"

    llm: LLMConfig = field(default_factory=LLMConfig)
    exa: ExaConfig = field(default_factory=ExaConfig)
    e2b: E2BConfig = field(default_factory=E2BConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

