"""Trace collection CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm.asyncio import tqdm_asyncio

from .agent.react_agent import ReActAgent, ReActTrace
from .bbh.dataset import BBH_SUBTASKS, load_bbh_dataset
from .clients.exa import ExaClient
from .clients.openrouter import OpenRouterClient
from .config import AgentConfig, E2BConfig, ExaConfig, LLMConfig, RunConfig
from .gaia.dataset import load_gaia_dataset
from .rgym.dataset import list_reasoning_gym_datasets, load_reasoning_gym_tasks
from .tools.browserbase_playwright import BrowserbasePlaywrightTool
from .tools.disabled import DisabledE2BPythonTool, DisabledWebContentsTool, DisabledWebSearchTool
from .tools.e2b_python import E2BPythonTool
from .tools.video_understanding import VideoUnderstandingTool
from .tools.web_contents import WebContentsTool
from .tools.web_search import WebSearchTool
from .trace_task import TraceTask


TraceType = ReActTrace


@dataclass
class TraceResult:
    task_id: str
    source: str
    trace_path: str
    success: bool
    error: Optional[str]
    total_time: float
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "trace_path": self.trace_path,
            "success": self.success,
            "error": self.error,
            "total_time": self.total_time,
            "total_cost": self.total_cost,
        }


_BOXED_RE = re.compile(r"\\+boxed\{([\s\S]*?)\}")
_DECIMAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _safe_file_stem_from_task_id(task_id: str) -> str:
    return str(task_id).replace(":", "-")


def _normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    if not s:
        return ""

    last_boxed: Optional[str] = None
    for m in _BOXED_RE.finditer(s):
        last_boxed = m.group(1)
    if last_boxed is not None:
        s = str(last_boxed).strip()

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    return " ".join(s.split())


def _maybe_parse_decimal(s: str) -> Optional[Decimal]:
    s2 = str(s).strip().replace(",", "")
    if not _DECIMAL_RE.match(s2):
        return None
    try:
        return Decimal(s2)
    except InvalidOperation:
        return None


def _answers_match(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    if pred.casefold() == gold.casefold():
        return True
    p = _maybe_parse_decimal(pred)
    g = _maybe_parse_decimal(gold)
    if p is not None and g is not None:
        return p == g
    return False


def _is_correct(pred: Optional[str], gold: str) -> Optional[bool]:
    gold_n = _normalize_answer(gold)
    if not gold_n:
        return None
    pred_n = _normalize_answer(pred)
    return _answers_match(pred_n, gold_n)


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


def _resolve_allowed_tools(raw_allowed: str) -> List[str]:
    supported = {"web_search", "web_contents", "execute_python", "browser", "understand_video"}
    raw = str(raw_allowed or "all").strip().lower()

    browser_configured = bool(os.getenv("BROWSERBASE_API_KEY") and os.getenv("BROWSERBASE_PROJECT_ID"))

    if raw in {"all", "*"}:
        out = set(supported)
        if not browser_configured:
            out.discard("browser")
        return sorted(out)

    out = {p.strip().lower() for p in raw.split(",") if p.strip()}
    unknown = sorted(list(out - supported))
    if unknown:
        raise ValueError(f"Unknown tools in --allowed-tools: {unknown}. Supported: {sorted(list(supported))} or 'all'.")
    if "browser" in out and not browser_configured:
        raise RuntimeError(
            "Tool 'browser' was requested via --allowed-tools but Browserbase is not configured. "
            "Set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID."
        )
    return sorted(out)


def _check_required_keys(*, allowed_tools: Sequence[str]) -> None:
    missing: List[str] = []
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")

    tool_set = set(str(t).strip().lower() for t in allowed_tools)
    if tool_set & {"web_search", "web_contents"}:
        if not os.getenv("EXA_API_KEY"):
            missing.append("EXA_API_KEY")
    if "execute_python" in tool_set:
        if not os.getenv("E2B_API_KEY"):
            missing.append("E2B_API_KEY")
    if "browser" in tool_set:
        if not os.getenv("BROWSERBASE_API_KEY"):
            missing.append("BROWSERBASE_API_KEY")
        if not os.getenv("BROWSERBASE_PROJECT_ID"):
            missing.append("BROWSERBASE_PROJECT_ID")

    if missing:
        print(f"Error: Missing required environment variables: {missing}")
        for k in missing:
            print(f"  export {k}='your-api-key'")
        raise SystemExit(1)


def _convert_tasks_with_prefix(source: str, tasks: Iterable[TraceTask]) -> List[TraceTask]:
    out: List[TraceTask] = []
    prefix = f"{source}:"
    for t in tasks:
        tid = str(t.task_id)
        if not tid.startswith(prefix):
            tid = prefix + tid
        out.append(
            TraceTask(
                task_id=tid,
                question=t.question,
                final_answer=t.final_answer,
                source=source,
                split=t.split or t.source or "",
                metadata=dict(t.metadata or {}),
            )
        )
    return out


def _load_source_tasks(args: argparse.Namespace, source: str) -> List[TraceTask]:
    source = str(source).strip().lower()
    seed = int(getattr(args, "seed", 0) or 0)
    limit = getattr(args, "limit", None)
    per_source_limit = getattr(args, "limit_per_source", None)
    cap = per_source_limit if per_source_limit is not None else limit

    if source in {"reasoning-gym", "reasoning_gym", "rgym"}:
        rg_root_raw = getattr(args, "rg_root", None)
        rg_root = str(rg_root_raw).strip() if rg_root_raw else None
        if rg_root == "":
            rg_root = None
        if bool(getattr(args, "rg_list_datasets", False)):
            names = list_reasoning_gym_datasets(rg_root=rg_root)
            print("\n".join(names))
            raise SystemExit(0)

        rg_seed = int(getattr(args, "rg_seed", None) or seed)
        rg_size = int(getattr(args, "rg_size", None) or (cap if cap is not None else 200))
        tasks = load_reasoning_gym_tasks(
            rg_root=rg_root,
            config_path=getattr(args, "rg_config", None),
            dataset_name=getattr(args, "rg_dataset", None),
            dataset_config_json=getattr(args, "rg_dataset_config_json", None),
            size=rg_size,
            seed=rg_seed,
            limit=cap,
            append_boxed_instruction=not bool(getattr(args, "rg_no_boxed_instruction", False)),
        )
        return tasks

    if source == "bbh":
        subtasks = getattr(args, "bbh_subtasks", None)
        if bool(getattr(args, "bbh_all_subtasks", False)) or not subtasks:
            subtasks = list(BBH_SUBTASKS)
        bbh_limit_per_subtask = getattr(args, "bbh_limit_per_subtask", None)
        bbh = load_bbh_dataset(subtasks=list(subtasks), limit_per_subtask=bbh_limit_per_subtask)
        out: List[TraceTask] = []
        for t in bbh:
            out.append(
                TraceTask(
                    task_id=str(t.task_id),
                    question=str(t.question),
                    final_answer=str(t.final_answer),
                    source="bbh",
                    split="test",
                    metadata={"subtask": str(getattr(t, "subtask", "") or "")},
                )
            )
        if cap is not None:
            out = out[: int(cap)]
        return _convert_tasks_with_prefix("bbh", out)

    if source == "gaia":
        subset = str(getattr(args, "gaia_subset", "2023_all"))
        split = str(getattr(args, "gaia_split", "validation"))
        gaia = load_gaia_dataset(subset=subset, split=split, filter_files=True)
        out: List[TraceTask] = []
        for t in gaia:
            out.append(
                TraceTask(
                    task_id=str(t.task_id),
                    question=str(t.question),
                    final_answer=str(t.final_answer),
                    source="gaia",
                    split=str(split),
                    metadata={"level": int(getattr(t, "level", 0) or 0)},
                )
            )
        if cap is not None:
            out = out[: int(cap)]
        return _convert_tasks_with_prefix("gaia", out)

    raise ValueError(f"Unknown source: {source}")


class TraceCollector:
    def __init__(
        self,
        *,
        config: RunConfig,
        allowed_tools: Sequence[str],
        output_dir: str,
        run_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.allowed_tools = [str(x) for x in allowed_tools]
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        self.output_dir = Path(output_dir) / self.run_id
        self.traces_dir = self.output_dir / "traces"
        self.reasoning_dir = self.output_dir / "reasoning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

        self.semaphore = asyncio.Semaphore(int(config.concurrency))
        self.results: List[TraceResult] = []

        self._stats_lock = asyncio.Lock()
        self._done_count: int = 0
        self._steps_sum: int = 0
        self._scored_count: int = 0
        self._correct_count: int = 0

    def _write_reasoning_archive(self, *, task_id: str, archive: Any) -> None:
        if not archive:
            return
        if not isinstance(archive, list):
            return
        path = self.reasoning_dir / f"{_safe_file_stem_from_task_id(task_id)}.json"
        payload = {
            "task_id": str(task_id),
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "agent": "react",
            "calls": archive,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    async def _note_one_done(
        self,
        *,
        total_steps: int,
        is_correct: Optional[bool],
        pbar: Optional[tqdm_asyncio],
    ) -> None:
        async with self._stats_lock:
            self._done_count += 1
            self._steps_sum += int(total_steps)
            if is_correct is not None:
                self._scored_count += 1
                if bool(is_correct):
                    self._correct_count += 1

            if pbar is not None:
                if not bool(getattr(self.config, "verbose", False)):
                    mean_steps = (self._steps_sum / max(1, self._done_count)) if self._done_count else 0.0
                    postfix: Dict[str, Any] = {"mean_steps": f"{mean_steps:.2f}"}
                    if self._scored_count > 0:
                        acc = self._correct_count / self._scored_count
                        postfix["correct"] = f"{acc*100:.1f}% ({self._correct_count}/{self._scored_count})"
                    pbar.set_postfix(postfix)
                pbar.update(1)

    async def _build_agent(self) -> ReActAgent:
        llm_client = OpenRouterClient(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            max_tokens=self.config.llm.max_tokens,
            timeout=self.config.llm.timeout,
            reasoning_effort=self.config.llm.reasoning_effort,
            reasoning_exclude=self.config.llm.reasoning_exclude,
        )

        tool_set = set(self.allowed_tools)

        browser_tool = None
        browser_configured = bool(os.getenv("BROWSERBASE_API_KEY") and os.getenv("BROWSERBASE_PROJECT_ID"))
        if "browser" in tool_set and browser_configured:
            try:
                browser_tool = BrowserbasePlaywrightTool()
            except Exception:
                browser_tool = None

        if tool_set & {"web_search", "web_contents"}:
            exa_client = ExaClient(timeout=self.config.exa.timeout)
            web_search_tool = WebSearchTool(exa_client=exa_client, default_num_results=self.config.exa.num_results)
            web_contents_tool = WebContentsTool(
                exa_client=exa_client,
                max_characters=self.config.exa.max_characters,
                livecrawl=self.config.exa.livecrawl,
            )
        else:
            web_search_tool = DisabledWebSearchTool()  # type: ignore[assignment]
            web_contents_tool = DisabledWebContentsTool()  # type: ignore[assignment]

        if "execute_python" in tool_set:
            e2b_tool = E2BPythonTool(
                sandbox_timeout=self.config.e2b.sandbox_timeout,
                execution_timeout=self.config.e2b.execution_timeout,
            )
        else:
            e2b_tool = DisabledE2BPythonTool()  # type: ignore[assignment]

        video_tool = VideoUnderstandingTool() if "understand_video" in tool_set else None

        return ReActAgent(
            llm_client=llm_client,
            web_search_tool=web_search_tool,
            web_contents_tool=web_contents_tool,
            e2b_tool=e2b_tool,
            browser_tool=browser_tool,
            video_tool=video_tool,
            allowed_tools=sorted(list(tool_set)),
            max_steps=self.config.agent.max_steps,
            max_context_chars=self.config.agent.max_context_chars,
            keep_start_steps=self.config.agent.keep_start_steps,
            keep_end_steps=self.config.agent.keep_end_steps,
            max_parse_retries=self.config.agent.max_parse_retries,
            verbose=bool(getattr(self.config, "verbose", False)),
        )

    async def run_one(self, task: TraceTask, *, pbar: Optional[tqdm_asyncio] = None) -> TraceResult:
        total_steps = 0
        correct: Optional[bool] = None
        async with self.semaphore:
            trace_path = self.traces_dir / f"{_safe_file_stem_from_task_id(task.task_id)}.json"
            try:
                agent = await self._build_agent()
                trace: TraceType = await agent.run(task.task_id, task.question)

                total_steps = len(getattr(trace, "steps", []) or [])
                correct = _is_correct(getattr(trace, "final_answer", None), str(getattr(task, "final_answer", "") or ""))

                payload: Dict[str, Any] = trace.to_dict()
                payload["task"] = task.to_dict()

                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

                self._write_reasoning_archive(
                    task_id=task.task_id,
                    archive=getattr(trace, "_reasoning_archive", None),
                )

                return TraceResult(
                    task_id=task.task_id,
                    source=task.source or "",
                    trace_path=str(trace_path),
                    success=bool(getattr(trace, "success", False)),
                    error=getattr(trace, "error", None),
                    total_time=float(getattr(trace, "total_time", 0.0) or 0.0),
                    total_cost=float(getattr(trace, "total_cost", 0.0) or 0.0),
                )
            except Exception as e:
                correct = _is_correct(None, str(getattr(task, "final_answer", "") or ""))
                stub = {
                    "task_id": task.task_id,
                    "question": task.question,
                    "steps": [],
                    "final_answer": None,
                    "success": False,
                    "error": str(e),
                    "total_cost": 0.0,
                    "total_time": 0.0,
                    "task": task.to_dict(),
                }
                try:
                    with open(trace_path, "w", encoding="utf-8") as f:
                        json.dump(stub, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                return TraceResult(
                    task_id=task.task_id,
                    source=task.source or "",
                    trace_path=str(trace_path),
                    success=False,
                    error=str(e),
                    total_time=0.0,
                    total_cost=0.0,
                )
            finally:
                await self._note_one_done(total_steps=total_steps, is_correct=correct, pbar=pbar)

    async def run(self, tasks: List[TraceTask]) -> None:
        with open(self.output_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "agent": "react",
                    "model": self.config.llm.model,
                    "temperature": self.config.llm.temperature,
                    "top_p": self.config.llm.top_p,
                    "max_tokens": self.config.llm.max_tokens,
                    "reasoning_effort": self.config.llm.reasoning_effort,
                    "reasoning_exclude": self.config.llm.reasoning_exclude,
                    "max_steps": self.config.agent.max_steps,
                    "allowed_tools": list(self.allowed_tools),
                    "concurrency": self.config.concurrency,
                    "num_tasks": len(tasks),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        async def run_with_progress() -> List[TraceResult]:
            with tqdm_asyncio(total=len(tasks), desc="Collecting traces") as pbar:
                coros = [self.run_one(t, pbar=pbar) for t in tasks]
                return await asyncio.gather(*coros, return_exceptions=False)

        self.results = await run_with_progress()

        with open(self.output_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        total_cost = sum(float(r.total_cost or 0.0) for r in self.results)
        total_time = sum(float(r.total_time or 0.0) for r in self.results)
        ok = sum(1 for r in self.results if r.success)
        summary = {
            "run_id": self.run_id,
            "num_tasks": len(self.results),
            "num_success": ok,
            "num_failed": len(self.results) - ok,
            "total_cost": total_cost,
            "total_time": total_time,
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nSaved traces to: {self.output_dir}")
        print(f"  - traces/: {len(self.results)}")
        print("  - tasks.jsonl, results.jsonl, summary.json, config.json")
        if not bool(getattr(self.config, "verbose", False)):
            mean_steps = (self._steps_sum / max(1, self._done_count)) if self._done_count else 0.0
            print(f"  - mean_total_steps: {mean_steps:.2f}")
            if self._scored_count > 0:
                acc = self._correct_count / self._scored_count
                print(f"  - correct_rate: {acc*100:.2f}% ({self._correct_count}/{self._scored_count})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect ReAct traces across GAIA/BBH/ReasoningGym",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["bbh"],
        choices=["bbh", "gaia", "reasoning-gym"],
        help="One or more sources to collect from",
    )

    # General run controls
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="runs/collect")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Global cap across all sources (applied after concatenation)")
    p.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="Cap tasks per source (applied during source load). If set, overrides --limit for per-source loading.",
    )
    p.add_argument("--verbose", action="store_true")

    # LLM / agent
    p.add_argument("--model", type=str, default="google/gemini-2.5-flash-lite-preview-09-2025")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-top-p", action="store_true", default=False)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--reasoning-effort", type=str, default="medium", choices=["none", "low", "medium", "high"])
    p.add_argument("--reasoning-exclude", action="store_true", default=False)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--allowed-tools", type=str, default="all")

    # BBH
    p.add_argument("--bbh-subtasks", nargs="+", default=None)
    p.add_argument("--bbh-all-subtasks", action="store_true", default=False)
    p.add_argument("--bbh-limit-per-subtask", type=int, default=None)

    # GAIA
    p.add_argument("--gaia-subset", type=str, default="2023_all")
    p.add_argument("--gaia-split", type=str, default="validation")

    # Reasoning Gym
    p.add_argument(
        "--rg-root",
        type=str,
        default=None,
        help="Optional path to a local reasoning-gym repo root. Omit if `reasoning_gym` is installed.",
    )
    p.add_argument(
        "--rg-list-datasets",
        dest="rg_list_datasets",
        action="store_true",
        default=False,
        help="List available Reasoning Gym dataset names and exit (use with --sources reasoning-gym).",
    )
    p.add_argument("--rg-config", type=str, default=None, help="Reasoning Gym dataset config (YAML/JSON).")
    p.add_argument("--rg-dataset", type=str, default=None, help="Single Reasoning Gym dataset name (ignored if --rg-config).")
    p.add_argument(
        "--rg-dataset-config-json",
        type=str,
        default=None,
        help="JSON object with dataset-specific config fields for --rg-dataset.",
    )
    p.add_argument("--rg-size", type=int, default=None, help="How many Reasoning Gym tasks to generate for this run.")
    p.add_argument("--rg-seed", type=int, default=None, help="Seed for Reasoning Gym generation (defaults to --seed).")
    p.add_argument(
        "--rg-no-boxed-instruction",
        action="store_true",
        default=False,
        help="Do not append a compatibility note about \\boxed{...} to RG questions.",
    )

    return p.parse_args()


async def main() -> None:
    args = parse_args()
    _load_dotenv_if_present()

    # Special mode: list Reasoning Gym datasets without requiring any API keys.
    if bool(getattr(args, "rg_list_datasets", False)):
        srcs = [str(s).strip().lower() for s in list(getattr(args, "sources", []) or [])]
        if srcs and all(s in {"reasoning-gym", "reasoning_gym", "rgym"} for s in srcs):
            rg_root_raw = getattr(args, "rg_root", None)
            rg_root = str(rg_root_raw).strip() if rg_root_raw else None
            if rg_root == "":
                rg_root = None
            names = list_reasoning_gym_datasets(rg_root=rg_root)
            print("\n".join(names))
            return

    allowed_tools = _resolve_allowed_tools(str(getattr(args, "allowed_tools", "all")))
    _check_required_keys(allowed_tools=allowed_tools)

    top_p = None if bool(getattr(args, "no_top_p", False)) else float(getattr(args, "top_p", 0.95))
    config = RunConfig(
        concurrency=int(getattr(args, "concurrency", 8)),
        verbose=bool(getattr(args, "verbose", False)),
        run_id=getattr(args, "run_id", None),
        output_dir=str(getattr(args, "output_dir", "runs/collect")),
        llm=LLMConfig(
            model=str(getattr(args, "model")),
            temperature=float(getattr(args, "temperature", 1.0)),
            top_p=top_p,
            max_tokens=int(getattr(args, "max_tokens", 32768)),
            reasoning_effort=str(getattr(args, "reasoning_effort", "medium")),
            reasoning_exclude=bool(getattr(args, "reasoning_exclude", False)),
        ),
        exa=ExaConfig(),
        e2b=E2BConfig(),
        agent=AgentConfig(
            max_steps=int(getattr(args, "max_steps", 100)),
            allowed_tools=str(getattr(args, "allowed_tools", "all")),
        ),
    )

    tasks: List[TraceTask] = []
    for src in list(getattr(args, "sources", []) or []):
        tasks.extend(_load_source_tasks(args, str(src)))

    if getattr(args, "limit", None) is not None and tasks:
        tasks = tasks[: int(getattr(args, "limit"))]

    if not tasks:
        print("No tasks selected.")
        raise SystemExit(1)

    collector = TraceCollector(
        config=config,
        allowed_tools=allowed_tools,
        output_dir=str(getattr(args, "output_dir", "runs/collect")),
        run_id=getattr(args, "run_id", None),
    )
    await collector.run(tasks)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()

