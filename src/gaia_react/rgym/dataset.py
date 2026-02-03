"""
Reasoning Gym (reasoning_gym) dataset adapter for trace collection.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..trace_task import TraceTask


def _repo_root() -> Path:
    # <repo>/src/gaia_react/rgym/dataset.py -> parents[4] == <repo>
    return Path(__file__).resolve().parents[4]


def default_reasoning_gym_root() -> Path:
    """
    Heuristic default for a local clone.

    Preferred:
    - $REASONING_GYM_ROOT (or $RGYM_ROOT)
    - sibling to this repo (common when both are cloned under the same workspace dir)
    - inside this repo
    - current working directory
    """
    env = os.getenv("REASONING_GYM_ROOT") or os.getenv("RGYM_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    candidates = [
        _repo_root().parent / "reasoning-gym",
        _repo_root() / "reasoning-gym",
        Path.cwd() / "reasoning-gym",
    ]
    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return candidates[0]


def _ensure_reasoning_gym_on_path(rg_root: Optional[str]) -> None:
    if not rg_root:
        return
    p = Path(str(rg_root)).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Reasoning Gym repo root not found: {p}")
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _import_reasoning_gym(rg_root: Optional[str]) -> Any:
    _ensure_reasoning_gym_on_path(rg_root)
    try:
        import reasoning_gym  # type: ignore
    except ModuleNotFoundError as e:
        msg = str(e)
        raise ModuleNotFoundError(
            "Failed to import `reasoning_gym`.\n"
            "- If you have a local clone, pass --rg-root /path/to/reasoning-gym\n"
            "- Ensure dependencies are installed (Reasoning Gym requires packages like sympy).\n"
            "Suggested fixes:\n"
            "  pip install reasoning-gym\n"
            "  # or:\n"
            f"  pip install -e {default_reasoning_gym_root()}\n"
            f"\nOriginal error: {msg}"
        ) from e
    return reasoning_gym


@contextlib.contextmanager
def _suppress_generation_noise():
    """
    Some RG datasets generate problems by executing snippets of Python code.

    Those snippets can emit:
    - `SyntaxWarning: invalid escape sequence ...` (from generated code strings)
    - debug `print(...)` output (e.g., metaclass demos)

    We silence stdout/stderr + those warnings during dataset construction/item generation
    so `collect_traces.py` logs stay clean.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)
        warnings.filterwarnings("ignore", message="invalid escape sequence", category=DeprecationWarning)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


def list_reasoning_gym_datasets(*, rg_root: Optional[str]) -> List[str]:
    reasoning_gym = _import_reasoning_gym(rg_root)
    from reasoning_gym.factory import DATASETS  # type: ignore

    return sorted(list(DATASETS.keys()))


@dataclass
class RGDatasetSpec:
    name: str
    weight: float = 1.0
    config: Dict[str, Any] = None


def _parse_rg_config_file(path: str, *, default_size: int, default_seed: int) -> Tuple[int, int, List[RGDatasetSpec]]:
    p = Path(str(path)).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"RG config not found: {p}")

    text = p.read_text(encoding="utf-8")
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "RG config appears to be YAML, but PyYAML is not available. "
                "Install Reasoning Gym deps (pip install reasoning-gym) or provide JSON instead."
            ) from e
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid RG config: expected mapping/object, got {type(data)}")

    # Format 1: HF helper config
    if "reasoning_gym" in data and isinstance(data.get("reasoning_gym"), dict):
        rg = data["reasoning_gym"]
        size = int(rg.get("dataset_size") or default_size)
        seed = int(rg.get("seed") or default_seed)
        ds_map = rg.get("datasets") or {}
        if not isinstance(ds_map, dict) or not ds_map:
            raise ValueError("RG config: reasoning_gym.datasets must be a non-empty mapping")
        specs: List[RGDatasetSpec] = []
        for name, cfg in ds_map.items():
            if not isinstance(cfg, dict):
                cfg = {}
            specs.append(
                RGDatasetSpec(
                    name=str(name),
                    weight=float(cfg.get("weight", 1.0)),
                    config=dict(cfg.get("config") or {}),
                )
            )
        return size, seed, specs

    # Format 2: CompositeConfig YAML-like
    if "datasets" in data and isinstance(data.get("datasets"), list):
        size = int(data.get("size") or default_size)
        seed = int(data.get("seed") or default_seed)
        specs: List[RGDatasetSpec] = []
        for ds in data.get("datasets") or []:
            if not isinstance(ds, dict) or not ds.get("name"):
                continue
            specs.append(
                RGDatasetSpec(
                    name=str(ds.get("name")),
                    weight=float(ds.get("weight", 1.0)),
                    config=dict(ds.get("config") or {}),
                )
            )
        if not specs:
            raise ValueError("RG config: datasets list is empty or invalid")
        return size, seed, specs

    raise ValueError(
        "Unrecognized RG config format. Provide either:\n"
        "- a JSON/YAML with top-level `reasoning_gym: {dataset_size, datasets: {...}}`, OR\n"
        "- a JSON/YAML with top-level `datasets: [ {name, weight, config}, ... ]`."
    )


def load_reasoning_gym_tasks(
    *,
    rg_root: Optional[str],
    config_path: Optional[str],
    dataset_name: Optional[str],
    dataset_config_json: Optional[str],
    size: int,
    seed: int,
    limit: Optional[int] = None,
    append_boxed_instruction: bool = True,
) -> List[TraceTask]:
    reasoning_gym = _import_reasoning_gym(rg_root)
    from reasoning_gym.factory import create_dataset, DATASETS  # type: ignore

    size = max(0, int(size))
    seed = int(seed)

    specs: List[RGDatasetSpec] = []
    if config_path:
        cfg_size, cfg_seed, cfg_specs = _parse_rg_config_file(str(config_path), default_size=size or 500, default_seed=seed)
        size = cfg_size
        seed = cfg_seed
        specs = cfg_specs
    else:
        name = str(dataset_name or "").strip()
        if not name:
            raise ValueError("For reasoning-gym source, you must pass either --rg-config PATH or --rg-dataset NAME")
        specs = [RGDatasetSpec(name=name, weight=1.0, config={})]

    if dataset_config_json and len(specs) == 1 and not config_path:
        try:
            cfg = json.loads(str(dataset_config_json))
        except Exception as e:
            raise ValueError("--rg-dataset-config-json must be valid JSON") from e
        if not isinstance(cfg, dict):
            raise ValueError("--rg-dataset-config-json must be a JSON object (mapping)")
        specs[0].config = dict(cfg)

    for s in specs:
        if s.name not in DATASETS:
            raise ValueError(
                f"Reasoning Gym dataset '{s.name}' not found/registered. Use --rg-list-datasets to see available names."
            )

    ds_obj: Any
    if len(specs) == 1:
        cfg = dict(specs[0].config or {})
        cfg.setdefault("seed", seed)
        cfg.setdefault("size", size)
        with _suppress_generation_noise():
            ds_obj = create_dataset(specs[0].name, **cfg)
    else:
        from reasoning_gym.composite import DatasetSpec  # type: ignore

        ds_specs = [DatasetSpec(name=s.name, weight=float(s.weight), config=dict(s.config or {})) for s in specs]
        with _suppress_generation_noise():
            ds_obj = create_dataset("composite", seed=seed, size=size, datasets=ds_specs)

    cap = min(size, int(limit)) if limit is not None else size
    cap = max(0, int(cap))

    tasks: List[TraceTask] = []
    for idx in range(cap):
        with _suppress_generation_noise():
            entry = ds_obj[idx]
        if not isinstance(entry, dict):
            continue
        q = str(entry.get("question") or "")
        a = str(entry.get("answer") or "")
        meta = entry.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {"raw_metadata": meta}

        src_ds = str(meta.get("source_dataset") or (specs[0].name if len(specs) == 1 else "composite"))

        if append_boxed_instruction:
            q = (
                q.rstrip()
                + "\n\n"
                + "IMPORTANT: When you finish, put ONLY the final answer inside \\boxed{...} (no extra text). "
                + "If the task requests a multi-line format, encode it into a single line representation."
            )

        task_id = f"rg:{src_ds}:{seed}:{idx:06d}"
        tasks.append(
            TraceTask(
                task_id=task_id,
                question=q,
                final_answer=a,
                source="reasoning_gym",
                split="generated",
                metadata={
                    "rg_dataset": src_ds,
                    "rg_index": int(idx),
                    "rg_seed": int(seed),
                    "rg_size": int(size),
                    "rg_config_path": str(config_path) if config_path else None,
                    "rg_spec_names": [s.name for s in specs],
                    "rg_spec_weights": [float(s.weight) for s in specs],
                    "rg_spec_configs": [dict(s.config or {}) for s in specs],
                    **meta,
                },
            )
        )

    return tasks

