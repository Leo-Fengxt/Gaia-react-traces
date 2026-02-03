"""
GAIA Dataset Loader.

Loads the GAIA benchmark dataset and filters out rows with files.
"""

from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GAIATask:
    """A single GAIA task."""

    task_id: str
    question: str
    level: int
    final_answer: str
    file_name: str = ""
    file_path: str = ""

    @property
    def has_files(self) -> bool:
        return bool(self.file_name) or bool(self.file_path)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "level": self.level,
            "final_answer": self.final_answer,
            "file_name": self.file_name,
            "file_path": self.file_path,
        }


def load_gaia_dataset(
    subset: str = "2023_all",
    split: str = "validation",
    filter_files: bool = True,
) -> List[GAIATask]:
    """
    Load the GAIA benchmark dataset.

    Note: GAIA is a gated dataset. You must accept the dataset terms on HF.
    """
    import os

    try:
        from datasets import load_dataset
        from datasets.exceptions import DatasetNotFoundError
    except ImportError as e:
        raise ImportError("datasets package required. Install with: pip install datasets") from e

    # Optional: login using HF_TOKEN if present (do not add to git credentials)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    try:
        dataset = load_dataset("gaia-benchmark/GAIA", subset, split=split)
    except DatasetNotFoundError as e:
        if "gated" in str(e).lower() or "authentication" in str(e).lower():
            raise RuntimeError(
                "GAIA is a gated dataset. Please:\n"
                "1. Go to https://huggingface.co/datasets/gaia-benchmark/GAIA\n"
                "2. Accept the dataset terms\n"
                "3. Set HF_TOKEN env var: export HF_TOKEN='your-token'\n"
                "   (Get token from: https://huggingface.co/settings/tokens)\n"
                f"Original error: {e}"
            ) from e
        raise

    tasks: List[GAIATask] = []

    for row in dataset:
        task_id = row.get("task_id", "")
        question = row.get("Question", "")
        level_raw = row.get("Level", 0)
        try:
            level = int(level_raw)
        except Exception:
            level = 0
        final_answer = row.get("Final answer", "")
        file_name = row.get("file_name", "") or ""
        file_path = row.get("file_path", "") or ""

        task = GAIATask(
            task_id=task_id,
            question=question,
            level=level,
            final_answer=final_answer,
            file_name=file_name,
            file_path=file_path,
        )

        if filter_files and task.has_files:
            continue

        tasks.append(task)

    return tasks


def load_gaia_by_level(
    subset: str = "2023_all",
    split: str = "validation",
    filter_files: bool = True,
) -> dict[int, List[GAIATask]]:
    tasks = load_gaia_dataset(subset, split, filter_files)
    by_level: dict[int, List[GAIATask]] = {1: [], 2: [], 3: []}
    for task in tasks:
        if task.level in by_level:
            by_level[task.level].append(task)
        else:
            by_level[task.level] = [task]
    return by_level


def get_dataset_stats(tasks: List[GAIATask]) -> dict:
    by_level: dict[int, int] = {}
    for task in tasks:
        by_level[task.level] = by_level.get(task.level, 0) + 1
    return {
        "total_tasks": len(tasks),
        "by_level": by_level,
        "has_files": sum(1 for t in tasks if t.has_files),
        "text_only": sum(1 for t in tasks if not t.has_files),
    }

