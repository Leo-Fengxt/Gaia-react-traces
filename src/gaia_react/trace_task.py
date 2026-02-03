"""
Common task schema for trace collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TraceTask:
    """
    A single task to run the agent on and save a trace for.
    """

    task_id: str
    question: str
    final_answer: str = ""
    source: str = ""
    split: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "final_answer": self.final_answer,
            "source": self.source,
            "split": self.split,
            "metadata": self.metadata,
        }

