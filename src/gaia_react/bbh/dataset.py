"""
BBH (BIG-Bench Hard) Dataset Loader and Transformer.

Loads BBH from HuggingFace (lukaemon/bbh) and transforms MCQ tasks
to open-ended GAIA-style format suitable for ReAct trace collection.
"""

from __future__ import annotations

import re
import uuid
import hashlib
from typing import List, Optional
from dataclasses import dataclass, field


# BBH subtasks that are GAIA-like (need reasoning + computation, no external knowledge)
BBH_SUBTASKS = [
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "navigate",
    "penguins_in_a_table",
    "web_of_lies",
    "date_understanding",
    "reasoning_about_colored_objects",
    "temporal_sequences",
    "hyperbaton",
    "snarks",
    "disambiguation_qa",
    "geometric_shapes",
    "boolean_expressions",
    "causal_judgement",
    "movie_recommendation",
    "ruin_names",
    "salient_translation_error_detection",
    "formal_fallacies",
    "sports_understanding",
    "object_counting",
    "dyck_languages",
    "word_sorting",
    "multistep_arithmetic_two",
]


@dataclass
class BBHTask:
    """A single BBH task converted to GAIA-style format."""

    task_id: str
    question: str
    final_answer: str
    subtask: str = ""
    original_input: str = ""
    original_target: str = ""
    choices: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "final_answer": self.final_answer,
            "subtask": self.subtask,
            "original_input": self.original_input,
            "original_target": self.original_target,
            "choices": self.choices,
        }


def _generate_task_id(subtask: str, input_text: str) -> str:
    """Generate a deterministic UUID-like task ID from subtask + input."""
    content = f"{subtask}:{input_text}"
    hash_bytes = hashlib.sha256(content.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


def _transform_to_open_ended(input_text: str, target: str, subtask: str) -> tuple[str, str]:
    """
    Transform a BBH MCQ into an open-ended GAIA-style question.
    """
    lines = input_text.strip().split("\n")

    option_pattern = re.compile(r"^\s*\(([A-Z])\)\s*(.+)$")
    options_header_pattern = re.compile(r"^\s*Options:\s*$", re.IGNORECASE)

    question_lines = []
    options = {}
    in_options = False

    for line in lines:
        if options_header_pattern.match(line):
            in_options = True
            continue

        match = option_pattern.match(line)
        if match:
            in_options = True
            letter = match.group(1)
            content = match.group(2).strip()
            options[letter] = content
        elif not in_options:
            question_lines.append(line)

    base_question = "\n".join(question_lines).strip()

    # Remove trailing incomplete phrases
    base_question = re.sub(r"\s+(has|is|are|was|were)\s*$", "", base_question, flags=re.IGNORECASE)

    answer_match = re.match(r"\(([A-Z])\)", target.strip())
    if answer_match and options:
        answer_letter = answer_match.group(1)
        ground_truth = options.get(answer_letter, target.strip())
    else:
        ground_truth = target.strip()

    if subtask.startswith("tracking_shuffled_objects"):
        if "dancing" in base_question.lower() or "partner" in base_question.lower():
            transformed = (
                f"{base_question}.\n\nWho is the person mentioned at the end paired with? Put your answer in \\boxed{{}}."
            )
        else:
            transformed = (
                f"{base_question}.\n\nWhat does the person mentioned at the end have? Put your answer in \\boxed{{}}."
            )
    elif subtask.startswith("logical_deduction"):
        transformed = f"{base_question}\n\nAnswer with a complete statement. Put your full answer in \\boxed{{}}."
    elif subtask in {"navigate", "web_of_lies", "causal_judgement"}:
        transformed = f"{base_question}\n\nAnswer Yes or No. Put your answer in \\boxed{{}}."
    elif subtask == "boolean_expressions":
        transformed = f"{base_question}\n\nAnswer True or False. Put your answer in \\boxed{{}}."
    elif subtask == "date_understanding":
        transformed = f"{base_question}\n\nPut your answer in \\boxed{{}} using MM/DD/YYYY format."
    elif subtask in {"object_counting", "multistep_arithmetic_two"}:
        transformed = f"{base_question}\n\nPut just the number in \\boxed{{}}."
    elif subtask == "word_sorting":
        transformed = f"{base_question}\n\nPut the sorted words (space-separated) in \\boxed{{}}."
    elif subtask == "dyck_languages":
        transformed = f"{base_question}\n\nPut the completing bracket sequence in \\boxed{{}}."
    elif subtask == "geometric_shapes":
        transformed = f"{base_question}\n\nPut the shape name in \\boxed{{}}."
    elif subtask == "hyperbaton":
        transformed = f"{base_question}\n\nPut the correctly ordered phrase in \\boxed{{}}."
    elif subtask == "snarks":
        transformed = f"{base_question}\n\nCopy the sarcastic statement exactly into \\boxed{{}}."
    elif subtask == "disambiguation_qa":
        transformed = f"{base_question}\n\nPut your answer in \\boxed{{}}. Use a complete statement or 'Ambiguous'."
    elif subtask == "formal_fallacies":
        transformed = f"{base_question}\n\nAnswer 'valid' or 'invalid'. Put your answer in \\boxed{{}}."
    elif subtask == "movie_recommendation":
        transformed = f"{base_question}\n\nPut the movie title in \\boxed{{}}."
    elif subtask == "ruin_names":
        transformed = f"{base_question}\n\nPut the punny name in \\boxed{{}}."
    elif subtask == "salient_translation_error_detection":
        transformed = f"{base_question}\n\nPut the error type in \\boxed{{}}."
    else:
        transformed = f"{base_question}\n\nPut your answer in \\boxed{{}}."

    return transformed, ground_truth


def load_bbh_dataset(
    subtasks: Optional[List[str]] = None,
    limit_per_subtask: Optional[int] = None,
) -> List[BBHTask]:
    """
    Load BBH dataset from HuggingFace and transform to GAIA-style format.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets package required. Install with: pip install datasets") from e

    if subtasks is None:
        subtasks = BBH_SUBTASKS

    tasks: List[BBHTask] = []

    for subtask in subtasks:
        try:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
        except Exception as e:
            print(f"Warning: Could not load subtask {subtask}: {e}")
            continue

        count = 0
        for row in ds:
            if limit_per_subtask and count >= limit_per_subtask:
                break

            input_text = row.get("input", "")
            target = row.get("target", "")

            if not input_text or not target:
                continue

            question, answer = _transform_to_open_ended(input_text, target, subtask)
            task_id = _generate_task_id(subtask, input_text)

            tasks.append(
                BBHTask(
                    task_id=task_id,
                    question=question,
                    final_answer=answer,
                    subtask=subtask,
                    original_input=input_text,
                    original_target=target,
                )
            )
            count += 1

    return tasks


def get_bbh_stats(tasks: List[BBHTask]) -> dict:
    by_subtask = {}
    for task in tasks:
        by_subtask[task.subtask] = by_subtask.get(task.subtask, 0) + 1
    return {"total_tasks": len(tasks), "by_subtask": by_subtask}

