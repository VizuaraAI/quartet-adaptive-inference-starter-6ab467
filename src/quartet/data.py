"""Dataset loading and answer scoring.

Three public datasets, chosen so that no single action wins everywhere:

    popqa    long-tail entity questions where the small model fails and
             retrieval helps (Mallen et al., 2023)
    gsm8k    grade-school maths where extra reasoning helps and retrieval
             does not
    hotpotqa multi-hop questions where retrieval plus reasoning are both
             needed, and delegation is often the only thing that works

Everything is pulled from the Hugging Face hub at first use. Nothing large is
committed to this repository.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

HF_SPECS = {
    "popqa": ("akariasai/PopQA", None, "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
}


@dataclass
class Example:
    qid: str
    question: str
    answers: list[str]
    dataset: str
    task: str  # "qa" or "math"
    meta: dict[str, Any] = field(default_factory=dict)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [value]
    return [str(value)]


def load_dataset_split(name: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    """Load one dataset and normalise it into `Example` records."""
    from datasets import load_dataset

    if name not in HF_SPECS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(HF_SPECS)}")
    repo, subset, split = HF_SPECS[name]
    raw = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
    raw = raw.shuffle(seed=seed)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))

    out: list[Example] = []
    for i, row in enumerate(raw):
        if name == "popqa":
            out.append(
                Example(
                    qid=f"popqa-{row.get('id', i)}",
                    question=row["question"],
                    answers=_as_list(row["possible_answers"]),
                    dataset=name,
                    task="qa",
                    # s_pop is subject-entity popularity: the cheapest possible
                    # signal that retrieval will be needed.
                    meta={"s_pop": row.get("s_pop"), "prop": row.get("prop")},
                )
            )
        elif name == "gsm8k":
            gold = row["answer"].split("####")[-1].strip()
            out.append(
                Example(
                    qid=f"gsm8k-{i}",
                    question=row["question"],
                    answers=[gold],
                    dataset=name,
                    task="math",
                )
            )
        else:  # hotpotqa
            out.append(
                Example(
                    qid=f"hotpotqa-{row.get('id', i)}",
                    question=row["question"],
                    answers=[row["answer"]],
                    dataset=name,
                    task="qa",
                    meta={"level": row.get("level"), "type": row.get("type")},
                )
            )
    return out


def load_splits(cfg) -> dict[str, list[Example]]:
    """Build train / val / test pools, mixed across the configured datasets."""
    import random

    per_dataset = cfg.data.train_size + cfg.data.val_size + cfg.data.test_size
    pool: list[Example] = []
    for name in cfg.data.datasets:
        pool.extend(load_dataset_split(name, limit=per_dataset, seed=cfg.data.seed))
    random.Random(cfg.data.seed).shuffle(pool)

    a, b = cfg.data.train_size, cfg.data.train_size + cfg.data.val_size
    return {
        "train": pool[:a],
        "val": pool[a:b],
        "test": pool[b : b + cfg.data.test_size],
    }


# ---------------------------------------------------------------- scoring ---

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in answers))


def contains_match(prediction: str, answers: Iterable[str]) -> float:
    """PopQA convention: credit if any gold answer appears in the output."""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) in pred for a in answers if a))


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    best = 0.0
    pred_tokens = normalize_answer(prediction).split()
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def final_number(text: str) -> str | None:
    """Last number in the string: how GSM8K answers are conventionally read."""
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def score(example: Example, prediction: str) -> float:
    """The single correctness scalar the policy is trained against."""
    if example.task == "math":
        pred = final_number(prediction)
        gold = final_number(example.answers[0]) if example.answers else None
        if pred is None or gold is None:
            return 0.0
        try:
            return float(abs(float(pred) - float(gold)) < 1e-4)
        except ValueError:
            return float(pred == gold)
    if example.dataset == "popqa":
        return contains_match(prediction, example.answers)
    return max(exact_match(prediction, example.answers), token_f1(prediction, example.answers))
