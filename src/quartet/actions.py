"""The four inference actions and the cost record each one produces."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any


class Action(IntEnum):
    """The action space of the policy. Keep this order stable: the policy head,
    the rollout tables and the results table in the paper all index by it."""

    ANSWER = 0
    REASON = 1
    RETRIEVE = 2
    DELEGATE = 3


ACTION_NAMES = ["answer", "reason", "retrieve", "delegate"]
NUM_ACTIONS = len(ACTION_NAMES)

#: Human-readable blurbs, used in logs and in the report tables.
ACTION_DESCRIPTIONS = {
    Action.ANSWER: "emit the draft answer already produced during gating",
    Action.REASON: "extended chain-of-thought, k sampled paths, majority vote",
    Action.RETRIEVE: "BM25 top-k passages prepended, small model answers",
    Action.DELEGATE: "escalate the query to the large model",
}


@dataclass
class ActionResult:
    """What executing one action on one query produced, including its cost.

    These four cost channels are exactly the four objectives in the paper, so
    every field here shows up in a column of the results table.
    """

    action: int
    prediction: str
    correct: float = 0.0
    latency_ms: float = 0.0
    energy_j: float = 0.0
    peak_memory_mib: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Objective names in the order used by every cost vector in the codebase.
COST_KEYS = ["latency_ms", "energy_j", "peak_memory_mib"]


def cost_vector(result: ActionResult) -> list[float]:
    """The (latency, energy, memory) vector the Lagrangian constrains."""
    return [getattr(result, key) for key in COST_KEYS]
