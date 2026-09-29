"""Gate features.

The rule that keeps this project honest: a feature may only be used if it is
already paid for by the shared gating prefill, or if it costs less than a few
percent of the cheapest action. Semantic entropy over sampled generations is a
much better correctness predictor, but it costs several extra decodes, so it
belongs in the REASON action, not in the gate. `FEATURE_COST_NOTE` records that
argument next to the code so the ablation table can quote it.
"""

from __future__ import annotations

import math
import re

FEATURE_NAMES = [
    "mean_logprob",
    "mean_entropy",
    "min_token_prob",
    "draft_len_tokens",
    "prompt_len_tokens",
    "question_words",
    "is_wh_question",
    "is_numeric_question",
    "has_rare_entity",
    "entity_count",
    "log_popularity",
    "probe_0",
    "probe_1",
    "probe_2",
    "probe_3",
    "probe_4",
    "probe_5",
    "probe_6",
    "probe_7",
]

BUDGET_FEATURE_NAMES = ["budget_latency", "budget_energy", "budget_memory"]

PROBE_DIM = 8

FEATURE_COST_NOTE = (
    "All features come from the single gating prefill that also produces the "
    "ANSWER action's draft, plus O(len(question)) string work. No feature "
    "requires an extra forward pass."
)

_NUMERIC = re.compile(r"\d")
_WH = re.compile(r"^\s*(who|what|when|where|which|why|how|whom|whose)\b", re.I)
_CAPS = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _project(hidden: list[float], dim: int = PROBE_DIM) -> list[float]:
    """Deterministic fixed-width summary of the mid-layer hidden state.

    Average-pooling into `dim` buckets rather than a learned projection: it
    costs nothing, needs no extra parameters, and keeps the feature vector the
    same width whichever backbone you swap in.
    """
    if not hidden:
        return [0.0] * dim
    size = max(1, len(hidden) // dim)
    buckets = []
    for i in range(dim):
        chunk = hidden[i * size : (i + 1) * size] or [0.0]
        buckets.append(sum(chunk) / len(chunk))
    return buckets


def extract(example, draft, budget: list[float] | None = None) -> list[float]:
    """Build the feature vector the policy consumes.

    Returns `FEATURE_NAMES` followed by `BUDGET_FEATURE_NAMES` when a budget is
    supplied, so the same policy can be queried at any operating point.
    """
    question = example.question
    words = question.split()
    popularity = example.meta.get("s_pop") if hasattr(example, "meta") else None

    values = [
        draft.mean_logprob,
        draft.mean_entropy,
        draft.min_token_prob,
        float(draft.completion_tokens),
        float(draft.prompt_tokens),
        float(len(words)),
        float(bool(_WH.match(question))),
        float(bool(_NUMERIC.search(question))),
        # Rare-entity proxy: a capitalised token the small model is unlikely to
        # have memorised. Mallen et al. show this is where retrieval pays off.
        float(len(_CAPS.findall(question)) > 0 and len(words) < 20),
        float(len(_CAPS.findall(question))),
        math.log1p(float(popularity)) if popularity else 0.0,
    ]
    values.extend(_project(draft.hidden))

    if budget is not None:
        # Log-scaled so the sweep multipliers are roughly linear in the input.
        values.extend(math.log1p(max(b, 0.0)) for b in budget)
    return values


def feature_names(with_budget: bool = True) -> list[str]:
    return FEATURE_NAMES + (BUDGET_FEATURE_NAMES if with_budget else [])


def feature_dim(with_budget: bool = True) -> int:
    return len(feature_names(with_budget))
