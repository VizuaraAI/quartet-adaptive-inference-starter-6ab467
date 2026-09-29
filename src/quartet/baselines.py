"""Baseline action-selection rules.

Each baseline is a function (features, accuracy, costs) -> action index per
query, so every row of the paper's results table is produced by the same
evaluation code. Adding a baseline means adding one function and one entry to
`BASELINES`; nothing else changes.
"""

from __future__ import annotations

from quartet.actions import Action, NUM_ACTIONS
from quartet.features import FEATURE_NAMES


def _column(name: str) -> int:
    return FEATURE_NAMES.index(name)


def fixed(action: int):
    """Always take the same action."""

    def rule(features, accuracy, costs):
        import numpy as np

        return np.full(len(features), int(action), dtype=int)

    return rule


def random_action(seed: int = 0):
    def rule(features, accuracy, costs):
        import numpy as np

        return np.random.default_rng(seed).integers(0, NUM_ACTIONS, size=len(features))

    return rule


def oracle():
    """Upper bound: pick the cheapest action among those that are correct.

    Not achievable in deployment (it reads the label), but it is the number
    that tells you how much headroom any gate could possibly have.
    """

    def rule(features, accuracy, costs):
        import numpy as np

        # Correctness first; the normalised cost only breaks ties, so the
        # tiny weight must stay well below the smallest accuracy gap of 1.
        norm = costs / np.maximum(costs.reshape(-1, 3).max(axis=0), 1e-6)
        penalty = norm.sum(axis=2) / 3.0
        return np.argmax(accuracy - 1e-3 * penalty, axis=1)

    return rule


def confidence_cascade(threshold: float = -0.55):
    """FrugalGPT-style: answer if the draft is confident, otherwise delegate."""
    col = _column("mean_logprob")

    def rule(features, accuracy, costs):
        import numpy as np

        confident = features[:, col] >= threshold
        return np.where(confident, int(Action.ANSWER), int(Action.DELEGATE))

    return rule


def two_action_router(seed: int = 0):
    """Hybrid-LLM-style: a learned binary choice between answer and delegate."""

    def rule(features, accuracy, costs):
        return _learned_subset(features, accuracy, [Action.ANSWER, Action.DELEGATE], seed)

    return rule


def retrieval_complexity_router(seed: int = 0):
    """Adaptive-RAG-style: choose among answer, retrieve and delegate."""

    def rule(features, accuracy, costs):
        return _learned_subset(
            features, accuracy, [Action.ANSWER, Action.RETRIEVE, Action.DELEGATE], seed
        )

    return rule


def _learned_subset(features, accuracy, subset, seed):
    """Train a plain accuracy-only classifier over a restricted action set.

    This is the honest version of the comparison: these baselines optimise
    accuracy alone, because that is what the papers they stand in for do.
    Their cost is then whatever it turns out to be, which is exactly the
    point QUARTET is making.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    subset = [int(a) for a in subset]
    labels = np.asarray(subset)[np.argmax(accuracy[:, subset], axis=1)]
    if len(np.unique(labels)) < 2:
        return labels
    x_tr, _, y_tr, _ = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    clf = GradientBoostingClassifier(random_state=seed).fit(x_tr, y_tr)
    return clf.predict(features)


BASELINES = {
    "answer_always": fixed(Action.ANSWER),
    "reason_always": fixed(Action.REASON),
    "retrieve_always": fixed(Action.RETRIEVE),
    "delegate_always": fixed(Action.DELEGATE),
    "random": random_action(),
    "confidence_cascade": confidence_cascade(),
    "two_action_router": two_action_router(),
    "retrieval_complexity_router": retrieval_complexity_router(),
    "oracle": oracle(),
}
