"""QUARTET: a budget-conditioned inference policy for small language models.

The policy sees one query plus a deployment-time budget vector and picks one
of four actions:

    0  ANSWER    emit the draft answer produced during gating
    1  REASON    extended chain-of-thought with self-consistency voting
    2  RETRIEVE  BM25 evidence, then answer with the small model
    3  DELEGATE  escalate to a larger model

It is trained to maximise accuracy subject to latency, energy and memory
budgets, so the same set of weights serves the whole trade-off curve.

Heavy dependencies (torch, transformers, datasets) are imported lazily inside
the functions that need them, so `import quartet` stays cheap and works even
before you have downloaded any model weights.
"""

__version__ = "0.1.0"

from quartet.actions import ACTION_NAMES, NUM_ACTIONS, Action

__all__ = ["Action", "ACTION_NAMES", "NUM_ACTIONS", "__version__"]
