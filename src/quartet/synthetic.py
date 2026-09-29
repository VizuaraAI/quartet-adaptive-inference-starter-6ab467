"""A synthetic rollout table, so the pipeline runs before you own a GPU hour.

This exists to let you exercise `train`, `evaluate` and `ablate` on day one and
to give the test suite something deterministic to check. The generator builds
in the structure the real problem has, and nothing more:

  * a latent difficulty per query, which the gate features report noisily
  * a latent "needs external knowledge" flag, which only RETRIEVE fixes
  * a latent "needs multi-step work" flag, which REASON and DELEGATE fix
  * costs that are ordered ANSWER < RETRIEVE < REASON < DELEGATE

Any number produced from this table is a property of the generator, not of a
language model. Never put one in the paper.
"""

from __future__ import annotations

import json
import os

from quartet.actions import NUM_ACTIONS
from quartet.features import FEATURE_NAMES


def generate(n: int = 4000, seed: int = 0) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    records = []

    for i in range(n):
        difficulty = rng.beta(2.0, 2.5)
        needs_knowledge = rng.random() < 0.35
        needs_steps = rng.random() < 0.30

        # Gate features: a noisy view of difficulty, plus the lexical signals.
        logprob = -0.25 - 1.5 * difficulty + rng.normal(0, 0.18)
        entropy = 0.4 + 2.2 * difficulty + rng.normal(0, 0.25)
        features = [
            logprob,
            max(entropy, 0.0),
            float(np.clip(np.exp(logprob) + rng.normal(0, 0.05), 0.0, 1.0)),
            float(rng.integers(4, 28)),
            float(rng.integers(12, 90)),
            float(rng.integers(6, 24)),
            float(rng.random() < 0.7),
            float(needs_steps and rng.random() < 0.8),
            float(needs_knowledge and rng.random() < 0.85),
            float(rng.integers(0, 4)),
            0.0 if needs_knowledge else float(rng.uniform(2.0, 9.0)),
        ]
        features += list(rng.normal(0, 1, size=len(FEATURE_NAMES) - len(features)))

        # Which actions actually succeed.
        base = 0.92 - 0.85 * difficulty
        p_answer = base * (0.25 if needs_knowledge else 1.0) * (0.35 if needs_steps else 1.0)
        p_reason = base * (0.30 if needs_knowledge else 1.0) * (0.85 if needs_steps else 1.0) + 0.05
        p_retrieve = base * (0.90 if needs_knowledge else 0.80) * (0.40 if needs_steps else 1.0)
        p_delegate = 0.97 - 0.45 * difficulty
        probs = [p_answer, p_reason, p_retrieve, p_delegate]

        results = {}
        for action, p in enumerate(probs):
            correct = float(rng.random() < np.clip(p, 0.01, 0.99))
            latency, energy, memory = _costs(action, rng)
            results[str(action)] = {
                "action": action,
                "prediction": "",
                "correct": correct,
                "latency_ms": latency,
                "energy_j": energy,
                "peak_memory_mib": memory,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "meta": {"synthetic": True},
            }

        records.append(
            {
                "qid": f"synthetic-{i}",
                "dataset": "synthetic",
                "task": "qa",
                "question": "",
                "features": [float(v) for v in features],
                "gate_latency_ms": 42.0,
                "gate_energy_j": 1.1,
                "results": results,
            }
        )
    return records


def _costs(action: int, rng):
    """Cost profiles roughly matching a 1.5B local model against a 14B one."""
    profile = {
        0: (0.0, 0.0, 3200),      # ANSWER reuses the gating pass
        1: (1850, 24.0, 4100),    # REASON: five sampled chains
        2: (620, 9.5, 5400),      # RETRIEVE: BM25 plus a longer context
        3: (2400, 88.0, 17800),   # DELEGATE: the large model
    }[action]
    jitter = lambda v, s: float(max(v * (1.0 + rng.normal(0, s)), 0.0))
    return jitter(profile[0], 0.18), jitter(profile[1], 0.15), jitter(profile[2], 0.05)


def write(path: str, n: int = 4000, seed: int = 0) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    records = generate(n, seed)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    assert all(len(r["results"]) == NUM_ACTIONS for r in records)
    return path
