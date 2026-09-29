"""Offline rollout collection.

For every training query we execute all four actions and record the outcome.
That gives a complete counterfactual table, so policy learning needs no
off-policy correction and no environment interaction: it becomes supervised
learning over a known payoff matrix. This is the expensive step (hours on one
GPU) and it only has to run once per model pair.

Output is JSON Lines, one object per query:

    {"qid": ..., "dataset": ..., "features": [...],
     "results": {"0": {...}, "1": {...}, "2": {...}, "3": {...}}}
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from quartet.actions import NUM_ACTIONS
from quartet.data import Example
from quartet.features import extract


def collect(engine, examples: Iterable[Example], out_path: str, resume: bool = True) -> str:
    """Run all four actions on every example and append them to `out_path`."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done: set[str] = set()
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"resuming: {len(done)} queries already rolled out")

    with open(out_path, "a", encoding="utf-8") as handle:
        for i, example in enumerate(examples):
            if example.qid in done:
                continue
            draft = engine.draft(example)
            record = {
                "qid": example.qid,
                "dataset": example.dataset,
                "task": example.task,
                "question": example.question,
                # Budget is appended at training time, so it is excluded here.
                "features": extract(example, draft, budget=None),
                "gate_latency_ms": draft.latency_ms,
                "gate_energy_j": draft.energy_j,
                "results": {},
            }
            for action in range(NUM_ACTIONS):
                try:
                    result = engine.run_action(action, example, draft)
                except Exception as exc:  # one bad action must not kill the run
                    print(f"  action {action} failed on {example.qid}: {exc}")
                    continue
                record["results"][str(action)] = result.to_dict()
            if len(record["results"]) == NUM_ACTIONS:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1} queries rolled out")
    return out_path


def load_rollouts(path: str) -> list[dict]:
    """Read back the rollout table, dropping any incomplete records."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if len(record.get("results", {})) == NUM_ACTIONS:
                records.append(record)
    return records


def to_arrays(records: list[dict]):
    """Stack the rollout table into the arrays the trainer expects.

    Returns (features, accuracy, costs) where
      features : (N, F)
      accuracy : (N, A)        correctness of each action
      costs    : (N, A, 3)     latency ms, energy J, peak memory MiB
    """
    import numpy as np

    features = np.asarray([r["features"] for r in records], dtype=np.float32)
    accuracy = np.zeros((len(records), NUM_ACTIONS), dtype=np.float32)
    costs = np.zeros((len(records), NUM_ACTIONS, 3), dtype=np.float32)
    for i, record in enumerate(records):
        for action in range(NUM_ACTIONS):
            result = record["results"][str(action)]
            accuracy[i, action] = result["correct"]
            costs[i, action] = [
                result["latency_ms"],
                result["energy_j"],
                result["peak_memory_mib"],
            ]
    return features, accuracy, costs
