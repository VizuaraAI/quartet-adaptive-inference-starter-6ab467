"""Evaluation: turn a set of per-query action choices into table rows.

The column names here are exactly the column names in the paper's Table 1 and
in `experiments/log_template.md`. Keep those three in sync and writing up
becomes copying numbers rather than reconciling them.
"""

from __future__ import annotations

import json
import os

from quartet.actions import ACTION_NAMES, NUM_ACTIONS


def summarise(actions, accuracy, costs, latency_slo_ms=None) -> dict:
    """Score one selection rule over the test pool."""
    import numpy as np

    actions = np.asarray(actions, dtype=int)
    rows = np.arange(len(actions))
    chosen_cost = costs[rows, actions]  # (N, 3)
    latency = chosen_cost[:, 0]

    result = {
        "accuracy": float(accuracy[rows, actions].mean() * 100.0),
        "latency_mean_ms": float(latency.mean()),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "energy_j": float(chosen_cost[:, 1].mean()),
        # The 95th percentile of the per-query peak, not the max. The max
        # saturates: one delegated query pins every method to the same number
        # and the column stops discriminating. Note that training constrains
        # the mean, so this column is a stricter read than the constraint.
        "peak_memory_mib": float(np.percentile(chosen_cost[:, 2], 95)),
    }
    result["accuracy_per_joule"] = result["accuracy"] / max(result["energy_j"], 1e-6)
    if latency_slo_ms:
        result["slo_hit_rate"] = float((latency <= latency_slo_ms).mean() * 100.0)
    for i, name in enumerate(ACTION_NAMES):
        result[f"share_{name}"] = float((actions == i).mean() * 100.0)
    return result


COLUMNS = [
    "accuracy",
    "latency_p95_ms",
    "energy_j",
    "peak_memory_mib",
    "accuracy_per_joule",
    "share_delegate",
]


def format_table(rows: dict[str, dict]) -> str:
    """Render the results as a fixed-width table for the terminal."""
    header = ["method"] + COLUMNS
    widths = [max(len(header[0]), *(len(k) for k in rows))] + [
        max(len(c), 9) for c in COLUMNS
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for name, values in rows.items():
        cells = [name.ljust(widths[0])]
        for col, width in zip(COLUMNS, widths[1:]):
            cells.append(f"{values.get(col, float('nan')):.2f}".rjust(width))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def pareto_sweep(bundle, features, accuracy, costs, cfg) -> list[dict]:
    """Sweep the budget multiplier through one trained policy.

    This produces panel (a) of Figure 2: because the budget is an input to the
    policy, the whole curve comes from a single set of weights.
    """
    from quartet.config import budget_vector
    from quartet.policy import act

    points = []
    for scale in cfg.budget.sweep:
        budget = budget_vector(cfg, scale)
        actions, _ = act(bundle, features, budget)
        row = summarise(actions, accuracy, costs, latency_slo_ms=budget[0])
        row["budget_scale"] = scale
        points.append(row)
    return points


def run_all(bundle, features, accuracy, costs, cfg) -> dict:
    """Every table row: baselines, QUARTET at nominal budget, and the sweep."""
    from quartet.baselines import BASELINES
    from quartet.config import budget_vector
    from quartet.policy import act

    slo = cfg.budget.latency_ms
    rows: dict[str, dict] = {}
    for name, rule in BASELINES.items():
        rows[name] = summarise(rule(features, accuracy, costs), accuracy, costs, slo)

    actions, _ = act(bundle, features, budget_vector(cfg, 1.0))
    rows["quartet"] = summarise(actions, accuracy, costs, slo)

    return {
        "table": rows,
        "sweep": pareto_sweep(bundle, features, accuracy, costs, cfg),
        "num_queries": int(len(features)),
        "num_actions": NUM_ACTIONS,
    }


def save(results: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return path
