"""The budget-conditioned policy and its constrained trainer.

Because the rollout table contains the outcome of *every* action on *every*
training query, the expected accuracy and expected cost under a stochastic
policy are both closed-form:

    E[accuracy] = sum_a pi(a|x) * y(x, a)
    E[cost_j]   = sum_a pi(a|x) * c_j(x, a)

so the Lagrangian is differentiable end to end and there is no sampling, no
REINFORCE estimator and no variance problem. The whole trainer is a few dozen
lines, and it runs on CPU in under a minute.

    maximise  E[accuracy] - sum_j lambda_j (E[cost_j] - budget_j)
    subject to lambda_j >= 0, updated by projected gradient ascent.
"""

from __future__ import annotations

import json
import os

from quartet.actions import ACTION_NAMES, NUM_ACTIONS

NUM_COSTS = 3


def build_policy(input_dim: int, hidden_dim: int = 128):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, NUM_ACTIONS),
    )


class Standardizer:
    """Z-scores features using training statistics only."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        import numpy as np

        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)
        return self

    def __call__(self, x):
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state):
        import numpy as np

        return cls(np.asarray(state["mean"]), np.asarray(state["std"]))


def _budget_features(budgets, base):
    """Encode the budget as log(budget / nominal budget).

    Encoding the raw magnitudes instead (log1p of 12000 MiB, say) gives the
    network three large, nearly constant inputs whose variation across the
    sweep is tiny next to their mean, and it simply learns to ignore them: the
    policy then returns the same action at every budget and the sweep is flat.
    A ratio is centred on zero, is the same size on all three axes, and is
    invariant to the hardware the nominal budget came from.
    """
    import numpy as np

    return np.log(
        np.maximum(budgets, 1e-6) / np.maximum(np.asarray(base, dtype=np.float32), 1e-6)
    )


def sample_budgets(base, multipliers, n, rng):
    """Draw one budget vector per training example from the sweep.

    Returns (budgets, picks); `picks` says which sweep level each example was
    drawn from, which the trainer needs in order to keep a separate Lagrange
    multiplier per level.
    """
    import numpy as np

    picks = rng.choice(len(multipliers), size=n)
    scales = np.asarray(multipliers, dtype=np.float32)[picks][:, None]
    return np.asarray(base, dtype=np.float32)[None, :] * scales, picks


def train(features, accuracy, costs, cfg, verbose=True):
    """Fit the budget-conditioned policy on the rollout table.

    features : (N, F)     gate features, budget columns NOT yet appended
    accuracy : (N, A)     correctness of each action
    costs    : (N, A, 3)  latency ms, energy J, peak memory MiB
    """
    import numpy as np
    import torch

    tc = cfg.train
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)

    base = np.asarray(
        [cfg.budget.latency_ms, cfg.budget.energy_j, cfg.budget.peak_memory_mib],
        dtype=np.float32,
    )
    # Costs are normalised by the nominal budget so the three constraints are
    # on a common scale and one shared dual learning rate works for all of
    # them. Without this, memory (thousands of MiB) swamps energy (tens of J).
    scale = np.maximum(base, 1e-6)

    scaler = Standardizer().fit(features)
    x_base = torch.tensor(scaler(features), dtype=torch.float32)
    y = torch.tensor(accuracy, dtype=torch.float32)
    c = torch.tensor(costs / scale[None, None, :], dtype=torch.float32)

    n = x_base.shape[0]
    num_levels = len(cfg.budget.sweep)
    model = build_policy(x_base.shape[1] + NUM_COSTS, tc.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)

    # One multiplier per (budget level, objective). A single pooled multiplier
    # averages a tight budget together with a loose one, and since neither is
    # ever satisfied at the pooled value the dual oscillates and the policy
    # thrashes between the cheapest and the dearest action.
    lam = torch.full((num_levels, NUM_COSTS), float(tc.lambda_init))

    history = []
    for epoch in range(tc.epochs):
        order = rng.permutation(n)
        epoch_acc, batches = 0.0, 0
        slack_sum = torch.zeros(num_levels, NUM_COSTS)
        slack_count = torch.zeros(num_levels, 1)
        ratio_sum = torch.zeros(NUM_COSTS)

        for start in range(0, n, tc.batch_size):
            idx = order[start : start + tc.batch_size]
            budgets, picks = sample_budgets(base, cfg.budget.sweep, len(idx), rng)
            bx = torch.tensor(_budget_features(budgets, base), dtype=torch.float32)
            pick_t = torch.tensor(picks, dtype=torch.long)
            # The constraint is "cost <= budget", and costs are expressed in
            # units of the nominal budget, so the target is the multiplier.
            target = torch.tensor(budgets / scale, dtype=torch.float32)

            x = torch.cat([x_base[idx], bx], dim=1)
            probs = torch.softmax(model(x), dim=1)

            exp_acc = (probs * y[idx]).sum(dim=1).mean()
            exp_cost = (probs.unsqueeze(-1) * c[idx]).sum(dim=1)
            slack = exp_cost - target
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()

            # Hinge, not the raw signed slack. Spending less than the budget
            # earns nothing, so once a query is inside its budget the only
            # remaining gradient is toward accuracy. With the signed slack the
            # policy is instead paid to keep shrinking cost and collapses onto
            # the cheapest action.
            penalty = (lam[pick_t] * torch.relu(slack)).sum(dim=1).mean()

            loss = -exp_acc + penalty - tc.entropy_bonus * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                epoch_acc += float(exp_acc.detach())
                ratio_sum += (exp_cost / target.clamp_min(1e-6)).mean(dim=0).detach()
                slack_sum.index_add_(0, pick_t, slack.detach())
                slack_count.index_add_(
                    0, pick_t, torch.ones(len(idx), 1)
                )
            batches += 1

        # Projected dual ascent, once per epoch on each level's mean slack.
        with torch.no_grad():
            mean_slack = slack_sum / slack_count.clamp_min(1.0)
            lam = (lam + tc.dual_lr * mean_slack).clamp(0.0, tc.lambda_max)

        history.append(
            {
                "epoch": epoch,
                "accuracy": epoch_acc / batches,
                "cost_ratio": (ratio_sum / batches).tolist(),
                "lambda_mean": lam.mean(dim=0).tolist(),
            }
        )
        if verbose and (epoch % 10 == 0 or epoch == tc.epochs - 1):
            h = history[-1]
            ratios = ", ".join(f"{v:.2f}" for v in h["cost_ratio"])
            lams = ", ".join(f"{v:.2f}" for v in h["lambda_mean"])
            print(
                f"epoch {epoch:3d}  acc {h['accuracy']:.4f}  "
                f"cost/budget [{ratios}]  mean lambda [{lams}]"
            )

    return {"model": model, "scaler": scaler, "base_budget": base, "history": history}


def act(bundle, features, budget):
    """Greedy action for a batch of queries at a given budget.

    features : (N, F) raw gate features
    budget   : list of 3 floats
    Returns (actions, probabilities).
    """
    import numpy as np
    import torch

    model, scaler = bundle["model"], bundle["scaler"]
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    budgets = np.repeat(np.asarray(budget, dtype=np.float32)[None, :], len(x), axis=0)
    inp = torch.tensor(
        np.concatenate(
            [scaler(x), _budget_features(budgets, bundle["base_budget"])], axis=1
        ),
        dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1).numpy()
    return probs.argmax(axis=1), probs


def save(bundle, directory):
    import torch

    os.makedirs(directory, exist_ok=True)
    torch.save(bundle["model"].state_dict(), os.path.join(directory, "policy.pt"))
    with open(os.path.join(directory, "policy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "scaler": bundle["scaler"].state_dict(),
                "base_budget": list(map(float, bundle["base_budget"])),
                "actions": ACTION_NAMES,
                "history": bundle["history"],
            },
            f,
            indent=2,
        )
    return directory


def load(directory, hidden_dim=128):
    import numpy as np
    import torch

    with open(os.path.join(directory, "policy_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = Standardizer.from_state(meta["scaler"])
    model = build_policy(len(scaler.mean) + NUM_COSTS, hidden_dim)
    model.load_state_dict(torch.load(os.path.join(directory, "policy.pt")))
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "base_budget": np.asarray(meta["base_budget"], dtype=np.float32),
        "history": meta["history"],
    }
