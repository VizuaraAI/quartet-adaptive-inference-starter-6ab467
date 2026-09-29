"""Command line entry points.

    python -m quartet.cli rollout   --config configs/default.yaml
    python -m quartet.cli train     --config configs/default.yaml
    python -m quartet.cli evaluate  --config configs/default.yaml
    python -m quartet.cli ablate    --config configs/default.yaml
    python -m quartet.cli demo      --question "..."

`rollout` needs a GPU and takes hours. `train`, `evaluate` and `ablate` read
the rollout table and run on CPU in under a minute, which is what makes the
iteration loop for this project fast once the table exists.
"""

from __future__ import annotations

import argparse
import json
import os

from quartet.config import load_config


def _split(records, cfg):
    """Deterministic train/test split of the rollout table by hash of qid."""
    train, test = [], []
    for record in records:
        bucket = hash((record["qid"], cfg.data.seed)) % 10
        (test if bucket < 3 else train).append(record)
    return train, test


def cmd_rollout(args):
    from quartet.data import load_splits
    from quartet.engine import Engine
    from quartet.retrieval import build_retriever
    from quartet.rollout import collect

    cfg = load_config(args.config)
    splits = load_splits(cfg)
    examples = splits["train"] + splits["val"] + splits["test"]
    if args.limit:
        examples = examples[: args.limit]

    retriever = build_retriever(cfg).load_or_build()
    engine = Engine(cfg, retriever=retriever)
    print(f"rolling out {len(examples)} queries x 4 actions")
    path = collect(engine, examples, cfg.rollout_path)
    print(f"wrote {path}")


def cmd_synth(args):
    """Write a synthetic rollout table so the pipeline runs with no GPU."""
    from quartet.synthetic import write

    cfg = load_config(args.config)
    path = write(cfg.rollout_path, n=args.n, seed=cfg.data.seed)
    print(f"wrote {args.n} synthetic rollouts to {path}")
    print("these numbers describe the generator, not a language model; "
          "never report them")


def cmd_train(args):
    from quartet.policy import save, train
    from quartet.rollout import load_rollouts, to_arrays

    cfg = load_config(args.config)
    records = load_rollouts(cfg.rollout_path)
    train_records, _ = _split(records, cfg)
    print(f"training on {len(train_records)} of {len(records)} rolled-out queries")

    features, accuracy, costs = to_arrays(train_records)
    bundle = train(features, accuracy, costs, cfg)
    print("saved to", save(bundle, cfg.output_dir))


def cmd_evaluate(args):
    from quartet.evaluate import format_table, run_all, save
    from quartet.policy import load
    from quartet.rollout import load_rollouts, to_arrays

    cfg = load_config(args.config)
    records = load_rollouts(cfg.rollout_path)
    _, test_records = _split(records, cfg)
    features, accuracy, costs = to_arrays(test_records)

    bundle = load(cfg.output_dir, cfg.train.hidden_dim)
    results = run_all(bundle, features, accuracy, costs, cfg)
    print(f"\n{len(test_records)} held-out queries\n")
    print(format_table(results["table"]))
    print("\nbudget sweep (accuracy %, energy J):")
    for point in results["sweep"]:
        print(
            f"  x{point['budget_scale']:<5} "
            f"acc {point['accuracy']:6.2f}   energy {point['energy_j']:7.2f}"
        )
    print("\nwrote", save(results, cfg.output_dir))


def cmd_ablate(args):
    """Action-set ablation: the experiment the paper's claim rests on."""
    import numpy as np

    from quartet.actions import Action
    from quartet.evaluate import summarise
    from quartet.policy import act, train
    from quartet.rollout import load_rollouts, to_arrays

    cfg = load_config(args.config)
    records = load_rollouts(cfg.rollout_path)
    train_records, test_records = _split(records, cfg)
    f_tr, a_tr, c_tr = to_arrays(train_records)
    f_te, a_te, c_te = to_arrays(test_records)

    subsets = {
        "answer+delegate": [Action.ANSWER, Action.DELEGATE],
        "answer+reason+delegate": [Action.ANSWER, Action.REASON, Action.DELEGATE],
        "answer+retrieve+delegate": [Action.ANSWER, Action.RETRIEVE, Action.DELEGATE],
        "all four": list(Action),
    }
    for name, subset in subsets.items():
        keep = [int(a) for a in subset]
        # Masking accuracy to -1 makes an excluded action never worth taking,
        # which restricts the action set without changing the network shape.
        masked = np.full_like(a_tr, -1.0)
        masked[:, keep] = a_tr[:, keep]
        bundle = train(f_tr, masked, c_tr, cfg, verbose=False)

        actions, probs = act(bundle, f_te, [cfg.budget.latency_ms,
                                            cfg.budget.energy_j,
                                            cfg.budget.peak_memory_mib])
        actions = np.asarray(keep)[probs[:, keep].argmax(axis=1)]
        row = summarise(actions, a_te, c_te, cfg.budget.latency_ms)
        print(
            f"{name:28s} acc {row['accuracy']:6.2f}   "
            f"energy {row['energy_j']:7.2f}   "
            f"p95 {row['latency_p95_ms']:8.1f}"
        )


def cmd_demo(args):
    """Run one question end to end and show what the policy decided."""
    from quartet.actions import ACTION_NAMES
    from quartet.data import Example
    from quartet.engine import Engine
    from quartet.features import extract
    from quartet.policy import act, load
    from quartet.retrieval import build_retriever

    cfg = load_config(args.config)
    example = Example(qid="demo", question=args.question, answers=[],
                      dataset="demo", task="qa")
    engine = Engine(cfg, retriever=build_retriever(cfg).load_or_build())

    draft = engine.draft(example)
    budget = [cfg.budget.latency_ms, cfg.budget.energy_j, cfg.budget.peak_memory_mib]
    features = extract(example, draft, budget=None)

    if os.path.exists(os.path.join(cfg.output_dir, "policy_meta.json")):
        actions, probs = act(load(cfg.output_dir, cfg.train.hidden_dim),
                             [features], budget)
        action = int(actions[0])
        print("action scores:", json.dumps(
            {n: round(float(p), 3) for n, p in zip(ACTION_NAMES, probs[0])}))
    else:
        action = 0
        print("no trained policy found; falling back to ANSWER")

    result = engine.run_action(action, example, draft)
    print(f"\naction    : {ACTION_NAMES[action]}")
    print(f"answer    : {result.prediction}")
    print(f"latency   : {result.latency_ms:.1f} ms")
    print(f"energy    : {result.energy_j:.2f} J")
    print(f"peak mem  : {result.peak_memory_mib:.0f} MiB")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="quartet")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic rollout table (no GPU)")
    p.add_argument("-n", type=int, default=4000)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("rollout", help="execute all four actions on every query")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_rollout)

    sub.add_parser("train", help="fit the policy").set_defaults(func=cmd_train)
    sub.add_parser("evaluate", help="score every baseline").set_defaults(func=cmd_evaluate)
    sub.add_parser("ablate", help="action-set ablation").set_defaults(func=cmd_ablate)

    p = sub.add_parser("demo", help="run one question end to end")
    p.add_argument("--question", required=True)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
