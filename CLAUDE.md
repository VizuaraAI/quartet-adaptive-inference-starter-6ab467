# CLAUDE.md

Instructions for the coding agent working on this repository with Siddharth.

## What this project is

QUARTET: a budget-conditioned inference policy for small language models. Per
query, it selects one of four actions (`ANSWER`, `REASON`, `RETRIEVE`,
`DELEGATE`) to maximise accuracy subject to latency, energy and memory
budgets. It is an eight-week research project aimed at a workshop paper. The
placeholder draft lives at `../paper/main.tex` and its section headings are
the real backlog.

The contribution is **not** "a router". Routers exist. It is (a) the union of
four action types under one gate, (b) energy and memory as first-class
objectives measured from hardware counters, and (c) the budget as a policy
input so one model covers the whole curve. When a change would weaken one of
those three, say so rather than making it.

## Layout

See the tree in `README.md`. The short version: `engine.py` executes actions,
`rollout.py` caches their outcomes, `policy.py` learns over that cache,
`evaluate.py` scores it, `cli.py` wires it together.

The rollout table is the hinge of the whole design. Collecting it costs GPU
hours; everything downstream reads it and runs on CPU in under a minute. Keep
that split. If you find yourself needing a GPU to test a training change, the
change is in the wrong module.

## Conventions

- **Lazy heavy imports.** `torch`, `transformers` and `datasets` are imported
  inside the functions that use them, never at module top level, so
  `import quartet` stays fast and works without model weights present. Keep
  doing this.
- **Action order is fixed**: answer, reason, retrieve, delegate. The policy
  head, the rollout JSON keys and the paper's table all index by it. Never
  reorder `Action`.
- **Cost vectors are always (latency_ms, energy_j, peak_memory_mib)**, in that
  order, everywhere.
- **Column names are shared** between `evaluate.COLUMNS`,
  `experiments/log_template.md` and Table 1 in the paper. Change one, change
  all three in the same commit.
- Comment the *why*, especially where a numerical choice is load-bearing.
  `policy.py` has two such places already; match that density, not more.
- No em dashes in prose or comments.

## Things that will bite you

- **A flat budget sweep means the policy is ignoring the budget.** Budgets are
  encoded as `log(budget / nominal)`, centred on zero. If you switch to raw
  magnitudes the network sees three near-constant large inputs and learns to
  ignore them. This has already happened once.
- **The hinge in the penalty is deliberate.** With the signed slack, spending
  under budget is rewarded and the policy collapses onto `ANSWER`.
- **One Lagrange multiplier per budget level.** A single pooled multiplier
  averages tight and loose budgets, satisfies neither, and the dual oscillates
  into a limit cycle.
- **Unbalanced budgets hide results.** If the latency cap binds far before the
  energy cap, the policy sits far under the energy budget and the energy axis
  of the results says nothing. Keep the three roughly equally tight.
- **`peak_memory_mib` is reported as a p95**, not a max: the max saturates
  because one delegated query pins every method to the same number. Training
  constrains the mean, which is a known looseness, flagged in the paper's
  Limitations.
- **Synthetic numbers are not results.** `synthetic.py` exists to exercise the
  plumbing. If a number came from it, it may not go in the paper. Say so
  explicitly whenever you report one.

## How to help week to week

Default loop: make the change, run `bash scripts/run_experiments.sh`, and
report the delta against the last logged run in `experiments/`. Do not report
a change as working without running it.

When adding an experiment, ask which section of `main.tex` it fills. If the
answer is "none", it is probably not worth the week.

Useful things to offer unprompted:

- Spotting when a new gate feature is not actually free, and pricing it.
- Noticing an action that is never selected at any budget, which usually means
  a cost is mismeasured rather than that the action is useless.
- Catching a baseline that has been given an unfair advantage or handicap. The
  accuracy-only baselines are *supposed* to ignore the budget; that is the
  comparison, not a bug.
- Keeping `references.bib` honest: open the arXiv page before adding an entry.
  Never write a citation from memory.

## Current state

The scaffold runs end to end on synthetic rollouts. Not yet done: the real
rollout collection, the `--limit` sanity pass on real data, energy calibration
on the target GPU, the depth-two (second decision) variant, and every ablation
in Section 5 of the paper.
