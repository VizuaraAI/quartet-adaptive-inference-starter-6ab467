# QUARTET

**Budget-conditioned action selection for small language model inference.**

Siddharth, Vizuara SLM Bootcamp. Eight weeks.

Most inference systems optimise one thing: accuracy. The handful that think
about cost squash it into a single scalar, usually dollars or tokens. A real
deployment does not experience cost that way. It has a latency target, a power
envelope and a memory ceiling at the same time, and all three move when the
hardware moves.

QUARTET treats each query as a decision. Given the query and a budget vector,
pick one of four actions:

| Action | What it does | Cost |
| --- | --- | --- |
| `ANSWER` | emit the draft the small model already produced while gating | free |
| `REASON` | sample k chains of thought, take the majority answer | slow, power-hungry |
| `RETRIEVE` | BM25 evidence prepended, small model answers | cheap compute, heavy context |
| `DELEGATE` | escalate to a model ~10x larger | expensive on every axis |

Two things make this different from the routing and cascade literature it
builds on. First, the action space is the **union** of the decisions that
separate papers have each studied alone (escalate, think longer, look it up).
Second, the budget vector is an **input to the policy**, so one trained set of
weights covers the whole trade-off curve instead of one operating point.

## Setup

```bash
git clone <your-repo-url> quartet && cd quartet
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

Python 3.10 or newer. Everything except the rollout step runs on CPU.

## The first three things to run

**1. Prove the plumbing works (5 minutes, no GPU, no downloads).**

```bash
bash scripts/quickstart.sh
```

This writes a synthetic rollout table, trains the policy on it, and prints the
full results table plus the budget sweep. You should see accuracy and energy
both rise monotonically as the budget multiplier grows, from roughly the
always-answer corner to roughly the always-delegate corner. That monotone
sweep is the single best smoke test in this repo: if it goes flat, the policy
has stopped reading the budget.

The numbers it prints come from the generator in `src/quartet/synthetic.py`.
They describe the generator. They are not results, and none of them belongs in
the paper.

**2. Measure your own hardware (10 minutes, needs the GPU you will report on).**

```bash
bash scripts/calibrate_energy.sh
```

Prints idle draw, joules per generated token and peak memory. Put those
numbers into `budget:` in `configs/default.yaml` and into the setup section of
the paper. Do this **before** the real rollout: budgets guessed rather than
measured are the fastest way to a degenerate policy, either one no action can
violate or one no action can satisfy.

**3. Collect a real rollout table (hours, needs a GPU).**

```bash
bash scripts/build_index.sh              # once, builds the BM25 index
bash scripts/run_rollouts.sh configs/default.yaml 200   # start small
```

Check the 200-query table by hand before launching the full run. `ANSWER`
should cost about nothing, `DELEGATE` should be the most expensive action on
all three axes, and no action should be correct on every query. If one action
wins everywhere, there is nothing to route and you need a harder dataset mix.
The full run resumes if interrupted, so killing it is safe.

Then iterate:

```bash
bash scripts/run_experiments.sh          # train + evaluate + ablate, under a minute
```

## Layout

```
configs/default.yaml      every knob, with the ones worth changing marked
scripts/                  quickstart, index build, calibration, rollouts, experiments
experiments/              your lab notebook; rows map to the paper's Table 1
src/quartet/
  actions.py              the four actions and the cost record they produce
  data.py                 PopQA / GSM8K / HotpotQA loading and answer scoring
  retrieval.py            BM25 index over Wikipedia passages
  meters.py               latency, NVML/RAPL energy, peak memory
  engine.py               the models, and the four actions built on them
  features.py             gate features, and the rule about what may be one
  rollout.py              run all four actions on every query, cache to JSONL
  policy.py               the budget-conditioned policy and its dual trainer
  baselines.py            fixed actions, cascade, routers, oracle
  evaluate.py             scoring, the results table, the Pareto sweep
  synthetic.py            fake rollouts so the pipeline runs before you have real ones
  cli.py                  synth / rollout / train / evaluate / ablate / demo
```

## How it trains

The rollout table holds the outcome of **every** action on **every** training
query, so expected accuracy and expected cost under a stochastic policy are
both closed form. That makes the constrained objective

```
maximise  E[accuracy]  subject to  E[cost_j] <= budget_j   for j in latency, energy, memory
```

differentiable end to end. No sampling, no REINFORCE, no variance problem. The
trainer is projected dual ascent on the Lagrange multipliers and takes well
under a minute on CPU. Two details in `policy.py` are load-bearing, and both
are commented where they live: the penalty uses a hinge rather than the signed
slack, and there is one multiplier per budget level rather than one pooled
multiplier.

## Two rules worth keeping

**A gate feature has to be free.** Every feature comes out of the single
prefill that also produces the `ANSWER` draft. Semantic entropy predicts
correctness better than anything else available, but it costs several extra
decodes, which is more than the cheapest action costs. So it lives inside the
`REASON` action, not in the gate. When you add a feature, say out loud what it
costs, and charge it.

**Log the run before you change anything.** `experiments/log_template.md` has
one row per run and its columns are the columns of Table 1 in the paper. An
unlogged run is a run you will repeat.

## Where this is going

`../paper/main.tex` is a placeholder draft of the paper this repo is meant to
produce: real structure, real citations, placeholder numbers. Every experiment
here fills a specific hole in it. The target is a workshop submission; see the
kit's roadmap and venue list.
