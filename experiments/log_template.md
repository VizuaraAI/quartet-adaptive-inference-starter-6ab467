# Experiment log

One block per run. The columns in the results table below are exactly the
columns of Table 1 in `../../paper/main.tex`, so writing up is copying rather
than reconciling. Fill the block in **before** you start the next run.

Copy the template, do not edit it in place.

---

## Run NNN: <one-line description>

- **Date:** YYYY-MM-DD
- **Commit:** `<git rev-parse --short HEAD>`
- **Config:** `configs/default.yaml` (note any overrides here)
- **Rollout table:** `artifacts/rollouts.jsonl`, N queries, real / synthetic
- **Hardware:** GPU model, driver, energy source (nvml / rapl / fallback)
- **Budget:** latency __ ms, energy __ J, memory __ MiB

### Question

What am I trying to find out, in one sentence? If this is blank, do not run it.

### Prediction

What do I expect to happen, and what would change my mind? Write this before
looking at the output. A run that cannot surprise you is not an experiment.

### Results

| Method | Accuracy (%) | Latency p95 (ms) | Energy (J/query) | Peak mem. (MiB) | Acc./energy (%/J) | Delegation rate (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Answer now (small) | | | | | | 0.0 |
| Reason (k=5) | | | | | | 0.0 |
| Retrieve (top-3) | | | | | | 0.0 |
| Delegate (large) | | | | | | 100.0 |
| Random action | | | | | | |
| Confidence cascade | | | | | | |
| Two-action router | | | | | | |
| Retrieval-complexity router | | | | | | |
| **QUARTET** | | | | | | |
| Oracle | | | | | | |

### Budget sweep

Feeds panel (a) of Figure 2. The sweep should be monotone in both columns; if
it is flat, the policy has stopped reading the budget.

| Multiplier | Accuracy (%) | Energy (J) | Latency p95 (ms) | Share answer / reason / retrieve / delegate |
| --- | --- | --- | --- | --- |
| x0.25 | | | | |
| x0.5 | | | | |
| x0.75 | | | | |
| x1.0 | | | | |
| x1.5 | | | | |
| x2.0 | | | | |
| x4.0 | | | | |

### Action-set ablation

Feeds Section 5. This is the table the paper's central claim rests on.

| Action set | Accuracy (%) | Energy (J) | Latency p95 (ms) |
| --- | --- | --- | --- |
| answer + delegate | | | |
| answer + reason + delegate | | | |
| answer + retrieve + delegate | | | |
| all four | | | |

### What happened

Did the prediction hold? Where it did not, what is the most likely
explanation, and what is the cheapest experiment that would distinguish
between the candidates?

### Sanity checks

- [ ] Budget sweep is monotone in accuracy and in energy
- [ ] No action is selected 0% of the time at *every* budget
- [ ] Energy source is nvml or rapl, not fallback (or the run is labelled estimated)
- [ ] Numbers are from a real rollout table, not `synthetic.py`
- [ ] Oracle is above every learned method (if not, something is mislabelled)

### Next

One concrete action. Not a list.

---
