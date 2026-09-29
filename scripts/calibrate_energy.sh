#!/usr/bin/env bash
# Measure this machine's idle draw and joules per generated token.
# Run this BEFORE setting the budgets in configs/default.yaml, and paste the
# numbers into meters.FALLBACK_J_PER_TOKEN and into the paper's setup section.
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import time

from quartet.config import load_config
from quartet.data import Example
from quartet.engine import Engine
from quartet.meters import measure

cfg = load_config("configs/default.yaml")

# Idle draw first: everything below is only meaningful relative to this.
with measure(cfg.measure_energy) as idle:
    time.sleep(5.0)
print(f"idle           : {idle.energy_j / 5.0:7.2f} W   (source: {idle.source})")
if idle.source == "fallback":
    print("  no power counters found; energy numbers will be analytic estimates")

engine = Engine(cfg)
example = Example(qid="cal", question="Name three prime numbers.", answers=[],
                  dataset="cal", task="qa")

for label in ("warmup", "measured"):
    draft = engine.draft(example)
    if label == "measured":
        tokens = max(draft.completion_tokens, 1)
        marginal = max(draft.energy_j - idle.energy_j / 5.0 * draft.latency_ms / 1000, 0)
        print(f"gate latency   : {draft.latency_ms:7.1f} ms")
        print(f"gate energy    : {draft.energy_j:7.2f} J  ({tokens} tokens)")
        print(f"J per token    : {marginal / tokens:7.4f}  (idle-corrected)")
        print(f"peak memory    : {draft.peak_memory_mib:7.0f} MiB")
PY
