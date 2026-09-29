#!/usr/bin/env bash
# Train, evaluate and ablate against an existing rollout table.
# CPU only, under a minute. This is the loop you will run dozens of times.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"

python -m quartet.cli --config "$CONFIG" train
python -m quartet.cli --config "$CONFIG" evaluate
echo
echo "=== action-set ablation ==="
python -m quartet.cli --config "$CONFIG" ablate
echo
echo "Copy the numbers into experiments/log_template.md before you change"
echo "anything. An unlogged run is a run you will have to repeat."
