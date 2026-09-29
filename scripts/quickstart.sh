#!/usr/bin/env bash
# The five-minute sanity check. No GPU, no model downloads.
# Builds a synthetic rollout table, trains the policy on it, and prints the
# full results table plus the budget sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"

python -m quartet.cli --config "$CONFIG" synth -n 4000
python -m quartet.cli --config "$CONFIG" train
python -m quartet.cli --config "$CONFIG" evaluate

echo
echo "Reminder: these numbers come from the synthetic generator in"
echo "src/quartet/synthetic.py. They tell you the plumbing works. They are"
echo "not results. Run scripts/run_rollouts.sh for real ones."
