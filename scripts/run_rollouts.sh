#!/usr/bin/env bash
# The expensive step: execute all four actions on every query.
# Needs a GPU. Budget several hours for the full pool; it resumes if killed,
# so run it with a small --limit first and confirm the costs look sane.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"
LIMIT="${2:-}"

if [[ -n "$LIMIT" ]]; then
  python -m quartet.cli --config "$CONFIG" rollout --limit "$LIMIT"
else
  python -m quartet.cli --config "$CONFIG" rollout
fi

echo
echo "Sanity check before you trust this table:"
echo "  * ANSWER latency and energy should be ~0 (it reuses the gating pass)"
echo "  * DELEGATE should be the most expensive action on all three axes"
echo "  * no action should be correct on every query, or routing is pointless"
