#!/usr/bin/env bash
# Build the BM25 passage index once. Takes a few minutes and a few GB of disk.
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
from quartet.config import load_config
from quartet.retrieval import build_retriever

cfg = load_config("configs/default.yaml")
retriever = build_retriever(cfg).build()
hits = retriever.search("Who composed the Brandenburg Concertos?", cfg.retrieval.top_k)
print(f"index ready at {cfg.retrieval.index_dir}; {len(hits)} passages returned")
for i, passage in enumerate(hits, 1):
    print(f"  [{i}] {passage[:120]}...")
PY
