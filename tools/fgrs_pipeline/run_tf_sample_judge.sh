#!/usr/bin/env bash
set -euo pipefail

# Expect secrets from environment, do not hardcode keys in script.
# Example:
#   export OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
#   export OPENAI_API_KEY="***"
# Optional explicit endpoint/model can also be passed through CLI args below.

# Resolve repository root relative to this script, then run the command
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# cd "$REPO_ROOT"

python tools/fgrs_pipeline/tf_sample_judge.py \
  --path datasets/dpo_data.jsonl \
  --image-root datasets/image_root \
  --limit 400 \
  --model Qwen2.5-VL-32B-Instruct \
  --criteria-pairs tools/fgrs_pipeline/output/reward_hierarchy_named_pairs.txt \
  --min-criteria-per-step 3 \
  --max-criteria-per-step 5 \
  --parent-distance-threshold 0.5 \
  --bge-model BAAI/bge-en-icl \
  --output tools/fgrs_pipeline/output/dpo_step_judgments.jsonl
