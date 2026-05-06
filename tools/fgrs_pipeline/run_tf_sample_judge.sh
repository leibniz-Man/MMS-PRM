#!/usr/bin/env bash
set -euo pipefail

# Set OpenAI proxy environment variables
export OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
export OPENAI_API_KEY="sk-pfQFCXggVvHdL3R1O3k5chxbva6l7Ai4v5SkkPdKYDw2t4DY"

# Resolve repository root relative to this script, then run the command
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# cd "$REPO_ROOT"

python tools/fgrs_pipeline/tf_sample_judge.py \
  --path datasets/dpo_data.jsonl \
  --image-root datasets/image_root \
  --limit 400 \
  --output tools/fgrs_pipeline/output/dpo_step_judgments.jsonl \
