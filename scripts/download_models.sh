#!/usr/bin/env bash
# scripts/download_models.sh — pull all model weights from HuggingFace Hub.
#
# Total: ~2.7 GB
#   wildfire-staging/base    (LiquidAI/LFM2.5-VL-450M, 900 MB)
#   wildfire-staging/adapter (YujiYamaguchi/lfm2-5-vl-450m-wildfire, 100 MB)
#   sft-grpo                 (todo1111/LFM2.5-VL-450M-sft-grpo-S64, 900 MB)
#
# After this, docker-compose.yaml's default mount paths just work:
#   WILDFIRE_MODEL_DIR=./models/wildfire-staging
#   LFM2_AGENT_MODEL_DIR=./models/sft-grpo
#
# Skip this if you already have weights elsewhere — set
# WILDFIRE_MODEL_DIR / LFM2_AGENT_MODEL_DIR in .env to point at them.
#
# Override the default repo IDs with these env vars if needed:
#   WILDFIRE_BASE_REPO     (default LiquidAI/LFM2.5-VL-450M)
#   WILDFIRE_ADAPTER_REPO  (default YujiYamaguchi/lfm2-5-vl-450m-wildfire)
#   LFM2_AGENT_REPO        (default todo1111/LFM2.5-VL-450M-sft-grpo-S64)
#
set -euo pipefail

cd "$(dirname "$0")/.."

WILDFIRE_BASE_REPO="${WILDFIRE_BASE_REPO:-LiquidAI/LFM2.5-VL-450M}"
WILDFIRE_ADAPTER_REPO="${WILDFIRE_ADAPTER_REPO:-YujiYamaguchi/lfm2-5-vl-450m-wildfire}"
LFM2_AGENT_REPO="${LFM2_AGENT_REPO:-todo1111/LFM2.5-VL-450M-sft-grpo-S64}"

HF="${HF:-uv run hf}"

mkdir -p models/wildfire-staging models/sft-grpo

echo "[1/3] $WILDFIRE_BASE_REPO  →  models/wildfire-staging/base"
$HF download "$WILDFIRE_BASE_REPO" \
    --local-dir models/wildfire-staging/base

echo "[2/3] $WILDFIRE_ADAPTER_REPO  →  models/wildfire-staging/adapter"
$HF download "$WILDFIRE_ADAPTER_REPO" \
    --local-dir models/wildfire-staging/adapter

echo "[3/3] $LFM2_AGENT_REPO  →  models/sft-grpo"
$HF download "$LFM2_AGENT_REPO" \
    --local-dir models/sft-grpo

cat <<'EOF'

Models downloaded. Make sure these are set in .env:

    WILDFIRE_MODEL_DIR=./models/wildfire-staging
    LFM2_AGENT_MODEL_DIR=./models/sft-grpo

then bring up the GPU services:

    docker compose up -d
EOF
