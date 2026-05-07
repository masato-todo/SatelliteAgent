#!/usr/bin/env bash
# scripts/download_models.sh — pull all model weights from HuggingFace Hub.
#
# Total: ~2.8 GB
#   wildfire-staging/base    (LiquidAI/LFM2.5-VL-450M, 900 MB)  ← shared base
#   wildfire-staging/adapter (YujiYamaguchi/lfm2-5-vl-450m-wildfire, 100 MB)
#   wildfire-precursor-staging/adapter
#                            (YujiYamaguchi/lfm2-5-vl-450m-wildfire-precursor-pair14_7, 100 MB)
#                            base is symlinked from wildfire-staging/base
#                            so it isn't downloaded twice.
#   sft-grpo                 (todo1111/LFM2.5-VL-450M-sft-grpo-S64, 900 MB)
#
# After this, docker-compose.yaml's default mount paths just work:
#   WILDFIRE_MODEL_DIR=./models/wildfire-staging
#   WILDFIRE_PRECURSOR_ADAPTER_DIR=./models/wildfire-precursor-staging/adapter
#   (WILDFIRE_PRECURSOR_BASE_DIR defaults to wildfire-staging/base, since
#    both LoRAs are trained against the same LFM2.5-VL-450M.)
#   LFM2_AGENT_MODEL_DIR=./models/sft-grpo
#
# Skip this if you already have weights elsewhere — set the corresponding
# *_MODEL_DIR vars in .env to point at them.
#
# Override the default repo IDs with these env vars if needed:
#   WILDFIRE_BASE_REPO              (default LiquidAI/LFM2.5-VL-450M)
#   WILDFIRE_ADAPTER_REPO           (default YujiYamaguchi/lfm2-5-vl-450m-wildfire)
#   WILDFIRE_PRECURSOR_REPO         (default YujiYamaguchi/lfm2-5-vl-450m-wildfire-precursor-pair14_7)
#   LFM2_AGENT_REPO                 (default todo1111/LFM2.5-VL-450M-sft-grpo-S64)
#
set -euo pipefail

cd "$(dirname "$0")/.."

WILDFIRE_BASE_REPO="${WILDFIRE_BASE_REPO:-LiquidAI/LFM2.5-VL-450M}"
WILDFIRE_ADAPTER_REPO="${WILDFIRE_ADAPTER_REPO:-YujiYamaguchi/lfm2-5-vl-450m-wildfire}"
WILDFIRE_PRECURSOR_REPO="${WILDFIRE_PRECURSOR_REPO:-YujiYamaguchi/lfm2-5-vl-450m-wildfire-precursor-pair14_7}"
LFM2_AGENT_REPO="${LFM2_AGENT_REPO:-todo1111/LFM2.5-VL-450M-sft-grpo-S64}"

HF="${HF:-uv run hf}"

mkdir -p models/wildfire-staging \
         models/wildfire-precursor-staging \
         models/sft-grpo

echo "[1/4] $WILDFIRE_BASE_REPO  →  models/wildfire-staging/base"
$HF download "$WILDFIRE_BASE_REPO" \
    --local-dir models/wildfire-staging/base

echo "[2/4] $WILDFIRE_ADAPTER_REPO  →  models/wildfire-staging/adapter"
$HF download "$WILDFIRE_ADAPTER_REPO" \
    --local-dir models/wildfire-staging/adapter

echo "[3/4] $WILDFIRE_PRECURSOR_REPO  →  models/wildfire-precursor-staging/adapter"
$HF download "$WILDFIRE_PRECURSOR_REPO" \
    --local-dir models/wildfire-precursor-staging/adapter
# Note: the precursor LoRA is trained against the same LFM2.5-VL-450M as
# the wildfire LoRA. docker-compose.yaml mounts wildfire-staging/base
# directly into the precursor container, so the base is only fetched
# once (in step 1).

echo "[4/4] $LFM2_AGENT_REPO  →  models/sft-grpo"
$HF download "$LFM2_AGENT_REPO" \
    --local-dir models/sft-grpo

cat <<'EOF'

Models downloaded. Make sure these are set in .env:

    WILDFIRE_MODEL_DIR=./models/wildfire-staging
    WILDFIRE_PRECURSOR_MODEL_DIR=./models/wildfire-precursor-staging
    LFM2_AGENT_MODEL_DIR=./models/sft-grpo

then bring up the GPU services:

    docker compose up -d
EOF
