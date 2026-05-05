#!/usr/bin/env bash
# vLLM serve script for the trained LFM2.5-VL agent (S62-S65).
#
# Usage:
#   MODEL_PATH=/path/to/lfm2vl-sft-grpo-checkpoint  bash scripts/serve_vllm_lfm2.sh
#
# Requires:
#   - vllm >= 0.19  (for ToolParserManager.import_tool_parser)
#   - The custom parser at agent/lfm2_tool_parser.py
#
# After this server is up, drive it via:
#   from agent.lfm2_agent import run_lfm2_agent
#   result = run_lfm2_agent(case_id, precompute_root=..., vllm_url="http://localhost:8000/v1")

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./models/LFM2.5-VL-450M-sft-grpo}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: model directory not found: $MODEL_PATH" >&2
    echo "Set MODEL_PATH to the trained S62/S64/S65 checkpoint." >&2
    exit 1
fi

PARSER_PATH="$(cd "$(dirname "$0")/.." && pwd)/agent/lfm2_tool_parser.py"
if [[ ! -f "$PARSER_PATH" ]]; then
    echo "ERROR: parser plugin not found: $PARSER_PATH" >&2
    exit 1
fi

echo "Starting vLLM with LFM2 custom tool parser"
echo "  MODEL_PATH:    $MODEL_PATH"
echo "  PARSER_PLUGIN: $PARSER_PATH"
echo "  PORT:          $PORT"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo

exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-parser-plugin "$PARSER_PATH" \
    --tool-call-parser lfm2_pythonic
