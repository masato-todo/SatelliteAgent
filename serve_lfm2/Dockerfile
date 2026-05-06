# vLLM image patched for LFM2.5-VL serving on aarch64 + cu13.
#
# Why each layer is needed:
# - cu130-nightly base: vLLM 0.16+ wheels, CUDA 13.0 toolchain, Blackwell
#   PTX (covers GB10's compute capability 12.x).
# - transformers 5.5.0 pin: 5.x is when Lfm2Vl entered transformers, but
#   newer releases shuffle private symbols vLLM's adapter relies on.
#   5.5.0 is the version author of LFM2.5-VL confirmed working.
# - sed patch on lfm2.py: vLLM's bundled adapter reads
#   `config.block_ff_dim` (the old transformers attribute name).
#   transformers 5.x renamed it to `intermediate_size`, so the launch
#   would crash with AttributeError without this rename.
# - lfm2_tool_parser.py copy: Python plugin loaded via
#   `--tool-parser-plugin /agent/lfm2_tool_parser.py` so vLLM can decode
#   the LFM2.5-VL-450M-sft-grpo Python-style tool calls
#   (`[fn(arg=...)]`) into OpenAI tool_calls JSON.
#
# Build:
#   docker build -t satelliteagent/lfm2-agent:cu130 -f serve_lfm2/Dockerfile .
#
# Run (bind a checkpoint at /model):
#   docker run --gpus all -p 8086:8000 \
#     -v /path/to/LFM2.5-VL-450M-sft-grpo:/model:ro \
#     satelliteagent/lfm2-agent:cu130 \
#     --model /model --served-model-name LFM2.5-VL-450M-sft-grpo \
#     --port 8000 --host 0.0.0.0 \
#     --gpu-memory-utilization 0.5 --max-model-len 4096 \
#     --enforce-eager --trust-remote-code \
#     --enable-auto-tool-choice \
#     --tool-parser-plugin /agent/lfm2_tool_parser.py \
#     --tool-call-parser lfm2_pythonic
#
# Or run via the repo root docker-compose.yaml.

FROM vllm/vllm-openai:cu130-nightly

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers==5.5.0" \
    "tokenizers==0.22.2" \
    "huggingface_hub==1.9.0" \
    "mistral_common==1.11.0" \
    "safetensors==0.6.2"

RUN sed -i 's/config\.block_ff_dim/config.intermediate_size/g' \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/lfm2.py

COPY agent/lfm2_tool_parser.py /agent/lfm2_tool_parser.py

# Entrypoint (`vllm serve`) is inherited from the base image.
EXPOSE 8000
