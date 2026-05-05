# LFM2.5-VL SFT/GRPO model — integration guide

The Kaggle notebooks **S62-S65** train and evaluate a multi-turn agent based on
LFM2.5-VL-450M. This doc describes how to serve the trained checkpoint with
vLLM and call it from the SatelliteAgent app.

## Best result so far

| Notebook | Stage | Accuracy (n=96 test split) |
|---|---|---|
| S46-S55 | single-turn SFT (no agent) | 70.8% (S54, image+spectral+mixup) |
| S62 | multi-turn SFT (1 epoch, with images) | 51.0% |
| S63 | multi-turn SFT (3 epoch, **no images at inference**) | **74.0%** |
| S64 | S63 + GRPO 30 step | **75.0%** |
| S65 | S63 + GRPO 100 step | TBD (in flight) |

**Key insight (S63)**: when images are passed to the model at inference time,
its vision attention dominates and it skips `compute_index_delta` (the spectral
data-gathering tool it was trained to call first). Removing images from the
user prompt unblocks the SFT-trained tool flow → +24pp accuracy.

## What's where

```
agent/
├── lfm2_agent.py          # multi-turn agent loop (THE inference SSOT)
│                            - SYS_PROMPT, TOOLS (4 tools, matches SFT trajectory)
│                            - run_lfm2_agent(...) entry point
│                            - execute_tool() reads precompute YAML offline
└── lfm2_tool_parser.py    # vLLM custom parser plugin
                              (LFM2 emits <|tool_call_start|>[fn(args)]<|tool_call_end|>
                              which no shipped vLLM parser handles)

scripts/
└── serve_vllm_lfm2.sh     # launches vLLM with the custom parser

docs/
└── INTEGRATION_LFM2VL.md  # this file
```

## Setup (other machine)

### 1. Pull this branch

```bash
git pull origin Branch/refactor
```

### 2. Get the trained model checkpoint

The checkpoint lives in the Kaggle kernel output for **S64 / S65**. Either:

**Option A — download from Kaggle CLI**:
```bash
# After running S65 to completion (do NOT delete weights in the cleanup cell)
kaggle kernels output titanic12/s65-grpo-long -p /tmp/s65 \
    --file-pattern '.*outputs/weights/.*'
# Result: /tmp/s65/outputs/weights/step_100/{model.safetensors, config.json, ...}
```

**Option B — re-upload as a Kaggle dataset** (recommended for repeat use):
```bash
# Download once, then:
kaggle datasets create -p /path/to/checkpoint  # set up dataset-metadata.json first
```

**Option C — direct file transfer** (rsync/scp from a Kaggle download).

The checkpoint is ~900 MB. Place it where the vLLM machine can read it.

### 3. Install vLLM (>= 0.19)

```bash
pip install vllm==0.19.0
# Plus your usual torch / cuda
```

### 4. Launch vLLM

```bash
MODEL_PATH=/path/to/lfm2vl-checkpoint  bash scripts/serve_vllm_lfm2.sh
# vLLM listens on :8000 with the lfm2_pythonic parser registered.
```

### 5. Get precompute data

The agent's `compute_index_delta` tool reads precomputed YAML files. Either:
- Download `titanic12/satelliteagent-precompute-v4` Kaggle dataset
- Or compute on the fly from raw Sentinel-2 (out of scope here)

### 6. Drive the agent

```python
from agent.lfm2_agent import run_lfm2_agent

result = run_lfm2_agent(
    case_id="mcd64a1_h09v04_202307_p4582_-12035",
    before_path="/data/raw_v4/curated_pairs/<case_id>/before.png",  # optional
    after_path="/data/raw_v4/curated_pairs/<case_id>/after.png",    # optional
    precompute_root="/data/precompute_v4",
    vllm_url="http://localhost:8000/v1",
    served_model="LFM2.5-VL-450M-sft-grpo",  # the dir name vLLM uses as model id
    include_images=False,  # 75% accuracy mode (recommended)
    # include_images=True,  # 51% accuracy mode (real VLM)
)

print(result["terminal"])      # "submit_to_ground" | "drop"
print(result["tool_call_log"]) # list of {name, args} per turn
print(result["raw_log"])       # per-turn finish_reason / content / n_tool_calls
```

## Critical settings (don't change without re-evaluating)

| setting | value | reason |
|---|---|---|
| `tool_choice` | `"required"` | Without this, model often answers in plain text |
| `temperature` | `0.0` | Deterministic for reproducibility |
| `max_turns` | `6` | Trained pattern needs 3 turns; 6 is safety |
| `include_images` | `False` | +24pp vs `True`. See "Key insight" above. |

## Wiring into app/server.py (TBD, separate task)

The app currently uses `agent.providers.GeminiProvider`. To switch to LFM2.5-VL:
1. Add a "lfm2vl" provider that wraps `run_lfm2_agent`
2. Select via env var `SAT_AGENT_PROVIDER=lfm2vl`
3. Pass `SAT_AGENT_VLLM_URL`, `SAT_AGENT_PRECOMPUTE_ROOT` etc.

The schema in `tools/schema.py` is for the OLD 14-tool ReAct agent and does
NOT match the trained LFM2.5-VL tool set. Use `agent.lfm2_agent.TOOLS`
(4 tools) for the LFM2 path; do not import from `tools/schema.py`.

## Troubleshooting

### vLLM error: `Tool parser 'lfm2_pythonic' not found`

The custom parser isn't registered. Either:
- Pass `--tool-parser-plugin /abs/path/to/agent/lfm2_tool_parser.py`
- Or import the file before vLLM starts (e.g. via `sitecustomize.py` on PYTHONPATH)

### Model returns text, not tool_calls

Check that `tool_choice="required"` is set in the request. With `"auto"` the
trained model often falls back to natural-language answers (especially with
images present).

### Per-category accuracy looks wrong

The model is biased toward `drop` for ambiguous cases — `pos_volcanic` and
`neg_hard_volcano` are both ~50%. This is a known limitation of the v4 dataset
+ 450M model (see `kaggle/DATASET_V4_FINDINGS.md`). GRPO marginally helps
`pos_deforestation`.
