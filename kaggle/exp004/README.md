# kaggle/exp004 — Precompute v3 (offline tool cache)

**Dataset id**: `<KAGGLE_USER>/satelliteagent-precompute-v3`

458 case zips, ~94 GB. Per-case offline cache of every tool's output so
GRPO rollouts can run **without any SimSat / STAC / Gemini calls**.

> Full schema, source breakdown, hard-negative rationale and load examples
> are in [docs/DATASET_V3.md](../../docs/DATASET_V3.md). Read that first.

## What it caches per case

```
<case_id>.zip  (each ~200 MB, ~100 files inside)
└── <case_id>/
    ├── classify_change.yaml                # Gemini result
    ├── compute_index/<INDEX>__{before,after}.png + .stats.yaml   # 6 indices × 2 = 12
    ├── compute_index_delta/<INDEX>.png + .stats.yaml             # 6 deltas
    ├── fetch_band/<band>__{before,after}.png + .stats.yaml       # 11 bands × 2 = 22
    ├── false_color/<combo>__{before,after}.png + .stats.yaml     # 5 combos × 2 = 10
    └── _stats.yaml                                                # build timing
```

Indices: NDVI, NDWI, MNDWI, NBR, NDBI, NDSI
Bands: blue, green, red, rededge1-3, nir, nir08, nir09, swir16, swir22
False-color combos: nir-red-green (CIR), swir22-nir-red (burn/lava),
swir16-nir-blue (urban), nir-swir16-red (agri), red-green-blue (true)

## What's new vs v2 (67 → 458 cases)

- ✚ 63 GDACS volcanic
- ✚ 131 PRODES Amazon deforestation
- ✚ ~91 negative biome-diverse
- ✚ 130 hard-negative (drop @ positive sites in stable years)
- ✗ 24 cases missing (SimSat transient outages during build) — these
  exist in raw-v3 but have no precompute zip. Detect with:
  ```python
  missing = [c["id"] for c in cases
             if not os.path.exists(f"{DATA_ROOT}/{c['id']}.zip")]
  ```

## PNG encoding (identical to live tool output)

| tool | mode | scaling |
|---|---|---|
| `fetch_band` | grayscale L | `(reflectance / 3000) × 255` |
| `false_color` | RGB | 3 bands stacked |
| `compute_index` | RGB | diverging red(-1) ↔ white(0) ↔ green(+1) |
| `compute_index_delta` | RGB | red(decrease) ↔ white(0) ↔ blue(increase) |

Pixel-identical to [tools/spectral.py](../../tools/spectral.py).

## Loading example

Each case is in its own zip at depth-1:

```python
import zipfile, yaml, io
DATA_ROOT = "/kaggle/input/satelliteagent-precompute-v3"

def load_tool_response(case_id: str, tool: str, **args):
    with zipfile.ZipFile(f"{DATA_ROOT}/{case_id}.zip") as z:
        if tool == "classify_change":
            return yaml.safe_load(z.read(f"{case_id}/classify_change.yaml"))["response"]
        if tool == "compute_index":
            stub = f"{case_id}/compute_index/{args['index']}__{args.get('which','after')}"
            return yaml.safe_load(z.read(f"{stub}.stats.yaml"))["response"]
        if tool == "compute_index_delta":
            stub = f"{case_id}/compute_index_delta/{args['index']}"
            return yaml.safe_load(z.read(f"{stub}.stats.yaml"))["response"]
        if tool == "fetch_band":
            stub = f"{case_id}/fetch_band/{args['band']}__{args.get('which','after')}"
            return yaml.safe_load(z.read(f"{stub}.stats.yaml"))["response"]
        if tool == "false_color":
            tag = "-".join(args["bands"])
            side = args.get("which", "after")
            return {"image_bytes": z.read(f"{case_id}/false_color/{tag}__{side}.png")}
        raise ValueError(f"unknown tool: {tool}")

# example
print(load_tool_response("mcd64a1_h03v06_202308_p2079_-15640",
                          "compute_index_delta", index="NBR"))
```

This lets `satelliteagent_env`'s tool wrapper hit the cache instead of
going to SimSat — GRPO rollout becomes fully offline.

## Re-upload

```bash
cd kaggle/exp004
./upload.sh                          # initial → kaggle datasets create
UPDATE=1 MSG="..." ./upload.sh       # subsequent → kaggle datasets version
```

`upload.sh` mirrors `data/precompute/` into `stage/` then runs
`kaggle datasets create -p stage --dir-mode zip`. Per-case directories
become 458 individual zips at the dataset root.

## Pairing with raw-v3

Every `<case_id>.zip` here corresponds 1:1 to an entry in
`<KAGGLE_USER>/satelliteagent-raw-v3` ([kaggle/exp003](../exp003/)) /
`canonical_dataset.yaml`. Use raw-v3 for image inspection / VLM input;
use this dataset for offline tool replay.
