# SatelliteAgent Dataset v3 — Overview for collaborators

This document describes the **two paired Kaggle datasets** that ship the
SatelliteAgent training data:

| Dataset | Kaggle id | Size | Purpose |
|---|---|---|---|
| **raw-v3** | `<KAGGLE_USER>/satelliteagent-raw-v3` | ~8.3 GB | Sentinel-2 Before/After PNG pairs + GT polygons + agent traces |
| **precompute-v3** | `<KAGGLE_USER>/satelliteagent-precompute-v3` | ~94 GB | Per-case offline cache of every tool's output (compute_index, fetch_band, false_color, classify_change, ...) so GRPO rollouts run with **zero SimSat / STAC / Gemini calls** |

Both are private datasets (CC-BY-NC-SA-4.0).

---

## 1. What's a "case"?

A **case** = one (lat, lon, before_date, after_date, size_km, expected_action)
tuple drawn from one of 5 disaster-event sources, plus its corresponding
Sentinel-2 imagery. Every case has a stable `id` that is the primary key
across both datasets.

### Sources (by `source` field in `canonical_dataset.yaml`)

| Source | Type | Count | Detection signal | id prefix |
|---|---|---|---|---|
| **MCD64A1** | wildfire (positive) | 51 | NBR delta, false_color SWIR-NIR | `mcd64a1_<tile>_<yyyymm>_<lat>_<lon>` |
| **GDACS_VO** | volcanic (positive) | 63 | NBR + SWIR thermal + ash albedo | `volcano_gdacs_<eventid>_<episode>` |
| **PRODES** | deforestation (positive) | 131 | NDVI / NBR drop, polygon overlay | `prodes_amazon_<uid>` |
| **negative** | drop expected (no change) | ~107 | (none) — biome-diverse boring scenes | `neg__<type>__<region>__<tile>` |
| **hard_negative** | drop expected at positive-source sites | 130 | (none) — same lat/lon as positives but in stable years | `hardneg_{volcano,forest,preburn}_<id>` |
| **TOTAL** | — | **482** | — | — |

> Why hard negatives matter: without them, an agent learns "forest visible
> → change happens" or "volcano visible → eruption". Hard negatives place
> the agent in the same visual context as positives but demand a `drop`
> decision because the time pair shows no real change.

### Counts at a glance (positive : negative ≈ 245 : 237)

- positives: fire 51 + volcanic 63 + deforestation 131 = **245**
- negatives: bored 107 + hard 130 = **237**

---

## 2. raw-v3 layout (~8.3 GB)

Top-level structure inside the Kaggle dataset:

```
satelliteagent-raw-v3/
├── canonical_dataset.yaml           # 482 entries (master list)
├── scene_catalog.yaml               # MCD64A1 wildfire scene metadata
├── curated_pairs.zip                # ←— 8.3 GB, 482 case dirs zipped together
│   └── <case_id>/
│       ├── before.png  (1019×1013, ~3 MB)
│       ├── after.png
│       └── meta.yaml
├── gt_polygons.zip                  # 5.6 MB
│   └── <case_id>.geojson            # WGS84 burn / clearing polygons
│                                    # (only for MCD64A1 + PRODES; others are point cases)
└── traces.zip                       # 288 KB, Phase 3 Gemini ReAct traces
```

### `canonical_dataset.yaml` schema

```yaml
created_at: '2026-05-02T...'
n_cases: 482
cases:
  - id:      mcd64a1_h03v06_202308_p2079_-15640
    label:   fire             # fire | volcanic | deforestation | no_change
    type:    positive         # positive | negative
    lat:     20.7892
    lon:     -156.4037
    size_km: 10.0
    request:
      before_date: '2023-06-02'
      after_date:  '2023-10-30'
      window_days: 30
    expected_resolved:
      before_datetime: '2023-06-02T20:34:51Z'
      after_datetime:  '2023-10-30T20:34:53Z'
    event:
      name:   ...
      period: ['2023-08-01', '2023-08-31']
```

### `curated_pairs/<case_id>/meta.yaml` schema

```yaml
scene_id:        prodes_amazon_754419
type:            positive
expected_action: submit_to_ground
lat:             -8.234
lon:             -54.123
size_km:         10.0
before:
  date:     '2021-04-15'
  key:      <cache_key>
  datetime: '2021-04-12T13:48:30Z'
  stats:    {cloud_proxy: 0.07, nodata_fraction: 0.0, usable: true, ...}
after:
  date:     '2024-08-13'
  key:      <cache_key>
  datetime: '2024-08-21T13:48:24Z'
  stats:    {cloud_proxy: 0.02, nodata_fraction: 0.0, usable: true, ...}
event:           {type: deforestation, start: null, end: null, name: null}
saved_at:        '2026-04-30T...'
```

> **Important: `curated_pairs.zip` is one big 8.3 GB file** (Kaggle's
> `--dir-mode zip` zips at depth-1, and the `curated_pairs/` directory is
> at depth 1). For typical use, extract just the case_ids you need:
>
> ```python
> import zipfile, io
> from PIL import Image
> z = zipfile.ZipFile("/kaggle/input/satelliteagent-raw-v3/curated_pairs.zip")
> with z.open(f"curated_pairs/{case_id}/before.png") as f:
>     before = Image.open(io.BytesIO(f.read()))
> ```

---

## 3. precompute-v3 layout (~94 GB)

Different layout — **each case is its own zip at depth-1** so you load
exactly what you need:

```
satelliteagent-precompute-v3/
├── _run_stats.yaml                       # build timing summary
└── <case_id>.zip   (× 458 files; ~24 missing due to SimSat transient errors)
    ├── classify_change.yaml              # Gemini classification result
    ├── compute_index/
    │   ├── NDVI__before.png   + .stats.yaml
    │   ├── NDVI__after.png    + .stats.yaml
    │   ├── NDWI__{before,after}.png + .stats.yaml
    │   ├── MNDWI__{before,after}.png + .stats.yaml
    │   ├── NBR__{before,after}.png + .stats.yaml
    │   ├── NDBI__{before,after}.png + .stats.yaml
    │   └── NDSI__{before,after}.png + .stats.yaml         # 6 indices × 2 sides = 12
    ├── compute_index_delta/
    │   ├── NDVI.png + .stats.yaml                          # 6 deltas
    │   └── ... (NDWI / MNDWI / NBR / NDBI / NDSI)
    ├── fetch_band/
    │   ├── blue__{before,after}.png + .stats.yaml
    │   ├── green__{before,after}.png + .stats.yaml
    │   ├── red, rededge1-3, nir, nir08, nir09, swir16, swir22 (× 2 sides) = 22 files
    ├── false_color/
    │   ├── nir-red-green__{before,after}.png + .stats.yaml      # CIR
    │   ├── swir22-nir-red__{before,after}.png + .stats.yaml     # burn / lava
    │   ├── swir16-nir-blue__{before,after}.png + .stats.yaml    # urban vs vegetation
    │   ├── nir-swir16-red__{before,after}.png + .stats.yaml     # agricultural
    │   └── red-green-blue__{before,after}.png + .stats.yaml     # true color
    └── _stats.yaml                                          # per-case build timing
```

### Per-case totals
~100 files / ~200 MB per case → 458 cases × 200 MB ≈ 92 GB.

### Cache file schemas

`compute_index/<INDEX>__<side>.stats.yaml`:
```yaml
args: {index: NBR, which: after}
response:
  index: NBR
  stats:
    min: -0.83, max: 1.03, mean: -0.06, median: -0.06
    frac_decrease_strong: 0.166   # |idx| > 0.2 negative side
    frac_increase_strong: 0.057
```

`compute_index_delta/<INDEX>.stats.yaml`:
```yaml
args: {index: NBR}
response:
  index: NBR
  delta_stats:
    min: -1.05, max: 0.83, mean: -0.12, median: -0.10
    frac_decrease_strong: 0.43
    frac_increase_strong: 0.02
```

`fetch_band/<band>__<side>.stats.yaml`:
```yaml
args: {band: nir, which: after}
response:
  band: nir
  datetime:    '2023-10-30T19:00:00Z'
  cloud_cover: 4.0
  source:      sentinel-2b
  stats: {min: 12, max: 9876, mean: 1234.5, std: 850.2}
```

`classify_change.yaml`:
```yaml
args: {}
response:
  classes: [{name: fire, confidence: 0.98}]
  bboxes: [[300, 300, 600, 400]]
  source: gemini
  model:  gemini-2.5-flash
```

### PNG encoding (identical to live tool output)

| tool | mode | scaling |
|---|---|---|
| `fetch_band` | grayscale L | `(reflectance / 3000) × 255` (clip 0-255) |
| `false_color` | RGB | 3 bands stacked with same scaling |
| `compute_index` | RGB | diverging red(-1) ↔ white(0) ↔ green(+1) |
| `compute_index_delta` | RGB | red(decrease) ↔ white(0) ↔ blue(increase), vmax=0.4 saturation |

---

## 4. Loading example (Kaggle notebook)

```python
import zipfile, yaml, io
from pathlib import Path
from PIL import Image

RAW_ROOT       = "/kaggle/input/satelliteagent-raw-v3"
PRECOMP_ROOT   = "/kaggle/input/satelliteagent-precompute-v3"

# ----- master list -----
canonical = yaml.safe_load(open(f"{RAW_ROOT}/canonical_dataset.yaml"))
cases     = canonical["cases"]   # 482 entries

# ----- load one case's Before/After PNG (from raw) -----
def load_pair(case_id):
    with zipfile.ZipFile(f"{RAW_ROOT}/curated_pairs.zip") as z:
        with z.open(f"curated_pairs/{case_id}/before.png") as f:
            before = Image.open(io.BytesIO(f.read())).copy()
        with z.open(f"curated_pairs/{case_id}/after.png") as f:
            after  = Image.open(io.BytesIO(f.read())).copy()
    return before, after

# ----- replay any tool from precompute -----
def load_tool_response(case_id, tool, **args):
    """Returns the same dict that the live tool would return."""
    case_zip = f"{PRECOMP_ROOT}/{case_id}.zip"
    with zipfile.ZipFile(case_zip) as z:
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
            side = args.get("which","after")
            png_path = f"{case_id}/false_color/{tag}__{side}.png"
            png_bytes = z.read(png_path)
            return {"image_bytes": png_bytes}
        raise ValueError(f"unknown tool: {tool}")

# ----- example: pull NBR delta for a wildfire case -----
case_id = "mcd64a1_h03v06_202308_p2079_-15640"
nbr_d = load_tool_response(case_id, "compute_index_delta", index="NBR")
print(nbr_d["delta_stats"]["frac_decrease_strong"])   # should be high for burn scar
```

---

## 5. Caveats / things to know

1. **24 cases missing from precompute-v3**: SimSat (the local Sentinel-2
   proxy used during build) had transient outages. Affected case_ids
   exist in `canonical_dataset.yaml` and `raw-v3/curated_pairs/` but have
   **no zip in precompute-v3**. Detect with:
   ```python
   import os
   missing = [c["id"] for c in cases if not os.path.exists(f"{PRECOMP_ROOT}/{c['id']}.zip")]
   ```

2. **size_km is uniformly 10 km × 10 km** (1019×1013 pixel @ 10 m/px). All
   sources curated at this scale.

3. **gt_polygons exist only for MCD64A1 + PRODES** (the two sources with
   real polygon datasets). Volcanic/negative/hard_negative are point
   cases — `gt_polygons/<id>.geojson` will not exist.

4. **Hard negative date strategy**: pre-event year(s) with the same
   lat/lon as a positive case. See `hard_negative_cases.yaml` in
   `data/metadata/disaster_m3/` of the source repo.

5. **License**: CC-BY-NC-SA 4.0. Underlying data: MCD64A1 (NASA),
   GDACS (EU JRC), PRODES (INPE Brazil), Sentinel-2 (ESA Copernicus).

6. **Reproducibility**: build scripts are in the source repo:
   - catalogs: `scripts/collect_*.py` (mcd64a1, volcanic, deforestation,
     negatives, hard_negatives)
   - Phase 2 (curated_pairs): `scripts/auto_fill_pairs.py`
   - precompute build: `scripts/build_precompute_v2.py`
   - Kaggle upload: `kaggle/exp003/upload.sh` (raw), `kaggle/exp004/upload.sh` (precompute)

---

## 6. Quick start for a collaborator

```python
# 1. Add both datasets to your Kaggle notebook (Add Data button)
# 2. Pick a case
case = next(c for c in cases if c["label"] == "deforestation")
case_id = case["id"]

# 3. Look at the imagery
before, after = load_pair(case_id)
display(before, after)   # in Jupyter

# 4. Look at what the tool would return (no SimSat/Gemini call needed)
print(load_tool_response(case_id, "classify_change"))
print(load_tool_response(case_id, "compute_index_delta", index="NDVI"))
```

That's it — the precompute means you can replay any tool call entirely
offline, which is the whole point of v3.
