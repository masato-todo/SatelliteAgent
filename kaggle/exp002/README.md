# kaggle/exp002 — Tool response precompute (offline cache)

**Dataset id**: `<KAGGLE_USER>/satelliteagent-precompute-v1`

Phase B として `scripts/build_precompute_v2.py` が **Sentinel-2 / Gemini API へのアクセスなしに全ツール出力を返せるようにキャッシュ** したもの。Kaggle 等のオフライン環境で GRPO rollout を回すために必要。

## 何が入っているか

67 case 全てについて、TOOL_SPEC §4 に基づく以下の事前計算結果:

```
<case_id>/
├── classify_change.yaml                    # Gemini VLM の分類結果 (1 件)
├── compute_index/
│   ├── NDVI__before.png  + .stats.yaml     # 6 index × 2 sides = 12
│   ├── NDVI__after.png   + .stats.yaml
│   ├── NDWI__{before,after}.png/yaml
│   ├── MNDWI__{before,after}.png/yaml
│   ├── NBR__{before,after}.png/yaml
│   ├── NDBI__{before,after}.png/yaml
│   └── NDSI__{before,after}.png/yaml
├── compute_index_delta/
│   ├── NDVI.png + .stats.yaml              # 6 index, after - before
│   ├── NDWI.png + .stats.yaml
│   ├── MNDWI.png + .stats.yaml
│   ├── NBR.png   + .stats.yaml
│   ├── NDBI.png  + .stats.yaml
│   └── NDSI.png  + .stats.yaml
├── fetch_band/
│   ├── blue__{before,after}.png  + .stats.yaml   # 11 bands × 2 sides = 22
│   ├── green__{before,after}.png/yaml
│   ├── red__{before,after}.png/yaml
│   ├── rededge1__{before,after}.png/yaml
│   ├── rededge2__{before,after}.png/yaml
│   ├── rededge3__{before,after}.png/yaml
│   ├── nir__{before,after}.png/yaml
│   ├── nir08__{before,after}.png/yaml
│   ├── nir09__{before,after}.png/yaml
│   ├── swir16__{before,after}.png/yaml
│   └── swir22__{before,after}.png/yaml
├── false_color/
│   ├── nir-red-green__{before,after}.png + .yaml      # 5 combos × 2 sides = 10
│   ├── swir22-nir-red__{before,after}.png/yaml         # burn severity
│   ├── swir16-nir-blue__{before,after}.png/yaml        # urban vs vegetation
│   ├── nir-swir16-red__{before,after}.png/yaml         # agricultural
│   └── red-green-blue__{before,after}.png/yaml         # true color
└── _stats.yaml                              # per-case timing breakdown
```

per-case **約 100 ファイル / 42 MB**、67 case 合計 **約 6,700 ファイル / 2.9 GB**。

## 各 yaml のスキーマ例

### `classify_change.yaml`
```yaml
args: {}
response:
  classes: [{name: fire, confidence: 0.98}]
  bboxes: [[300, 300, 600, 400]]
  source: gemini
  model:  gemini-2.5-flash
```

### `compute_index/<INDEX>__<side>.stats.yaml`
```yaml
args: {index: NBR, which: after}
response:
  index: NBR
  stats:
    min:    -0.825
    max:     1.032
    mean:   -0.058
    median: -0.062
    frac_decrease_strong: 0.166   # |idx| > 0.2 の負側比率
    frac_increase_strong: 0.057
```

### `compute_index_delta/<INDEX>.stats.yaml`
```yaml
args: {index: NBR}
response:
  index: NBR
  delta_stats:
    min: -1.05, max: 0.83, mean: -0.12, median: -0.10
    frac_decrease_strong: 0.43    # 焼け跡など強い減少領域
    frac_increase_strong: 0.02
```

### `fetch_band/<band>__<side>.stats.yaml`
```yaml
args: {band: nir, which: after}
response:
  band: nir
  datetime:    "2023-10-30T19:00:00Z"
  cloud_cover: 4.0
  source:      sentinel-2b
  stats: {min: 12, max: 9876, mean: 1234.5, std: 850.2}
```

### `_stats.yaml` (per-case timing)
```yaml
case_id: mcd64a1_h03v06_202308_+2079_-15640
timing:
  fetch:                10.2     # 2 STAC searches (before + after, multi-band)
  fetch_band:            1.2     # 22 PNG saves
  compute_index:         2.7     # numpy formulas
  compute_index_delta:   1.6
  false_color:           2.5
  classify_change:       8.4     # Gemini API
  total:                26.6
```

## PNG エンコード仕様 (v1 ツールと完全互換)

| ツール | mode | スケーリング |
|---|---|---|
| `fetch_band` | grayscale L | `(reflectance / 3000) × 255` (clip 0-255) |
| `false_color` | RGB | 3 bands を上記スケーリングで stack |
| `compute_index` | RGB | `_colormap_signed`: 赤(-1) ↔ 白(0) ↔ 緑(+1) diverging |
| `compute_index_delta` | RGB | `_colormap_delta`: 赤(減) ↔ 白(0) ↔ 青(増)、vmax=0.4 saturation |

これらは [tools/spectral.py](../../tools/spectral.py) の元実装と pixel-level で同一。

## case 内訳

`<KAGGLE_USER>/satelliteagent-raw-v1` (= exp001) と同じ 67 case:

| type | count | expected_action | 由来 |
|---|---|---|---|
| positive (MCD64A1 wildfire) | 51 | submit_to_ground | Phase 1 で MCD64A1 から検出した焼け跡 |
| negative (no_change) | 14 | drop | 世界各地の「異常なし」シーン (砂漠/海/極地など) |
| negative (cloud_blocked) | 2 | drop | After が雲で判定不能なケース |

balance 51:16 = 76:24 (submit:drop)。GRPO 学習時は 「always submit ゲーミング」防止のため、reward 関数で drop の重み増す等の工夫推奨。

## アップロード方法

```bash
cd kaggle/exp002
./upload.sh                          # 初回 → kaggle datasets create
UPDATE=1 MSG="update" ./upload.sh    # 更新 → kaggle datasets version
```

`upload.sh` は `data/precompute/` から毎回 stage を再生成。`stage/` は gitignore (再生成可能なので git に乗せない)。

## Kaggle notebook での読み出し例

```python
DATA_ROOT = "/kaggle/input/satelliteagent-precompute-v1"

import yaml
from pathlib import Path

def load_tool_response(case_id: str, tool: str, **args):
    """Replay any tool call against the precomputed cache."""
    case_dir = Path(DATA_ROOT) / case_id
    if tool == "classify_change":
        return yaml.safe_load((case_dir / "classify_change.yaml").read_text())["response"]
    if tool == "compute_index":
        idx, side = args["index"], args.get("which", "after")
        stub = case_dir / "compute_index" / f"{idx}__{side}"
        return yaml.safe_load(stub.with_suffix(".stats.yaml").read_text())["response"]
    if tool == "compute_index_delta":
        stub = case_dir / "compute_index_delta" / args["index"]
        return yaml.safe_load(stub.with_suffix(".stats.yaml").read_text())["response"]
    if tool == "fetch_band":
        stub = case_dir / "fetch_band" / f"{args['band']}__{args.get('which','after')}"
        return yaml.safe_load(stub.with_suffix(".stats.yaml").read_text())["response"]
    if tool == "false_color":
        tag = "-".join(args["bands"])
        side = args.get("which", "after")
        return {"image_path": str(case_dir / "false_color" / f"{tag}__{side}.png")}
    raise ValueError(f"unknown tool: {tool}")
```

`satelliteagent_env` の `setup_state` で precompute_root を読み込み、tool wrapper でこの関数を介して on-call 時にキャッシュ参照する形に組み込めば、GRPO rollout が完全オフラインで回せる。
