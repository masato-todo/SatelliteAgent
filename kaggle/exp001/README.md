# kaggle/exp001 — Raw data upload to Kaggle Dataset

**Dataset id**: `<KAGGLE_USER>/satelliteagent-raw-v1`

Phase 1-3 のローカル成果物を **そのまま (raw)** Kaggle に上げる。Kaggle 側の notebook で resize / cases YAML 分割 / SFT format 変換などを行う方針 (再 upload 不要に保つため)。

## Contents

```
stage/                                  # gitignored, ./upload.sh で再生成
├── curated_pairs/<scene_id>/           # Phase 2 確定済 67 scene
│   ├── before.png  (1019×1013, ~2MB)
│   ├── after.png   (~2MB)
│   └── meta.yaml
├── canonical_dataset.yaml              # 67 case の lat/lon/dates/expected_action
├── scene_catalog.yaml                  # Phase 1 MCD64A1 burn polygon catalog
├── gt_polygons/<id>.geojson            # MCD64A1 burn polygon (WGS84)
├── traces/<scene_id>__YYYYMMDDTHHMMSSZ.yaml   # Phase 3 Gemini ReAct traces (68本)
└── dataset-metadata.json
```

合計 ~219 MB。

## Re-upload

```bash
cd kaggle/exp001
./upload.sh                # 初回 → kaggle datasets create
UPDATE=1 MSG="update" ./upload.sh   # 更新 → kaggle datasets version
```

`upload.sh` は `data/` から毎回 stage を再生成するので、ローカル `data/` を更新したら走らせ直すだけ。

## Kaggle 側での前処理 (別 notebook 想定)

DATA_SPEC.md の §3 schema に整形する作業は Kaggle 上で完結:
- 画像 resize (384 or 512)
- `canonical_dataset.yaml` → `eval/cases/<id>.yaml` 分割
- splits/{train,val,test}.txt 生成
- manifest.json
- (任意) SFT Stage 1 jsonl 生成 (`traces/` から)
- (任意) SFT Stage 0 fmt-warmup 合成

これらは別の `satelliteagent-data-prep` notebook で出力 → `satelliteagent-data-processed` dataset として保存する流れを想定。
