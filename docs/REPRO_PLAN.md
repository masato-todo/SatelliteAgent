# SatelliteAgent — Reproducibility Plan

新しい開発者が `git clone` → 1コマンドで eval / UI / agent loop を再現できる状態にするためのロードマップ。

---

## 1. 現状インベントリ (2026-05-06)

### Branches
- `Branch/refactor` (HEAD, 現作業): FireEdge GT 統合、Fetch FireEdge frame ボタン、`detect_wildfire` の eval 一致 fix、realtime SimSat agent
- `origin/feat/toolcall`: ReAct loop 簡略化、operator instructions textarea、anti-fabrication guards、`DATASET_V3.md` / `INTEGRATION_LFM2VL.md`、kaggle exp003/004、`config/providers.yaml` のローカル vLLM デフォルト化
- 共通祖先からの divergence: `27 files, +2133/-53`

### Services (現状ローカルで手動起動)
| port | コンテナ / プロセス | 役割 |
|------|---------------------|------|
| 7860 | `python -m app.server` | UI + REST API |
| 8085 | `serve_lfm/Dockerfile` (`lfm25-vl-trl-ft:latest` ベース) | wildfire LoRA (LFM2.5-VL-450M-wildfire) |
| 8086 | `vllm-lfm2-pin:latest` (cu130 + transformers 5.5.0 + sed patch) | LFM2.5-VL-450M-sft-grpo S64 (multi-turn agent) |
| 9005 | external (別チームの SimSat) | Sentinel-2 mock backend |

### データサイズ (リポジトリ ` git push` 不可サイズ)
| path | size | repo に含めるか |
|------|------|------|
| `data/scenarios/` | 24G | ✗ (.gitignore — fetch cache、生成物) |
| `data/precompute/` | 19G | ✗ (.gitignore — `build_precompute_v2.py` の生成物) |
| `data/curated_pairs/` | 1.4G | ✗ (.gitignore — canonical pair PNG) |
| `data/derived/` | 1.1G | ✗ (.gitignore — composite cache) |
| `data/metadata/` | 178M | △ 一部 (yaml だけ ~600K を含める、CSV/raw は除外) |
| `data/gt_polygons/` | 5.6M | ○ 含める (xBD damage overlay 用) |
| `data/wildfire_composites/` | 4K (現在ほぼ空) | ✗ |
| `data/scene_catalog.yaml` (19K) / `canonical_dataset.yaml` (195K) | < 1M | ○ |
| eval jsonl (`eval_*.jsonl`) | 各 5-100K | ○ commit 済み eval 履歴として |

### Secrets (`.env` に存在、push 厳禁)
- `GOOGLE_API_KEY` (Gemini classify_change)
- `FIRMS_MAP_KEY` (FIRMS データ収集 — `scripts/collect_firms_fire.py`)
- (Kaggle: `~/.kaggle/kaggle.json` 個別管理)

---

## 2. Phase 1 — Branch 統合 (`feat/toolcall` → `Branch/refactor`)

### 統合対象
`feat/toolcall` の差分 (`git diff feat/toolcall...Branch/refactor --stat` 抜粋):

| ファイル | feat/toolcall 側 | refactor 側 (現状) | 統合方針 |
|---|---|---|---|
| `agent/lfm2_agent.py` | (新規 458行) | (refactor 側で realtime SimSat 化) | refactor 側を優先。`feat/toolcall` の `forced_tool_steps`/`user_instructions` を移植 |
| `agent/lfm2_tool_parser.py` | (新規 96行) | 同 | feat/toolcall 側を採用 (差分なし想定、要 diff 確認) |
| `app/server.py` | +361 (region-bind, instructions param, run_lfm2 endpoint) | refactor で detect_wildfire fix + FetchRequest.before_window_days | 手動 merge: 両方の機能を残す |
| `app/static/js/dm3-fetch.js` | +44 | refactor で FireEdge fetch ボタン追加 | 両方残す |
| `app/static/js/main.js` | +51 (instructions textarea wiring) | refactor で FireEdge button wiring | 両方残す |
| `app/static/index.html` | +2 (instructions textarea) | refactor で FireEdge button + window_days min=1 | 両方残す |
| `tools/stubs.py` | +35 (anti-fabrication) | (差分なしっぽい) | feat/toolcall 側採用 |
| `config/providers.yaml` | local vLLM デフォルト化 | 同 | feat/toolcall 側採用 |
| `kaggle/exp003/`, `kaggle/exp004/` | (新規) | - | そのまま取り込み |
| `docs/DATASET_V3.md`, `docs/INTEGRATION_LFM2VL.md` | (新規) | - | そのまま取り込み |
| `scripts/serve_vllm_lfm2.sh` | (新規 48行) | 同 | feat/toolcall 側採用 |

### 手順
1. `Branch/refactor` 作業をすべて commit (現在 `M` がある分)
2. `git fetch origin feat/toolcall`
3. 一時統合ブランチを切る: `git checkout -b integrate/refactor-toolcall`
4. `git merge origin/feat/toolcall` — conflict 発生想定箇所:
   - `app/server.py` (FetchRequest 周辺、build_tool_registry)
   - `app/static/index.html` (button 列 + textarea)
   - `app/static/js/main.js` (boot wiring)
   - `agent/lfm2_agent.py` (realtime vs forced-step)
5. ファイルごとに手動 resolve (上の方針表に従う)
6. smoke test (Phase 5) で eval 数値が崩れないことを確認
7. `Branch/refactor` に fast-forward → `git push origin Branch/refactor`
8. PR で `main` に merge

---

## 3. Phase 2 — データ最小集合の確定

### 含めるもの (~10MB max)
```
data/
├── metadata/disaster_m3/*.yaml   (~600K — 全 case yaml)
├── gt_polygons/                  (5.6M — xBD damage overlay)
├── scene_catalog.yaml            (19K)
├── canonical_dataset.yaml        (195K)
├── eval_hf_simsat.jsonl          (15K — eval ベースライン履歴)
├── eval_hf.jsonl                 (6.5K)
└── README.md                     (← 新規。各 yaml の出自と再生成コマンド)
```

### `.gitignore` に追加
```
data/scenarios/
data/precompute/
data/curated_pairs/
data/derived/
data/raw_mcd64a1/
data/wildfire_composites/
data/traces/        # 個別実験トレース、共有しない
data/metadata/*.csv # disaster_m3_xbd_buildings.csv 等の重い CSV
*.log
.env
.venv/
satelliteagent_env/
satellite_agent.egg-info/
__pycache__/
```

### Smoke 用最小キャッシュ (任意 zip 配布、< 50MB)
- 1 FireEdge fire case (`fireedge_train_firms_pos_001`) の Before/After PNG + meta sidecar
- 1 fail-known negative case
- 配布先候補: GitHub Release asset (LFS は使わない)

### データ再生成スクリプト (既存、`scripts/`)
| スクリプト | 出力 | 外部依存 |
|---|---|---|
| `collect_fireedge_hf.py` | `fireedge_hf_cases.yaml` | HF datasets `YujiYamaguchi/fireedge-sentinel2-wildfire` |
| `collect_firms_fire.py` | `firms_fire_cases.yaml` | NASA FIRMS API (`FIRMS_MAP_KEY`) |
| `collect_ems.py` | `ems_cases.yaml` | Copernicus EMS |
| `collect_volcanic.py` | `volcanic_cases.yaml` | GDACS |
| `collect_deforestation.py` | `deforestation_cases.yaml` | PRODES |
| `collect_algal_bloom.py` | `algal_bloom_cases.yaml` | (内部リスト) |
| `collect_negatives_v2.py` | `negative_cases.yaml` | 乱数生成 |
| `collect_hard_negatives.py` | `hard_negative_cases.yaml` | (内部リスト) |
| `build_scene_catalog.py` | `scene_catalog.yaml` | xBD GeoJSON |
| `build_precompute_v2.py` | `data/precompute/<case>/` | SimSat (heavy, ~30min) |

→ いずれも include する。**各スクリプトの先頭 docstring に "Reproduces: data/metadata/.../X.yaml" を入れる** リンクを追加する。

---

## 4. Phase 3 — Service コンテナ整備

**設計判断**: モデルサーバは **2 つに分離維持** (1 docker-compose で起動するので体感は 1 コマンド)。

vLLM 1 process 化は検証済みで断念:
- `--lora-modules` 経由は **vLLM が `multi_modal_projector.linear_*` の LoRA を refuse**、`vision_tower.*` も silently 無視 → wildfire LoRA は読めない (`vllm/lora/lora_model.py:check_unexpected_modules` で fail)
- 事前 merge した `LFM2.5-VL-450M-Wildfire-merged` を vLLM で serve する案も検証 → FireEdge test split で **recall=0.0** (transformers+peft の baseline 0.933 が再現せず)、merge が VL 部分込みで正しく行われていない疑い
- → 現状の transformers+peft (wildfire) + vLLM (agent) 構成が **eval 再現性として唯一動作確認できている**

### 3.1 LFM2 agent vLLM (port 8086) — `vllm-lfm2-pin:latest` を Dockerfile 化
新規 `serve_lfm2/Dockerfile`:
```dockerfile
FROM vllm/vllm-openai:cu130-nightly
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    transformers==5.5.0 tokenizers==0.22.2 \
    huggingface_hub==1.9.0 mistral_common==1.11.0 safetensors==0.6.2
# transformers 5.x renamed Lfm2Config.block_ff_dim → intermediate_size
RUN sed -i 's/config\.block_ff_dim/config.intermediate_size/g' \
    /usr/local/lib/python*/dist-packages/vllm/model_executor/models/lfm2.py
COPY agent/lfm2_tool_parser.py /agent/lfm2_tool_parser.py
EXPOSE 8086
```
build:  `docker build -t satelliteagent/lfm2-agent:cu130 -f serve_lfm2/Dockerfile .`
run: `scripts/serve_vllm_lfm2.sh` (既存、`MODEL_PATH` だけ env で指定)

### 3.2 wildfire LoRA (port 8085) — 既存 `serve_lfm/` (transformers+peft)
- `serve_lfm/Dockerfile` は OK (`lfm25-vl-trl-ft:latest` ベース、peft が adapter を runtime 適用)
- `serve_lfm/server.py` / `requirements.txt` はそのまま
- 起動スクリプトを `scripts/serve_lfm_wildfire.sh` として抽出
- LoRA path: `LFM_BASE_DIR` / `LFM_ADAPTER_DIR` env で指定 (デフォルト `/models/base`, `/models/adapter`)

### 3.3 SimSat (port 9005) — **Optional**
upstream `DPhi-Space/SimSat` (AGPL-3.0) に **タイル境界 mosaic 対応**等のローカルパッチが当たっている。我々の repo には patch ファイルだけ同梱し、bootstrap で「Optional」として apply する方式を採る (vendor すると AGPL が伝染するため)。

**含めるもの (我々の repo 側、~5KB):**
```
patches/simsat/
├── README.md                            # 上流 SHA = 52f5619、各パッチの根拠
├── 001-mosaic-tile-boundary.patch       # sentinel_provider.py: odc.stac.load(items, groupby="solar_day")
│                                        # AOI が MGRS 境界をまたいでも nodata≈0 で取得
├── 002-fastapi-sync-endpoints.patch     # api.py: async def → def (event loop block 回避)
└── 003-compose-sim-only.patch           # docker-compose.yaml: dashboard 削除、sim のみ
```

**bootstrap での分岐:**
```bash
if [[ "${WITH_SIMSAT:-0}" == "1" ]]; then
  SIMSAT_SHA="52f5619"
  git clone https://github.com/DPhi-Space/SimSat.git vendor/SimSat
  (cd vendor/SimSat && git checkout $SIMSAT_SHA \
    && git apply ../../patches/simsat/*.patch)
  docker compose -f vendor/SimSat/docker-compose.yaml up -d
else
  echo "Skipping SimSat setup. Set SIMSAT_API_URL in .env to a reachable instance."
fi
```

**`.env.example` 側:**
```
SIMSAT_API_URL=http://localhost:9005   # チーム共有 endpoint がある場合はそれを書く
```

**何故 patch 必須か (再現性インパクト):**
- mosaic patch が無いと AOI が S2 タイル境界に近い case (FireEdge 300 cases の数十件相当) で after image が半分黒く欠ける → SWIR percentile clip が崩れる → wildfire LFM の prediction が不安定化
- eval baseline (`recall=0.933`) を再現するには patch 適用必須
- ただし上記の通り「ローカルで SimSat 立てる場合」のみ。共有 endpoint に繋ぐ運用なら patch も不要

**ライセンス注意:** SimSat 本体は AGPL-3.0。patch を当てたコードを **公開 fork として配布する場合は fork 側も AGPL** に従う必要あり。本 repo の `patches/` は単なる diff 配布なので AGPL の感染対象外。

### 3.4 docker-compose.yaml (新規、ルート直下)
```yaml
services:
  app:
    build: .              # uv venv 内蔵の slim イメージ
    ports: ["7860:7860"]
    env_file: .env
    volumes: ["./data:/app/data"]
    depends_on: [lfm-wildfire, lfm2-agent]
  lfm-wildfire:
    build: ./serve_lfm
    ports: ["8085:8000"]
    deploy: {resources: {reservations: {devices: [{capabilities: [gpu]}]}}}
  lfm2-agent:
    build: ./serve_lfm2
    ports: ["8086:8086"]
    volumes: ["${MODEL_PATH:-./models/LFM2.5-VL-450M-sft-grpo}:/model"]
    command: ["bash", "/scripts/serve_vllm_lfm2_inside.sh"]
    deploy: {resources: {reservations: {devices: [{capabilities: [gpu]}]}}}
```

---

## 5. Phase 4 — Shell エントリポイント (`scripts/`)

すべて `set -euo pipefail` + `--help` 対応で揃える。

| スクリプト | 役割 |
|---|---|
| `scripts/bootstrap.sh` | `uv venv` → `uv pip install -e ".[simsat,geo]"` → `.env.example` を `.env` にコピー → 必須 env をユーザに prompt → **Optional**: `WITH_SIMSAT=1` のとき `vendor/SimSat` を clone + 上流 SHA pin + `patches/simsat/*.patch` apply + compose up |
| `scripts/run_app.sh` | `.venv/bin/python -m app.server` (env: APP_HOST/PORT) |
| `scripts/run_lfm_wildfire.sh` | port 8085 のコンテナ起動 (既存 `serve_lfm/docker-compose.yaml` 経由) |
| `scripts/serve_vllm_lfm2.sh` | port 8086、既存。docs を加筆 (MODEL_PATH の取得方法 → HF or kaggle artifact) |
| `scripts/collect_all_metadata.sh` | `collect_*.py` を順に呼んで `data/metadata/disaster_m3/*.yaml` を再生成 (FIRMS/Copernicus key 必要) |
| `scripts/eval_wildfire_full.sh` | `eval_wildfire_hf_simsat.py --use-sentinel-datetime --window-days 1` を test/val/train で順に走らせ、jsonl を `data/` に書き出す |
| `scripts/smoke_test.sh` | Phase 5 参照 |
| `scripts/dump_compose_logs.sh` | `docker compose logs --tail=200 lfm-wildfire lfm2-agent app` (debug 用) |

---

## 6. Phase 5 — Smoke test (`scripts/smoke_test.sh`)

CI でも回せる最小再現テスト。  
**前提**: `.env` 設定済み、SimSat 9005 reachable、LFM 8085 reachable。

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1) サーバ起動
nohup .venv/bin/python -m app.server > /tmp/smoke.log 2>&1 &
APP_PID=$!
trap "kill $APP_PID" EXIT
until curl -sf http://localhost:7860/api/templates > /dev/null; do sleep 1; done

# 2) FireEdge fire case を fetch
FETCH=$(curl -sf -X POST http://localhost:7860/api/fetch \
  -H 'Content-Type: application/json' \
  -d '{"lat":7.58296,"lon":29.53288,"before_date":"2024-09-16",
       "after_date":"2025-03-15T08:38:09Z","size_km":5.0,
       "window_days":1,"before_window_days":30}')
B_KEY=$(jq -r '.before.key' <<<"$FETCH")
A_KEY=$(jq -r '.after.key'  <<<"$FETCH")
A_DT=$(jq  -r '.after.meta.datetime' <<<"$FETCH")

[[ "$A_DT" == "2025-03-15T08:38:09Z" ]] || { echo "After STAC mismatch: $A_DT"; exit 1; }

# 3) detect_wildfire を invoke
RES=$(curl -sf -X POST http://localhost:7860/api/tool/invoke \
  -H 'Content-Type: application/json' \
  -d "{\"tool_name\":\"detect_wildfire\",\"arguments\":{\"which\":\"after\"},
       \"before_key\":\"$B_KEY\",\"after_key\":\"$A_KEY\"}")

FIRE=$(jq -r '.observation.fire_detected' <<<"$RES")
DT=$(jq   -r '.observation.sentinel.datetime' <<<"$RES")

[[ "$FIRE" == "true" ]] || { echo "expected fire=true, got $FIRE"; exit 1; }
[[ "$DT"   == "2025-03-15T08:38:09Z" ]] || { echo "wildfire SimSat mismatch: $DT"; exit 1; }

echo "smoke: OK"
```

これで `detect_wildfire` ↔ `eval_wildfire_hf_simsat.py` の同条件 invariant を CI で守れる。

---

## 7. Phase 6 — Documentation 整備

### 既存に追加 / 改訂
- `README.md` — Quickstart 5行 (`bootstrap.sh` → `docker compose up` → `run_app.sh`)
- `docs/REPRO_PLAN.md` (本ファイル — 完了したら章ごとに ✓ を入れる)
- `docs/SIMSAT_SETUP.md` (新規) — SimSat URL の入手 / mock 起動方法
- `docs/EVAL.md` (新規) — `eval_wildfire_hf_simsat.py` のフラグ意味、期待値 (test recall 0.933)
- `docs/INTEGRATION_LFM2VL.md` (feat/toolcall 由来、merge 時にそのまま) — Docker pin 手順、sed patch の根拠
- `docs/DATASET_V3.md` (同) — DM3 v3 dataset
- `.env.example` (新規) — 全 env のリスト + 取得方法のコメント

---

## 8. 実行順序サマリ

1. ✅ (本ファイル)
2. Phase 1: branch merge — まず実施 (整合性が崩れる前に)
3. Phase 2: `.gitignore` 拡張 + metadata yaml の整理 + smoke 用 mini-cache zip
4. Phase 3: `serve_lfm2/Dockerfile` 抽出 + `docker-compose.yaml` 追加
5. Phase 4: shell 7本を `scripts/` に追加
6. Phase 5: `smoke_test.sh` を CI (GitHub Actions) で回す
7. Phase 6: README / docs 仕上げ

---

## 9. 将来のクリーンアップ (Phase 7, optional)

vLLM 1 backend 化は eval 再現を犠牲にしないと達成不可 (Phase 3 設計判断参照)。後日トライするなら:

- **7.1 wildfire LoRA を VL 部分込みで正しく re-merge** → vLLM merged ckpt 化を再挑戦
  - 現 `LFM2.5-VL-450M-Wildfire-merged` は merge 不完全 (FireEdge recall 0.0)
  - peft `merge_and_unload()` の VL handling、もしくは元 merge スクリプトの調査が必要
  - 成功すれば serve_lfm 廃止 → vLLM 1 image 2 process (1 docker-compose service にまとめ可)
- **7.2 vLLM 上流に LFM2-VL の `multi_modal_projector` LoRA サポート PR**
  - 現 `vllm/lora/lora_model.py:check_unexpected_modules` が hardcode で reject
  - `vllm/model_executor/models/lfm2_vl.py` 側で projector 層を LoRA target に登録する必要
  - mainline merge されれば `--lora-modules` 経路で 1 process 化可能

---

## 10. 未解決事項 (要確認)

- **モデル配布**: `LFM2.5-VL-450M-sft-grpo` S64 checkpoint をどこに置く?
  - 候補: HuggingFace Hub にアップ / Kaggle dataset / Release asset (LFS)
  - サイズ ~900MB → HF Hub が現実的
- **FireEdge LoRA**: `LiquidAI/LFM2.5-VL-450M-wildfire` は YujiYamaguchi 公開済み → そのまま参照可
- **SimSat の公開可否**: 別チーム所有なので mock を書くか、本番 endpoint を docs にだけ書くか
- **CI コスト**: GPU が要るので GitHub-hosted runner では smoke test 完走しない。self-hosted runner or 抜粋テスト (no-GPU の API 単体) に絞るか
- **Branch 名**: 統合後は `Branch/refactor` を `main` に rebase すべきか、`develop` を新設するか
