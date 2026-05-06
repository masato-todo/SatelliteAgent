# 実験計画

旧 DM3 ベースの構成 (Phase 1〜6) は、xBD per-image を 10m Sentinel-2 で使うと AOI が重複して unique scene が実質 10件以下にしかならず scale しないことが判明したため破棄。新 Phase 1 から再設計する。

> 設計原則: **10m S2 で判別可能な広域変化** (burn scar / flood inundation / clear-cut 等) を **全球規模** で集める。建物単体スケール (~10m 以下) の damage は使わない。

---

## Phase 1 — Disaster scene catalog discovery

**目的**: 10m Sentinel-2 で判別可能な disaster scene を全球規模で発見し、`data/scene_catalog.yaml` に永続化。後続 phase 全ての single source of truth とする。

**現状**: 未着手 (旧 DM3 ベース収集は破棄)

### ソース優先度

| 順位 | ソース | カバー | 単位 | 備考 |
|---|---|---|---|---|
| 1 | **MCD64A1** (MODIS Burned Area) | 全球月次 | 500m burn polygon | wildfire 主軸、年数万件 |
| 2 | **NASA FIRMS** (VIIRS/MODIS active fire) | 全球日次 | point + FRP | MCD64A1 が取れない最近 event を補完 |
| 3 | **Copernicus EMS** rapid mapping | 国際支援要請 only | event polygon | flood/storm/landslide の多様性、年数百件 |
| 4 (option) | Hansen Global Forest Change | 全球年次 | 30m loss pixel | 山火事と区別困難なら後段 |

### フィルタ条件

- 期間: **2017-01 以降** (Sentinel-2 globally cover 完了)
- 災害域 **≥1km²** (10m S2 で十分判別、AOI 10km に内包)
- 隣接 event 間距離 **≥20km** (AOI 重複回避のクラスタ間引き)
- (option) 雲被覆少ない地域・季節を優先

### ツール / スクリプト

`scripts/build_scene_catalog.py` (新規):
- MCD64A1: NASA Earthdata 経由で月次 HDF を download → polygon 抽出 → filter
- FIRMS: CSV API で active fire points → DBSCAN クラスタリング → 仮 polygon
- EMS: WMS/WFS or static archive scrape
- 出力: `data/scene_catalog.yaml` + `data/gt_polygons/<id>.geojson`

### 出力スキーマ

```yaml
scenes:
  - id: fire_au_blackforest_20200103
    event_type: wildfire        # wildfire | flood | storm | landslide | volcanic | deforestation
    lat: -35.123
    lon: 138.567
    event_period: [2020-01-03, 2020-01-12]
    affected_area_km2: 14.2
    source: MCD64A1
    gt_polygon_uri: data/gt_polygons/fire_au_blackforest_20200103.geojson
    notes: ""
```

### スケール目安

- 初期: wildfire 主軸 ~500 scenes
- 追加: EMS で flood/storm/landslide ~200 scenes
- 合計目標 **700 scenes** (旧 DM3 80件比 9倍、unique 比だと数十倍)

### 成功基準

- `scene_catalog.yaml` に 500+ entries
- 各 scene について GT polygon が GeoJSON で参照可能
- 抽出 10 件のスポット確認で S2 Before/After に **NBR Δ > 0.2** の signal が見える

### 前提セットアップ

- NASA Earthdata 無料アカウント (Earthdata Login OAuth)
- 依存: `earthaccess`, `geopandas`, `rasterio` (pyproject `geo` extra で対応済)

### 残タスク

- [ ] Earthdata token 取得手順をドキュメント化
- [ ] `build_scene_catalog.py` 骨格実装
- [ ] **smoke test**: 1 burn scar を S2 で実画確認 → signal 実在を実証 (実装前にやる価値高い)
- [ ] スケール検証: filter 後の残数を実測

---

## Phase 2 — Curate Before/After pairs

**目的**: Phase 1 で得た scene catalog (positive: MCD64A1 wildfire, negative: `negative_cases.yaml`) に対して、S2 で signal 判定可能な Before/After 日付ペアを確定し、`canonical_dataset.yaml` + `data/curated_pairs/<scene_id>/{before,after}.png` に永続化。後続 Phase 5/6 はこれを唯一の入力とする。

**現状**: 進行中 (positive 51/60, negative 16/16 確定済 ≒ 67 entries)

### 入力

- `data/scene_catalog.yaml` (Phase 1 の MCD64A1 wildfire 60 scenes、event_period が pixel-DOY から正確に抽出済)
- `data/metadata/disaster_m3/negative_cases.yaml` (人手curated 16 scenes、no_change / cloud_blocked)

### 出力

- `data/canonical_dataset.yaml` — 各 scene について `(lat, lon, size_km, request.before_date, request.after_date, expected_resolved_datetime, type, expected_action)` を記録
- `data/curated_pairs/<scene_id>/before.png + after.png + meta.yaml` — GRPO 学習が直接 Image.open する canonical 画像
- `data/scenarios/<key>.png` — SimSat fetch cache (副産物、shared)

### 候補日付の探索方針

**Positive (MCD64A1 catalog)**
- AOI: **10 km × 10 m** (probe = final、別解像度の finalize なし)
- Before anchor: `event_start - {14, 30, 60, 90}d`
- After  anchor: `event_end   + {0, 7, 14, 21, 30}d`
- Selection: usable=true & cloud_proxy<0.30 & nodata<0.20 のうち `cloud + nodata` 最小

**Negative (no_change / cloud_blocked)**
- AOI: 10 km × 10 m
- 日付: `negative_cases.yaml` 固定 (人手選定済)
- Selection: image_available のみ要求 (`usable` フラグは disaster 検出向けで負例には不適切)
  - ただし nodata < 0.30 は要求 (タイル境界に大半が外れた scene を除外)
  - cloud_blocked タイプは雲が要件なので cloud check 完全 skip

### ツール

| ツール | 用途 |
|---|---|
| `scripts/auto_phase2.py` | 旧 DM3 cases (xBD/BRIGHT) 用の自動探索 (legacy) |
| `scripts/auto_fill_pairs.py` | **MCD64A1 + Negative 一括処理** (本phase主軸) |
| UI **`Save Before/After`** ボタン | 1 case ずつ目視確認後に curate |
| UI **`Use cache`** ボタン | 確定済 pair の即時再表示 |

UI には保存状態をマーカーで表示:
- **★** = canonical 確定済 (`canonical_dataset.yaml` にエントリあり)
- **●** = この lat/lon に cache 2 件以上 (= Before/After 揃う可能性)
- **◐** = cache 1 件 (片側のみ)

### 成功基準

- 全 catalog scene について canonical entry 存在 (`★` フル充足)
- positive: 焼け跡 polygon が AOI 内に収まっており、After で植生反射が変化
- negative: 期待通り「変化なし」or「雲で判定不能」が視覚的に確認できる
- 失敗 case (Sentinel-2 カバレッジ穴等) は理由付きで catalog から除外

### SimSat タイル境界対応

旧計画では「nodata>50% を probe で弾く」だったが、**SimSat fork の `sentinel_provider.py` を mosaic 対応に修正済** (`odc.stac.load([best_items], groupby="solar_day")`)。AOI が MGRS 境界をまたいでも同日付の隣接タイルから合成 → nodata≈0 で取得可能になった。これにより検出された焼け跡が AOI 端にあるケースでも救えるようになっている。

### Negative の品質判定が positive と異なる理由

`tools/quality.py` の `usable` フラグは `cloud_proxy<0.5 AND dark_fraction<0.10 AND edge_density>2.0` を要求 (= disaster 検証画像として最適化)。砂漠・海・雲被覆など **意図的に "退屈" な negative scene** はこれを満たさないので、Phase 2 での負例 fetch は専用ロジックで緩める ([auto_fill_pairs.py:process_negative](../scripts/auto_fill_pairs.py))。

### 残タスク

- [ ] auto_fill_pairs FAIL の 9 件を手動確認 (Sentinel coverage 不足 / 別 anchor 試行)
- [ ] catalog scene の `affected_area_km2` が 1 km² ぴったりの境界値を再評価
- [ ] negative の地理多様性: 現状 16 件 → 目標 30+ で `random` 系 (海洋/極地) 自動追加検討

---

## Phase 3 — SFT trace collection (Gemini-as-teacher)

**目的**: SFT warmup の学習データとして、canonical 全 scene に対する **完全な ReAct trace** (thought → action → observation → ... → final) を Gemini-2.5-flash で収集。LFM2.5-VL-450M (素) は tool calling が壊滅的なので、Gemini を教師として trace の **形式** を学ばせる distillation 路線。

**現状**: 完了 (2026-04-26)、67 scene × 1 replica の trace を保存 + baseline accuracy 確定。

### なぜ Gemini-2.5-flash か

| 項目 | 値 |
|---|---|
| ReAct 実走可否 | ✓ (Phase 1〜2 動作確認済) |
| 価格 | 無料枠 250 req/日 (67 scene × 数 step ≒ 200-300 calls、ほぼ枠内) |
| 速度 | 1 trace 平均 25 秒 (実測) |
| 精度 | 76.5% (実測、後述) |

**精度は怪しいことを許容する**:
- SFT の主目的は **tool 呼び順 + 引数 JSON 形式の獲得** であって正答率ではない
- 後段 GRPO で reward signal により正答率は補正される
- ただし全 trace を盲信せず、`expected_action` 不一致 trace は SFT から除外 or weight 下げる前提

### 前提条件

- Phase 1, 2 完了済 (`data/canonical_dataset.yaml` に 67 entries)
- `~/.env` または環境に `GOOGLE_API_KEY` 設定 (https://aistudio.google.com/apikey で取得)
- SatelliteAgent サーバ稼働中 (port 7860)
- `config/providers.yaml` に gemini provider 定義済
- SimSat 稼働中 (cache miss 時の Sentinel-2 取得用、ほぼ HIT するので無くても可)

### 再現手順

```bash
# 1. サーバ起動
cd ~/work/SatelliteAgent
APP_HOST=0.0.0.0 nohup .venv/bin/python -m app.server > server.log 2>&1 &

# 2. (必要なら) curated_pairs の cache が落ちている場合は scripts/auto_fill_pairs.py 再実行
uv run python scripts/auto_fill_pairs.py    # 既存はスキップ

# 3. trace 一括収集 (~30 分)
uv run python scripts/collect_agent_traces.py
```

### スクリプト構成 (`scripts/collect_agent_traces.py`)

各 canonical entry に対して:
1. `/api/fetch` を叩いて Before/After を確実に cache (skip if HIT)
2. `/api/run_agent?scene_id=<id>&provider=gemini&model=gemini-2.5-flash` を SSE で消費
3. server 側が `data/traces/agent/<scene_id>__YYYYMMDDTHHMMSSZ.yaml` に **auto-save**
4. final.action と canonical の `expected_action` を比較し OK / MISS をログ出力

オプション:
```bash
uv run python scripts/collect_agent_traces.py --limit 5         # smoke test
uv run python scripts/collect_agent_traces.py --replicas 3      # per-scene 多様性
uv run python scripts/collect_agent_traces.py --no-skip-existing # 強制再収集
uv run python scripts/collect_agent_traces.py --only-id <id>    # 単発
```

サーバ側 auto-save は `app/server.py:_save_agent_trace` で実装。UI から Run Agent を押した場合も同じ path で保存される (ボタン押下不要)。

### 出力スキーマ

`data/traces/agent/<scene_id>__YYYYMMDDTHHMMSSZ.yaml`:
```yaml
metadata:
  scene_id: mcd64a1_h03v06_202308_p2079_-15640
  scenario_type: positive          # positive | negative
  expected_action: submit_to_ground
  expected_class: fire
  lat: 20.789
  lon: -156.404
  size_km: 10.0
  before_date: '2023-06-02'
  after_date: '2023-10-30'
  before_key: af4a2227ff
  after_key:  5b86762a9b
  provider: gemini
  model:    gemini-2.5-flash
  collected_at: '2026-04-26T02:18:33+00:00'
events:
  - {type: thought, text: "Classify the change first..."}
  - {type: action,  name: classify_change, arguments: {}}
  - {type: observation, name: classify_change, result: {classes: [{name: fire, confidence: 0.98}], bboxes: [...]}}
  ...
final:
  type: final
  name: submit_to_ground
  result: {status: ok, report_id: r-0001, attached: true, attached_crop_key: null}
gt_match: true
```

### Baseline 数値 (2026-04-26 実測)

| カテゴリ | OK | MISS | accuracy |
|---|---:|---:|---:|
| Positive (MCD64A1 wildfire) | 45 | 7 | **86.5%** |
| Negative (no_change / cloud_blocked) | 7 | 9 | **43.8%** |
| **全体** | **52** | **16** | **76.5%** |

総 trace = 68 (smoke test 1 件 + 本番 67 件)、所要 27 分 (1 trace 平均 25秒、Gemini step数 7-19 events)。

**観察**:
- Positive (wildfire) は 86.5% — Gemini は焼け跡 polygon 一致で正しく submit できる
- **Negative は 43.8% — Gemini は drop 判断が苦手、誤って submit_to_ground しがち**
- ⇒ SFT 後 / GRPO 後の主たる改善目標は **drop 判断の精度向上**

### UI 連携

- 右サイドバー `Run Agent` ボタンで単発実行 + auto-save (1 case 単位の検証用)
- `📂 Traces` ボタンで一覧 (`[AGENT]` / `[HUMAN]` バッジ + `GT✓` / `GT✗` バッジで一目で良し悪し判別)
- 各 trace 詳細から `🗑 Delete` で個別削除可能

### 成功基準 (達成済 ✓)

- [x] 全 67 canonical scene について 1 trace 以上取得
- [x] `gt_match` メタを記録、baseline accuracy 数値化
- [x] 失敗 trace も除外せず保存 (後段判断のため)
- [x] UI から手動 Run でも同じ path で auto-save される (操作整合)

### 残タスク (任意・後段で必要なら)

- [ ] `--replicas 3` で per-scene 多様性増 (336 trace ≒ 無料枠ギリ、temperature 振り分け代替案あり)
- [ ] MISS trace の人手 trace 上書き (Phase 4 = Human Annotate モード)
- [ ] SFT データセット export スクリプト (yaml → jsonl 変換 + train/val split)

---

## Phase 4 以降 — 詳細化予定

| Phase | 内容 | 状況 |
|---|---|---|
| 4 | Human Annotate (高品質 trace 上書き) | UI Recording モード既存。Phase 3 trace の bad ケースを人手修正する用途。任意 |
| 5 | SFT warmup | Gemini trace で LFM2.5-VL に tool calling + 出力形式を仕込む。LoRA で開始、別 repo (Kaggle GPU 等) |
| 6 | GRPO | reward = expected_action 一致 + 帯域効率。Phase 5 後の policy fine-tune |

---

## 横断的な懸念 (現時点)

| 項目 | メモ |
|---|---|
| Earthdata 認証 | OAuth token の管理。`.env` に置いて gitignore |
| MCD64A1 の lag | 月次配信、最新月は 2-3週遅れ。最新event 取りたい時は FIRMS で代替 |
| Sentinel-2 タイル境界 nodata | 旧 DM3 で問題化した nodata>50% を Phase 2 probe で同様に弾く |
| event_type の多様性 | wildfire 偏重だと train が "burn detector" になりがち。catalog 段で type 比率を制御 |
| GT polygon 解像度 | MCD64A1 500m 単位なので、10m S2 と比べると粗い。pixel-perfect な metric は不可、area-overlap で評価 |

---

## 関連ドキュメント

- [SPECTRAL_INDEX_GUIDE.md](SPECTRAL_INDEX_GUIDE.md): NBR/NDVI/MNDWI/NDBI の物理的意味と読み方
- [TOOL_SPEC.md](TOOL_SPEC.md): tool スキーマ
- [ARCHITECTURE.md](ARCHITECTURE.md): system 構成
- [REMOTE_DEPLOY.md](REMOTE_DEPLOY.md): リモートサーバ構築手順
