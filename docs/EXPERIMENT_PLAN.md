# 実験計画 — Phase 1〜6

DM3 / xBD を gold データソースに、LFM2-VL ベースの Tool-Using Agent を SFT → GRPO で学習するための段階構成。各 Phase の入出力・解像度方針・成功基準を明示する。

> 設計原則: **Annotate session の cache が GRPO の environment になる**。よって Phase 4 (cache build) は native 10m / 高 fidelity、Phase 1〜2 (探索) は coarse OK、Phase 6 (RL training) は cache のみ参照で SimSat 経由なし。

> Agent の action policy には **submit と drop の両方**がある。drop は「downlink を消費しない正しい判断」であり、negative scenarios (no_change, cloudy 等) を学習材料に**含めない**と GRPO で submit を多発するバイアスを学んでしまう。よって Phase 1 で先に negative を集める。

---

## Phase 1 — Negative scenario collection

**目的**: Agent が `drop()` で正しく終わるべき scenario を集める。「変化なし」「雲で判定不能」「ノイズ的な季節変動だけ」など、submit すべきでない場面のデータセットを最初に固める。

**現状**: 未着手

**Scenario types** (全部 1 つの dataset として扱う):

| Type | 構成 | 期待 final action | 件数目安 |
|---|---|---|---|
| **no_change_pre_pre** | xBD 各 lat/lon、両 Before/After が災害前 (例: 災害-180d, -30d) | **`drop()`** | 8 |
| **no_change_post_post** | xBD 各 lat/lon、両 Before/After が災害後復旧期 (例: 災害+12mo, +16mo) | **`drop()`** | 8 |
| **cloud_blocked** | After が雲で >70% 不明瞭 (`cloud_proxy > 0.7`) | **`drop()`** | 5-10 |
| (optional) random_no_change | サハラ砂漠・海洋・極地等 random 非災害地点 | **`drop()`** | 5-10 |

**Tool / UI**:
- 既存 Annotate UI を流用 (Phase 2 と同じ workflow)
- ドロップダウンに negative scenario を出すために server.py 側で **疑似 case を canonical_dataset に append** する必要あり
- Find clearer Before/After で雲量制御 (cloud は逆に **雲が多い候補**を選ぶ)

**解像度方針**:
- `auto_resolution_meters(size_km)` (Phase 2 と同じ coarse 設定)

**成功基準**:
- 各 negative type の date pair が SimSat で取れる
- no_change 系: 全 index `frac_strong_decrease/increase < 5%` を確認
- cloud 系: After の `cloud_proxy > 0.7` を確認
- 全 ~30 sub-case が「drop が正解」と人間判断可能なシグナル分布を持つ

**残タスク**:
- xBD 8 イベントごとに pre_pre + post_post の date pair を発見 (16 件)
- cloud_blocked: santa_rosa post_disaster の cloudy day (例: 2017-10-15) のような既知の悪天日を 5-10 件
- (optional) random non-disaster 5-10 件

---

## Phase 2 — Positive (xBD) date discovery

**目的**: DM3 各イベントで使える Before/After 日付を見つける。SimSat が S2 シーンを返せて、雲が薄く、災害シグナルが見えるペアを確定。

**現状**: 進行中 (santa_rosa, palu_tsunami, hurricane_florence で動作確認済)

**Tool / UI**:
- DisasterM3 case dropdown (◆ 49 precise / ◇ 41 coarse)
- Find clearer Before/After candidates (event_start/end をアンカーに ±N日)
- Footprint slider 50km デフォルト
- get_change_stats / compute_index_delta で signal 確認

**解像度方針**:
- `auto_resolution_meters(size_km)` (50km → 30m, 30km → 20m, 10km → 10m)
- 速度優先。**fidelity 不問**(Phase 4 で再 fetch する)

**成功基準** (各 8 イベントで):
- Before/After 両方 SimSat available=true
- After は disaster_period より後で雲≤30%
- 該当 disaster_type の主要 index で `frac_strong_decrease/increase > 5%`

**残タスク**:
- 残り 6 イベント (hurricane_harvey/florence/michael, midwest_flooding, socal_fire, guatemala_volcano) で良い Before/After ペアを発見

---

## Phase 3 — Canonical pair lock

**目的**: Phase 1 で発見した positive + negative ペアを `data/canonical_dataset.yaml` に**永続化**。1ケース1構成として、replay 時に毎回同じシーンが取れることを保証。

**入力**: Phase 1 の探索結果 (各ケースの best positive + 2 negative date sets)

**出力**: `data/canonical_dataset.yaml`

```yaml
# 例
cases:
  - id: xbd_santa_rosa_wildfire_00000063
    label: fire
    type: positive
    lat: 38.4735
    lon: -122.7491
    size_km: 50
    request:
      before_date: 2017-06-21
      after_date:  2017-11-06
      window_days: 30
    expected_resolved:
      before_datetime: 2017-06-19T18:57:28Z
      after_datetime:  2017-11-06T18:55:55Z
    event:
      name: Tubbs Fire (Northern California)
      period: [2017-10-08, 2017-10-31]

  - id: xbd_santa_rosa_wildfire_00000063__neg_pre
    label: no_change
    type: negative_pre_pre
    lat: 38.4735
    lon: -122.7491
    size_km: 50
    request:
      before_date: 2017-04-15
      after_date:  2017-08-15
      window_days: 30
    note: "両方 disaster 前 (2017-10-08 より前)、季節変化のみのコントロール"

  - id: xbd_santa_rosa_wildfire_00000063__neg_post
    label: no_change
    type: negative_post_post
    lat: 38.4735
    lon: -122.7491
    size_km: 50
    request:
      before_date: 2018-06-21
      after_date:  2018-10-21
      window_days: 30
    note: "両方 disaster 後 (2018年、復旧期)、植生回復の季節変動のみ"
```

**Dataset 構成** (Phase 1 で negative、Phase 2 で positive を集めた結果):

| タイプ | 由来 | expected action | 数 |
|---|---|---|---|
| **disaster_event** (positive) | xBD precise (1ケースずつ TOP damaged) | `submit_to_ground(class)` | ~49 |
| **no_change_pre_pre** | xBD lat/lon、両 Before/After 災害前 | `drop()` | ~8 |
| **no_change_post_post** | xBD lat/lon、両 Before/After 災害後復旧期 | `drop()` | ~8 |
| **cloud_blocked** | After が雲 >70% | `drop()` | ~5-10 |
| (optional) easy_negatives | random 非災害地点 (砂漠・海洋等) | `drop()` | ~5-10 |

合計 ~75-85 ケース。positive : negative ≈ 1 : 0.5〜0.7 (シンプル設定)。

**生成方法**:
- Option A: 人間が UI で全 sub-case 確認 → yaml 手書き (品質高、時間大)
- Option B: スクリプト `scripts/build_canonical_dataset.py` で
  - Phase 1, 2 の探索結果を集約
  - SimSat probe で各日付の resolved_datetime を確定
  - signal 妥当性チェック (positive は signal あり、negative は signal なし、cloud は cloud_proxy>0.7)
  - 自動 yaml 生成
- 推奨: **B で叩き台 → A で問題ケースを修正**

**成功基準**: 全 ~80 ケースで再現可能な (lat, lon, dates, size_km, expected_resolved_datetime, expected_action) が yaml に記録されている。

---

## Phase 4 — High-resolution cache build

**目的**: canonical yaml を読み、**全ケース全派生画像を native 10m でローカルに pre-fetch**。Phase 5 / 6 では SimSat に一切触らない状態を作る。

**入力**: `data/canonical_dataset.yaml`

**Tool**: `scripts/prewarm_cache.py` (新規、~80行)

```bash
python scripts/prewarm_cache.py \
    --dataset data/canonical_dataset.yaml \
    --resolution 10 \
    --indices NBR,NDVI,MNDWI,NDBI \
    --workers 4
```

処理内容 (各ケース):
1. Before/After RGB PNG (SimSat fetch、10m)
2. Before/After 4 bands array (NDVI/NBR/MNDWI/NDBI 計算用)
3. compute_index 4種 × 2 sides = 8 index PNG
4. compute_index_delta 4種 = 4 delta PNG
5. cache key + sidecar JSON 永続化

**解像度方針**: **常に 10m** (auto_resolution の override)

**容量目安**:
- 1 ケース ≈ 10m PNG ×3 (RGB + 4 idx ×2 + 4 delta) = 11 PNG
- 50km @ 10m PNG ≈ 30-50MB/枚 (圧縮後)
- 11 × 40MB ≈ 450MB / ケース
- **80 ケース × 450MB ≈ 36GB** ⚠

→ 容量妥協案:
- (a) サイズ落とす: **30km @ 10m** に変更 (3000×3000、~15MB/枚 → 12GB total) ← 推奨
- (b) cache を圧縮 (PNG → WebP lossy) → 半減
- (c) array bands は npz 圧縮 → 1/3

**成功基準**:
- canonical 全ケースが 10m cache に存在
- replay スクリプトで cache だけから get_change_stats / compute_index_delta が走る (SimSat 切断状態でも OK)

---

## Phase 5 — Human Annotate (gold trace 収集)

**目的**: Phase 4 cache を使って **各 canonical ケースで人間 gold trace を1〜N本録画**。Recording mode で reasoning 込みで記録。

**入力**: Phase 4 cache + canonical yaml

**Tool**: 既存の Annotate UI (右サイドバー Recording)

**作業 (positive case)**:
- ドロップダウンでケース選択 → Fetch (cache hit、即時)
- get_change_stats → 数値確認
- compute_index_delta → 視覚確認
- (必要なら) classify_change で VLM 比較
- bbox draw で attention event
- pan/zoom で view event
- **Submit Report** → Stop Recording → 自動保存

**作業 (negative case = no_change_pre_pre / post_post / cloud_blocked)**:
- get_change_stats で全 strong↓/↑ 弱い (cloud は cloud_proxy>0.7) 確認
- Thought に「季節変動のみ」or「After は雲に被覆されてて判定不能」
- **Drop ボタン** → trace に `final.action: drop` が記録される
- Stop Recording → 自動保存
- 短い trace (3-5 events) で OK、それも"drop と判断する正しい手順"の学習材料

**出力**: `data/traces/human/*.yaml` (1ケース 1〜数本)

**成功基準**: 全 ~80 ケースで trace 1本以上、final action が canonical の `expected_action` (submit or drop) と一致。

---

## Phase 6 — GRPO training

**目的**: trace + cache を使って LFM2-VL agent を SFT (warmup) → GRPO 学習。

**入力**:
- `data/traces/human/*.yaml` (gold)
- `data/scenarios/*.png` + sidecar (environment)
- `data/canonical_dataset.yaml` (case index + GT label + expected_action)

**Stages**:
1. **SFT warmup**: trace の (state, action) ペアで模倣学習。Tool 呼び順を学ぶ。
2. **GRPO**: 自由ロールアウト。reward = expected_action 一致 + bandwidth efficiency penalty。
3. **Eval**: held-out canonical ケースで accuracy 測定。

**インフラ**: 別レポジトリ予定 (Kaggle GPU pod など)。本リポジトリは trace + cache の export のみ責任。

**Reward 関数 (案)**:
```
# Positive case (GT.expected_action = submit_to_ground)
reward = + 1.0  if final.change_type == GT.mapped_class    (correct class)
       - 0.3  if final.change_type ≠ GT.mapped_class       (wrong class, submitted)
       - 1.0  if final.action == drop                       (false negative, missed disaster)

# Negative case (GT.expected_action = drop)
reward = + 1.0  if final.action == drop                     (correct silence)
       - 0.5  if final.action == submit_to_ground          (false positive, wasted bandwidth)

# Both
       - 0.2 × n_tools_called    (efficiency penalty)
       - 0.1 × attach_image      (bandwidth penalty)
```

**成功基準**: held-out で 70%+ action correctness & n_tools_called ≤ trace median × 1.5。

---

## 横断的な懸念

| 項目 | 対処 |
|---|---|
| **cache 削除事故** | `data/scenarios/` を `.gitignore` 入りつつ、**canonical yaml** だけを repo 管理 → cache が消えても Phase 4 で再生成可能 |
| **STAC catalog drift** | Phase 3 で記録した `expected_resolved_datetime` と Phase 4 で実際取れる datetime を比較 → ズレたら manifest 更新 (年単位の保守) |
| **負ケース偏り** | pre_pre / post_post / cloud_blocked / random を散らして collect、特定パターンへの過学習を防止 |
| **classify_change の VLM 依存** | 学習中は Gemini 不在の前提 → trace の classify_change observation を frozen として使う or 学習 reward から除外 |
| **解像度ミスマッチ (Annotate vs RL)** | Phase 5 と Phase 4 を厳密に同一 cache key で動かす。Annotate 時に Phase 4 cache hit してることを画面に表示 (`(cached, full 10m)` 等) |

---

## 進め方の提案 (短期 1-2週間)

1. **Phase 1 (Negative collection)**: xBD 8 イベントで negative_pre_pre + negative_post_post 各1ペア + cloud_blocked 5-10件 を annotate UI で発見 (~1日)
2. **Phase 2 (Positive xBD discovery)** 完了: 残り 6 イベントで positive ペアを発見 (~1日、既に santa_rosa/palu/florence 確認済)
3. **Phase 3 + Phase 4 を統合スクリプト化** (`scripts/build_and_warm.py`、~200行)
   - Phase 1, 2 結果から canonical yaml を auto 生成
   - 即座に 10m 解像度で pre-warm
   - 失敗ケースは log → 手動修正フィードバック
   - 30km @ 10m で容量 ~12GB
4. **Phase 5 を本格的に**: ~80 trace を 1 週間かけて録画 (1 ケース 5-10 分、positive は分析厚く、negative は drop で短く)
5. **Phase 6** は別タスク (Kaggle GPU 用、別 repo)

---

## 関連ドキュメント

- [ARCHITECTURE.md](ARCHITECTURE.md): 全体システム構成
- [SPECTRAL_INDEX_GUIDE.md](SPECTRAL_INDEX_GUIDE.md): index の物理的意味と読み方
- [TOOL_SPEC.md](TOOL_SPEC.md): tool スキーマ
- [EVAL_DATA_DESIGN.md](EVAL_DATA_DESIGN.md): 評価データ設計 (本ドキュメントと役割整理が必要 — TODO)
