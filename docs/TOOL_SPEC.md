# ツール仕様

エージェントのツール層は VLM (LFM2.5-VL 等) が呼び出す関数の集合。本ドキュメントはチーム全員の **single source of truth**。

- スキーマ定義: [`tools/schema.py`](../tools/schema.py) の `TOOL_SCHEMAS`
- 終端ツール: [`tools/schema.py`](../tools/schema.py) の `TERMINAL_TOOLS = {submit_to_ground, drop}`
- Phase 1 モック実装: [`tools/stubs.py`](../tools/stubs.py)
- 本実装: [`tools/spectral.py`](../tools/spectral.py), [`tools/vision.py`](../tools/vision.py), [`tools/scorer.py`](../tools/scorer.py), [`tools/quality.py`](../tools/quality.py), [`tools/classifier_*.py`](../tools/)

## 1. 設計原則 (Invariants)

1. **数値演算はツール側で**: VLM は浮動小数点演算をしない。`compute_area` / `compute_index` / `estimate_size` / `check_downlink_budget` のような数値計算は決定論的にツールが処理する。
2. **読み取り系ツールは冪等**: 同じ入力は何度呼んでも同じ出力 (副作用なし)。失敗時のリトライが安全。
3. **終端アクションは 1 回のみ**: `submit_to_ground` または `drop` のいずれかで ReAct ループは必ず終了する。
4. **VLM は画像、ツールは数値**: VLM は人間が見る画像 (PNG) で判断する。spectral index 等の数値解釈はツールが実施し、結果を VLM に dict で返す。
5. **case コンテキストはサーバ側でバインド**: lat/lon/size_km/timestamp 等の case 固有メタは `make_*` ファクトリでクロージャに閉じ込め、VLM には公開しない。VLM が見る引数は `band` / `index` / `which` などの「何を計算するか」のみ。

## 2. ツール一覧

凡例: 終端 = ReAct loop 終了 / impl = 本実装ファイル / data = データソース / offline = Kaggle offline 動作可否 (✓=可 / △=事前データ準備で可 / ✗=不可)

| # | カテゴリ | ツール | 概要 | 入力 (引数: 型, 必須) | 出力 (戻り値) | 終端 | impl | data | offline |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Vision | `classify_change` | LFM2-VL で before/after を分類しクラス候補 + bbox を返す | `image_before:str`✓, `image_after:str`✓ | `{candidates:[{class, confidence, bbox}, ...]}` | | classifier_gemini/openai | LLM API | ✗ |
| 2 | Vision | `fetch_band` | Sentinel-2 単一バンドをグレースケール PNG で取得 (band 名 = `coastal/blue/green/red/rededge1-3/nir/nir08/nir09/swir16/swir22/aot/scl/visual/wvp`) | `band:str`✓, `which:str`("before"/"after"=after) | `{png_path, stats:{min,max,mean}}` | | spectral.py `make_fetch_band` | Sentinel-2 多バンド | △ |
| 3 | Vision | `false_color` | 任意 3 バンドで RGB 合成 (例: `nir-red-green` 植生 / `swir22-nir-red` 焼失重度) | `bands:str[3]`✓, `which:str`(=after) | `{png_path}` | | spectral.py `make_false_color` | Sentinel-2 多バンド | △ |
| 4 | Vision | `compute_index` | 標準 spectral index (`NDVI/NDWI/MNDWI/NBR/NDBI/NDSI`) を計算しヒートマップ + 統計 | `index:str`✓, `which:str`(=after) | `{png_path, stats:{min,max,mean,std}}` | | spectral.py `make_compute_index` | Sentinel-2 多バンド | △ |
| 5 | Vision (未登録) | `compute_index_delta` | before/after の index 差 `Δ=After-Before` を計算 (赤=減少 / 青=増加) | `index:str`✓ | `{png_path, delta_stats:{frac_strong_decrease, frac_strong_increase, mean, median}}` | | spectral.py `make_compute_index_delta` (※スキーマ要追加) | Sentinel-2 × 2 時点 | △ |
| 6 | Vision | `zoom_in` | bbox を 512×512 に LANCZOS アップサンプル (before/after 両方) | `bbox:int[4]`✓ `[x,y,w,h]` | `{before_png, after_png}` | | vision.py `make_zoom_in` | ローカル PNG | ✓ |
| 7 | Context | `get_region_info` | lat/lon → 地域名・国・人口分類・近隣インフラ | `lat:num`✓, `lon:num`✓ | `{region, country, population, infra_nearby:[...]}` | | (未着手) | reverse geocoding | △ |
| 8 | Context | `get_history` | 当該地点の過去 N 日の onboard 報告履歴 | `lat:num`✓, `lon:num`✓, `days:int`(=30) | `[{timestamp, report_id, summary}, ...]` | | (未着手) | history DB or state | ✓ |
| 9 | Context | `compute_area` | bbox を地理空間 km² に換算 | `bbox:int[4]`✓ | `{area_km2:float}` | | (純粋数式) | なし | ✓ |
| 10 | Budget | `check_downlink_budget` | 残ダウンリンクバイト / 残時間 | (なし) | `{bytes_remaining:int, seconds_until_window_close:int}` | | (未着手) | rollout state | ✓ |
| 11 | Budget | `estimate_size` | report の送信バイト数見積もり | `report_id:str`✓, `with_image:bool`✓ | `{bytes:int}` | | (未着手) | state + length | ✓ |
| 12 | Action | `compose_report` | 報告書ドラフト作成 (送信はしない) | `change_type:str`✓, `urgency:int(0-10)`✓, `description:str`✓, `attach_image:bool`(=false) | `{report_id:str}` | | (state のみ) | state | ✓ |
| 13 | Action | `submit_to_ground` | 作成済み報告を地上に送信 | `report_id:str`✓, `attach_image:bool`✓ | `{status:"ok", report_id, attached:bool}` | ✓ | (state のみ) | state | ✓ |
| 14 | Action | `drop` | 何も送信せず破棄 | (なし) | `{status:"dropped"}` | ✓ | (state のみ) | なし | ✓ |

スキーマ未登録の補助実装 (将来 `TOOL_SCHEMAS` に追加するか整理する候補):
- `get_change_stats` ([scorer.py](../tools/scorer.py)): 4 指数 (NBR/NDVI/MNDWI/NDBI) の delta 統計を一括返却
- `capture_crop` ([vision.py](../tools/vision.py)): bbox の切り出しユーティリティ
- `assess_image_quality` ([quality.py](../tools/quality.py)): 画像品質指標 (cloud_proxy, brightness, edge_density)

## 3. 実装の差し替え戦略 (Phase 1 → 本番)

Phase 1 では `tools.stubs.STUB_TOOLS` がすべてのツールを dummy で提供する。本実装はスキーマを保ったまま category 別ファイルに drop-in:

```python
from tools.stubs import STUB_TOOLS
from tools.spectral import make_compute_index, make_compute_index_delta
from tools.vision import make_zoom_in

# RL env の setup_state で case context をバインドして registry を組む
def build_registry(case_meta, before_path, after_path):
    return {
        **STUB_TOOLS,
        "compute_index":       make_compute_index(**case_meta),
        "compute_index_delta": make_compute_index_delta(**case_meta),
        "zoom_in":             make_zoom_in(before_path, after_path),
        # ...
    }
```

`satelliteagent_env._build_env_class` の `update_tool_args` でも同様に case-bound tool factory を使える。

## 4. オフライン実行のために集めるデータ

Kaggle offline で全ツールを動かすには、`eval/precompute_tool_responses/<case_id>/` 以下に下記を case 毎に事前生成する。zoom_in / 純粋数式系 / state 系は precompute 不要 (on-call で動く)。

| ツール | 生成物 | 1 case あたり | 67 case 合計の目安 |
|---|---|---|---|
| `classify_change` | `classify_change.yaml` (`{candidates:[{class, confidence, bbox}]}`) ※ online VLM で 1 回計算 or dummy or ground-truth hardcode | 1 YAML | 67 |
| `fetch_band` | `bands/<band>__{before,after}.png` + `bands/<band>__{before,after}.stats.yaml` (16 bands × 2 時点) | 32 PNG + 32 YAML | 約 2,150 PNG |
| `false_color` | `false_color/<r>_<g>_<b>__{before,after}.png` (代表 5 組合せ × 2 時点) | 10 PNG | 約 670 PNG |
| `compute_index` | `index/<NAME>__{before,after}.png` + `.stats.yaml` (6 index × 2 時点) | 12 PNG + 12 YAML | 約 800 PNG |
| `compute_index_delta` | `index_delta/<NAME>.png` + `.delta_stats.yaml` (6 index) | 6 PNG + 6 YAML | 約 400 PNG |
| `get_region_info` | `region_info.yaml` (`{region, country, population, infra_nearby}`) ※ online reverse geocoding | 1 YAML | 67 |
| `get_history` | (env state で持つので precompute 不要) | — | — |

**precompute 不要 (on-call で動く)**:
- `zoom_in`: 既存 visualize 済 PNG (before/after) から実時間切り出し
- `compute_area`: 純粋数式 (size_km と画像解像度から計算)
- `check_downlink_budget` / `estimate_size` / `compose_report` / `submit_to_ground` / `drop`: rollout state 操作のみ

**規模感**: 約 4,000 ファイル / 500MB-1GB。precompute スクリプト (`eval/scripts/build_precompute.py` を新規作成) で online 環境で 1 回生成し、`titanic12/satelliteagent-precompute-v1` として Kaggle Dataset 化する。

## 5. テスト

各ツールは `tools/schema.py` の JSONSchema に対するテストでカバー予定 (TODO):
- 入力 schema validation
- 戻り値 shape 検証
- stub と本実装で同じ shape を返すこと
