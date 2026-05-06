# feat/toolcall — Local vLLM (LFM2.5-VL) ReAct integration

このブランチで `feat/toolcall` として実装した内容と、現時点で残している
HACK / TODO のメモ。

## 1. リリースした機能

### 1.1 ローカル vLLM (LFM2.5-VL-1.6B) を既定 VLM プロバイダ化
- `config/providers.yaml`
  - `lfm25_vl_local` を先頭・`default: true` に。
  - `base_url: http://127.0.0.1:8002/v1` (hermes tool parser, max-model-len 16384)。
  - 嘘のメタデータが入っていた `lfm25_vl_local_450m` エントリを削除。
- `app/server.py::_resolve_provider_cfg`
  - 既定 / 不明 provider を silent fallback せず **明示的に 400/503** を返す。
  - Gemini を選んでいるが `GOOGLE_API_KEY` 未設定の場合も 503。

### 1.2 OpenAI 互換 ReAct ループ (`agent/react_loop_openai.py`)
- vLLM の OpenAI 互換 chat completions エンドポイント経由で実行。
- **`forced_tool_steps` (既定 2)**: 最初の `forced_tool_steps` 回のターンだけ
  `tool_choice="required"` にし、それ以降は `"auto"`。
  以前の `ready_to_decide` フェーズ遷移は撤廃。
- **Operator instructions**: `run_react_openai(..., user_instructions=...)`
  でユーザ自由記述指示を user メッセージ末尾に連結。
- **ツールカタログ**: system prompt に `_tool_catalog_block()` を埋め込み、
  ツール名・必須引数・mandatory call sequence を明示。
- **Terminal の error 再ループ**: `submit_to_ground` などが
  `{"status":"error", ...}` を返したターンは終了せず、observation として
  モデルに見せて続行。

### 1.3 捏造防止ガード
- **report_id 必須化** (`tools/stubs.py`)
  - `compose_report` が発番した `report_id` をモジュール内 `_COMPOSED_REPORTS`
    dict に保持。
  - `submit_to_ground` は未知 id を `{"status":"error",...}` で reject。
- **Region context-bind** (`tools/region.py` + `app/server.py`)
  - `app/server.py::_reverse_geocode()` で fetch 時に Nominatim `/reverse`
    を 1 回叩き、結果を sidecar JSON に保存。
  - `build_tool_registry` で `make_get_region_info(lat, lon, region_payload)`
    factory により lat/lon/region をクロージャに束ねる。
  - `tools/schema.py` から `get_region_info` / `get_history` の lat/lon 引数を
    削除 → モデルが座標を捏造する余地を排除。

### 1.4 UI: Operator instructions 入力欄
- `app/static/index.html` に `<details><textarea id="agent-instructions">` を
  Trace パネル上部に追加 (折りたたみ式・任意入力)。
- `app/static/js/main.js::runAgent()` で textarea の値を
  `&instructions=...` クエリに付与。
- `/api/run_agent` が `instructions` を受けて `run_react_openai` に転送。

### 1.5 mandatory call sequence の固定化
- `agent/react_loop.py` SYSTEM_PROMPT:
  1. 調査 (`classify_change` → spectral 系を 1 回以上)
  2. `compose_report(...)` → `report_id` を取得
  3. `submit_to_ground(report_id, reason, ...)` または `drop(reason)`

## 2. HACK / TODO

### HACK
- **`_resolve_provider_cfg` の `kind` 暗黙判定**
  - `base_url` の有無で OpenAI 互換とみなしている箇所がある。providers.yaml の
    `kind` を必須化し、明示しないと起動時に reject する形に直したい。
- **`_COMPOSED_REPORTS` がプロセス内グローバル**
  - 多プロセス worker やリスタートで失われる。本番化するなら sidecar JSON か
    sqlite に永続化すべき。
- **Nominatim 直叩き**
  - User-Agent はベタ書き。レート制御 (1 req/s) も未実装。検証用途には十分
    だが、本番ではキャッシュ層 + 自前ジオコーダに置換が必要。
- **`forced_tool_steps=2` のマジックナンバー**
  - シナリオ難度に応じて変えたい。providers.yaml 側に持たせるか、
    `/api/run_agent` のクエリで上書き可能にするか検討。

### TODO
- [ ] `instructions` を Trace に明示的に出すイベント (`operator_instructions`)
      を追加。現状は user メッセージに混ざるため、保存トレースから事後に
      区別しづらい。
- [ ] `submit_to_ground` の `reason` に `report_id` で参照したメトリクスが
      含まれているかを軽くバリデート (例: 数値が 1 つも無ければ警告)。
- [ ] `compose_report` 引数 (change_type / urgency / description) の妥当性
      チェック。urgency は 0–10 範囲の clamping のみ実装済。
- [ ] vLLM 側で長時間アイドルすると KV cache が解放されてレイテンシが跳ねる
      ことがある。warmup ジョブの追加。
- [ ] Frontend: instructions textarea の値を localStorage に保存して
      リロード後も復元する。
- [ ] テスト: `_COMPOSED_REPORTS` を介した submit reject の自動テストを
      追加 (現状はサニティスクリプトのみ)。
- [ ] `agent/react_loop.py` (Anthropic 系) の SYSTEM_PROMPT は同期したが、
      ループ実装側はまだ `forced_tool_steps` 相当が未導入。Gemini/Anthropic
      経路を本格復活させる際に同じ簡素化を適用すること。
