# SatelliteAgent — Phase 5 本番投入のためのデータ仕様

S0–S8 (Kaggle RTX 6000 Pro オフライン × LFM2.5-VL × prime-rl × satelliteagent_env tool-calling) の最小スタックは toy データで動作確定済み。本書は **toy → 本番 (Phase 5b)** で必要となる **データセットの構造・スキーマ・配置・Kaggle Dataset 化方法** を 1 枚にまとめたもの。Phase 2 担当者へそのまま渡せる。

## 0. 設計の前提 ([RunAgent](../agent/react_loop.py) と整合)

- **タスクは 2 値**: 各 case で AI は `submit_to_ground(...)` か `drop()` のいずれかで終了する
- **system prompt は完全固定** ([react_loop.py:15-29](../agent/react_loop.py#L15-L29) — "You are an onboard satellite operator agent..."。case 毎には変えない)
- **user message も構造固定** ("Before image..." + 画像 / "After image..." + 画像 / "Analyze the pair..." の 3 パート)
- **case 毎に違うのは**: (a) 画像 2 枚 (before/after) (b) tool が見るメタ情報 (lat/lon/size_km/timestamps) (c) 正解 action と判定材料
- **triplet/variant 設計は採用しない** (RunAgent に variant 概念が無いため)

prime-rl 視点では env が下記を渡す:

```python
{
  "prompt": [system_固定, user_固定_with_2images],
  "info": {"context": {...lat/lon/...}, "expected": {...action/...}}
}
```

reward 計算は [eval/validators/common.py](../eval/validators/common.py) (Phase 2 で実装) が `info.expected` を参照して行う。

---

## 1. 必要なデータの全体像

| 種別 | 用途 | 形式 | 必須? |
|---|---|---|---|
| **eval cases (YAML)** | GRPO の rollout 入力 + reward ground truth | 1 ファイル / case | **必須** |
| **画像** | VLM の vision 入力 (before/after) | PNG/JPEG | **必須** |
| **precompute_tool_responses** | tool 実行を rollout 中にキャッシュ | 1 ファイル / case | 推奨 (実 tool が重い/非決定論なら必須) |
| **SFT trace (Stage 0)** | tool-call format warmup | JSONL | **必須** (LFM2.5-VL base は tool 自発呼び出ししない) |
| **SFT trace (Stage 1, golden trajectory)** | domain warmup | JSONL | **任意** (Stage 0 で十分なら省略可) |

Validators は Python コード ([eval/validators/common.py](../eval/validators/common.py)) なのでデータでは無く、SatelliteAgent リポジトリの `Branch/refactor` に commit すれば env-prep notebook が clone して取り込む (S8 で確立した経路)。

---

## 2. Kaggle Dataset 構造

**1 個の Kaggle Dataset** (`<KAGGLE_USER>/satelliteagent-data` など) に全部まとめる。S6/S7/S8 で確立した notebook の `dataset_sources` で参照する。

```
satelliteagent-data/
├── eval/
│   ├── cases/
│   │   ├── case_001.yaml
│   │   ├── case_002.yaml
│   │   └── ...
│   ├── images/
│   │   ├── case_001/
│   │   │   ├── before.png
│   │   │   └── after.png
│   │   ├── case_002/
│   │   │   ├── before.png
│   │   │   └── after.png
│   │   └── ...
│   └── precompute_tool_responses/
│       ├── case_001.yaml
│       └── ...
├── sft/
│   ├── stage0_fmt_warmup/
│   │   └── data.jsonl
│   └── stage1_domain/                # 任意 (golden trajectory がある場合のみ)
│       └── data.jsonl
├── splits/
│   ├── train.txt                      # 1 行 1 case_id
│   ├── val.txt
│   └── test.txt
└── manifest.json
```

prime-rl notebook 側からは `/kaggle/input/satelliteagent-data/` にマウント。

```json
{
  "dataset_sources": ["<KAGGLE_USER>/satelliteagent-data"],
  "kernel_sources": [
    "<KAGGLE_USER>/prime-rl-offline-prep",
    "<KAGGLE_USER>/satelliteagent-env-prep"
  ]
}
```

---

## 3. スキーマ詳細

### 3.1 eval/cases/case_NNN.yaml

`satelliteagent_env.load_environment(toy=False)` がこれを読んで Dataset 1 行に展開する。

```yaml
case_id: flood_sylhet_2024
version: "1.0"
description: "Sylhet (BD) 2024 monsoon flooding — populated area"

# === RunAgent の context_from_keys 相当 (tool が参照する事実情報) ===
context:
  lat: 24.90
  lon: 91.87
  size_km: 10.0
  before_ts: "2024-05-15T03:42:11Z"
  after_ts:  "2024-08-20T03:46:55Z"
  region_description: "Sylhet, Bangladesh — populated, monsoon-prone"

# === 画像 (Dataset root からの相対パス) ===
images:
  before: images/flood_sylhet_2024/before.png
  after:  images/flood_sylhet_2024/after.png

# === reward 計算用 ground truth (validators が参照) ===
expected:
  action: submit_to_ground             # submit_to_ground | drop  (必須)
  attach_image: true                   # 任意 (validators/common.py の attach_image_match で使う)
  urgency: high                        # low | medium | high
  change_type: flood                   # flood | wildfire | landuse_change | none | ...
  rationale: |                         # 人手レビュー用、validator は読まない
    Visible flood expansion over populated area; bandwidth justified.
```

#### 必須 vs 任意

| フィールド | 必須? | 用途 |
|---|---|---|
| `case_id` | 必須 | 一意識別子、splits や precompute と紐付け |
| `context.lat/lon/size_km/before_ts/after_ts` | 必須 | tool が画像 fetch/index 計算で使う |
| `images.before/after` | 必須 | VLM 入力 |
| `expected.action` | **必須** | 主 reward (drop/submit) |
| `expected.attach_image` | 任意 | 細かい reward (画像添付の判断) |
| `expected.urgency` | 任意 | 細かい reward |
| `expected.change_type` | 任意 | 細かい reward |

最小構成は `case_id` + `context` + `images` + `expected.action` のみで動く。

#### バランス制約

- `expected.action` は **drop : submit ≒ 40:60 〜 60:40** を厳守 (always-submit / always-drop のゲーミング防止)
- 詳細は [docs/EVAL_DATA_DESIGN.md](../docs/EVAL_DATA_DESIGN.md) §3 (シナリオの多様性) を参照

### 3.2 eval/precompute_tool_responses/case_NNN.yaml

`satelliteagent_env` の `setup_state` で読み込み、tool 実行時にキャッシュ参照する。**実 tool 呼ばずに済む = rollout 高速 & 決定論**。

```yaml
case_id: flood_sylhet_2024

responses:
  - tool: detect_change                 # tool 関数名
    args:                               # canonical kwargs (sorted JSON dump など)
      image_a: before
      image_b: after
    response:                           # tool が返すべき dict
      detected: true
      type: flood
      confidence: 0.92
      affected_area_km2: 12.5

  - tool: spectral_index
    args:
      image: after
      index: ndwi
    response:
      mean: 0.34
      max: 0.78

  # submit_to_ground / drop は terminal なのでキャッシュ不要 (validator が拾う)
```

未キャッシュの (tool, args) を呼ばれた場合は `{"error": "not precomputed"}` を返して env がそのターンを reward 0 にする運用が無難。

### 3.3 sft/stage0_fmt_warmup/data.jsonl

**目的**: tool-call の wire format (Hermes 形式 `<tool_call>{"name": ..., "arguments": ...}</tool_call>`) を覚えさせる。**最小 50–200 行で OK**、画像は合成 / 流用可。

各行 = 1 サンプル (parquet 化前は JSONL で書いて、prime-rl 側は parquet folder に変換):

```json
{
  "messages": [
    {"role": "system", "content": "You are an onboard satellite operator agent..."},
    {"role": "user", "content": [
      {"type": "image", "path": "/abs/path/before.png"},
      {"type": "image", "path": "/abs/path/after.png"},
      {"type": "text",  "text": "Analyze the pair and decide what to report to ground."}
    ]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "submit_to_ground",
                    "arguments": "{\"report_id\": \"r1\", \"attach_image\": true, \"urgency\": \"high\", \"change_type\": \"flood\"}"}}
    ]}
  ]
}
```

format さえ正しければ scenario の reasonability は問わない (合成 case で OK)。

### 3.4 sft/stage1_domain/data.jsonl (任意)

**目的**: case と 1:1 対応の **golden trajectory** で domain 知識を教える。`expected.action` から自動生成可能なら省略してもよい (Stage 0 + GRPO で収束する想定)。

```json
{
  "case_id": "flood_sylhet_2024",
  "messages": [
    {"role": "system", "content": "You are an onboard satellite operator agent..."},
    {"role": "user", "content": [
      {"type": "image", "path": "images/flood_sylhet_2024/before.png"},
      {"type": "image", "path": "images/flood_sylhet_2024/after.png"},
      {"type": "text",  "text": "Analyze the pair and decide what to report to ground."}
    ]},
    {"role": "assistant", "tool_calls": [
      {"id": "1", "type": "function",
       "function": {"name": "detect_change",
                    "arguments": "{\"image_a\": \"before\", \"image_b\": \"after\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "1",
     "content": "{\"detected\": true, \"type\": \"flood\", \"confidence\": 0.92}"},
    {"role": "assistant", "tool_calls": [
      {"id": "2", "type": "function",
       "function": {"name": "submit_to_ground",
                    "arguments": "{\"report_id\": \"flood_sylhet_2024\", \"attach_image\": true, \"urgency\": \"high\", \"change_type\": \"flood\"}"}}
    ]}
  ]
}
```

### 3.5 splits/{train,val,test}.txt

```
flood_sylhet_2024
wildfire_california_2023
desert_clear_001
...
```

train で SFT/GRPO、val で best-of-N 評価、test で最終リーダーボード提出スコア。

### 3.6 manifest.json

```json
{
  "dataset_version": "0.1.0",
  "created": "2026-04-30",
  "case_count": 100,
  "splits": {"train": 70, "val": 15, "test": 15},
  "action_balance": {"submit_to_ground": 60, "drop": 40},
  "image_format": "png",
  "image_max_resolution": [512, 512]
}
```

---

## 4. 規模感の目安

| 項目 | 最小 (smoke) | 推奨 (本番) | 上限の参考 |
|---|---|---|---|
| case 数 | 30 | 100–200 | RTX 6000 Pro 単機なら 500 でも回る |
| 画像 / case | 2 (before+after 固定) | 2 | RunAgent と一致 |
| 画像解像度 | 256×256 | 384×384 | LFM2.5-VL は 512×512 まで現実的 |
| Stage 0 SFT 行 | 50 | 100–200 | format 学習なので少なくて良い |
| Stage 1 SFT 行 | (省略可) | case と同数 | golden trajectory なら 1:1 |
| Dataset 総サイズ | < 100 MB | 100–500 MB | Kaggle Dataset 上限は 100 GB |

---

## 5. データ作成 → アップロードの流れ

### 5.1 Phase 2 の作業

1. `eval/cases/case_NNN.yaml` を 100–200 件作る (実衛星 or シミュレーション、drop/submit のバランス厳守)
2. 対応する before/after 画像を `eval/images/<case_id>/{before,after}.png` に配置
3. `eval/validators/common.py` を実装 — reward 関数群 (`action_match` 必須、`attach_image_match` / `urgency_match` / `change_type_match` は任意で重み付け)
4. `eval/precompute_tool_responses/<case_id>.yaml` を生成 — tool ごとに本物実装を 1 回呼んでキャッシュするスクリプトを書く (例: `scripts/build_precompute.py`)
5. SFT Stage 0 trace を生成 (`scripts/build_stage0_traces.py` 想定 — 100-200 行)

### 5.2 Kaggle Dataset 化

```bash
# 上記ディレクトリ構成で dataset/ を準備
cd dataset
kaggle datasets init -p .
# dataset-metadata.json を編集 (id: <KAGGLE_USER>/satelliteagent-data)
kaggle datasets create -p .

# 以後の更新
kaggle datasets version -p . -m "v0.1.0 — initial 100 cases"
```

### 5.3 prime-rl notebook 側の変更 (S8 → S9 仮称)

[s8 notebook](notebooks/s8_lfm2vl_rl_satellite/) の clone を流用、変更は 4 点だけ:

1. `kernel-metadata.json` の `dataset_sources` に `<KAGGLE_USER>/satelliteagent-data` を追加
2. orch.toml に `data_root` を渡し、toy フラグを外す:
   ```toml
   [[train.env]]
   id = "satelliteagent_env"
   [train.env.kwargs]
   toy = false
   data_root = "/kaggle/input/satelliteagent-data"
   ```
3. trainer.toml の `seq_len` を画像トークン分増やす (384×384 画像 2 枚 ≈ +400 token 程度)
4. `tool_choice = "required"` は SFT Stage 0 後は外す (自発呼び出しを学習させる)

---

## 6. SatelliteAgent 側で必要な実装 (`load_environment(toy=False)`)

[satelliteagent_env/__init__.py](../satelliteagent_env/__init__.py) の `NotImplementedError` ブランチを埋める。骨子:

```python
def load_environment(toy: bool = True, data_root: str | None = None, **kwargs):
    if toy:
        ...  # 既存
    # === 本番 ===
    import yaml, glob
    from datasets import Dataset
    from eval.validators.common import (
        action_match,
        attach_image_match,   # 任意
        urgency_match,        # 任意
        change_type_match,    # 任意
    )

    SYSTEM_PROMPT = open("agent/react_loop.py").read()  # SYSTEM_PROMPT 定数を import するのが理想

    cases = sorted(glob.glob(f"{data_root}/eval/cases/*.yaml"))
    rows = []
    for path in cases:
        c = yaml.safe_load(open(path))
        before = f"{data_root}/{c['images']['before']}"
        after  = f"{data_root}/{c['images']['after']}"
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text",  "text": "Before image (previous satellite pass over this location):"},
                    {"type": "image", "path": before},
                    {"type": "text",  "text": "After image (current pass, same location):"},
                    {"type": "image", "path": after},
                    {"type": "text",  "text": "Analyze the pair and decide what to report to ground. "
                                              "Use your tools. End with submit_to_ground(...) or drop()."},
                ]},
            ],
            "info": {
                "case_id": c["case_id"],
                "context": c["context"],
                "expected": c["expected"],
                "data_root": data_root,
            },
        })
    dataset = Dataset.from_list(rows)

    rubric = vf.Rubric(
        funcs=[action_match, attach_image_match, urgency_match, change_type_match],
        weights=[1.0, 0.3, 0.2, 0.2],   # action が主、他は補助
    )

    SatelliteToolEnv = _build_env_class()  # setup_state で precompute load + state inject
    return SatelliteToolEnv(
        dataset=dataset,
        tools=[_expose_for_vf(t) for t in REAL_TOOLS],  # tools/ から
        rubric=rubric,
        max_turns=8,
    )
```

`setup_state` 側で `precompute_tool_responses/<case_id>.yaml` を読み込んで `state["tool_cache"]` に格納、`update_tool_args` でキャッシュ check → cached なら直接返す wrapper を tool に巻く。

---

## 7. 残作業チェックリスト (Phase 2 担当者向け)

- [ ] case YAML を 100 件作成 (drop/submit バランス 40:60〜60:40)
- [ ] 対応 before/after 画像準備 (実データ or 合成、最大 512×512)
- [ ] precompute_tool_responses を生成 (tool 実装が固まり次第)
- [ ] [eval/validators/common.py](../eval/validators/common.py) 実装 (`action_match` 必須、他オプション、verifiers 形式 = `async def f(completion, info, **kw) -> float`)
- [ ] SFT Stage 0 trace 生成スクリプト (`scripts/build_stage0_traces.py`) 作成
- [ ] splits ファイル + manifest.json
- [ ] Kaggle Dataset 化 (`kaggle datasets create`)
- [ ] [satelliteagent_env/__init__.py](../satelliteagent_env/__init__.py) の `load_environment(toy=False)` 実装
- [ ] s8 notebook の clone (s9_lfm2vl_grpo_real?) を作って `dataset_sources` 追加 + `toy = false` 渡し
