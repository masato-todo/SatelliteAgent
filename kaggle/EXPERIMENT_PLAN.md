# 実験プラン

## 0. 制約と前提

- **GPU**: Kaggle RTX 6000 Pro (Blackwell, 96GB VRAM)
- **Internet**: 強制 OFF (`enable_internet: false` 必須)
- **Competition紐付け必須**: `competition_sources: ["nvidia-nemotron-model-reasoning-challenge"]` がないとRTX 6000 Proが選べない
- **push時のフラグ**: `kaggle kernels push --accelerator NvidiaRtxPro6000` ((Kaggle CLI 2.0; metadataの`accelerator`指定だけでは効かない)
- **Python**: prime-rl は `~=3.12.0` 必須 (Kaggle 既設で 3.12.12 あり、別途用意不要だった)
- **時間**: ハッカソン締切 2026-05-06

## 1. 段階ゴール

| Stage | ゴール | 完了判定 | 状態 |
|---|---|---|---|
| **S0** | オフラインパッケージ構築 | wheels一式 + prime-rl src + HF model/dataset を Kaggle notebook の output として確保 | ✅ 2026-04-26 完了 (275 wheels) |
| **S1** | prime-rl が **import できる** | RTX 6000 Pro オフラインで `import prime_rl, torch, vllm, transformers` 全成功、GPU認識 | ✅ 2026-04-26 PASS |
| **S2** | reverse-text **SFT が完走** | `uv run sft @ ...` (or `prime-rl sft`) が `max_steps=5 batch=1 seq=512` で正常終了 | ✅ 2026-04-27 PASS |
| **S3** | reverse-text **RL pipeline が回る** | inference + orchestrator + trainer 3プロセスが起動、orchestrator が rollout 生成、reward 計算、trainer が training loop 開始 | ✅ 2026-04-27 PASS (3-process method) |
| **S4** | **LFM2.5-VL ロード + generate 検証** | `AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM2.5-VL-450M")` 成功 + dummy画像で generate 動作 + chat template 確認 | ✅ 2026-04-27 PASS |
| **S5** | SatelliteAgent 統合 (toy) | satelliteagent_env (SSOT glue) + 2-prep notebook構造 + tool_call_parser 配線まで確認 | ✅ 2026-04-27 toy PASS / 本番は Phase 2-4 後 |
| **S6** | **LFM2.5-VL × prime-rl SFT** | `prime_rl.entrypoints.sft` が LFM2.5-VL を VLM 設定でロードし synthetic VLM データで loss が下がる + vLLM serve 確認 | ✅ 2026-04-27 PASS (loss 3.25→0.94, peak 3.8 GiB) |
| **S7** | **LFM2.5-VL × prime-rl RL (reverse-text)** | LFM2.5-VL を 3-process RL で 2 steps 完走、`RL trainer finished!` まで到達 | ✅ 2026-04-27 PASS (peak 4.9 GiB) |
| **S8** | **LFM2.5-VL × prime-rl RL (satelliteagent_env)** | LFM2.5-VL + tool-calling toy env で 3-process RL pipeline 完走 | ✅ 2026-04-27 PASS (peak 4.7 GiB, 115.7s) |

## 2. 各 Stage の詳細

### Stage 0: オフラインパッケージ構築 ✅ 完了

**当初計画と違ったポイント**:
- Kaggle Dataset 化は不要だった → notebook の output を `kernel_sources` で参照すれば十分
- Python 3.12 と uv は Kaggle 環境に既設なので `python-build-standalone` 同梱不要
- flash-attn は Stage 1-3 不要なので除外

**実際に動いた構成** ([notebooks/s0_prep/](notebooks/s0_prep/), Kaggle: `<KAGGLE_USER>/prime-rl-offline-prep`):

1. CPU notebook (Internet ON) で `python3.12 -m pip download` 実行
2. `uv export --no-hashes --no-emit-project` で git+/URL 含む reqs.txt 生成 (uvはこの目的だけに使用)
3. `pip download -r reqs.txt --extra-index-url cu128 --extra-index-url primeintellect` で全 wheel 取得
4. torch cu128 を別途 `pip download --index-url cu128 --no-deps` で確実に
5. sdist (.zip/.tar.gz) を `pip wheel` で wheel 化 (オフライン側で hatchling 等の build backend が無いと詰むため)
6. `pip wheel /kaggle/working/prime-rl --no-deps` で prime-rl 自体も wheel 化
7. HF snapshot で `Qwen3-0.6B` と `reverse-text-dataset` を `/kaggle/working/output/{models,datasets}/` へ
8. 出力構造:
   ```
   /kaggle/working/output/
   ├── wheels/         (275個, ~5.8GB)
   ├── prime_rl_src/   (src tree)
   ├── models/Qwen3-0.6B/
   ├── datasets/reverse-text/
   └── manifest.json
   ```

**踏んだ罠**:
- `uv pip download` は uv 0.9.x には存在しない → 普通の pip を使う
- `uv sync` がシステム Python を流用してキャッシュが空になる (managed-python 強制も効かなかった)
- `pip download` で git+ パッケージは sdist (.zip) で保存される → prep 段階で wheel 化必須
- `--no-build-isolation` で `pip wheel prime-rl` すると hatchling が無くて失敗 → build isolation を有効に

### Stage 1: import 通過 ✅ PASS

**実際に動いた install 手順** ([notebooks/s1_import_test/](notebooks/s1_import_test/), Kaggle: `<KAGGLE_USER>/prime-rl-s1-import-test`):

1. `kernel_sources: ["<KAGGLE_USER>/prime-rl-offline-prep"]` で prep の output を `/kaggle/input/notebooks/<KAGGLE_USER>/prime-rl-offline-prep/output/` にマウント
2. **wheel フィルタリングが必須**:
   - `torch-` / `torchvision-` / `torchaudio-` / `nvidia_*` で始まる wheel は **install しない** (Kaggle既設のtorch+NCCLとABI衝突する)
   - 重複バージョン (例: torch-2.10+cu128 と torch-2.11+cu128 が両方DLされた) は最高版のみ採用
3. install コマンド:
   ```bash
   python3.12 -m pip install --no-index --no-build-isolation --pre --no-deps <filtered_wheels>
   ```
   - `--no-deps` 必須: vllm 0.19 が `transformers<5` を pin、prime-rl は dev版transformers 5.x を `transformers_v5_compat` plugin で実行時パッチする (pip resolverはこれを理解できない)
   - `--pre` 必須: verifiers-0.1.12.dev6 等の dev リリース
   - `--no-build-isolation` 必須: 残ったsdistがあった場合に hatchling 等の取得をしようとしないように
4. 動作実績の組合せ:
   - torch 2.10.0+cu128 (Kaggle 既設、我々のwheelは使わず)
   - transformers 5.5.0 (我々の wheel)
   - vllm 0.19.0 (我々の wheel)
   - prime_rl 0.4.0 (我々の wheel)
   - GPU: NVIDIA RTX PRO 6000 Blackwell, 102GB VRAM, sm12.0

### Stage 2: reverse-text SFT ✅ PASS

S1 で確立した install を再利用。Qwen3-0.6B + reverse-text dataset で `max_steps=5 batch=1 seq=512`、67s で `SFT trainer finished!`、peak VRAM 11.3GB。`prime-rl sft` の起動経路 (`python3.12 -m prime_rl.entrypoints.sft @ <toml>`) と `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled` の組合せがそのまま S6 にも流用可能と判明。

### Stage 3: reverse-text RL ✅ PASS

**当初計画と違ったポイント**:
- `rl` unified entrypoint は **>=2 GPUs 必須** (CHANGELOG 2026-02-23 で `--trainer-gpu-ids` 削除済み)
- 単GPUは **3プロセス手動起動** が必須

**実際に動いた構成** ([notebooks/s3_rl_smoke/](notebooks/s3_rl_smoke/)):

3つのTOMLに分割し、3プロセスで起動:
```python
# inference (background): vLLM, gpu_memory_utilization=0.5, port=8000
python3.12 -m prime_rl.entrypoints.inference @ infer.toml

# orchestrator (background): no GPU, talks to vLLM via HTTP
python3.12 -m prime_rl.orchestrator.orchestrator @ orch.toml

# trainer (foreground via torchrun for distributed env vars)
python3.12 -m torch.distributed.run --nproc-per-node 1 \
  -m prime_rl.trainer.rl.train @ trainer.toml
```

**踏んだ罠**:
- `VLLM_API_KEY` 設定 → vLLM が認証要求 → health check には `Authorization: Bearer` ヘッダ必要
- trainer 直叩きは `RANK env var expected` で失敗 → `torch.distributed.run` で wrap
- orchestrator は reverse-text Environment が `verifiers.load_environment("reverse-text")` で `PrimeIntellect/Reverse-Text-RL` HF dataset を読みに行く → prep で local_dir snapshot + S3 で `sed -i 's|PrimeIntellect/Reverse-Text-RL|<local_path>|g'` で patch
- `HF_DATASETS_CACHE` を read-only マウントに向けると lock file 作成失敗 → `/kaggle/working/hf_cache` に書込可能パスを設定
- 最終的な「停止」は zero_advantage filter による設計通りの挙動 (base Qwen3-0.6B は reverse-text できないので全 rollout reward=0 → group内分散ゼロ → 学習意味なしと判断)

**確認できたこと**:
1. inference (vLLM) + orchestrator + trainer の3プロセス協調動作
2. orchestrator → vLLM の HTTP 経由 rollout 生成
3. reward 計算 (LCS ratio)
4. trainer の RL training loop 起動

実プロダクションで RL を回すには SFT-tuned base が必要 (例の `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`)。SatelliteAgent では Phase 4 (Stage 0+1 SFT) → Phase 5 (GRPO) の順で進めるので問題なし。

### Stage 4: LFM2.5-VL ロード + generate 検証 ✅ PASS

**実際に動いた構成** ([notebooks/s4_lfm2_vl_load/](notebooks/s4_lfm2_vl_load/)):

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
processor = AutoProcessor.from_pretrained("/path/to/LFM2.5-VL-450M")
model = AutoModelForImageTextToText.from_pretrained(
    "/path/to/LFM2.5-VL-450M", dtype=torch.bfloat16, device_map="cuda"
)
# dummy 224x224 blue image + "What dominant color do you see?" prompt
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
```

**結果**:
- `Lfm2VlProcessor` + `Lfm2VlForConditionalGeneration`, 448.7M params
- Load後 VRAM: **0.9 GB** (RTX 6000 Pro 102GB の 1% 以下)
- Generate 時間: 0.9s (64 tokens)
- 出力: "The dominant color in this image is blue. The entire background is filled with a solid, vibrant blue color..."
- Chat template: **ChatML** 形式 (`<|im_start|>system/user/assistant<|im_end|>`) → SatelliteAgent ReAct trace との整合性 OK

**踏んだ罠**:
- numpy の version 衝突 — 我々の wheel `numpy 2.2.6` で Kaggle 既設 scipy が壊れる (`cannot import name '_center' from numpy._core.umath`)
- 解決: `SKIP_PREFIXES` に `numpy-` `scipy-` `scikit_learn-` `pandas-` 追加 → Kaggle 既設の科学計算スタックを温存

**残り**: prime-rl の trainer に LFM2.5-VL を VLM として食わせる検証は Stage 5 で実施 (SatelliteAgent 統合と同時にやる方が効率的)。

### Stage 5: SatelliteAgent 統合 ✅ toy PASS (2026-04-27)

**実装した SSOT 構造**:

```
SatelliteAgent/
├── tools/                              # ← 既存、変更なし
├── eval/                               # ← 既存 (Phase 2 で中身充実)
└── satelliteagent_env/                 # NEW (~150 LOC, glue only)
    └── __init__.py                     # load_environment + state injection
```

`satelliteagent_env/__init__.py` は:
- `from tools.stubs import STUB_TOOLS, submit_to_ground, drop` で SSOT 参照
- `_expose_for_vf()` wrapper で `**_extra` (pydantic v2 が嫌う leading-underscore varkw) を strip
- `load_environment(toy=True)` で toy dataset (10 シナリオ × submit/drop 二択) + action_match reward を提供
- `toy=False` は Phase 2 の eval/cases triplets + eval/validators/common.py を待つ

**Kaggle 構成 (2 prep + S5)**:

- [notebooks/s0_prep/](notebooks/s0_prep/) `<KAGGLE_USER>/prime-rl-offline-prep` — 重い prep (15-20min)
- [notebooks/s0b_env_prep/](notebooks/s0b_env_prep/) `<KAGGLE_USER>/satelliteagent-env-prep` — SatelliteAgent wheel build のみ (~2min、SatelliteAgent 更新時に再ビルド)
- [notebooks/s5_satelliteagent_rl/](notebooks/s5_satelliteagent_rl/) `<KAGGLE_USER>/prime-rl-s5-satelliteagent-rl` — 両 prep を `kernel_sources` で参照、`id = "satelliteagent_env"` で toy RL

**確認できたこと**:
1. SatelliteAgent wheel build (`pip wheel . --no-deps`) 成功、`satelliteagent_env` が verifiers から resolve される
2. orchestrator が env worker 起動、ToolEnv 経由で `submit_to_ground` / `drop` を model に提示
3. vLLM の `tool_call_parser = "hermes"` で Qwen3 系の tool call protocol が成立
4. trainer が training loop に到達

**踏んだ罠 (memory に追記)**:
- pydantic v2 が tool 関数の leading-underscore varkw (`**_extra`) を弾く → env package で sig wrapper
- vLLM の auto tool choice には `tool_call_parser` 明示が必要 (default "auto" でも 400 で弾かれる)
- base Qwen3-0.6B は **tool 呼び出しを無視してテキスト返答** → empty trajectory ループ。Phase 4 (SFT Stage 0 fmt warmup) で format 教えてから RL に入る ARCHITECTURE 通りの順序が必要。toy 検証では `tool_choice = "required"` で強制可能

**Phase 2-4 着手後の本番統合 (Stage 5b)**:
- toy dataset → `eval/cases/triplets/*.yaml` loader
- toy rubric (action_match のみ) → `eval/validators/common.py` の action / attach_image / urgency / change_type / trajectory_validity / bandwidth_saved を組み込み
- StatefulToolEnv の `setup_state` で precompute_tool_responses cache を inject
- SFT Stage 0 で format 教えた adapter を base に GRPO へ

`load_environment(toy=False)` の TODO ブランチを Phase 2 と同期して埋める。

### Stage 6: LFM2.5-VL × prime-rl SFT pipeline ✅ PASS (2026-04-27)

**目的**: prime-rl の SFT trainer / vLLM inference が **本番モデル LFM2.5-VL** を扱えるかの最小検証。S2 (Qwen3-0.6B) で動いただけでは LFM2.5-VL での動作保証にならない。

**実装** ([notebooks/s6_lfm2vl_pipeline/](notebooks/s6_lfm2vl_pipeline/), Kaggle: `<KAGGLE_USER>/prime-rl-s6-lfm2vl-pipeline`):

合成 VLM データ (4色 RGB 画像 + "What color?" → 色名 の 4 サンプル) を parquet 形式 (`<dir>/parquet/train/data.parquet`) で書き出し、SFT 2 steps を回す。

**結果**:
- Test C (SFT 2 steps): Loss 3.2542 → 0.9406, peak VRAM 3.8 GiB, `SFT trainer finished!`
- Test D (vLLM serve): `GET /v1/models 200 OK` で起動確認

**踏んだ罠と patch**:

1. **`prime_rl.utils.vlm.VLM_REGISTRY` に lfm2_vl が無い** → 起動時 `KeyError`。install 後に runtime patch (`PATCH 1`):
   ```python
   # qwen3_vl エントリの直前に lfm2_vl を挿入
   "lfm2_vl": VLMModelInfo(vision_encoder_attr="model.vision_tower",
                           language_model_attr="model.language_model"),
   ```
2. **`Lfm2DecoderLayer` に `mlp` 属性が無い** (LFM2 は MoE ではないので) → trainer の `configure_moe_ep_backend` が `transformer_block.mlp` 直接アクセスで AttributeError。`PATCH 2` で `getattr(transformer_block, "mlp", None)` ガードに置換。
3. **Siglip2 vision tower が flash-attn 未対応** → trainer.toml で `attn = "sdpa"` 必須。
4. **VLM ロードは bf16 必須** → `optimization_dtype = "bfloat16"`, `reduce_dtype = "bfloat16"`。
5. **`max_steps = 0` で `KeyError: 'perf/peak_memory'`** (Test B 失敗) — perf metrics 未記録なのに flush で参照される prime-rl 側のバグ。`max_steps>=1` で回避 (Test C で代替)。
6. **`[data.fake]` は SFT trainer に効かない** → 実データ (parquet) が必須。`save_to_disk` だと `load_dataset` 経路 (prime-rl が使う方) では読めない → 必ず parquet folder で出す。
7. **kaggle output retention**: `/kaggle/working/outputs/` 配下に書いたファイルは確実に `kaggle kernels output` で取得できる。直下のファイルやサブディレクトリの一部は取得されない場合があるので、**デバッグ用ログは必ず `outputs/` 以下に書く**。
8. **ckpt 抑制**: trainer.toml から `[ckpt]` セクションを削除すると checkpoint が書かれない (ダウンロードを軽くする目的)。

### Stage 7: LFM2.5-VL × prime-rl RL pipeline (reverse-text toy) ✅ PASS (2026-04-27)

**目的**: 「LFM2.5-VL で **RL pipeline 自体が回るか**」を tool/env の複雑性無しで切り分ける。reverse-text は text-only env なので tool 配線とは独立に検証できる。

**実装** ([notebooks/s7_lfm2vl_rl_revtext/](notebooks/s7_lfm2vl_rl_revtext/), Kaggle: `<KAGGLE_USER>/prime-rl-s7-lfm2vl-rl-revtext`):

S3 (Qwen3-0.6B reverse-text RL) の 3-process 構造をベースに、モデルだけ LFM2.5-VL に差し替え + S6 で見つけた 2 patch + `[model.vlm]` + bf16 + sdpa を入れた trainer.toml。

**結果**:
- Step 0: Loss 0.0038, Entropy 1.2465, Mismatch KL 0.0358, 81s (Inductor compile)
- Step 1: Loss -0.0008, Entropy 5.2441, Mismatch KL 0.0418, 0.28s (キャッシュ効果)
- Peak VRAM: **4.9 GiB / 95 GiB** (RTX 6000 Pro の 5%)
- `RL trainer finished!` SUCCESS、`outputs/run_default/broadcasts/step_1/model.safetensors` まで書かれる

**確認できたこと**:
1. 3-process pipeline (vLLM + orchestrator + trainer) が **VLM ロードでも安定** (S6 patch で十分)
2. `weight broadcast (filesystem, sharded safetensors)` が VLM でも機能
3. reverse-text env worker が VLM model 相手にも rollout を生成
4. RL trainer の loss/entropy/KL metric が記録される

reverse-text の reward は base モデルが解けず低いままだが、**pipeline の動作確認** が目的なので問題なし (S2/S3 と同じ位置付け)。

### Stage 8: LFM2.5-VL × satelliteagent_env tool-calling RL ✅ PASS (2026-04-27)

**目的**: S7 の VLM RL pipeline + S5 の tool-calling env を結合し、ハッカソン本番設計 (LFM2.5-VL × ツール呼び出し × GRPO) の最小スタックが回ることを確認。

**実装** ([notebooks/s8_lfm2vl_rl_satellite/](notebooks/s8_lfm2vl_rl_satellite/), Kaggle: `<KAGGLE_USER>/prime-rl-s8-lfm2vl-rl-satellite`):

S5 の構造 (satelliteagent_env toy + 2-prep notebook) + S7 の VLM patch (PATCH 1, PATCH 2) + bf16/sdpa + `tool_call_parser = "hermes"` + `tool_choice = "required"` (smoke 検証用、base モデルが自発的に tool 呼ばないので強制)。

**結果**:
- Step 0: Loss 0.0003, Entropy 0.4182, Mismatch KL 0.1712, 83.40s, peak 4.0 GiB
- Step 1: Loss 0.0002, Entropy 0.4184, Mismatch KL 0.1417, 0.38s, peak 4.7 GiB
- `RL trainer finished!` / trainer exit=0, total 115.7s

**確認できたこと**:
1. **vLLM が LFM2.5-VL を tool_call_parser="hermes" で serve** — `POST /v1/chat/completions` 全部 200 で応答 (S8 v1/v2 の時点で確認済み)
2. ChatML chat_template が Hermes 形式 tool call と互換 (LFM2.5-VL でも parser エラー無し)
3. 3-process pipeline が tool-calling env でも完走 (rollout 生成 → reward 計算 → trainer step 完了)
4. `tool_choice = "required"` で base モデルから tool emission を強制可能

**踏んだ罠**:
- **`_SatelliteToolEnv.update_tool_args` の シグネチャミスマッチ** (S5 で潜伏、S8 で `tool_choice=required` により顕在化): verifiers が 5 positional 渡すのに env 側 4 受けで `TypeError`。`*args, **kwargs` 受けに修正してフォワード/バックワード互換に。修正は `satelliteagent_env/__init__.py` 1 行 → GitHub に push → env-prep 再 push (Kaggle は GitHub から clone する構造)。
- **GitHub 経由 wheel ビルドの教訓**: ローカル修正 + env-prep 再 push だけでは反映されない (clone 元が GitHub の `Branch/refactor`)。`git push origin Branch/refactor` 必須。
- **base モデルは自発的に tool 呼ばない**: `tool_choice = "required"` を `[train.sampling.extra_body]` に設定して強制 (本番は SFT Stage 0 fmt warmup で format 教えてから外す)。

### Stage 5b (本番): 次のマイルストーン

S0-S8 で「LFM2.5-VL × prime-rl × ツール呼び出し × Kaggle RTX 6000 Pro オフライン」のパイプラインが揃った。残るは **データと SFT warmup**:

1. Phase 2 完成: `eval/cases/triplets/*.yaml` (本物 triplet) と `eval/validators/common.py` (action / attach_image / urgency / change_type / trajectory_validity / bandwidth_saved)
2. `satelliteagent_env.load_environment(toy=False)` の `NotImplementedError` ブランチを実装 (triplet loader + `setup_state` で precompute_tool_responses inject + StatefulToolEnv の state 配線)
3. SFT Stage 0 (fmt warmup): tool 呼び出し format を教える小規模 SFT → adapter or full ckpt を base に
4. SFT Stage 1 (domain): satellite scenario の golden trajectory で domain 学習
5. GRPO (Stage 2): Stage 1 base + 全 validator で RL、`tool_choice = "required"` を外して自発 tool 呼び出しを学習させる

## 3. 残リスク

| リスク | 影響 | 緩和策 |
|---|---|---|
| ~~LFM2.5-VL が prime-rl の trainer ローダで動かない~~ | ~~S4 で詰む~~ | **解決済み (S6/S7)**: 2 つの runtime patch (VLM_REGISTRY 追加 + mlp guard) + bf16/sdpa 設定で SFT/RL 共に動作 |
| ~~reverse-text Environment が `[envs]` extra から抜け落ちている~~ | ~~S3 で 'env not found'~~ | **解決済み**: primeintellect index 経由で reverse-text wheel 取得 + sed で local path patch |
| ~~flash-attn を後から要求された~~ | ~~S2/S3 で性能不足~~ | **解決済み**: SFT/RL ともに `attn = "sdpa"` で動作 (LFM2 の Siglip2 vision tower は flash-attn 非対応) |
| ~~LFM2.5-VL の tool_call_parser が Qwen3 系 "hermes" と互換でない~~ | ~~S8 で tool が呼ばれない / 400 で弾かれる~~ | **解決済み (S8)**: `tool_call_parser = "hermes"` でそのまま動く (LFM2 の ChatML chat_template と Hermes 形式 tool call が互換)。base モデルは自発呼び出ししない傾向だが `[train.sampling.extra_body] tool_choice = "required"` で強制可能 |

## 4. 現在地 (2026-04-27)

- [x] prime-rl 公式ドキュメント / コード調査
- [x] 必要依存と単GPU RL VRAM ~48GB の確認
- [x] LFM2.5-VL の transformers 5.1+ 対応確認
- [x] **Stage 0: オフラインパッケージ構築完了 (275 wheels + flash-attn + reverse-text wheel + HF data snapshot)**
- [x] **Stage 1: prime-rl import on RTX 6000 Pro オフライン PASS**
- [x] **Stage 2: reverse-text SFT PASS (5 steps, 67s, peak VRAM 11.3GB)**
- [x] **Stage 3: reverse-text RL pipeline PASS (3-process method)**
- [x] **Stage 4: LFM2.5-VL-450M ロード + generate PASS (Lfm2VlForConditionalGeneration, 0.9GB VRAM, ChatML)**
- [x] **Stage 5 (toy): satelliteagent_env scaffold + Kaggle 2-prep 構造 + RL pipeline 配線 PASS**
- [x] **Stage 6: LFM2.5-VL × prime-rl SFT PASS (synthetic VLM data, loss 3.25 → 0.94, peak 3.8 GiB) + vLLM serve PASS**
- [x] **Stage 7: LFM2.5-VL × prime-rl RL pipeline PASS (reverse-text, 3-process, 2 steps, peak 4.9 GiB)**
- [x] **Stage 8: LFM2.5-VL × satelliteagent_env (tool-calling toy) RL pipeline PASS (3-process, 2 steps, peak 4.7 GiB, 115.7s, hermes parser, tool_choice=required)**
- [ ] Stage 5b (本番): Phase 2 (eval/cases + eval/validators) 着手後に toy → 本物データに差し替え。SFT Stage 0 fmt warmup 後 GRPO
