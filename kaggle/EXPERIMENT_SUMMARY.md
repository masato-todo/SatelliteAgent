# SatelliteAgent — 実験精度まとめ

LFM2.5-VL-450M による多時期 Sentinel-2 変化検出 (`submit_to_ground` / `drop` の二値判定)
についての主要 10 条件の overall + per-class 精度。

データ source: 各 Kaggle kernel の `outputs/eval_results.json` または `*_predictions.json`
をローカルに pull → `SatelliteAgent/kaggle/eval_consolidated.json` に正規化 → 本 md を生成。

## Test split

96 件の固定 test split (stratified 80/20, seed=0)。内訳:

| category | n | type |
|---|---:|---|
| pos_fire | 11 | positive (submit) |
| pos_volcanic | 13 | positive (submit) |
| pos_deforestation | 26 | positive (submit) |
| neg_soft | 21 | negative (drop), 容易 |
| neg_hard_volcano | 10 | negative (drop), hard pair |
| neg_hard_forest | 9 | negative (drop), hard pair |
| neg_hard_preburn | 6 | negative (drop), hard pair |
| **計** | **96** | positive 50 / negative 46 |

`hard_*` は positive と同じ lat/lon の異なる時期 (visual prior 不可)。

**注意**:
- S43 (#5) は v4 全 485 件で評価、しかも positive/negative の 2 分類のみ → per-class 欄は `—`。
- S46 (#6) は古い 100 件 split (test split 標準化前) → n=100。割合比較は OK。

## Overall

| # | 条件 | source | n | overall |
|---|---|---|---:|---:|
| 1 | Rule-based (spectral indices, no model) | S56 best_rule=R5_combined | 96 | **67.7%** |
| 2 | Gemini 2.5 Flash (binary) | S57 head-to-head | 96 | **67.7%** |
| 3 | LFM2.5-VL-450M base, with images | S64 base | 96 | **20.8%** |
| 4 | LFM2.5-VL-450M base, no images | S64 base_no_image | 96 | **52.1%** |
| 5 | Base + GRPO only (no SFT) | S43 after_grpo_mt (multi-turn) | 485 | **2.7%** |
| 6 | Single-turn SFT (image only, no spectral) | S46 after_sft | 100 | **61.0%** |
| 7 | Single-turn SFT (image + spectral tags + mixup) | S54 after_sft | 96 | **70.8%** |
| 8 | Multi-turn agent SFT (no images at inference) | S64 after_sft_no_image | 96 | **74.0%** |
| 9 | Multi-turn agent SFT + GRPO 30 step ★ best ★ | S64 after_grpo_no_image | 96 | **75.0%** |
| 10 | Multi-turn agent SFT + GRPO 100 step | S65 after_grpo_no_image | 96 | **74.0%** |

## Per-class accuracy

| 条件 | pos_fire<br>(n=11) | pos_volcanic<br>(n=13) | pos_deforestation<br>(n=26) | neg_soft<br>(n=21) | neg_hard_volcano<br>(n=10) | neg_hard_forest<br>(n=9) | neg_hard_preburn<br>(n=6) |
|---|---|---|---|---|---|---|---|
| Rule-based (spectral indices, no model) | 90.9% | 84.6% | 84.6% | 81.0% | 10.0% | 22.2% | 33.3% |
| Gemini 2.5 Flash (binary) | 45.5% | 100.0% | 84.6% | 76.2% | 20.0% | 22.2% | 83.3% |
| LFM2.5-VL-450M base, with images | 27.3% | 30.8% | 3.8% | 28.6% | 10.0% | 44.4% | 16.7% |
| LFM2.5-VL-450M base, no images | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Base + GRPO only (no SFT) | — | — | — | — | — | — | — |
| Single-turn SFT (image only, no spectral) | 81.8% | 7.7% | 22.2% | 95.7% | 90.0% | 90.0% | 83.3% |
| Single-turn SFT (image + spectral tags + mixup) | 100.0% | 38.5% | 61.5% | 85.7% | 60.0% | 100.0% | 50.0% |
| Multi-turn agent SFT (no images at inference) | 100.0% | 69.2% | 80.8% | 85.7% | 10.0% | 77.8% | 66.7% |
| Multi-turn agent SFT + GRPO 30 step ★ best | 100.0% | 69.2% | 84.6% | 85.7% | 10.0% | 77.8% | 66.7% |
| Multi-turn agent SFT + GRPO 100 step | 100.0% | 69.2% | 76.9% | 85.7% | 20.0% | 77.8% | 66.7% |

## 各条件の補足

### 1. Rule-based (spectral indices, no model)

- **Source**: `S56 best_rule=R5_combined`
- **n / overall**: 96 / 67.7%
- **Notes**: ETL ceiling, no ML model

### 2. Gemini 2.5 Flash (binary)

- **Source**: `S57 head-to-head`
- **n / overall**: 96 / 67.7%
- **Notes**: mc_overall=0.6250

### 3. LFM2.5-VL-450M base, with images

- **Source**: `S64 base`
- **n / overall**: 96 / 20.8%
- **Notes**: image attention dominates

### 4. LFM2.5-VL-450M base, no images

- **Source**: `S64 base_no_image`
- **n / overall**: 96 / 52.1%
- **Notes**: tool flow unblocked

### 5. Base + GRPO only (no SFT)

- **Source**: `S43 after_grpo_mt (multi-turn)`
- **n / overall**: 485 / 2.7%
- **Notes**: multi-turn collapsed (None bug, no terminal). Single-turn: after_grpo=0.5155 (~base 0.5196, no movement). positive_acc=0.0082 negative_acc=0.0458. n=485 (full v4, not 96-test split).

### 6. Single-turn SFT (image only, no spectral)

- **Source**: `S46 after_sft`
- **n / overall**: 100 / 61.0%
- **Notes**: base=0.4900, n=100 (older test split)

### 7. Single-turn SFT (image + spectral tags + mixup)

- **Source**: `S54 after_sft`
- **n / overall**: 96 / 70.8%
- **Notes**: best single-turn; mixup α∈[0.3,0.7]; spectral binary tags from S50

### 8. Multi-turn agent SFT (no images at inference)

- **Source**: `S64 after_sft_no_image`
- **n / overall**: 96 / 74.0%
- **Notes**: tool-calling flow re-enabled

### 9. Multi-turn agent SFT + GRPO 30 step ★ best

- **Source**: `S64 after_grpo_no_image`
- **n / overall**: 96 / 75.0%
- **Notes**: GRPO sweet spot

### 10. Multi-turn agent SFT + GRPO 100 step

- **Source**: `S65 after_grpo_no_image`
- **n / overall**: 96 / 74.0%
- **Notes**: over-train regression (-1pp vs 30 step)

## ストーリー (#1→#10 で読む)

1. **#1 Rule (67.7%)** と **#2 Gemini (67.7%)** が「天井」。spectral indices のしきい値 rule
   だけで Gemini と並ぶ点に注目 — pos クラスは spectral 単体で十分判別可能。
2. **#3 base with images (20.8%)**: LFM2.5-VL-450M は無学習だと画像 attention に支配され、
   tool-calling の system prompt を無視。
3. **#4 base no-image (52.1%)**: 画像を抜くだけで「全部 submit」bias で 52% (positive 比率)
   に到達。これは「学習効果ゼロ」のフロアと同等。
4. **#5 GRPO only (2.7%)**: SFT 無しで multi-turn agentic RL を回すと、analyze→terminal の
   sequence が学習されず None 多発で崩壊 (S42-S45)。reward 設計を変えても (analyze_then_terminal)
   崩壊を止められず、SFT で trajectory pattern を植え付ける必要があると判明。
5. **#6 SFT visual only (61.0%)**: image-only SFT。`neg_*` 90%+ で強いが `pos_volcanic` 7.7%
   / `pos_deforestation` 22.2% — 画像だけでは spectral 必須クラスが見えない。
6. **#7 SFT image+spectral+mixup (70.8%)**: spectral binary tag + mixup augmentation で
   `pos_fire` 100%、overall 70.8%。single-turn 経路の最高。
7. **#8 multi-turn SFT no-image (74.0%)**: 4-tool agent trajectory を SFT で学習、推論時に
   画像を渡さない (image attention の罠回避)。これだけで +24pp vs 画像入り。
8. **#9 multi-turn SFT + GRPO 30 step (75.0%) ★ best**: GRPO で `pos_deforestation`
   80.8%→84.6% など微改善。
9. **#10 GRPO 100 step (74.0%)**: 30 step が sweet spot。100 step では `pos_deforestation`
   が 84.6%→76.9% と落ち、over-train regression。

## 元データ

- 統合 JSON: `SatelliteAgent/kaggle/eval_consolidated.json`
- 各 kernel の生 output (Kaggle):
  - S43: `titanic12/prime-rl-s43-lfm2vl-v4-reward`
  - S46: `titanic12/sft-s46-lfm2vl-v4-visual`
  - S54: `titanic12/sft-s54-aug-mixup`
  - S56: `titanic12/etl-s56-rule-baseline`
  - S57: `titanic12/s57-gemini-head-to-head`
  - S64: `titanic12/s64-sft-grpo-no-image`
  - S65: `titanic12/s65-grpo-long`
- 取得コマンド: `kaggle kernels output <id> -p <dir> --file-pattern '.*\.json'`
