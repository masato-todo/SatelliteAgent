# SatelliteAgent

オンボード LFM2-VL エージェントが ReAct ループで衛星ダウンリンク帯域を最適化する PoC。
1084 件の DisasterM3 / FireEdge GT に対して 3 種類の VLM (Gemini / 局所 vLLM 1.6B / 学習済 LFM2.5-VL-450M-sft-grpo)
を切替えて走らせ、Before/After Sentinel-2 frame の change classification → submit_to_ground / drop を判定する。

設計詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、再現手順の全体ロードマップは [docs/REPRO_PLAN.md](docs/REPRO_PLAN.md) 参照。

---

## 必要なもの

- Linux (動作確認は Ubuntu 24.04 / aarch64 GB10)
- Python 3.10+ (`uv` 推奨。`pip install uv` で入る)
- Docker (SimSat と LFM サービスをコンテナで起動するため)
- GPU (LFM 推論サーバを動かす場合のみ。Gemini path だけなら不要)

## クローン直後のセットアップ

```bash
git clone https://github.com/masato-todo/SatelliteAgent.git
cd SatelliteAgent
uv sync --extra simsat --extra geo
```

これでアプリ本体の Python 環境ができる。`data/` 以下は metadata 一式 (~6MB)
が同梱済みなので追加ダウンロード不要。

## 必要な 3 サービス

| サーバ | port | 役割 |
|---|---:|---|
| **SimSat** | 9005 | Sentinel-2 mock backend (lat/lon/timestamp → S2 image) — `vendor/SimSat` をクローンしてパッチ適用、`patches/simsat/README.md` 参照 |
| **wildfire LoRA** | 8085 | `detect_wildfire` ツールが叩く FireEdge LoRA (transformers + peft) |
| **LFM2 agent vLLM** | 8086 | 学習済 LFM2.5-VL-450M-sft-grpo (Run Agent の `lfm25_vl_sft_grpo` provider) |

LoRA + vLLM の 2 つはルートの `docker-compose.yaml` でまとめて立てられる:

```bash
# 必須 env を .env に書く (or 同コマンドの直前で export)
echo "WILDFIRE_MODEL_DIR=$HOME/path/to/wildfire-staging"  >> .env
echo "LFM2_AGENT_MODEL_DIR=$HOME/path/to/sft-grpo-ckpt" >> .env

docker compose up -d              # build + 起動 (~5 分)
docker compose logs -f lfm2-agent # vLLM の load 完了待ち
```

詳細は `docker-compose.yaml` 上部のコメントと `services/agent/Dockerfile`、`docs/INTEGRATION_LFM2VL.md` 参照。

(任意: Settings ⚙ で **Gemini を使う場合は `GOOGLE_API_KEY` を `.env` に**。
ローカル vLLM 1.6B も別途使いたい場合は port 8002 に立てる、これは `config/providers.yaml` の `lfm25_vl_local` 既定エントリ。)

## アプリ起動

```bash
uv run python -m app.server
```

ブラウザで <http://localhost:7860> を開く。

## ブラウザ操作 (動作確認の最短手順)

1. 左サイドバー **DisasterM3 case** ドロップダウンから例えば 🔥 FireEdge GT 内の `fireedge_train_firms_pos_001` を選択
2. **🔥 Fetch FireEdge frame** をクリック (= sentinel_datetime + window=1d で訓練時の S2 frame をピン)
3. After map に画像が表示されたら `detect_wildfire 🔥` ツールボタン → fire_detected: true が返れば成功
4. 上部 ⚙ Settings で provider を `lfm25_vl_sft_grpo` に切替 → **Run Agent** で multi-turn agent が逐次 SSE で trace を出力

## 環境変数 (任意)

通常は不要。以下は使うシナリオがある時だけ `.env` に書く:

```bash
# Settings ⚙ で Gemini を選ぶ場合のみ
GOOGLE_API_KEY=...

# scripts/collect_firms_fire.py で FIRMS データを再収集する場合のみ
FIRMS_MAP_KEY=...

# SimSat がリモートにある場合のみ (デフォルト http://localhost:9005)
SIMSAT_API_URL=http://...
```

## ディレクトリ構成

```
app/                    FastAPI バックエンド + ES module フロント
  ├─ server.py          /api/fetch, /api/run_agent (SSE) ...
  └─ static/            index.html + app.css + js/
agent/                  ReAct ループ実装
  ├─ react_loop_openai.py  OpenAI 互換 (Gemini, 局所 vLLM 1.6B 共用)
  ├─ react_loop.py         google-genai SDK 経由の Gemini 経路
  ├─ lfm2_agent.py         学習済 LFM2.5-VL-450M-sft-grpo 用 multi-turn loop
  └─ lfm2_tool_parser.py   pythonic tool parser plugin (vLLM 用)
tools/                  Vision / spectral / region / wildfire / ...
simsat_client/          SimSat API wrapper
config/providers.yaml   VLM provider catalog (UI Settings dropdown が読む)
services/
  ├─ wildfire/          wildfire LoRA serve コンテナ (transformers+peft, port 8085)
  └─ agent/             LFM2.5-VL-450M-sft-grpo serve コンテナ (vLLM, port 8086)
scripts/                データ収集 / 評価 / Docker 起動補助
data/                   selectively tracked — see [data/README.md](data/README.md)
docs/
  ├─ REPRO_PLAN.md      再現性整備のフェーズ別計画
  ├─ ARCHITECTURE.md
  ├─ INTEGRATION_LFM2VL.md   vLLM cu130 + transformers 5.5.0 + sed patch
  └─ DATASET_V3.md
patches/simsat/         SimSat fork へのローカルパッチ (タイル境界 mosaic 対応 ほか)
```

## License

MIT
