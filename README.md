# SatelliteAgent

AI in Space Hackathon (Liquid AI × DPhi Space, 2026-04-13 〜 2026-05-06) 提出作品。

## 概要

オンボード LFM2-VL エージェントが ReAct ループでツール群を呼び出し、衛星のダウンリンク帯域を自律最適化する。

- **Orchestrator**: LFM2-VL (ReAct で意思決定)
- **Specialist**: LFM2-VL (変化検知分類、LoRA adapter 切替)
- **Tools**: 11種 (vision / context / budget / action)
- **Platform想定**: NVIDIA Orin 16GB
- **UI**: FastAPI + Leaflet Mission Control Dashboard

設計の全体は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) を参照。

## Quickstart

```bash
# 依存インストール
uv sync
# または: pip install -e .

# .env 準備
cp .env.example .env
# ANTHROPIC_API_KEY を設定 (Phase 1 は Claude Opus で ReAct をドライブ)

# SimSat 起動 (別シェル、SimSat/ の docker-compose up -d)
# サーバー起動
python -m app.server
```

ブラウザで http://localhost:7860 を開く。起動時に自動で Sentinel-2 を取得、Template 切替 / 座標 / 日付 / Footprint / 検索ウィンドウが全てUIで調整可能。Run Agent で ReAct トレースを SSE で流す。

## ディレクトリ構成

```
app/                  FastAPI + Leaflet Mission Control Dashboard
  ├─ server.py        FastAPI エンドポイント (/api/fetch, /api/run_agent SSE)
  └─ static/          index.html, app.js (Leaflet), app.css
agent/                Orchestrator (ReAct loop, LLM providers)
  ├─ react_loop.py
  ├─ providers.py     Claude / LFM2-VL 差し替え層
  └─ prompts/
tools/                ツール層
  ├─ schema.py        JSONSchema 定義
  ├─ stubs.py         Phase 1 モック実装
  ├─ vision.py        classify_change, fetch_band, zoom_in
  ├─ context.py       get_region_info, get_history, compute_area
  ├─ budget.py        check_downlink_budget, estimate_size
  ├─ action.py        compose_report, submit_to_ground, drop
  └─ validator.py     schema validation + retry + fallback
simsat_client/        SimSat API wrapper (Phase 2)
data/
  ├─ scenarios/       テストシナリオ (flood / fire / deforestation)
  └─ region_db/
eval/                 Bandwidth saving など評価ハーネス
models/               FT adapter (gitignore、HF Hub 配布)
docs/
  ├─ ARCHITECTURE.md
  ├─ TOOL_SPEC.md
  └─ DEMO_SCENARIOS.md
```

## 開発フェーズ

1. **Phase 1 (現在)**: 全ツールをモック化 + Claude Opus で ReAct ループを駆動。UIと仕様の固め込み。
2. **Phase 2**: モックを実ツール (SimSat API, SQLite, classify_change 等) に順次差し替え。
3. **Phase 3**: Orchestrator を LFM2-VL (FT済 adapter) に差し替え。

## 役割分担

| 領域 | 担当 |
|---|---|
| `app/`, `tools/`, `agent/providers.py`, `simsat_client/`, `eval/` | コア実装 |
| `agent/react_loop.py` プロンプト設計, FTデータ生成 | チーム |
| `models/` (adapter 学習) | チーム |
| `docs/` | 共同 |

## License

MIT