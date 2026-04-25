# SatelliteAgent

オンボード LFM2-VL エージェントが ReAct ループで衛星ダウンリンク帯域を最適化する。
設計詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 参照。

## Quickstart (ローカル)

```bash
uv sync --extra simsat
cp .env.example .env       # GOOGLE_API_KEY を設定

# SimSat 起動 (別シェル)
cd SimSat && docker compose up -d --build sim

# SatelliteAgent 起動
uv run python -m app.server
```

ブラウザで http://localhost:7860 → DM3 96 ケース (xBD 80 + negative 16) から Before/After を選んで `Run Agent`。

## Quickstart (リモート GPU サーバ)

[docs/REMOTE_DEPLOY.md](docs/REMOTE_DEPLOY.md) 参照。
ローカルからは `ssh -L 7860:localhost:7860 user@gpu-host` でトンネル → http://localhost:7860 でブラウザアクセス。

## データ

`data/` は git 管理外。bootstrap (DM3 metadata 配布、cache prewarm 等) は [docs/REMOTE_DEPLOY.md](docs/REMOTE_DEPLOY.md) の手順に従う。

## ディレクトリ構成

```
app/
  ├─ server.py        FastAPI: /api/fetch, /api/run_agent (SSE), /api/disasterm3/cases ...
  └─ static/
      ├─ index.html
      ├─ app.css
      └─ js/          ES modules (state-utils, maps, tools, dm3-fetch, annotate-traces, main)
agent/                Orchestrator (ReAct loop, LLM providers)
tools/                ツール層 (vision / context / budget / action / quality)
simsat_client/        SimSat API wrapper
scripts/
  ├─ prewarm_cache.py    DM3 96 ケースを 50km/10m で pre-fetch
  ├─ collect_negatives.py STAC 経由で negative シナリオ収集
  └─ sync_cache.sh       ローカル → リモート rsync
data/                 (gitignore) scenarios/, metadata/, traces/, region_db/
eval/                 評価ハーネス
models/               (gitignore) FT adapter
docs/
```

## License

MIT
