# リモートサーバ deploy 手順

GRPO 学習用キャッシュを大量に保管するため、SimSat + SatelliteAgent を別Linuxサーバ上で動かす運用。Phase 3 (cache build) と Phase 5 (training) は同じサーバで完結する。

**キャッシュ配置方針:**
- ローカル / リモートどちらも `SatelliteAgent/data/scenarios/` (デフォルト) に置く
- ローカルで集めた cache は `scripts/sync_cache.sh` (rsync) でリモートへ転送
- リモートで追加 fetch しても同じ場所に積まれる → 双方向で増分同期可能

## 前提

- Linux + Docker (Compose v2) インストール済み
- Python 3.10+ と [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- インターネット接続 (Element84 STAC へアクセス)
- リポジトリ clone 権限

## 1. リポジトリ取得

```bash
ssh user@gpu-host
cd /opt   # 任意の場所

# SatelliteAgent (本リポジトリ)
git clone https://github.com/masato-todo/SatelliteAgent.git
cd SatelliteAgent
git checkout Branch/refactor   # 現状の作業ブランチ
cd ..

git clone https://github.com/DPhi-Space/SimSat.git
```

## 2. SimSat 起動

```bash
cd SimSat
docker compose up -d --build sim
docker ps --filter name=fakesat-sim
# ヘルスチェック
curl -s http://localhost:9005/ | head -c 100
# → {"message":"Simulation API is online"}
cd ..
```

## 3. SatelliteAgent 環境構築

```bash
cd SatelliteAgent
uv sync --extra simsat   # FastAPI / pillow / numpy / yaml / pystac-client / rasterio / shapely
```

`uv sync` で `.venv/` が自動生成され、`uv.lock` 通りに依存をインストール。`--extra simsat` は negative collection で使う `pystac_client` 等を含む。

## 4. SatelliteAgent 起動

cache はデフォルトで `SatelliteAgent/data/scenarios/`、trace は `SatelliteAgent/data/traces/human/` に保存される (環境変数で上書きしない限り)。

```bash
uv run python -m app.server
# 起動ログで確認:
# [startup] CACHE_DIR = .../SatelliteAgent/data/scenarios
# [startup] TRACES_DIR = .../SatelliteAgent/data/traces/human
# [startup] DisasterM3 cases loaded: 96 (positive/neutral=80, negative=16)
```

別端末で疎通確認:
```bash
curl -s http://localhost:7860/api/disasterm3/cases | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'cases={d[\"count\"]}')"
```

## 5. ローカルで Cache pre-warm

全 96 ケース (positive 80 + negative 16) の Before/After を **50km × 10m** で pre-fetch。
**ローカル機 (or リモートどちらでも)** で実行可:

```bash
cd SatelliteAgent
uv run python scripts/prewarm_cache.py
```

ログ例:
```
[1/3] SAT_BASE = http://localhost:7860
[2/3] Fetching DM3 case list ...
     -> 96 cases queued
[3/3] Pre-warming Before/After at 10m ...
  [  1/96] xbd_santa_rosa_wildfire_00000063     OK   42.1s  B[N] cloud=0.00 nd=0.00  A[N] cloud=0.00 nd=0.00
  [  2/96] xbd_santa_rosa_wildfire_00000079     OK   38.3s  B[N] cloud=0.00 nd=0.00  A[N] cloud=0.00 nd=0.00
  ...
[done] 96 cases in 4280s wall, 4280s aggregate fetch
       OK: 92   FAIL: 4   already cached: 0
```

時間目安: 1ケース ~30-60秒 cold fetch、96ケース で **~1.5時間**。バックグラウンド推奨:
```bash
nohup uv run python scripts/prewarm_cache.py > prewarm.log 2>&1 &
```

容量目安:
- 確定ペアのみ: ~96 × 2 × 30MB ≈ **5GB**
- 派生 (compute_index_delta 等含む) も増えれば: **10-20GB**

## 6. ローカル → リモートへ rsync

ローカルで集めた cache をリモートの **同じパス (`SatelliteAgent/data/scenarios/`)** に転送:

```bash
# ローカル機で
cd SatelliteAgent
REMOTE=user@gpu-host:/opt/SatelliteAgent/data/scenarios ./scripts/sync_cache.sh
```

`sync_cache.sh` は `--update --partial-dir` 付き rsync なので:
- 既にリモートにあるファイルは送らない (増分転送)
- 中断しても次回 resume

何度叩いても OK。リモート側で追加 fetch しても上書きしない。

DRYRUN で送信予定だけ確認:
```bash
REMOTE=user@gpu-host:/opt/SatelliteAgent/data/scenarios DRYRUN=1 ./scripts/sync_cache.sh
```

## 7. ローカル PC からの UI 確認

UI 操作はローカル機ブラウザで可。リモートの SatelliteAgent に向ければ、cache hit は HTTP 越し:

```bash
# SSH トンネル経由
ssh -L 7860:localhost:7860 user@gpu-host
# → http://localhost:7860/ でアクセス
```

`/api/image/{key}` 経由で 30MB PNG が転送されるので、LAN なら ~1秒、WAN なら数秒。

## 8. GRPO 学習時 (Phase 5)

学習スクリプトは remote の `SatelliteAgent/data/scenarios/` をローカル disk read で参照:
```python
from pathlib import Path
from PIL import Image
CACHE = Path("/opt/SatelliteAgent/data/scenarios")
img = Image.open(CACHE / f"{key}.png")
```

SimSat / SatelliteAgent 経由不要、I/O は最速。

---

## トラブルシュート

### SimSat fetch が遅い / timeout
- Element84 STAC への帯域が足りない → 国内ミラーや帯域確認
- 50km × 10m は重い → `--resolution 30` に下げる:
  ```bash
  uv run python scripts/prewarm_cache.py --resolution 30
  ```
  ※ ただし resolution が変わると cache key も変わる (10m と 30m は別ファイル) ので、
  GRPO 用は 10m で確定させる。30m は debug/低帯域時のみ。

### キャッシュが書き込まれない
- `data/scenarios/` の権限確認: `ls -la SatelliteAgent/data/scenarios`
- 起動ログで `[startup] CACHE_DIR = ...` が想定通りか確認

### 同時実行で重複 fetch
- prewarm script は逐次実行 (1並列)
- 高速化したい場合 `--workers N` を script に追加 (TBD)

### キャッシュ場所を変えたい (大容量別ディスクに置く等)
デフォルト (`SatelliteAgent/data/scenarios/`) で困った時のみ環境変数で上書き:
```bash
export SAT_CACHE_DIR=/mnt/nvme/sat_cache
export SAT_TRACES_DIR=/mnt/nvme/sat_traces
```
ただしリモート ↔ ローカルでパスが食い違うと rsync 先も合わせて変更が必要。
**通常はデフォルトのままで運用するのが管理が楽。**
