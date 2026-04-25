# リモートサーバ deploy 手順

GRPO 学習用キャッシュを大量に保管するため、SimSat + SatelliteAgent を別Linuxサーバ上で動かす運用。Phase 3 (cache build) と Phase 5 (training) は同じサーバで完結する。

## 前提

- Linux + Docker (Compose v2) インストール済み
- Python 3.11+ + venv
- インターネット接続 (Element84 STAC へアクセス)
- リポジトリ clone 権限

## 1. リポジトリ取得

```bash
ssh user@gpu-host
cd /opt   # 任意の場所
git clone <YOUR_REPO_URL> mz_ai_appmaker
cd mz_ai_appmaker
```

## 2. SimSat 起動

```bash
cd SimSat
docker compose up -d --build sim
docker ps --filter name=fakesat-sim
# ヘルスチェック
curl -s http://localhost:9005/ | head -c 100
# → {"message":"Simulation API is online"}
```

## 3. SatelliteAgent 環境構築

```bash
cd ../SatelliteAgent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # FastAPI / pillow / numpy / yaml / etc
pip install pystac_client                # STAC discovery 用 (negative collection)
```

## 4. キャッシュ保管先 (大容量ディスク) 指定

```bash
# 例: /data/sat_cache が大容量 SSD/HDD のマウントポイント
mkdir -p /data/sat_cache /data/sat_traces

# 環境変数でリダイレクト
export SAT_CACHE_DIR=/data/sat_cache
export SAT_TRACES_DIR=/data/sat_traces
```

`.env` などで永続化する場合:
```bash
cat > .env <<'EOF'
SAT_CACHE_DIR=/data/sat_cache
SAT_TRACES_DIR=/data/sat_traces
SIMSAT_API_URL=http://localhost:9005
GOOGLE_API_KEY=...   # 必要なら
EOF
```

## 5. SatelliteAgent 起動

```bash
.venv/bin/python -m app.server
# 起動ログで確認:
# [startup] CACHE_DIR = /data/sat_cache
# [startup] TRACES_DIR = /data/sat_traces
# [startup] DisasterM3 cases loaded: 96 (positive/neutral=80, negative=16)
```

別端末で疎通確認:
```bash
curl -s http://localhost:7860/api/disasterm3/cases | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'cases={d[\"count\"]}')"
```

## 6. Cache pre-warm

全 96 ケース (positive 80 + negative 16) の Before/After を **50km × 10m** で pre-fetch:

```bash
cd SatelliteAgent
.venv/bin/python scripts/prewarm_cache.py
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
nohup .venv/bin/python scripts/prewarm_cache.py > prewarm.log 2>&1 &
```

容量目安:
- 確定ペアのみ: ~96 × 2 × 30MB ≈ **5GB**
- 派生 (compute_index_delta 等含む) も増えれば: **10-20GB**

## 7. ローカル PC からの確認

UI 操作はローカルでも可能。SAT_BASE を remote に向ければ、cache hit は remote 経由 = HTTP 越し:

```bash
# ローカル開発機の Webブラウザで:
http://gpu-host:7860/

# or SSH トンネル:
ssh -L 7860:localhost:7860 user@gpu-host
# → http://localhost:7860/ でアクセス
```

`/api/image/{key}` 経由で 30MB PNG が転送されるので、LAN なら ~1秒、WAN なら数秒。

## 8. GRPO 学習時 (Phase 5)

学習スクリプトは remote の `/data/sat_cache` をローカル disk read で参照:
```python
from PIL import Image
img = Image.open(f"/data/sat_cache/{key}.png")
```

SimSat / SatelliteAgent 経由不要、I/O は最速。

---

## トラブルシュート

### SimSat fetch が遅い / timeout
- Element84 STAC への帯域が足りない → 国内ミラーや帯域確認
- 50km × 10m は重い → `--resolution 30` に下げる:
  ```bash
  python scripts/prewarm_cache.py --resolution 30
  ```

### キャッシュが書き込まれない
- `SAT_CACHE_DIR` の権限確認: `ls -la /data/sat_cache`
- 起動ログで `[startup] CACHE_DIR = ...` が想定通りか確認

### 同時実行で重複 fetch
- prewarm script は逐次実行 (1並列)
- 高速化したい場合 `--workers N` を script に追加 (TBD)

---

## ローカル → リモート移行 (もし既にローカルで集めた cache あれば)

`scripts/sync_cache.sh` で rsync 転送:
```bash
REMOTE=user@gpu-host:/data/sat_cache ./scripts/sync_cache.sh
```

増分転送なので何度叩いても OK。
