#!/usr/bin/env bash
# scripts/smoke_test.sh — minimal end-to-end invariant check.
#
# Boots the app, fetches the FireEdge GT case fireedge_train_firms_pos_001,
# invokes detect_wildfire, and asserts that:
#   - the After-side cached SimSat datetime equals the case's
#     sentinel_datetime (= the exact training-time STAC item is pinned)
#   - detect_wildfire's own SimSat refetch returns the *same* datetime
#     (= eval_wildfire_hf_simsat.py --use-sentinel-datetime --window-days 1
#      same conditions)
#   - fire_detected = true
#
# Pass = the repo is wired correctly: SimSat 9005 reachable, wildfire
# LoRA 8085 reachable, mosaic patch present, detect_wildfire eval-parity
# binding intact.
#
# Pre-requisites that are NOT auto-checked:
#   - SimSat container or remote endpoint accepting requests
#   - wildfire LoRA container on :8085 (docker compose up -d)
#   - .venv populated (./setup.sh)
#
# Usage:
#   ./scripts/smoke_test.sh
#
# Exits 0 on success, non-zero on first failure with a message.
#
set -euo pipefail

cd "$(dirname "$0")/.."

APP_PORT="${APP_PORT:-7860}"
APP_URL="http://localhost:${APP_PORT}"
EXPECTED_SDT="2025-03-15T08:38:09Z"

# 1. Make sure dependencies look like they're up.
echo "[1/4] checking SimSat / wildfire LoRA reachability..."
curl -sf -m 3 "${SIMSAT_API_URL:-http://localhost:9005}/" >/dev/null \
    || { echo "  FAIL: SimSat unreachable. Start it (see patches/simsat/README.md)"; exit 2; }
curl -sf -m 3 "${LFM_WILDFIRE_BASE_URL:-http://localhost:8085/v1}/models" >/dev/null \
    || { echo "  FAIL: wildfire LoRA :8085 unreachable. 'docker compose up -d lfm-wildfire'"; exit 2; }

# 2. Boot the app server in the background.
echo "[2/4] booting app server on :${APP_PORT}..."
nohup env APP_PORT="${APP_PORT}" uv run python -m app.server \
    > /tmp/satelliteagent_smoke.log 2>&1 &
APP_PID=$!
trap 'kill "$APP_PID" 2>/dev/null || true' EXIT

deadline=$(( $(date +%s) + 60 ))
until curl -sf -m 2 "${APP_URL}/api/templates" >/dev/null 2>&1; do
    [[ $(date +%s) -gt $deadline ]] && {
        echo "  FAIL: app didn't come up within 60s"
        tail -30 /tmp/satelliteagent_smoke.log
        exit 3
    }
    sleep 1
done
echo "  ready."

# 3. Pre-fetch the FireEdge case via the dedicated FireEdge button path.
echo "[3/4] /api/fetch (FireEdge fire case)..."
FETCH_JSON=$(curl -sf -X POST "${APP_URL}/api/fetch" \
    -H 'Content-Type: application/json' \
    -d '{"lat":7.58296,"lon":29.53288,"before_date":"2024-09-16",
         "after_date":"2025-03-15T08:38:09Z","size_km":5.0,
         "window_days":1,"before_window_days":30}')

A_KEY=$(echo "$FETCH_JSON" | uv run python -c "import sys,json; print(json.load(sys.stdin)['after']['key'])")
B_KEY=$(echo "$FETCH_JSON" | uv run python -c "import sys,json; print(json.load(sys.stdin)['before']['key'])")
A_DT=$(echo "$FETCH_JSON"  | uv run python -c "import sys,json; print(json.load(sys.stdin)['after']['meta'].get('datetime',''))")

if [[ "$A_DT" != "$EXPECTED_SDT" ]]; then
    echo "  FAIL: After STAC datetime is '$A_DT', expected '$EXPECTED_SDT'"
    echo "        ↳ SimSat returned a different scene. Check the mosaic patch."
    exit 4
fi
echo "  After dt = $A_DT  (matches sentinel_datetime ✓)"

# 4. Run detect_wildfire and assert eval-parity.
echo "[4/4] /api/tool/invoke detect_wildfire..."
RESULT=$(curl -sf -X POST "${APP_URL}/api/tool/invoke" \
    -H 'Content-Type: application/json' \
    -d "{\"tool_name\":\"detect_wildfire\",\"arguments\":{\"which\":\"after\"},
         \"before_key\":\"$B_KEY\",\"after_key\":\"$A_KEY\"}")

FIRE=$(echo "$RESULT"  | uv run python -c "import sys,json; print(json.load(sys.stdin)['observation'].get('fire_detected'))")
WF_DT=$(echo "$RESULT" | uv run python -c "import sys,json; print((json.load(sys.stdin)['observation'].get('sentinel') or {}).get('datetime',''))")

if [[ "$WF_DT" != "$EXPECTED_SDT" ]]; then
    echo "  FAIL: wildfire SimSat datetime '$WF_DT' != expected '$EXPECTED_SDT'"
    echo "        ↳ detect_wildfire is not eval-parity. Check make_detect_wildfire."
    exit 5
fi
if [[ "$FIRE" != "True" ]]; then
    echo "  FAIL: fire_detected = $FIRE (expected True for fireedge_train_firms_pos_001)"
    exit 6
fi
echo "  fire_detected = $FIRE,  sentinel.dt = $WF_DT  ✓"
echo
echo "smoke: OK"
