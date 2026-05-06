#!/usr/bin/env bash
# setup.sh — first-run bootstrap for SatelliteAgent.
#
# Idempotent: safe to re-run. Each step is skipped if its output is
# already in place.
#
# Steps:
#   1. Verify uv is installed.
#   2. Create .venv and install Python deps via uv sync.
#   3. Create an empty .env template if one isn't there yet.
#   4. (Optional, WITH_SIMSAT=1) Clone DPhi-Space/SimSat at the pinned
#      SHA into vendor/SimSat and apply patches/simsat/*.patch.
#   5. docker compose up -d on the two GPU services
#      (assumes WILDFIRE_MODEL_DIR / LFM2_AGENT_MODEL_DIR are set in
#      .env or already exported — see scripts/download_models.sh).
#
# Usage:
#   ./setup.sh                     # core install only
#   WITH_SIMSAT=1 ./setup.sh       # also clone+patch+run SimSat
#
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/5] checking uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "  ERROR: uv is not installed."
    echo "  Install it from https://github.com/astral-sh/uv (e.g. 'pip install uv')"
    exit 1
fi
echo "  ok ($(uv --version))"

echo "[2/5] uv sync..."
uv sync --extra simsat --extra geo

echo "[3/5] .env template..."
if [ -f .env ]; then
    echo "  .env already exists — leaving it alone"
else
    cat > .env <<'EOF'
# Optional — only when Settings ⚙ → Provider = Gemini
# GOOGLE_API_KEY=

# Optional — only when running scripts/collect_firms_fire.py
# FIRMS_MAP_KEY=

# Optional override (default: http://localhost:9005)
# SIMSAT_API_URL=http://localhost:9005

# Required by docker-compose.yaml — see scripts/download_models.sh
# WILDFIRE_MODEL_DIR=./models/wildfire-staging
# LFM2_AGENT_MODEL_DIR=./models/sft-grpo
EOF
    echo "  wrote .env (all keys commented out — uncomment as needed)"
fi

echo "[4/5] SimSat (optional)..."
if [[ "${WITH_SIMSAT:-0}" == "1" ]]; then
    SIMSAT_SHA="52f5619330c1edbb2e330b2961a1a551bebc0d69"
    if [ ! -d vendor/SimSat ]; then
        echo "  cloning DPhi-Space/SimSat into vendor/SimSat..."
        mkdir -p vendor
        git clone https://github.com/DPhi-Space/SimSat.git vendor/SimSat
    fi
    pushd vendor/SimSat >/dev/null
    git fetch --quiet origin
    git checkout --quiet "$SIMSAT_SHA"
    if git apply --check ../../patches/simsat/*.patch >/dev/null 2>&1; then
        git apply ../../patches/simsat/*.patch
        echo "  applied patches/simsat/*.patch"
    else
        echo "  patches already applied (or repo dirty) — skipping"
    fi
    popd >/dev/null
    echo "  starting SimSat container on :9005..."
    docker compose -f vendor/SimSat/docker-compose.yaml up -d sim
else
    echo "  skipped (set WITH_SIMSAT=1 to enable)."
    echo "  If you already have a reachable SimSat, set SIMSAT_API_URL in .env."
fi

echo "[5/5] GPU services (docker compose up -d)..."
if grep -qE '^[[:space:]]*WILDFIRE_MODEL_DIR=' .env 2>/dev/null \
   || [[ -n "${WILDFIRE_MODEL_DIR:-}" ]]; then
    docker compose up -d
else
    echo "  WILDFIRE_MODEL_DIR / LFM2_AGENT_MODEL_DIR are not set yet."
    echo "  Run ./scripts/download_models.sh first, then 'docker compose up -d'."
fi

cat <<'EOF'

Setup complete. Next:

    ./scripts/download_models.sh        # ~3 GB pull from HF Hub
    docker compose up -d                # start GPU services (if not done above)
    uv run python -m app.server         # start the app

then open http://localhost:7860
EOF
