#!/usr/bin/env bash
# Recreate stage/ from SatelliteAgent data/ and upload to Kaggle as
# <KAGGLE_USER>/satelliteagent-raw-v3.
#
# v3 adds GDACS volcanic + PRODES deforestation + hard-negative cases on
# top of the v2 (MCD64A1 wildfire + biome-diverse negative). flood/storm/
# quake/landslide were tried via EMS but excluded — S2 signal too weak.
# This is the *raw* dataset (Phase 1-3 outputs as-is). Kaggle notebooks do
# their own resize / format conversion so we don't have to re-upload when
# tuning resolution or splits.
#
# Usage:
#   cd kaggle/exp003
#   ./upload.sh                # first time → datasets create
#   UPDATE=1 ./upload.sh       # subsequent → datasets version

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
SRC="$PROJ_ROOT/data"
STAGE="$EXP_DIR/stage"

echo "[1/3] Refreshing $STAGE from $SRC"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$SRC/curated_pairs"      "$STAGE/"
cp    "$SRC/canonical_dataset.yaml" "$STAGE/"
[ -f "$SRC/scene_catalog.yaml" ] && cp    "$SRC/scene_catalog.yaml"     "$STAGE/" || true
[ -d "$SRC/gt_polygons" ]        && cp -r "$SRC/gt_polygons"            "$STAGE/" || true
cp -r "$SRC/traces/agent"        "$STAGE/traces"
cp    "$EXP_DIR/dataset-metadata.json" "$STAGE/"

echo "[2/3] Stage contents:"
du -sh "$STAGE"/* | sed 's/^/  /'
echo "  curated scenes : $(ls "$STAGE/curated_pairs" | wc -l)"
echo "  agent traces   : $(ls "$STAGE/traces" | wc -l)"

echo "[3/3] Uploading to Kaggle ..."
if [ "${UPDATE:-0}" = "1" ]; then
  kaggle datasets version -p "$STAGE" -m "${MSG:-update}" --dir-mode zip
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
echo "Done."
