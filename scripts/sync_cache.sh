#!/usr/bin/env bash
# Sync the local Sentinel-2 cache to a remote training server via rsync.
#
# Usage:
#   REMOTE=user@host:/path/to/sat_cache ./scripts/sync_cache.sh
#   REMOTE=user@host:/path/to/sat_cache DRYRUN=1 ./scripts/sync_cache.sh
#
# Defaults source to data/scenarios/ (override with LOCAL=).
# Adds --update so already-present remote files aren't re-sent.
# Adds --partial-dir so interrupted transfers can resume.

set -eu

LOCAL="${LOCAL:-data/scenarios}"
REMOTE="${REMOTE:?Set REMOTE=user@host:/path}"
DRYRUN="${DRYRUN:-}"

if [ ! -d "$LOCAL" ]; then
  echo "Local cache dir does not exist: $LOCAL" >&2
  exit 1
fi

LOCAL_SIZE=$(du -sh "$LOCAL" | cut -f1)
LOCAL_COUNT=$(find "$LOCAL" -type f | wc -l)
echo "Source: $LOCAL  ($LOCAL_COUNT files, $LOCAL_SIZE)"
echo "Target: $REMOTE"

FLAGS="-avz --update --partial-dir=.rsync-partial --info=progress2"
if [ -n "$DRYRUN" ]; then
  echo "DRYRUN: rsync $FLAGS --dry-run $LOCAL/ $REMOTE/"
  rsync $FLAGS --dry-run "$LOCAL/" "$REMOTE/"
else
  rsync $FLAGS "$LOCAL/" "$REMOTE/"
fi

echo "Done."
