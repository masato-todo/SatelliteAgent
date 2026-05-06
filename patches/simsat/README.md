# SimSat patches

Local fixes against [DPhi-Space/SimSat](https://github.com/DPhi-Space/SimSat)
(AGPL-3.0). Distributed here as `*.patch` files only — the upstream sources
themselves are NOT vendored, to avoid the AGPL copyleft propagating into
this project.

## Pinned upstream SHA

```
52f5619330c1edbb2e330b2961a1a551bebc0d69   (main, "updated readme", 2026-04)
```

Patches assume this exact commit. If upstream advances, the patches may
need to be re-baselined (`git apply --3way` or manual rebase).

## Patches

| # | File | What it does | Why we need it |
|---|---|---|---|
| 001 | `001-mosaic-tile-boundary.patch` | `src/sim/ImagingProviders/sentinel_provider.py`: group STAC items by acquisition date, pick the date whose union covers the AOI best, then mosaic with `odc.stac.load(items, groupby="solar_day")`. Also fix single-band PNG output to use mode="L". | Without it, AOIs straddling MGRS tile boundaries return ~50% nodata. The wildfire LFM's percentile-clip preprocessing breaks on those, dropping `eval_wildfire_hf_simsat.py` recall well below 0.933. |
| 002 | `002-fastapi-sync-endpoints.patch` | `src/sim/api.py`: switch four `async def` endpoints to `def`. | `odc.stac.load` is synchronous (S3 fetches block) and was running inside the asyncio loop, blocking the event loop. FastAPI runs sync `def` endpoints in a threadpool, fixing latency under load. |
| 003 | `003-compose-sim-only.patch` | `docker-compose.yaml`: drop the `dashboard` service and Django dependency, leave just `sim` exposed on port 9005. | We only need the `sim` service for SatelliteAgent. The full FakeSat dashboard adds a Django container plus volumes that we never use. |

## Apply

```bash
# Optional setup — only needed if you want SimSat running locally.
SIMSAT_SHA=52f5619330c1edbb2e330b2961a1a551bebc0d69
git clone https://github.com/DPhi-Space/SimSat.git vendor/SimSat
(cd vendor/SimSat \
  && git checkout "$SIMSAT_SHA" \
  && git apply ../../patches/simsat/*.patch)
docker compose -f vendor/SimSat/docker-compose.yaml up -d sim
```

If you can already point at a reachable SimSat instance (set
`SIMSAT_API_URL` in `.env` to that URL), this whole step is optional.

## License note

The unmodified `001/002/003` diffs target AGPL-3.0 source. Distributing
diffs ("derivative work" interpretation is debated for tiny patches but
strict reading is permissive) is consistent with AGPL — the *applied*
copy ends up under AGPL and lives only in the per-developer
`vendor/SimSat/` clone, which is git-ignored from this repo. Nothing
under AGPL is committed here.
