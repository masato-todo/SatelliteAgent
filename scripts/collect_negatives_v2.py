"""Auto-collect MORE negative scenarios (target +100 cases over v1).

Differences from v1 (`collect_negatives.py`):
  - **Append mode**: existing `negative_cases.yaml` is loaded first; new
    candidates whose tile/id already appears are skipped. The 16 v1 cases
    survive untouched.
  - **Wider region set**: ocean center / polar coast / island / desert core /
    high mountain regions added on top of v1's 14 biome regions.
  - **Per-candidate verify_no_change** with multiple season pairs
    (2021/2022/2023/2024 × spring↔autumn) — v1 only tried 2 pairs.
  - **Looser thresholds for biome=ocean** (background can be dark, so
    `nodata < 0.5` instead of 0.3) — pluggable via REGION-level config.
  - **Mid-save**: yaml is rewritten every `--save-every` cases so a kill
    mid-run preserves work.

Output: `data/metadata/disaster_m3/negative_cases.yaml` (default in-place).

Usage:
  # smoke (1 region, ~5 cases)
  python scripts/collect_negatives_v2.py --only-region ocean_pacific_central --target 5

  # full (~100 new cases on top of existing 16)
  python scripts/collect_negatives_v2.py --target 100
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml
from pystac_client import Client


SAT_BASE = "http://localhost:7860"
STAC_URL = "https://earth-search.aws.element84.com/v1"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "negative_cases.yaml"


# Season pairs to try, in priority order.
# Each entry = (before_date, after_date, datetime_range_for_stac, label).
SEASON_PAIRS = [
    ("2022-04-15", "2022-09-15", "2022-04-01/2022-09-30", "2022_spring_autumn"),
    ("2023-04-15", "2023-09-15", "2023-04-01/2023-09-30", "2023_spring_autumn"),
    ("2021-04-15", "2021-09-15", "2021-04-01/2021-09-30", "2021_spring_autumn"),
    ("2024-04-15", "2024-09-15", "2024-04-01/2024-09-30", "2024_spring_autumn"),
    # Southern-hemisphere swap (austral spring/autumn).
    ("2022-10-15", "2023-03-15", "2022-10-01/2023-03-31", "2022s_austral"),
]


# Region config:
#   id, biome, bbox=[W,S,E,N], n=target candidates from STAC,
#   max_cloud_threshold, max_nodata_threshold (looser for ocean)
SEARCH_REGIONS: list[dict[str, Any]] = [
    # ---- v1 14 regions (n bumped 3-5 → 6-8) ----
    {"id": "europe_west",            "biome": "urban_temperate",  "bbox": [-5, 43, 15, 55],  "n": 7},
    {"id": "europe_east",            "biome": "crop_temperate",   "bbox": [20, 45, 40, 55],  "n": 7},
    {"id": "north_america_central",  "biome": "crop_grassland",   "bbox": [-105, 35, -85, 48], "n": 8},
    {"id": "south_america_central",  "biome": "forest_tropical",  "bbox": [-65, -15, -45, -3], "n": 7},
    {"id": "north_africa",           "biome": "desert",           "bbox": [-5, 18, 30, 32],  "n": 7},
    {"id": "africa_central",         "biome": "forest_tropical",  "bbox": [12, -10, 35, 5],  "n": 6},
    {"id": "africa_south",           "biome": "savanna",          "bbox": [18, -28, 32, -20],"n": 6},
    {"id": "middle_east",            "biome": "desert",           "bbox": [38, 22, 55, 35],  "n": 6},
    {"id": "central_asia",           "biome": "steppe",           "bbox": [55, 40, 80, 50],  "n": 7},
    {"id": "south_asia",             "biome": "crop_subtropical", "bbox": [73, 22, 88, 30],  "n": 6},
    {"id": "east_asia",              "biome": "urban_subtropical","bbox": [105, 30, 125, 42],"n": 7},
    {"id": "se_asia",                "biome": "forest_tropical",  "bbox": [98, -2, 116, 12], "n": 6},
    {"id": "australia_inland",       "biome": "desert",           "bbox": [125, -28, 140, -22],"n": 6},
    {"id": "siberia",                "biome": "boreal",           "bbox": [70, 55, 130, 68], "n": 6},

    # ---- v2 newly added: ocean / polar / island / desert_core / mountain ----
    {"id": "ocean_pacific_central",  "biome": "ocean",            "bbox": [-160, -10, -130, 10], "n": 4, "max_nodata": 0.6},
    {"id": "ocean_atlantic_central", "biome": "ocean",            "bbox": [-40, -10, -20, 10],  "n": 4, "max_nodata": 0.6},
    {"id": "ocean_indian_central",   "biome": "ocean",            "bbox": [60, -15, 80, 5],     "n": 4, "max_nodata": 0.6},

    {"id": "polar_north",            "biome": "polar",            "bbox": [-70, 70, 30, 80],    "n": 4, "max_nodata": 0.5},
    {"id": "polar_south_coast",      "biome": "polar",            "bbox": [-60, -68, 0, -62],   "n": 3, "max_nodata": 0.5},

    {"id": "island_pacific",         "biome": "island_tropical",  "bbox": [155, -20, 175, -5],  "n": 4, "max_nodata": 0.5},
    {"id": "island_caribbean",       "biome": "island_tropical",  "bbox": [-78, 14, -60, 22],   "n": 4, "max_nodata": 0.5},
    {"id": "island_indonesia",       "biome": "island_tropical",  "bbox": [120, -8, 135, 0],    "n": 4, "max_nodata": 0.5},

    {"id": "desert_sahara_core",     "biome": "desert_core",      "bbox": [5, 22, 25, 28],      "n": 5},
    {"id": "desert_gobi_core",       "biome": "desert_core",      "bbox": [95, 41, 110, 45],    "n": 5},
    {"id": "desert_atacama",         "biome": "desert_core",      "bbox": [-71, -25, -68, -19], "n": 4},

    {"id": "mountain_himalaya",      "biome": "mountain",         "bbox": [80, 28, 95, 32],     "n": 4},
    {"id": "mountain_andes",         "biome": "mountain",         "bbox": [-72, -30, -68, -18], "n": 4},
    {"id": "mountain_rockies",       "biome": "mountain",         "bbox": [-115, 38, -105, 48], "n": 4},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 60.0  # overridden by --request-timeout


def fetch_pair(lat: float, lon: float, before_date: str, after_date: str,
               size_km: float, window_days: int) -> tuple[dict, dict]:
    body = {
        "lat": lat, "lon": lon,
        "before_date": before_date, "after_date": after_date,
        "size_km": size_km, "window_days": window_days,
    }
    r = requests.post(f"{SAT_BASE}/api/fetch", json=body, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    d = r.json()
    return d.get("before", {}), d.get("after", {})


def verify_no_change(lat: float, lon: float, size_km: float, window_days: int,
                     before_date: str, after_date: str,
                     max_cloud: float, max_nodata: float):
    try:
        b, a = fetch_pair(lat, lon, before_date, after_date, size_km, window_days)
    except Exception as e:
        return False, {"error": f"fetch_pair raised: {type(e).__name__}: {e}"}
    bm, am = b.get("meta", {}), a.get("meta", {})
    if not bm.get("image_available") or not am.get("image_available"):
        return False, {"reason": "image not available"}
    bk, ak = b.get("key"), a.get("key")
    if not (bk and ak):
        return False, {"reason": "missing keys"}
    cp_b = (bm.get("stats") or {}).get("cloud_proxy", 1.0)
    cp_a = (am.get("stats") or {}).get("cloud_proxy", 1.0)
    nd_b = (bm.get("stats") or {}).get("nodata_fraction", 0.0)
    nd_a = (am.get("stats") or {}).get("nodata_fraction", 0.0)
    if cp_b > max_cloud or cp_a > max_cloud:
        return False, {"reason": f"too cloudy (b={cp_b:.2f}, a={cp_a:.2f})"}
    if nd_b > max_nodata or nd_a > max_nodata:
        return False, {"reason": f"too much nodata (b={nd_b:.2f}, a={nd_a:.2f})"}
    return True, {
        "before_datetime": bm.get("datetime"),
        "after_datetime":  am.get("datetime"),
        "before_key": bk, "after_key": ak,
        "cloud_b": cp_b, "cloud_a": cp_a,
        "nodata_b": nd_b, "nodata_a": nd_a,
    }


# ---------------------------------------------------------------------------
# STAC discovery: pick S2 tile centroids from real items in each region
# ---------------------------------------------------------------------------

def stac_pick_tile_centers(region_bbox: list[float], n_wanted: int,
                            datetime_range: str, max_cloud_pct: float = 10.0):
    client = Client.open(STAC_URL)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=region_bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_pct}},
        max_items=300,
    )
    items = list(search.items())
    by_tile = {}
    for it in items:
        tile = it.properties.get("s2:mgrs_tile") or it.id
        if tile in by_tile:
            continue
        bb = it.bbox  # [W, S, E, N]
        by_tile[tile] = {
            "tile": tile,
            "lat": (bb[1] + bb[3]) / 2.0,
            "lon": (bb[0] + bb[2]) / 2.0,
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "datetime": it.datetime.isoformat() if it.datetime else None,
        }
        if len(by_tile) >= n_wanted:
            break
    return list(by_tile.values())


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_existing(path: Path) -> tuple[list[dict], set[str], set[tuple[float, float]]]:
    if not path.exists():
        return [], set(), set()
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cases = doc.get("cases", []) if isinstance(doc, dict) else []
    ids = {c.get("id") for c in cases if c.get("id")}
    coords = {(round(float(c["lat"]), 3), round(float(c["lon"]), 3))
              for c in cases if c.get("lat") is not None and c.get("lon") is not None}
    return cases, ids, coords


def save_yaml(path: Path, cases: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "cases": cases,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_one_region(region: dict, existing_ids: set[str],
                        existing_coords: set[tuple[float, float]],
                        target_remaining: int,
                        max_cloud: float, default_max_nodata: float,
                        size_km: float, window_days: int,
                        max_consecutive_fail: int = 6,
                        max_seasons_per_candidate: int = 2) -> Iterable[dict]:
    rmax_nodata = float(region.get("max_nodata", default_max_nodata))
    print(f"\n=== {region['id']:30s} biome={region['biome']:18s} bbox={region['bbox']} n_wanted={region['n']} ===", flush=True)
    candidates: list[dict] = []
    seen_tiles: set[str] = set()
    for (b_date, a_date, dt_range, label) in SEASON_PAIRS:
        if len(candidates) >= region["n"] * 2:
            break
        try:
            centers = stac_pick_tile_centers(region["bbox"], region["n"] * 2, dt_range)
        except Exception as e:
            print(f"  [{label}] STAC search FAIL: {e}", flush=True)
            continue
        new_centers = [c for c in centers if c["tile"] not in seen_tiles]
        seen_tiles.update(c["tile"] for c in new_centers)
        if not new_centers:
            continue
        print(f"  [{label}] +{len(new_centers)} new tile candidates", flush=True)
        candidates.extend([(c, b_date, a_date, label) for c in new_centers])
    if not candidates:
        return

    yielded = 0
    consecutive_fail = 0
    for c, b_date, a_date, label in candidates:
        if target_remaining <= 0:
            return
        if consecutive_fail >= max_consecutive_fail:
            print(f"  [give up] {region['id']}: {consecutive_fail} consecutive failures, moving on", flush=True)
            return
        case_id = f"neg__no_change__{region['id']}__{c['tile']}"
        if case_id in existing_ids:
            print(f"  [skip dup id] {case_id}", flush=True)
            continue
        coord_key = (round(c["lat"], 3), round(c["lon"], 3))
        if coord_key in existing_coords:
            print(f"  [skip dup coord] {coord_key}", flush=True)
            continue

        # Try this season pair first; if it fails, fall through to a few others.
        # Cap to max_seasons_per_candidate to avoid burning hours on one stubborn tile.
        attempted = []
        ok = False; info: dict[str, Any] = {}
        season_order = [(b_date, a_date, label)] + [
            (sb, sa, sl) for (sb, sa, _r, sl) in SEASON_PAIRS if sl != label
        ]
        for try_b, try_a, try_label in season_order[:max_seasons_per_candidate]:
            attempted.append(try_label)
            ok, info = verify_no_change(c["lat"], c["lon"], size_km, window_days,
                                         try_b, try_a, max_cloud, rmax_nodata)
            if ok:
                b_date, a_date = try_b, try_a
                label = try_label
                break

        tag = f"tile={c['tile']:8s} lat={c['lat']:+7.3f} lon={c['lon']:+8.3f}"
        if not ok:
            print(f"  FAIL {tag} after seasons {attempted}: {info.get('reason') or info.get('error','?')}", flush=True)
            consecutive_fail += 1
            continue
        print(f"  OK   {tag} season={label} cp_b={info['cloud_b']:.2f} cp_a={info['cloud_a']:.2f} nd_b={info['nodata_b']:.2f} nd_a={info['nodata_a']:.2f}", flush=True)
        existing_ids.add(case_id)
        existing_coords.add(coord_key)
        target_remaining -= 1
        yielded += 1
        consecutive_fail = 0
        yield {
            "id":              case_id,
            "parent_event":    None,
            "negative_type":   "no_change",
            "biome":           region["biome"],
            "expected_action": "drop",
            "lat":             round(c["lat"], 4),
            "lon":             round(c["lon"], 4),
            "size_km":         size_km,
            "before_date":     b_date,
            "after_date":      a_date,
            "window_days":     window_days,
            "stac_tile":       c["tile"],
            "season_label":    label,
            "note":            f"{region['biome']} ({region['id']}, mgrs {c['tile']})",
            "verified":        info,
        }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--target", type=int, default=100,
                   help="Stop after collecting this many NEW cases")
    p.add_argument("--only-region", default=None,
                   help="Restrict to one region id (debug/smoke)")
    p.add_argument("--save-every", type=int, default=5,
                   help="Re-write yaml every N successful cases")
    p.add_argument("--size-km", type=float, default=50.0)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--max-cloud", type=float, default=0.5)
    p.add_argument("--max-nodata", type=float, default=0.3,
                   help="Default per-side nodata fraction max (regions can override)")
    p.add_argument("--request-timeout", type=float, default=60.0,
                   help="Per-fetch timeout (s). Lower = faster fail; higher = more patient")
    p.add_argument("--max-seasons", type=int, default=2,
                   help="Max season pairs to try per candidate before giving up")
    p.add_argument("--max-fail", type=int, default=6,
                   help="Per-region: bail after this many consecutive verify failures")
    args = p.parse_args()

    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = float(args.request_timeout)

    out_path = Path(args.out)
    cases, existing_ids, existing_coords = load_existing(out_path)
    print(f"[start] existing: {len(cases)} cases (ids: {len(existing_ids)})")

    regions = SEARCH_REGIONS
    if args.only_region:
        regions = [r for r in SEARCH_REGIONS if r["id"] == args.only_region]
        if not regions:
            print(f"FAIL: --only-region {args.only_region} not found", file=sys.stderr)
            return 2

    new_added = 0
    target_remaining = args.target
    save_pending = 0
    try:
        for region in regions:
            if target_remaining <= 0:
                break
            for new_case in collect_one_region(
                region, existing_ids, existing_coords, target_remaining,
                max_cloud=args.max_cloud, default_max_nodata=args.max_nodata,
                size_km=args.size_km, window_days=args.window_days,
                max_consecutive_fail=args.max_fail,
                max_seasons_per_candidate=args.max_seasons,
            ):
                cases.append(new_case)
                new_added += 1
                target_remaining -= 1
                save_pending += 1
                if save_pending >= args.save_every:
                    save_yaml(out_path, cases)
                    print(f"  [save] {len(cases)} total ({new_added} new) → {out_path}")
                    save_pending = 0
    except KeyboardInterrupt:
        print("\n[interrupt] saving partial results...", file=sys.stderr)
    except Exception as e:
        print(f"\n[fatal] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        save_yaml(out_path, cases)
        print(f"\n[done] +{new_added} new cases (total {len(cases)}) → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
