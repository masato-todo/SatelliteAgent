"""Auto-collect negative scenarios for the SFT/GRPO dataset.

Strategy: query STAC (Element84 sentinel-2-l2a) for actual S2 items in
diverse regions, use each item's centroid as a guaranteed-tile-center
lat/lon. Then probe SimSat to confirm the image fetch yields low cloud +
low nodata. This avoids the "lat/lon picked arbitrarily → SimSat returns
half-empty PNG" problem.

Two categories produced:
  - no_change: ~50 cases, biome-diverse, geographically spread.
               Date pair: 6 months apart in 2022. Verified low cloud + nodata.
  - cloud_blocked: xBD event lat/lon with a known cloudy day on After.
                   Tied to xBD only because we already have those coords.

Output: data/metadata/disaster_m3/negative_cases.yaml
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml
from pystac_client import Client


SAT_BASE = "http://localhost:7860"
STAC_URL = "https://earth-search.aws.element84.com/v1"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "negative_cases.yaml"

DEFAULT_BEFORE_DATE = "2022-04-15"
DEFAULT_AFTER_DATE  = "2022-09-15"

# Biome-diverse search regions. For each, query STAC and take a few items
# whose centroid we'll adopt as a "guaranteed-tile-center" lat/lon.
# bbox = [west, south, east, north]
SEARCH_REGIONS = [
    {"id": "europe_west",     "biome": "urban_temperate", "bbox": [-5,  43, 15, 55],  "n": 4},
    {"id": "europe_east",     "biome": "crop_temperate",  "bbox": [20, 45, 40, 55],  "n": 4},
    {"id": "north_america_central", "biome": "crop_grassland", "bbox": [-105, 35, -85, 48], "n": 5},
    {"id": "south_america_central", "biome": "forest_tropical", "bbox": [-65, -15, -45, -3], "n": 4},
    {"id": "north_africa",    "biome": "desert",          "bbox": [-5, 18, 30, 32],  "n": 4},
    {"id": "africa_central",  "biome": "forest_tropical", "bbox": [12, -10, 35, 5],  "n": 3},
    {"id": "africa_south",    "biome": "savanna",         "bbox": [18, -28, 32, -20],"n": 3},
    {"id": "middle_east",     "biome": "desert",          "bbox": [38, 22, 55, 35],  "n": 3},
    {"id": "central_asia",    "biome": "steppe",          "bbox": [55, 40, 80, 50],  "n": 4},
    {"id": "south_asia",      "biome": "crop_subtropical","bbox": [73, 22, 88, 30],  "n": 3},
    {"id": "east_asia",       "biome": "urban_subtropical","bbox": [105, 30, 125, 42],"n": 4},
    {"id": "se_asia",         "biome": "forest_tropical", "bbox": [98, -2, 116, 12], "n": 3},
    {"id": "australia_inland","biome": "desert",          "bbox": [125, -28, 140, -22],"n": 3},
    {"id": "siberia",         "biome": "boreal",          "bbox": [70, 55, 130, 68], "n": 3},
]


# ---------------------------------------------------------------------------

def fetch_dm3_cases() -> list[dict]:
    r = requests.get(f"{SAT_BASE}/api/disasterm3/cases", timeout=30)
    r.raise_for_status()
    return r.json().get("cases", [])


def fetch_pair(lat: float, lon: float, before_date: str, after_date: str,
               size_km: float, window_days: int) -> tuple[dict, dict]:
    body = {
        "lat": lat, "lon": lon,
        "before_date": before_date, "after_date": after_date,
        "size_km": size_km, "window_days": window_days,
    }
    r = requests.post(f"{SAT_BASE}/api/fetch", json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    return d.get("before", {}), d.get("after", {})


def parse_iso(s: str) -> datetime:
    if "T" not in s:
        s = s + "T00:00:00+00:00"
    elif s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def verify_no_change(lat, lon, size_km, window_days, before_date, after_date,
                     max_cloud=0.5, max_nodata=0.3):
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


def verify_cloud_blocked(lat, lon, size_km, window_days, before_date, after_date,
                         min_cloud_after=0.7, max_nodata=0.3):
    try:
        b, a = fetch_pair(lat, lon, before_date, after_date, size_km, window_days)
    except Exception as e:
        return False, {"error": f"fetch_pair raised: {type(e).__name__}: {e}"}
    bm, am = b.get("meta", {}), a.get("meta", {})
    if not am.get("image_available"):
        return False, {"reason": "after image not available"}
    cp_a = (am.get("stats") or {}).get("cloud_proxy", 0)
    nd_a = (am.get("stats") or {}).get("nodata_fraction", 0)
    info = {
        "after_datetime": am.get("datetime"),
        "before_key": b.get("key"), "after_key": a.get("key"),
        "cloud_b": (bm.get("stats") or {}).get("cloud_proxy"),
        "cloud_a": cp_a,
        "nodata_a": nd_a,
    }
    if nd_a > max_nodata:
        info["reason"] = f"after has too much nodata (nd_a={nd_a:.2f})"
        return False, info
    if cp_a < min_cloud_after:
        info["reason"] = f"after not cloudy enough (cp_a={cp_a:.2f}, want >{min_cloud_after})"
        return False, info
    return True, info


# ---------------------------------------------------------------------------
# STAC discovery: pick S2 tile centroids from real items in each region
# ---------------------------------------------------------------------------

def stac_pick_tile_centers(region_bbox, n_wanted, datetime_range):
    """Query STAC for low-cloud S2 items in the region. Return tile centroids."""
    client = Client.open(STAC_URL)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=region_bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": 10}},
        max_items=200,
    )
    items = list(search.items())
    if not items:
        return []
    # Dedupe by mgrs tile (s2:mgrs_tile in properties) so we get distinct
    # geographic positions, not multiple acquisitions of the same tile.
    by_tile = {}
    for it in items:
        tile = it.properties.get("s2:mgrs_tile") or it.id
        if tile in by_tile:
            continue
        # Use bbox center
        bb = it.bbox  # [west, south, east, north]
        centroid = {
            "tile": tile,
            "lat": (bb[1] + bb[3]) / 2.0,
            "lon": (bb[0] + bb[2]) / 2.0,
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "datetime": it.datetime.isoformat() if it.datetime else None,
        }
        by_tile[tile] = centroid
        if len(by_tile) >= n_wanted:
            break
    return list(by_tile.values())


def collect_no_change_via_stac():
    print(f"[1/2] STAC discovery across {len(SEARCH_REGIONS)} regions...")
    cases = []
    seq = 0
    for region in SEARCH_REGIONS:
        print(f"\n  Region: {region['id']:25s} biome={region['biome']:18s} bbox={region['bbox']}")
        try:
            centers = stac_pick_tile_centers(
                region["bbox"], region["n"],
                datetime_range="2022-04-01/2022-09-30",
            )
        except Exception as e:
            print(f"    STAC search failed: {e}")
            continue
        print(f"    found {len(centers)} candidate tile centers")
        for c in centers:
            seq += 1
            print(f"    [{seq}] tile={c['tile']:8s} lat={c['lat']:+7.3f} lon={c['lon']:+8.3f} (stac cloud={c['cloud_cover']:.1f})", end=" ")
            ok, info = verify_no_change(c["lat"], c["lon"], 50.0, 30,
                                         DEFAULT_BEFORE_DATE, DEFAULT_AFTER_DATE)
            if not ok:
                # Try one alternate season
                ok, info = verify_no_change(c["lat"], c["lon"], 50.0, 30,
                                             "2023-04-15", "2023-09-15")
            if not ok:
                print(f"FAIL: {info.get('reason') or info.get('error', '?')}")
                continue
            cases.append({
                "id":              f"neg__no_change__{region['id']}__{c['tile']}",
                "parent_event":    None,
                "negative_type":   "no_change",
                "biome":           region["biome"],
                "expected_action": "drop",
                "lat":             round(c["lat"], 4),
                "lon":             round(c["lon"], 4),
                "size_km":         50.0,
                "before_date":     DEFAULT_BEFORE_DATE if info.get("before_datetime", "").startswith("2022-04") or info.get("before_datetime", "").startswith("2022-05") or info.get("before_datetime", "").startswith("2022-03") else "2023-04-15",
                "after_date":      DEFAULT_AFTER_DATE  if info.get("after_datetime", "").startswith("2022-09") or info.get("after_datetime", "").startswith("2022-08") or info.get("after_datetime", "").startswith("2022-10") else "2023-09-15",
                "window_days":     30,
                "stac_tile":       c["tile"],
                "note":            f"{region['biome']} ({region['id']}, mgrs {c['tile']})",
                "verified":        info,
            })
            print(f"OK cloud_b={info['cloud_b']:.2f} cloud_a={info['cloud_a']:.2f} nd_b={info['nodata_b']:.2f} nd_a={info['nodata_a']:.2f}")
    return cases


def collect_cloud_blocked_for_xbd_events():
    print("\n[2/2] cloud_blocked from xBD event lat/lons...")
    all_cases = fetch_dm3_cases()
    precise = [c for c in all_cases if c.get("precise")]
    by_event = {}
    for c in precise:
        e = c["event"]
        if e not in by_event:
            by_event[e] = c

    out = []
    for event_name, case in by_event.items():
        print(f"  -> {event_name:25s}", end=" ")
        if not case.get("event_start"):
            print("(no event_start)"); continue
        lat, lon = case["lat"], case["lon"]
        size_km = case.get("size_km", 50.0)
        event_start = parse_iso(case["event_start"])
        offsets = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 365,
                   -15, -30, -45, -60, -90, -120, -180, -240, -300, -365]
        chosen = None
        for offset_days in offsets:
            target = event_start + timedelta(days=offset_days)
            if target.date() > datetime.now(timezone.utc).date():
                continue
            target_str = date_str(target)
            before = date_str(target - timedelta(days=30))
            ok, info = verify_cloud_blocked(lat, lon, size_km, 30, before, target_str)
            if ok:
                chosen = (before, target_str, info)
                break
        if not chosen:
            print("FAIL"); continue
        before, after, info = chosen
        out.append({
            "id":              f"neg__cloud_blocked__{event_name}",
            "parent_event":    event_name,
            "negative_type":   "cloud_blocked",
            "expected_action": "drop",
            "lat":             lat,
            "lon":             lon,
            "size_km":         size_km,
            "before_date":     before,
            "after_date":      after,
            "window_days":     30,
            "note":            f"After {info.get('after_datetime', after)[:10]} は雲被覆 cloud_proxy={info.get('cloud_a', 0):.2f}",
            "verified":        info,
        })
        print(f"OK cp={info.get('cloud_a', 0):.2f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--skip-cloud", action="store_true")
    args = parser.parse_args()

    cases = collect_no_change_via_stac()
    if not args.skip_cloud:
        cases.extend(collect_cloud_blocked_for_xbd_events())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    print(f"\n[done] {len(cases)} negative cases → {out_path}")


if __name__ == "__main__":
    main()
