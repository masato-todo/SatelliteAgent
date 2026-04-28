"""Phase 1 — Build disaster scene catalog from MCD64A1 burn-area polygons.

Iterates over `config/catalog_regions.yaml` (region × month list), pulls
MCD64A1 tiles, polygonizes burned pixels, and appends qualifying scenes to
`data/scene_catalog.yaml` with their GT polygon as GeoJSON sidecar.

Filters:
  - polygon area >= MIN_AREA_KM2 (default 1.0 km²)
  - centroid >= DEDUP_DISTANCE_KM from any existing scene (default 20 km)

Usage:
    uv run python scripts/build_scene_catalog.py
    uv run python scripts/build_scene_catalog.py --max-per-region 5
    uv run python scripts/build_scene_catalog.py --dry-run    # report only, no save
    uv run python scripts/build_scene_catalog.py --regions config/my_targets.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import yaml
import earthaccess
import pyproj
from pyhdf.SD import SD, SDC
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform


PROJ_ROOT     = Path(__file__).resolve().parent.parent
CATALOG_PATH  = PROJ_ROOT / "data" / "scene_catalog.yaml"
GT_DIR        = PROJ_ROOT / "data" / "gt_polygons"
RAW_DIR       = PROJ_ROOT / "data" / "raw_mcd64a1"

# MODIS sinusoidal grid constants
R              = 6371007.181
TILE_W         = 1111950.5196667
PIXELS_PER_TILE = 2400
PIXEL_SIZE     = TILE_W / PIXELS_PER_TILE
SIN_PROJ       = f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={R} +units=m +no_defs"

MIN_AREA_KM2       = 1.0
DEDUP_DISTANCE_KM  = 20.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_e = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R_e * asin(sqrt(a))


def month_end_day(year: int, month: int) -> int:
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def process_region(name, year, month, bbox, max_per_region, proj_to_wgs):
    """Search + download + polygonize one region/month. Returns list of candidate dicts."""
    end_day = month_end_day(year, month)
    start = f"{year:04d}-{month:02d}-01"
    end   = f"{year:04d}-{month:02d}-{end_day:02d}"

    results = earthaccess.search_data(
        short_name="MCD64A1",
        temporal=(start, end),
        bounding_box=tuple(bbox),
    )
    if not results:
        return []

    files = earthaccess.download(results, str(RAW_DIR))

    out: list[dict] = []
    for fpath in files:
        fp = Path(fpath)
        m = re.search(r'\.h(\d{2})v(\d{2})\.', fp.name)
        if not m:
            continue
        h_idx, v_idx = int(m.group(1)), int(m.group(2))
        x_ul = -20015109.354 + h_idx * TILE_W
        y_ul =  10007554.677 - v_idx * TILE_W
        transform = Affine(PIXEL_SIZE, 0, x_ul, 0, -PIXEL_SIZE, y_ul)
        try:
            sd_ds = SD(str(fp), SDC.READ)
            burn_doy = sd_ds.select("Burn Date").get()
            sd_ds.end()
        except Exception as e:
            print(f"    pyhdf error on {fp.name}: {type(e).__name__}: {e}")
            continue
        if (burn_doy > 0).sum() == 0:
            continue
        burned_mask = (burn_doy > 0).astype(np.uint8)
        for geom, _ in shapes(burned_mask, mask=burned_mask > 0, transform=transform):
            poly_sin = shape(geom)
            area_km2 = poly_sin.area / 1e6
            if area_km2 < MIN_AREA_KM2:
                continue
            c = poly_sin.centroid
            lon, lat = proj_to_wgs(c.x, c.y)
            # Compute actual burn date range from per-pixel DOY values inside polygon
            poly_mask = rasterize(
                [(poly_sin, 1)], out_shape=burn_doy.shape,
                transform=transform, fill=0, dtype=np.uint8,
            )
            doys = burn_doy[(poly_mask == 1) & (burn_doy > 0)]
            min_doy = int(doys.min()) if doys.size else None
            max_doy = int(doys.max()) if doys.size else None
            out.append({
                "area_km2": area_km2,
                "lat": lat,
                "lon": lon,
                "tile": f"h{h_idx:02d}v{v_idx:02d}",
                "poly_sin": poly_sin,
                "min_doy": min_doy,
                "max_doy": max_doy,
            })
    out.sort(key=lambda e: -e["area_km2"])
    return out[:max_per_region]


def make_scene_id(tile: str, year: int, month: int, lat: float, lon: float) -> str:
    return (
        f"mcd64a1_{tile}_{year:04d}{month:02d}_"
        f"{int(round(lat * 100)):+05d}_{int(round(lon * 100)):+06d}"
    )


def add_scene(entry, region_name, year, month, scenes, proj_to_wgs):
    sid = make_scene_id(entry["tile"], year, month, entry["lat"], entry["lon"])
    if any(s["id"] == sid for s in scenes):
        return False, "dup id"
    for s in scenes:
        d_km = haversine_km(entry["lat"], entry["lon"], s["lat"], s["lon"])
        if d_km < DEDUP_DISTANCE_KM:
            return False, f"too close to {s['id'][-25:]} ({d_km:.1f}km)"

    GT_DIR.mkdir(parents=True, exist_ok=True)
    poly_wgs = shp_transform(proj_to_wgs, entry["poly_sin"])
    gj_path = GT_DIR / f"{sid}.geojson"
    with open(gj_path, "w", encoding="utf-8") as f:
        json.dump({
            "type": "Feature",
            "id": sid,
            "geometry": mapping(poly_wgs),
            "properties": {"region": region_name, "area_km2": round(entry["area_km2"], 2)},
        }, f)

    end_day = month_end_day(year, month)
    if entry.get("min_doy") and entry.get("max_doy"):
        period_start = date(year, 1, 1) + timedelta(days=entry["min_doy"] - 1)
        period_end   = date(year, 1, 1) + timedelta(days=entry["max_doy"] - 1)
    else:
        period_start = date(year, month, 1)
        period_end   = date(year, month, end_day)
    scenes.append({
        "id": sid,
        "event_type": "wildfire",
        "lat": float(entry["lat"]),
        "lon": float(entry["lon"]),
        "event_period": [period_start.isoformat(), period_end.isoformat()],
        "affected_area_km2": float(round(entry["area_km2"], 2)),
        "source": "MCD64A1",
        "gt_polygon_uri": str(gj_path.relative_to(PROJ_ROOT)),
        "region": region_name,
    })
    return True, "added"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="config/catalog_regions.yaml")
    ap.add_argument("--max-per-region", type=int, default=10,
                    help="cap candidates per region (top by area), default 10")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added; do not write files")
    args = ap.parse_args()

    print(f"[init] earthaccess.login(strategy='netrc') ...")
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        print("FAIL: not authenticated. Check ~/.netrc.")
        return 2

    proj_to_wgs = pyproj.Transformer.from_crs(SIN_PROJ, "EPSG:4326", always_xy=True).transform

    regions_path = (PROJ_ROOT / args.regions) if not Path(args.regions).is_absolute() else Path(args.regions)
    with open(regions_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    targets = cfg.get("targets", [])
    if not targets:
        print(f"FAIL: no targets in {regions_path}")
        return 3

    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            scenes = (yaml.safe_load(f) or {}).get("scenes", [])
    else:
        scenes = []
    print(f"[init] {len(targets)} targets, existing scenes: {len(scenes)}")

    n_added_total = n_skipped_total = 0
    for i, t in enumerate(targets, 1):
        name = t["name"]; year = int(t["year"]); month = int(t["month"]); bbox = t["bbox"]
        print(f"\n[{i}/{len(targets)}] {name}  {year}-{month:02d}  bbox={bbox}")
        try:
            cands = process_region(name, year, month, bbox, args.max_per_region, proj_to_wgs)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue
        if not cands:
            print(f"  -> 0 candidates")
            continue

        n_added = n_skip = 0
        for c in cands:
            if args.dry_run:
                print(f"    DRY  area={c['area_km2']:.1f}km²  lat={c['lat']:+8.4f} lon={c['lon']:+9.4f}  {c['tile']}")
                continue
            ok, why = add_scene(c, name, year, month, scenes, proj_to_wgs)
            if ok:
                n_added += 1
            else:
                n_skip += 1
        print(f"  -> {len(cands)} candidates, +{n_added} added, {n_skip} skipped")
        n_added_total += n_added
        n_skipped_total += n_skip

        # Persist after each region so a crash does not lose work
        if not args.dry_run:
            with open(CATALOG_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump({"scenes": scenes}, f, sort_keys=False, allow_unicode=True)

    print(f"\n[done] +{n_added_total} new scenes, {n_skipped_total} skipped (catalog total: {len(scenes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
