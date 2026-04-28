"""MCD64A1 smoke test: download one month, polygonize burn pixels, list top areas.

Verifies that:
  1. earthaccess login works (~/.netrc)
  2. MCD64A1 HDF4 can be read
  3. Burn polygons can be extracted with sensible lat/lon centroids

Usage:
    uv run python scripts/mcd64a1_smoke.py
    uv run python scripts/mcd64a1_smoke.py --year 2023 --month 8 --bbox -160 18 -155 22
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import earthaccess
from pyhdf.SD import SD, SDC
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
import pyproj


PROJ_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJ_ROOT / "data" / "scene_catalog.yaml"
GT_DIR       = PROJ_ROOT / "data" / "gt_polygons"


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_mcd64a1"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--month", type=int, default=8)
    ap.add_argument("--bbox", nargs=4, type=float, default=[-160, 18, -155, 22],
                    metavar=("lon_min", "lat_min", "lon_max", "lat_max"),
                    help="bounding box (default: Hawaii area, Aug 2023 = Lahaina fire)")
    ap.add_argument("--top", type=int, default=10, help="show top-N largest burns")
    ap.add_argument("--save", action="store_true",
                    help=f"append detections to {CATALOG_PATH.relative_to(PROJ_ROOT)}")
    args = ap.parse_args()

    print(f"[1/5] earthaccess.login(strategy='netrc') ...")
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        print("FAIL: not authenticated. Check ~/.netrc.")
        return 2

    start = f"{args.year:04d}-{args.month:02d}-01"
    end_day = 28 if args.month == 2 else 30 if args.month in (4, 6, 9, 11) else 31
    end = f"{args.year:04d}-{args.month:02d}-{end_day:02d}"
    print(f"[2/5] search MCD64A1 temporal=({start}, {end}) bbox={args.bbox} ...")
    results = earthaccess.search_data(
        short_name="MCD64A1",
        temporal=(start, end),
        bounding_box=tuple(args.bbox),
    )
    print(f"     -> {len(results)} granule(s) found")
    if not results:
        print("FAIL: no granules in this bbox/time. Try a different month.")
        return 3

    print(f"[3/5] downloading {len(results)} granule(s) to {RAW_DIR} ...")
    files = earthaccess.download(results, str(RAW_DIR))
    print(f"     -> {len(files)} file(s) downloaded")

    # MODIS sinusoidal grid constants
    R = 6371007.181
    TILE_W = 1111950.5196667
    PIXELS_PER_TILE = 2400
    PIXEL_SIZE = TILE_W / PIXELS_PER_TILE
    sin_proj = f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={R} +units=m +no_defs"
    proj_to_wgs = pyproj.Transformer.from_crs(sin_proj, "EPSG:4326", always_xy=True).transform

    print(f"[4/5] reading Burn Date for each tile ...")
    import re
    enriched: list[dict] = []
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
            print(f"  {fp.name}: pyhdf error {type(e).__name__}: {e}")
            continue
        n_burn = int((burn_doy > 0).sum())
        print(f"  {fp.name}  tile=h{h_idx:02d}v{v_idx:02d}  burned_pixels={n_burn}")
        if n_burn == 0:
            continue
        burned_mask = (burn_doy > 0).astype(np.uint8)
        for geom, _ in shapes(burned_mask, mask=burned_mask > 0, transform=transform):
            poly_sin = shape(geom)
            area_km2 = poly_sin.area / 1e6
            if area_km2 < 1.0:
                continue
            c = poly_sin.centroid
            lon, lat = proj_to_wgs(c.x, c.y)
            enriched.append({
                "area_km2": area_km2,
                "lat": lat,
                "lon": lon,
                "tile": f"h{h_idx:02d}v{v_idx:02d}",
                "poly_sin": poly_sin,
            })

    print(f"[5/5] {len(enriched)} polygons >=1km² across all tiles")
    enriched.sort(key=lambda e: -e["area_km2"])

    print(f"     -> {len(enriched)} polygons >=1km²")
    print()
    print(f"  rank  area_km²    lat        lon         tile")
    print(f"  ----  --------    --------   --------    ------")
    for i, e in enumerate(enriched[: args.top], 1):
        print(f"  {i:>4}  {e['area_km2']:>8.2f}    {e['lat']:>+8.4f}  {e['lon']:>+9.4f}    {e['tile']}")

    if args.save and enriched:
        save_to_catalog(enriched[: args.top], args.year, args.month, proj_to_wgs)

    print()
    print(f"[done] Use the dropdown in the UI (group: MCD64A1) to inspect Before/After.")
    return 0


def save_to_catalog(entries: list[dict], year: int, month: int,
                    proj_to_wgs) -> None:
    """Append detected scenes to scene_catalog.yaml + write each polygon as GeoJSON."""
    GT_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        scenes = doc.get("scenes") or []
    else:
        scenes = []
    seen_ids = {s.get("id") for s in scenes}

    from shapely.ops import transform as shp_transform
    n_added = 0
    for e in entries:
        sid = (
            f"mcd64a1_{e['tile']}_{year:04d}{month:02d}_"
            f"{int(round(e['lat']*100)):+05d}_{int(round(e['lon']*100)):+06d}"
        )
        if sid in seen_ids:
            continue
        # Reproject polygon sin -> WGS84 and write GeoJSON
        poly_wgs = shp_transform(proj_to_wgs, e["poly_sin"])
        gj_path = GT_DIR / f"{sid}.geojson"
        with open(gj_path, "w", encoding="utf-8") as f:
            json_dump_geojson(f, poly_wgs, sid)
        from datetime import date, timedelta
        period_start = date(year, month, 1).isoformat()
        end_day = 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31
        period_end = date(year, month, end_day).isoformat()
        scenes.append({
            "id": sid,
            "event_type": "wildfire",
            "lat": float(e["lat"]),
            "lon": float(e["lon"]),
            "event_period": [period_start, period_end],
            "affected_area_km2": float(round(e["area_km2"], 2)),
            "source": "MCD64A1",
            "gt_polygon_uri": str(gj_path.relative_to(PROJ_ROOT)),
        })
        seen_ids.add(sid)
        n_added += 1

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenes": scenes}, f, sort_keys=False, allow_unicode=True)
    print(f"     saved: +{n_added} new scenes (catalog total: {len(scenes)})")


def json_dump_geojson(f, poly, sid: str) -> None:
    import json
    feat = {
        "type": "Feature",
        "id": sid,
        "geometry": mapping(poly),
        "properties": {},
    }
    json.dump(feat, f)


if __name__ == "__main__":
    sys.exit(main())
