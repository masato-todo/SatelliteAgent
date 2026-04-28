"""Refresh event_period in scene_catalog.yaml from MCD64A1 per-pixel Burn Date.

Existing entries had event_period = whole search-month (placeholder). This
re-reads the cached MCD64A1 HDF tiles and computes the actual min/max burn
day-of-year inside each polygon.

Usage:
    uv run python scripts/refresh_burn_dates.py
    uv run python scripts/refresh_burn_dates.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml
import pyproj
from pyhdf.SD import SD, SDC
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


PROJ_ROOT     = Path(__file__).resolve().parent.parent
CATALOG_PATH  = PROJ_ROOT / "data" / "scene_catalog.yaml"
GT_DIR        = PROJ_ROOT / "data" / "gt_polygons"
RAW_DIR       = PROJ_ROOT / "data" / "raw_mcd64a1"

R              = 6371007.181
TILE_W         = 1111950.5196667
PIXELS_PER_TILE = 2400
PIXEL_SIZE     = TILE_W / PIXELS_PER_TILE
SIN_PROJ       = f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={R} +units=m +no_defs"


def find_hdf_for(tile: str, year: int, month: int) -> Path | None:
    end_day = 29 if (month == 2 and year % 4 == 0) else 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31
    # Granule date is encoded as A{YYYY}{DOY}, where DOY = day-of-year of month start.
    candidates = list(RAW_DIR.glob(f"MCD64A1.A{year:04d}*.{tile}.*.hdf"))
    # filter to ones whose DOY falls in the target month
    for fp in candidates:
        m = re.search(r'\.A(\d{4})(\d{3})\.', fp.name)
        if not m:
            continue
        granule_year = int(m.group(1)); doy = int(m.group(2))
        gd = date(granule_year, 1, 1) + timedelta(days=doy - 1)
        if gd.year == year and gd.month == month:
            return fp
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CATALOG_PATH.exists():
        print(f"FAIL: {CATALOG_PATH} not found")
        return 2
    with open(CATALOG_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    scenes = doc.get("scenes") or []
    print(f"[init] {len(scenes)} scenes in catalog")

    proj_to_sin = pyproj.Transformer.from_crs("EPSG:4326", SIN_PROJ, always_xy=True).transform

    n_updated = n_skip = n_fail = 0
    for sc in scenes:
        sid = sc["id"]
        m = re.match(r"mcd64a1_(h\d{2}v\d{2})_(\d{4})(\d{2})_", sid)
        if not m:
            n_skip += 1
            continue
        tile = m.group(1); year = int(m.group(2)); month = int(m.group(3))

        hdf_path = find_hdf_for(tile, year, month)
        if not hdf_path:
            print(f"  {sid}: HDF not in {RAW_DIR}, skip")
            n_skip += 1
            continue

        # Load polygon from sidecar GeoJSON
        gj_path = PROJ_ROOT / sc["gt_polygon_uri"]
        if not gj_path.exists():
            n_skip += 1
            continue
        with open(gj_path) as f:
            feat = json.load(f)
        poly_wgs = shape(feat["geometry"])
        poly_sin = shp_transform(proj_to_sin, poly_wgs)

        # Set up MODIS sinusoidal transform for this tile
        h_idx = int(tile[1:3]); v_idx = int(tile[4:6])
        x_ul = -20015109.354 + h_idx * TILE_W
        y_ul =  10007554.677 - v_idx * TILE_W
        transform = Affine(PIXEL_SIZE, 0, x_ul, 0, -PIXEL_SIZE, y_ul)

        try:
            sd_ds = SD(str(hdf_path), SDC.READ)
            burn_doy = sd_ds.select("Burn Date").get()
            sd_ds.end()
        except Exception as e:
            print(f"  {sid}: pyhdf error {type(e).__name__}: {e}")
            n_fail += 1
            continue

        poly_mask = rasterize(
            [(poly_sin, 1)], out_shape=burn_doy.shape,
            transform=transform, fill=0, dtype=np.uint8,
        )
        doys = burn_doy[(poly_mask == 1) & (burn_doy > 0)]
        if doys.size == 0:
            print(f"  {sid}: 0 burned pixels inside polygon (?), skip")
            n_skip += 1
            continue

        min_doy, max_doy = int(doys.min()), int(doys.max())
        new_start = (date(year, 1, 1) + timedelta(days=min_doy - 1)).isoformat()
        new_end   = (date(year, 1, 1) + timedelta(days=max_doy - 1)).isoformat()

        old = sc.get("event_period")
        if old != [new_start, new_end]:
            print(f"  {sid}  {old} -> [{new_start}, {new_end}]  ({doys.size}px)")
            if not args.dry_run:
                sc["event_period"] = [new_start, new_end]
            n_updated += 1
        else:
            n_skip += 1

    if not args.dry_run and n_updated:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
        print(f"\n[done] updated {n_updated} entries in {CATALOG_PATH.relative_to(PROJ_ROOT)}")
    else:
        print(f"\n[done] {n_updated} updates pending, {n_skip} unchanged, {n_fail} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
