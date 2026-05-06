"""Collect deforestation events as Phase 1 catalog entries.

Source: PRODES Amazon Biome (INPE / TerraBrasilis), exposed via OGC WFS at
https://terrabrasilis.dpi.inpe.br/geoserver/prodes-amazon-nb/ows .
The `yearly_deforestation_biome` layer holds per-year clearing polygons
already detected from Sentinel-2 by INPE — exactly the events we want.

Per docs/EXPERIMENT_PLAN.md Phase 1: this script ONLY builds catalog
metadata (id / lat / lon / dates / area). NO SimSat call. Phase 2's
`auto_fill_pairs.py` does best-pair selection.

S2 detectability of forest loss:
  - NDVI / NBR delta is large (-0.4 〜 -0.7) for fresh clear-cut
  - 10m resolution captures patches ≥1 ha cleanly
  - clearing pattern: rectangular / road-aligned (vs irregular wildfire scar)

Output: `data/metadata/disaster_m3/deforestation_cases.yaml`.
Each case carries `area_km2`, `state` (AM/PA/MT/...), and PRODES `image_date`.

Usage:
  python scripts/collect_deforestation.py                              # year>=2020, area>5km², all
  python scripts/collect_deforestation.py --area-min 10 --target 100   # bigger only
  python scripts/collect_deforestation.py --start-year 2022
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


WFS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-amazon-nb/ows"
TYPE_NAME = "prodes-amazon-nb:yearly_deforestation_biome"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "deforestation_cases.yaml"
GT_DIR = ROOT / "data" / "gt_polygons"


def fetch_features(start_year: int, end_year: int, area_min: float,
                    max_count: int = 5000) -> list[dict]:
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": TYPE_NAME, "outputFormat": "application/json",
        "count": str(max_count),
        "CQL_FILTER": f"year >= {start_year} AND year <= {end_year} AND area_km > {area_min}",
    }
    url = f"{WFS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "SatelliteAgent/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d.get("features", [])


def multipolygon_centroid(coords: list) -> tuple[float, float] | None:
    """Approximate centroid via mean of all vertices (good enough for AOI center)."""
    xs: list[float] = []
    ys: list[float] = []

    def walk(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)) and len(obj) >= 2:
                xs.append(float(obj[0]))
                ys.append(float(obj[1]))
            else:
                for x in obj:
                    walk(x)

    walk(coords)
    if not xs:
        return None
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if "T" not in s:
            s = s + "T00:00:00"
        if not s.endswith("Z") and "+" not in s[-6:]:
            s = s + "+00:00"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cases = doc.get("cases", []) if isinstance(doc, dict) else []
    ids = {c.get("id") for c in cases if c.get("id")}
    return cases, ids


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--area-min", type=float, default=5.0,
                   help="Minimum polygon area in km² (default 5.0)")
    p.add_argument("--size-km", type=float, default=10.0)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--before-offset-days", type=int, default=-365,
                   help="before_date = image_date + offset (negative). Default -365 to dampen seasonal bias.")
    p.add_argument("--after-offset-days", type=int, default=14,
                   help="after_date  = image_date + offset.")
    args = p.parse_args()

    print(f"[1/2] Fetching PRODES yearly_deforestation polygons "
          f"(year ∈ [{args.start_year},{args.end_year}], area>={args.area_min} km²)...", flush=True)
    feats = fetch_features(args.start_year, args.end_year, args.area_min)
    print(f"      got {len(feats)} polygons", flush=True)

    out_path = Path(args.out)
    cases, existing_ids = load_existing(out_path)
    print(f"      existing yaml: {len(cases)} cases", flush=True)

    print(f"\n[2/2] Building catalog entries...", flush=True)
    new_added = skipped_dup = skipped_bad = 0
    target_remaining = args.target if args.target is not None else 10**9

    # Sort by area descending so target=N picks the biggest first (most visible signal)
    feats_sorted = sorted(
        feats,
        key=lambda f: (f.get("properties") or {}).get("area_km", 0.0),
        reverse=True,
    )

    for f in feats_sorted:
        if target_remaining <= 0:
            break
        props = f.get("properties", {})
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates") or []
        cen = multipolygon_centroid(coords)
        if cen is None:
            skipped_bad += 1
            continue
        lat, lon = cen
        uid = props.get("uid") or props.get("uuid")
        case_id = f"prodes_amazon_{uid}"
        if case_id in existing_ids:
            skipped_dup += 1
            continue
        et = parse_iso(props.get("image_date"))
        if et is None:
            skipped_bad += 1
            continue
        before_date = (et + timedelta(days=args.before_offset_days)).strftime("%Y-%m-%d")
        after_date  = (et + timedelta(days=args.after_offset_days)).strftime("%Y-%m-%d")
        # Save GT polygon for UI overlay (mirrors MCD64A1 burn_polygon path)
        GT_DIR.mkdir(parents=True, exist_ok=True)
        gt_path = GT_DIR / f"{case_id}.geojson"
        gt_path.write_text(json.dumps({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id":       case_id,
                "source":   "PRODES",
                "year":     props.get("year"),
                "area_km2": float(props.get("area_km", 0.0)),
                "state":    props.get("state"),
                "image_date": props.get("image_date"),
            },
        }))
        cases.append({
            "id":              case_id,
            "prodes_uid":      uid,
            "source":          "PRODES",
            "category":        "Deforestation",
            "event_type":      "deforestation",
            "expected_action": "submit_to_ground",
            "name":            f"PRODES {props.get('class_name','?')} {props.get('state','?')} ({props.get('area_km',0):.1f} km²)",
            "country":         "Brazil",
            "state":           props.get("state"),
            "path_row":        props.get("path_row"),
            "scene_id":        props.get("scene_id"),
            "main_class":      props.get("main_class"),
            "class_name":      props.get("class_name"),
            "year":            props.get("year"),
            "area_km2":        round(float(props.get("area_km", 0.0)), 3),
            "lat":             round(lat, 4),
            "lon":             round(lon, 4),
            "size_km":         args.size_km,
            "before_date":     before_date,
            "after_date":      after_date,
            "window_days":     args.window_days,
            "image_date":      props.get("image_date"),
            "satellite":       props.get("satellite"),
            "sensor":          props.get("sensor"),
            "gt_polygon_uri":  f"data/gt_polygons/{case_id}.geojson",
        })
        existing_ids.add(case_id)
        new_added += 1
        target_remaining -= 1

    save_yaml(out_path, cases)
    print(f"\n[done] +{new_added} new (skip dup={skipped_dup}, bad={skipped_bad}); total {len(cases)} → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
