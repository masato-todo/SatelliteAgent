"""Collect FireEdge-style POS / NEG cases from NASA FIRMS active fire API.

Mirrors apps/fireedge/finetune/dataset_builder.py:
  - POS:        FIRMS VIIRS_SNPP_SP detection coords; after = detect_date + 2d
                (SWIR thermal signature still warm - matches LoRA training)
  - NEG (temporal): same coord; after = detect_date - 180d (no fire baseline)

Output: data/metadata/disaster_m3/firms_fire_cases.yaml

Requires:
  FIRMS_MAP_KEY env var (free at https://firms.modaps.eosdis.nasa.gov/api/area/).

Usage:
  FIRMS_MAP_KEY=xxx python scripts/collect_firms_fire.py
  FIRMS_MAP_KEY=xxx python scripts/collect_firms_fire.py --target 50 --days 5
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml

# Load FIRMS_MAP_KEY (and other secrets) from .env at repo root.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "firms_fire_cases.yaml"

# 8 fire-prone regions, same as fireedge/finetune
FIRMS_AREAS: dict[str, str] = {
    "west_africa":  "5,5,30,15",
    "east_africa":  "25,0,45,15",
    "seasia":       "95,5,140,20",
    "amazon":       "-70,-15,-45,5",
    "australia":    "130,-35,155,-15",
    "cent_africa":  "10,-10,35,5",
    "us_west":      "-125,35,-105,50",
    "siberia":      "80,50,130,65",
}
FIRMS_PRODUCT = os.environ.get("FIRMS_PRODUCT", "VIIRS_SNPP_SP")
# Default matches FireEdge training (SP = Standard Processing, post-processed,
# stable confidence labels). NRT is for last few days when SP not yet ready.

# FireEdge training period (apps/fireedge/finetune/dataset_builder.py):
#   FIRMS_TRAIN_START    = "2025-02-01"
#   FIRMS_TRAIN_WINDOW   = 60   (= 2025-02-01 .. 2025-04-01, 60 days)
TRAIN_START_DEFAULT = os.environ.get("FIRMS_TRAIN_START", "2025-02-01")
TRAIN_WINDOW_DEFAULT = int(os.environ.get("FIRMS_TRAIN_WINDOW", "60"))

# FireEdge constants
SHIFT_DAYS    = 2     # POS: detect_date + 2d
NEG_OFFSET    = 180   # NEG temporal: detect_date - 180d
SIZE_KM       = 5.0
WINDOW_DAYS   = 12    # SimSat search window (matches FireEdge ±12d)


def fetch_firms_area(map_key: str, bbox: str, days: int,
                     start_date: str | None = None) -> list[dict]:
    """FIRMS Area CSV API. Max 5 days. start_date YYYY-MM-DD optional
    (omitted = "last `days` from now")."""
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{map_key}/{FIRMS_PRODUCT}/{bbox}/{days}")
    if start_date:
        url += f"/{start_date}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    if text.startswith("Invalid"):
        raise RuntimeError(f"FIRMS error: {text[:200]}")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--target", type=int, default=80,
                    help="Number of POS cases (NEG temporal mirrors 1:1)")
    ap.add_argument("--start-date", default=TRAIN_START_DEFAULT,
                    help=f"FIRMS_TRAIN_START (default {TRAIN_START_DEFAULT}). Empty = last N days.")
    ap.add_argument("--window-days", type=int, default=TRAIN_WINDOW_DEFAULT,
                    help=f"Total days from start-date (default {TRAIN_WINDOW_DEFAULT}). FIRMS API needs 5d chunks.")
    ap.add_argument("--frp-min", type=float, default=0.0,
                    help="Min FRP (MW). FireEdge training uses 0 (no filter).")
    ap.add_argument("--no-neg", action="store_true",
                    help="Skip NEG temporal pairs (POS only)")
    ap.add_argument("--cell-deg", type=float, default=0.05,
                    help="Dedup grid size (~5 km at equator)")
    args = ap.parse_args()

    map_key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_API_KEY")
    if not map_key:
        print("ERROR: set FIRMS_MAP_KEY (free at https://firms.modaps.eosdis.nasa.gov/api/area/)",
              file=sys.stderr)
        return 2

    # Build 5-day chunks across the [start_date, start_date+window_days) range,
    # mirroring fineturne/dataset_builder.py:collect_firms (5d API limit).
    use_window = bool(args.start_date)
    chunks: list[tuple[str | None, int]] = []
    if use_window:
        from datetime import datetime as _dt, timedelta as _td
        start_dt = _dt.strptime(args.start_date, "%Y-%m-%d")
        for offset in range(0, args.window_days, 5):
            chunk_start = (start_dt + _td(days=offset)).strftime("%Y-%m-%d")
            chunk_days  = min(5, args.window_days - offset)
            chunks.append((chunk_start, chunk_days))
        print(f"[1/3] Fetching FIRMS {FIRMS_PRODUCT} ({len(FIRMS_AREAS)} regions × {len(chunks)} 5d-chunks "
              f"from {args.start_date} +{args.window_days}d)...", flush=True)
    else:
        chunks.append((None, 5))
        print(f"[1/3] Fetching FIRMS {FIRMS_PRODUCT} ({len(FIRMS_AREAS)} regions, last 5 days)...",
              flush=True)

    detections: list[dict] = []
    for region, bbox in FIRMS_AREAS.items():
        region_total = 0
        region_kept = 0
        for chunk_start, chunk_days in chunks:
            try:
                rows = fetch_firms_area(map_key, bbox, chunk_days, chunk_start)
            except Exception as e:
                print(f"  {region:14s} {chunk_start or 'NRT':10s}: FAIL {e}", flush=True)
                continue
            region_total += len(rows)
            for r in rows:
                try:
                    lat  = float(r["latitude"])
                    lon  = float(r["longitude"])
                    frp  = float(r.get("frp", 0))
                    conf = r.get("confidence", "")
                except (KeyError, ValueError):
                    continue
                # FireEdge accepts NRT="nominal"/"high" + SP="n"/"h"; rejects "l"/"low".
                if conf in ("l", "low"):
                    continue
                if frp < args.frp_min:
                    continue
                detections.append({
                    "lat": lat, "lon": lon, "frp": frp,
                    "date": r["acq_date"], "confidence": conf, "region": region,
                })
                region_kept += 1
        print(f"  {region:14s}: {region_total:>5} raw → {region_kept:>4} kept "
              f"(frp≥{args.frp_min}, conf≠low)", flush=True)
    print(f"      total: {len(detections)} detections", flush=True)

    # Dedup by lat/lon grid cell, then sort by FRP desc
    seen: set[tuple[float, float]] = set()
    unique: list[dict] = []
    cell = args.cell_deg
    for d in detections:
        key = (round(d["lat"] / cell) * cell, round(d["lon"] / cell) * cell)
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    unique.sort(key=lambda x: -x["frp"])
    selected = unique[:args.target]
    print(f"      dedup → {len(unique)}, top {len(selected)} by FRP", flush=True)

    print(f"\n[2/3] Building POS + NEG cases...", flush=True)
    cases: list[dict] = []
    for d in selected:
        try:
            et = datetime.strptime(d["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        after_pos = et + timedelta(days=SHIFT_DAYS)
        before_pos = et - timedelta(days=NEG_OFFSET)  # baseline 6mo prior

        # ASCII-safe id (no '+' / weird chars; lat sign as p/m).
        lat_str = f"{'p' if d['lat'] >= 0 else 'm'}{abs(d['lat']):.3f}".replace(".", "")
        lon_str = f"{'p' if d['lon'] >= 0 else 'm'}{abs(d['lon']):.3f}".replace(".", "")
        date_str = d["date"].replace("-", "")
        cid_pos = f"firms_pos_{d['region']}_{date_str}_{lat_str}_{lon_str}"

        cases.append({
            "id":              cid_pos,
            "source":          "FIRMS",
            "category":        "Fire",
            "event_type":      "fire",
            "label":           "fire",
            "expected_action": "submit_to_ground",
            "lat":             round(d["lat"], 4),
            "lon":             round(d["lon"], 4),
            "size_km":         SIZE_KM,
            "before_date":     before_pos.strftime("%Y-%m-%d"),
            "after_date":      after_pos.strftime("%Y-%m-%d"),
            "window_days":     WINDOW_DAYS,
            "event_time":      et.isoformat(),
            "frp":             d["frp"],
            "region":          d["region"],
            "firms_confidence": d["confidence"],
            "is_firms_pos":    True,
        })

        if not args.no_neg:
            cid_neg = cid_pos.replace("firms_pos_", "firms_neg_")
            after_neg  = et - timedelta(days=NEG_OFFSET)
            before_neg = et - timedelta(days=NEG_OFFSET + 30)
            cases.append({
                "id":              cid_neg,
                "source":          "FIRMS",
                "category":        "Fire-Negative",
                "event_type":      "no_change",
                "label":           "no_change",
                "expected_action": "drop",
                "negative_type":   "firms_temporal",
                "lat":             round(d["lat"], 4),
                "lon":             round(d["lon"], 4),
                "size_km":         SIZE_KM,
                "before_date":     before_neg.strftime("%Y-%m-%d"),
                "after_date":      after_neg.strftime("%Y-%m-%d"),
                "window_days":     WINDOW_DAYS,
                "event_time":      et.isoformat(),
                "region":          d["region"],
                "is_firms_neg":    True,
            })

    print(f"      built: {len(cases)} cases (POS + NEG mix)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases":    len(cases),
        "source":     "FIRMS VIIRS_SNPP_SP",
        "shift_days": SHIFT_DAYS,
        "neg_offset_days": NEG_OFFSET,
        "size_km":    SIZE_KM,
        "cases":      cases,
    }
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    tmp.replace(out_path)

    n_pos = sum(1 for c in cases if c.get("is_firms_pos"))
    n_neg = sum(1 for c in cases if c.get("is_firms_neg"))
    print(f"\n[3/3] saved {len(cases)} cases (POS={n_pos}, NEG={n_neg}) → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
