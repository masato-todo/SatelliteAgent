"""Collect volcanic eruption events as Phase 1 catalog entries.

Source: GDACS (Global Disaster Alert and Coordination System) Volcano (VO)
event list. Returns GeoJSON FeatureCollection with point centroids, dates,
country, and alert level (Green/Orange/Red).

Per docs/EXPERIMENT_PLAN.md Phase 1 — this script ONLY builds catalog
metadata (id / lat / lon / dates / alert level). Image fetch and best-pair
selection are Phase 2's job (`scripts/auto_fill_pairs.py`). NO SimSat call.

S2 detectability of volcanic events:
  - lava flow → SWIR/NIR strong, NBR drop
  - ash deposit → high-albedo gray, NDVI drop
  - Best with size_km=10 (typical lava extents 1-10 km).

Output: `data/metadata/disaster_m3/volcanic_cases.yaml`.

Usage:
  python scripts/collect_volcanic.py                 # all 100ish events
  python scripts/collect_volcanic.py --target 30
  python scripts/collect_volcanic.py --min-alertlevel Orange   # exclude Green
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


GDACS_API = "https://www.gdacs.org/gdacsapi/api/Events/Geteventlist/SEARCH"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "volcanic_cases.yaml"

ALERTLEVEL_RANK = {"Green": 0, "Orange": 1, "Red": 2}


def fetch_volcanic_events(start_year: int, end_year: int,
                            alertlevels: str = "Green,Orange,Red") -> list[dict]:
    """Fetch VO events between years (inclusive). Returns GeoJSON Feature list."""
    qs = (
        f"eventlist=VO"
        f"&fromdate={start_year}-01-01"
        f"&todate={end_year}-12-31"
        f"&alertlevel={alertlevels.replace(',', '%2C')}"
    )
    url = f"{GDACS_API}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "SatelliteAgent/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d.get("features", [])


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
    p.add_argument("--start-year", type=int, default=2017)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--min-alertlevel", default="Green",
                   choices=["Green", "Orange", "Red"],
                   help="Drop events below this alert level (Green=all)")
    p.add_argument("--size-km", type=float, default=10.0)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--before-offset-days", type=int, default=-90)
    p.add_argument("--after-offset-days", type=int, default=14)
    args = p.parse_args()

    print(f"[1/2] Fetching volcanic events from GDACS ({args.start_year}–{args.end_year})...", flush=True)
    feats = fetch_volcanic_events(args.start_year, args.end_year)
    print(f"      got {len(feats)} events", flush=True)

    threshold = ALERTLEVEL_RANK[args.min_alertlevel]
    selected = [
        f for f in feats
        if ALERTLEVEL_RANK.get(f.get("properties", {}).get("alertlevel", "Green"), 0) >= threshold
    ]
    print(f"      filtered → {len(selected)} (alertlevel ≥ {args.min_alertlevel})", flush=True)

    out_path = Path(args.out)
    cases, existing_ids = load_existing(out_path)
    print(f"      existing yaml: {len(cases)} cases", flush=True)

    print(f"\n[2/2] Building catalog entries...", flush=True)
    new_added = skipped_dup = skipped_bad = 0
    target_remaining = args.target if args.target is not None else 10**9
    for f in selected:
        if target_remaining <= 0:
            break
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            skipped_bad += 1
            continue
        lon, lat = float(coords[0]), float(coords[1])
        event_id = props.get("eventid")
        episode_id = props.get("episodeid", 1)
        eventname = props.get("eventname") or props.get("name") or f"VO_{event_id}"
        case_id = f"volcano_gdacs_{event_id}_{episode_id}"
        if case_id in existing_ids:
            skipped_dup += 1
            continue
        et = parse_iso(props.get("fromdate"))
        end_t = parse_iso(props.get("todate"))
        if et is None:
            skipped_bad += 1
            continue
        before_date = (et + timedelta(days=args.before_offset_days)).strftime("%Y-%m-%d")
        after_date  = (et + timedelta(days=args.after_offset_days)).strftime("%Y-%m-%d")
        cases.append({
            "id":              case_id,
            "gdacs_event_id":  event_id,
            "gdacs_episode":   episode_id,
            "source":          "GDACS_VO",
            "category":        "Volcano",
            "event_type":      "volcanic",
            "expected_action": "submit_to_ground",
            "name":            eventname,
            "country":         props.get("country"),
            "iso3":            props.get("iso3"),
            "alertlevel":      props.get("alertlevel"),
            "alertscore":      props.get("alertscore"),
            "lat":             round(lat, 4),
            "lon":             round(lon, 4),
            "size_km":         args.size_km,
            "before_date":     before_date,
            "after_date":      after_date,
            "window_days":     args.window_days,
            "event_time":      et.isoformat(),
            "event_end":       end_t.isoformat() if end_t else None,
            "report_url":      (props.get("url") or {}).get("report"),
        })
        existing_ids.add(case_id)
        new_added += 1
        target_remaining -= 1

    save_yaml(out_path, cases)
    print(f"\n[done] +{new_added} new (skip dup={skipped_dup}, bad={skipped_bad}); total {len(cases)} → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
