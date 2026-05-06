"""Collect Copernicus EMS Rapid Mapping activations as Phase 1 catalog entries.

Per docs/EXPERIMENT_PLAN.md Phase 1: this script ONLY builds catalog metadata
(id / lat / lon / dates / category). Image fetch, cloud/nodata filtering and
best-pair selection are Phase 2's job (`scripts/auto_fill_pairs.py`).
We do NOT call SimSat here.

Output: `data/metadata/disaster_m3/ems_cases.yaml`. Per-case fields:
  - id, ems_code, source=EMS, category, event_type
  - lat, lon, size_km
  - event_time, before_date, after_date (deterministic anchors; Phase 2 may
    refine via probe)
  - name, countries, n_aois, n_products, closed, gdacs_id

Categories included (complement MCD64A1 wildfire):
  Flood, Storm, Earthquake, Mass movement (= landslide).

Usage:
  python scripts/collect_ems.py                       # all included → yaml
  python scripts/collect_ems.py --only-category Flood  # one category
  python scripts/collect_ems.py --target 50            # cap N
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml


EMS_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "ems_cases.yaml"

INCLUDED_CATEGORIES = {"Flood", "Storm", "Earthquake", "Mass movement"}

CATEGORY_TO_EVENT_TYPE = {
    "Flood":         "flood",
    "Storm":         "storm",
    "Earthquake":    "earthquake",
    "Mass movement": "landslide",
}

# WKT POINT format: "POINT (lon lat)"
WKT_POINT_RE = re.compile(r"POINT\s*\(\s*([\-\d.]+)\s+([\-\d.]+)\s*\)")


def fetch_all_activations() -> list[dict]:
    items: list[dict] = []
    url: str | None = EMS_API + "?limit=100&offset=0"
    while url:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
        items.extend(d.get("results", []))
        url = d.get("next")
    return items


def parse_centroid(wkt: str) -> tuple[float, float] | None:
    m = WKT_POINT_RE.search(wkt or "")
    if not m:
        return None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lat, lon


def parse_event_time(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        elif "+" not in s and "T" in s:
            s = s + "+00:00"
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
    p.add_argument("--target", type=int, default=None,
                   help="Cap number of NEW cases (default: include all)")
    p.add_argument("--only-category", default=None,
                   help="Restrict to one EMS category (Flood / Storm / Earthquake / Mass movement)")
    p.add_argument("--size-km", type=float, default=10.0)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--before-offset-days", type=int, default=-90,
                   help="before_date = eventTime + offset (negative). Phase 2 refines via probe.")
    p.add_argument("--after-offset-days", type=int, default=14,
                   help="after_date  = eventTime + offset.")
    args = p.parse_args()

    print(f"[1/2] Fetching activation list from EMS API...", flush=True)
    activations = fetch_all_activations()
    print(f"      got {len(activations)} activations", flush=True)

    cats_filter = INCLUDED_CATEGORIES
    if args.only_category:
        cats_filter = {args.only_category}
    selected = [it for it in activations if it.get("category") in cats_filter]
    print(f"      filtered → {len(selected)} (categories: {sorted(cats_filter)})", flush=True)

    out_path = Path(args.out)
    cases, existing_ids = load_existing(out_path)
    print(f"      existing yaml: {len(cases)} cases", flush=True)

    print(f"\n[2/2] Building catalog entries (no SimSat fetch — Phase 2 handles imagery)...", flush=True)
    new_added = 0
    skipped_dup = 0
    skipped_bad = 0
    target_remaining = args.target if args.target is not None else 10**9
    for it in selected:
        if target_remaining <= 0:
            break
        code = it.get("code") or ""
        cat = it.get("category") or ""
        event_type = CATEGORY_TO_EVENT_TYPE.get(cat, cat.lower().replace(" ", "_"))
        case_id = f"ems_{code}"
        if case_id in existing_ids:
            skipped_dup += 1
            continue
        cen = parse_centroid(it.get("centroid", ""))
        et  = parse_event_time(it.get("eventTime", ""))
        if cen is None or et is None:
            skipped_bad += 1
            continue
        lat, lon = cen
        before_date = (et + timedelta(days=args.before_offset_days)).strftime("%Y-%m-%d")
        after_date  = (et + timedelta(days=args.after_offset_days)).strftime("%Y-%m-%d")
        cases.append({
            "id":              case_id,
            "ems_code":        code,
            "source":          "EMS",
            "category":        cat,
            "event_type":      event_type,
            "expected_action": "submit_to_ground",
            "name":            it.get("name"),
            "countries":       it.get("countries") or [],
            "lat":             round(lat, 4),
            "lon":             round(lon, 4),
            "size_km":         args.size_km,
            "before_date":     before_date,
            "after_date":      after_date,
            "window_days":     args.window_days,
            "event_time":      et.isoformat(),
            "n_aois":          it.get("n_aois"),
            "n_products":      it.get("n_products"),
            "closed":          it.get("closed"),
            "gdacs_id":        it.get("gdacsId"),
        })
        existing_ids.add(case_id)
        new_added += 1
        target_remaining -= 1

    save_yaml(out_path, cases)
    print(f"\n[done] +{new_added} new cases (skipped dup={skipped_dup}, bad={skipped_bad}); total {len(cases)} → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
