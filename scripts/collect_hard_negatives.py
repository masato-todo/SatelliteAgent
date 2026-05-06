"""Collect HARD-NEGATIVE catalog entries — same lat/lon as positive cases,
but in stable (pre-event) periods so before/after look identical.

Why: existing `negative_cases.yaml` contains biome-diverse but boring scenes
(deserts, oceans, polar). With those alone, an agent can spuriously learn
"forest visible → change happens" or "volcano visible → eruption". Hard
negatives place the agent in the SAME visual context as positives, then
demand a `drop` decision because the time pair shows no real change.

Strategy per source:
  - Volcanic (GDACS):    event_time - 2y → seasonal pair (apr-sep) of that year
  - Deforestation (PRODES): image_date - 3y → pair ~3y before clearing
  - Wildfire (MCD64A1):  event_period[0] - 2y → pair 2y before burn

Output: `data/metadata/disaster_m3/hard_negative_cases.yaml` (separate
file from the existing negative_cases.yaml so curation stays clean).

Per docs/EXPERIMENT_PLAN.md Phase 1: catalog only, no SimSat call.
Phase 2's `auto_fill_pairs.py --only-hard-negative` actually fetches
imagery and verifies that the date pair is indeed stable (low cloud,
both sides usable).

Usage:
  python scripts/collect_hard_negatives.py
  python scripts/collect_hard_negatives.py --target-volcanic 50 --target-deforestation 50 --target-wildfire 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "hard_negative_cases.yaml"
VOLCANIC_PATH      = ROOT / "data" / "metadata" / "disaster_m3" / "volcanic_cases.yaml"
DEFORESTATION_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "deforestation_cases.yaml"
WILDFIRE_CATALOG   = ROOT / "data" / "scene_catalog.yaml"  # MCD64A1 catalog


def load_yaml_cases(path: Path, key: str = "cases") -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if isinstance(doc, dict):
        # scene_catalog.yaml uses "scenes", others use "cases"
        return doc.get(key) or doc.get("scenes") or []
    return []


def parse_year(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(s[:4])
    except Exception:
        return None


def make_volcanic_hardneg(case: dict, lookback_years: int = 2) -> dict | None:
    et_year = parse_year(case.get("event_time"))
    if et_year is None:
        return None
    target_y = et_year - lookback_years
    return {
        "id":              f"hardneg_volcano_{case['gdacs_event_id']}_{case['gdacs_episode']}",
        "parent_source":   "GDACS_VO",
        "parent_id":       case["id"],
        "negative_type":   "stable_volcano",
        "biome":           "volcano",
        "expected_action": "drop",
        "lat":             float(case["lat"]),
        "lon":             float(case["lon"]),
        "size_km":         float(case.get("size_km", 10.0)),
        "before_date":     f"{target_y}-04-15",
        "after_date":      f"{target_y}-09-15",
        "window_days":     30,
        "note":            f"Same site as {case.get('name','volcano')} but {lookback_years}y before eruption (quiet)",
    }


def make_deforestation_hardneg(case: dict, lookback_years: int = 3) -> dict | None:
    img_y = parse_year(case.get("image_date"))
    if img_y is None:
        return None
    target_y = img_y - lookback_years
    if target_y < 2017:  # S2 global coverage starts ~2017
        return None
    return {
        "id":              f"hardneg_forest_{case['prodes_uid']}",
        "parent_source":   "PRODES",
        "parent_id":       case["id"],
        "negative_type":   "stable_forest",
        "biome":           "forest_amazon",
        "expected_action": "drop",
        "lat":             float(case["lat"]),
        "lon":             float(case["lon"]),
        "size_km":         float(case.get("size_km", 10.0)),
        "before_date":     f"{target_y}-04-15",
        "after_date":      f"{target_y}-09-15",
        "window_days":     30,
        "note":            f"Same site as PRODES clearing {case.get('year')} but {lookback_years}y before (forest intact)",
    }


def make_wildfire_hardneg(scene: dict, lookback_years: int = 2) -> dict | None:
    """scene from data/scene_catalog.yaml (MCD64A1 list)."""
    period = scene.get("event_period")
    burn_year = None
    if isinstance(period, list) and period:
        burn_year = parse_year(period[0])
    if burn_year is None:
        return None
    target_y = burn_year - lookback_years
    if target_y < 2017:
        return None
    return {
        "id":              f"hardneg_preburn_{scene['id']}",
        "parent_source":   "MCD64A1",
        "parent_id":       scene["id"],
        "negative_type":   "pre_burn",
        "biome":           "forest_or_grassland",
        "expected_action": "drop",
        "lat":             float(scene["lat"]),
        "lon":             float(scene["lon"]),
        "size_km":         float(scene.get("size_km") or 10.0),
        "before_date":     f"{target_y}-04-15",
        "after_date":      f"{target_y}-09-15",
        "window_days":     30,
        "note":            f"Same site as MCD64A1 burn {burn_year} but {lookback_years}y before (vegetation intact)",
    }


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
    p.add_argument("--target-volcanic", type=int, default=50)
    p.add_argument("--target-deforestation", type=int, default=50)
    p.add_argument("--target-wildfire", type=int, default=30)
    p.add_argument("--volcanic-lookback", type=int, default=2)
    p.add_argument("--deforestation-lookback", type=int, default=3)
    p.add_argument("--wildfire-lookback", type=int, default=2)
    args = p.parse_args()

    print("[1/4] Loading parent catalogs...", flush=True)
    vol_cases = load_yaml_cases(VOLCANIC_PATH)
    def_cases = load_yaml_cases(DEFORESTATION_PATH)
    wf_scenes = load_yaml_cases(WILDFIRE_CATALOG, key="scenes")
    print(f"      volcanic={len(vol_cases)}  deforestation={len(def_cases)}  wildfire(MCD64A1)={len(wf_scenes)}", flush=True)

    out: list[dict] = []

    print(f"\n[2/4] Volcanic hard negatives (lookback {args.volcanic_lookback}y)...", flush=True)
    n = 0
    for c in vol_cases:
        if n >= args.target_volcanic:
            break
        hn = make_volcanic_hardneg(c, args.volcanic_lookback)
        if hn:
            out.append(hn); n += 1
    print(f"      +{n}", flush=True)

    print(f"\n[3/4] Deforestation hard negatives (lookback {args.deforestation_lookback}y)...", flush=True)
    n = 0
    for c in def_cases:
        if n >= args.target_deforestation:
            break
        hn = make_deforestation_hardneg(c, args.deforestation_lookback)
        if hn:
            out.append(hn); n += 1
    print(f"      +{n}", flush=True)

    print(f"\n[4/4] Wildfire hard negatives (lookback {args.wildfire_lookback}y)...", flush=True)
    n = 0
    for s in wf_scenes:
        if n >= args.target_wildfire:
            break
        hn = make_wildfire_hardneg(s, args.wildfire_lookback)
        if hn:
            out.append(hn); n += 1
    print(f"      +{n}", flush=True)

    save_yaml(Path(args.out), out)
    print(f"\n[done] {len(out)} hard-negative cases → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
