"""Automate Phase 2 (good Before/After date discovery) for DM3 positive cases.

Flow per case:
  1. Probe Before candidates around event_start (10km @ 10m, ~5s each).
  2. Probe After  candidates around event_end   (10km @ 10m).
  3. Pick the best pair (usable=True, lowest cloud_proxy + nodata_fraction).
  4. Append the resolved (lat, lon, dates, event period) to canonical_dataset.yaml.
     The probe images are themselves the training cache — no separate finalize.

Usage:
    uv run python scripts/auto_phase2.py
    uv run python scripts/auto_phase2.py --only-event socal_fire
    uv run python scripts/auto_phase2.py --limit 5
    uv run python scripts/auto_phase2.py --output data/canonical_dataset.yaml --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml


SAT_BASE = os.environ.get("SAT_BASE", "http://localhost:7860")
CACHE_DIR = Path(os.environ.get(
    "SAT_CACHE_DIR",
    Path(__file__).resolve().parent.parent / "data" / "scenarios"
))

# Probe at small AOI / native res (SimSat ignores resolution_meters anyway).
# 10km/10m ≈ 3MB per image, ~5s each. The probe IS the training cache —
# no separate 50km finalize, since 10km centered on the xBD disaster centroid
# is enough context and keeps the cache footprint small.
PROBE_SIZE_KM = 10.0
PROBE_RESOLUTION_M = 10

# Reduced offsets to keep total fetches manageable.
BEFORE_OFFSETS = [14, 30, 60, 90]
AFTER_OFFSETS  = [0, 7, 14, 21, 30]

CLOUD_MAX = 0.30
NODATA_MAX = 0.20


def fetch_dm3_cases() -> list[dict]:
    r = requests.get(f"{SAT_BASE}/api/disasterm3/cases", timeout=30)
    r.raise_for_status()
    return r.json().get("cases", [])


def probe_candidates(side: str, lat: float, lon: float, anchor_date: str,
                     fallback_after_date: str) -> list[dict]:
    """side = 'before' or 'after'. Returns list of {target_date, key, meta}."""
    offsets = BEFORE_OFFSETS if side == "before" else AFTER_OFFSETS
    body = {
        "lat": lat,
        "lon": lon,
        "after_date": fallback_after_date,
        "anchor_date": anchor_date,
        "size_km": PROBE_SIZE_KM,
        "resolution_meters": PROBE_RESOLUTION_M,
        "offsets_days": offsets,
    }
    r = requests.post(f"{SAT_BASE}/api/{side}_candidates", json=body, timeout=300)
    r.raise_for_status()
    return r.json().get("candidates", [])


def score_candidate(c: dict) -> float | None:
    """Lower is better. None if disqualified."""
    meta = c.get("meta") or {}
    if not meta.get("image_available"):
        return None
    stats = meta.get("stats") or {}
    if not stats.get("usable"):
        return None
    cloud = stats.get("cloud_proxy", 1.0)
    nodata = stats.get("nodata_fraction", 1.0)
    if cloud > CLOUD_MAX or nodata > NODATA_MAX:
        return None
    return cloud + nodata


def pick_best(candidates: list[dict]) -> dict | None:
    scored = [(score_candidate(c), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s is not None]
    if not scored:
        return None
    scored.sort(key=lambda sc: sc[0])
    return scored[0][1]


def cleanup_rejected(candidates: list[dict], picked_key: str) -> int:
    """Delete cache files (PNG + meta.json) for candidates other than the picked one."""
    n = 0
    for c in candidates:
        k = c.get("key")
        if not k or k == picked_key:
            continue
        for suffix in (".png", ".meta.json"):
            p = CACHE_DIR / f"{k}{suffix}"
            if p.exists():
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
    return n


def load_existing_yaml(path: Path) -> dict:
    if not path.exists():
        return {"cases": []}
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if "cases" not in doc:
        doc["cases"] = []
    return doc


def save_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def case_to_yaml_entry(c: dict, before_meta: dict, after_meta: dict,
                       before_date: str, after_date: str) -> dict:
    return {
        "id": c["id"],
        "label": c.get("mapped_class") or c.get("disaster_type"),
        "type": "positive",
        "lat": float(c["lat"]),
        "lon": float(c["lon"]),
        "size_km": PROBE_SIZE_KM,
        "request": {
            "before_date": before_date,
            "after_date":  after_date,
            "window_days": 30,
        },
        "expected_resolved": {
            "before_datetime": before_meta.get("datetime"),
            "after_datetime":  after_meta.get("datetime"),
        },
        "event": {
            "name":   c.get("event_name") or c.get("event"),
            "period": [c.get("event_start"), c.get("event_end")],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-event", help="filter to a single DM3 event (substring match on c['event'])")
    ap.add_argument("--only-id", help="filter to a single case id")
    ap.add_argument("--limit", type=int, default=0, help="cap number of cases processed (0 = all)")
    ap.add_argument("--output", default="data/canonical_dataset.yaml")
    ap.add_argument("--force", action="store_true", help="re-process cases already in the yaml")
    args = ap.parse_args()

    out_path = Path(args.output).resolve()
    doc = load_existing_yaml(out_path)
    seen_ids = {e["id"] for e in doc.get("cases", []) if "id" in e}

    print(f"[1/4] SAT_BASE = {SAT_BASE}")
    print(f"[2/4] output   = {out_path}  (existing entries: {len(seen_ids)})")
    print(f"[3/4] Fetching DM3 case list ...")
    all_cases = fetch_dm3_cases()
    cases = [c for c in all_cases if not c.get("is_negative")]
    if args.only_event:
        cases = [c for c in cases if args.only_event in (c.get("event") or "")]
    if args.only_id:
        cases = [c for c in cases if c.get("id") == args.only_id]
    if args.limit:
        cases = cases[: args.limit]
    print(f"     -> {len(cases)} positive cases queued (of {len(all_cases)} total)")
    print(f"[4/4] Probing @ {PROBE_SIZE_KM}km / {PROBE_RESOLUTION_M}m ...")

    n_ok = n_skip = n_fail = 0
    t_start = time.time()
    for i, c in enumerate(cases, 1):
        cid = c["id"]
        tag = f"  [{i:>3}/{len(cases)}] {cid:<55}"

        if cid in seen_ids and not args.force:
            print(f"{tag} SKIP (already in yaml)")
            n_skip += 1
            continue

        anchor_b = c.get("event_start") or c.get("before_date")
        anchor_a = c.get("event_end")   or c.get("after_date")
        if not (anchor_b and anchor_a):
            print(f"{tag} FAIL  (missing event_start/end)")
            n_fail += 1
            continue

        t0 = time.time()
        try:
            before_cands = probe_candidates("before", c["lat"], c["lon"], anchor_b, c["after_date"])
            after_cands  = probe_candidates("after",  c["lat"], c["lon"], anchor_a, c["after_date"])
        except Exception as e:
            print(f"{tag} FAIL  probe-error: {type(e).__name__}: {e}")
            n_fail += 1
            continue

        before_pick = pick_best(before_cands)
        after_pick  = pick_best(after_cands)
        probe_elapsed = time.time() - t0

        if not before_pick or not after_pick:
            b_have = sum(1 for x in before_cands if score_candidate(x) is not None)
            a_have = sum(1 for x in after_cands  if score_candidate(x) is not None)
            print(f"{tag} FAIL  no usable pair  probe={probe_elapsed:.1f}s  good_b={b_have}/{len(before_cands)} good_a={a_have}/{len(after_cands)}")
            n_fail += 1
            continue

        before_date = before_pick["target_date"]
        after_date  = after_pick["target_date"]

        n_cleaned  = cleanup_rejected(before_cands, before_pick["key"])
        n_cleaned += cleanup_rejected(after_cands,  after_pick["key"])

        bm = before_pick.get("meta") or {}
        am = after_pick.get("meta")  or {}
        b_stats = bm.get("stats") or {}
        a_stats = am.get("stats") or {}

        entry = case_to_yaml_entry(c, bm, am, before_date, after_date)
        doc["cases"] = [e for e in doc["cases"] if e.get("id") != cid] + [entry]
        save_yaml(out_path, doc)
        seen_ids.add(cid)

        print(f"{tag} OK    probe={probe_elapsed:.1f}s  cleaned={n_cleaned} files  "
              f"B[{before_date}] cloud={b_stats.get('cloud_proxy', 0):.2f} nd={b_stats.get('nodata_fraction', 0):.2f}  "
              f"A[{after_date}] cloud={a_stats.get('cloud_proxy', 0):.2f} nd={a_stats.get('nodata_fraction', 0):.2f}")
        n_ok += 1

    elapsed = time.time() - t_start
    print(f"\n[done] {len(cases)} cases in {elapsed:.0f}s")
    print(f"       OK: {n_ok}   FAIL: {n_fail}   SKIP: {n_skip}")
    print(f"       canonical_dataset.yaml entries: {len(doc.get('cases', []))}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
