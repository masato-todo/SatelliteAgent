"""Fill in canonical Before/After pairs for Negative + MCD64A1 + Volcanic scenes.

For each unsaved scene in /api/disasterm3/cases:
  - MCD64A1 wildfire: probe candidates around event_period, pick best, save.
  - Negative: fetch the predefined before/after dates directly, save if usable.
  - Volcanic (GDACS_VO): probe with wider offsets and cloud<0.40 (volcanoes
    sit in cloudy mountainous / island regions).
Skips scenes already present in canonical_dataset.yaml.

Usage:
    uv run python scripts/auto_fill_pairs.py
    uv run python scripts/auto_fill_pairs.py --only-mcd64a1
    uv run python scripts/auto_fill_pairs.py --only-negative
    uv run python scripts/auto_fill_pairs.py --only-volcanic
    uv run python scripts/auto_fill_pairs.py --limit 5 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests


SAT_BASE = os.environ.get("SAT_BASE", "http://localhost:7860")

PROBE_SIZE_KM      = 10.0
BEFORE_OFFSETS     = [14, 30, 60, 90]
AFTER_OFFSETS      = [0, 7, 14, 21, 30]
CLOUD_MAX          = 0.30
NODATA_MAX         = 0.20


def list_cases() -> list[dict]:
    r = requests.get(f"{SAT_BASE}/api/disasterm3/cases", timeout=30)
    r.raise_for_status()
    return r.json().get("cases", [])


def probe(side: str, lat: float, lon: float, anchor: str, fallback: str) -> list[dict]:
    body = {
        "lat": lat, "lon": lon,
        "after_date":  fallback,
        "anchor_date": anchor,
        "size_km":     PROBE_SIZE_KM,
        "resolution_meters": 10,
        "offsets_days": BEFORE_OFFSETS if side == "before" else AFTER_OFFSETS,
    }
    r = requests.post(f"{SAT_BASE}/api/{side}_candidates", json=body, timeout=300)
    r.raise_for_status()
    return r.json().get("candidates", [])


def score(cand: dict) -> float | None:
    m = cand.get("meta") or {}
    if not m.get("image_available"):
        return None
    s = m.get("stats") or {}
    if not s.get("usable"):
        return None
    cloud = s.get("cloud_proxy", 1.0)
    nd    = s.get("nodata_fraction", 1.0)
    if cloud > CLOUD_MAX or nd > NODATA_MAX:
        return None
    return cloud + nd


def pick_best(cands: list[dict]) -> dict | None:
    scored = [(score(c), c) for c in cands]
    scored = [(s, c) for s, c in scored if s is not None]
    if not scored:
        return None
    scored.sort(key=lambda sc: sc[0])
    return scored[0][1]


def fetch_pair(lat: float, lon: float, before_date: str, after_date: str,
               size_km: float) -> tuple[dict, dict]:
    """Direct /api/fetch — used for negatives (predefined dates)."""
    body = {
        "lat": lat, "lon": lon,
        "before_date": before_date,
        "after_date":  after_date,
        "size_km":     size_km,
        "window_days": 30,
        "resolution_meters": 10,
    }
    r = requests.post(f"{SAT_BASE}/api/fetch", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d.get("before") or {}, d.get("after") or {}


def save_pair(case: dict, before_date: str, after_date: str, size_km: float,
              before_key: str, after_key: str) -> dict:
    body = {
        "scene_id":      case["id"],
        "lat":           float(case["lat"]),
        "lon":           float(case["lon"]),
        "before_date":   before_date,
        "after_date":    after_date,
        "size_km":       size_km,
        "before_key":    before_key,
        "after_key":     after_key,
        "label":         case.get("mapped_class") or case.get("disaster_type"),
        "event_type":    case.get("disaster_type"),
        "event_start":   case.get("event_start"),
        "event_end":     case.get("event_end"),
        "event_name":    case.get("event_name"),
        "is_negative":   bool(case.get("is_negative")),
        "negative_type": case.get("negative_type"),
        "expected_action": case.get("expected_action"),
    }
    r = requests.post(f"{SAT_BASE}/api/scene/save_pair", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def usable_meta(meta: dict) -> bool:
    if not meta.get("image_available"):
        return False
    s = meta.get("stats") or {}
    if not s.get("usable"):
        return False
    if (s.get("cloud_proxy") or 0) > CLOUD_MAX:
        return False
    if (s.get("nodata_fraction") or 0) > NODATA_MAX:
        return False
    return True


def process_mcd64a1(case: dict, dry: bool) -> tuple[bool, str]:
    lat = float(case["lat"]); lon = float(case["lon"])
    anchor_b = case.get("event_start") or case.get("before_date")
    anchor_a = case.get("event_end")   or case.get("after_date")
    if not (anchor_b and anchor_a):
        return False, "no event_start/end"
    try:
        b_cands = probe("before", lat, lon, anchor_b, case.get("after_date") or anchor_a)
        a_cands = probe("after",  lat, lon, anchor_a, case.get("after_date") or anchor_a)
    except Exception as e:
        return False, f"probe error: {type(e).__name__}: {e}"
    bp, ap = pick_best(b_cands), pick_best(a_cands)
    if not bp or not ap:
        gb = sum(1 for c in b_cands if score(c) is not None)
        ga = sum(1 for c in a_cands if score(c) is not None)
        return False, f"no usable pair (good_b={gb}/{len(b_cands)} good_a={ga}/{len(a_cands)})"
    if dry:
        return True, f"DRY  B[{bp['target_date']}] A[{ap['target_date']}]"
    res = save_pair(case, bp["target_date"], ap["target_date"], PROBE_SIZE_KM,
                    bp["key"], ap["key"])
    return True, f"saved B[{bp['target_date']}] A[{ap['target_date']}]  -> {res.get('saved_dir')}"


def process_volcanic(case: dict, dry: bool) -> tuple[bool, str]:
    """Volcanic events from GDACS. Wider offsets (volcanoes in cloudy
    mountainous/island regions) and cloud<0.40 (vs 0.30 default)."""
    lat = float(case["lat"]); lon = float(case["lon"])
    anchor_b = case.get("before_date")
    anchor_a = case.get("after_date")
    if not (anchor_b and anchor_a):
        return False, "no dates"

    # Probe wider before window (1 year back to dampen seasonal bias) and
    # longer after window (eruptions / lava cooling visible for months).
    body_b = {
        "lat": lat, "lon": lon,
        "after_date": anchor_a, "anchor_date": anchor_b,
        "size_km": PROBE_SIZE_KM, "resolution_meters": 10,
        "offsets_days": [30, 60, 90, 180, 365],
    }
    body_a = {
        "lat": lat, "lon": lon,
        "after_date": anchor_a, "anchor_date": anchor_a,
        "size_km": PROBE_SIZE_KM, "resolution_meters": 10,
        "offsets_days": [0, 7, 14, 21, 30, 45, 60, 90],
    }
    try:
        rb = requests.post(f"{SAT_BASE}/api/before_candidates", json=body_b, timeout=300)
        ra = requests.post(f"{SAT_BASE}/api/after_candidates",  json=body_a, timeout=300)
        rb.raise_for_status(); ra.raise_for_status()
        b_cands = rb.json().get("candidates", [])
        a_cands = ra.json().get("candidates", [])
    except Exception as e:
        return False, f"probe error: {type(e).__name__}: {e}"

    cloud_max = 0.40   # volcano-specific
    def vol_score(cand):
        m = cand.get("meta") or {}
        if not m.get("image_available"):
            return None
        s = m.get("stats") or {}
        if not s.get("usable"):
            return None
        cl = s.get("cloud_proxy", 1.0)
        nd = s.get("nodata_fraction", 1.0)
        if cl > cloud_max or nd > NODATA_MAX:
            return None
        return cl + nd

    bs = sorted([(vol_score(c), c) for c in b_cands if vol_score(c) is not None], key=lambda x: x[0])
    as_ = sorted([(vol_score(c), c) for c in a_cands if vol_score(c) is not None], key=lambda x: x[0])
    if not bs or not as_:
        return False, f"no usable pair (good_b={len(bs)}/{len(b_cands)} good_a={len(as_)}/{len(a_cands)})"
    bp, ap = bs[0][1], as_[0][1]
    if dry:
        return True, f"DRY  B[{bp['target_date']}] A[{ap['target_date']}] alert={case.get('alertlevel')}"
    res = save_pair(case, bp["target_date"], ap["target_date"], PROBE_SIZE_KM,
                    bp["key"], ap["key"])
    return True, f"saved B[{bp['target_date']}] A[{ap['target_date']}] alert={case.get('alertlevel')}  -> {res.get('saved_dir')}"


def process_deforestation(case: dict, dry: bool) -> tuple[bool, str]:
    """PRODES deforestation events. Amazon basin is heavily cloudy; widen
    before window to 1 year (dampen seasonal NDVI variation) and accept
    cloud<0.40 (vs 0.30 default)."""
    lat = float(case["lat"]); lon = float(case["lon"])
    anchor_b = case.get("before_date")
    anchor_a = case.get("after_date")
    if not (anchor_b and anchor_a):
        return False, "no dates"

    body_b = {
        "lat": lat, "lon": lon,
        "after_date": anchor_a, "anchor_date": anchor_b,
        "size_km": PROBE_SIZE_KM, "resolution_meters": 10,
        "offsets_days": [0, 30, 60, 90, 180, 365],
    }
    body_a = {
        "lat": lat, "lon": lon,
        "after_date": anchor_a, "anchor_date": anchor_a,
        "size_km": PROBE_SIZE_KM, "resolution_meters": 10,
        "offsets_days": [0, 7, 14, 21, 30, 60, 90],
    }
    try:
        rb = requests.post(f"{SAT_BASE}/api/before_candidates", json=body_b, timeout=300)
        ra = requests.post(f"{SAT_BASE}/api/after_candidates",  json=body_a, timeout=300)
        rb.raise_for_status(); ra.raise_for_status()
        b_cands = rb.json().get("candidates", [])
        a_cands = ra.json().get("candidates", [])
    except Exception as e:
        return False, f"probe error: {type(e).__name__}: {e}"

    cloud_max = 0.40
    def def_score(cand):
        m = cand.get("meta") or {}
        if not m.get("image_available"):
            return None
        s = m.get("stats") or {}
        if not s.get("usable"):
            return None
        cl = s.get("cloud_proxy", 1.0)
        nd = s.get("nodata_fraction", 1.0)
        if cl > cloud_max or nd > NODATA_MAX:
            return None
        return cl + nd

    bs = sorted([(def_score(c), c) for c in b_cands if def_score(c) is not None], key=lambda x: x[0])
    as_ = sorted([(def_score(c), c) for c in a_cands if def_score(c) is not None], key=lambda x: x[0])
    if not bs or not as_:
        return False, f"no usable pair (good_b={len(bs)}/{len(b_cands)} good_a={len(as_)}/{len(a_cands)})"
    bp, ap = bs[0][1], as_[0][1]
    if dry:
        return True, f"DRY  B[{bp['target_date']}] A[{ap['target_date']}] area={case.get('area_km2')}km²"
    res = save_pair(case, bp["target_date"], ap["target_date"], PROBE_SIZE_KM,
                    bp["key"], ap["key"])
    return True, f"saved B[{bp['target_date']}] A[{ap['target_date']}] area={case.get('area_km2')}km²  -> {res.get('saved_dir')}"


def process_negative(case: dict, dry: bool) -> tuple[bool, str]:
    """Negative scenes are intentionally boring (desert/ocean/cloud) so the
    server's strict `usable` flag (which expects high edge density and low
    dark/cloud fractions) does not apply. We only require that Sentinel-2
    returned actual pixels and the AOI is mostly inside one tile."""
    lat = float(case["lat"]); lon = float(case["lon"])
    bd, ad = case.get("before_date"), case.get("after_date")
    size = float(case.get("size_km", 10.0))
    if not (bd and ad):
        return False, "no dates"
    try:
        b, a = fetch_pair(lat, lon, bd, ad, size)
    except Exception as e:
        return False, f"fetch error: {type(e).__name__}: {e}"
    bm, am = b.get("meta") or {}, a.get("meta") or {}
    if not (bm.get("image_available") and am.get("image_available")):
        return False, f"image not available (B={bm.get('image_available')} A={am.get('image_available')})"
    NODATA_NEG_MAX = 0.30
    bs = bm.get("stats") or {}; as_ = am.get("stats") or {}
    bnd = bs.get("nodata_fraction") or 0.0
    and_ = as_.get("nodata_fraction") or 0.0
    if bnd > NODATA_NEG_MAX or and_ > NODATA_NEG_MAX:
        return False, f"too much nodata (B nd={bnd:.2f} A nd={and_:.2f})"
    nt = case.get("negative_type") or "?"
    if dry:
        return True, f"DRY  B[{bd}] A[{ad}]  type={nt}  nd_b={bnd:.2f} nd_a={and_:.2f}"
    res = save_pair(case, bd, ad, size, b.get("key"), a.get("key"))
    return True, f"saved B[{bd}] A[{ad}]  -> {res.get('saved_dir')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-mcd64a1", action="store_true")
    ap.add_argument("--only-negative", action="store_true")
    ap.add_argument("--only-volcanic", action="store_true")
    ap.add_argument("--only-deforestation", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[1/3] SAT_BASE = {SAT_BASE}")
    cases = list_cases()
    print(f"[2/3] {len(cases)} total cases")

    targets: list[dict] = []
    for c in cases:
        if c.get("canonical_pairs"):
            continue  # already saved
        is_neg = bool(c.get("is_negative"))
        is_mcd = c.get("source") == "MCD64A1"
        is_vol = bool(c.get("is_volcanic"))
        is_def = bool(c.get("is_deforestation"))
        if args.only_mcd64a1 and not is_mcd:
            continue
        if args.only_negative and not is_neg:
            continue
        if args.only_volcanic and not is_vol:
            continue
        if args.only_deforestation and not is_def:
            continue
        if not (is_neg or is_mcd or is_vol or is_def):
            continue
        targets.append(c)
    if args.limit:
        targets = targets[: args.limit]
    print(f"[3/3] {len(targets)} unsaved targets queued")

    n_ok = n_fail = 0
    t_start = time.time()
    for i, c in enumerate(targets, 1):
        if c.get("is_negative"):
            kind = "NEG"
        elif c.get("is_volcanic"):
            kind = "VOL"
        elif c.get("is_deforestation"):
            kind = "DEF"
        else:
            kind = "MCD"
        tag = f"  [{i:>3}/{len(targets)}] {kind} {c['id']:<55}"
        t0 = time.time()
        try:
            if c.get("is_negative"):
                ok, msg = process_negative(c, args.dry_run)
            elif c.get("is_volcanic"):
                ok, msg = process_volcanic(c, args.dry_run)
            elif c.get("is_deforestation"):
                ok, msg = process_deforestation(c, args.dry_run)
            else:
                ok, msg = process_mcd64a1(c, args.dry_run)
        except Exception as e:
            ok, msg = False, f"unexpected: {type(e).__name__}: {e}"
        elapsed = time.time() - t0
        flag = "OK  " if ok else "FAIL"
        print(f"{tag} {flag}  {elapsed:.1f}s  {msg}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n[done] {len(targets)} processed in {time.time() - t_start:.0f}s  OK={n_ok}  FAIL={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
