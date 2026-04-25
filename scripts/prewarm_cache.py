"""Pre-fetch Before/After image pairs for every DM3 case at 50km / 10m.

This populates the canonical training cache used by Phase 4 (Annotate) and
Phase 5 (GRPO). Each fetch is sent through SatelliteAgent's /api/fetch so
sidecar metadata + quality stats are computed and saved alongside the PNG.

Usage:
    # Default: hits localhost:7860, fetches at 50km/10m
    python scripts/prewarm_cache.py

    # Different SatelliteAgent instance
    SAT_BASE=http://gpu-host:7860 python scripts/prewarm_cache.py

    # Lower resolution if disk-constrained
    python scripts/prewarm_cache.py --resolution 30

    # Limit to a specific subset for testing
    python scripts/prewarm_cache.py --only-precise   # xBD positives only
    python scripts/prewarm_cache.py --only-negative  # negatives only
    python scripts/prewarm_cache.py --limit 5

Cache files land wherever the server's SAT_CACHE_DIR points (default
data/scenarios/). For remote storage, run this on the host that has the
cache mounted.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import requests


SAT_BASE = os.environ.get("SAT_BASE", "http://localhost:7860")


def fetch_dm3_cases() -> list[dict]:
    r = requests.get(f"{SAT_BASE}/api/disasterm3/cases", timeout=30)
    r.raise_for_status()
    return r.json().get("cases", [])


def fetch_pair(c: dict, resolution_meters: int) -> tuple[bool, dict]:
    body = {
        "lat": c["lat"], "lon": c["lon"],
        "before_date": c["before_date"], "after_date": c["after_date"],
        "size_km": c.get("size_km", 50.0),
        "window_days": 30,
        "resolution_meters": resolution_meters,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{SAT_BASE}/api/fetch", json=body, timeout=600)
    except requests.RequestException as e:
        return False, {"error": f"{type(e).__name__}: {e}", "elapsed_s": time.time() - t0}
    elapsed = time.time() - t0
    if not r.ok:
        return False, {"error": f"HTTP {r.status_code}", "elapsed_s": elapsed}
    d = r.json()
    bm = (d.get("before") or {}).get("meta", {})
    am = (d.get("after")  or {}).get("meta", {})
    info = {
        "elapsed_s": round(elapsed, 1),
        "before_cached": bm.get("cached"),
        "after_cached":  am.get("cached"),
        "before_avail":  bm.get("image_available"),
        "after_avail":   am.get("image_available"),
        "before_cloud":  (bm.get("stats") or {}).get("cloud_proxy"),
        "after_cloud":   (am.get("stats") or {}).get("cloud_proxy"),
        "before_nodata": (bm.get("stats") or {}).get("nodata_fraction"),
        "after_nodata":  (am.get("stats") or {}).get("nodata_fraction"),
        "before_key":    (d.get("before") or {}).get("key"),
        "after_key":     (d.get("after")  or {}).get("key"),
    }
    ok = bool(info["before_avail"]) and bool(info["after_avail"])
    return ok, info


def fmt(info):
    bc = "C" if info.get("before_cached") else "N" if info.get("before_avail") else "x"
    ac = "C" if info.get("after_cached")  else "N" if info.get("after_avail")  else "x"
    bcl = info.get("before_cloud"); acl = info.get("after_cloud")
    bnd = info.get("before_nodata"); a_nd = info.get("after_nodata")
    return (
        f"B[{bc}] cloud={bcl:.2f} nd={bnd:.2f}  "
        f"A[{ac}] cloud={acl:.2f} nd={a_nd:.2f}"
        if all(v is not None for v in [bcl, acl, bnd, a_nd])
        else f"B[{bc}] A[{ac}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10,
                        help="resolution_meters (default 10 for full fidelity)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap how many cases to fetch (debug)")
    parser.add_argument("--only-precise", action="store_true")
    parser.add_argument("--only-negative", action="store_true")
    args = parser.parse_args()

    print(f"[1/3] SAT_BASE = {SAT_BASE}")
    print(f"[2/3] Fetching DM3 case list ...")
    cases = fetch_dm3_cases()
    if args.only_precise:
        cases = [c for c in cases if c.get("precise")]
    if args.only_negative:
        cases = [c for c in cases if c.get("is_negative")]
    if args.limit:
        cases = cases[:args.limit]
    print(f"     -> {len(cases)} cases queued")
    print(f"[3/3] Pre-warming Before/After at {args.resolution}m ...")

    started = datetime.now(timezone.utc)
    n_ok = 0
    n_fail = 0
    n_cached_hit = 0
    total_seconds = 0.0
    for i, c in enumerate(cases, 1):
        cid = c.get("id", "?")
        print(f"  [{i:3d}/{len(cases)}] {cid:55s}", end=" ", flush=True)
        ok, info = fetch_pair(c, args.resolution)
        total_seconds += info.get("elapsed_s", 0)
        if ok:
            n_ok += 1
            if info.get("before_cached") and info.get("after_cached"):
                n_cached_hit += 1
            print(f"OK  {info['elapsed_s']:5.1f}s  {fmt(info)}")
        else:
            n_fail += 1
            err = info.get("error") or f"avail b={info.get('before_avail')} a={info.get('after_avail')}"
            print(f"FAIL  {info.get('elapsed_s', 0):5.1f}s  {err}")

    finished = datetime.now(timezone.utc)
    duration = (finished - started).total_seconds()
    print()
    print(f"[done] {len(cases)} cases in {duration:.0f}s wall, {total_seconds:.0f}s aggregate fetch")
    print(f"       OK: {n_ok}   FAIL: {n_fail}   already cached: {n_cached_hit}")


if __name__ == "__main__":
    main()
