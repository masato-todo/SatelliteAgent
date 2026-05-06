"""Bulk-collect ReAct traces by hitting /api/run_agent for every canonical scene.

Server side already auto-saves to data/traces/agent/<scene_id>__<timestamp>.yaml,
so this script only orchestrates the calls + reports stats.

Usage:
    uv run python scripts/collect_agent_traces.py
    uv run python scripts/collect_agent_traces.py --provider gemini --model gemini-2.5-flash
    uv run python scripts/collect_agent_traces.py --replicas 3
    uv run python scripts/collect_agent_traces.py --only-id mcd64a1_h03v06_202308_p2079_-15640
    uv run python scripts/collect_agent_traces.py --skip-existing       # default
    uv run python scripts/collect_agent_traces.py --no-skip-existing
    uv run python scripts/collect_agent_traces.py --limit 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml


PROJ_ROOT       = Path(__file__).resolve().parent.parent
CANONICAL_PATH  = PROJ_ROOT / "data" / "canonical_dataset.yaml"
AGENT_TRACES    = PROJ_ROOT / "data" / "traces" / "agent"
SAT_BASE        = os.environ.get("SAT_BASE", "http://localhost:7860")


def cache_key(lat: float, lon: float, ts: str, size_km: float, res: int = 10) -> str:
    """Mirror app.server._cache_key (md5 of normalized request)."""
    if "T" not in ts:
        ts = ts + "T00:00:00Z"
    elif not ts.endswith("Z"):
        ts = ts + "Z"
    return hashlib.md5(f"{lat:.4f}_{lon:.4f}_{ts}_{size_km}_r{res}".encode()).hexdigest()[:10]


def safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s.strip())


def has_recent_trace(scene_id: str) -> bool:
    slug = safe_slug(scene_id)
    return any(p.name.startswith(f"{slug}__") for p in AGENT_TRACES.glob("*.yaml"))


def ensure_cache(case: dict) -> tuple[str, str] | None:
    """Trigger /api/fetch so the Before/After PNGs are in cache before run_agent."""
    body = {
        "lat":         float(case["lat"]),
        "lon":         float(case["lon"]),
        "before_date": case["request"]["before_date"],
        "after_date":  case["request"]["after_date"],
        "size_km":     float(case["size_km"]),
        "window_days": int(case["request"].get("window_days", 30)),
        "resolution_meters": 10,
    }
    try:
        r = requests.post(f"{SAT_BASE}/api/fetch", json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"    fetch error: {type(e).__name__}: {e}")
        return None
    bk = (d.get("before") or {}).get("key")
    ak = (d.get("after")  or {}).get("key")
    if not (bk and ak):
        return None
    return bk, ak


def stream_run_agent(scene_id: str, before_key: str, after_key: str,
                     provider: str, model: str, timeout: float = 600.0) -> tuple[str, str]:
    """SSE-consume /api/run_agent. Returns (final_action, last_status_msg)."""
    qp = {
        "before_key": before_key, "after_key": after_key,
        "provider":   provider,   "model": model,
        "scene_id":   scene_id,
    }
    url = f"{SAT_BASE}/api/run_agent"
    final_action = "(none)"
    last_msg = ""
    n_events = 0
    try:
        with requests.get(url, params=qp, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                if not raw.startswith(b"data:"):
                    continue
                try:
                    payload = json.loads(raw[5:].strip().decode())
                except json.JSONDecodeError:
                    continue
                n_events += 1
                t = payload.get("type")
                if t == "final":
                    final_action = payload.get("name") or payload.get("action") or "(unknown)"
                elif t == "error":
                    last_msg = payload.get("text", "") or last_msg
                elif t == "note":
                    last_msg = payload.get("text", "")
    except Exception as e:
        return "(error)", f"{type(e).__name__}: {e}"
    return final_action, f"{n_events} events. {last_msg}".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model",    default="gemini-2.5-flash")
    ap.add_argument("--replicas", type=int, default=1, help="how many traces per scene")
    ap.add_argument("--only-id",  default=None)
    ap.add_argument("--limit",    type=int, default=0)
    ap.add_argument("--skip-existing",    dest="skip", action="store_true",  default=True)
    ap.add_argument("--no-skip-existing", dest="skip", action="store_false")
    args = ap.parse_args()

    if not CANONICAL_PATH.exists():
        print(f"FAIL: {CANONICAL_PATH} not found")
        return 2
    with open(CANONICAL_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cases = doc.get("cases") or []
    if args.only_id:
        cases = [c for c in cases if c.get("id") == args.only_id]
    if args.limit:
        cases = cases[: args.limit]

    AGENT_TRACES.mkdir(parents=True, exist_ok=True)

    print(f"[init] SAT_BASE={SAT_BASE}  provider={args.provider}  model={args.model}")
    print(f"[init] {len(cases)} canonical entries × {args.replicas} replicas")

    n_ok = n_fail = n_skip = 0
    t_start = time.time()
    for i, case in enumerate(cases, 1):
        sid = case["id"]
        for rep in range(args.replicas):
            tag = f"  [{i:>3}/{len(cases)} r{rep+1}/{args.replicas}] {sid:<55}"
            if args.skip and args.replicas == 1 and has_recent_trace(sid):
                print(f"{tag} SKIP (existing trace)")
                n_skip += 1
                continue
            keys = ensure_cache(case)
            if not keys:
                print(f"{tag} FAIL (cache fetch failed)")
                n_fail += 1
                continue
            bk, ak = keys
            t0 = time.time()
            final_action, msg = stream_run_agent(sid, bk, ak, args.provider, args.model)
            elapsed = time.time() - t0
            expected = case.get("expected_action") or (
                "drop" if case.get("type") == "negative" else "submit_to_ground"
            )
            ok = final_action == expected
            flag = "OK  " if ok else "MISS"
            if final_action.startswith("(") and final_action != "(none)":
                flag = "FAIL"
                n_fail += 1
            elif ok:
                n_ok += 1
            else:
                n_fail += 1
            print(f"{tag} {flag}  {elapsed:5.1f}s  final={final_action} (expected={expected})  {msg}")

    elapsed_total = time.time() - t_start
    print(f"\n[done] {len(cases)*args.replicas} runs in {elapsed_total:.0f}s  OK={n_ok}  MISS/FAIL={n_fail}  SKIP={n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
