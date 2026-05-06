"""Phase B precompute — replay TOOL_SPEC §4 enumeration for every canonical
case so Kaggle/offline GRPO can serve tool responses from cache.

Output (relative to repo root):
    data/precompute/<case_id>/
        classify_change.yaml
        compute_index/<INDEX>__{before,after}.{png,stats.yaml}
        compute_index_delta/<INDEX>.{png,stats.yaml}
        fetch_band/<band>__{before,after}.{png,stats.yaml}
        false_color/<R>-<G>-<B>__{before,after}.png
        _stats.json     # per-case timing breakdown

A run-level summary lands at data/precompute/_run_stats.json.

Usage:
    uv run python scripts/build_precompute.py
    uv run python scripts/build_precompute.py --limit 3
    uv run python scripts/build_precompute.py --only-id mcd64a1_h03v06_202308_p2079_-15640
    uv run python scripts/build_precompute.py --tools compute_index_delta classify_change
    uv run python scripts/build_precompute.py --skip-existing      # default
    uv run python scripts/build_precompute.py --no-skip-existing
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml


PROJ_ROOT      = Path(__file__).resolve().parent.parent
CANONICAL_PATH = PROJ_ROOT / "data" / "canonical_dataset.yaml"
OUT_ROOT       = PROJ_ROOT / "data" / "precompute"
SAT_BASE       = os.environ.get("SAT_BASE", "http://localhost:7860")

INDICES = ["NDVI", "NDWI", "MNDWI", "NBR", "NDBI", "NDSI"]
# Drop rarely-used bands (coastal/aot/scl/visual/wvp) to shave ~10 calls/case.
# Agent reasoning works fine without them; add back if a use case appears.
BANDS = ["blue", "green", "red",
         "rededge1", "rededge2", "rededge3",
         "nir", "nir08", "nir09",
         "swir16", "swir22"]
FALSE_COLOR_COMBOS = [
    ("nir", "red", "green"),       # standard vegetation
    ("swir22", "nir", "red"),      # burn severity
    ("swir16", "nir", "blue"),     # urban vs vegetation
    ("nir", "swir16", "red"),      # agricultural
    ("red", "green", "blue"),      # true color
]
SIDES = ["before", "after"]


def cache_key(lat, lon, ts, size_km, res=10):
    if "T" not in ts: ts = ts + "T00:00:00Z"
    elif not ts.endswith("Z"): ts = ts + "Z"
    return hashlib.md5(f"{lat:.4f}_{lon:.4f}_{ts}_{size_km}_r{res}".encode()).hexdigest()[:10]


def keys_from_case(case):
    lat = float(case["lat"]); lon = float(case["lon"])
    size = float(case["size_km"])
    bd = case["request"]["before_date"]; ad = case["request"]["after_date"]
    return cache_key(lat, lon, bd, size), cache_key(lat, lon, ad, size)


def invoke(before_key, after_key, tool, args, timeout=120):
    body = {"before_key": before_key, "after_key": after_key,
            "tool_name": tool, "arguments": args}
    r = requests.post(f"{SAT_BASE}/api/tool/invoke", json=body, timeout=timeout)
    if not r.ok:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    return (r.json() or {}).get("observation") or {}


def fetch_image(key, dest):
    r = requests.get(f"{SAT_BASE}/api/image/{key}", timeout=60)
    if not r.ok:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def write_yaml(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def args_id(args):
    return "_".join(f"{k}={v}" for k, v in sorted(args.items())).replace("/", "_")


def precompute_classify_change(case_dir, before_key, after_key):
    obs = invoke(before_key, after_key, "classify_change", {})
    write_yaml(case_dir / "classify_change.yaml",
               {"args": {}, "response": obs})
    return [("classify_change", "()", "error" not in obs)]


def precompute_compute_index(case_dir, before_key, after_key):
    rows = []
    for idx in INDICES:
        for side in SIDES:
            args = {"index": idx, "which": side}
            obs = invoke(before_key, after_key, "compute_index", args)
            ok = "error" not in obs
            rows.append((f"compute_index", f"{idx}/{side}", ok))
            stub = case_dir / "compute_index" / f"{idx}__{side}"
            write_yaml(stub.with_suffix(".stats.yaml"),
                       {"args": args, "response": obs})
            png_key = (obs or {}).get("image_key")
            if ok and png_key:
                fetch_image(png_key, stub.with_suffix(".png"))
    return rows


def precompute_compute_index_delta(case_dir, before_key, after_key):
    rows = []
    for idx in INDICES:
        args = {"index": idx}
        obs = invoke(before_key, after_key, "compute_index_delta", args)
        ok = "error" not in obs
        rows.append(("compute_index_delta", idx, ok))
        stub = case_dir / "compute_index_delta" / idx
        write_yaml(stub.with_suffix(".stats.yaml"),
                   {"args": args, "response": obs})
        png_key = (obs or {}).get("image_key")
        if ok and png_key:
            fetch_image(png_key, stub.with_suffix(".png"))
    return rows


def precompute_fetch_band(case_dir, before_key, after_key):
    rows = []
    for band in BANDS:
        for side in SIDES:
            args = {"band": band, "which": side}
            obs = invoke(before_key, after_key, "fetch_band", args)
            ok = "error" not in obs
            rows.append(("fetch_band", f"{band}/{side}", ok))
            stub = case_dir / "fetch_band" / f"{band}__{side}"
            write_yaml(stub.with_suffix(".stats.yaml"),
                       {"args": args, "response": obs})
            png_key = (obs or {}).get("image_key")
            if ok and png_key:
                fetch_image(png_key, stub.with_suffix(".png"))
    return rows


def precompute_false_color(case_dir, before_key, after_key):
    rows = []
    for combo in FALSE_COLOR_COMBOS:
        for side in SIDES:
            args = {"bands": list(combo), "which": side}
            obs = invoke(before_key, after_key, "false_color", args)
            ok = "error" not in obs
            tag = f"{combo[0]}-{combo[1]}-{combo[2]}__{side}"
            rows.append(("false_color", tag, ok))
            stub = case_dir / "false_color" / tag
            write_yaml(stub.with_suffix(".stats.yaml"),
                       {"args": args, "response": obs})
            png_key = (obs or {}).get("image_key")
            if ok and png_key:
                fetch_image(png_key, stub.with_suffix(".png"))
    return rows


TOOL_FNS = {
    "classify_change":     precompute_classify_change,
    "compute_index":       precompute_compute_index,
    "compute_index_delta": precompute_compute_index_delta,
    "fetch_band":          precompute_fetch_band,
    "false_color":         precompute_false_color,
}


def case_already_done(case_dir):
    """Crude check: a 'done' marker is whether classify_change.yaml exists AND
    compute_index_delta dir has all 6 indices."""
    if not (case_dir / "classify_change.yaml").exists():
        return False
    delta_dir = case_dir / "compute_index_delta"
    if not delta_dir.exists():
        return False
    return sum(1 for _ in delta_dir.glob("*.stats.yaml")) == len(INDICES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-id", default=None)
    ap.add_argument("--tools", nargs="+", default=list(TOOL_FNS.keys()),
                    choices=list(TOOL_FNS.keys()))
    ap.add_argument("--skip-existing",   dest="skip", action="store_true",  default=True)
    ap.add_argument("--no-skip-existing", dest="skip", action="store_false")
    ap.add_argument("--parallel", type=int, default=1,
                    help="number of cases to process concurrently (1 = sequential)")
    args = ap.parse_args()

    if not CANONICAL_PATH.exists():
        print(f"FAIL: {CANONICAL_PATH} not found"); return 2
    with open(CANONICAL_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cases = doc.get("cases") or []
    if args.only_id:
        cases = [c for c in cases if c.get("id") == args.only_id]
    if args.limit:
        cases = cases[: args.limit]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[init] SAT_BASE={SAT_BASE}")
    print(f"[init] {len(cases)} cases × {len(args.tools)} tool(s) → {OUT_ROOT.relative_to(PROJ_ROOT)}")

    run_stats = {
        "started_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases":       len(cases),
        "tools":         args.tools,
        "skip_existing": args.skip,
        "per_tool_total_seconds": {t: 0.0 for t in args.tools},
        "per_tool_call_count":     {t: 0   for t in args.tools},
        "per_tool_fail_count":     {t: 0   for t in args.tools},
        "per_case": [],
    }

    def process_one(idx_case):
        i, case = idx_case
        cid = case["id"]
        case_dir = OUT_ROOT / cid
        if args.skip and case_already_done(case_dir):
            return (i, cid, None, "SKIP (already done)")
        bk, ak = keys_from_case(case)
        case_dir.mkdir(parents=True, exist_ok=True)

        case_t0 = time.time()
        case_results = {}
        for tool in args.tools:
            tool_t0 = time.time()
            rows = TOOL_FNS[tool](case_dir, bk, ak)
            tool_dt = time.time() - tool_t0
            n_ok   = sum(1 for _, _, ok in rows if ok)
            n_fail = sum(1 for _, _, ok in rows if not ok)
            case_results[tool] = {
                "elapsed_s": round(tool_dt, 2), "n_calls": len(rows),
                "n_ok": n_ok, "n_fail": n_fail,
            }
        case_dt = time.time() - case_t0
        write_yaml(case_dir / "_stats.yaml", {
            "case_id": cid, "elapsed_s": round(case_dt, 1),
            "tools": case_results, "before_key": bk, "after_key": ak,
        })
        return (i, cid, case_results, f"OK   {case_dt:5.1f}s")

    t_run = time.time()
    if args.parallel <= 1:
        results_iter = (process_one(ic) for ic in enumerate(cases, 1))
    else:
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=args.parallel)
        results_iter = ex.map(process_one, list(enumerate(cases, 1)))

    for i, cid, case_results, msg in results_iter:
        if case_results is None:
            print(f"  [{i:>3}/{len(cases)}] {cid:<55} {msg}")
            continue
        for tool, r in case_results.items():
            run_stats["per_tool_total_seconds"][tool] += r["elapsed_s"]
            run_stats["per_tool_call_count"][tool]    += r["n_calls"]
            run_stats["per_tool_fail_count"][tool]    += r["n_fail"]
        run_stats["per_case"].append({"case_id": cid, "tools": case_results})
        line = f"  [{i:>3}/{len(cases)}] {cid:<55} {msg}"
        for tool in args.tools:
            r = case_results[tool]
            line += f"  {tool[:8]}={r['n_ok']}/{r['n_calls']}({r['elapsed_s']:.1f}s)"
        print(line)

    run_stats["total_elapsed_s"] = round(time.time() - t_run, 1)
    run_stats["finished_at"]     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_yaml(OUT_ROOT / "_run_stats.json", run_stats)

    print(f"\n[done] {len(cases)} cases in {run_stats['total_elapsed_s']:.0f}s")
    for t in args.tools:
        n = run_stats["per_tool_call_count"][t]
        s = run_stats["per_tool_total_seconds"][t]
        f = run_stats["per_tool_fail_count"][t]
        if n:
            print(f"  {t:<22} {n:>4} calls in {s:>6.1f}s  ({s/n:.2f}s avg)  fail={f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
