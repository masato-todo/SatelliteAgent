"""Phase B precompute v2 — eliminate redundant SimSat fetches.

v1 (build_precompute.py) called /api/tool/invoke for every (band, side) and
every (index, side), so SimSat re-ran STAC search + odc.stac.load redundantly.
A single 11-band fetch costs almost the same as a 1-band fetch (latency-dominated),
so this version fetches ALL bands once per side and computes everything else
locally with numpy.

Per case the only network calls are now:
    1. fetch_sentinel_array(BANDS, before_ts) — one HTTP/STAC round-trip
    2. fetch_sentinel_array(BANDS, after_ts)  — one HTTP/STAC round-trip
    3. POST /api/tool/invoke for classify_change (Gemini)

Output schema is identical to v1.

Usage:
    uv run python scripts/build_precompute_v2.py
    uv run python scripts/build_precompute_v2.py --limit 3
    uv run python scripts/build_precompute_v2.py --skip-existing       # default
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from simsat_client.sentinel import fetch_sentinel_array, SimSatError
from tools.spectral import _colormap_signed, _colormap_delta  # reuse v1 PNG colormap


CANONICAL_PATH = PROJ_ROOT / "data" / "canonical_dataset.yaml"
OUT_ROOT       = PROJ_ROOT / "data" / "precompute"
SAT_BASE       = os.environ.get("SAT_BASE", "http://localhost:7860")
SIMSAT_BASE    = os.environ.get("SIMSAT_API_URL", "http://localhost:9005")

BANDS = ["blue", "green", "red",
         "rededge1", "rededge2", "rededge3",
         "nir", "nir08", "nir09",
         "swir16", "swir22"]

# index definition: (a, b) such that index = (a - b) / (a + b)
INDEX_DEFS = {
    "NDVI":  ("nir",    "red"),
    "NDWI":  ("green",  "nir"),
    "MNDWI": ("green",  "swir16"),
    "NBR":   ("nir",    "swir22"),
    "NDBI":  ("swir16", "nir"),
    "NDSI":  ("green",  "swir16"),
}

FALSE_COLOR_COMBOS = [
    ("nir", "red", "green"),       # vegetation
    ("swir22", "nir", "red"),      # burn severity
    ("swir16", "nir", "blue"),     # urban vs vegetation
    ("nir", "swir16", "red"),      # agricultural
    ("red", "green", "blue"),      # true color
]
SIDES = ["before", "after"]


# ---------------- helpers (no /api/tool/invoke) ----------------

def scale_rgb(arr_uint16: np.ndarray) -> np.ndarray:
    """Same convention as SimSat's image_to_png: divide raw reflectance by 3000."""
    return (arr_uint16.astype(np.float32) / 3000 * 255).clip(0, 255).astype(np.uint8)


def colormap_index(idx_arr: np.ndarray) -> np.ndarray:
    """Identical to v1 compute_index PNG: red↔white↔green diverging."""
    return _colormap_signed(idx_arr)


def colormap_delta(idx_arr: np.ndarray, vmax: float = 0.4) -> np.ndarray:
    """Identical to v1 compute_index_delta PNG: red↔white↔blue diverging."""
    return _colormap_delta(idx_arr, vmax=vmax)


def index_array(ds: dict[str, np.ndarray], a: str, b: str) -> np.ndarray:
    a_arr = ds[a].astype(np.float32)
    b_arr = ds[b].astype(np.float32)
    denom = a_arr + b_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (a_arr - b_arr) / denom, np.nan)


def stats_of(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None,
                "frac_decrease_strong": 0.0, "frac_increase_strong": 0.0}
    return {
        "min":    float(finite.min()),
        "max":    float(finite.max()),
        "mean":   float(finite.mean()),
        "median": float(np.median(finite)),
        "frac_decrease_strong": float((arr < -0.2).sum() / arr.size),
        "frac_increase_strong": float((arr >  0.2).sum() / arr.size),
    }


def write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def save_png(arr: np.ndarray, path: Path, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)


# ---------------- per-case work ----------------

def fetch_both_sides(case: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict]:
    """Return (ds_before, ds_after, meta_before, meta_after) where ds_* maps band→array.

    SentinelArray.array is shape (n_bands, H, W) dtype=uint16. We split into a
    dict for ergonomic access by name.
    """
    lat = float(case["lat"]); lon = float(case["lon"])
    size = float(case["size_km"])
    bd = case["request"]["before_date"]
    ad = case["request"]["after_date"]

    sa_b = fetch_sentinel_array(lat=lat, lon=lon, timestamp=bd, bands=BANDS,
                                size_km=size, base_url=SIMSAT_BASE,
                                resolution_meters=10, window_days=30, timeout=300)
    sa_a = fetch_sentinel_array(lat=lat, lon=lon, timestamp=ad, bands=BANDS,
                                size_km=size, base_url=SIMSAT_BASE,
                                resolution_meters=10, window_days=30, timeout=300)
    ds_b = {name: sa_b.array[i] for i, name in enumerate(sa_b.band_names)}
    ds_a = {name: sa_a.array[i] for i, name in enumerate(sa_a.band_names)}
    return ds_b, ds_a, sa_b.metadata, sa_a.metadata


def precompute_case(case: dict, case_dir: Path) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    timing = {}

    # 1. Two STAC fetches (entire band set)
    t0 = time.time()
    ds_b, ds_a, meta_b, meta_a = fetch_both_sides(case)
    timing["fetch"] = round(time.time() - t0, 2)

    # 2. fetch_band — per band per side, save grayscale PNG + stats yaml
    t0 = time.time()
    for side, ds, meta in (("before", ds_b, meta_b), ("after", ds_a, meta_a)):
        for band in BANDS:
            arr = ds[band]
            stub = case_dir / "fetch_band" / f"{band}__{side}"
            save_png(scale_rgb(arr), stub.with_suffix(".png"), mode="L")
            write_yaml(stub.with_suffix(".stats.yaml"), {
                "args": {"band": band, "which": side},
                "response": {
                    "band": band,
                    "datetime":    meta.get("date") if isinstance(meta.get("date"), str) else None,
                    "cloud_cover": meta.get("cloud_cover"),
                    "source":      meta.get("platform"),
                    "stats": {"min": float(arr.min()), "max": float(arr.max()),
                              "mean": float(arr.mean()), "std": float(arr.std())},
                },
            })
    timing["fetch_band"] = round(time.time() - t0, 2)

    # 3. compute_index — pseudocolor PNG + stats yaml per index per side
    t0 = time.time()
    idx_arrays = {"before": {}, "after": {}}
    for idx_name, (a_band, b_band) in INDEX_DEFS.items():
        for side, ds in (("before", ds_b), ("after", ds_a)):
            idx = index_array(ds, a_band, b_band)
            idx_arrays[side][idx_name] = idx
            stub = case_dir / "compute_index" / f"{idx_name}__{side}"
            save_png(colormap_index(idx), stub.with_suffix(".png"), mode="RGB")
            write_yaml(stub.with_suffix(".stats.yaml"), {
                "args": {"index": idx_name, "which": side},
                "response": {"index": idx_name, "stats": stats_of(idx)},
            })
    timing["compute_index"] = round(time.time() - t0, 2)

    # 4. compute_index_delta — after - before for each index
    t0 = time.time()
    for idx_name in INDEX_DEFS:
        delta = idx_arrays["after"][idx_name] - idx_arrays["before"][idx_name]
        stub = case_dir / "compute_index_delta" / idx_name
        save_png(colormap_delta(delta, vmax=0.4), stub.with_suffix(".png"), mode="RGB")
        write_yaml(stub.with_suffix(".stats.yaml"), {
            "args": {"index": idx_name},
            "response": {"index": idx_name, "delta_stats": stats_of(delta)},
        })
    timing["compute_index_delta"] = round(time.time() - t0, 2)

    # 5. false_color — RGB stack of 3 bands per combo per side
    t0 = time.time()
    for combo in FALSE_COLOR_COMBOS:
        tag = "-".join(combo)
        for side, ds in (("before", ds_b), ("after", ds_a)):
            r, g, b = (scale_rgb(ds[combo[i]]) for i in range(3))
            rgb = np.stack([r, g, b], axis=-1)
            stub = case_dir / "false_color" / f"{tag}__{side}"
            save_png(rgb, stub.with_suffix(".png"), mode="RGB")
            write_yaml(stub.with_suffix(".stats.yaml"), {
                "args": {"bands": list(combo), "which": side},
                "response": {"bands": list(combo)},
            })
    timing["false_color"] = round(time.time() - t0, 2)

    # 6. classify_change — still goes through Gemini via /api/tool/invoke
    # (the only call that needs the SatelliteAgent server)
    t0 = time.time()
    bk = case.get("_before_key"); ak = case.get("_after_key")
    if bk and ak:
        body = {"before_key": bk, "after_key": ak,
                "tool_name": "classify_change", "arguments": {}}
        try:
            r = requests.post(f"{SAT_BASE}/api/tool/invoke", json=body, timeout=120)
            obs = (r.json() or {}).get("observation") if r.ok else {"error": f"HTTP {r.status_code}"}
        except requests.RequestException as e:
            obs = {"error": f"{type(e).__name__}: {e}"}
    else:
        obs = {"error": "missing before_key / after_key"}
    write_yaml(case_dir / "classify_change.yaml",
               {"args": {}, "response": obs})
    timing["classify_change"] = round(time.time() - t0, 2)

    timing["total"] = round(sum(v for k, v in timing.items() if k != "total"), 2)
    return timing


def cache_key(lat: float, lon: float, ts: str, size_km: float, res: int = 10) -> str:
    import hashlib
    if "T" not in ts: ts = ts + "T00:00:00Z"
    elif not ts.endswith("Z"): ts = ts + "Z"
    return hashlib.md5(f"{lat:.4f}_{lon:.4f}_{ts}_{size_km}_r{res}".encode()).hexdigest()[:10]


def case_already_done(case_dir: Path) -> bool:
    if not (case_dir / "classify_change.yaml").exists():
        return False
    return sum(1 for _ in (case_dir / "compute_index_delta").glob("*.stats.yaml")) == len(INDEX_DEFS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-id", default=None)
    ap.add_argument("--skip-existing",   dest="skip", action="store_true",  default=True)
    ap.add_argument("--no-skip-existing", dest="skip", action="store_false")
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

    # Precompute the cache_keys so classify_change can reuse the SatelliteAgent
    # /api/tool/invoke endpoint (which needs cached PNGs by key).
    for c in cases:
        c["_before_key"] = cache_key(c["lat"], c["lon"], c["request"]["before_date"], c["size_km"])
        c["_after_key"]  = cache_key(c["lat"], c["lon"], c["request"]["after_date"],  c["size_km"])

    print(f"[init] SAT_BASE={SAT_BASE}  SIMSAT={SIMSAT_BASE}")
    print(f"[init] {len(cases)} cases × multi-band (1 fetch / side) → {OUT_ROOT.relative_to(PROJ_ROOT)}")

    run_stats = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "n_cases": len(cases), "per_case": []}
    t_run = time.time()
    n_ok = n_skip = n_fail = 0

    for i, case in enumerate(cases, 1):
        cid = case["id"]; case_dir = OUT_ROOT / cid
        if args.skip and case_already_done(case_dir):
            print(f"  [{i:>3}/{len(cases)}] {cid:<55} SKIP")
            n_skip += 1
            continue
        try:
            t = precompute_case(case, case_dir)
            print(f"  [{i:>3}/{len(cases)}] {cid:<55} OK   total={t['total']:5.1f}s  "
                  f"fetch={t['fetch']:.1f} band={t['fetch_band']:.1f} idx={t['compute_index']:.1f} "
                  f"delta={t['compute_index_delta']:.1f} fc={t['false_color']:.1f} cls={t['classify_change']:.1f}")
            run_stats["per_case"].append({"case_id": cid, "timing": t})
            n_ok += 1
        except SimSatError as e:
            print(f"  [{i:>3}/{len(cases)}] {cid:<55} FAIL  {type(e).__name__}: {str(e)[:100]}")
            n_fail += 1
        except Exception as e:
            print(f"  [{i:>3}/{len(cases)}] {cid:<55} FAIL  {type(e).__name__}: {str(e)[:100]}")
            n_fail += 1

    run_stats["total_elapsed_s"] = round(time.time() - t_run, 1)
    run_stats["n_ok"]   = n_ok
    run_stats["n_fail"] = n_fail
    run_stats["n_skip"] = n_skip
    write_yaml(OUT_ROOT / "_run_stats.yaml", run_stats)

    print(f"\n[done] {len(cases)} cases in {run_stats['total_elapsed_s']:.0f}s  "
          f"OK={n_ok} FAIL={n_fail} SKIP={n_skip}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
