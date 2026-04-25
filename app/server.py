"""FastAPI server for SatelliteAgent Mission Control.

Endpoints:
    GET  /                       serves index.html
    GET  /static/*               static assets
    GET  /api/templates          returns location templates
    POST /api/fetch              fetches before/after Sentinel-2 images from SimSat
    GET  /api/image/{key}        serves a cached image
    GET  /api/run_agent          SSE stream of ReAct events
    POST /api/tool/invoke        invoke a single tool (Annotate mode)
    POST /api/trace/save         save a human annotation trace as YAML
    GET  /api/traces             list saved human traces
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.react_loop import run_react
from agent.providers import GeminiProvider
from simsat_client import fetch_sentinel_image, SimSatError
from tools.stubs import STUB_TOOLS
from tools.vision import make_zoom_in, make_capture_crop
from tools.spectral import (
    make_fetch_band,
    make_false_color,
    make_compute_index,
    make_compute_index_delta,
)
from tools.scorer import make_get_change_stats
from tools.quality import assess_image_quality_impl, STATS_SCHEMA
from tools.classifier_gemini import make_classify_change as make_classify_change_gemini


def _build_provider():
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiProvider()
        except Exception as e:
            print(f"[startup] Gemini provider disabled: {e}")
    else:
        print("[startup] GOOGLE_API_KEY not set - Run Agent will return an error until set")
    return None


PROVIDER = _build_provider()


APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
# Cache directory is overridable so the same code can write to a network
# mount (SSHFS / NFS / SMB) when training storage lives on a remote server.
# Defaults to repo-local data/scenarios for development.
CACHE_DIR = Path(os.environ.get("SAT_CACHE_DIR", APP_DIR.parent / "data" / "scenarios"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR = Path(os.environ.get("SAT_TRACES_DIR", APP_DIR.parent / "data" / "traces" / "human"))
TRACES_DIR.mkdir(parents=True, exist_ok=True)
print(f"[startup] CACHE_DIR = {CACHE_DIR}")
print(f"[startup] TRACES_DIR = {TRACES_DIR}")


def build_tool_registry(before_path: str, after_path: str,
                        context: dict[str, Any] | None = None) -> dict[str, Callable]:
    """Per-request tool registry shared by agent (ReAct) and human annotation.

    `context` carries (lat, lon, size_km, before_ts, after_ts) so that the
    spectral tools (fetch_band / false_color / compute_index) can fetch fresh
    bands from SimSat. Without context they remain stubbed.
    """
    reg: dict[str, Callable] = {**STUB_TOOLS}
    reg["zoom_in"] = make_zoom_in(before_path, after_path)
    reg["capture_crop"] = make_capture_crop(before_path, after_path)
    # classify_change: Gemini interim until team's LFM2-VL classifier LoRA ships
    reg["classify_change"] = make_classify_change_gemini(before_path, after_path, PROVIDER)
    if context:
        reg["fetch_band"]          = make_fetch_band(**context)
        reg["false_color"]         = make_false_color(**context)
        reg["compute_index"]       = make_compute_index(**context)
        reg["compute_index_delta"] = make_compute_index_delta(**context)
        reg["get_change_stats"]    = make_get_change_stats(**context)
    return reg


LOCATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "flood: Sylhet (BD) 2024": {
        "lat": 24.90, "lon": 91.87,
        "before": "2024-05-15", "after": "2024-08-20", "size_km": 10.0,
        "note": "Monsoon flooding, Bangladesh",
    },
    "wildfire: Park Fire (CA) 2024": {
        "lat": 39.91, "lon": -121.62,
        "before": "2024-07-01", "after": "2024-08-15", "size_km": 20.0,
        "note": "Park Fire, Northern California",
    },
    "deforestation: Mato Grosso (BR)": {
        "lat": -9.50, "lon": -55.00,
        "before": "2023-08-01", "after": "2024-10-01", "size_km": 15.0,
        "note": "Amazon rainforest clearing",
    },
    "volcano: La Palma 2021": {
        "lat": 28.61, "lon": -17.87,
        "before": "2021-08-01", "after": "2021-11-15", "size_km": 8.0,
        "note": "Cumbre Vieja eruption",
    },
    "urban: Dubai expansion": {
        "lat": 25.10, "lon": 55.15,
        "before": "2018-01-01", "after": "2024-01-01", "size_km": 15.0,
        "note": "Coastal urban growth",
    },
    "earthquake: Turkey 2023": {
        "lat": 37.17, "lon": 37.04,
        "before": "2023-01-15", "after": "2023-02-20", "size_km": 10.0,
        "note": "Kahramanmaraş earthquake aftermath",
    },
    "flood: Pakistan 2022": {
        "lat": 26.60, "lon": 68.30,
        "before": "2022-06-01", "after": "2022-09-01", "size_km": 20.0,
        "note": "Sindh province floods",
    },
}


def _normalize_ts(d: str) -> str:
    s = d.strip()
    if "T" not in s:
        s = f"{s}T00:00:00Z"
    elif not s.endswith("Z"):
        s = s + "Z"
    return s


def _cache_key(lat: float, lon: float, ts: str, size_km: float, resolution_meters: int = 10) -> str:
    """Hash includes resolution so 10m and 30m caches don't collide."""
    return hashlib.md5(
        f"{lat:.4f}_{lon:.4f}_{ts}_{size_km}_r{resolution_meters}".encode()
    ).hexdigest()[:10]


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.meta.json"


def _load_meta(key: str) -> dict | None:
    p = _meta_path(key)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_meta(key: str, meta: dict) -> None:
    try:
        with open(_meta_path(key), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass  # non-fatal


def _fetch_one(lat: float, lon: float, ts: str, size_km: float, window_days: int,
               resolution_meters: int = 10) -> tuple[str | None, dict[str, Any]]:
    key = _cache_key(lat, lon, _normalize_ts(ts), size_km, resolution_meters)
    path = CACHE_DIR / f"{key}.png"
    request_info = {"lat": lat, "lon": lon, "ts": _normalize_ts(ts),
                    "size_km": size_km, "resolution_meters": resolution_meters}
    if path.exists() and path.stat().st_size > 0:
        meta = _load_meta(key) or {}
        # Backfill stats if missing OR if stats schema has changed.
        current_stats = meta.get("stats") or {}
        if current_stats.get("_schema") != STATS_SCHEMA:
            stats = assess_image_quality_impl(str(path))
            meta = {**meta, "stats": stats}
            _save_meta(key, {**meta, "request": meta.get("request") or request_info})
        return key, {"cached": True, "path": str(path), **meta}
    # Negative cache (previously queried and no image was available)
    meta_only = _load_meta(key)
    if meta_only is not None and meta_only.get("image_available") is False:
        return None, {"cached": True, **meta_only}
    try:
        result = fetch_sentinel_image(
            lat=lat, lon=lon, timestamp=_normalize_ts(ts),
            size_km=size_km, window_days=window_days,
            resolution_meters=resolution_meters,
        )
        result.image.save(path)
        stats = assess_image_quality_impl(str(path))
        meta_to_save = {**result.metadata, "stats": stats, "request": request_info}
        _save_meta(key, meta_to_save)
        return key, {"cached": False, **result.metadata, "stats": stats}
    except SimSatError as e:
        msg = str(e)
        # Only persist as a negative cache for *deterministic* "no data here"
        # answers. Transient infra failures (timeout, unreachable, empty body)
        # would otherwise poison the cache forever — see incident 2026-04-25.
        msg_low = msg.lower()
        transient = (
            "unreachable" in msg_low
            or "timed out" in msg_low
            or "empty body" in msg_low
            or "5" == msg_low.lstrip().split(":", 1)[0][:1]  # "SimSat 5xx: ..."
        )
        if not transient:
            _save_meta(key, {"image_available": False, "error": msg, "request": request_info})
        return None, {"error": msg, "image_available": False, "transient": transient}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}", "transient": True}


def _context_from_keys(before_key: str, after_key: str) -> dict[str, Any] | None:
    """Reconstruct (lat, lon, size_km, before_ts, after_ts) from cached sidecars.

    Tool factories need these to bind fresh SimSat calls (fetch_band, false_color,
    compute_index). We prefer the stored `request` block; fall back to footprint
    center + image datetime when the sidecar predates the request-params change.
    """
    bm = _load_meta(before_key) or {}
    am = _load_meta(after_key)  or {}

    def _resolve(m):
        req = m.get("request") or {}
        if req:
            return req.get("lat"), req.get("lon"), req.get("size_km"), req.get("ts")
        fp = m.get("footprint")
        if fp and len(fp) == 4:
            lon_c = (fp[0] + fp[2]) / 2.0
            lat_c = (fp[1] + fp[3]) / 2.0
            return lat_c, lon_c, m.get("size_km") or 10.0, m.get("datetime")
        return None, None, None, None

    lat_b, lon_b, sz_b, ts_b = _resolve(bm)
    lat_a, lon_a, sz_a, ts_a = _resolve(am)
    lat = lat_a if lat_a is not None else lat_b
    lon = lon_a if lon_a is not None else lon_b
    size_km = sz_a if sz_a is not None else sz_b
    if lat is None or lon is None or size_km is None or ts_a is None or ts_b is None:
        return None
    return {
        "lat": float(lat), "lon": float(lon),
        "size_km": float(size_km),
        "before_ts": ts_b, "after_ts": ts_a,
    }


class FetchRequest(BaseModel):
    lat: float
    lon: float
    before_date: str
    after_date: str
    size_km: float = Field(default=10.0, ge=1, le=100)
    window_days: int = Field(default=30, ge=1, le=180)
    # Default to native 10m so the cached pair matches the GRPO training env.
    # Cold fetch at 50km/10m takes ~30-60s; subsequent calls hit the cache.
    resolution_meters: int = Field(default=10, ge=10, le=120)


app = FastAPI(title="SatelliteAgent")


DM3_CSV = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "disaster_m3_image_metadata.csv"
DM3_XBD_CSV = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "disaster_m3_xbd_image_summary.csv"
DM3_XBD_BUILDINGS_CSV = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "disaster_m3_xbd_buildings.csv"

DM3_TYPE_MAP = {
    "flooding":   "flood",
    "tsunami":    "flood",
    "hurricane":  "flood",
    "fire":       "fire",
    "wildfire":   "fire",
    "volcano":    "volcanic_activity",
    "earthquake": "earthquake_damage",
    "wind":       "earthquake_damage",
    "conflict":   "earthquake_damage",
    "explosion":  "earthquake_damage",
    "landslide":  "earthquake_damage",
}


# Real-world disaster active periods. Used to show users the actual event
# window (vs the xBD Maxar capture date which can be days after).
EVENT_PERIODS: dict[str, tuple[str, str, str]] = {
    # event_name → (start, end, common_name)
    "guatemala_volcano":   ("2018-06-03", "2018-08-19", "Volcán de Fuego eruption"),
    "hurricane_florence":  ("2018-09-13", "2018-09-18", "Hurricane Florence landfall + flooding"),
    "hurricane_harvey":    ("2017-08-25", "2017-09-02", "Hurricane Harvey landfall + Houston flooding"),
    "hurricane_michael":   ("2018-10-10", "2018-10-12", "Hurricane Michael landfall (Cat 5)"),
    "hurricane_matthew":   ("2016-10-04", "2016-10-09", "Hurricane Matthew Caribbean+SE US"),
    "mexico_earthquake":   ("2017-09-19", "2017-09-19", "Puebla earthquake M7.1"),
    "midwest_flooding":    ("2019-03-13", "2019-06-30", "Missouri/Arkansas River basin flooding"),
    "palu_tsunami":        ("2018-09-28", "2018-09-28", "Sulawesi M7.5 earthquake + tsunami + liquefaction"),
    "santa_rosa_wildfire": ("2017-10-08", "2017-10-31", "Tubbs Fire (Northern California)"),
    "socal_fire":          ("2018-11-08", "2018-11-25", "Camp + Woolsey Fires"),
}


def _parse_iso_utc(ts: str):
    from datetime import datetime, timezone as tz
    if "T" in ts:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(ts + "T00:00:00+00:00")
    return dt.astimezone(tz.utc)


def _load_xbd_precise_cases(top_per_event: int = 5) -> tuple[list[dict[str, Any]], set[str]]:
    """Load per-image xBD cases with precise centroids.

    Picks the top N post-disaster images per event ranked by
    (n_destroyed + n_major_damage), pairs each with the matching
    _pre_disaster.png to get the real Before capture date.

    Returns (cases, covered_events) so the caller can skip event-level
    xBD rows in the original DM3 CSV.
    """
    import csv
    from datetime import timedelta

    # Events whose Before/After dates fall outside SimSat (Element84 S2 L2A)
    # availability — verified 2026-04-24 with 30/60/120-day windows:
    #   hurricane_matthew:  pre=2013-01-05 is before Sentinel-2A launch (2015-06)
    #   mexico_earthquake:  pre=2017-01-04 has no L2A coverage for ~120 days
    SIMSAT_UNAVAILABLE = {"hurricane_matthew", "mexico_earthquake"}

    if not DM3_XBD_CSV.exists():
        return [], set()
    with open(DM3_XBD_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    idx: dict[str, dict] = {r["image"]: r for r in rows}
    by_event: dict[str, list[tuple[int, dict]]] = {}
    for r in rows:
        if "post_disaster" not in r["image"]:
            continue
        if r["event"] in SIMSAT_UNAVAILABLE:
            continue
        try:
            dmg = int(r["n_destroyed"]) + int(r["n_major_damage"])
        except (ValueError, KeyError):
            continue
        by_event.setdefault(r["event"], []).append((dmg, r))

    cases: list[dict[str, Any]] = []
    covered: set[str] = set()
    for event, items in by_event.items():
        items.sort(key=lambda x: -x[0])
        picks = [r for dmg, r in items[:top_per_event] if dmg > 0]
        if not picks:
            continue
        covered.add(event)
        for r in picks:
            pre_name = r["image"].replace("_post_disaster.", "_pre_disaster.")
            pre = idx.get(pre_name)
            try:
                post_dt = _parse_iso_utc(r["capture_date"])
            except Exception:
                continue
            if pre:
                try:
                    pre_dt = _parse_iso_utc(pre["capture_date"])
                except Exception:
                    pre_dt = None
            else:
                pre_dt = None
            try:
                lat = float(r["center_lat"]); lon = float(r["center_lon"])
            except (ValueError, KeyError):
                continue
            mapped = DM3_TYPE_MAP.get(r["disaster_type"], "earthquake_damage")
            n_destroyed = int(r["n_destroyed"])
            n_major = int(r["n_major_damage"])
            n_minor = int(r["n_minor_damage"])
            n_none = int(r["n_no_damage"])
            total = int(r["n_buildings"])
            image_id = r["image"].split("_post_")[0].replace(f"{event}_", "", 1)
            # Push After 14 days past xBD's Maxar capture date so SimSat's
            # backward-only window safely picks up a post-disaster S2 pass.
            # xBD post_disaster is 2-19 days after the actual event; +14d more
            # also lets smoke/clouds clear so the change becomes visually crisp.
            after_request_dt = post_dt + timedelta(days=14)
            period = EVENT_PERIODS.get(event)
            cases.append({
                "id": f"xbd_{event}_{image_id}",
                "source": "xBD",
                "event": event,
                "disaster_type": r["disaster_type"],
                "mapped_class": mapped,
                "capture_date": post_dt.strftime("%Y-%m-%d"),
                "after_date":   after_request_dt.strftime("%Y-%m-%d"),
                "before_date":  (pre_dt or (post_dt - timedelta(days=180))).strftime("%Y-%m-%d"),
                "lat": lat,
                "lon": lon,
                "location": event.replace("_", " "),
                "image": r["image"],
                "size_km": 50.0,
                "precise": True,
                "event_start": period[0] if period else None,
                "event_end":   period[1] if period else None,
                "event_name":  period[2] if period else None,
                "damage": {
                    "destroyed": n_destroyed,
                    "major":     n_major,
                    "minor":     n_minor,
                    "no_damage": n_none,
                    "total":     total,
                },
            })
    return cases, covered


def _load_dm3_cases() -> list[dict[str, Any]]:
    """Sample DisasterM3 CSV: one row per (source, event) to maximize variety.

    Events already covered by precise xBD cases are skipped — those are
    loaded separately by _load_xbd_precise_cases.
    """
    import csv
    import random
    from datetime import timedelta

    xbd_cases, xbd_covered = _load_xbd_precise_cases()

    if not DM3_CSV.exists():
        return list(xbd_cases)
    with open(DM3_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_event: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["source"].lower() == "xbd" and r["event"] in xbd_covered:
            continue
        key = (r["source"], r["event"])
        by_event.setdefault(key, []).append(r)

    rng = random.Random(42)
    picks = [rng.choice(v) for v in by_event.values()]

    cases: list[dict[str, Any]] = list(xbd_cases)
    for r in picks:
        try:
            dt = _parse_iso_utc(r["capture_date"])
        except Exception:
            continue
        after_date  = dt.strftime("%Y-%m-%d")
        before_date = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
        mapped = DM3_TYPE_MAP.get(r["disaster_type"], "earthquake_damage")
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except ValueError:
            continue
        period = EVENT_PERIODS.get(r["event"])
        cases.append({
            "id": f"{r['source'].lower()}_{r['event']}",
            "source": r["source"],
            "event": r["event"],
            "disaster_type": r["disaster_type"],
            "mapped_class": mapped,
            "capture_date": after_date,
            "before_date": before_date,
            "after_date":  after_date,
            "lat": lat,
            "lon": lon,
            "location": r["location"],
            "image": r["image"],
            "size_km": 50.0,
            "precise": False,
            "event_start": period[0] if period else None,
            "event_end":   period[1] if period else None,
            "event_name":  period[2] if period else None,
        })
    cases.sort(key=lambda c: (0 if c.get("precise") else 1, c["source"], c["disaster_type"], c["event"], c.get("image", "")))
    return cases


def _load_negative_cases() -> list[dict[str, Any]]:
    """Load negative scenarios (drop is the expected action) from a hand- or
    script-curated YAML. File is optional — runs no-op if absent.
    """
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "negative_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load negative_cases.yaml: {e}")
        return []
    raw_cases = doc.get("cases", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for r in raw_cases:
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        size_km = float(r.get("size_km", 50.0))
        before_date = r["before_date"]
        after_date  = r["after_date"]
        out.append({
            "id":              r["id"],
            "source":          "Negative",
            "event":           r.get("parent_event") or r["id"],
            "disaster_type":   r.get("negative_type", "no_change"),  # pre_pre / post_post / cloud_blocked / random
            "mapped_class":    "no_change",
            "expected_action": r.get("expected_action", "drop"),
            "negative_type":   r.get("negative_type"),
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        r.get("note", ""),
            "image":           "",
            "size_km":         size_km,
            "precise":         False,  # not from xBD per-image
            "is_negative":     True,
        })
    return out


DM3_CASES: list[dict[str, Any]] = _load_dm3_cases() + _load_negative_cases()
_n_pos = sum(1 for c in DM3_CASES if not c.get("is_negative"))
_n_neg = sum(1 for c in DM3_CASES if c.get("is_negative"))
print(f"[startup] DisasterM3 cases loaded: {len(DM3_CASES)} (positive/neutral={_n_pos}, negative={_n_neg})")


# ---- xBD per-building damage polygons (for After/Before image overlay) ----

DAMAGE_SUBTYPES = ("destroyed", "major-damage", "minor-damage")


def _load_xbd_buildings_by_event() -> dict[str, list[dict[str, Any]]]:
    """Load xBD buildings CSV, index by event, keep only rows we may overlay.

    Returns a dict mapping event -> list of rows with keys:
      {subtype, centroid_lat, centroid_lon, polygon_wkt}

    WKT is NOT parsed here (would explode memory and startup time).
    We parse lazily per-request after the centroid AOI filter.
    """
    import csv
    by_event: dict[str, list[dict[str, Any]]] = {}
    if not DM3_XBD_BUILDINGS_CSV.exists():
        return by_event
    with open(DM3_XBD_BUILDINGS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subtype") not in DAMAGE_SUBTYPES:
                continue
            try:
                lat = float(r["centroid_lat"]); lon = float(r["centroid_lon"])
            except (ValueError, KeyError):
                continue
            by_event.setdefault(r["event"], []).append({
                "subtype": r["subtype"],
                "centroid_lat": lat,
                "centroid_lon": lon,
                "polygon_wkt": r["polygon_wkt"],
            })
    return by_event


XBD_BUILDINGS_BY_EVENT: dict[str, list[dict[str, Any]]] = _load_xbd_buildings_by_event()
print(f"[startup] xBD damaged buildings loaded: "
      f"{sum(len(v) for v in XBD_BUILDINGS_BY_EVENT.values()):,} "
      f"across {len(XBD_BUILDINGS_BY_EVENT)} events")


def _parse_wkt_polygon(wkt: str) -> list[tuple[float, float]] | None:
    """Parse 'POLYGON ((lon lat, lon lat, ...))' → list of (lat, lon) points.

    Returns None if parsing fails. Inner rings (holes) are discarded —
    we only use the outer ring for rendering.
    """
    try:
        start = wkt.index("((") + 2
        end = wkt.index("))", start)
        inner = wkt[start:end]
        # Outer ring is before the first '),(' if any
        outer = inner.split("),(")[0]
        pts: list[tuple[float, float]] = []
        for coord in outer.split(","):
            coord = coord.strip()
            if not coord:
                continue
            lon_s, lat_s = coord.split()[:2]
            pts.append((float(lat_s), float(lon_s)))
        return pts if len(pts) >= 3 else None
    except (ValueError, IndexError):
        return None


@app.get("/api/disasterm3/cases")
def api_dm3_cases() -> dict[str, Any]:
    return {"count": len(DM3_CASES), "cases": DM3_CASES}


@app.get("/api/xbd/damage_overlay")
def api_xbd_damage_overlay(
    event: str,
    lat: float,
    lon: float,
    size_km: float,
) -> dict[str, Any]:
    """Return damaged-building polygons (WGS84) that fall inside the AOI.

    Frontend converts WGS84 → Leaflet CRS.Simple pixel coords using the
    AOI center + size_km + image dimensions it already knows.
    """
    import math

    rows = XBD_BUILDINGS_BY_EVENT.get(event, [])
    if not rows:
        return {"event": event, "polygons": [], "counts": {}, "note": "no xBD buildings for this event"}

    half_km = size_km / 2.0
    deg_per_km_lat = 1.0 / 110.574
    deg_per_km_lon = 1.0 / (111.320 * max(math.cos(math.radians(lat)), 1e-6))
    half_lat = half_km * deg_per_km_lat
    half_lon = half_km * deg_per_km_lon
    lat_min, lat_max = lat - half_lat, lat + half_lat
    lon_min, lon_max = lon - half_lon, lon + half_lon

    polygons: list[dict[str, Any]] = []
    counts = {"destroyed": 0, "major-damage": 0, "minor-damage": 0}
    for r in rows:
        if not (lat_min <= r["centroid_lat"] <= lat_max):
            continue
        if not (lon_min <= r["centroid_lon"] <= lon_max):
            continue
        pts = _parse_wkt_polygon(r["polygon_wkt"])
        if pts is None:
            continue
        polygons.append({
            "subtype": r["subtype"],
            "points": pts,  # [(lat, lon), ...]
        })
        counts[r["subtype"]] = counts.get(r["subtype"], 0) + 1

    return {
        "event": event,
        "aoi": {"lat": lat, "lon": lon, "size_km": size_km},
        "polygons": polygons,
        "counts": counts,
    }


@app.get("/api/templates")
def api_templates() -> dict[str, Any]:
    base = os.environ.get("SIMSAT_API_URL", "http://localhost:9005")
    if PROVIDER is not None:
        provider_info = {"kind": "gemini", "model": getattr(PROVIDER, "model", "?")}
    else:
        provider_info = {"kind": "none", "model": "no API key"}
    return {
        "simsat_url": base,
        "templates": LOCATION_TEMPLATES,
        "provider": provider_info,
    }


class BeforeCandidatesRequest(BaseModel):
    lat: float
    lon: float
    after_date: str
    # When set (typically the event_start of the selected DM3 case), candidates
    # are computed as anchor_date - offset_days. Otherwise falls back to after_date.
    anchor_date: str | None = None
    size_km: float = Field(default=10.0, ge=1, le=100)
    # Offsets < 11d would risk SimSat returning the same S2 scene as the
    # current After (10-day backward-only window → overlapping hits).
    offsets_days: list[int] = Field(default_factory=lambda: [14, 30, 60, 90, 180, 365])
    window_days: int = Field(default=10, ge=1, le=60)


def _parse_date_utc(s: str):
    if "T" in s:
        ref = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        ref = datetime.fromisoformat(f"{s}T00:00:00+00:00")
    return ref.astimezone(timezone.utc)


@app.post("/api/before_candidates")
def api_before_candidates(req: BeforeCandidatesRequest) -> dict[str, Any]:
    """Probe multiple past dates near `anchor_date` (or `after_date`) and return
    cloud-cover/metadata for each.

    When the caller passes anchor_date (typically the disaster's event_start),
    every candidate lands safely pre-disaster regardless of the current After
    value. Caches both hits and misses as metadata sidecars.
    """
    ref_str = req.anchor_date or req.after_date
    try:
        ref = _parse_date_utc(ref_str)
    except ValueError as e:
        raise HTTPException(400, f"invalid date: {e}")

    def probe(offset_days: int) -> dict[str, Any]:
        target = (ref - timedelta(days=offset_days)).strftime("%Y-%m-%d")
        key, meta = _fetch_one(req.lat, req.lon, target, req.size_km, req.window_days)
        return {
            "target_date": target,
            "offset_days": offset_days,
            "key": key,
            "meta": meta,
        }

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(probe, req.offsets_days))

    return {"candidates": results, "anchor_date": ref.strftime("%Y-%m-%d")}


class AfterCandidatesRequest(BaseModel):
    lat: float
    lon: float
    after_date: str  # current After date — used as the starting point if no anchor
    # When set (typically the event_end of the selected DM3 case), candidates
    # are computed as anchor_date + offset_days. Otherwise falls back to after_date.
    anchor_date: str | None = None
    size_km: float = Field(default=10.0, ge=1, le=100)
    # +0 = at event end (peak flood for short-lived disasters);
    # +14, +30 = post-event with cleaner skies (better for fires/burn scars).
    offsets_days: list[int] = Field(default_factory=lambda: [0, 3, 7, 14, 21, 30, 45])
    window_days: int = Field(default=10, ge=1, le=60)


@app.post("/api/after_candidates")
def api_after_candidates(req: AfterCandidatesRequest) -> dict[str, Any]:
    """Probe future dates after `anchor_date` (or `after_date`) for clearer imagery.

    When the caller passes anchor_date (typically the disaster's event_end),
    candidates span the active disaster period through ~6 weeks post-event
    so the user can find the right balance of "during/after disaster" + "clear sky".
    """
    ref_str = req.anchor_date or req.after_date
    try:
        ref = _parse_date_utc(ref_str)
    except ValueError as e:
        raise HTTPException(400, f"invalid date: {e}")

    def probe(offset_days: int) -> dict[str, Any]:
        target = (ref + timedelta(days=offset_days)).strftime("%Y-%m-%d")
        key, meta = _fetch_one(req.lat, req.lon, target, req.size_km, req.window_days)
        return {
            "target_date": target,
            "offset_days": offset_days,
            "key": key,
            "meta": meta,
        }

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(probe, req.offsets_days))

    return {"candidates": results, "anchor_date": ref.strftime("%Y-%m-%d")}


import requests as _requests  # lazy local alias to avoid shadowing

@app.get("/api/geocode")
def api_geocode(q: str) -> dict[str, Any]:
    """Forward a location query to OpenStreetMap Nominatim.

    Free + no API key, but rate-limited (1 req/s per IP). The Nominatim
    ToS requires a descriptive User-Agent; we send one here.
    """
    q = (q or "").strip()
    if len(q) < 2:
        return {"results": []}
    try:
        r = _requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 6, "addressdetails": 0},
            headers={"User-Agent": "SatelliteAgent/0.1 (hackathon, no commercial use)"},
            timeout=6,
        )
        if r.status_code != 200:
            return {"results": [], "error": f"Nominatim HTTP {r.status_code}"}
        items = r.json()
        return {
            "results": [
                {
                    "display_name": x.get("display_name"),
                    "lat": float(x.get("lat", 0.0)),
                    "lon": float(x.get("lon", 0.0)),
                    "type": x.get("type"),
                    "class": x.get("class"),
                    "importance": x.get("importance"),
                }
                for x in items
            ]
        }
    except _requests.RequestException as e:
        return {"results": [], "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"results": [], "error": f"{type(e).__name__}: {e}"}


@app.post("/api/fetch")
def api_fetch(req: FetchRequest) -> dict[str, Any]:
    b_key, b_meta = _fetch_one(req.lat, req.lon, req.before_date, req.size_km,
                                req.window_days, req.resolution_meters)
    a_key, a_meta = _fetch_one(req.lat, req.lon, req.after_date,  req.size_km,
                                req.window_days, req.resolution_meters)
    return {
        "request": req.model_dump(),
        "before": {"key": b_key, "meta": b_meta, "date": req.before_date},
        "after":  {"key": a_key, "meta": a_meta, "date": req.after_date},
    }


@app.get("/api/image/{key}")
def api_image(key: str):
    if not key.replace("_", "").isalnum():
        raise HTTPException(400, "invalid key")
    path = CACHE_DIR / f"{key}.png"
    if not path.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/run_agent")
def api_run_agent(before_key: str, after_key: str):
    before_path = CACHE_DIR / f"{before_key}.png"
    after_path  = CACHE_DIR / f"{after_key}.png"
    if not before_path.exists() or not after_path.exists():
        raise HTTPException(400, "fetch images first")

    context = _context_from_keys(before_key, after_key)
    tool_registry = build_tool_registry(str(before_path), str(after_path), context)

    def generate():
        try:
            for event in run_react(str(before_path), str(after_path),
                                   provider=PROVIDER, tool_registry=tool_registry):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'agent crashed: {type(e).__name__}: {e}'})}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---- Human annotation endpoints ----------------------------------------

class ToolInvokeRequest(BaseModel):
    before_key: str
    after_key: str
    tool_name: str
    arguments: dict[str, Any] = {}


@app.post("/api/tool/invoke")
def api_tool_invoke(req: ToolInvokeRequest) -> dict[str, Any]:
    before_path = CACHE_DIR / f"{req.before_key}.png"
    after_path  = CACHE_DIR / f"{req.after_key}.png"
    if not before_path.exists() or not after_path.exists():
        raise HTTPException(400, "images not found; fetch first")

    context = _context_from_keys(req.before_key, req.after_key)
    registry = build_tool_registry(str(before_path), str(after_path), context)
    if req.tool_name not in registry:
        raise HTTPException(400, f"unknown tool: {req.tool_name}")

    try:
        result = registry[req.tool_name](**req.arguments)
    except TypeError as e:
        raise HTTPException(400, f"argument mismatch for {req.tool_name}: {e}")
    except Exception as e:
        return {"observation": {"error": f"{type(e).__name__}: {e}"}}

    return {"observation": result}


class TraceSaveRequest(BaseModel):
    metadata: dict[str, Any]
    events: list[dict[str, Any]]
    final: dict[str, Any]


def _safe_slug(s: str, default: str = "unknown") -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in s.strip())
    return cleaned or default


@app.post("/api/trace/save")
def api_trace_save(req: TraceSaveRequest) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenario = _safe_slug(str(req.metadata.get("scenario_id", "unknown")))
    profile = _safe_slug(str(req.metadata.get("profile", "default")))
    annotator = _safe_slug(str(req.metadata.get("annotator", "human")))

    filename = f"{scenario}__{profile}__{annotator}__{ts}.yaml"
    path = TRACES_DIR / filename

    metadata = dict(req.metadata)
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    doc = {
        "metadata": metadata,
        "events": req.events,
        "final": req.final,
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)

    rel = path.relative_to(APP_DIR.parent)
    return {"saved_path": str(rel), "filename": filename}


@app.get("/api/traces")
def api_list_traces() -> dict[str, Any]:
    files = sorted(TRACES_DIR.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:200]:
        try:
            with open(f, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            final = doc.get("final", {}) if isinstance(doc, dict) else {}
            events = doc.get("events", []) if isinstance(doc, dict) else []
            out.append({
                "filename": f.name,
                "scenario_id": meta.get("scenario_id", "?"),
                "profile": meta.get("profile", "?"),
                "annotator": meta.get("annotator", "?"),
                "created_at": meta.get("created_at", ""),
                "final_action": final.get("action", "?"),
                "final_change_type": final.get("change_type"),
                "n_events": len(events),
                "size_bytes": f.stat().st_size,
            })
        except Exception as e:
            out.append({"filename": f.name, "error": str(e)})
    return {"traces": out}


def _resolve_trace_path(filename: str) -> Path:
    # Reject anything that could escape TRACES_DIR.
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".yaml"):
        raise HTTPException(400, f"invalid filename: {filename}")
    p = TRACES_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"trace not found: {filename}")
    return p


@app.get("/api/traces/{filename}")
def api_get_trace(filename: str) -> dict[str, Any]:
    p = _resolve_trace_path(filename)
    with open(p, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return {"filename": filename, "doc": doc}


@app.delete("/api/traces/{filename}")
def api_delete_trace(filename: str) -> dict[str, Any]:
    p = _resolve_trace_path(filename)
    p.unlink()
    return {"deleted": filename}


# ---- Static files & index -----------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
