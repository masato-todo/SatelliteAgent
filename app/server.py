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

import requests as _requests
import yaml
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.react_loop import run_react
from agent.react_loop_openai import run_react_openai
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
from tools.wildfire import make_detect_wildfire
from tools.scorer import make_get_change_stats
from tools.quality import assess_image_quality_impl, STATS_SCHEMA
from tools.region import make_get_region_info
from tools.classifier_gemini import make_classify_change as make_classify_change_gemini
from tools.classifier_openai import make_classify_change as make_classify_change_openai


def _build_provider():
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiProvider()
        except Exception as e:
            print(f"[startup] Gemini provider disabled: {e}")
    else:
        print("[startup] GOOGLE_API_KEY not set - Gemini path disabled (local vLLM provider is used by default)")
    return None


PROVIDER = _build_provider()


def _load_providers_config(app_dir: Path) -> list[dict]:
    path = app_dir.parent / "config" / "providers.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return doc.get("providers") or []
    except Exception as e:
        print(f"[startup] WARN: failed to load providers.yaml: {e}")
        return []


def _resolve_provider_cfg(provider_name: str | None) -> dict:
    """Pick the provider config for this request.

    Resolution order:
      1. caller passed a `provider_name` -> must match an entry by name,
         otherwise HTTP 400 (no silent substitution — debugging nightmare).
      2. caller passed nothing -> the entry whose `default: true` is set.
      3. no `default: true` anywhere -> first configured entry (deterministic).

    Returns the cfg dict; never returns None.
    """
    if not PROVIDERS_CFG:
        raise HTTPException(503, "no VLM providers configured (config/providers.yaml)")
    if provider_name:
        cfg = next((p for p in PROVIDERS_CFG if p.get("name") == provider_name), None)
        if cfg is None:
            available = [p.get("name") for p in PROVIDERS_CFG]
            raise HTTPException(
                400,
                f"unknown provider '{provider_name}'. available: {available}",
            )
        return cfg
    cfg = next((p for p in PROVIDERS_CFG if p.get("default") is True), None)
    if cfg is not None:
        return cfg
    return PROVIDERS_CFG[0]


def _make_classify_change(before_path: str, after_path: str,
                          provider_name: str | None, model: str | None) -> Callable:
    """Resolve provider config + return classify_change callable.

    Raises HTTPException with a precise status code so the UI/SSE surfaces
    misconfiguration instead of silently swapping providers.
    """
    cfg = _resolve_provider_cfg(provider_name)
    kind = cfg.get("kind")
    chosen_model = model or cfg.get("default_model")
    if kind == "gemini":
        if PROVIDER is None:
            raise HTTPException(
                503,
                "Gemini provider selected but GOOGLE_API_KEY/GEMINI_API_KEY is not set",
            )
        bound = type(PROVIDER)(model=chosen_model) if chosen_model else PROVIDER
        return make_classify_change_gemini(before_path, after_path, bound)
    if kind == "openai_compat":
        api_key_env = cfg.get("api_key_env")
        api_key = os.environ.get(api_key_env, "dummy") if api_key_env else "dummy"
        return make_classify_change_openai(
            before_path, after_path,
            base_url=cfg["base_url"], model=chosen_model, api_key=api_key,
        )
    if kind == "lfm2_multiturn":
        # The 450M sft-grpo agent emits Python-style tool calls, not OpenAI
        # tool_calls JSON, and is invoked via /api/run_agent's lfm2_multiturn
        # branch. classify_change as a standalone tool is not supported here —
        # lazy-fail so build_tool_registry doesn't blow up.
        def _unsupported(**_kwargs):
            return {"error": (
                f"classify_change is not supported with provider "
                f"'{cfg.get('name')}' (kind=lfm2_multiturn). "
                f"Switch to a Gemini or openai_compat provider in Settings."
            )}
        return _unsupported
    raise HTTPException(500, f"provider '{cfg.get('name')}' has unknown kind '{kind}'")


APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
# Cache directory is overridable so the same code can write to a network
# mount (SSHFS / NFS / SMB) when training storage lives on a remote server.
# Defaults to repo-local data/scenarios for development.
CACHE_DIR = Path(os.environ.get("SAT_CACHE_DIR", APP_DIR.parent / "data" / "scenarios"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR = Path(os.environ.get("SAT_DERIVED_DIR", APP_DIR.parent / "data" / "derived"))
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR = Path(os.environ.get("SAT_TRACES_DIR", APP_DIR.parent / "data" / "traces" / "human"))
TRACES_DIR.mkdir(parents=True, exist_ok=True)
print(f"[startup] CACHE_DIR = {CACHE_DIR}")
print(f"[startup] TRACES_DIR = {TRACES_DIR}")

PROVIDERS_CFG = _load_providers_config(APP_DIR)
print(f"[startup] VLM providers: {[p.get('name') for p in PROVIDERS_CFG]}")


def build_tool_registry(before_path: str, after_path: str,
                        context: dict[str, Any] | None = None,
                        provider_name: str | None = None,
                        model: str | None = None) -> dict[str, Callable]:
    """Per-request tool registry shared by agent (ReAct) and human annotation.

    `context` carries (lat, lon, size_km, before_ts, after_ts) so that the
    spectral tools (fetch_band / false_color / compute_index) can fetch fresh
    bands from SimSat. Without context they remain stubbed.
    """
    reg: dict[str, Callable] = {**STUB_TOOLS}
    reg["zoom_in"] = make_zoom_in(before_path, after_path)
    reg["capture_crop"] = make_capture_crop(before_path, after_path)
    reg["classify_change"] = _make_classify_change(before_path, after_path, provider_name, model)
    if context:
        # Strip non-SimSat-fetch keys before splatting into spectral factories.
        # - region_info: reverse-geocoded sidecar (feat/toolcall, anti-fabrication)
        # - before_actual_dt/after_actual_dt: SimSat-returned STAC datetimes
        #   (Branch/refactor, used by detect_wildfire eval-parity binding)
        ctx = dict(context)
        region_payload  = ctx.pop("region_info", None)
        before_actual   = ctx.pop("before_actual_dt", None)
        after_actual    = ctx.pop("after_actual_dt",  None)
        reg["fetch_band"]          = make_fetch_band(**ctx)
        reg["false_color"]         = make_false_color(**ctx)
        reg["compute_index"]       = make_compute_index(**ctx)
        reg["compute_index_delta"] = make_compute_index_delta(**ctx)
        reg["get_change_stats"]    = make_get_change_stats(**ctx)
        reg["get_region_info"]     = make_get_region_info(
            lat=ctx["lat"], lon=ctx["lon"], region_payload=region_payload,
        )
        # detect_wildfire pins the *actual* SimSat-returned STAC item (= the
        # scene the user sees on the map) so the LFM input matches what eval
        # at scripts/eval_wildfire_hf_simsat.py --use-sentinel-datetime sends.
        reg["detect_wildfire"] = make_detect_wildfire(
            lat=ctx["lat"], lon=ctx["lon"], size_km=ctx["size_km"],
            before_ts=before_actual or ctx["before_ts"],
            after_ts=after_actual  or ctx["after_ts"],
        )
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


def _reverse_geocode(lat: float, lon: float) -> dict[str, Any] | None:
    """Resolve (lat, lon) to a region descriptor via OSM Nominatim /reverse.

    Best-effort: returns None on any failure. Used to populate the cache
    sidecar at fetch time so the agent's `get_region_info` tool returns
    real data instead of a stub fixture.

    HACK: Direct call to public Nominatim with hardcoded UA. No caching
    or rate limiting (Nominatim's policy is 1 req/s, sustained use must
    self-host or use a paid provider).
    TODO: cache by (round(lat,3), round(lon,3)) to avoid redundant calls
    when the same scene is fetched repeatedly.
    """
    try:
        r = _requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10,
                    "addressdetails": 1},
            headers={"User-Agent": "SatelliteAgent/0.1 (hackathon, no commercial use)"},
            timeout=6,
        )
        if r.status_code != 200:
            return None
        data = r.json() or {}
        addr = data.get("address") or {}
        country = addr.get("country")
        country_code = (addr.get("country_code") or "").upper() or None
        # Nominatim returns various keys depending on density; pick the most
        # specific human-meaningful place name available.
        region = (
            addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("municipality") or addr.get("county")
            or addr.get("state_district") or addr.get("state")
            or addr.get("region")
        )
        return {
            "display_name": data.get("display_name"),
            "region":       region,
            "state":        addr.get("state"),
            "country":      country,
            "country_code": country_code,
        }
    except Exception:
        return None


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
        dirty = False
        if current_stats.get("_schema") != STATS_SCHEMA:
            stats = assess_image_quality_impl(str(path))
            meta = {**meta, "stats": stats}
            dirty = True
        # Backfill region info for sidecars written before reverse-geocode
        # was wired in.
        if "region_info" not in meta:
            meta = {**meta, "region_info": _reverse_geocode(lat, lon)}
            dirty = True
        if dirty:
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
        region_info = _reverse_geocode(lat, lon)
        meta_to_save = {**result.metadata, "stats": stats, "request": request_info,
                        "region_info": region_info}
        _save_meta(key, meta_to_save)
        return key, {"cached": False, **result.metadata, "stats": stats,
                     "region_info": region_info}
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

    Also returns `before_actual_dt`/`after_actual_dt` — the SimSat-returned STAC
    item datetimes — so detect_wildfire can pin the *exact same* scene the user
    sees on the map (request.ts vs SimSat-picked item can differ; see
    eval_wildfire_hf_simsat.py for why this matters for matching the LoRA).
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
    # Region info was attached at fetch time; prefer the after-side sidecar.
    region_info = am.get("region_info") or bm.get("region_info")
    return {
        "lat": float(lat), "lon": float(lon),
        "size_km": float(size_km),
        "before_ts": ts_b, "after_ts": ts_a,
        "before_actual_dt": bm.get("datetime") or ts_b,
        "after_actual_dt":  am.get("datetime") or ts_a,
        "region_info": region_info,
    }


class FetchRequest(BaseModel):
    lat: float
    lon: float
    before_date: str
    after_date: str
    size_km: float = Field(default=10.0, ge=1, le=100)
    window_days: int = Field(default=30, ge=1, le=180)
    # Optional per-side override. Used by the FireEdge fetch path: tight
    # window=1 on After (pin the training STAC item via sentinel_datetime),
    # wider window on Before so sdt-180d can fall on a real S2 capture.
    before_window_days: int | None = Field(default=None, ge=1, le=180)
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


def _load_ems_cases() -> list[dict[str, Any]]:
    """Load Copernicus EMS Rapid Mapping cases (positive, expected_action=submit_to_ground).
    Schema mirrors `_load_negative_cases`. File is optional.
    """
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "ems_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load ems_cases.yaml: {e}")
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
        countries = r.get("countries") or []
        loc = ", ".join(countries) if countries else (r.get("name") or "")
        out.append({
            "id":              r["id"],
            "source":          "EMS",
            "event":           r.get("ems_code") or r["id"],
            "disaster_type":   r.get("event_type", "other"),
            "mapped_class":    r.get("event_type", "other"),
            "expected_action": r.get("expected_action", "submit_to_ground"),
            "ems_code":        r.get("ems_code"),
            "ems_category":    r.get("category"),
            "name":            r.get("name"),
            "countries":       countries,
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        loc,
            "image":           "",
            "size_km":         size_km,
            "precise":         False,
            "is_ems":          True,
        })
    return out


def _load_volcanic_cases() -> list[dict[str, Any]]:
    """Load GDACS volcanic event cases (positive, expected_action=submit_to_ground)."""
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "volcanic_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load volcanic_cases.yaml: {e}")
        return []
    raw_cases = doc.get("cases", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for r in raw_cases:
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        size_km = float(r.get("size_km", 10.0))
        before_date = r["before_date"]
        after_date  = r["after_date"]
        loc = r.get("country") or r.get("name") or ""
        out.append({
            "id":              r["id"],
            "source":          "GDACS_VO",
            "event":           r.get("name") or r["id"],
            "disaster_type":   "volcanic",
            "mapped_class":    "volcanic",
            "expected_action": r.get("expected_action", "submit_to_ground"),
            "alertlevel":      r.get("alertlevel"),
            "name":            r.get("name"),
            "country":         r.get("country"),
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        loc,
            "image":           "",
            "size_km":         size_km,
            "precise":         False,
            "is_volcanic":     True,
        })
    return out


def _load_deforestation_cases() -> list[dict[str, Any]]:
    """Load PRODES deforestation cases (positive, expected_action=submit_to_ground)."""
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "deforestation_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load deforestation_cases.yaml: {e}")
        return []
    raw_cases = doc.get("cases", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for r in raw_cases:
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        size_km = float(r.get("size_km", 10.0))
        before_date = r["before_date"]
        after_date  = r["after_date"]
        loc = f"Brazil/{r.get('state','?')}"
        out.append({
            "id":              r["id"],
            "source":          "PRODES",
            "event":           r.get("name") or r["id"],
            "disaster_type":   "deforestation",
            "mapped_class":    "deforestation",
            "expected_action": r.get("expected_action", "submit_to_ground"),
            "area_km2":        r.get("area_km2"),
            "year":            r.get("year"),
            "state":           r.get("state"),
            "name":            r.get("name"),
            "country":         "Brazil",
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        loc,
            "image":           "",
            "size_km":         size_km,
            "precise":         False,
            "is_deforestation": True,
        })
    return out


def _load_algal_bloom_cases() -> list[dict[str, Any]]:
    """Load hand-curated harmful algal bloom cases."""
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "algal_bloom_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load algal_bloom_cases.yaml: {e}")
        return []
    raw_cases = doc.get("cases", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for r in raw_cases:
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        size_km = float(r.get("size_km", 20.0))
        before_date = r["before_date"]
        after_date  = r["after_date"]
        loc = f"{r.get('country','?')}/{r.get('region','?')}"
        out.append({
            "id":              r["id"],
            "source":          "HAB",
            "event":           r.get("name") or r["id"],
            "disaster_type":   "algal_bloom",
            "mapped_class":    "algal_bloom",
            "expected_action": r.get("expected_action", "submit_to_ground"),
            "species":         r.get("species"),
            "bloom_color":     r.get("bloom_color"),
            "name":            r.get("name"),
            "region":          r.get("region"),
            "country":         r.get("country"),
            "notes":           r.get("notes"),
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        loc,
            "image":           "",
            "size_km":         size_km,
            "precise":         False,
            "is_algal_bloom":  True,
        })
    return out


def _load_hard_negative_cases() -> list[dict[str, Any]]:
    """Load HARD NEGATIVE cases — same lat/lon as positive sources but in
    stable (pre-event) periods. Behaves like negative but flagged separately
    so UI can show them in their own optgroup."""
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "hard_negative_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load hard_negative_cases.yaml: {e}")
        return []
    raw_cases = doc.get("cases", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for r in raw_cases:
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        size_km = float(r.get("size_km", 10.0))
        before_date = r["before_date"]
        after_date  = r["after_date"]
        out.append({
            "id":              r["id"],
            "source":          "HARD_NEG",
            "event":           r.get("parent_id") or r["id"],
            "disaster_type":   r.get("negative_type", "no_change"),
            "mapped_class":    "no_change",
            "expected_action": r.get("expected_action", "drop"),
            "negative_type":   r.get("negative_type"),
            "parent_source":   r.get("parent_source"),
            "parent_id":       r.get("parent_id"),
            "biome":           r.get("biome"),
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        r.get("note", ""),
            "image":           "",
            "size_km":         size_km,
            "precise":         False,
            "is_negative":     True,
            "is_hard_negative": True,
        })
    return out


def _load_fireedge_hf_cases() -> list[dict[str, Any]]:
    """Load YujiYamaguchi/fireedge-sentinel2-wildfire HF dataset cases.
    Each case carries `sentinel_datetime` so a window=1 SimSat fetch
    reproduces the exact training-time S2 frame the LoRA saw.
    """
    path = APP_DIR.parent / "data" / "metadata" / "disaster_m3" / "fireedge_hf_cases.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load fireedge_hf_cases.yaml: {e}")
        return []
    raw_cases = doc.get("cases", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for r in raw_cases:
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        size_km = float(r.get("size_km", 5.0))
        before_date = r["before_date"]
        after_date  = r["after_date"]
        label_str   = r.get("label") or r.get("event_type") or "no_change"
        is_fire     = label_str == "fire"
        out.append({
            "id":              r["id"],
            "source":          "FireEdge_HF",
            "event":           r["id"],
            "disaster_type":   "fire" if is_fire else "no_change",
            "mapped_class":    "fire" if is_fire else "no_change",
            "expected_action": "submit_to_ground" if is_fire else "drop",
            "split":           r.get("fireedge_split"),
            "fireedge_source": r.get("fireedge_source"),
            "sentinel_datetime": r.get("sentinel_datetime"),
            "query_date":      r.get("query_date"),
            "capture_date":    after_date,
            "after_date":      after_date,
            "before_date":     before_date,
            "lat": lat, "lon": lon,
            "location":        f"FireEdge {r.get('fireedge_split')}/{r.get('fireedge_source')}",
            "image":           "",
            "size_km":         size_km,
            "window_days":     int(r.get("window_days", 1)),
            "precise":         False,
            "is_fireedge":     True,
            # Negatives carry is_negative so the existing scorer picks up
            # 'drop' as expected action; positives stay positive (= submit).
            "is_negative":     not is_fire,
            "negative_type":   None if is_fire else r.get("fireedge_source"),
        })
    return out


DM3_CASES: list[dict[str, Any]] = _load_dm3_cases() + _load_negative_cases() + _load_hard_negative_cases() + _load_ems_cases() + _load_volcanic_cases() + _load_deforestation_cases() + _load_algal_bloom_cases() + _load_fireedge_hf_cases()
_n_pos = sum(1 for c in DM3_CASES if not any(c.get(f) for f in ("is_negative","is_ems","is_volcanic","is_deforestation","is_algal_bloom")))
_n_neg     = sum(1 for c in DM3_CASES if c.get("is_negative") and not c.get("is_hard_negative"))
_n_hardneg = sum(1 for c in DM3_CASES if c.get("is_hard_negative"))
_n_ems = sum(1 for c in DM3_CASES if c.get("is_ems"))
_n_vol = sum(1 for c in DM3_CASES if c.get("is_volcanic"))
_n_def = sum(1 for c in DM3_CASES if c.get("is_deforestation"))
_n_hab = sum(1 for c in DM3_CASES if c.get("is_algal_bloom"))
_n_fe  = sum(1 for c in DM3_CASES if c.get("is_fireedge"))
print(f"[startup] DisasterM3 cases loaded: {len(DM3_CASES)} (positive/neutral={_n_pos}, negative={_n_neg}, hard_negative={_n_hardneg}, ems={_n_ems}, volcanic={_n_vol}, deforestation={_n_def}, hab={_n_hab}, fireedge={_n_fe})")


def _load_scene_catalog() -> list[dict[str, Any]]:
    """Load Phase 1 scene_catalog.yaml entries (MCD64A1 etc.) and convert to
    case-shape compatible with the DM3 dropdown. Re-read on every API call so
    appending to the catalog (via mcd64a1_smoke.py --save) is picked up live.
    """
    from datetime import date, timedelta
    path = APP_DIR.parent / "data" / "scene_catalog.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[startup] WARN: failed to load scene_catalog.yaml: {e}")
        return []
    out: list[dict[str, Any]] = []
    for s in doc.get("scenes", []):
        try:
            lat = float(s["lat"]); lon = float(s["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        period = s.get("event_period") or []
        ev_start = period[0] if len(period) >= 1 else None
        ev_end   = period[1] if len(period) >= 2 else ev_start
        # 60d before period start / 60d after period end as default Before/After.
        try:
            d_start = date.fromisoformat(ev_start)
            d_end   = date.fromisoformat(ev_end)
            before_d = (d_start - timedelta(days=60)).isoformat()
            after_d  = (d_end   + timedelta(days=60)).isoformat()
        except Exception:
            before_d = ev_start
            after_d  = ev_end
        et = s.get("event_type", "wildfire")
        out.append({
            "id":            s["id"],
            "source":        s.get("source", "Catalog"),
            "event":         s["id"],
            "disaster_type": et,
            "mapped_class":  "fire" if et == "wildfire" else et,
            "capture_date":  ev_end,
            "before_date":   before_d,
            "after_date":    after_d,
            "lat": lat, "lon": lon,
            "location":      f"{s.get('source','?')} burn ({s.get('affected_area_km2', '?')} km²)",
            "image":         "",
            "size_km":       10.0,
            "precise":       True,
            "event_start":   ev_start,
            "event_end":     ev_end,
            "event_name":    f"{s.get('affected_area_km2','?')} km² burn",
        })
    return out


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
    # Cache hit detection is loose: count any cached scene at this case's
    # (lat, lon) regardless of size/date/resolution. The frontend uses this
    # only as an "explored / not explored" hint; the actual fetch still keys
    # on the exact (lat, lon, date, size, res) tuple.
    by_latlon: dict[tuple[float, float], int] = {}
    for meta_p in CACHE_DIR.glob("*.meta.json"):
        try:
            d = json.loads(meta_p.read_text())
        except Exception:
            continue
        req = d.get("request") or {}
        lat = req.get("lat"); lon = req.get("lon")
        if lat is None or lon is None:
            continue
        png = CACHE_DIR / f"{meta_p.stem.replace('.meta','')}.png"
        if not png.exists():
            continue
        by_latlon[(round(lat, 4), round(lon, 4))] = by_latlon.get((round(lat, 4), round(lon, 4)), 0) + 1

    # Collect Phase 2 results from canonical_dataset.yaml as a separate
    # field so the UI can offer a "Use cache" dropdown distinct from the
    # default Fetch Images action (which keeps DM3-original size/dates).
    canonical_path = APP_DIR.parent / "data" / "canonical_dataset.yaml"
    canonical_by_id: dict[str, list[dict]] = {}
    if canonical_path.exists():
        try:
            with open(canonical_path, encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            for e in doc.get("cases", []):
                cid = e.get("id")
                if not cid:
                    continue
                req = e.get("request") or {}
                canonical_by_id.setdefault(cid, []).append({
                    "size_km":     e.get("size_km"),
                    "before_date": req.get("before_date"),
                    "after_date":  req.get("after_date"),
                    "before_resolved": (e.get("expected_resolved") or {}).get("before_datetime"),
                    "after_resolved":  (e.get("expected_resolved") or {}).get("after_datetime"),
                })
        except Exception:
            pass

    all_cases = DM3_CASES + _load_scene_catalog()
    out: list[dict[str, Any]] = []
    for c in all_cases:
        lat = c.get("lat"); lon = c.get("lon")
        n_cached = 0
        if lat is not None and lon is not None:
            n_cached = by_latlon.get((round(lat, 4), round(lon, 4)), 0)
        out.append({
            **c,
            "cached_count":    n_cached,
            "cached_before":   n_cached >= 2,
            "cached_after":    n_cached >= 2,
            "canonical_pairs": canonical_by_id.get(c.get("id", ""), []),
        })
    return {"count": len(out), "cases": out}


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
    # Reflect the configured catalog default (single source of truth).
    # The UI overrides this per-request via its own selector.
    try:
        cfg = _resolve_provider_cfg(None)
        provider_info = {
            "name":  cfg.get("name"),
            "kind":  cfg.get("kind"),
            "model": cfg.get("default_model") or "?",
        }
    except HTTPException:
        provider_info = {"kind": "none", "model": "no provider configured"}
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
    resolution_meters: int = Field(default=10, ge=10, le=120)


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
        key, meta = _fetch_one(req.lat, req.lon, target, req.size_km, req.window_days,
                               req.resolution_meters)
        return {
            "target_date": target,
            "offset_days": offset_days,
            "key": key,
            "meta": meta,
        }

    with ThreadPoolExecutor(max_workers=2) as ex:
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
    resolution_meters: int = Field(default=10, ge=10, le=120)


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
        key, meta = _fetch_one(req.lat, req.lon, target, req.size_km, req.window_days,
                               req.resolution_meters)
        return {
            "target_date": target,
            "offset_days": offset_days,
            "key": key,
            "meta": meta,
        }

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(probe, req.offsets_days))

    return {"candidates": results, "anchor_date": ref.strftime("%Y-%m-%d")}


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
    before_window = req.before_window_days if req.before_window_days is not None else req.window_days
    b_key, b_meta = _fetch_one(req.lat, req.lon, req.before_date, req.size_km,
                                before_window, req.resolution_meters)
    a_key, a_meta = _fetch_one(req.lat, req.lon, req.after_date,  req.size_km,
                                req.window_days, req.resolution_meters)
    return {
        "request": req.model_dump(),
        "before": {"key": b_key, "meta": b_meta, "date": req.before_date},
        "after":  {"key": a_key, "meta": a_meta, "date": req.after_date},
    }


class SavePairRequest(BaseModel):
    scene_id: str
    lat: float
    lon: float
    before_date: str
    after_date: str
    size_km: float
    # When the UI has resolved keys (e.g., from a candidate click that may have
    # used a different target_date than what's now in before_date), pass them
    # directly so the save bypasses re-computing the cache key.
    before_key: str | None = None
    after_key: str | None = None
    label: str | None = None
    event_type: str | None = None
    event_start: str | None = None
    event_end: str | None = None
    event_name: str | None = None
    is_negative: bool = False
    negative_type: str | None = None
    expected_action: str | None = None


@app.post("/api/scene/save_pair")
def api_save_pair(req: SavePairRequest) -> dict[str, Any]:
    """Append the current Before/After pair to canonical_dataset.yaml + copy
    PNGs into data/curated_pairs/<scene_id>/ for easy filesystem inspection."""
    canonical_path = APP_DIR.parent / "data" / "canonical_dataset.yaml"
    curated_dir   = APP_DIR.parent / "data" / "curated_pairs" / req.scene_id
    curated_dir.mkdir(parents=True, exist_ok=True)

    bk = req.before_key or _cache_key(req.lat, req.lon, _normalize_ts(req.before_date), req.size_km, 10)
    ak = req.after_key  or _cache_key(req.lat, req.lon, _normalize_ts(req.after_date),  req.size_km, 10)
    bp = CACHE_DIR / f"{bk}.png"; ap = CACHE_DIR / f"{ak}.png"
    bm = _load_meta(bk) or {}
    am = _load_meta(ak) or {}
    if not (bp.exists() and ap.exists()):
        raise HTTPException(400, "Before/After not in cache yet — press Fetch Images first")

    # Copy PNGs + sidecar to curated dir
    import shutil
    for src, name in ((bp, "before.png"), (ap, "after.png")):
        shutil.copy2(src, curated_dir / name)
    with open(curated_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "scene_id":   req.scene_id,
            "type":       "negative" if req.is_negative else "positive",
            "expected_action": (req.expected_action or "drop") if req.is_negative else "submit_to_ground",
            "negative_type":   req.negative_type if req.is_negative else None,
            "lat":        req.lat,
            "lon":        req.lon,
            "size_km":    req.size_km,
            "before":     {"date": req.before_date, "key": bk, "datetime": bm.get("datetime"), "stats": bm.get("stats")},
            "after":      {"date": req.after_date,  "key": ak, "datetime": am.get("datetime"), "stats": am.get("stats")},
            "event":      {"type": req.event_type, "start": req.event_start, "end": req.event_end, "name": req.event_name},
            "saved_at":   datetime.now(timezone.utc).isoformat(),
        }, f, sort_keys=False, allow_unicode=True)

    # Append/replace entry in canonical_dataset.yaml
    if canonical_path.exists():
        with open(canonical_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    else:
        doc = {}
    cases = doc.get("cases") or []
    cases = [c for c in cases if c.get("id") != req.scene_id]
    # The canonical entry's request.before_date/after_date must be the *target*
    # date that produced the cache key (not the resolved S2 datetime), so that
    # auto_phase2 / Use cache can re-compute the same key. Pull from meta when
    # available; fall back to the form value.
    def _date_from_ts(meta, fallback):
        ts = (meta.get("request") or {}).get("ts")
        return ts.split("T", 1)[0] if isinstance(ts, str) and "T" in ts else fallback
    canonical_before_date = _date_from_ts(bm, req.before_date)
    canonical_after_date  = _date_from_ts(am, req.after_date)
    entry = {
        "id":      req.scene_id,
        "label":   req.label or req.event_type or ("no_change" if req.is_negative else "wildfire"),
        "type":    "negative" if req.is_negative else "positive",
        "lat":     float(req.lat),
        "lon":     float(req.lon),
        "size_km": float(req.size_km),
        "request": {"before_date": canonical_before_date, "after_date": canonical_after_date, "window_days": 30},
        "expected_resolved": {"before_datetime": bm.get("datetime"), "after_datetime": am.get("datetime")},
        "event":   {"name": req.event_name, "period": [req.event_start, req.event_end]},
    }
    if req.is_negative:
        entry["expected_action"] = req.expected_action or "drop"
        if req.negative_type:
            entry["negative_type"] = req.negative_type
    cases.append(entry)
    doc["cases"] = cases
    with open(canonical_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    return {
        "saved_dir": str(curated_dir.relative_to(APP_DIR.parent)),
        "canonical_entries": len(cases),
        "before_key": bk, "after_key": ak,
    }


@app.get("/api/scene/burn_polygon/{scene_id}")
def api_scene_burn_polygon(scene_id: str):
    """Return the GT burn polygon for an MCD64A1 catalog scene."""
    if not all(c.isalnum() or c in "_+-" for c in scene_id):
        raise HTTPException(400, "invalid id")
    p = APP_DIR.parent / "data" / "gt_polygons" / f"{scene_id}.geojson"
    if not p.exists():
        raise HTTPException(404, "polygon not found")
    return FileResponse(p, media_type="application/geo+json")


@app.get("/api/image/{key}")
def api_image(key: str):
    if not key.replace("_", "").isalnum():
        raise HTTPException(400, "invalid key")
    for d in (CACHE_DIR, DERIVED_DIR):
        path = d / f"{key}.png"
        if path.exists():
            return FileResponse(path, media_type="image/png")
    raise HTTPException(404, "image not found")


AGENT_TRACES_DIR = APP_DIR.parent / "data" / "traces" / "agent"
AGENT_TRACES_DIR.mkdir(parents=True, exist_ok=True)


def _lookup_canonical_entry(scene_id: str | None) -> dict[str, Any]:
    if not scene_id:
        return {}
    canonical_path = APP_DIR.parent / "data" / "canonical_dataset.yaml"
    if not canonical_path.exists():
        return {}
    try:
        with open(canonical_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        for c in doc.get("cases", []):
            if c.get("id") == scene_id:
                return c
    except Exception:
        pass
    return {}


def _save_agent_trace(scene_id: str, events: list[dict], provider: str | None,
                      model: str | None, before_key: str, after_key: str) -> str | None:
    if not scene_id or not events:
        return None
    canon = _lookup_canonical_entry(scene_id)
    expected_action = canon.get("expected_action") or (
        "drop" if canon.get("type") == "negative" else "submit_to_ground"
    )
    final_ev = next((e for e in reversed(events) if e.get("type") == "final"), None)
    actual_action = (final_ev or {}).get("name") or (final_ev or {}).get("action")
    gt_match = (actual_action == expected_action) if final_ev else None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{_safe_slug(scene_id)}__{ts}.yaml"
    path = AGENT_TRACES_DIR / fname
    doc = {
        "metadata": {
            "scene_id":         scene_id,
            "scenario_type":    canon.get("type"),
            "expected_action":  expected_action,
            "expected_class":   canon.get("label"),
            "lat":              canon.get("lat"),
            "lon":              canon.get("lon"),
            "size_km":          canon.get("size_km"),
            "before_date":      (canon.get("request") or {}).get("before_date"),
            "after_date":       (canon.get("request") or {}).get("after_date"),
            "before_key":       before_key,
            "after_key":        after_key,
            "provider":         provider,
            "model":            model,
            "collected_at":     datetime.now(timezone.utc).isoformat(),
        },
        "events": events,
        "final":  final_ev,
        "gt_match": gt_match,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return str(path.relative_to(APP_DIR.parent))


def _run_lfm2_as_events(scene_id: str, base_url: str,
                         served_model: str | None,
                         before_path: Path, after_path: Path):
    """Look up the case, then delegate straight to agent.lfm2_agent's
    streaming generator. Each tool_call / observation reaches the SSE
    consumer the moment it happens (no longer "after-the-fact replay").
    """
    from agent.lfm2_agent import iter_lfm2_agent

    cid = scene_id
    if not all(c.isalnum() or c in "_+-" for c in cid):
        raise HTTPException(400, "invalid scene_id")
    case = next((c for c in DM3_CASES + _load_scene_catalog() if c.get("id") == cid), None)
    if case is None:
        raise HTTPException(404, f"scene_id not found in DM3 catalog: {cid}")
    if not (case.get("before_date") and case.get("after_date")):
        raise HTTPException(400, f"case missing before_date/after_date: {cid}")

    case_meta = {
        "lat":         float(case["lat"]),
        "lon":         float(case["lon"]),
        "before_date": case["before_date"],
        "after_date":  case["after_date"],
        "size_km":     float(case.get("size_km", 10.0)),
    }
    served = served_model or os.environ.get("LFM2_AGENT_MODEL", "LFM2.5-VL-450M-sft-grpo")
    yield from iter_lfm2_agent(
        case_id=cid, case_meta=case_meta,
        before_path=str(before_path), after_path=str(after_path),
        vllm_url=base_url, served_model=served,
        include_images=False, max_turns=6, temperature=0.0,
    )


@app.get("/api/run_agent")
def api_run_agent(before_key: str, after_key: str,
                  provider: str | None = None, model: str | None = None,
                  scene_id: str | None = None,
                  instructions: str | None = None):
    before_path = CACHE_DIR / f"{before_key}.png"
    after_path  = CACHE_DIR / f"{after_key}.png"
    if not before_path.exists() or not after_path.exists():
        raise HTTPException(400, "fetch images first")

    cfg = _resolve_provider_cfg(provider)
    context = _context_from_keys(before_key, after_key)
    tool_registry = build_tool_registry(str(before_path), str(after_path), context,
                                        provider_name=provider, model=model)

    def generate():
        events_collected: list[dict] = []
        try:
            if cfg.get("kind") == "openai_compat":
                api_key_env = cfg.get("api_key_env")
                api_key = os.environ.get(api_key_env, "dummy") if api_key_env else "dummy"
                events = run_react_openai(
                    str(before_path), str(after_path),
                    base_url=cfg["base_url"],
                    model=model or cfg.get("default_model"),
                    api_key=api_key,
                    tool_registry=tool_registry,
                    user_instructions=instructions,
                )
            elif cfg.get("kind") == "gemini":
                if PROVIDER is None:
                    raise HTTPException(
                        503,
                        "Gemini provider selected but GOOGLE_API_KEY/GEMINI_API_KEY is not set",
                    )
                events = run_react(str(before_path), str(after_path),
                                   provider=PROVIDER, tool_registry=tool_registry)
            elif cfg.get("kind") == "lfm2_multiturn":
                # The 450M sft-grpo agent has its own multi-turn loop with
                # case_meta-based realtime SimSat fetch. We run it to
                # completion (no streaming yet — see REPRO_PLAN Phase 7),
                # then replay the trace as SSE events so the UI renders
                # identically to the openai_compat / gemini paths.
                if not scene_id:
                    raise HTTPException(400, (
                        "lfm2_multiturn provider requires a DM3 case to be "
                        "selected (scene_id missing). Pick a case in the "
                        "DisasterM3 dropdown before clicking Run Agent."
                    ))
                events = _run_lfm2_as_events(
                    scene_id=scene_id,
                    base_url=cfg["base_url"],
                    served_model=model or cfg.get("default_model"),
                    before_path=before_path, after_path=after_path,
                )
            else:
                raise HTTPException(500, f"unknown provider kind '{cfg.get('kind')}'")
            for event in events:
                events_collected.append(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "text": f"agent crashed: {type(e).__name__}: {e}"}
            events_collected.append(err)
            yield f"data: {json.dumps(err)}\n\n"
        finally:
            saved = _save_agent_trace(scene_id, events_collected, provider, model,
                                      before_key, after_key)
            if saved:
                yield f"data: {json.dumps({'type': 'note', 'text': f'trace saved → {saved}'}, ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---- Human annotation endpoints ----------------------------------------

class ToolInvokeRequest(BaseModel):
    before_key: str
    after_key: str
    tool_name: str
    arguments: dict[str, Any] = {}
    provider: str | None = None
    model: str | None = None


class RunLfm2AgentRequest(BaseModel):
    """Drive agent.lfm2_agent.run_lfm2_agent with a curated case_id.

    Looks up the case in canonical_dataset.yaml (or a prebuilt curated_pair
    directory) so the front-end only has to pass the case_id.
    """
    scene_id: str
    include_images: bool = False  # 75% accuracy mode (S63 finding)
    max_turns: int = 6
    temperature: float = 0.0
    vllm_url: str | None = None
    served_model: str | None = None
    precompute_root: str | None = None


@app.post("/api/run_lfm2_agent")
def api_run_lfm2_agent(req: RunLfm2AgentRequest) -> dict[str, Any]:
    """Run the LFM2.5-VL multi-turn SFT/GRPO agent on one case.

    Wires the case_id → curated_pairs/<id>/{before,after}.png + the offline
    precompute cache, then invokes agent.lfm2_agent.run_lfm2_agent.
    Returns the terminal action and tool-call log so the UI can render it
    the same way the Gemini ReAct agent's traces are rendered.
    """
    from agent.lfm2_agent import run_lfm2_agent
    cid = req.scene_id
    if not all(c.isalnum() or c in "_+-" for c in cid):
        raise HTTPException(400, "invalid scene_id")
    repo_root = APP_DIR.parent
    curated   = repo_root / "data" / "curated_pairs" / cid
    before_p  = curated / "before.png"
    after_p   = curated / "after.png"
    have_pair = before_p.exists() and after_p.exists()

    # Build case_meta (lat/lon/dates/size_km) so tools.spectral.compute_index_delta_impl
    # can be called live by the agent. MCD64A1 cases live in scene_catalog
    # (re-read each call), the rest in DM3_CASES (loaded at startup).
    case = next((c for c in DM3_CASES + _load_scene_catalog() if c.get("id") == cid), None)
    if case is None:
        raise HTTPException(404, f"scene_id not found in DM3 catalog: {cid}")
    case_meta = {
        "lat":         float(case["lat"]),
        "lon":         float(case["lon"]),
        "before_date": case.get("before_date"),
        "after_date":  case.get("after_date"),
        "size_km":     float(case.get("size_km", 10.0)),
    }
    if not (case_meta["before_date"] and case_meta["after_date"]):
        raise HTTPException(400, f"case missing before_date/after_date: {cid}")

    vllm_url     = req.vllm_url     or os.environ.get("LFM2_AGENT_VLLM_URL", "http://localhost:8086/v1")
    served_model = req.served_model or os.environ.get("LFM2_AGENT_MODEL", "LFM2.5-VL-450M-sft-grpo")

    try:
        result = run_lfm2_agent(
            case_id=cid,
            case_meta=case_meta,
            before_path=str(before_p) if have_pair else None,
            after_path=str(after_p)   if have_pair else None,
            vllm_url=vllm_url,
            served_model=served_model,
            include_images=req.include_images,
            max_turns=req.max_turns,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(500, f"run_lfm2_agent failed: {type(e).__name__}: {e}")

    # Strip any large message bodies before returning to the browser, and
    # extract observations (role="tool" messages) so the UI can interleave
    # them with the action log.
    msgs_compact: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for m in result.get("messages") or []:
        if not isinstance(m, dict):
            msgs_compact.append(m)
            continue
        c = m.get("content")
        if isinstance(c, list):
            scrubbed = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    scrubbed.append({"type": "image_url", "image_url": {"url": "<stripped>"}})
                else:
                    scrubbed.append(part)
            msgs_compact.append({"role": m.get("role"), "content": scrubbed})
        else:
            msgs_compact.append(m)
        if m.get("role") == "tool":
            observations.append({
                "name": m.get("name"),
                "tool_call_id": m.get("tool_call_id"),
                "content": m.get("content"),
            })

    return {
        "scene_id":       cid,
        "terminal":       result.get("terminal"),
        "tool_call_log":  result.get("tool_call_log") or [],
        "observations":   observations,
        "raw_log":        result.get("raw_log") or [],
        "messages":       msgs_compact,
        "vllm_url":       vllm_url,
        "served_model":   served_model,
        "case_meta":      case_meta,
        "include_images": req.include_images,
    }


@app.get("/api/providers")
def api_providers() -> dict[str, Any]:
    """Expose VLM provider/model catalog so the UI can render selectors."""
    out = []
    for p in PROVIDERS_CFG:
        out.append({
            "name":          p.get("name"),
            "kind":          p.get("kind"),
            "models":        p.get("models") or [],
            "default_model": p.get("default_model"),
        })
    return {"providers": out}


@app.post("/api/tool/invoke")
def api_tool_invoke(req: ToolInvokeRequest) -> dict[str, Any]:
    before_path = CACHE_DIR / f"{req.before_key}.png"
    after_path  = CACHE_DIR / f"{req.after_key}.png"
    if not before_path.exists() or not after_path.exists():
        raise HTTPException(400, "images not found; fetch first")

    context = _context_from_keys(req.before_key, req.after_key)
    registry = build_tool_registry(str(before_path), str(after_path), context,
                                   provider_name=req.provider, model=req.model)
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
    files: list[tuple[Path, str]] = []
    for d, kind in ((TRACES_DIR, "human"), (AGENT_TRACES_DIR, "agent")):
        for p in d.glob("*.yaml"):
            files.append((p, kind))
    files.sort(key=lambda fk: fk[0].stat().st_mtime, reverse=True)
    out = []
    for f, kind in files[:300]:
        try:
            with open(f, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            final = doc.get("final", {}) if isinstance(doc, dict) else {}
            events = doc.get("events", []) if isinstance(doc, dict) else []
            out.append({
                "filename":     f.name,
                "kind":         kind,
                "scenario_id":  meta.get("scenario_id") or meta.get("scene_id") or "?",
                "profile":      meta.get("profile") or meta.get("provider") or "?",
                "annotator":    meta.get("annotator") or meta.get("model") or "?",
                "created_at":   meta.get("created_at") or meta.get("collected_at") or "",
                "final_action": final.get("action") or final.get("name") or "?",
                "final_change_type": final.get("change_type"),
                "expected_action":   meta.get("expected_action"),
                "gt_match":          doc.get("gt_match") if isinstance(doc, dict) else None,
                "n_events":     len(events),
                "size_bytes":   f.stat().st_size,
            })
        except Exception as e:
            out.append({"filename": f.name, "kind": kind, "error": str(e)})
    return {"traces": out}


def _resolve_trace_path(filename: str) -> Path:
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".yaml"):
        raise HTTPException(400, f"invalid filename: {filename}")
    for d in (TRACES_DIR, AGENT_TRACES_DIR):
        p = d / filename
        if p.exists() and p.is_file():
            return p
    raise HTTPException(404, f"trace not found: {filename}")


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
    uvicorn.run(app, host=os.environ.get("APP_HOST", "127.0.0.1"), port=int(os.environ.get("APP_PORT", "7860")))
