"""Spectral tools for the onboard VLM agent.

Three tools, all producing PNG artifacts the agent (or human annotator) can look at:

- `fetch_band_ts` : single band, grayscale PNG. Any Sentinel-2 band SimSat accepts.
- `false_color`   : 3-band RGB composite (e.g. NIR-Red-Green for vegetation).
- `compute_index` : standard spectral index (NDVI, NDWI, MNDWI, NBR, NDBI, NDSI).

All three accept lat/lon/timestamp/size_km directly so the tools are self-contained
and require no outer "context" injection — the agent fills these from the prompt.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from simsat_client import (
    ALL_BANDS,
    INDEX_DEFINITIONS,
    SimSatError,
    fetch_sentinel_array,
    fetch_sentinel_image_bands,
)


OUTPUT_DIR = Path(__file__).parent.parent / "data" / "derived"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_ts(d: str) -> str:
    s = d.strip()
    if "T" not in s:
        s = f"{s}T00:00:00Z"
    elif not s.endswith("Z"):
        s = s + "Z"
    return s


def _save_png(img: Image.Image, prefix: str) -> tuple[str, str]:
    uid = uuid.uuid4().hex[:10]
    key = f"{prefix}_{uid}"
    path = OUTPUT_DIR / f"{key}.png"
    img.save(path)
    return key, str(path)


def _stretch_to_u8(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Linear-stretch a float or int array to uint8 using lo/hi percentiles."""
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).any():
        return np.zeros(a.shape, dtype=np.uint8)
    finite = a[np.isfinite(a)]
    lo_v, hi_v = np.percentile(finite, (lo, hi))
    if hi_v <= lo_v:
        return np.zeros(a.shape, dtype=np.uint8)
    out = (a - lo_v) / (hi_v - lo_v)
    out = np.clip(out, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def _colormap_signed(a: np.ndarray) -> np.ndarray:
    """Map signed [-1, 1] index to an RGB pseudocolor (red → white → green)."""
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    # Simple diverging red (-1) - white (0) - green (+1) palette.
    pos = np.clip(a, 0.0, 1.0)
    neg = np.clip(-a, 0.0, 1.0)
    r = ((1 - pos) * 255).astype(np.uint8)
    g = ((1 - neg) * 255).astype(np.uint8)
    b = (np.minimum(1 - pos, 1 - neg) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _colormap_delta(a: np.ndarray, vmax: float = 0.4) -> np.ndarray:
    """Diverging heatmap for index deltas (After - Before).

    vmax = saturation point. |delta| >= vmax maps to full color.
    - Red  = decrease (veg loss / burn scar / building collapse)
    - Blue = increase (new vegetation / new water / cloud)
    - White = no change
    NaN pixels rendered black.
    """
    a = np.asarray(a, dtype=np.float32)
    nan_mask = ~np.isfinite(a)
    a = np.nan_to_num(a, nan=0.0)
    norm = np.clip(a / max(vmax, 1e-6), -1.0, 1.0)
    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)
    r = ((1.0 - pos) * 255).astype(np.uint8)
    g = (np.clip(1.0 - pos - neg, 0.0, 1.0) * 255).astype(np.uint8)
    b = ((1.0 - neg) * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    if nan_mask.any():
        rgb[nan_mask] = [0, 0, 0]
    return rgb


# ---- Tool implementations -----------------------------------------------

def fetch_band_impl(
    band: str,
    lat: float,
    lon: float,
    timestamp: str,
    size_km: float = 10.0,
    window_days: int = 10,
) -> dict[str, Any]:
    """Return a grayscale PNG of a single Sentinel-2 band."""
    if band not in ALL_BANDS:
        return {"error": f"unknown band '{band}'. Valid: {list(ALL_BANDS)}"}
    try:
        result = fetch_sentinel_image_bands(
            lat=lat, lon=lon, timestamp=_normalize_ts(timestamp),
            bands=[band], size_km=size_km, window_days=window_days,
        )
    except SimSatError as e:
        return {"error": str(e)}
    key, _ = _save_png(result.image, prefix=f"band_{band}")
    return {
        "image_key": key,
        "band": band,
        "cloud_cover": result.metadata.get("cloud_cover"),
        "datetime": result.metadata.get("datetime"),
        "source": result.metadata.get("source"),
    }


def false_color_impl(
    bands: list[str],
    lat: float,
    lon: float,
    timestamp: str,
    size_km: float = 10.0,
    window_days: int = 10,
) -> dict[str, Any]:
    """RGB composite from 3 bands."""
    if not isinstance(bands, list) or len(bands) != 3:
        return {"error": f"bands must be a list of exactly 3 band names, got {bands}"}
    unknown = [b for b in bands if b not in ALL_BANDS]
    if unknown:
        return {"error": f"unknown band(s) {unknown}. Valid: {list(ALL_BANDS)}"}
    try:
        result = fetch_sentinel_image_bands(
            lat=lat, lon=lon, timestamp=_normalize_ts(timestamp),
            bands=bands, size_km=size_km, window_days=window_days,
        )
    except SimSatError as e:
        return {"error": str(e)}
    key, _ = _save_png(result.image, prefix="fc_" + "_".join(bands))
    return {
        "image_key": key,
        "bands_rgb": bands,
        "cloud_cover": result.metadata.get("cloud_cover"),
        "datetime": result.metadata.get("datetime"),
        "source": result.metadata.get("source"),
    }


def compute_index_impl(
    index: str,
    lat: float,
    lon: float,
    timestamp: str,
    size_km: float = 10.0,
    window_days: int = 10,
    pseudocolor: bool = True,
) -> dict[str, Any]:
    """Compute a standard spectral index and return as a viewable PNG."""
    idx_up = index.upper()
    if idx_up not in INDEX_DEFINITIONS:
        return {"error": f"unknown index '{index}'. Valid: {list(INDEX_DEFINITIONS)}"}
    a_name, b_name = INDEX_DEFINITIONS[idx_up]
    try:
        result = fetch_sentinel_array(
            lat=lat, lon=lon, timestamp=_normalize_ts(timestamp),
            bands=[a_name, b_name], size_km=size_km, window_days=window_days,
        )
    except SimSatError as e:
        return {"error": str(e)}

    names = [n.lower() for n in result.band_names]
    try:
        i_a = names.index(a_name)
        i_b = names.index(b_name)
    except ValueError:
        return {"error": f"returned bands {result.band_names} missing {a_name} or {b_name}"}

    a = result.array[i_a].astype(np.float32)
    b = result.array[i_b].astype(np.float32)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        idx_arr = np.where(denom > 0, (a - b) / denom, np.nan)

    if pseudocolor:
        rgb = _colormap_signed(idx_arr)
        img = Image.fromarray(rgb, "RGB")
    else:
        u8 = _stretch_to_u8(idx_arr)
        img = Image.fromarray(u8, "L")

    key, _ = _save_png(img, prefix=f"idx_{idx_up.lower()}")
    finite = idx_arr[np.isfinite(idx_arr)]
    stats = {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
    }
    return {
        "image_key": key,
        "index": idx_up,
        "bands_used": [a_name, b_name],
        "stats": stats,
        "cloud_cover": result.metadata.get("cloud_cover"),
        "datetime": result.metadata.get("datetime"),
        "source": result.metadata.get("source"),
    }


def compute_index_delta_impl(
    index: str,
    lat: float,
    lon: float,
    before_ts: str,
    after_ts: str,
    size_km: float = 10.0,
    window_days: int = 10,
    vmax: float = 0.4,
) -> dict[str, Any]:
    """Compute the per-pixel delta (After - Before) of a spectral index.

    Fetches both time points, computes the index on each, subtracts, and
    renders the diverging heatmap on the After side. Useful for change
    detection: red = index decreased, blue = increased.
    """
    idx_up = index.upper()
    if idx_up not in INDEX_DEFINITIONS:
        return {"error": f"unknown index '{index}'. Valid: {list(INDEX_DEFINITIONS)}"}
    a_name, b_name = INDEX_DEFINITIONS[idx_up]

    def _index_at(ts: str):
        try:
            result = fetch_sentinel_array(
                lat=lat, lon=lon, timestamp=_normalize_ts(ts),
                bands=[a_name, b_name], size_km=size_km, window_days=window_days,
            )
        except SimSatError as e:
            return None, str(e), None
        names = [n.lower() for n in result.band_names]
        try:
            i_a = names.index(a_name)
            i_b = names.index(b_name)
        except ValueError:
            return None, f"bands {result.band_names} missing {a_name} or {b_name}", None
        a = result.array[i_a].astype(np.float32)
        b = result.array[i_b].astype(np.float32)
        denom = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            idx_arr = np.where(denom > 0, (a - b) / denom, np.nan)
        return idx_arr, None, result.metadata

    before_idx, err_b, before_meta = _index_at(before_ts)
    if err_b is not None:
        return {"error": f"before: {err_b}"}
    after_idx, err_a, after_meta = _index_at(after_ts)
    if err_a is not None:
        return {"error": f"after: {err_a}"}

    b_dt = (before_meta or {}).get("datetime")
    a_dt = (after_meta  or {}).get("datetime")
    if b_dt and a_dt and b_dt == a_dt:
        return {
            "error": (
                f"Before and After resolved to the same S2 scene ({str(b_dt)[:10]}). "
                f"Pick a Before date >10 days away from the current After "
                f"(SimSat's 10-day backward search causes overlap otherwise)."
            ),
            "before_datetime": b_dt,
            "after_datetime":  a_dt,
            "index": idx_up,
        }

    if before_idx.shape != after_idx.shape:
        # Crop to the common overlap
        h = min(before_idx.shape[0], after_idx.shape[0])
        w = min(before_idx.shape[1], after_idx.shape[1])
        before_idx = before_idx[:h, :w]
        after_idx  = after_idx[:h, :w]

    delta = after_idx - before_idx
    rgb = _colormap_delta(delta, vmax=vmax)
    img = Image.fromarray(rgb, "RGB")
    key, _ = _save_png(img, prefix=f"delta_{idx_up.lower()}")

    finite = delta[np.isfinite(delta)]
    stats = {
        "min":    float(finite.min())    if finite.size else None,
        "max":    float(finite.max())    if finite.size else None,
        "mean":   float(finite.mean())   if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "frac_decrease_strong": float(np.mean(finite < -vmax * 0.5)) if finite.size else None,
        "frac_increase_strong": float(np.mean(finite >  vmax * 0.5)) if finite.size else None,
    }
    return {
        "image_key": key,
        "index": idx_up,
        "delta_vmax": vmax,
        "stats": stats,
        "before_datetime": before_meta.get("datetime") if before_meta else None,
        "after_datetime":  after_meta.get("datetime")  if after_meta  else None,
    }


# ---- Factories --------------------------------------------------------

def make_fetch_band(lat: float, lon: float, size_km: float,
                    before_ts: str, after_ts: str) -> Callable:
    """Bind lat/lon/size_km so the agent only has to say band + which."""
    def fetch_band(band: str, which: str = "after", **_ignored) -> dict[str, Any]:
        ts = before_ts if which == "before" else after_ts
        return fetch_band_impl(band, lat, lon, ts, size_km)
    return fetch_band


def make_false_color(lat: float, lon: float, size_km: float,
                     before_ts: str, after_ts: str) -> Callable:
    def false_color(bands: list[str], which: str = "after", **_ignored) -> dict[str, Any]:
        ts = before_ts if which == "before" else after_ts
        return false_color_impl(bands, lat, lon, ts, size_km)
    return false_color


def make_compute_index(lat: float, lon: float, size_km: float,
                       before_ts: str, after_ts: str) -> Callable:
    def compute_index(index: str, which: str = "after", **_ignored) -> dict[str, Any]:
        ts = before_ts if which == "before" else after_ts
        return compute_index_impl(index, lat, lon, ts, size_km)
    return compute_index


def make_compute_index_delta(lat: float, lon: float, size_km: float,
                             before_ts: str, after_ts: str) -> Callable:
    def compute_index_delta(index: str, **_ignored) -> dict[str, Any]:
        return compute_index_delta_impl(index, lat, lon, before_ts, after_ts, size_km)
    return compute_index_delta
