"""SimSat Sentinel-2 client.

Thin wrapper around SimSat's `/data/image/sentinel` endpoint. SimSat itself
wraps AWS STAC Sentinel-2 L2A — bands are pulled from real imagery and
returned as PNG with metadata in an HTTP header, or as a base64-encoded raw
array when `return_type=array`.

Available band names (per SimSat's sentinel_provider.py):
  coastal (B01)  blue (B02)  green (B03)  red (B04)
  rededge1-3 (B05-07)  nir (B08)  nir08 (B08A)  nir09 (B09)
  swir16 (B11)  swir22 (B12)  aot  scl  visual  wvp
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image


DEFAULT_BASE = os.environ.get("SIMSAT_API_URL", "http://localhost:9005")

# Canonical band list (SimSat names). Keep in sync with sentinel_provider.
ALL_BANDS: tuple[str, ...] = (
    "coastal", "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09",
    "swir16", "swir22",
    "aot", "scl", "visual", "wvp",
)


class SimSatError(RuntimeError):
    pass


@dataclass
class SentinelImage:
    image: Image.Image
    metadata: dict[str, Any]


@dataclass
class SentinelArray:
    """Raw multi-band array returned from SimSat with `return_type=array`."""
    array: np.ndarray  # shape: (n_bands, H, W)
    band_names: list[str]
    metadata: dict[str, Any]


def auto_resolution_meters(size_km: float) -> int:
    """Pick a reasonable Sentinel-2 resolution for fast exploration (Phase 1).

    Phase 1 (date discovery / candidate exploration) prioritises speed —
    coarser fetches use the S2 COG overview pyramid (30m fetch ≈ 9x cheaper
    than 10m on the same bbox). Trade-off: zoom-in fidelity is reduced,
    which is fine for "is this the right pair?" checks but NOT for the
    final cache used in GRPO training.

    Phase 3 prewarm uses native 10m explicitly via the `resolution_meters`
    parameter on each fetch function (overrides this default).
    """
    if size_km <= 10:
        return 10
    if size_km <= 30:
        return 20
    if size_km <= 60:
        return 30
    return 60


def fetch_sentinel_image(
    lat: float,
    lon: float,
    timestamp: str,
    size_km: float = 5.0,
    bands: list[str] | None = None,
    window_days: int = 10,
    base_url: str | None = None,
    timeout: float = 180.0,
    resolution_meters: int | None = None,
) -> SentinelImage:
    """Fetch a Sentinel-2 PNG at (lat, lon) near `timestamp`.

    Args:
        timestamp: ISO-8601 UTC, e.g. "2024-08-15T00:00:00Z".
        size_km: footprint size in km.
        bands: spectral bands (default ["red","green","blue"] for true-color).
        window_days: acceptable time window to search for a cloud-free scene.
    """
    base = (base_url or DEFAULT_BASE).rstrip("/")
    bands = bands or ["red", "green", "blue"]
    res = int(resolution_meters if resolution_meters is not None else auto_resolution_meters(size_km))
    params: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp,
        "size_km": size_km,
        "spectral_bands": bands,
        "window_seconds": int(window_days * 24 * 60 * 60),
        "return_type": "png",
        "resolution_meters": res,
    }
    url = f"{base}/data/image/sentinel"
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise SimSatError(f"SimSat unreachable at {url}: {e}") from e
    if r.status_code != 200:
        raise SimSatError(f"SimSat {r.status_code}: {r.text[:200]}")
    if not r.content:
        raise SimSatError("SimSat returned empty body (no image available in window)")
    try:
        meta = json.loads(r.headers.get("sentinel_metadata", "{}"))
    except json.JSONDecodeError:
        meta = {}
    if meta.get("image_available") is False:
        raise SimSatError(f"No Sentinel-2 image available near {timestamp} "
                          f"(lat={lat}, lon={lon}, window_days={window_days})")
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return SentinelImage(image=img, metadata=meta)


def fetch_sentinel_image_bands(
    lat: float,
    lon: float,
    timestamp: str,
    bands: list[str],
    size_km: float = 5.0,
    window_days: int = 10,
    base_url: str | None = None,
    timeout: float = 60.0,
    resolution_meters: int | None = None,
) -> SentinelImage:
    """Fetch SimSat as PNG composed from the given 1 or 3 bands.

    SimSat's `image_to_png` accepts exactly 1 band (grayscale) or 3 bands (RGB).
    """
    if len(bands) not in (1, 3):
        raise SimSatError(f"bands must be a list of 1 or 3 entries, got {bands}")
    base = (base_url or DEFAULT_BASE).rstrip("/")
    res = int(resolution_meters if resolution_meters is not None else auto_resolution_meters(size_km))
    params: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp,
        "size_km": size_km,
        "spectral_bands": bands,
        "window_seconds": int(window_days * 24 * 60 * 60),
        "return_type": "png",
        "resolution_meters": res,
    }
    url = f"{base}/data/image/sentinel"
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise SimSatError(f"SimSat unreachable at {url}: {e}") from e
    if r.status_code != 200:
        raise SimSatError(f"SimSat {r.status_code}: {r.text[:200]}")
    if not r.content:
        raise SimSatError(f"SimSat returned empty body for bands {bands}")
    try:
        meta = json.loads(r.headers.get("sentinel_metadata", "{}"))
    except json.JSONDecodeError:
        meta = {}
    if meta.get("image_available") is False:
        raise SimSatError(f"No Sentinel-2 image available at {lat},{lon} near {timestamp}")
    img = Image.open(io.BytesIO(r.content))
    if len(bands) == 3:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
    return SentinelImage(image=img, metadata=meta)


def fetch_sentinel_array(
    lat: float,
    lon: float,
    timestamp: str,
    bands: list[str],
    size_km: float = 5.0,
    window_days: int = 10,
    base_url: str | None = None,
    timeout: float = 120.0,
    resolution_meters: int | None = None,
) -> SentinelArray:
    """Fetch SimSat with `return_type=array` and decode to a numpy array.

    Supports any number of bands (unlike the PNG endpoint). Needed for indices
    like NDVI/NDWI that require arithmetic on raw band values.
    """
    base = (base_url or DEFAULT_BASE).rstrip("/")
    res = int(resolution_meters if resolution_meters is not None else auto_resolution_meters(size_km))
    params: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp,
        "size_km": size_km,
        "spectral_bands": bands,
        "window_seconds": int(window_days * 24 * 60 * 60),
        "return_type": "array",
        "resolution_meters": res,
    }
    url = f"{base}/data/image/sentinel"
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise SimSatError(f"SimSat unreachable at {url}: {e}") from e
    if r.status_code != 200:
        raise SimSatError(f"SimSat {r.status_code}: {r.text[:200]}")
    try:
        payload = r.json()
    except ValueError as e:
        raise SimSatError(f"SimSat returned non-JSON array payload: {e}") from e
    meta = payload.get("sentinel_metadata") or {}
    if meta.get("image_available") is False or payload.get("image") is None:
        raise SimSatError(f"No Sentinel-2 image available at {lat},{lon} near {timestamp}")
    image = payload["image"]
    array_meta = image.get("metadata") or {}
    shape = tuple(array_meta.get("shape", ()))
    dtype = array_meta.get("dtype", "uint16")
    band_names = list(array_meta.get("bands", []) or bands)
    try:
        raw = base64.b64decode(image["image"])
    except Exception as e:
        raise SimSatError(f"failed to decode base64 array: {e}") from e
    try:
        arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
    except Exception as e:
        raise SimSatError(f"failed to reshape array (shape={shape}, dtype={dtype}): {e}") from e
    return SentinelArray(array=arr, band_names=band_names, metadata=meta)


# Index definitions: (index_name) -> (numerator_band, denominator_band) such that
# index = (a - b) / (a + b). All standard spectral-ratio indices fit this shape.
INDEX_DEFINITIONS: dict[str, tuple[str, str]] = {
    "NDVI":  ("nir", "red"),
    "NDWI":  ("green", "nir"),
    "MNDWI": ("green", "swir16"),
    "NBR":   ("nir", "swir22"),
    "NDBI":  ("swir16", "nir"),
    "NDSI":  ("green", "swir16"),
}
