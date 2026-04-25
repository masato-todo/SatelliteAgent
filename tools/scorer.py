"""Spectral change statistics tool — `get_change_stats`.

Given a Before/After Sentinel-2 pair, fetches multi-band arrays, computes
all standard spectral index deltas, and returns purely numerical statistics
per index. NO class judgements — that's the agent's job.

The output includes universal interpretation hints (what each index measures)
but nothing about which disaster class is more likely. This avoids baking
DM3 dataset priors into the tool, so the agent must reason from scratch
based on the numbers + other tools (e.g., get_region_info).

Future: LFM2-VL classifier LoRA can consume this exact stats dict as part
of its prompt, with `classify_change` providing the class verdict.

Index reference:
  - NBR   = (NIR-SWIR)/(NIR+SWIR) — burn detection (USGS Key & Benson 2006)
  - NDVI  = (NIR-Red)/(NIR+Red)   — vegetation health
  - MNDWI = (Green-SWIR)/(Green+SWIR) — water (Xu 2006)
  - NDBI  = (SWIR-NIR)/(SWIR+NIR) — bright-SWIR surfaces (built, bare, char)
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from simsat_client import (
    SimSatError,
    fetch_sentinel_array,
)


REQUIRED_BANDS = ["green", "red", "nir", "swir16"]


# Literature thresholds for "strong" change per index (used to compute the
# frac_strong_{decrease,increase} fields below).
STRONG_THRESHOLDS = {
    "NBR":   0.27,  # Key & Benson 2006: dNBR > 0.27 = moderate-low burn
    "NDVI":  0.20,  # commonly used for "significant vegetation change"
    "MNDWI": 0.30,  # Xu 2006: water apparition threshold
    "NDBI":  0.10,  # subtle by design — built-up changes are slow
}


INTERPRETATION_HINTS = {
    "NBR":   "Strong DECREASE → burn / charred vegetation (SWIR up). Strong INCREASE → vegetation regrowth.",
    "NDVI":  "DECREASE → vegetation loss (any cause: burn, ash, mud, clearing). INCREASE → vegetation gain / seasonal greening.",
    "MNDWI": "INCREASE → new water surface (flood, river expansion). DECREASE → water receded / drained.",
    "NDBI":  "INCREASE → bright-SWIR surface appeared (built-up, bare soil, ash, charred ground — NOT specific to buildings). DECREASE → vegetation or water replaced built/bare surface.",
}


def _normalize_ts(d: str) -> str:
    s = d.strip()
    if "T" not in s:
        s = f"{s}T00:00:00Z"
    elif not s.endswith("Z"):
        s = s + "Z"
    return s


def _normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (a - b) / denom, np.nan)


def _delta_stats(after_idx: np.ndarray, before_idx: np.ndarray, threshold: float) -> dict[str, float] | None:
    delta = after_idx - before_idx
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return None
    return {
        "mean":                 float(finite.mean()),
        "median":               float(np.median(finite)),
        "std":                  float(finite.std()),
        "min":                  float(finite.min()),
        "max":                  float(finite.max()),
        "p10":                  float(np.percentile(finite, 10)),
        "p90":                  float(np.percentile(finite, 90)),
        "frac_strong_decrease": float(np.mean(finite < -threshold)),
        "frac_strong_increase": float(np.mean(finite >  threshold)),
        "threshold_strong":     threshold,
    }


def get_change_stats_impl(
    lat: float,
    lon: float,
    before_ts: str,
    after_ts: str,
    size_km: float = 5.0,
    window_days: int = 10,
) -> dict[str, Any]:
    """Fetch all required bands for Before+After once, compute every spectral
    index delta, and return per-index statistics. No class judgements.
    """
    try:
        b = fetch_sentinel_array(
            lat=lat, lon=lon, timestamp=_normalize_ts(before_ts),
            bands=REQUIRED_BANDS, size_km=size_km, window_days=window_days,
        )
        a = fetch_sentinel_array(
            lat=lat, lon=lon, timestamp=_normalize_ts(after_ts),
            bands=REQUIRED_BANDS, size_km=size_km, window_days=window_days,
        )
    except SimSatError as e:
        return {"error": str(e)}

    bnames = [n.lower() for n in b.band_names]
    anames = [n.lower() for n in a.band_names]

    def grab(arr, names, band):
        try:
            i = names.index(band)
        except ValueError:
            return None
        return arr[i].astype(np.float32)

    bands_b = {k: grab(b.array, bnames, k) for k in REQUIRED_BANDS}
    bands_a = {k: grab(a.array, anames, k) for k in REQUIRED_BANDS}
    if any(v is None for v in bands_b.values()) or any(v is None for v in bands_a.values()):
        return {"error": f"missing required bands. before={bnames} after={anames}"}

    pairs = {
        "NBR":   (_normalized_diff(bands_b["nir"],    bands_b["swir16"]),
                  _normalized_diff(bands_a["nir"],    bands_a["swir16"])),
        "NDVI":  (_normalized_diff(bands_b["nir"],    bands_b["red"]),
                  _normalized_diff(bands_a["nir"],    bands_a["red"])),
        "MNDWI": (_normalized_diff(bands_b["green"],  bands_b["swir16"]),
                  _normalized_diff(bands_a["green"],  bands_a["swir16"])),
        "NDBI":  (_normalized_diff(bands_b["swir16"], bands_b["nir"]),
                  _normalized_diff(bands_a["swir16"], bands_a["nir"])),
    }

    indices: dict[str, Any] = {}
    for name, (idx_b, idx_a) in pairs.items():
        h = min(idx_b.shape[0], idx_a.shape[0])
        w = min(idx_b.shape[1], idx_a.shape[1])
        stats = _delta_stats(idx_a[:h, :w], idx_b[:h, :w], threshold=STRONG_THRESHOLDS[name])
        if stats is not None:
            indices[name] = stats

    return {
        "indices": indices,
        "interpretation_hints": INTERPRETATION_HINTS,
        "before_datetime": (b.metadata or {}).get("datetime"),
        "after_datetime":  (a.metadata or {}).get("datetime"),
        "source": "spectral_stats",
    }


def make_get_change_stats(lat: float, lon: float, size_km: float,
                          before_ts: str, after_ts: str) -> Callable:
    def get_change_stats(**_ignored) -> dict[str, Any]:
        return get_change_stats_impl(lat, lon, before_ts, after_ts, size_km)
    return get_change_stats
