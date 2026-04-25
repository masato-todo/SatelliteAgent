"""Pixel-level image quality stats.

Computed directly from the PNG (no STAC / ground metadata).
Designed to be cheap enough to run onboard: numpy only, <10 ms per image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Bump this when the stats formula changes so the server backfills stale sidecars.
STATS_SCHEMA = "v3_nodata"


def assess_image_quality_impl(image_path: str) -> dict[str, Any]:
    """Compute pixel stats from a saved PNG (Tier 1 + Tier 2 only).

    Returns a JSON-serializable dict. Values are intentionally raw numbers
    plus a boolean `usable` so the UI and agent can reason about them.
    """
    p = Path(image_path)
    if not p.exists() or p.stat().st_size == 0:
        return {"error": "image missing or empty"}
    try:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if arr.size == 0:
        return {"error": "empty array"}

    h, w, _ = arr.shape
    arr_f = arr.astype(np.float32)

    # Tier 0: data coverage. Pixels that are exactly (0,0,0) on all channels
    # are SimSat / STAC fill (tile boundary, no S2 coverage at that bbox edge).
    # A high nodata_fraction means the AOI partially missed the satellite swath
    # and the image is unusable as evidence regardless of cloud.
    nodata_fraction = float((arr.sum(axis=-1) == 0).mean())

    # Tier 1: brightness / darkness
    brightness_mean = float(arr_f.mean())
    brightness_std  = float(arr_f.std())
    white_fraction  = float((arr.min(axis=-1) > 240).mean())
    dark_fraction   = float((arr.max(axis=-1) < 30).mean())
    dynamic_range   = float(arr.max()) - float(arr.min())

    # HSV-style saturation: normalized by brightness (max channel).
    # Dark forest/water stay "saturated" here because (max-min)/max is large
    # even when absolute diffs are small. Only truly achromatic pixels hit <0.1.
    max_c = arr_f.max(axis=-1)
    min_c = arr_f.min(axis=-1)
    saturation = np.where(max_c > 1e-6, (max_c - min_c) / np.maximum(max_c, 1e-6), 0.0)

    # Cloud-like = BRIGHT AND LOW-SATURATION (both required).
    pixel_brightness = arr_f.mean(axis=-1)
    bright_mask  = pixel_brightness > 180
    low_sat_mask = saturation < 0.10
    cloud_like_fraction = float((bright_mask & low_sat_mask).mean())

    # Tier 2: edge density via simple gradient on the mean channel
    gray = pixel_brightness
    gy, gx = np.gradient(gray)
    edge_mag = np.hypot(gx, gy)
    edge_density = float(edge_mag.mean())

    # Cloud proxy: max of saturated-white pixels and the bright+low-sat heuristic.
    cloud_proxy = float(max(white_fraction, cloud_like_fraction))

    # Usability heuristic. nodata wins over cloud — a half-empty PNG is useless
    # for visual analysis even if the visible part is clear.
    usable = (
        nodata_fraction < 0.20
        and cloud_proxy < 0.5
        and dark_fraction < 0.10
        and edge_density > 2.0
    )

    return {
        "_schema": STATS_SCHEMA,
        "size": [w, h],
        "nodata_fraction": round(nodata_fraction, 3),
        "brightness_mean": round(brightness_mean, 2),
        "brightness_std":  round(brightness_std, 2),
        "dynamic_range":   round(dynamic_range, 1),
        "white_fraction":  round(white_fraction, 3),
        "dark_fraction":   round(dark_fraction, 3),
        "cloud_like_fraction": round(cloud_like_fraction, 3),
        "edge_density":    round(edge_density, 2),
        "cloud_proxy":     round(cloud_proxy, 3),
        "usable": usable,
    }


def summary_label(stats: dict[str, Any]) -> str:
    """Short human-readable label for a stats dict (UI use)."""
    if not stats or "error" in stats:
        return f"stats error: {stats.get('error', 'n/a')}" if stats else "no stats"
    cp = stats.get("cloud_proxy")
    ed = stats.get("edge_density")
    usable = stats.get("usable", False)
    tag = "clear" if (cp is not None and cp < 0.2) else ("ok" if (cp is not None and cp < 0.5) else "CLOUDY")
    parts = []
    if cp is not None:  parts.append(f"cloud_proxy {cp:.2f} ({tag})")
    if ed is not None:  parts.append(f"edges {ed:.1f}")
    if not usable:      parts.append("NOT usable")
    return " · ".join(parts)
