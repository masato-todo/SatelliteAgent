"""Region info tool — context-bound.

Lat/lon are bound at registry-build time so the VLM cannot pass spurious
coordinates (LFM2.5-VL was observed hallucinating Tokyo coords for a
scene over a different country). The actual reverse-geocoded payload is
captured at fetch time and stored in the cache sidecar; this tool just
surfaces it.
"""
from __future__ import annotations

from typing import Any, Callable


def make_get_region_info(
    lat: float,
    lon: float,
    region_payload: dict[str, Any] | None,
) -> Callable[..., dict[str, Any]]:
    """Return a callable that ignores any kwargs and returns the bound region.

    `region_payload` is whatever was cached during /api/fetch (Nominatim
    /reverse response, normalized). When unavailable we still return the
    coordinates so downstream reasoning is honest about the gap.
    """
    def get_region_info(**_ignored) -> dict[str, Any]:
        out: dict[str, Any] = {"lat": lat, "lon": lon}
        if region_payload:
            out.update(region_payload)
        else:
            out["note"] = "reverse-geocode lookup unavailable for this scene"
        return out

    return get_region_info
