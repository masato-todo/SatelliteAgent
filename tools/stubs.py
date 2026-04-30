"""Phase 1 stubs: plausible fixtures for every tool.

All shapes must conform to `tools/schema.py`. Real implementations will
replace these one by one as Phase 2 progresses.
"""
from __future__ import annotations

from typing import Any, Callable


def classify_change(image_before: str, image_after: str) -> dict[str, Any]:
    return {
        "classes": [
            {"flood": 0.62},
            {"cloud": 0.28},
            {"no_change": 0.10},
        ],
        "bboxes": [[120, 80, 60, 40]],
    }


def fetch_band(lat: float, lon: float, band: str) -> dict[str, Any]:
    return {"image_path": f"stub_{band}.png", "lat": lat, "lon": lon, "band": band}


def zoom_in(bbox: list[int], **_ignored) -> dict[str, Any]:
    # Legacy stub; in the live app this is replaced per-request by
    # tools.vision.make_zoom_in(before_path, after_path).
    return {"image_path": "stub_zoom.png", "bbox": bbox}


def get_region_info(lat: float, lon: float) -> dict[str, Any]:
    return {
        "region": "Sylhet",
        "country": "BD",
        "populated": True,
        "infra_nearby": ["road", "settlement"],
    }


def get_history(lat: float, lon: float, days: int = 30) -> list[dict[str, Any]]:
    return []


def compute_area(bbox: list[int]) -> dict[str, float]:
    _, _, w, h = bbox
    return {"area_km2": round(w * h * 0.0001, 2)}


def check_downlink_budget() -> dict[str, int]:
    return {"remaining_bytes": 4_200_000, "window_sec_left": 180}


def estimate_size(report_id: str, with_image: bool) -> dict[str, int]:
    return {"bytes": 420_000 if with_image else 2_000}


_REPORT_COUNTER = [0]


def compose_report(
    change_type: str,
    urgency: int,
    description: str,
    attach_image: bool = False,
) -> dict[str, str]:
    _REPORT_COUNTER[0] += 1
    return {"report_id": f"r-{_REPORT_COUNTER[0]:04d}"}


def submit_to_ground(
    report_id: str,
    reason: str,
    attach_image: bool = False,
    attach_crop_key: str | None = None,
    **_extra,
) -> dict[str, Any]:
    """Transmit a report to ground.

    Args:
        report_id: identifier for this report.
        reason: free-text justification. Cite which spectral index and
            numbers led to the decision (e.g. "NBR delta frac_decrease_strong
            = 0.78, well above burn threshold 0.27 -> wildfire").
    """
    return {
        "status": "ok",
        "report_id": report_id,
        "reason": reason,
        "attached": attach_image,
        "attached_crop_key": attach_crop_key,
    }


def drop(reason: str) -> dict[str, str]:
    """Drop the data without transmitting.

    Args:
        reason: free-text justification. Cite which spectral index and
            numbers led to the no-change decision (e.g. "NBR delta mean
            ~ 0.0, no fire signal; classify_change top class no_change").
    """
    return {"status": "dropped", "reason": reason}


STUB_TOOLS: dict[str, Callable[..., Any]] = {
    "classify_change": classify_change,
    "fetch_band": fetch_band,
    "zoom_in": zoom_in,
    "get_region_info": get_region_info,
    "get_history": get_history,
    "compute_area": compute_area,
    "check_downlink_budget": check_downlink_budget,
    "estimate_size": estimate_size,
    "compose_report": compose_report,
    "submit_to_ground": submit_to_ground,
    "drop": drop,
}