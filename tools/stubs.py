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


def get_history(days: int = 30, **_ignored) -> list[dict[str, Any]]:
    return []


def compute_area(bbox: list[int]) -> dict[str, float]:
    _, _, w, h = bbox
    return {"area_km2": round(w * h * 0.0001, 2)}


def check_downlink_budget() -> dict[str, int]:
    return {"remaining_bytes": 4_200_000, "window_sec_left": 180}


def estimate_size(report_id: str, with_image: bool) -> dict[str, int]:
    return {"bytes": 420_000 if with_image else 2_000}


_REPORT_COUNTER = [0]
# Per-process registry of compose_report() outputs so submit_to_ground()
# can verify the agent isn't fabricating report_ids.
#
# HACK: in-memory only. Lost on server restart and not shared across
# uvicorn workers. For multi-worker / persistent setups, replace with a
# sidecar JSON or sqlite store keyed by report_id.
# TODO: also persist the composed payload alongside the trace so that
# submitted reports can be inspected after the fact.
_COMPOSED_REPORTS: dict[str, dict[str, Any]] = {}


def compose_report(
    change_type: str,
    urgency: int,
    description: str,
    attach_image: bool = False,
) -> dict[str, str]:
    _REPORT_COUNTER[0] += 1
    rid = f"r-{_REPORT_COUNTER[0]:04d}"
    _COMPOSED_REPORTS[rid] = {
        "change_type": change_type,
        "urgency": int(urgency),
        "description": description,
        "attach_image": bool(attach_image),
    }
    return {"report_id": rid}


def submit_to_ground(
    report_id: str,
    reason: str,
    attach_image: bool = False,
    attach_crop_key: str | None = None,
    **_extra,
) -> dict[str, Any]:
    """Transmit a report to ground.

    Args:
        report_id: identifier returned by an earlier compose_report() call.
            Submitting an unknown id is rejected — the agent must compose
            before submitting.
        reason: free-text justification. Cite which spectral index and
            numbers led to the decision (e.g. "NBR delta frac_decrease_strong
            = 0.78, well above burn threshold 0.27 -> wildfire").
    """
    if report_id not in _COMPOSED_REPORTS:
        return {
            "status": "error",
            "error": (
                f"unknown report_id '{report_id}'. Call compose_report() "
                "first and reuse the report_id it returns."
            ),
            "known_report_ids": sorted(_COMPOSED_REPORTS.keys())[-5:],
        }
    composed = _COMPOSED_REPORTS[report_id]
    return {
        "status": "ok",
        "report_id": report_id,
        "reason": reason,
        "attached": attach_image,
        "attached_crop_key": attach_crop_key,
        "composed": composed,
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