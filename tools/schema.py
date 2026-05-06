"""JSON Schema definitions for all agent tools.

The Orchestrator sees these schemas via its LLM provider's tool-use API.
Stubs and real implementations both must conform to these shapes.

Grouped by category:
    Vision : classify_change, fetch_band, zoom_in
    Context: get_region_info, get_history, compute_area
    Budget : check_downlink_budget, estimate_size
    Action : compose_report, submit_to_ground, drop
"""
from __future__ import annotations

from typing import Any


TOOL_SCHEMAS: list[dict[str, Any]] = [
    # ---- Vision -----------------------------------------------------
    {
        "name": "classify_change",
        "description": (
            "Run the onboard LFM2-VL change classifier on a before/after image pair. "
            "Returns candidate change classes with confidences and bounding boxes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_before": {"type": "string", "description": "Path or handle of earlier image"},
                "image_after": {"type": "string", "description": "Path or handle of later image"},
            },
            "required": ["image_before", "image_after"],
        },
    },
    {
        "name": "fetch_band",
        "description": (
            "Fetch a single Sentinel-2 spectral band as a grayscale image for "
            "the current location. Lat/lon/size_km/timestamp are bound server-side; "
            "just pick a band and which side (before or after). "
            "Useful to inspect SWIR (swir16/swir22) for burn scars, NIR (nir) for "
            "vegetation & water contrast, or rededge bands for vegetation stress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "band": {
                    "type": "string",
                    "enum": [
                        "coastal", "blue", "green", "red",
                        "rededge1", "rededge2", "rededge3",
                        "nir", "nir08", "nir09",
                        "swir16", "swir22",
                        "aot", "scl", "visual", "wvp",
                    ],
                },
                "which": {"type": "string", "enum": ["before", "after"], "default": "after"},
            },
            "required": ["band"],
        },
    },
    {
        "name": "false_color",
        "description": (
            "Build an RGB false-color composite from any 3 Sentinel-2 bands for the "
            "current location. Useful for visual interpretation: "
            "nir-red-green (vegetation), swir22-nir-red (burn severity), "
            "swir16-nir-blue (urban vs vegetation), etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": (
                        "Three band names mapped to R, G, B channels in that order. "
                        "Valid: coastal, blue, green, red, rededge1-3, nir, nir08, "
                        "nir09, swir16, swir22, aot, scl, visual, wvp."
                    ),
                },
                "which": {"type": "string", "enum": ["before", "after"], "default": "after"},
            },
            "required": ["bands"],
        },
    },
    {
        "name": "compute_index",
        "description": (
            "Compute a standard spectral index from Sentinel-2 bands and return a "
            "pseudocolor map. Supported indices:\n"
            "- NDVI  (vegetation)         = (nir - red)       / (nir + red)\n"
            "- NDWI  (water, veg-based)   = (green - nir)     / (green + nir)\n"
            "- MNDWI (water, urban)       = (green - swir16)  / (green + swir16)\n"
            "- NBR   (burn ratio)         = (nir - swir22)    / (nir + swir22)\n"
            "- NDBI  (built-up)           = (swir16 - nir)    / (swir16 + nir)\n"
            "- NDSI  (snow)               = (green - swir16)  / (green + swir16)\n"
            "Returns a PNG plus min/max/mean statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "enum": ["NDVI", "NDWI", "MNDWI", "NBR", "NDBI", "NDSI"],
                },
                "which": {"type": "string", "enum": ["before", "after"], "default": "after"},
            },
            "required": ["index"],
        },
    },
    {
        "name": "zoom_in",
        "description": (
            "Zoom into a suspicious bounding box to re-examine the region at higher effective "
            "resolution. Returns 512x512 square crops of both before and after images upscaled "
            "via LANCZOS, so you can compare the area in detail. Use after classify_change "
            "returns a bbox you want to look at more closely, or whenever you need a finer "
            "look at a specific part of the current scene. Minimum bbox side after squaring is "
            "32 pixels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "[x, y, w, h] in pixel coords of the current after-image. "
                        "Will be squared to max(w,h) centered, and clipped to image bounds."
                    ),
                },
            },
            "required": ["bbox"],
        },
    },
    # ---- Context ----------------------------------------------------
    {
        "name": "get_region_info",
        "description": (
            "Return region/country/admin info for the CURRENT scene. The "
            "coordinates are bound server-side from the fetch request — "
            "no arguments needed and you cannot override them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_history",
        "description": (
            "Return past onboard reports for the CURRENT location within "
            "`days`. Coordinates are bound server-side."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 30},
            },
        },
    },
    {
        "name": "compute_area",
        "description": "Compute km² for a pixel-space bounding box.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["bbox"],
        },
    },
    # ---- Budget -----------------------------------------------------
    {
        "name": "check_downlink_budget",
        "description": "Return remaining downlink bytes and seconds until window closes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "estimate_size",
        "description": "Estimate bytes a composed report will consume, with or without image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "with_image": {"type": "boolean"},
            },
            "required": ["report_id", "with_image"],
        },
    },
    # ---- Action -----------------------------------------------------
    {
        "name": "compose_report",
        "description": "Create a report. Returns a report_id. Does not transmit yet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "change_type": {"type": "string"},
                "urgency": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"},
                "attach_image": {"type": "boolean", "default": False},
            },
            "required": ["change_type", "urgency", "description"],
        },
    },
    {
        "name": "submit_to_ground",
        "description": (
            "Transmit a composed report to the ground station. Terminal action. "
            "`reason` MUST cite which spectral index/numbers led to the decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string",
                              "description": "The id returned by compose_report. Required."},
                "reason": {"type": "string",
                           "description": "Why we are transmitting (cite indices/numbers)."},
                "attach_image": {"type": "boolean", "default": True},
                "crop_key": {"type": "string",
                             "description": "Optional zoom_in crop key to attach."},
            },
            "required": ["report_id", "reason", "attach_image"],
        },
    },
    {
        "name": "drop",
        "description": "Discard everything; no transmission. Terminal action. `reason` MUST justify the no-change decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string",
                           "description": "Why we are not transmitting (cite indices/numbers)."},
            },
            "required": ["reason"],
        },
    },
]


TERMINAL_TOOLS: frozenset[str] = frozenset({"submit_to_ground", "drop"})
"""Tools that end the ReAct loop when called."""