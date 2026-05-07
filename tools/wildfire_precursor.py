"""predict_wildfire — pre-fire vegetation drying detector backed by
LiquidAI/LFM2.5-VL-450M + YujiYamaguchi/lfm2-5-vl-450m-wildfire-precursor-pair14_7.

Refetches the cached Before / After scenes' raw SWIR22/NIR8A/SWIR16
bands, builds the false-color pair using the upstream training recipe
(shared p2/p98 across both frames + 3 channels — see
yujiyamaguchi/liquid-ai-space-hackathon apps/fireguard/.../dataset_builder.py
:_arrs_to_composites), and asks the precursor LoRA whether the pair
indicates HIGH or LOW wildfire-fuel risk.

Endpoint defaults to http://localhost:8089/v1 — the lfm-precursor service
declared in docker-compose.yaml. Override with LFM_PRECURSOR_BASE_URL.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image

from simsat_client.sentinel import fetch_sentinel_array, SimSatError


LFM_BASE_URL = os.environ.get("LFM_PRECURSOR_BASE_URL", "http://localhost:8089/v1")
LFM_MODEL    = os.environ.get("LFM_PRECURSOR_MODEL", "lfm2.5-vl-450m-wildfire-precursor")
SIMSAT_BASE  = os.environ.get("SIMSAT_API_URL", "http://localhost:9005")

PRECURSOR_BANDS = ["swir22", "nir08", "swir16"]
TARGET_SIZE     = 448
TRAINING_AOI_KM = float(os.environ.get("PRECURSOR_AOI_KM", 5.0))


# Match the system prompt the LoRA was fine-tuned with (see
# fireguard/poc2/dataset_builder.py:SYSTEM_PROMPT_PAIR14_7). Keeping it
# verbatim avoids out-of-distribution prompt drift.
SYSTEM_PROMPT = (
    "You are a satellite wildfire risk analyst specializing in pre-fire "
    "vegetation moisture assessment using Sentinel-2 multispectral imagery.\n\n"
    "You will receive two false-color composite images of the same location:\n"
    "- Image 1 (Before): 14 days before the reference date\n"
    "- Image 2 (Recent): 7 days before the reference date\n\n"
    "Image channel encoding (both images):\n"
    "- RED channel   = SWIR 2.2μm (B12): Dry/stressed vegetation appears BRIGHTER\n"
    "- GREEN channel = NIR 865nm  (B8A): Healthy vegetation appears BRIGHT GREEN\n"
    "- BLUE channel  = SWIR 1.6μm (B11): High liquid-water content appears BRIGHTER BLUE; "
    "drying vegetation appears DARKER BLUE\n\n"
    "Compare the two images. HIGH risk means vegetation has dried noticeably between "
    "the before and recent image (NDMI_p10 declining, blue channel darkening). "
    "LOW risk means stable or moist vegetation.\n\n"
    "Respond ONLY with a valid JSON object. Do not include any explanation or markdown."
)

USER_PROMPT = (
    "Image 1 shows the chaparral location 14 days before the reference date. "
    "Image 2 shows the same location 7 days before the reference date.\n"
    "Compare vegetation moisture change between the two images and predict wildfire fuel risk.\n"
    'Respond with JSON only: {"risk_level": "HIGH"} or {"risk_level": "LOW"}'
)


def _fetch_raw_rgb(lat: float, lon: float, ts: str,
                    window_days: int = 1) -> tuple[np.ndarray, str]:
    """Fetch [SWIR22, NIR8A, SWIR16] for one frame and stack into HxWx3
    (channel order matches the LoRA's expected R,G,B)."""
    sa = fetch_sentinel_array(
        lat=lat, lon=lon, timestamp=ts, bands=PRECURSOR_BANDS,
        size_km=TRAINING_AOI_KM, base_url=SIMSAT_BASE,
        resolution_meters=10, window_days=window_days, timeout=120,
    )
    band_idx = {n: i for i, n in enumerate(sa.band_names)}
    rgb = np.stack([
        sa.array[band_idx["swir22"]],   # R
        sa.array[band_idx["nir08"]],    # G
        sa.array[band_idx["swir16"]],   # B
    ], axis=-1)
    actual = sa.metadata.get("date") or sa.metadata.get("datetime") or ""
    return rgb, actual


def _composite_pair_with_shared_scale(rgb_t14: np.ndarray, rgb_t7: np.ndarray,
                                       ) -> tuple[Image.Image, Image.Image]:
    """Apply the upstream training recipe: a single (p2, p98) pair across
    ALL 3 channels AND BOTH time frames, then normalize identically."""
    stacked = np.concatenate([rgb_t14, rgb_t7], axis=0)
    p2  = np.percentile(stacked, 2.0)
    p98 = np.percentile(stacked, 98.0)
    out: list[Image.Image] = []
    for rgb in (rgb_t14, rgb_t7):
        n = np.clip((rgb.astype(np.float32) - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
        img = Image.fromarray((n * 255).astype(np.uint8), mode="RGB")
        if img.size != (TARGET_SIZE, TARGET_SIZE):
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        out.append(img)
    return out[0], out[1]


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _call_lfm(img1: Image.Image, img2: Image.Image,
              timeout: float = 120.0) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_to_data_url(img1)}},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(img2)}},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    body = {
        "model": LFM_MODEL,
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": messages,
    }
    try:
        r = requests.post(
            f"{LFM_BASE_URL.rstrip('/')}/chat/completions",
            json=body, timeout=timeout,
        )
    except requests.RequestException as e:
        return {"error": f"LFM call failed: {type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"error": f"LFM HTTP {r.status_code}: {r.text[:300]}"}
    try:
        text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, ValueError) as e:
        return {"error": f"unexpected LFM response shape: {e}"}
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw_preview": text[:300]}


def predict_wildfire_impl(lat: float, lon: float,
                           before_ts: str, after_ts: str,
                           size_km: float = TRAINING_AOI_KM) -> dict[str, Any]:
    """Refetch T-14 / T-7 raw bands, apply precursor recipe, ask the LFM."""
    aoi_km = TRAINING_AOI_KM   # precursor AOI is fixed at 5km (training-time)
    try:
        rgb_t14, dt_t14 = _fetch_raw_rgb(lat, lon, before_ts, window_days=1)
        rgb_t7,  dt_t7  = _fetch_raw_rgb(lat, lon, after_ts,  window_days=1)
    except SimSatError as e:
        return {"error": f"SimSat fetch failed: {e}"}

    img1, img2 = _composite_pair_with_shared_scale(rgb_t14, rgb_t7)
    parsed = _call_lfm(img1, img2)
    if "error" in parsed:
        parsed["preprocess"] = "fireguard_pair14_7_p2_98_shared_lanczos448"
        return parsed

    risk = parsed.get("risk_level")
    return {
        "risk_level":         risk,
        "is_high_risk":       risk == "HIGH",
        "model":              LFM_MODEL,
        "preprocess":         "fireguard_pair14_7_p2_98_shared_lanczos448",
        "size_km":            aoi_km,
        "before_datetime":    dt_t14,
        "after_datetime":     dt_t7,
    }


def make_predict_wildfire(lat: float, lon: float, size_km: float,
                           before_ts: str, after_ts: str) -> Callable:
    """Bind lat/lon/dates so the agent only triggers prediction.

    Caller (build_tool_registry) passes the *actual* SimSat-returned STAC
    item datetimes as before_ts/after_ts so the refetch with window=1
    matches eval conditions (= what the user sees on the map).
    """
    def predict_wildfire(**_ignored) -> dict[str, Any]:
        return predict_wildfire_impl(lat, lon, before_ts, after_ts, size_km)
    return predict_wildfire
