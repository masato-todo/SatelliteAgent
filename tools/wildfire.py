"""detect_wildfire — single-image wildfire detector backed by
LiquidAI/LFM2.5-VL-450M + YujiYamaguchi wildfire LoRA.

Preprocessing must match FireEdge's training pipeline (model_report.md §4):
  - Bands: SWIR22 (B12), SWIR16 (B11), NIR (B08); RGB = (SWIR22, SWIR16, NIR)
  - Percentile clip 2-98%, **shared scale across SWIR22+SWIR16** (preserves
    burn-scar red), **NIR independent**.
  - Lanczos resize to 448×448 (LFM 2.5-VL vision-encoder native).
The factory needs (lat, lon, size_km, before_ts, after_ts).
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image

from simsat_client.sentinel import fetch_sentinel_array, SimSatError


# OpenAI-compat endpoint of the LFM wildfire docker container.
LFM_BASE_URL = os.environ.get("LFM_WILDFIRE_BASE_URL", "http://localhost:8085/v1")
LFM_MODEL    = os.environ.get("LFM_WILDFIRE_MODEL", "lfm2.5-vl-450m-wildfire")
SIMSAT_BASE  = os.environ.get("SIMSAT_API_URL", "http://localhost:9005")

# Channel encoding the LoRA was fine-tuned on.
FIRE_BANDS  = ["swir22", "swir16", "nir"]
TARGET_SIZE = 448  # LFM 2.5-VL vision encoder native input

# FireEdge training was done at AOI 5km × 5km (model_report.md §4).
# Inferring at the same scale matters — at 10km the post-resize spatial
# resolution doubles and small fire patches lose the thermal signature.
TRAINING_AOI_KM = float(os.environ.get("LFM_WILDFIRE_AOI_KM", 5.0))

DERIVED_DIR = Path(os.environ.get(
    "SAT_DERIVED_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "derived"),
))
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


# Match FireEdge fine-tuning prompt (src/interfaces.py:FIRE_DETECTION_FT_PROMPT).
# The LoRA was trained ONLY on this short form, so any longer / richer prompt
# pushes the model out of distribution and recall collapses.
SYSTEM_PROMPT = None  # No system role used during fine-tuning

USER_PROMPT = (
    "Examine this satellite false-color composite image "
    "(R=SWIR2.2μm, G=SWIR1.6μm, B=NIR).\n\n"
    "Does this scene contain active fire or burn scar?\n"
    'Respond with JSON only: {"fire_detected": true} or {"fire_detected": false}'
)


def _percentile_normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map [lo, hi] -> [0, 1] with clipping. Returns float32."""
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _build_fireedge_composite(swir22: np.ndarray, swir16: np.ndarray,
                               nir: np.ndarray) -> Image.Image:
    """FireEdge model_report.md §4 preprocessing:
    - SWIR22 + SWIR16: shared p2-p98 scale (preserves SWIR ratio = burn red)
    - NIR: independent p2-p98
    - Stack -> uint8 RGB -> Lanczos 448x448
    """
    # Shared SWIR scale
    swir_combined = np.concatenate([swir22.flatten(), swir16.flatten()])
    swir_lo, swir_hi = np.percentile(swir_combined, [2.0, 98.0])
    r = _percentile_normalize(swir22, swir_lo, swir_hi)
    g = _percentile_normalize(swir16, swir_lo, swir_hi)
    # Independent NIR
    nir_lo, nir_hi = np.percentile(nir, [2.0, 98.0])
    b = _percentile_normalize(nir, nir_lo, nir_hi)

    rgb = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
    ], axis=-1)  # H, W, 3
    img = Image.fromarray(rgb, mode="RGB")
    if img.size != (TARGET_SIZE, TARGET_SIZE):
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return img


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _call_lfm(image_data_url: str, timeout: float = 120.0) -> dict[str, Any]:
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": USER_PROMPT},
    ]})
    body = {
        "model": LFM_MODEL,
        "max_tokens": 64,        # response is just {"fire_detected": ...}
        "temperature": 0.1,
        "top_p": 0.9,
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
        data = r.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, ValueError) as e:
        return {"error": f"unexpected LFM response shape: {e}"}
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw_preview": text[:300]}
    return parsed


def _to_change_classes(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Bolt-on classify_change-compatible classes list. Since the
    fine-tuned LoRA only outputs `fire_detected`, we synthesize a
    confidence (1.0 if true, 0.0 if false) so downstream scoring works."""
    if parsed.get("fire_detected"):
        return [{"name": "fire", "confidence": 1.0}]
    return [{"name": "no_change", "confidence": 1.0}]


def detect_wildfire_impl(lat: float, lon: float, timestamp: str,
                          size_km: float = 10.0,
                          window_days: int = 12) -> dict[str, Any]:
    """Fetch raw SWIR22/SWIR16/NIR, apply FireEdge §4 preprocessing,
    send to LFM wildfire LoRA. AOI is forced to TRAINING_AOI_KM (5km) so
    the post-Lanczos spatial resolution matches what the LoRA saw at
    training time. window_days defaults to 12 (FireEdge training value);
    pass 1 if `timestamp` is already the actual sentinel_datetime so SimSat
    returns the exact same STAC item the LoRA saw."""
    aoi_km = TRAINING_AOI_KM
    try:
        sa = fetch_sentinel_array(
            lat=lat, lon=lon, timestamp=timestamp, bands=FIRE_BANDS,
            size_km=aoi_km, base_url=SIMSAT_BASE,
            resolution_meters=10, window_days=window_days, timeout=120,
        )
    except SimSatError as e:
        return {"error": f"SimSat fetch failed: {e}"}

    band_idx = {name: i for i, name in enumerate(sa.band_names)}
    try:
        swir22 = sa.array[band_idx["swir22"]]
        swir16 = sa.array[band_idx["swir16"]]
        nir    = sa.array[band_idx["nir"]]
    except KeyError as e:
        return {"error": f"missing band in SimSat response: {e}"}

    composite = _build_fireedge_composite(swir22, swir16, nir)

    # Save the composite (the actual model input) to the cache so UI / eval
    # can inspect it via /api/image/{composite_image_key}.png
    import hashlib
    key_input = f"wildfire|{lat:.4f}|{lon:.4f}|{timestamp[:10]}|{size_km}"
    composite_key = "wf_" + hashlib.md5(key_input.encode()).hexdigest()[:10]
    composite_path = DERIVED_DIR / f"{composite_key}.png"
    composite.save(composite_path)

    data_url = _image_to_data_url(composite)
    parsed = _call_lfm(data_url)
    if "error" in parsed:
        parsed["preprocess"] = "fireedge_p2_98_lanczos448"
        parsed["composite_image_key"] = composite_key
        return parsed
    # FT_PROMPT only outputs {fire_detected: bool}. Synthesize the
    # FireEdge-style fields so existing eval / UI code keeps working.
    fire = bool(parsed.get("fire_detected"))
    parsed.setdefault("fire_confidence", 1.0 if fire else 0.0)
    parsed.setdefault("smoke_detected", False)
    parsed.setdefault("smoke_confidence", 0.0)
    parsed.setdefault("severity", "MEDIUM" if fire else "NONE")
    parsed.setdefault("description", "active fire detected" if fire else "no fire detected")
    parsed["classes"]             = _to_change_classes(parsed)
    parsed["bboxes"]              = []
    parsed["model"]               = LFM_MODEL
    parsed["source"]              = "lfm25_vl_wildfire"
    parsed["preprocess"]          = "fireedge_p2_98_lanczos448_ft_prompt"
    parsed["composite_image_key"] = composite_key
    parsed["sentinel"] = {
        "datetime":    sa.metadata.get("date") or sa.metadata.get("datetime"),
        "cloud_cover": sa.metadata.get("cloud_cover"),
        "platform":    sa.metadata.get("platform"),
    }
    return parsed


def make_detect_wildfire(lat: float, lon: float, size_km: float,
                          before_ts: str, after_ts: str) -> Callable:
    """Bind lat/lon/dates so the agent only specifies which side.

    Caller (build_tool_registry) passes the *actual* SimSat-returned STAC
    item datetime (cached meta `datetime`) as before_ts/after_ts so that
    detect_wildfire_impl with window_days=1 refetches the exact same scene
    shown on the map. This matches eval_wildfire_hf_simsat.py
    --use-sentinel-datetime --window-days 1.
    """
    def detect_wildfire(which: str = "after", **_ignored) -> dict[str, Any]:
        ts = before_ts if which == "before" else after_ts
        return detect_wildfire_impl(lat, lon, ts, size_km, window_days=1)
    return detect_wildfire
