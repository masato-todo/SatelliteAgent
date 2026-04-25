"""Real implementations for vision tools.

Phase 1 provides:
- zoom_in: crop + LANCZOS upscale from the current image pair.
- classify_change / fetch_band: still stubbed in tools.stubs (real versions land in Phase 2).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from PIL import Image


ZOOM_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scenarios"
ZOOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_BBOX_SIDE = 32
DEFAULT_OUTPUT_SIZE = 512


def _square_crop_box(
    bbox: list[int], img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Return a PIL-compatible (x0, y0, x1, y1) square crop centered on bbox.

    The square side = max(w, h). Clipped to image bounds; the resulting
    rectangle may be non-square if clipping reduces one dimension.
    """
    x, y, w, h = bbox
    side = max(w, h)
    cx = x + w / 2.0
    cy = y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return x0, y0, x1, y1


def zoom_in_impl(
    bbox: list[int],
    before_path: str,
    after_path: str,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> dict[str, Any]:
    """Crop the given bbox from both images (squared, clipped) and upscale to output_size.

    Deterministic: same inputs -> same output (different uuid per call for file isolation).
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"error": "bbox must be [x, y, w, h]"}
    try:
        bbox_int = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return {"error": f"bbox values must be integers, got {bbox}"}

    try:
        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")
    except FileNotFoundError as e:
        return {"error": f"image not found: {e}"}
    except Exception as e:
        return {"error": f"failed to load images: {type(e).__name__}: {e}"}

    img_w, img_h = after.size
    x0, y0, x1, y1 = _square_crop_box(bbox_int, img_w, img_h)
    w = x1 - x0
    h = y1 - y0

    if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
        return {
            "error": (
                f"bbox too small after squaring and clipping: {w}x{h}, "
                f"minimum {MIN_BBOX_SIDE}x{MIN_BBOX_SIDE}"
            )
        }

    zoomed_before = before.crop((x0, y0, x1, y1)).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    zoomed_after = after.crop((x0, y0, x1, y1)).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )

    uid = uuid.uuid4().hex[:10]
    before_key = f"zoom_before_{uid}"
    after_key = f"zoom_after_{uid}"
    zoomed_before.save(ZOOM_OUTPUT_DIR / f"{before_key}.png")
    zoomed_after.save(ZOOM_OUTPUT_DIR / f"{after_key}.png")

    zoom_ratio = round(output_size / max(w, h), 2)
    return {
        "zoomed_before_key": before_key,
        "zoomed_after_key": after_key,
        "crop_pixel_bbox": [x0, y0, w, h],
        "original_image_size": [img_w, img_h],
        "zoom_ratio": zoom_ratio,
    }


def make_zoom_in(before_path: str, after_path: str) -> Callable[..., dict[str, Any]]:
    """Factory binding a zoom_in tool to a specific image pair for the current session."""
    def zoom_in(bbox: list[int], **_ignored) -> dict[str, Any]:
        return zoom_in_impl(bbox, before_path, after_path)
    return zoom_in


def capture_crop_impl(
    bbox: list[int],
    after_path: str,
    side: str = "after",
    before_path: str | None = None,
) -> dict[str, Any]:
    """Crop the given bbox from the After image (or Before if side=before).

    Unlike zoom_in this preserves the native resolution (no upscaling) — the
    crop is meant as concrete visual evidence to attach to a submitted report.
    Bbox is in pixel coords of the After image (matches the user's draw on the
    After Leaflet map).
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"error": "bbox must be [x, y, w, h]"}
    try:
        bbox_int = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return {"error": f"bbox values must be integers, got {bbox}"}

    src_path = before_path if (side == "before" and before_path) else after_path
    try:
        img = Image.open(src_path).convert("RGB")
    except FileNotFoundError as e:
        return {"error": f"image not found: {e}"}
    except Exception as e:
        return {"error": f"failed to load image: {type(e).__name__}: {e}"}

    x, y, w, h = bbox_int
    img_w, img_h = img.size
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {"error": f"bbox is empty after clipping: {x1 - x0}x{y1 - y0}"}

    crop = img.crop((x0, y0, x1, y1))
    uid = uuid.uuid4().hex[:10]
    key = f"crop_{side}_{uid}"
    out_path = ZOOM_OUTPUT_DIR / f"{key}.png"
    crop.save(out_path)

    return {
        "image_key": key,
        "side": side,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "original_image_size": [img_w, img_h],
        "size_bytes": out_path.stat().st_size,
    }


def make_capture_crop(before_path: str, after_path: str) -> Callable[..., dict[str, Any]]:
    def capture_crop(bbox: list[int], side: str = "after", **_ignored) -> dict[str, Any]:
        return capture_crop_impl(bbox, after_path, side=side, before_path=before_path)
    return capture_crop
