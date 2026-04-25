"""Interim classify_change backed by Gemini Vision (Structured Output).

Uses Gemini's structured output feature (response_mime_type + Pydantic
response_schema) so the output is guaranteed well-formed JSON conforming
to the ClassifyResult schema. No manual JSON parsing.

The final system replaces this with LFM2-VL classifier LoRA; the tool
schema and response shape stay the same.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


ChangeClass = Literal[
    "no_change",
    "flood",
    "fire",
    "deforestation",
    "urban_growth",
    "volcanic_activity",
    "earthquake_damage",
    "cloud",
]


class ClassItem(BaseModel):
    name: ChangeClass
    confidence: float = Field(ge=0.0, le=1.0)


class ClassifyResult(BaseModel):
    classes: list[ClassItem] = Field(
        description="Candidate change classes with confidences, sorted desc. 1-3 entries."
    )
    bboxes: list[list[int]] = Field(
        default_factory=list,
        description="0-3 bboxes [x, y, w, h] in pixel coords of the AFTER image, highlighting suspicious regions.",
    )


CLASSIFY_PROMPT = """You are a Sentinel-2 CHANGE-DETECTION classifier running onboard a satellite.

Your job: compare BEFORE (previous pass) and AFTER (current pass) of the SAME location, and decide WHAT CHANGED between them. Not "what is in the image" — "what is different between the two".

Reasoning procedure (follow in order):
1. Identify regions visible in BOTH images (i.e. not obscured by clouds in either).
2. In those jointly-visible regions, compare: is there meaningful land-surface change?
   - If yes, classify the change type (flood / fire / deforestation / urban_growth / volcanic_activity / earthquake_damage).
   - If no, output "no_change" as the primary class.
3. Only use "cloud" as the primary class when the AFTER image is so obscured that NO comparison is possible at all (>80% of the scene unusable).
4. If clouds cover only part of AFTER but the visible area shows no change → primary = "no_change", "cloud" can be a secondary (lower-confidence) class.

Important:
- A half-clouded scene is NOT "95% cloud". If the un-clouded half clearly shows no change, say "no_change" with high confidence.
- Be honest about uncertainty. Don't over-commit.
- bboxes: 0-3 rectangles [x, y, w, h] in AFTER image pixel coords, pointing at the CHANGE location (not the cloud location). Empty if the primary class is "no_change" or if change is diffuse.
"""


def _image_part(path: str):
    from google.genai import types
    p = Path(path)
    data = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime)


def _call_gemini_classify(before_path: str, after_path: str, provider) -> dict[str, Any]:
    from google.genai import types

    contents = [
        types.Content(role="user", parts=[
            types.Part(text="BEFORE (previous satellite pass):"),
            _image_part(before_path),
            types.Part(text="AFTER (current pass over the same location):"),
            _image_part(after_path),
            types.Part(text=CLASSIFY_PROMPT),
        ])
    ]
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ClassifyResult,
        # Thinking helps here — comparison reasoning is exactly what we want.
        # Budget modest so the structured JSON still fits.
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_budget=2048),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    try:
        resp = provider.client.models.generate_content(
            model=provider.model,
            contents=contents,
            config=cfg,
        )
    except Exception as e:
        return {"error": f"Gemini call failed: {type(e).__name__}: {e}"}

    # Prefer response.parsed (auto-parsed Pydantic instance)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, ClassifyResult):
        result = parsed.model_dump()
    else:
        # Fallback: parse text manually
        text = getattr(resp, "text", "") or ""
        try:
            result = ClassifyResult.model_validate_json(text).model_dump()
        except Exception as e:
            return {
                "error": f"failed to parse structured output: {type(e).__name__}: {e}",
                "raw_preview": text[:400],
            }

    result["source"] = "gemini"
    result["model"] = getattr(provider, "model", "?")
    return result


def make_classify_change(
    before_path: str,
    after_path: str,
    provider,
) -> Callable[..., dict[str, Any]]:
    """Factory: returns a classify_change callable bound to the image pair + provider.

    If provider is None, returns a stub (stable output) so the UI keeps working.
    """
    if provider is None:
        def classify_change_stub(**_kwargs) -> dict[str, Any]:
            return {
                "classes": [{"name": "no_change", "confidence": 0.0}],
                "bboxes": [],
                "source": "stub",
                "note": "no LLM provider configured",
            }
        return classify_change_stub

    def classify_change(**_kwargs) -> dict[str, Any]:
        return _call_gemini_classify(before_path, after_path, provider)

    return classify_change
