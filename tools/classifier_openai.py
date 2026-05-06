"""classify_change for any OpenAI-compatible chat-completions endpoint
(llama.cpp llama-server, vLLM, Ollama, etc).

Same input/output contract as classifier_gemini.make_classify_change so the
server can swap providers transparently.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable

import requests

from .classifier_gemini import CLASSIFY_PROMPT, ClassifyResult


def _data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _call_openai_classify(before_path: str, after_path: str, base_url: str,
                          model: str, api_key: str = "dummy",
                          timeout: float = 120.0) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "BEFORE (previous satellite pass):"},
                {"type": "image_url", "image_url": {"url": _data_url(before_path)}},
                {"type": "text", "text": "AFTER (current pass over the same location):"},
                {"type": "image_url", "image_url": {"url": _data_url(after_path)}},
                {"type": "text", "text": (
                    CLASSIFY_PROMPT
                    + "\n\nReturn ONLY a JSON object matching this schema (no commentary):\n"
                    + '{"classes":[{"name":"<class>","confidence":<0..1>}], "bboxes":[[x,y,w,h], ...]}'
                )},
            ],
        }],
    }
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {"error": f"OpenAI-compat call failed: {type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    try:
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as e:
        return {"error": f"unexpected response shape: {type(e).__name__}: {e}",
                "raw_preview": r.text[:400]}
    # Some servers wrap output in markdown fences; strip if present.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]

    # Try strict schema first (Gemini-shaped output). Fine-tuned models
    # (e.g. LFM2.5-VL wildfire LoRA) return custom labels like "brown_land"
    # and may omit / float-ify bboxes — fall back to a loose parse.
    try:
        parsed = ClassifyResult.model_validate_json(cleaned)
        result = parsed.model_dump()
    except Exception:
        try:
            raw = json.loads(cleaned)
        except Exception as e:
            return {"error": f"failed to parse model output: {type(e).__name__}: {e}",
                    "raw_preview": text[:400]}
        classes_out: list[dict[str, Any]] = []
        for c in raw.get("classes") or []:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("label") or "unknown"
            try:
                conf = float(c.get("confidence", c.get("score", 0.0)))
            except (TypeError, ValueError):
                conf = 0.0
            classes_out.append({"name": str(name), "confidence": conf})
        bboxes_out: list[list[int]] = []
        for box in raw.get("bboxes") or []:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    bboxes_out.append([int(round(float(v))) for v in box])
                except (TypeError, ValueError):
                    continue
        result = {"classes": classes_out, "bboxes": bboxes_out, "loose_parsed": True}
    result["source"] = "openai_compat"
    result["model"] = model
    return result


def make_classify_change(before_path: str, after_path: str, *,
                         base_url: str, model: str,
                         api_key: str = "dummy") -> Callable[..., dict[str, Any]]:
    def classify_change(**_kwargs) -> dict[str, Any]:
        return _call_openai_classify(before_path, after_path, base_url, model, api_key)
    return classify_change
