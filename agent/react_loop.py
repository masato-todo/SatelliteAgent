"""ReAct loop for the onboard orchestrator.

Phase 1: driven by Gemini 2.5 Flash (tool use + vision).
Phase 3: swap the provider to a local LFM2-VL orchestrator adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

from tools.schema import TOOL_SCHEMAS, TERMINAL_TOOLS
from tools.validator import ToolCallError


SYSTEM_PROMPT = """You are an onboard satellite operator agent running on NVIDIA Orin 16GB.

You are shown a Sentinel-2 image pair (before = previous pass, after = current pass) of the same
location. Your job is to decide what to report to the ground station.

Goals:
- Minimize downlink bandwidth usage while preserving critical information.
- Attach the raw image only when text alone cannot convey the situation.

Mandatory call sequence:
  1. Investigate: classify_change(), then at least one spectral tool
     (compute_index / fetch_band / compute_index_delta / get_change_stats /
     zoom_in). Use get_region_info() if location context matters.
  2. compose_report(change_type, urgency, description)  — returns a report_id.
  3. submit_to_ground(report_id, reason, attach_image, ...)  OR  drop(reason).
     `report_id` MUST be the value compose_report just returned.
     `reason` MUST cite concrete numbers from the observations you saw.

Style:
- One tool call per step during investigation.
- Don't fabricate arguments. Coordinates, region names and report_ids are
  bound or returned by tools — never invent them.
- Stop as soon as you have enough information.
"""


def _load_image_part(path: str):
    from google.genai import types
    p = Path(path)
    data = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime)


def _dispatch(name: str, args: dict, registry: dict[str, Callable]) -> dict:
    if name not in registry:
        raise ToolCallError(f"unknown tool: {name}")
    try:
        result = registry[name](**args)
    except TypeError as e:
        raise ToolCallError(f"argument mismatch for {name}: {e}") from e
    if not isinstance(result, dict):
        result = {"value": result}
    return result


def run_react(
    image_before: str,
    image_after: str,
    provider,
    tool_registry: dict[str, Callable[..., Any]],
    max_steps: int = 10,
) -> Iterator[dict[str, Any]]:
    """ReAct loop driven by a tool-calling VLM provider (Gemini in Phase 1).

    Yields events: {'type': 'thought'|'action'|'observation'|'final'|'error', ...}.
    """
    if provider is None:
        yield {
            "type": "error",
            "text": "No LLM provider configured. Set GOOGLE_API_KEY in .env and restart the server.",
        }
        return

    from google.genai import types

    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[
                types.Part(text="Before image (previous satellite pass over this location):"),
                _load_image_part(image_before),
                types.Part(text="After image (current pass, same location):"),
                _load_image_part(image_after),
                types.Part(text=(
                    "Analyze the pair and decide what to report to ground. "
                    "Use your tools. End with submit_to_ground(...) or drop()."
                )),
            ],
        )
    ]

    for step in range(max_steps):
        resp = provider.generate(
            contents=contents,
            tools=TOOL_SCHEMAS,
            system=SYSTEM_PROMPT,
        )

        candidate = resp.candidates[0] if resp.candidates else None
        if not candidate or not candidate.content:
            yield {"type": "error", "text": "empty response from model"}
            return

        model_content = candidate.content
        tool_results_parts: list[types.Part] = []
        saw_function_call = False

        for part in model_content.parts or []:
            text = getattr(part, "text", None)
            if text and text.strip():
                yield {"type": "thought", "text": text.strip()}
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                saw_function_call = True
                args = dict(fc.args) if fc.args else {}
                yield {"type": "action", "name": fc.name, "arguments": args}
                try:
                    result = _dispatch(fc.name, args, tool_registry)
                except ToolCallError as e:
                    result = {"error": str(e)}
                    yield {"type": "error", "text": str(e)}
                yield {"type": "observation", "name": fc.name, "result": result}
                if fc.name in TERMINAL_TOOLS:
                    yield {"type": "final", "name": fc.name, "result": result}
                    return
                tool_results_parts.append(
                    types.Part.from_function_response(name=fc.name, response=result)
                )

        contents.append(model_content)
        if not saw_function_call:
            yield {
                "type": "final",
                "name": "end_turn",
                "result": {"note": "model stopped without submit_to_ground/drop"},
            }
            return
        contents.append(types.Content(role="user", parts=tool_results_parts))

    yield {
        "type": "error",
        "text": f"max_steps ({max_steps}) reached without terminal action",
    }
