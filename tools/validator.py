"""Schema validation + retry + fallback for tool calls.

Design goals:
- Catch malformed tool calls from the LLM, retry with error feedback.
- After N failures, fall back to a deterministic pipeline so the system
  always produces output (hackathon requirement: "must run without debugging").
"""
from __future__ import annotations

from typing import Any, Callable


class ToolCallError(Exception):
    """Raised when a tool call cannot be validated or executed."""


def validate_and_dispatch(
    name: str,
    arguments: dict[str, Any],
    registry: dict[str, Callable[..., Any]],
) -> Any:
    """Validate then dispatch a tool call. Phase 1: dispatch-only, validation TBD."""
    if name not in registry:
        raise ToolCallError(f"unknown tool: {name}")
    try:
        return registry[name](**arguments)
    except TypeError as e:
        raise ToolCallError(f"argument mismatch for {name}: {e}") from e


def run_deterministic_fallback(
    image_before: str,
    image_after: str,
    classifier: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Hard-coded pipeline when the agent fails repeatedly.

    Always yields a valid report decision so the demo never hangs.
    """
    classes = classifier(image_before, image_after).get("classes", [])
    top = classes[0] if classes else {"no_change": 1.0}
    top_name, top_conf = next(iter(top.items()))

    if top_name == "no_change" or top_conf < 0.4:
        return {"action": "drop"}
    if top_name in {"flood", "fire", "earthquake"}:
        return {"action": "send_full", "urgency": 8, "change_type": top_name}
    return {"action": "send_text", "urgency": 5, "change_type": top_name}