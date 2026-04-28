"""Reward functions for satelliteagent_env (verifiers Rubric format).

Each function follows the verifiers signature:
    async def f(completion, info, **kw) -> float

`completion` is the list of assistant/tool messages produced during the
rollout. `info` is the per-row dict the env attached at dataset build time.

Phase 5b minimal set: only `action_match` is required to land a smoke pass.
The other validators are stubs we'll extend once the basic loop is healthy.
"""
from __future__ import annotations

from typing import Any

TERMINAL = {"submit_to_ground", "drop"}


def _terminal_call(completion: list[Any]) -> tuple[str | None, dict | None]:
    """Pick up the first terminal tool call (submit_to_ground / drop) and
    return (name, arguments_dict). If none found, returns (None, None).
    """
    if not isinstance(completion, list):
        return None, None
    for msg in completion:
        tcs = getattr(msg, "tool_calls", None) or []
        for tc in tcs:
            name = getattr(tc, "name", None)
            if name in TERMINAL:
                args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
                return name, args
    return None, None


def _expected(info: dict | None) -> dict:
    return ((info or {}).get("expected")) or {}


# === required: 1.0 / 0.0 binary on terminal action --------------------------

async def action_match(completion, info, **_kw) -> float:
    target = _expected(info).get("action")
    name, _ = _terminal_call(completion)
    if name is None:
        return 0.0
    return 1.0 if name == target else 0.0


# === optional (use after smoke is green) ------------------------------------

async def attach_image_match(completion, info, **_kw) -> float:
    """Reward 1.0 if the assistant's submit_to_ground attach_image matches
    expected.attach_image. Vacuous reward (1.0) when expected says drop or
    when expectation is missing -- this validator should not punish on cases
    where there is nothing to attach.
    """
    exp = _expected(info)
    if exp.get("action") == "drop":
        return 1.0
    target = exp.get("attach_image")
    if target is None:
        return 1.0
    name, args = _terminal_call(completion)
    if name != "submit_to_ground":
        return 0.0
    got = bool(args.get("attach_image", False))
    return 1.0 if got == bool(target) else 0.0


async def urgency_match(completion, info, **_kw) -> float:
    exp = _expected(info)
    target = exp.get("urgency")
    if target is None or exp.get("action") == "drop":
        return 1.0
    name, args = _terminal_call(completion)
    if name != "submit_to_ground":
        return 0.0
    got = args.get("urgency")
    return 1.0 if got == target else 0.0


async def change_type_match(completion, info, **_kw) -> float:
    exp = _expected(info)
    target = exp.get("change_type")
    if target is None or exp.get("action") == "drop":
        return 1.0
    name, args = _terminal_call(completion)
    if name != "submit_to_ground":
        return 0.0
    got = args.get("change_type")
    return 1.0 if got == target else 0.0
