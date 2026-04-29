"""Reward functions for satelliteagent_env (verifiers Rubric format).

Each function follows the verifiers signature:
    async def f(completion, info, **kw) -> float

`completion` is the list of assistant/tool messages produced during the
rollout. `info` is the per-row dict the env attached at dataset build time.

Phase 5b minimal set: only `action_match` is required to land a smoke pass.
The other validators are stubs we'll extend once the basic loop is healthy.
"""
from __future__ import annotations

import json
from typing import Any

TERMINAL = {"submit_to_ground", "drop"}


def _terminal_call(completion: list[Any]) -> tuple[str | None, dict | None]:
    """Pick up the first terminal tool call (submit_to_ground / drop) and
    return (name, arguments_dict). If none found, returns (None, None).

    `completion` may also be a list of plain dicts (verifiers env-server form)
    or have `tool_calls` whose entries are dicts/strings instead of attribute
    objects, depending on how the env serialised them. We normalise here so
    that downstream validators can treat the result as a dict.
    """
    if not isinstance(completion, list):
        return None, None
    for msg in completion:
        # msg can be a dict or an object; normalise tool_calls extraction
        tcs = getattr(msg, "tool_calls", None)
        if tcs is None and isinstance(msg, dict):
            tcs = msg.get("tool_calls")
        tcs = tcs or []
        for tc in tcs:
            # tc can be: object with .name/.arguments, dict with name/arguments,
            # or even a JSON string.
            if isinstance(tc, str):
                try:
                    tc_obj = json.loads(tc)
                except Exception:
                    continue
                name = tc_obj.get("name")
                args = tc_obj.get("arguments") or tc_obj.get("args") or {}
            elif isinstance(tc, dict):
                name = tc.get("name") or (tc.get("function") or {}).get("name")
                args = tc.get("arguments") or tc.get("args") \
                    or (tc.get("function") or {}).get("arguments") or {}
            else:
                name = getattr(tc, "name", None)
                args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            if name in TERMINAL:
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


# === schema-shaping rewards ------------------------------------------------

def _tool_messages(completion: list[Any]) -> list[Any]:
    """Return the tool-role messages from a completion list."""
    out = []
    if not isinstance(completion, list):
        return out
    for msg in completion:
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role == "tool":
            out.append(msg)
    return out


def _tool_msg_content_str(msg: Any) -> str:
    """Extract a tool message's content as a string for substring checks."""
    c = getattr(msg, "content", None)
    if c is None and isinstance(msg, dict):
        c = msg.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # multimodal content parts
        parts = []
        for p in c:
            if isinstance(p, dict):
                parts.append(p.get("text") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(c)


async def valid_tool_args(completion, info, **_kw) -> float:
    """Reward the fraction of lookup-tool calls whose args were actually
    accepted by the env (i.e., the tool DID NOT return an error string).

    Rationale: LFM2.5-VL-450M is small enough that it hallucinates
    `compute_index(index='after', which='before')` etc. when only
    `action_match` is rewarded. Giving partial credit when a lookup tool
    actually finds its cached entry pushes the model toward the correct
    schema (NBR/NDVI/... for `index`, B3/B4/... for `band`, etc.) WITHOUT
    requiring the rollout to reach a terminal action — so the reward signal
    survives even when max_turns is hit.

    Returns 0.0 if no tool response is found at all (e.g., model never used
    a tool). Returns 1.0 if every tool response is non-error.
    """
    msgs = _tool_messages(completion)
    if not msgs:
        return 0.0
    n_total = len(msgs)
    n_ok = 0
    for m in msgs:
        s = _tool_msg_content_str(m)
        # Tool implementations in `satelliteagent_env` return
        # `{'error': '...'}` on cache miss / bad args. After ToolEnv.call_tool
        # str()s the dict, the substring `'error'` appears in the message.
        # We treat any non-error response as "args were valid".
        if "'error'" not in s and '"error"' not in s and not s.startswith("{'error"):
            n_ok += 1
    return n_ok / n_total


async def terminal_reached(completion, info, **_kw) -> float:
    """Reward 1.0 if the rollout reached ANY terminal action (submit/drop),
    else 0.0. Independent of correctness — just rewards termination.

    Helps when max_turns is the binding constraint and most rollouts get
    cut off mid-loop with reward 0; this gives the model a small but
    consistent signal to terminate at all.
    """
    name, _ = _terminal_call(completion)
    return 1.0 if name in TERMINAL else 0.0
