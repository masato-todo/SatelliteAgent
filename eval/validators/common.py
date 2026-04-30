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


async def balanced_action_match(completion, info, **_kw) -> float:
    """Class-weighted action match. The only reward we actually care about,
    weighted to neutralize the dataset's positive prior.

    Dataset is 51 positive : 16 negative, so an "always submit" policy
    gets ~0.76 raw accuracy. To make that policy uncompetitive, we weight
    a correct drop ~3× a correct submit (51/16 ≈ 3.19 → rounded to 3.0).

    Returns:
        0.0  if the terminal action is missing or wrong.
        1.0  if expected="submit_to_ground" and the model submitted.
        3.0  if expected="drop" and the model dropped.

    Expected per-rollout averages:
      - always-submit policy:  0.76 * 1.0 + 0.24 * 0   = 0.76
      - always-drop   policy:  0.76 * 0   + 0.24 * 3.0 = 0.72
      - perfect agent:         0.76 * 1.0 + 0.24 * 3.0 = 1.48

    Both trivial policies sit near 0.75, so the model has to actually
    distinguish positives from negatives to climb above that.
    """
    target = _expected(info).get("action")
    name, _ = _terminal_call(completion)
    if name is None or name != target:
        return 0.0
    return 1.0 if target == "submit_to_ground" else 3.0


_INDEX_TERMS = (
    "nbr", "ndvi", "ndwi", "mndwi", "ndbi", "ndsi",
    "delta", "frac_decrease", "frac_increase",
    "swir", "nir", "rededge",
)
_ALL_INDICES = {"NBR", "NDVI", "NDWI", "MNDWI", "NDBI", "NDSI"}


def _fetched_indices(completion) -> set:
    """Return the set of spectral indices the model actually fetched
    via compute_index / compute_index_delta in this rollout.

    Calling either tool with no `index` arg returns all six → treated
    as fetching the full set.
    """
    fetched: set = set()
    if not isinstance(completion, list):
        return fetched
    for msg in completion:
        tcs = getattr(msg, "tool_calls", None)
        if tcs is None and isinstance(msg, dict):
            tcs = msg.get("tool_calls")
        for tc in (tcs or []):
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except Exception:
                    continue
            if isinstance(tc, dict):
                fn_name = tc.get("name") or (tc.get("function") or {}).get("name")
                fn_args = (
                    tc.get("arguments")
                    or tc.get("args")
                    or (tc.get("function") or {}).get("arguments")
                    or {}
                )
            else:
                fn_name = getattr(tc, "name", None)
                fn_args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
            if fn_name not in ("compute_index", "compute_index_delta"):
                continue
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}
            if not isinstance(fn_args, dict):
                continue
            idx = fn_args.get("index")
            if isinstance(idx, str) and idx in _ALL_INDICES:
                fetched.add(idx)
            else:
                fetched.update(_ALL_INDICES)
    return fetched


async def reason_grounded(completion, info, **_kw) -> float:
    """LEGACY (S27): +0.5 if reason mentions any index term.

    Allows the copy-paste exploit (model parrots an example reason from
    the system prompt regardless of action). Replaced by
    `reason_grounded_correct` from S28 onward.
    """
    name, args = _terminal_call(completion)
    if name is None:
        return 0.0
    reason = args.get("reason") if isinstance(args, dict) else None
    if not isinstance(reason, str) or len(reason.strip()) < 8:
        return 0.0
    low = reason.lower()
    return 0.5 if any(t in low for t in _INDEX_TERMS) else 0.0


async def reason_grounded_correct(completion, info, **_kw) -> float:
    """+0.5 only when ALL of:
      1. terminal action matches expected (correctness),
      2. reason mentions an index NAME that was actually fetched via
         compute_index / compute_index_delta in this rollout (true
         grounding, not parroting an example).

    Closes the S27 copy-paste exploit: a wrong-action rollout with a
    plausible-looking reason now gets 0.0, and even correct rollouts
    must have actually queried the index they cite.
    """
    target = _expected(info).get("action")
    name, args = _terminal_call(completion)
    if name is None or name != target:
        return 0.0
    reason = args.get("reason") if isinstance(args, dict) else None
    if not isinstance(reason, str) or len(reason.strip()) < 8:
        return 0.0
    fetched = _fetched_indices(completion)
    if not fetched:
        return 0.0
    low = reason.lower()
    return 0.5 if any(idx.lower() in low for idx in fetched) else 0.0


async def lookup_called(completion, info, **_kw) -> float:
    """+0.2 if at least one successful (non-error) lookup tool message
    appears BEFORE the terminal call. Independent of action correctness.

    Purpose: provide GRPO with a variance signal when all rollouts in a
    group commit to the same terminal action. Without this, batches with
    8 identical "submit_to_ground" rollouts have zero advantage and the
    model gets no learning signal for changing investigation behavior.

    The reward is intentionally small (0.2 << balanced_action_match's
    1.0/3.0) so it cannot pay for the S22 exploit ("call cheap tool then
    blind submit"). It only nudges initial behavior to start with a
    lookup call so the model can then learn to commit correctly.
    """
    if not isinstance(completion, list):
        return 0.0
    for msg in completion:
        tcs = getattr(msg, "tool_calls", None)
        if tcs is None and isinstance(msg, dict):
            tcs = msg.get("tool_calls")
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    tc_name = tc.get("name") or (tc.get("function") or {}).get("name")
                else:
                    tc_name = getattr(tc, "name", None)
                if tc_name in TERMINAL:
                    return 0.0  # terminal hit before any successful lookup
        s = _tool_msg_content_str(msg)
        if s and "'error'" not in s and '"error"' not in s and not s.startswith("{'error"):
            role = getattr(msg, "role", None)
            if role is None and isinstance(msg, dict):
                role = msg.get("role")
            if role == "tool":
                return 0.2
    return 0.0


async def grounded_action_match(completion, info, **_kw) -> float:
    """Reward 1.0 only when the terminal action is correct AND was preceded
    by at least one successful (non-error) lookup tool call.

    Rationale: S21 traces showed that the model exploits the positive
    prior by submitting blindly without ever calling a lookup tool.
    `action_match` rewards both "blind submit that happens to be right"
    and "grounded submit that's right" identically. This reward function
    only credits grounded decisions, so blind correct submissions stop
    receiving the bonus signal — pushing the model toward an
    "investigate-then-decide" policy.

    A tool message counts as "successful" when:
      - its role == "tool"
      - its content does NOT contain `'error'` (env returns
        `{'error': '...'}` strings on cache miss / bad args).
      - it appears BEFORE the terminal tool call in the completion
        order.

    Returns 0.0 if either condition fails (wrong action, or right action
    but no successful evidence preceded it).
    """
    if not isinstance(completion, list):
        return 0.0
    target = _expected(info).get("action")
    name, _ = _terminal_call(completion)
    if name is None or name != target:
        return 0.0

    # Walk completion in order; ensure a non-error tool message exists
    # BEFORE the first terminal tool call.
    for msg in completion:
        # Did we hit the terminal call? Stop searching for evidence after.
        tcs = getattr(msg, "tool_calls", None)
        if tcs is None and isinstance(msg, dict):
            tcs = msg.get("tool_calls")
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    tc_name = tc.get("name") or (tc.get("function") or {}).get("name")
                else:
                    tc_name = getattr(tc, "name", None)
                if tc_name in TERMINAL:
                    # No evidence accumulated before the terminal call.
                    return 0.0
        # Otherwise, check tool messages
        s = _tool_msg_content_str(msg)
        if s and "'error'" not in s and '"error"' not in s and not s.startswith("{'error"):
            role = getattr(msg, "role", None)
            if role is None and isinstance(msg, dict):
                role = msg.get("role")
            if role == "tool":
                return 1.0  # found evidence before terminal
    return 0.0
