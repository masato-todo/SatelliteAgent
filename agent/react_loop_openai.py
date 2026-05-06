"""ReAct loop for OpenAI-compatible chat-completions endpoints
(llama.cpp llama-server, vLLM, Ollama).

Same event contract as react_loop.run_react so the SSE stream consumed by
the UI does not need to know which backend produced it.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

from tools.schema import TOOL_SCHEMAS, TERMINAL_TOOLS
from tools.validator import ToolCallError

from .react_loop import SYSTEM_PROMPT, _dispatch


def _data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _to_openai_tools() -> list[dict]:
    """Convert our internal tool schemas to OpenAI 'function' format."""
    out = []
    for s in TOOL_SCHEMAS:
        out.append({
            "type": "function",
            "function": {
                "name":        s["name"],
                "description": s["description"],
                "parameters":  s["input_schema"],
            },
        })
    return out


def _tool_catalog_block() -> str:
    """Render the tool list as a compact bullet catalog.

    LFM2.5-VL doesn't always read `tools=` reliably; surfacing the names +
    one-line descriptions in the system prompt nudges it to call only
    *valid* tool names instead of hallucinating ones.
    """
    lines = ["Available tools (call by exact name; do not invent others):"]
    for s in TOOL_SCHEMAS:
        desc = (s.get("description") or "").strip().split("\n", 1)[0]
        if len(desc) > 140:
            desc = desc[:137] + "..."
        params = (s.get("input_schema") or {}).get("properties") or {}
        required = set((s.get("input_schema") or {}).get("required") or [])
        if params:
            arg_bits = []
            for k in params:
                arg_bits.append(f"{k}*" if k in required else k)
            args = "(" + ", ".join(arg_bits) + ")"
        else:
            args = "()"
        marker = " [TERMINAL]" if s["name"] in TERMINAL_TOOLS else ""
        lines.append(f"- {s['name']}{args}{marker}: {desc}")
    lines.append("(* = required argument)")
    lines.append(
        "Workflow: 1) classify_change, 2) at least one spectral tool "
        "(compute_index / fetch_band / compute_index_delta / zoom_in), "
        "3) compose_report -> use the returned report_id, "
        "4) submit_to_ground(report_id, reason, ...) OR drop(reason). "
        "`reason` MUST cite concrete numbers from observations."
    )
    return "\n".join(lines)


SYSTEM_PROMPT_WITH_TOOLS = SYSTEM_PROMPT + "\n\n" + _tool_catalog_block()


def run_react_openai(
    image_before: str,
    image_after: str,
    base_url: str,
    model: str,
    api_key: str,
    tool_registry: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    timeout: float = 180.0,
    user_instructions: str | None = None,
    forced_tool_steps: int = 2,
) -> Iterator[dict[str, Any]]:
    """ReAct loop driven by an OpenAI-compatible /chat/completions endpoint.

    Args:
        user_instructions: optional free-text appended to the user turn so
            the operator can specify what to look for (e.g. "focus on the
            western shoreline; ignore cloud cover"). Empty/None falls back
            to the generic prompt.
        forced_tool_steps: number of leading steps where `tool_choice` is
            forced to "required". After that the model may answer with text
            only (so a final natural-language summary is allowed). 0 means
            never force; very high values reproduce the old always-required
            behavior. Default 2 = force investigation kickoff.
    """
    tools = _to_openai_tools()
    base_instruction = (
        "Analyze the pair and decide what to report to ground. "
        "Use your tools. End with submit_to_ground(report_id, reason, ...) "
        "or drop(reason)."
    )
    if user_instructions and user_instructions.strip():
        user_text = (
            base_instruction
            + "\n\nOperator instructions: " + user_instructions.strip()
        )
    else:
        user_text = base_instruction
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOLS},
        {"role": "user", "content": [
            {"type": "text", "text": "BEFORE (previous satellite pass over this location):"},
            {"type": "image_url", "image_url": {"url": _data_url(image_before)}},
            {"type": "text", "text": "AFTER (current pass, same location):"},
            {"type": "image_url", "image_url": {"url": _data_url(image_after)}},
            {"type": "text", "text": user_text},
        ]},
    ]

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    for step in range(max_steps):
        # `tool_choice="required"` for the first few turns guarantees the
        # model actually picks up its tools instead of returning a chatty
        # "I would inspect..." answer. After that we switch to "auto" so
        # the model is free to terminate with a tool call AND prepend a
        # natural-language summary in the same turn.
        force_tool = step < forced_tool_steps
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required" if force_tool else "auto",
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            yield {"type": "error", "text": f"OpenAI-compat call failed: {type(e).__name__}: {e}"}
            return
        if r.status_code != 200:
            yield {"type": "error", "text": f"HTTP {r.status_code}: {r.text[:300]}"}
            return
        try:
            data = r.json()
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as e:
            yield {"type": "error", "text": f"bad response shape: {type(e).__name__}: {e}",
                   "raw_preview": r.text[:300]}
            return

        content = (msg.get("content") or "").strip()
        if content:
            yield {"type": "thought", "text": content}

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # No tool call: treat as the model's final natural-language answer.
            # The flow is: tool gating (compose_report -> submit_to_ground)
            # already prevents fabricated submits, so giving up here is safe.
            yield {"type": "final", "name": "end_turn",
                   "result": {"note": "model stopped without submit_to_ground/drop",
                              "text": content}}
            return

        # Append the assistant turn so the model sees its own tool_calls next round
        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
        })

        terminal_hit = False
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            yield {"type": "action", "name": name, "arguments": args}

            try:
                result = _dispatch(name, args, tool_registry)
            except ToolCallError as e:
                result = {"error": str(e)}
                yield {"type": "error", "text": str(e)}

            yield {"type": "observation", "name": name, "result": result}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or name,
                "content": json.dumps(result, ensure_ascii=False),
            })

            # Only treat submit_to_ground/drop as terminal when they actually
            # succeeded. A rejected submit (e.g. unknown report_id) returns
            # status="error" and the model gets another turn to recover.
            if name in TERMINAL_TOOLS and isinstance(result, dict) \
                    and result.get("status") not in ("error",) \
                    and "error" not in result:
                yield {"type": "final", "name": name, "result": result}
                terminal_hit = True
                break

        if terminal_hit:
            return

    yield {"type": "error",
           "text": f"max_steps ({max_steps}) reached without terminal action"}
