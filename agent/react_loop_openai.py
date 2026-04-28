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


def run_react_openai(
    image_before: str,
    image_after: str,
    base_url: str,
    model: str,
    api_key: str,
    tool_registry: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    timeout: float = 180.0,
) -> Iterator[dict[str, Any]]:
    """ReAct loop driven by an OpenAI-compatible /chat/completions endpoint."""
    tools = _to_openai_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "BEFORE (previous satellite pass over this location):"},
            {"type": "image_url", "image_url": {"url": _data_url(image_before)}},
            {"type": "text", "text": "AFTER (current pass, same location):"},
            {"type": "image_url", "image_url": {"url": _data_url(image_after)}},
            {"type": "text", "text": (
                "Analyze the pair and decide what to report to ground. "
                "Use your tools. End with submit_to_ground(...) or drop()."
            )},
        ]},
    ]

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    for step in range(max_steps):
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
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
            # Model gave a text answer without invoking a terminal tool.
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

            if name in TERMINAL_TOOLS:
                yield {"type": "final", "name": name, "result": result}
                terminal_hit = True
                break

        if terminal_hit:
            return

    yield {"type": "error",
           "text": f"max_steps ({max_steps}) reached without terminal action"}
