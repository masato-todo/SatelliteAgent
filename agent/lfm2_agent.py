"""Multi-turn agent loop for the LFM2.5-VL SFT/GRPO model served by vLLM.

This is the inference-time SSOT for the trained LFM2.5-VL agent (S62-S65).

Tool set (4 tools, matching SFT trajectory format):
  - compute_index_delta(index="all"): read precomputed spectral binary tags
  - analyze(evidence, interpretation, recommended_action): commit reasoning
  - submit_to_ground(reason): terminal — transmit
  - drop(reason):              terminal — discard

Critical settings (discovered via S62-S64 ablations):
  - tool_choice="required": grammar-constrained generation, force tool emission
  - include_images=False:   removing images from the user prompt unblocks
                            the SFT-trained tool-calling pattern (75% acc vs
                            51% with images). The model was trained with images
                            but at inference time vision context dominates and
                            the agent skips compute_index_delta. Without
                            images the trained agent flow runs cleanly.

Usage:
    from agent.lfm2_agent import run_lfm2_agent
    result = run_lfm2_agent(
        case_id="mcd64a1_h09v04_202307_p4582_-12035",
        before_path="/path/to/before.png",
        after_path="/path/to/after.png",
        precompute_root="/path/to/precompute_v4",
        vllm_url="http://localhost:8000/v1",
        served_model="LFM2.5-VL-450M",
        include_images=False,  # set True to test image-mode (lower accuracy)
    )
    print(result["terminal"])      # "submit_to_ground" or "drop"
    print(result["tool_call_log"]) # list of {name, args}
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
import urllib.error
from typing import Any

import yaml


SYS_PROMPT = (
    "You are an onboard satellite operator agent on Earth-observation duty.\n\n"
    "You are shown a Sentinel-2 image pair (before / after) of the same location. "
    "Decide whether to transmit a report to ground (submit_to_ground) or discard the data (drop).\n\n"
    "Use compute_index_delta() (with no arguments) to get all 6 spectral indices at once. "
    "Then call analyze() to commit your reasoning, then call submit_to_ground() or drop()."
)


TOOLS = [
    {"type": "function", "function": {
        "name": "compute_index_delta",
        "description": (
            "Returns spectral delta tags (burn, vegetation, water, built-up, snow) "
            "for the image pair. Call with no arguments to get all 6 indices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "enum": ["all", "NBR", "NDVI", "NDWI", "MNDWI", "NDBI", "NDSI"],
                    "description": "Index name. Default 'all' returns binary tags for all 6.",
                },
            },
            "required": [],
        },
    }},
    {"type": "function", "function": {
        "name": "analyze",
        "description": "Commit reasoning before terminal action. Cite spectral evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "evidence":     {"type": "string"},
                "interpretation": {
                    "type": "string",
                    "enum": ["significant_change", "no_significant_change"],
                },
                "recommended_action": {
                    "type": "string",
                    "enum": ["submit_to_ground", "drop"],
                },
            },
            "required": ["evidence", "interpretation", "recommended_action"],
        },
    }},
    {"type": "function", "function": {
        "name": "submit_to_ground",
        "description": "Transmit the report. Use when significant change worth ground attention.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    }},
    {"type": "function", "function": {
        "name": "drop",
        "description": "Discard the data. Use when no significant change.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    }},
]


TERMINAL_TOOLS = frozenset({"submit_to_ground", "drop"})

INDICES = ["NBR", "NDVI", "NDWI", "MNDWI", "NDBI", "NDSI"]
LEVEL_RANK = {"NONE": 0, "MODERATE": 1, "STRONG": 2}


# ---------------------------------------------------------------------------
# Precompute reading (offline tool implementation)
# ---------------------------------------------------------------------------

def _level(v):
    if v is None: return "NONE"
    if v >= 0.30: return "STRONG"
    if v >= 0.10: return "MODERATE"
    return "NONE"


def read_compute_index_delta(precompute_root: str, case_id: str, index: str):
    """Read one index's delta_stats from precompute YAML. Returns dict or None."""
    path = os.path.join(precompute_root, case_id, "compute_index_delta", f"{index}.stats.yaml")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return (data.get("response") or {}).get("delta_stats") or {}
    except Exception:
        return None


def case_features(precompute_root: str, case_id: str):
    """Read all 6 indices for a case. Returns dict[index → delta_stats] or None."""
    deltas = {}
    for idx in INDICES:
        d = read_compute_index_delta(precompute_root, case_id, idx)
        if d is None:
            return None
        deltas[idx] = d
    return deltas


def tags_of(feats: dict) -> dict[str, str]:
    """Convert raw spectral deltas to disaster-type binary tags (S50 style)."""
    nbr   = feats["NBR"]; ndvi  = feats["NDVI"]; ndwi  = feats["NDWI"]
    mndwi = feats["MNDWI"]; ndbi  = feats["NDBI"]; ndsi  = feats["NDSI"]
    return {
        "burn":    _level(nbr.get("frac_decrease_strong")),
        "veg":     _level(ndvi.get("frac_decrease_strong")),
        "water":   max(
            [_level(ndwi.get("frac_decrease_strong")),  _level(ndwi.get("frac_increase_strong")),
             _level(mndwi.get("frac_decrease_strong")), _level(mndwi.get("frac_increase_strong"))],
            key=lambda x: LEVEL_RANK[x],
        ),
        "builtup": _level(ndbi.get("frac_increase_strong")),
        "snow":    max(
            [_level(ndsi.get("frac_decrease_strong")), _level(ndsi.get("frac_increase_strong"))],
            key=lambda x: LEVEL_RANK[x],
        ),
    }


def build_observation(feats: dict) -> str:
    """Tool observation in the EXACT format used during SFT training (S62-S64)."""
    tags = tags_of(feats)
    means = {idx: feats[idx].get("mean") or 0.0 for idx in INDICES}
    return (
        f"burn_or_biomass_loss: {tags['burn']} (NBR mean={means['NBR']:+.2f})\n"
        f"vegetation_loss:      {tags['veg']} (NDVI mean={means['NDVI']:+.2f})\n"
        f"water_change:         {tags['water']} (NDWI mean={means['NDWI']:+.2f}, MNDWI mean={means['MNDWI']:+.2f})\n"
        f"built_up_change:      {tags['builtup']} (NDBI mean={means['NDBI']:+.2f})\n"
        f"snow_change:          {tags['snow']} (NDSI mean={means['NDSI']:+.2f})"
    )


def execute_tool(name: str, args: dict, *, precompute_root: str, case_id: str) -> str:
    """Execute one tool call. Returns observation string for the tool message."""
    if name == "compute_index_delta":
        feats = case_features(precompute_root, case_id)
        if feats is None:
            return "error: precompute missing"
        which = (args or {}).get("index", "all")
        if which == "all" or which is None:
            return build_observation(feats)
        if which in INDICES:
            d = feats[which]
            return (f"{which}: mean={d.get('mean'):+.3f}, "
                    f"frac_decrease={d.get('frac_decrease_strong'):.3f}, "
                    f"frac_increase={d.get('frac_increase_strong'):.3f}")
        return f"error: unknown index {which!r}"
    if name == "analyze":
        action = (args or {}).get("recommended_action", "?")
        return f"noted, call {action}"
    if name in TERMINAL_TOOLS:
        return "ok"
    return f"error: unknown tool {name!r}"


# ---------------------------------------------------------------------------
# Multi-turn agent loop
# ---------------------------------------------------------------------------

def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def initial_messages(
    before_path: str | None,
    after_path: str | None,
    *,
    include_images: bool = False,
) -> list[dict]:
    """Build initial messages.

    include_images=False (default) — text-only prompt. Reaches 75% accuracy
        with the SFT-trained model (vs 51% with images). Discovered in S63.
    include_images=True — original VLM prompt with before/after images.
    """
    user_content: list[dict[str, Any]] = []
    if include_images and before_path and after_path:
        user_content.extend([
            {"type": "text",      "text": "Before image (previous pass):"},
            {"type": "image_url", "image_url": {"url": _b64_image(before_path)}},
            {"type": "text",      "text": "After image (current pass):"},
            {"type": "image_url", "image_url": {"url": _b64_image(after_path)}},
        ])
    user_content.append({
        "type": "text",
        "text": "Investigate with your tools, then decide submit_to_ground or drop.",
    })
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]


def chat_complete(
    vllm_url: str,
    served_model: str,
    messages: list[dict],
    *,
    api_key: str = "dummy",
    max_tokens: int = 256,
    temperature: float = 0.0,
    tool_choice: str = "required",
    timeout: float = 120.0,
) -> dict:
    """Single /chat/completions call. tool_choice='required' is critical
    (forces vLLM to emit tool_calls — without it, model often answers in text)."""
    body = {
        "model": served_model, "messages": messages,
        "tools": TOOLS, "tool_choice": tool_choice,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    url = vllm_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_lfm2_agent(
    case_id: str,
    *,
    before_path: str | None = None,
    after_path: str | None = None,
    precompute_root: str,
    vllm_url: str = "http://localhost:8000/v1",
    served_model: str = "LFM2.5-VL-450M",
    api_key: str = "dummy",
    include_images: bool = False,
    max_turns: int = 6,
    temperature: float = 0.0,
) -> dict:
    """Run the multi-turn agent loop until terminal action or max_turns.

    Returns:
        {
            "terminal": "submit_to_ground" | "drop" | "max_turns_exceeded",
            "tool_call_log": [{"name": str, "args": dict}, ...],
            "raw_log": [{"turn": int, "finish": str, "n_tool_calls": int, ...}, ...],
            "messages": full message history,
        }
    """
    messages = initial_messages(before_path, after_path, include_images=include_images)
    terminal = None
    tool_call_log: list[dict] = []
    raw_log: list[dict] = []

    for turn in range(max_turns):
        try:
            resp = chat_complete(
                vllm_url, served_model, messages,
                api_key=api_key, temperature=temperature,
            )
        except Exception as e:
            terminal = f"HTTP_ERR_{type(e).__name__}"
            raw_log.append({"turn": turn, "error": str(e)})
            break

        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        content = msg.get("content")
        finish = resp["choices"][0].get("finish_reason")
        raw_log.append({
            "turn": turn,
            "finish": finish,
            "content": content,
            "n_tool_calls": len(tcs),
        })

        if not tcs:
            # Model answered in plain text — treat as drop.
            terminal = "drop"
            break

        # Append assistant turn so model sees its own tool_calls
        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": tcs,
        })

        tc = tcs[0]
        fn = tc.get("function", {})
        fname = fn.get("name")
        try:
            fargs = json.loads(fn.get("arguments") or "{}")
        except Exception:
            fargs = {}
        tool_call_log.append({"name": fname, "args": fargs})

        if fname in TERMINAL_TOOLS:
            terminal = fname
            break

        # Execute tool, append observation
        obs = execute_tool(fname, fargs, precompute_root=precompute_root, case_id=case_id)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"t{turn}"),
            "name": fname,
            "content": str(obs),
        })

    if terminal is None:
        terminal = "max_turns_exceeded"

    return {
        "terminal": terminal,
        "tool_call_log": tool_call_log,
        "raw_log": raw_log,
        "messages": messages,
    }
