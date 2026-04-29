"""SatelliteAgent prime-rl environment (verifiers framework).

Single source of truth: this package is a thin wrapper that wires SatelliteAgent's
existing `tools/` and (eventually) `eval/validators/` into the verifiers
StatefulToolEnv interface used by prime-rl orchestrator.

Loaded by prime-rl via `id = "satelliteagent_env"` in orchestrator config.
The verifiers loader does:
    importlib.import_module("satelliteagent_env").load_environment(**kwargs)

Only the glue (Dataset row format, state injection, reward signature adapter)
lives here. Tool implementations and reward logic remain in `tools/` and `eval/`.

Status: Phase 5 toy. Exercises submit_to_ground / drop tool-calling end-to-end
on a tiny hand-crafted dataset so we can verify the rollout + reward pipeline
before the real Phase 2 triplet data lands.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TYPE_CHECKING

# Lazy imports: `verifiers` and `datasets` are heavy and only needed when
# prime-rl actually loads the env. SatelliteAgent app dev/test doesn't need
# them, so we defer until `load_environment()` is called.
if TYPE_CHECKING:
    import verifiers as vf
    from datasets import Dataset

# === SSOT imports from SatelliteAgent ===
from tools.stubs import STUB_TOOLS, submit_to_ground, drop


# === module-import-time debug breadcrumb =====================================
# Writes UNCONDITIONALLY (no env var gate) so we can verify whether THIS module
# is being imported by the env worker at all. Catches: stale wheel deployed,
# env var stripped by mp.spawn, module failing to import. Path is hardcoded so
# even if env vars are missing the file appears.
def _import_breadcrumb() -> None:
    try:
        import os as _os, time as _time, sys as _sys
        path = "/kaggle/working/outputs/satelliteagent_import.log"
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"{_time.time():.3f} pid={_os.getpid()} ppid={_os.getppid()} "
                f"argv0={_sys.argv[0] if _sys.argv else '?'} "
                f"DEBUG_LOG_set={_os.environ.get('SATELLITEAGENT_DEBUG_LOG', '<UNSET>')!r} "
                f"TRACE_PATH_set={_os.environ.get('SATELLITEAGENT_TRACE_PATH', '<UNSET>')!r}\n"
            )
    except Exception:
        pass


_import_breadcrumb()


def _expose_for_vf(fn: Callable) -> Callable:
    """Strip leading-underscore ``**_extra``-style varkw from a function's
    signature so verifiers' pydantic-v2 backed tool converter accepts it.

    SatelliteAgent's stubs use ``**_extra`` / ``**_ignored`` to swallow
    forward-compat kwargs from upstream callers, but pydantic forbids field
    names starting with underscore (``Fields must not use names with leading
    underscores; e.g., use 'extra' instead of '_extra'``). We expose a wrapper
    with the varkw removed; runtime behaviour is unchanged.
    """
    sig = inspect.signature(fn)
    has_underscore_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD and p.name.startswith("_")
        for p in sig.parameters.values()
    )
    if not has_underscore_varkw:
        return fn

    new_params = [
        p for p in sig.parameters.values()
        if not (p.kind == inspect.Parameter.VAR_KEYWORD and p.name.startswith("_"))
    ]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]
    return wrapper
# from eval.validators.common import (   # Phase 2 TODO
#     action_match,
#     attach_image_match,
#     trajectory_validity,
#     ...
# )
# from eval.runner import load_triplets   # Phase 2 TODO


# --- Toy dataset (replace with Phase 2 triplet loader) -------------------

_TOY_ROWS: list[dict[str, str]] = [
    {"scenario": "Major flood detected in densely populated city, water rising rapidly.",     "expected_action": "submit_to_ground"},
    {"scenario": "Empty desert with no vegetation or structures, nothing of interest.",       "expected_action": "drop"},
    {"scenario": "Wildfire burning across forest near a town with visible smoke plume.",     "expected_action": "submit_to_ground"},
    {"scenario": "Calm open ocean with no ships, weather, or anomalies visible.",            "expected_action": "drop"},
    {"scenario": "Volcanic eruption with large ash cloud over inhabited region.",            "expected_action": "submit_to_ground"},
    {"scenario": "Routine view of cloudless agricultural fields, normal pattern.",            "expected_action": "drop"},
    {"scenario": "Major earthquake damage visible in urban area with collapsed buildings.",   "expected_action": "submit_to_ground"},
    {"scenario": "Quiet rural landscape with rivers and small farms, nothing unusual.",      "expected_action": "drop"},
    {"scenario": "Oil spill spreading across coastal waters near port.",                       "expected_action": "submit_to_ground"},
    {"scenario": "Plain snow-covered mountains with no signs of activity.",                   "expected_action": "drop"},
]

_SYSTEM_PROMPT = (
    "You are an onboard satellite imaging agent. For each scenario you receive, "
    "you must decide whether to transmit a report to the ground or drop the data. "
    "You have exactly two tools available:\n"
    "  - submit_to_ground(report_id, attach_image=False): use when the scene is "
    "    urgent / important / requires human attention.\n"
    "  - drop(): use when the scene is uninteresting and not worth bandwidth.\n"
    "You MUST call exactly one of these two tools. Do not produce any other "
    "text response."
)


def _toy_dataset() -> "Dataset":
    from datasets import Dataset

    rows = []
    for r in _TOY_ROWS:
        rows.append({
            "prompt": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": r["scenario"]},
            ],
            "info": {
                "scenario": r["scenario"],
                "expected_action": r["expected_action"],
            },
        })
    return Dataset.from_list(rows)


def _toy_rubric() -> "vf.Rubric":
    """Reward 1.0 if the model called the expected terminal action, else 0.0.

    Picks up the first terminal call (`submit_to_ground` or `drop`) and
    compares to `info["expected_action"]`.
    """
    import verifiers as vf

    TERMINAL = {"submit_to_ground", "drop"}

    async def action_match(completion, info, **kw) -> float:
        if not isinstance(completion, list):
            return 0.0
        expected = info.get("expected_action")
        for msg in completion:
            tcs = getattr(msg, "tool_calls", None) or []
            for tc in tcs:
                name = getattr(tc, "name", None)
                if name in TERMINAL:
                    return 1.0 if name == expected else 0.0
        return 0.0  # no terminal action called

    return vf.Rubric(funcs=[action_match], weights=[1.0])


def _scrub_content_for_trace(content):
    """Strip base64 image bodies from message `content` so traces stay small."""
    if isinstance(content, str):
        return content[:1000]
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                out.append(repr(part)[:200]); continue
            ptype = part.get("type")
            if ptype == "text":
                out.append({"type": "text", "text": (part.get("text") or "")[:1000]})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if isinstance(url, str) and url.startswith("data:"):
                    out.append({"type": "image_url", "image_url": {"url": url[:64] + f"...<{len(url)}b>"}})
                else:
                    out.append({"type": "image_url", "image_url": {"url": str(url)[:200]}})
            else:
                out.append({"type": ptype, "_truncated": True})
        return out
    return repr(content)[:300]


def _summarize_message(msg) -> dict:
    """Convert a chat message (dict or pydantic model) into a JSON-safe summary.

    Captures role, scrubbed content, tool_calls (name + parsed JSON args) and
    tool result content. Designed for offline analysis of multi-turn traces.
    """
    if hasattr(msg, "model_dump"):
        try:
            d = msg.model_dump()
        except Exception:
            d = {}
    elif isinstance(msg, dict):
        d = msg
    else:
        d = {"_repr": repr(msg)[:300]}

    role = d.get("role") or getattr(msg, "role", None)
    out = {"role": role}

    content = d.get("content", getattr(msg, "content", None))
    if content is not None:
        out["content"] = _scrub_content_for_trace(content)

    tool_calls = d.get("tool_calls") or getattr(msg, "tool_calls", None) or []
    if tool_calls:
        tcs = []
        import json as _json
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                try: td = tc.model_dump()
                except Exception: td = {}
            elif isinstance(tc, dict):
                td = tc
            else:
                td = {}
            fn = td.get("function") or {}
            name = fn.get("name") or td.get("name") or getattr(tc, "name", None)
            raw_args = fn.get("arguments") if isinstance(fn, dict) else None
            if raw_args is None:
                raw_args = td.get("arguments")
            parsed = raw_args
            if isinstance(raw_args, str):
                try: parsed = _json.loads(raw_args)
                except Exception: parsed = raw_args[:500]
            tcs.append({"name": name, "args": parsed})
        out["tool_calls"] = tcs

    if role == "tool":
        out["tool_call_id"] = d.get("tool_call_id")
        out["name"] = d.get("name")
    return out


_PRECOMPUTE_TOOL_NAMES = {
    "compute_index", "compute_index_delta", "fetch_band", "false_color", "classify_change",
}


def _debug_log(msg: str) -> None:
    """Append a line to ``$SATELLITEAGENT_DEBUG_LOG`` when set.

    Opt-in via env var so production runs don't pollute disk. Writes one line
    per call, opens-and-closes per call so a process crash never loses buffered
    output.
    """
    try:
        import os as _os, time as _time
        path = _os.environ.get("SATELLITEAGENT_DEBUG_LOG", "").strip()
        if not path:
            return
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{_time.time():.3f} pid={_os.getpid()} {msg}\n")
    except Exception:
        pass


def _build_env_class(precompute_root: str | None = None):
    """Build the StatefulToolEnv subclass at call time so `verifiers` import
    is deferred. Returns the class object.

    Args:
        precompute_root: when set, update_tool_args injects `case_dir =
            precompute_root/<case_id>` into precompute-lookup tool calls so
            the offline tools can read per-case PNG/YAML from disk.
    """
    import verifiers as vf
    import os
    _precompute_root = precompute_root

    class _SatelliteToolEnv(vf.StatefulToolEnv):
        """Wraps SatelliteAgent tools so the deterministic per-rollout state
        (case_id, precompute case_dir, terminal flag) is injected into each
        tool call.
        """

        async def setup_state(self, state):
            # The pinned verifiers (prime-rl rev 77a9f28) does
            # `state = await self.setup_state(state)`, so we MUST return state.
            # Master verifiers later switched to "mutate in place, return ignored",
            # but returning state remains compatible with both.
            try:
                _debug_log(
                    f"setup_state ENTER state_type={type(state).__name__} "
                    f"state_keys={list(state.keys())[:20] if hasattr(state, 'keys') else 'NO_KEYS'} "
                    f"info_type={type(state.get('info')).__name__ if hasattr(state, 'get') else 'NO_GET'}"
                )
                info = state.get("info") or {}
                state["scenario"] = info.get("scenario", "")
                state["expected_action"] = info.get("expected_action", "")
                state["case_id"] = info.get("case_id", "")
                state["terminal_called"] = False
                state["terminal_action"] = None
                state["tool_call_log"] = []
                _debug_log(f"setup_state OK case_id={state.get('case_id')!r}")
                return state
            except Exception as _e:
                import traceback as _tb
                _debug_log(f"setup_state RAISED {_e!r}\n{_tb.format_exc()}")
                raise

        def update_tool_args(
            self,
            tool_name: str,
            tool_args: dict,
            messages,
            state,
            **kwargs,
        ) -> dict:
            try:
                _debug_log(
                    f"update_tool_args ENTER name={tool_name} "
                    f"args_keys={list(tool_args.keys()) if isinstance(tool_args, dict) else type(tool_args).__name__} "
                    f"state_type={type(state).__name__} "
                    f"state_has_get={hasattr(state, 'get')}"
                )
                # case_id may be missing if setup_state hasn't run for this state
                # (defensive); fall back to info.case_id.
                case_id = (state or {}).get("case_id")
                if not case_id:
                    case_id = ((state or {}).get("info") or {}).get("case_id", "")
                    if case_id and isinstance(state, dict):
                        state["case_id"] = case_id

                # Inject case_dir ONLY for precompute lookup tools. submit_to_ground
                # / drop have no `case_dir` parameter and would TypeError on the
                # injected kwarg.
                if _precompute_root and case_id and tool_name in _PRECOMPUTE_TOOL_NAMES:
                    tool_args["case_dir"] = os.path.join(_precompute_root, case_id)

                if isinstance(state, dict):
                    log = state.setdefault("tool_call_log", [])
                    logged_args = {k: v for k, v in tool_args.items() if k != "case_dir"}
                    log.append({"name": tool_name, "args": logged_args})
                    if tool_name in {"submit_to_ground", "drop"}:
                        state["terminal_called"] = True
                        state["terminal_action"] = tool_name
                _debug_log(f"update_tool_args OK name={tool_name} case_id={case_id!r}")
                return tool_args
            except Exception as _e:
                import traceback as _tb
                _debug_log(f"update_tool_args RAISED name={tool_name} {_e!r}\n{_tb.format_exc()}")
                raise

        @vf.stop
        async def terminal_tool_called(self, state, **kwargs):
            """Stop the rollout as soon as `submit_to_ground` or `drop` is
            called. Without this, vLLM with `tool_choice=required` would force
            the model to keep emitting tool calls every turn until max_turns
            is hit.
            """
            try:
                if state is None:
                    _debug_log("terminal_tool_called STATE_IS_NONE")
                    return False
                return bool(state.get("terminal_called"))
            except Exception as _e:
                import traceback as _tb
                _debug_log(f"terminal_tool_called RAISED {_e!r}\n{_tb.format_exc()}")
                raise

        async def rollout(self, *args, **kwargs):
            """Wrap parent rollout to dump a JSONL trace per rollout when
            ``SATELLITEAGENT_TRACE_PATH`` is set. Failures are swallowed so a
            broken tracer never breaks training.
            """
            try:
                _debug_log(
                    f"rollout ENTER nargs={len(args)} kwargs_keys={list(kwargs.keys())}"
                )
                result = await super().rollout(*args, **kwargs)
                _debug_log(
                    f"rollout RETURNED type={type(result).__name__} "
                    f"keys={list(result.keys())[:25] if hasattr(result, 'keys') else 'NO_KEYS'}"
                )
            except Exception as _e:
                import traceback as _tb
                _debug_log(f"rollout RAISED {_e!r}\n{_tb.format_exc()}")
                raise

            try:
                _maybe_dump_trace(result, args, kwargs)
            except Exception as _e:
                # Emit one stderr line so we notice silent breakage but never
                # raise into the orchestrator's gather.
                import sys as _sys
                print(f"[satelliteagent_env] trace dump failed: {_e!r}", file=_sys.stderr)
                _debug_log(f"_maybe_dump_trace RAISED {_e!r}")
            return result

    return _SatelliteToolEnv


def _maybe_dump_trace(rollout_result, call_args, call_kwargs) -> None:
    """If SATELLITEAGENT_TRACE_PATH is set, append one JSON line summarizing
    the rollout (case_id, terminal action, full message trace with images
    scrubbed, tool-call ledger). Best-effort; tolerant to verifiers shape
    changes.

    `MultiTurnEnv.rollout()` returns a `State` (dict-like) directly. We pull
    `state["completion"]` for the chat trace and read our setup_state-added
    fields (case_id, tool_call_log, terminal_*) from it.
    """
    import os, json, time, threading
    path = os.environ.get("SATELLITEAGENT_TRACE_PATH", "").strip()
    if not path:
        return

    # MultiTurnEnv.rollout() returns the State (dict). Tuple/list returns are
    # supported only as a safety net for non-MultiTurn envs.
    completion, state = None, None
    if isinstance(rollout_result, dict):
        state = rollout_result
        completion = state.get("completion") or state.get("messages")
    elif isinstance(rollout_result, tuple) and len(rollout_result) >= 2:
        completion, state = rollout_result[0], rollout_result[1]
    elif isinstance(rollout_result, list):
        completion = rollout_result

    state = state if isinstance(state, dict) else {}

    case_id = state.get("case_id") or (state.get("info") or {}).get("case_id")
    expected = state.get("expected_action")
    if not expected:
        exp = (state.get("info") or {}).get("expected") or {}
        expected = exp.get("action") if isinstance(exp, dict) else None

    msgs_summary = []
    if isinstance(completion, list):
        for m in completion:
            try:
                msgs_summary.append(_summarize_message(m))
            except Exception as e:
                msgs_summary.append({"_summary_error": repr(e)})

    entry = {
        "ts": time.time(),
        "pid": os.getpid(),
        "case_id": case_id,
        "expected_action": expected,
        "terminal_called": bool(state.get("terminal_called")),
        "terminal_action": state.get("terminal_action"),
        "tool_call_log": state.get("tool_call_log") or [],
        "n_messages": len(msgs_summary),
        "messages": msgs_summary,
    }

    line = json.dumps(entry, ensure_ascii=False, default=str)
    # Best-effort thread-safe append. Multiple workers may race; we accept
    # interleaved writes since each line is one full JSON object.
    lock = _maybe_dump_trace.__dict__.setdefault("_lock", threading.Lock())
    with lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# --- Precompute lookup tools (Phase 5b real-tool path) -------------------

def _read_yaml(path):
    import yaml as _yaml
    try:
        with open(path) as f:
            return _yaml.safe_load(f)
    except Exception:
        return None


def compute_index(
    index: str,
    which: str = "after",
    case_dir: str = "",
):
    """Compute a Sentinel-2 spectral index (NDVI/NDWI/MNDWI/NBR/NDBI/NDSI) and
    return its pseudo-color heatmap PNG path plus statistics. Bound to the
    current case automatically; `case_dir` is set by the env.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    stats_path = _os.path.join(case_dir, "compute_index", f"{index}__{which}.stats.yaml")
    png_path   = _os.path.join(case_dir, "compute_index", f"{index}__{which}.png")
    data = _read_yaml(stats_path)
    if data is None:
        return {"error": f"not cached: compute_index(index={index}, which={which})"}
    out = dict(data.get("response") or {})
    if _os.path.isfile(png_path):
        out["png_path"] = png_path
    return out


def compute_index_delta(
    index: str,
    case_dir: str = "",
):
    """Compute Δ = After - Before for a spectral index. Returns a diverging
    heatmap (red=decrease, blue=increase) PNG path plus delta stats. `case_dir`
    is set by the env.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    stats_path = _os.path.join(case_dir, "compute_index_delta", f"{index}.stats.yaml")
    png_path   = _os.path.join(case_dir, "compute_index_delta", f"{index}.png")
    data = _read_yaml(stats_path)
    if data is None:
        return {"error": f"not cached: compute_index_delta(index={index})"}
    out = dict(data.get("response") or {})
    if _os.path.isfile(png_path):
        out["png_path"] = png_path
    return out


def fetch_band(
    band: str,
    which: str = "after",
    case_dir: str = "",
):
    """Fetch a single Sentinel-2 grayscale band as PNG plus min/max/mean
    stats. `case_dir` is set by the env automatically.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    stats_path = _os.path.join(case_dir, "fetch_band", f"{band}__{which}.stats.yaml")
    png_path   = _os.path.join(case_dir, "fetch_band", f"{band}__{which}.png")
    data = _read_yaml(stats_path)
    if data is None:
        return {"error": f"not cached: fetch_band(band={band}, which={which})"}
    out = dict(data.get("response") or {})
    if _os.path.isfile(png_path):
        out["png_path"] = png_path
    return out


def false_color(
    bands: list[str],
    which: str = "after",
    case_dir: str = "",
):
    """Build an RGB false-color composite from 3 Sentinel-2 bands.
    Available combos in the precompute cache:
    nir-red-green / swir22-nir-red / swir16-nir-blue / nir-swir16-red /
    red-green-blue. `case_dir` is set by the env.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    if not isinstance(bands, list) or len(bands) != 3:
        return {"error": "`bands` must be a list of exactly 3 band names"}
    combo = "-".join(bands)
    png_path = _os.path.join(case_dir, "false_color", f"{combo}__{which}.png")
    if not _os.path.isfile(png_path):
        return {"error": f"not cached: false_color(bands={bands}, which={which})"}
    return {"png_path": png_path, "bands": bands, "which": which}


def classify_change(
    image_before: str = "before",
    image_after: str = "after",
    case_dir: str = "",
):
    """Run the offline change classifier (precomputed Gemini result) on the
    case's before/after pair. Returns candidate classes with confidences.
    `case_dir` is set by the env automatically; the image_* args are kept
    only to mirror the production schema.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    path = _os.path.join(case_dir, "classify_change.yaml")
    data = _read_yaml(path)
    if data is None:
        return {"error": "not cached: classify_change"}
    return data.get("response") or {}


_REAL_TOOLS = [compute_index, compute_index_delta, fetch_band, false_color, classify_change]


# --- Real dataset (Phase 5b: read raw canonical_dataset.yaml directly) ----

# Mirrors agent/react_loop.py SYSTEM_PROMPT so the env runs the model under
# the same system prompt as production RunAgent.
_REAL_SYSTEM_PROMPT = """You are an onboard satellite operator agent running on NVIDIA Orin 16GB.

You are shown a Sentinel-2 image pair (before = previous pass, after = current pass) of the same
location. Your job is to decide what to report to the ground station.

Goals:
- Minimize downlink bandwidth usage while preserving critical information.
- Attach the raw image only when text alone cannot convey the situation.
- Always terminate with exactly one of: submit_to_ground(...) or drop().

Style:
- Think briefly in natural language before each tool call (one sentence).
- One tool call per step.
- Stop as soon as you have enough information.
"""


def _parse_burn_area_km2(event_name: str | None) -> float | None:
    """Extract km² from an MCD64A1 event name like '18.03 km² burn'."""
    if not event_name:
        return None
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*km", event_name)
    return float(m.group(1)) if m else None


def _derive_expected(case: dict) -> dict:
    """Map a canonical_dataset.yaml row to a uniform `expected` dict.

    Positives in raw v1 only carry event area; we derive urgency / attach /
    change_type heuristically so action_match works today and the optional
    validators have something to grade once we turn them on. Negatives carry
    `expected_action: drop` directly.
    """
    if case.get("type") == "positive":
        area_km2 = _parse_burn_area_km2((case.get("event") or {}).get("name")) or 0.0
        urgency = "high" if area_km2 >= 100 else ("medium" if area_km2 >= 10 else "low")
        return {
            "action": "submit_to_ground",
            "attach_image": area_km2 >= 50.0,
            "urgency": urgency,
            "change_type": "wildfire",
        }
    # negative
    return {
        "action": case.get("expected_action", "drop"),
        "attach_image": False,
        "urgency": "low",
        "change_type": "none",
    }


def _image_part(path: str) -> dict:
    """Build an OpenAI-compatible chat-completions image part from a local
    file path. prime-rl orchestrator (trajectories.py) expects the
    `image_url` form with a `data:image/...;base64,...` URL; the raw
    `{type: image, path: ...}` shape is rejected by vLLM with
    `501 Unknown part type: image`.
    """
    import base64, os
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _real_dataset(data_root: str) -> "Dataset":
    import yaml
    from datasets import Dataset

    canonical_path = f"{data_root}/canonical_dataset.yaml"
    with open(canonical_path) as f:
        canonical = yaml.safe_load(f)

    rows: list[dict] = []
    for case in canonical.get("cases") or []:
        case_id = case["id"]
        case_dir = f"{data_root}/curated_pairs/{case_id}"
        before = f"{case_dir}/before.png"
        after  = f"{case_dir}/after.png"
        resolved = case.get("expected_resolved") or {}
        expected = _derive_expected(case)

        rows.append({
            "prompt": [
                # All `content` fields are lists of parts -- HF datasets rejects
                # mixed string/list content across rows of the same column with
                # "cannot mix list and non-list, non-null values".
                {"role": "system", "content": [
                    {"type": "text", "text": _REAL_SYSTEM_PROMPT},
                ]},
                {"role": "user", "content": [
                    {"type": "text",  "text": "Before image (previous satellite pass over this location):"},
                    _image_part(before),
                    {"type": "text",  "text": "After image (current pass, same location):"},
                    _image_part(after),
                    {"type": "text",  "text": (
                        "Analyze the pair and decide what to report to ground. "
                        "Use your tools. End with submit_to_ground(...) or drop()."
                    )},
                ]},
            ],
            "info": {
                "case_id": case_id,
                "type": case.get("type"),
                "context": {
                    "lat": case.get("lat"),
                    "lon": case.get("lon"),
                    "size_km": case.get("size_km"),
                    "before_ts": resolved.get("before_datetime"),
                    "after_ts":  resolved.get("after_datetime"),
                },
                "expected": expected,
            },
        })

    return Dataset.from_list(rows)


def _real_rubric(weights: dict | None = None) -> "vf.Rubric":
    """Build the Rubric from eval.validators.common.

    Default weights (S16 v6 traces showed action_match alone gives no signal
    to a 450M model that fails to call tools with valid args):
    - action_match: 1.0 — primary correctness signal.
    - valid_tool_args: 0.3 — partial credit for calling lookup tools with
        args the env actually accepts (index='NBR', band='B11', etc.).
        Crucial for small VLMs that otherwise loop on hallucinated args.
    - terminal_reached: 0.1 — small bias toward actually terminating
        (rather than running out of max_turns).
    - attach_image / urgency / change_type: weight 0 (off until basic loop
        learns to terminate correctly).
    """
    import verifiers as vf
    from eval.validators.common import (
        action_match,
        attach_image_match,
        urgency_match,
        change_type_match,
        valid_tool_args,
        terminal_reached,
    )

    w = {
        "action": 1.0,
        "valid_tool_args": 0.3,
        "terminal_reached": 0.1,
        "attach": 0.0,
        "urgency": 0.0,
        "change_type": 0.0,
    }
    if weights:
        w.update(weights)
    return vf.Rubric(
        funcs=[
            action_match,
            valid_tool_args,
            terminal_reached,
            attach_image_match,
            urgency_match,
            change_type_match,
        ],
        weights=[
            w["action"],
            w["valid_tool_args"],
            w["terminal_reached"],
            w["attach"],
            w["urgency"],
            w["change_type"],
        ],
    )


def load_environment(
    toy: bool = True,
    data_root: str | None = None,
    precompute_root: str | None = None,
    rubric_weights: dict | None = None,
    max_turns: int = 1,
    **kwargs: Any,
) -> "vf.Environment":
    """Entry point invoked by `verifiers.load_environment("satelliteagent_env")`.

    Args:
        toy: when True use the hand-crafted submit-or-drop toy dataset.
            When False, read `canonical_dataset.yaml` from `data_root` and
            build the real dataset.
        data_root: required when toy=False. Path to the directory containing
            `canonical_dataset.yaml` and `curated_pairs/<case_id>/...`
            (`<KAGGLE_USER>/satelliteagent-raw-v1`).
        precompute_root: optional. Path to the precompute cache
            (`<KAGGLE_USER>/satelliteagent-precompute-v1`). When set, the env
            exposes `compute_index` / `compute_index_delta` / `fetch_band` /
            `false_color` / `classify_change` as offline lookup tools that
            read PNG/YAML from `<precompute_root>/<case_id>/...`. When
            unset, only `submit_to_ground` and `drop` are exposed.
        rubric_weights: optional dict overriding default per-validator weights.
        max_turns: orchestrator multi-turn cap. 1 = single tool_call (smoke),
            >=2 = lookup tools usable before final terminal action.
    """
    _import_breadcrumb()  # second breadcrumb: load_environment was actually called
    try:
        import os as _o, time as _t
        with open("/kaggle/working/outputs/satelliteagent_import.log", "a", encoding="utf-8") as _f:
            _f.write(
                f"{_t.time():.3f} pid={_o.getpid()} load_environment ENTER "
                f"toy={toy} data_root={data_root!r} precompute_root={precompute_root!r} "
                f"max_turns={max_turns}\n"
            )
    except Exception: pass

    SatelliteToolEnv = _build_env_class(precompute_root)
    # Terminal tools always exposed. Precompute lookup tools are added via
    # `add_tool(..., args_to_skip=["case_dir"])` so the model NEVER sees
    # `case_dir` in the schema (env injects it from state.case_id at call
    # time). Otherwise the model hallucinates `case_dir="case_dir"` literally.
    base_tools: list[Callable] = [_expose_for_vf(submit_to_ground), _expose_for_vf(drop)]

    if toy:
        env = SatelliteToolEnv(
            dataset=_toy_dataset(),
            tools=base_tools,
            rubric=_toy_rubric(),
            max_turns=max_turns,
        )
        if precompute_root:
            for t in _REAL_TOOLS:
                env.add_tool(t, args_to_skip=["case_dir"])
        return env

    if not data_root:
        raise ValueError("load_environment(toy=False) requires data_root=<path>")

    env = SatelliteToolEnv(
        dataset=_real_dataset(data_root),
        tools=base_tools,
        rubric=_real_rubric(rubric_weights),
        # When precompute_root is set, lookup tools should be reachable
        # before the terminal action -> caller passes max_turns >= 2.
        # Without precompute_root, max_turns=1 forces a clean 1-turn rollout
        # (assistant emits one terminal tool_call, env stops via @vf.stop).
        max_turns=max_turns,
    )
    try:
        import os as _o, time as _t
        with open("/kaggle/working/outputs/satelliteagent_import.log", "a", encoding="utf-8") as _f:
            _f.write(
                f"{_t.time():.3f} pid={_o.getpid()} env_constructed "
                f"tool_map_keys={list(getattr(env, 'tool_map', {}).keys())}\n"
            )
    except Exception: pass
    if precompute_root:
        for t in _REAL_TOOLS:
            try:
                env.add_tool(t, args_to_skip=["case_dir"])
            except Exception as _e:
                import traceback as _tb, os as _o, time as _t
                with open("/kaggle/working/outputs/satelliteagent_import.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        f"{_t.time():.3f} pid={_o.getpid()} add_tool RAISED "
                        f"tool={getattr(t, '__name__', '?')} err={_e!r}\n{_tb.format_exc()}\n"
                    )
                raise
    try:
        import os as _o, time as _t
        with open("/kaggle/working/outputs/satelliteagent_import.log", "a", encoding="utf-8") as _f:
            _f.write(
                f"{_t.time():.3f} pid={_o.getpid()} load_environment RETURN "
                f"tool_map_keys={list(getattr(env, 'tool_map', {}).keys())}\n"
            )
    except Exception: pass
    return env
