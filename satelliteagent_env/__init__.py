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
    # Vision tools that read precomputed YAML/PNG from <case_dir>/...
    "compute_index", "compute_index_delta", "fetch_band", "false_color",
    "classify_change", "zoom_in",
    # Context tools that read per-case info from <case_dir>/...
    "get_region_info", "get_history",
    # Action drafting & budget tools also receive case_dir (ignored or
    # used for state binding) — keeping them in the set lets the env
    # inject case_dir uniformly without breaking the tool functions.
    "compute_area", "check_downlink_budget", "estimate_size", "compose_report",
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


_VALID_INDICES = ["NBR", "NDVI", "NDWI", "MNDWI", "NDBI", "NDSI"]
_VALID_WHICHES = ["before", "after"]
# Sentinel-2 band names as used in the precompute cache
# (human-readable aliases, NOT the B2/B3/.../B12 codes).
_VALID_BANDS = [
    "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09",
    "swir16", "swir22",
]
_FALSE_COLOR_COMBOS = {
    "natural":    "red-green-blue",
    "color-ir":   "nir-red-green",
    "burn":       "swir22-nir-red",
    "vegetation": "swir16-nir-blue",
    "water":      "nir-swir16-red",
}


def compute_index(
    index: str = "",
    case_dir: str = "",
):
    """Sentinel-2 spectral index statistics over the scene, for both
    snapshots (before and after).

    All arguments are optional — calling with no arguments returns ALL
    six indices for both snapshots in one shot, which is the simplest
    way to get an overview.

    Args:
        index (string, optional): name of one specific spectral index to
            return. When omitted (default), every index below is returned.
              - "NBR"   — Normalized Burn Ratio (B8, B12). Sensitive to
                          burn scars and charred / dry surfaces.
              - "NDVI"  — Vegetation index (B8, B4). High on green plants.
              - "NDWI"  — Water index (B3, B8). High on open water.
              - "MNDWI" — Modified water index (B3, B11). Robust to
                          built-up areas.
              - "NDBI"  — Built-up index (B11, B8). High on urban surface.
              - "NDSI"  — Snow/ice index (B3, B11). High on snow.

    Returns:
        dict mapping `<INDEX>_<which>` to its stats {min, max, mean,
        median, frac_decrease_strong, frac_increase_strong} plus
        `png_path`. E.g. result["NBR_after"]["mean"].

    Example:
        compute_index() → all six indices × {before, after}
        compute_index(index="NBR") → only NBR for both snapshots

    A single-timepoint index tells you what is there NOW. To detect a
    CHANGE between the two passes, use `compute_index_delta` instead.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    sel = [index] if index in _VALID_INDICES else _VALID_INDICES
    out = {}
    for idx in sel:
        for wh in _VALID_WHICHES:
            stats_path = _os.path.join(
                case_dir, "compute_index", f"{idx}__{wh}.stats.yaml"
            )
            png_path = _os.path.join(
                case_dir, "compute_index", f"{idx}__{wh}.png"
            )
            data = _read_yaml(stats_path)
            if data is None:
                continue
            response = dict(data.get("response") or {})
            if _os.path.isfile(png_path):
                response["png_path"] = png_path
            out[f"{idx}_{wh}"] = response
    return out if out else {"error": "no compute_index data cached"}


def compute_index_delta(
    index: str = "",
    case_dir: str = "",
):
    """Δ = (After − Before) for Sentinel-2 spectral indices. Use this
    whenever the question is about CHANGE between the two passes (burn,
    flood, deforestation, urbanization, snow melt).

    `index` is optional. When omitted, deltas for ALL six indices are
    returned in one shot (overview). Specify `index` to focus on one.

    Args:
        index (string, optional): name of one spectral index to delta.
            When omitted (default), all six indices are returned.
              - "NBR"   — Δ drops (negative) on burn / charring.
              - "NDVI"  — Δ drops on vegetation loss (deforestation,
                          burn scar, drought, harvest).
              - "NDWI"  — Δ rises on flooding, drops on drying.
              - "MNDWI" — Same as NDWI but more robust to built-up land.
              - "NDBI"  — Δ rises on new built-up surfaces.
              - "NDSI"  — Δ tracks snow gain / loss.

    Returns:
        dict mapping each `<INDEX>` to its delta stats:
          - mean — Δ averaged over the scene (sign + magnitude).
          - frac_decrease_strong — fraction of pixels strongly negative.
          - frac_increase_strong — fraction of pixels strongly positive.
          - png_path — diverging heatmap (red = decrease, blue = increase).

    Example:
        compute_index_delta() → all six indices' deltas
        compute_index_delta(index="NBR") → only NBR's delta
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    sel = [index] if index in _VALID_INDICES else _VALID_INDICES
    out = {}
    for idx in sel:
        stats_path = _os.path.join(
            case_dir, "compute_index_delta", f"{idx}.stats.yaml"
        )
        png_path = _os.path.join(
            case_dir, "compute_index_delta", f"{idx}.png"
        )
        data = _read_yaml(stats_path)
        if data is None:
            continue
        response = dict(data.get("response") or {})
        if _os.path.isfile(png_path):
            response["png_path"] = png_path
        out[idx] = response
    return out if out else {"error": "no compute_index_delta data cached"}


def fetch_band(
    band: str = "",
    case_dir: str = "",
):
    """Sentinel-2 surface-reflectance bands as grayscale PNGs plus stats.
    Lower-level than `compute_index`; only use this when you need to see
    a raw band in isolation.

    `band` is optional. When omitted, all six bands' (after) stats are
    returned in one shot. The before snapshot is also included when
    cached.

    Args:
        band (string, optional): one of the human-readable Sentinel-2
            band aliases:
              - "blue"      (~490 nm)
              - "green"     (~560 nm)
              - "red"       (~665 nm)
              - "rededge1"  (~705 nm)
              - "rededge2"  (~740 nm)
              - "rededge3"  (~783 nm)
              - "nir"       (~842 nm). Vegetation = bright; water and
                            burn scars = dark.
              - "nir08"     (~865 nm)
              - "nir09"     (~945 nm)
              - "swir16"    (~1610 nm). Burned / water-stressed bright.
              - "swir22"    (~2200 nm). Strongest fire / burn-scar
                            response.
            When omitted (default), every band is returned.

    Returns:
        dict mapping `<band>_<which>` to {min, max, mean, std} + png_path.
        E.g. result["swir22_after"]["mean"].

    Example:
        fetch_band() → every band × {before, after}
        fetch_band(band="swir22") → only SWIR2 for both snapshots
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    sel = [band] if band in _VALID_BANDS else _VALID_BANDS
    out = {}
    for b in sel:
        for wh in _VALID_WHICHES:
            stats_path = _os.path.join(case_dir, "fetch_band", f"{b}__{wh}.stats.yaml")
            png_path   = _os.path.join(case_dir, "fetch_band", f"{b}__{wh}.png")
            data = _read_yaml(stats_path)
            if data is None:
                continue
            response = dict(data.get("response") or {})
            if _os.path.isfile(png_path):
                response["png_path"] = png_path
            out[f"{b}_{wh}"] = response
    return out if out else {"error": "no fetch_band data cached"}


def false_color(
    combo: str = "",
    case_dir: str = "",
):
    """RGB false-color composites of the after-pass scene. Useful for
    visual confirmation when a spectral index has flagged something —
    different band orderings make different surface types salient.

    `combo` is optional and is a single keyword (not a band list).
    Without `combo`, all five composites are returned at once.

    Args:
        combo (string, optional): which composite to return. One of:
              - "natural"     — true-color (R=red, G=green, B=blue).
              - "color-ir"    — color-IR (NIR/Red/Green). Vegetation = red.
              - "burn"        — burn-scar emphasis (SWIR2/NIR/Red). Burned
                                land = dark red / brown.
              - "vegetation"  — vegetation health (SWIR1/NIR/Blue).
              - "water"       — water + vegetation contrast (NIR/SWIR1/Red).
            When omitted (default), all five composites are returned.

    Returns:
        dict mapping each combo keyword to {png_path, bands}.

    Example:
        false_color() → all five composites
        false_color(combo="burn") → only the burn-emphasis view
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    sel = [combo] if combo in _FALSE_COLOR_COMBOS else list(_FALSE_COLOR_COMBOS.keys())
    out = {}
    for kw in sel:
        c = _FALSE_COLOR_COMBOS[kw]
        png_path = _os.path.join(case_dir, "false_color", f"{c}__after.png")
        if _os.path.isfile(png_path):
            out[kw] = {"png_path": png_path, "bands": c.split("-")}
    return out if out else {"error": "no false_color composites cached"}


def classify_change(
    image_before: str = "before",
    image_after: str = "after",
    case_dir: str = "",
):
    """Ask a general-purpose VLM (precomputed Gemini result) what kind of
    change is visible between the before/after RGB images. This is a
    coarse, RGB-only opinion — useful as a prior, but it can be wrong on
    subtle changes (e.g. burn scars often look like 'no_change' or 'cloud'
    in plain RGB). Cross-check with spectral evidence when stakes are high.

    Args:
        image_before, image_after: ignored — the env binds the current
            case automatically. Pass any non-empty strings (kept for
            schema parity with the production tool).

    Returns: dict with `classes` (list of {name, confidence}) and
    optional `bboxes`. Class names are free-form (e.g. "no_change",
    "cloud", "fire", "flood", "deforestation").
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context (env did not inject case_dir)"}
    path = _os.path.join(case_dir, "classify_change.yaml")
    data = _read_yaml(path)
    if data is None:
        return {"error": "not cached: classify_change"}
    return data.get("response") or {}


# --- Additional production tools (TOOL_SPEC: zoom_in / context / budget /
# action category). All argument-light or argument-free so the small VLM
# can call them without arg-hallucination. ----------------------------------

_ZOOM_TARGETS = {"center", "top-left", "top-right", "bottom-left", "bottom-right", "auto"}


def zoom_in(target: str = "center", case_dir: str = ""):
    """Return a zoomed-in view of part of the scene (before/after pair).

    Args:
        target (string, optional): which part to zoom into. Keyword:
              - "center" / "auto" — middle of the scene (default)
              - "top-left", "top-right",
                "bottom-left", "bottom-right" — quadrants

    Returns:
        dict with `before_png`, `after_png`, `target`. The PNGs are the
        full-scene paths; in production the env crops them.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context"}
    t = target if target in _ZOOM_TARGETS else "center"
    return {
        "target": t,
        "note": "training stub: full-scene PNGs returned without crop.",
        "before_png": _os.path.join(case_dir, "..", "before.png"),
        "after_png":  _os.path.join(case_dir, "..", "after.png"),
    }


def get_region_info(case_dir: str = ""):
    """Return geographic context for the current case.

    Args:
        (none — the env auto-binds the case.)

    Returns:
        dict with `region`, `country`, `populated`, `infra_nearby`,
        `lat`, `lon`. Read region context from the precompute cache;
        falls back to "unknown" entries when not cached.
    """
    import os as _os
    if not case_dir:
        return {"error": "no case context"}
    path = _os.path.join(case_dir, "region_info.yaml")
    data = _read_yaml(path)
    if data is not None:
        return data.get("response") or {}
    # fallback when not cached: harmless empty context
    return {
        "region": "unknown",
        "country": "unknown",
        "populated": None,
        "infra_nearby": [],
        "lat": None,
        "lon": None,
    }


def get_history(days: int = 30, case_dir: str = ""):
    """Return prior onboard reports for this location, last N days.

    Args:
        days (int, optional): lookback window in days. Default 30.

    Returns:
        dict with `history` (list of {timestamp, report_id, summary})
        and `days_searched`. In training there is no prior history, so
        the list is empty.
    """
    return {"history": [], "days_searched": int(days) if isinstance(days, int) else 30}


def compute_area(target: str = "scene", case_dir: str = ""):
    """Return the area in km^2 of the requested region.

    Args:
        target (string, optional): "scene" (default — full scene) or one
            of the zoom targets ("center", "top-left", ...). For a
            quadrant the area is roughly 1/4 of the full scene.

    Returns:
        dict with `area_km2`, `target`.
    """
    SCENE_AREA_KM2 = 25.0  # roughly 5km × 5km crop
    factor = 1.0 if target in ("scene", "auto", "center") else 0.25
    return {"target": target, "area_km2": round(SCENE_AREA_KM2 * factor, 2)}


def check_downlink_budget(case_dir: str = ""):
    """Return remaining downlink budget for the current pass window.

    Args:
        (none.)

    Returns:
        dict with `bytes_remaining`, `seconds_until_window_close`.
    """
    return {
        "bytes_remaining": 4_200_000,
        "seconds_until_window_close": 180,
    }


def estimate_size(with_image: bool = False, case_dir: str = ""):
    """Estimate the size in bytes of a report transmission.

    Args:
        with_image (bool, optional): True if the report attaches the
            raw image (~420 KB extra). Default False.

    Returns:
        dict with `bytes`, `with_image`.
    """
    base = 2000
    img = 420_000 if with_image else 0
    return {"bytes": base + img, "with_image": bool(with_image)}


def compose_report(
    change_type: str = "no_change",
    urgency: str = "low",
    attach_image: bool = False,
    case_dir: str = "",
):
    """Draft an onboard report locally. This DOES NOT transmit; you
    must follow it with `submit_to_ground` to actually downlink.

    Args:
        change_type (string, optional): label for the change. Free-form
            (e.g. "fire", "flood", "deforestation", "no_change").
        urgency (string, optional): one of "low", "medium", "high".
        attach_image (bool, optional): whether the planned report
            attaches the raw image.

    Returns:
        dict with `report_id` (auto-generated, pass it to
        `submit_to_ground`), plus the fields you provided.
    """
    import time as _time, random as _random
    rid = f"r-{int(_time.time())}-{_random.randint(1000, 9999)}"
    return {
        "report_id": rid,
        "change_type": change_type,
        "urgency": urgency,
        "attach_image": bool(attach_image),
    }


_REAL_TOOLS = [
    # Vision (6)
    classify_change,
    compute_index,
    compute_index_delta,
    fetch_band,
    false_color,
    zoom_in,
    # Context (3)
    get_region_info,
    get_history,
    compute_area,
    # Budget (2)
    check_downlink_budget,
    estimate_size,
    # Action drafting (1) — submit_to_ground / drop come from base_tools
    compose_report,
]


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

Approach:
- The question is whether something noteworthy CHANGED between before and after.
  Single-timepoint signals tell you what is there now; tools whose name contains
  "delta" / "change" tell you what is different. Use the latter when the
  question is about change.
- All lookup tools accept their main argument as OPTIONAL — calling with no
  arguments returns an overview (e.g. `compute_index_delta()` returns deltas
  for every index in one shot). Specify an argument only when you want to
  focus on one item.
- Each spectral index in `compute_index` / `compute_index_delta` is sensitive
  to a different kind of surface (vegetation, water, burn, built-up, snow).
  Read the per-tool docstrings; pick the index whose definition matches the
  kind of change you care about.
- A single tool call usually isn't enough to be confident — corroborate one
  signal with another (e.g. a numerical delta with a false-color visual,
  or with regional context from `get_region_info`).

Style:
- Think briefly in natural language before each tool call (one sentence
  explaining what you expect to see and why).
- One tool call per step.
- Stop as soon as you have enough evidence; don't keep calling tools after
  the picture is clear.
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
    """Single-reward rubric (S25): one class-balanced action_match.

    Earlier rubrics combined action_match / grounded_action_match /
    valid_tool_args / terminal_reached and the model learned to game the
    grounded reward by calling any cheap tool (compute_area) before
    submitting. With this minimal rubric, the only signal is whether the
    final action is correct, weighted to neutralize the dataset's
    positive prior (51 pos : 16 neg). See `balanced_action_match`.

    The other validators stay importable but are not in the active rubric.
    """
    import verifiers as vf
    from eval.validators.common import balanced_action_match

    return vf.Rubric(
        funcs=[balanced_action_match],
        weights=[1.0],
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
