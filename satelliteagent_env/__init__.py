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


def _build_env_class():
    """Build the StatefulToolEnv subclass at call time so `verifiers` import
    is deferred. Returns the class object."""
    import verifiers as vf

    class _SatelliteToolEnv(vf.StatefulToolEnv):
        """Wraps SatelliteAgent tools so the deterministic per-rollout state
        (cache lookups + downlink budget) is injected into each tool call.

        For the Phase 5 toy we do not need state injection -- submit_to_ground
        and drop are pure functions of their declared args. Phase 2 real env
        will set up case caches here.
        """

        async def setup_state(self, state):
            info = state.get("info") or {}
            state["scenario"] = info.get("scenario", "")
            state["expected_action"] = info.get("expected_action", "")
            return state

        def update_tool_args(self, tool_name: str, tool_args: dict, *args, **kwargs) -> dict:
            # verifiers' StatefulToolEnv has changed the trailing positional args
            # of update_tool_args between releases (state, messages, ...). The
            # toy env doesn't need any of them -- pure functions of tool_args --
            # so absorb anything extra to stay forward/backward compatible.
            return tool_args

    return _SatelliteToolEnv


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
                {"role": "system", "content": _REAL_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text",  "text": "Before image (previous satellite pass over this location):"},
                    {"type": "image", "path": before},
                    {"type": "text",  "text": "After image (current pass, same location):"},
                    {"type": "image", "path": after},
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
    """Build the Rubric from eval.validators.common. Phase 5b smoke uses
    only `action_match`; the optional validators are wired but weight=0 so
    they don't affect learning until we turn them on explicitly.
    """
    import verifiers as vf
    from eval.validators.common import (
        action_match,
        attach_image_match,
        urgency_match,
        change_type_match,
    )

    w = {"action": 1.0, "attach": 0.0, "urgency": 0.0, "change_type": 0.0}
    if weights:
        w.update(weights)
    return vf.Rubric(
        funcs=[action_match, attach_image_match, urgency_match, change_type_match],
        weights=[w["action"], w["attach"], w["urgency"], w["change_type"]],
    )


def load_environment(
    toy: bool = True,
    data_root: str | None = None,
    rubric_weights: dict | None = None,
    **kwargs: Any,
) -> "vf.Environment":
    """Entry point invoked by `verifiers.load_environment("satelliteagent_env")`.

    Args:
        toy: when True use the hand-crafted submit-or-drop toy dataset.
            When False, read `canonical_dataset.yaml` from `data_root` and
            build the real Phase 5b dataset (Phase 2 still needs to grow
            negative coverage and add precompute_tool_responses, but the
            shape is right for an end-to-end smoke).
        data_root: required when toy=False. Path to the directory containing
            `canonical_dataset.yaml` and `curated_pairs/<case_id>/...`.
            On Kaggle this is `/kaggle/input/satelliteagent-raw-v1`.
        rubric_weights: optional dict overriding default per-validator weights
            (keys: "action", "attach", "urgency", "change_type").
    """
    SatelliteToolEnv = _build_env_class()
    tools: list[Callable] = [_expose_for_vf(submit_to_ground), _expose_for_vf(drop)]

    if toy:
        return SatelliteToolEnv(
            dataset=_toy_dataset(),
            tools=tools,
            rubric=_toy_rubric(),
            max_turns=3,
        )

    if not data_root:
        raise ValueError("load_environment(toy=False) requires data_root=<path>")

    return SatelliteToolEnv(
        dataset=_real_dataset(data_root),
        tools=tools,
        rubric=_real_rubric(rubric_weights),
        max_turns=3,
    )
