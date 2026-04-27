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


def load_environment(toy: bool = True, **kwargs: Any) -> "vf.Environment":
    """Entry point invoked by `verifiers.load_environment("satelliteagent_env")`.

    Args:
        toy: when True (default) use the hand-crafted submit-or-drop toy
            dataset to exercise the tool-call + reward pipeline. When False
            use the (Phase 2 TODO) real triplet data + full tool set.
    """
    SatelliteToolEnv = _build_env_class()

    if toy:
        tools: list[Callable] = [_expose_for_vf(submit_to_ground), _expose_for_vf(drop)]
        return SatelliteToolEnv(
            dataset=_toy_dataset(),
            tools=tools,
            rubric=_toy_rubric(),
            max_turns=3,
        )

    # Phase 2 path: real triplets + full STUB_TOOLS / real tools
    raise NotImplementedError(
        "Real (non-toy) env requires Phase 2 eval/cases triplets and "
        "eval/validators/common.py to be in place."
    )
