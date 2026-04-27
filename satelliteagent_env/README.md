# satelliteagent_env

prime-rl / verifiers Environment package for SatelliteAgent.

## Single source of truth (no code duplication)

This package is **glue only**:

- Tool functions live in `tools/` (SatelliteAgent root). Import as
  `from tools.stubs import STUB_TOOLS` (Phase 1) or `from tools.vision import
  classify_change` (Phase 2+ real impls).
- Reward functions live in `eval/validators/common.py` (Phase 2 TODO).
  Import here and wrap as verifiers `Rubric` reward funcs.
- Triplet dataset lives in `eval/cases/triplets/*.yaml` (Phase 2 TODO).
  Read here and convert to a `datasets.Dataset` row format.

When `tools/` or `eval/validators/` change, this glue picks them up
automatically. The only manual update needed here is when:
- A tool's signature changes (rare)
- A new reward function is added to the rubric
- The dataset row schema changes

## Loaded by prime-rl as

```toml
[[orchestrator.train.env]]
id = "satelliteagent_env"
```

verifiers internally calls
`importlib.import_module("satelliteagent_env").load_environment(**kwargs)`.

## Status

| Component | State |
|---|---|
| Package scaffold + `load_environment` | ✅ Phase 5 stub |
| Tool wiring (STUB_TOOLS) | ✅ |
| `StatefulToolEnv` skeleton | ✅ minimal `setup_state` / `update_tool_args` |
| Real dataset (triplet YAMLs) | ❌ depends on Phase 2 |
| Real rubric (validators) | ❌ depends on Phase 2 |
| Rollout cache injection | ❌ depends on Phase 5 (precompute_tool_responses.py) |
| Kaggle prep wheel build | ❌ next: add `pip wheel SatelliteAgent` to prep |

See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) §7-8 for the full FT
pipeline design and [../kaggle/EXPERIMENT_PLAN.md](../kaggle/EXPERIMENT_PLAN.md)
for Stage 5 plan.

## Smoke test (after Phase 2 lands)

```bash
cd SatelliteAgent
pip install -e .
python -c "import satelliteagent_env; env = satelliteagent_env.load_environment(); print(env)"
```

For a Kaggle offline run:

1. prep notebook: `pip wheel . --no-deps -w wheels/` builds `satellite_agent-0.1.0-py3-none-any.whl`
2. S5 notebook installs it via `pip install --no-index --find-links wheels satellite-agent`
3. orchestrator config: `id = "satelliteagent_env"`
