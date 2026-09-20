# Paired batch comparison

AgentTraceLab compares baseline and candidate batches by `scenario_id`. This prevents an easy new
task from hiding a regression in an existing task and prevents a removed task from silently
disappearing from the denominator.

```bash
uv run agenttracelab compare-wasmhatch-batches \
  path/to/baseline/manifest.json \
  path/to/candidate/manifest.json \
  --output evaluation-results/comparison.json \
  --summary evaluation-results/comparison.md
```

The comparison reports:

- baseline-only and candidate-only scenarios;
- run-level pass-to-fail regressions and fail-to-pass improvements;
- deterministic check regressions and improvements for each paired scenario;
- per-scenario score changes and their paired mean;
- whether both batches use the same evidence mode.

`promote` requires at least one paired scenario, no removed baseline scenario, no run or check
regression, a fully passing candidate batch, and matching evidence modes. New candidate-only
scenarios remain visible but do not contribute to the paired mean.

The paired mean is descriptive only. AgentTraceLab does not claim statistical significance from a
small batch, and it does not treat synthetic fixtures as live provider evidence.
