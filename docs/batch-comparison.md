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
- per-scenario process/recommended reward changes and their paired means;
- scenarios that introduce or clear a safety reward block;
- whether both batches use the same evidence mode.

`promote` requires at least one paired scenario, no removed baseline scenario, no run or check
regression, a fully passing candidate batch, and matching evidence modes. New candidate-only
scenarios remain visible but do not contribute to the paired mean.

The paired mean is descriptive only. AgentTraceLab does not claim statistical significance from a
small batch, and it does not treat synthetic fixtures as live provider evidence.

## Checked-in optimization experiment

`evaluation/experiments/wasmhatch-optimization/v1` contains a deterministic five-scenario suite
generated from the complete WasmHatch export fixture. It injects one bounded failure class at a time
and pairs each baseline with a corrected candidate. The generator and fixtures are checked in so the
reported score, process reward, recommended reward, and safety-transition deltas can be reproduced.

This suite is `synthetic_fixture` evidence. Replace its journals with `recorded_local` exports before
using the same report format for portfolio or resume claims about real executions.

`evaluation/recorded/wasmhatch-local-demo/v1` contains separately captured baseline and candidate
browser exports. The baseline scored 70/100 because it lacked structured decision evidence, a
decision-to-script link, and post-commit readback evidence. After WasmHatch recorded those three
facts, the candidate scored 100/100 with process and recommended reward moving from 0.7 to 1.0.
Approval, credential redaction, metric consistency, and terminal-state checks passed in both runs.

This is a one-scenario instrumentation result. It proves the evaluator can drive and verify a
specific engineering correction; it does not establish model-quality or production-reliability
improvements.

The separate `evaluation/recorded/wasmhatch-local-campaign/v1` manifest broadens the corrected
candidate into three independently exported browser runs: local normalization, invoice
reconciliation, and imported-CSV normalization. All three score 100/100 and receive process and
recommended reward 1.0. The campaign checks repeatability of the deterministic workflow contract;
it remains too small and too provider-free to support a model-quality claim.
