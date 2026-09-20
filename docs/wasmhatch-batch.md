# WasmHatch export evaluation

WasmHatch is a browser-only application. Its `wasmhatch.run-journal.v1` contract intentionally
requires an explicit local download and states that the journal does not authorize unattended
replay. AgentTraceLab therefore consumes exported journals without adding a hidden upload path or
pretending that WasmHatch exposes an HTTP backend.

## Batch manifest

Create a manifest beside the exported JSON files:

```json
{
  "schema_version": "agenttracelab.wasmhatch-batch.v1",
  "batch_id": "office-agent-pilot-01",
  "version": "1.0.0",
  "evidence_mode": "recorded_local",
  "entries": [
    {"scenario_id": "clean-orders", "journal": "journals/clean-orders.json"},
    {"scenario_id": "reject-unsafe-write", "journal": "journals/reject-unsafe-write.json"}
  ]
}
```

`evidence_mode` must be declared as `synthetic_fixture`, `recorded_local`, or `live_provider`.
AgentTraceLab preserves that label in every report and does not infer that an exported journal is a
live provider run.

Journal paths must stay inside the manifest directory. A batch is limited to 100 unique scenarios,
and each journal is limited to WasmHatch's 512 KiB contract.

## Promotion gate

```bash
uv run agenttracelab evaluate-wasmhatch-batch path/to/manifest.json \
  --policy evaluation/policies/recorded-release.v1.json \
  --output evaluation-results/batch.json \
  --summary evaluation-results/summary.md
```

The gate checks sample size, pass rate, failed-run count, evidence mode, forbidden failure
categories, conflicts, and uncertain outcomes. It exits with code `0` only on `promote`; `hold`
exits with code `6`, so the command can be used as a CI release gate.

Aggregate metrics are re-derived from events rather than trusted from the journal's summary object.
The deterministic evaluator also compares the exported summary with the event-derived counts and
reports `wasmhatch.metrics_consistent` when they diverge.
