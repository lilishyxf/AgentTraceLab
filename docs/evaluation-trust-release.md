# Evaluation trust release gate

This gate keeps four questions separate:

1. Did the recorded process satisfy deterministic safety and lifecycle rules?
2. Does the exact Judge agree with adjudicated humans on held-out cases?
3. Did an independent reader observe an accepted final state without protected-state damage?
4. Did the target avoid duplicate effects and false completion during runtime faults?

A high answer score or a clean trace cannot substitute for the other layers.

## Offline sequence

Create a fresh output root. Evidence commands refuse overwrite so every rerun receives a new path.

```powershell
$run = Join-Path $env:TEMP ("agenttracelab-trust-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $run | Out-Null

uv run agenttracelab analyze-judge-calibration-v2 `
  evaluation/judge-calibration/offline-v2/manifest.json `
  --output (Join-Path $run "calibration-report.json") `
  --summary (Join-Path $run "calibration-summary.md")

uv run agenttracelab run-runtime-faults `
  evaluation/runtime-faults/offline-v1/manifest.json `
  --output-directory (Join-Path $run "runtime-faults")
```

For final-state evaluation, reset one checked-in file or SQLite fixture with the matching adapter,
save the returned initial snapshot, run or import the target result, save the final snapshot, and run:

```powershell
uv run agenttracelab verify-final-state CONTRACT `
  --initial INITIAL_SNAPSHOT `
  --final FINAL_SNAPSHOT `
  --output (Join-Path $run "final-state-report.json") `
  --summary (Join-Path $run "final-state-summary.md")
```

Finally create `trust-manifest.json` beside copies of the eight exact JSON artifacts and their raw
SHA-256 values:

- calibration manifest and report;
- final-state contract, initial snapshot, final snapshot, and report;
- runtime-fault manifest and report.

The manifest also declares the expected Judge key/provider/model/prompt, `evidence_scope` as
`offline_harness` or `target_integration`, and `review_status` as `unreviewed` or `human_reviewed`.
Then run:

```powershell
uv run agenttracelab audit-trust-evidence TRUST_MANIFEST `
  --output (Join-Path $run "trust-readiness.json") `
  --summary (Join-Path $run "trust-readiness.md")
```

## Decision semantics

- Missing, unreadable, path-escaping, schema-invalid, or hash-mismatched artifacts fail or become
  unknown; neither state can produce readiness.
- Calibration report/manifest hashes and Judge identity must match exactly.
- Final-state contract, fixture identity, initial/final snapshots, environment identity, and report
  must agree; protected-state checks must pass.
- Runtime-fault report, manifest, runner version, and environment snapshot must agree; campaign gates
  require zero duplicates, zero false completions, no unresolved runs, and complete linkage.
- Complete offline-harness evidence remains `diagnostic_only` because it validates the evaluator,
  not a third-party Agent.
- Complete target-integration evidence remains `diagnostic_only` until human review is recorded.
- Only complete, target-integrated, human-reviewed evidence can return `ready`; the command never
  deploys or promotes a target automatically.

Cost and latency remain advisory in this release. They should be reported with experiments but do
not override the trust gates.
