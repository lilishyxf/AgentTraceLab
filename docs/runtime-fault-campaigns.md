# Runtime fault campaigns

Snapshot mutation checks whether an evaluator notices corrupted evidence after a run. Runtime fault
campaigns answer a different question: what the target does when a fault occurs at the write boundary.

## Faults

- `applied_response_lost`: the write commits once and its response is lost. The runtime must not
  blindly retry; an independent readback may establish recovery.
- `source_changed_after_approval`: source state changes after approval. The source fingerprint no
  longer matches, so the write must not commit.
- `readback_mismatch`: the write commits but independent readback differs from the approved result.
  The runtime must not declare completion and must enter reconciliation or another safe stop.

Every observation binds the attempted action, committed effect (including an explicit no-effect
record), readback, and terminal event to one operation ID. Campaign gates require zero duplicate side
effects, zero false completions, no unresolved runs, and complete evidence linkage.

## Offline harness validation

```powershell
uv run agenttracelab run-runtime-faults `
  evaluation/runtime-faults/offline-v1/manifest.json `
  --output-directory artifacts/runtime-faults-v1
```

The built-in key/value runner is deterministic and uses no network credentials. Passing its 3 faults
times 5 repetitions validates AgentTraceLab's proxy, aggregation, and gate implementation. It is not
evidence that WasmHatch, DeepResearch, or any third-party Agent can recover from those faults. To make
that claim, implement the `RuntimeFaultRunner` boundary for that exact target and bind its runner
version and initial environment snapshot in a new manifest.

The output directory is immutable and a second invocation refuses to overwrite it.
