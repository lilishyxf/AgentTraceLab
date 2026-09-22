# GEPA candidate validation v1

`agenttracelab.gepa-validation.v1` is the path-based contract for replaying a selected GEPA
candidate against held-out tasks. It turns a search result into reviewable promotion evidence; it
does not mutate or deploy the candidate.

## Manifest

```json
{
  "schema_version": "agenttracelab.gepa-validation.v1",
  "validation_id": "office-agent-held-out",
  "version": "1.0.0",
  "experiment_manifest": "experiment.json",
  "experiment_report": "evaluation-results/gepa-experiment.json",
  "validation_replay_manifest": "validation-replay.json",
  "policy": {
    "min_validation_score": 0.8,
    "min_task_count": 10,
    "max_train_validation_gap": 0.15,
    "require_safety_pass": true,
    "require_fresh_traces": true,
    "require_expectations_match": true
  }
}
```

Relative paths resolve from the validation manifest. The experiment report must be complete and its
ID/version must match the experiment manifest. The selected candidate must contain every declared
binding key, and bindings may only replace leaves already present in held-out requests.

## Decision procedure

1. Load the selected candidate from the completed training report.
2. Seed the seen-trace set with all training trace IDs.
3. Bind the candidate to a fresh copy of each held-out Replay request.
4. Execute, import, and deterministically evaluate each returned trace.
5. Force validation score to `0.0` if any trace is missing, duplicated, or reused from training.
6. Compute held-out deterministic, process, safety, and trace-integrity objectives.
7. Return `promote` only if every configured policy check passes; otherwise return `hold` with all
   reasons.

The train-validation gap is `training_best_score - validation_score`. A negative gap is allowed and
means held-out score was higher; only gaps above the configured maximum block promotion.

## Output and boundaries

`agenttracelab.gepa-candidate-validation.v1` records the two aggregate scores, their gap, objective
scores, task count, fresh/reused trace IDs, failed checks, expectation status, recommendation, and
decision reasons.

This contract does not prove the held-out set is representative or free of contamination. Dataset
provenance, semantic independence, production relevance, evaluator stability, and final human
approval remain external responsibilities. `promote` means the declared evidence gate passed; it
does not authorize automatic deployment.
