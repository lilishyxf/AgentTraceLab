# GEPA experiment v1 specification

## Manifest

`agenttracelab.gepa-experiment.v1` defines one bounded candidate-search run. Required fields are
`experiment_id`, `version`, `seed_candidate`, `objective`, `replay_manifest`, and `bindings`.

Every binding maps one candidate key to an existing dot-separated field in each Replay task request.
Bindings cannot create new request fields. Candidate keys are unique; all bound keys must exist in
the seed candidate.

Budgets are bounded to 2-1000 metric calls and 1-100 candidate proposals. The default random seed is
zero.

## Evaluation

One GEPA evaluator invocation runs the complete Replay task set. Missing tasks contribute zero via
the fixed denominator. Safety-critical failures use `recommended_reward` and therefore contribute
zero to the primary score. Any trace ID reused within or across evaluator calls forces the whole
candidate score to zero.

Side information contains bounded diagnostics, failed check IDs, candidate fingerprint, trace IDs,
Replay expectation status, preservation constraints, and these higher-is-better objectives:

- `deterministic_gate`;
- `process_quality`;
- `safety_compliance`;
- `fresh_trace_integrity`.

## Report

`agenttracelab.gepa-experiment-report.v1` records the exact GEPA version, seed and best scores,
selected candidate, candidate fingerprints, metric-call count, all observed/reused trace IDs, and
every candidate evaluation. It never contains endpoint bearer tokens or reflection API keys.

## Non-goals

The contract does not assert that a search generalizes, that an LLM proposer is better than a
deterministic proposer, that weights were fine-tuned, or that a candidate is safe for automatic
production promotion.
