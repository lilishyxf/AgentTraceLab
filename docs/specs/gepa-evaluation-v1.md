# GEPA evaluation v1 specification

## Purpose

`agenttracelab.gepa-evaluation.v1` maps one stored trace and its latest deterministic evaluation to
the current GEPA `optimize_anything` evaluator shape without running GEPA or executing a candidate.

The contract is pinned to GEPA commit
`15ee314f9c7d34ec153b809d401f42f55c4dcd76`, where an evaluator returns `(score, info)` and may
provide a higher-is-better `scores` mapping for multi-objective Pareto tracking.

## Primary score

`score` equals AgentTraceLab's `TrainingRewardRecord.recommended_reward`:

- all deterministic gates pass: normalized process reward, normally `1.0`;
- a non-safety check fails: dense normalized process reward;
- approval-before-commit or secret-redaction fails: `0.0`.

This prevents a candidate with a safety-critical regression from receiving a positive primary
optimization signal.

## Actionable side information

`info` contains:

- `diagnostics`: bounded, content-free pass preservation guidance or ordered remediation;
- `scores`: `deterministic_gate`, `process_quality`, and `safety_compliance`, all higher is better;
- `failed_check_ids`: deterministic failed checks in evaluation order;
- `primary_failure_span_id`: earliest evidence-linked span when available;
- `preservation_constraints`: invariants an optimizer must not weaken to improve the score.

Prompt, response, task, observation, tool input/output, file contents, and credentials are excluded.

## Fresh execution invariant

The adapter scores evidence; it does not run a candidate. A valid GEPA evaluator must:

1. apply one candidate to a tunable Agent runner;
2. execute that candidate against the intended example;
3. ingest the resulting new trace and deterministic evaluation;
4. call `build_gepa_evaluation` for that exact trace;
5. return `record.score, record.info.model_dump(mode="json")`.

Reusing a fixed trace across candidates cannot measure candidate quality and invalidates the search.

## Surfaces

- Python: `build_gepa_evaluation(trace, evaluation)`.
- Service: `AgentTraceService.get_gepa_evaluation(trace_id)`.
- CLI: `gepa-evaluation TRACE_ID [--output PATH]`.
- HTTP: `GET /v1/integrations/gepa/evaluations/{trace_id}`.

All surfaces are read-only derivations from the stored trace and latest evaluation.

## Non-goals

This contract does not:

- install or run GEPA;
- define the candidate artifact or mutation boundary;
- call a reflection model;
- execute WasmHatch or another Agent;
- claim an optimization, self-evolution, or model-quality improvement.
