# Optimization Feedback v1 Specification

## Purpose

Turn one stored Agent trace and its deterministic evaluation into two framework-neutral artifacts:

1. actionable optimization feedback for reflective optimizers such as GEPA/DSPy;
2. an evidence-linked reward record for external training systems such as Agent Lightning or ART.

AgentTraceLab remains the evaluator and evidence authority. It does not train a model or mutate an
Agent configuration in this release.

## Failure localization

Every failed deterministic check becomes one localization finding containing its failure category,
summary, remediation hint, and zero or more implicated span IDs. Span IDs must be validated against
the source trace. Findings with span evidence are ordered by the earliest implicated span; global
findings remain after localized findings. The report names the earliest implicated span as the
primary failure point without claiming causal certainty.

Localization must not copy task, tool input/output, observation, reason, validation, or other raw
content into the feedback artifact.

## GEPA-compatible feedback

The optimizer artifact exposes:

- `score`: deterministic score normalized to `[0, 1]`;
- `feedback`: concise natural-language actionable side information;
- the deterministic gate result and evaluation identifiers;
- structured localization findings.

The contract deliberately avoids a hard dependency on GEPA or DSPy. A caller can convert the two
fields to `dspy.Prediction(score=..., feedback=...)` or log the feedback through GEPA's
`optimize_anything` evaluator.

A passing evaluation returns a preservation message instead of inventing improvements.

## Training reward record

The training artifact exposes separate signals rather than hiding policy in one score:

- `outcome_reward`: `1.0` when every required gate passes, otherwise `0.0`;
- `process_reward`: deterministic score normalized to `[0, 1]`;
- `recommended_reward`: process reward unless a safety-critical check fails, in which case `0.0`;
- one component per applicable deterministic check, linked to implicated spans when available.

Safety-critical checks are `safety.approval_before_commit` and `privacy.no_raw_secret`. Training
consumers may choose a different documented reward policy, but AgentTraceLab's recommended reward
must never reward a trace that violates either check.

The reward record contains no prompt or response content. Training content export remains an
explicit, separate, privacy-reviewed operation.

## Interfaces

- Python: `localize_failures(trace, evaluation)`,
  `build_optimization_feedback(trace, evaluation)`, and
  `build_training_reward(trace, evaluation)`.
- Service: latest optimization feedback and training reward by trace ID.
- HTTP: `GET /v1/optimization-feedback/{trace_id}` and
  `GET /v1/training-rewards/{trace_id}`.
- CLI: `optimization-feedback TRACE_ID` and `training-reward TRACE_ID`, each with optional
  `--output`.

## Non-goals

- running GEPA search;
- invoking a trainer or updating model weights;
- generating dynamic invariants with an LLM;
- claiming that an implicated span is the unique root cause;
- exporting raw conversation or tool content.
