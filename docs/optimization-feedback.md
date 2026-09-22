# Evaluation-to-optimization bridge

AgentTraceLab converts a stored trace and its latest deterministic evaluation into content-free
feedback, reward, and integration artifacts. It remains the evidence and evaluation layer; an
external optimizer or trainer owns candidate mutation and model updates.

## Failure localization

`localize_failures(trace, evaluation)` converts every failed deterministic check into a finding
with a failure category, remediation hint, and validated span references. The first referenced span
in execution order is reported as the earliest implicated span. That label is intentionally weaker
than “root cause”: deterministic evidence can identify where an invariant first failed without
proving that one span uniquely caused the final outcome.

The localization output does not contain task text, observations, action reasons, validation text,
or tool input/output. Global findings such as secret detection may therefore have no span ID.

## GEPA and DSPy

Generate feedback from the CLI:

```bash
uv run agenttracelab optimization-feedback TRACE_ID \
  --output evaluation-results/optimization-feedback.json
```

The JSON contract exposes a normalized `score` in `[0, 1]` and an actionable `feedback` string. A
DSPy metric can wrap those fields without making DSPy a runtime dependency of AgentTraceLab:

```python
import dspy

from agenttracelab import build_optimization_feedback


def metric_for_stored_trace(trace, evaluation):
    result = build_optimization_feedback(trace, evaluation)
    return dspy.Prediction(score=result.score, feedback=result.feedback)
```

GEPA's `optimize_anything` contract, verified against commit
`15ee314f9c7d34ec153b809d401f42f55c4dcd76`, accepts an evaluator result shaped as
`(score, info)`. AgentTraceLab 1.5 creates that pair as a versioned record:

```bash
uv run agenttracelab gepa-evaluation TRACE_ID \
  --output evaluation-results/gepa-evaluation.json
```

The record maps directly to an evaluator return value:

```python
from agenttracelab import build_gepa_evaluation


def evaluate_executed_candidate(trace, evaluation):
    result = build_gepa_evaluation(trace, evaluation)
    return result.score, result.info.model_dump(mode="json")
```

The candidate runner must execute the Agent, ingest the resulting fresh trace, and retrieve its
matching evaluation before constructing this artifact. Reusing an old trace for a new candidate
invalidates the experiment.

The primary `score` is the safety-gated recommended reward. `info.scores` contains three
higher-is-better objectives for GEPA's Pareto tracking: `deterministic_gate`, `process_quality`, and
`safety_compliance`. Diagnostics contain bounded, content-free failure summaries and remediation;
failed check IDs and the earliest evidence-linked span remain machine readable. Preservation
constraints tell the optimizer not to weaken approval, privacy, fresh-execution, or evaluation
integrity merely to increase a score.

Passing traces return a preservation message. The optimizer should not mutate already-correct
behavior merely to create a different candidate.

## Training reward record

Generate a reward record:

```bash
uv run agenttracelab training-reward TRACE_ID \
  --output evaluation-results/training-reward.json
```

The record keeps three rewards separate:

- `outcome_reward`: binary deterministic gate result;
- `process_reward`: fraction of applicable deterministic checks that passed;
- `recommended_reward`: process reward unless approval-before-commit or secret-redaction failed.

Every applicable check also becomes a component with its failure category and implicated span IDs.
This allows an external adapter to emit a terminal reward or construct denser step-level rewards.
Agent Lightning v1.0 can consume a mapped reward event alongside model-request events captured by
its Gateway; ART or another trainer can consume a dataset-specific transformation.

AgentTraceLab deliberately does not export prompts, responses, or tool content in this record.
Training data that includes content requires a separate explicit export with domain-specific PII,
licensing, retention, and authorization review.

## Agent Lightning v1.0 reward event

Agent Lightning's v1.0 contract, verified against commit
`ff9457587fb6ec900e16e93be9ad2d77409afa08`, uses append-only rollout events. A reward is created by
posting an `EventCreate` body with `event_type: "reward"` to
`/rollouts/{rollout_id}/attempt/{attempt_id}/events`. AgentTraceLab maps its safety-gated
`recommended_reward` to that contract:

```bash
uv run agenttracelab agent-lightning-reward-event TRACE_ID \
  --rollout-id EXISTING_ROLLOUT_ID \
  --attempt-id 0 \
  --output evaluation-results/agent-lightning-reward-event.json
```

The export is endpoint-ready but deliberately inert: it does not create a rollout, send a request,
or start training. The caller must bind the AgentTraceLab trace to an existing Agent Lightning
rollout. Rollout and attempt IDs are restricted to URL-safe identifiers so an evaluation artifact
cannot alter the destination path.

The reward body contains only the normalized value, a content-free pass-count message, the
`agenttracelab` source label, and a machine-readable reason. Prompt, response, task, observation,
tool input/output, and credentials are excluded.

## HTTP API

- `GET /v1/optimization-feedback/{trace_id}`
- `GET /v1/training-rewards/{trace_id}`
- `GET /v1/integrations/agent-lightning/reward-events/{trace_id}?rollout_id=...&attempt_id=0`
- `GET /v1/integrations/gepa/evaluations/{trace_id}`

Both endpoints derive their result from the stored trace and latest stored deterministic evaluation.
An unknown trace or missing evaluation returns `404`.

## Paired reward experiments

WasmHatch batch results embed both artifacts. Paired batch comparison v2 therefore reports process
and recommended reward deltas alongside deterministic check changes, and makes safety-block
transitions explicit. The generated experiment in
`evaluation/experiments/wasmhatch-optimization/v1` is the reproducible smoke test for this path.

## Current boundary

The `gepa-evaluation` adapter remains a read-only derivation and does not run search. AgentTraceLab
1.6 adds a separate, explicit `run-gepa-experiment` path that may invoke GEPA, a reflection model,
and the configured Agent replay endpoint. It does not post to an Agent Lightning Gateway, start an
RL trainer, fine-tune weights, or by itself prove self-improvement. See `docs/gepa-runtime.md` for
its execution, credential, and fresh-trace boundaries.
