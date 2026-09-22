# GEPA candidate-search runtime

AgentTraceLab 2.0 can run the official `gepa.optimize_anything` 0.1.4 engine against an Agent that
accepts bounded JSON requests and returns a supported trace. GEPA owns candidate selection and
reflection; AgentTraceLab owns candidate binding, execution evidence, deterministic evaluation,
safety gating, fresh-trace enforcement, and held-out promotion gates.

## Install

GEPA is isolated in an optional dependency because trace ingestion and evaluation do not require an
optimizer:

```bash
uv sync --extra dev --extra optimization
```

The extra pins `gepa==0.1.4`. AgentTraceLab supplies an OpenAI-compatible reflection callable, so
the runtime does not require GEPA's optional LiteLLM stack.

## Experiment contract

An `agenttracelab.gepa-experiment.v1` manifest contains:

- a named text `seed_candidate`;
- an objective and bounded background context;
- a relative or absolute Replay manifest path;
- explicit bindings from candidate keys to existing dot-separated request fields;
- metric-call, proposal, and random-seed budgets.

Bindings only overwrite existing object leaves. Missing paths, missing candidate keys, oversized
manifests, and oversized candidate text fail closed. This prevents optimization from silently
changing the request schema.

## Execution loop

For every GEPA metric call AgentTraceLab:

1. fingerprints the candidate without using its text as an identifier;
2. injects bound values into a fresh copy of every Replay request;
3. invokes the configured Agent endpoint with Replay's timeout, size, redirect, and credential
   controls;
4. adapts and deterministically evaluates every returned trace;
5. rejects missing or previously observed trace IDs by forcing the primary score to `0.0`;
6. returns score, diagnostics, failed checks, and four Pareto objectives to GEPA;
7. records the candidate lineage summary and every trace ID in the final report.

The four objectives are deterministic gate, process quality, safety compliance, and fresh-trace
integrity. All are higher-is-better.

## Live command

```bash
export AGENTTRACELAB_GEPA_API_KEY=your-key
uv run agenttracelab run-gepa-experiment path/to/experiment.json \
  --reflection-base-url https://provider.example/v1 \
  --reflection-model reflection-model \
  --output evaluation-results/gepa-experiment.json
```

The API key is read from the named environment variable and is never stored in the report. The
runtime remains CLI/Python-only because exposing arbitrary Replay targets through the public API
would create a server-side request forgery surface.

## Reproducible official-engine smoke

```bash
uv run python examples/run_gepa_runtime_smoke.py \
  --output evaluation-results/gepa-runtime-smoke-v1.json \
  --validation-output evaluation-results/gepa-validation-smoke-v1.json \
  --stability-output evaluation-results/gepa-stability-smoke-v1.json \
  --hard-cases-output evaluation-results/gepa-hard-cases-smoke-v1.json \
  --resolutions-output evaluation-results/gepa-hard-case-resolutions-smoke-v1.json \
  --curated-output evaluation-results/gepa-curated-challenge-set-smoke-v1.json
```

This invokes the installed GEPA engine with a deterministic proposer and mock Agent transport. The
checked-in summary records two candidates, four fresh metric calls, no reused trace IDs, and a score
change from `0.0` to `1.0`. The selected candidate then runs against two separately declared
held-out requests, scores `1.0`, has a `0.0` train-validation gap, and produces two new trace IDs.
It proves wiring and gate behavior only: fixture-backed mock responses do not demonstrate LLM
reflection, deployed-Agent generalization, fine-tuning, or model-quality improvement.

## Held-out validation contract

`agenttracelab.gepa-validation.v1` points to the immutable experiment manifest, its completed search
report, and a separately versioned Replay manifest. Paths are resolved relative to the validation
manifest. The policy supports:

- `min_validation_score`;
- `min_task_count`;
- `max_train_validation_gap`;
- required safety compliance;
- required expected-outcome matching;
- required fresh traces.

The validator binds only the search report's selected candidate. It initializes its seen-trace set
from the training report, runs every held-out task, and rejects trace reuse both within validation
and across training/validation. Example:

```bash
uv run agenttracelab validate-gepa-candidate path/to/validation.json \
  --output evaluation-results/gepa-candidate-validation.json
```

`promote` is returned only when all configured checks pass; all reasons are retained for `hold`.
The command never updates a prompt, model, service, or deployment. See
[`gepa-validation-v1.md`](specs/gepa-validation-v1.md) for the exact schema and boundaries.

## Repeated stability gate

One held-out pass can still be accidental for a stochastic Agent. An
`agenttracelab.gepa-stability.v1` manifest therefore repeats every held-out task, running both the
seed baseline and selected candidate on each trial. Execution order alternates by task/trial. All
runs share a seen-trace set initialized from the GEPA search report.

```bash
uv run agenttracelab run-gepa-stability-gate path/to/stability.json \
  --output evaluation-results/gepa-stability.json
```

The report includes paired mean delta and the lower endpoint of a Wilson score interval for the
candidate's pass rate. The gate also detects candidate safety regressions relative to a safe baseline. A confidence bound is
not a substitute for independent task curation: correlated prompts, shared outages, or fixture-based
responses can violate the Bernoulli-trial assumption. See
[`gepa-stability-v1.md`](specs/gepa-stability-v1.md).

## Hard-case feedback queue

The stability report can be transformed into a bounded review queue:

```bash
uv run agenttracelab mine-gepa-hard-cases \
  evaluation-results/gepa-stability.json \
  --dataset-id office-agent-challenge-set \
  --dataset-version 1.0.0 \
  --output evaluation-results/gepa-hard-cases.json
```

The miner classifies candidate failures, safety regressions, score regressions, and resolved
baseline failures. It groups matching failed-check signatures, prioritizes safety evidence, and
selects across signatures before allowing repeats. The artifact contains trace references rather
than prompts or tool payloads. Human review decides whether each pointer becomes a labeled training
case, a regression test, or is discarded. See
[`hard-case-mining-v1.md`](specs/hard-case-mining-v1.md).

## Adjudication and immutable curation

A reviewer resolves every selected case as `regression_guard`, `training_candidate`, or `reject`.
AgentTraceLab validates that decisions reference the exact queue version, contain no duplicate or
unknown IDs, and satisfy decision/rationale rules:

```bash
uv run agenttracelab curate-gepa-hard-cases \
  evaluation-results/gepa-hard-cases.json \
  path/to/resolutions.json \
  --output evaluation-results/curated-challenge-set-v1.json
```

Training candidates are assigned to train/validation/test by a deterministic hash of `task_id`, so
multiple trials of one task remain in the same split. Confirmed fixes become regression guards. The
CLI refuses to overwrite an existing output. The smoke uses an explicit `generated-synthetic`
review mode; only a real reviewer-created manifest may be described as human annotation. See
[`hard-case-curation-v1.md`](specs/hard-case-curation-v1.md).

## Promotion boundary

A live improvement claim additionally requires a tunable deployed Agent, independently curated
train and held-out tasks, a fixed evaluator version, retained run artifacts, and human review of the
chosen candidate. A synthetic smoke recommendation or training-set improvement is never sufficient
for production deployment.
