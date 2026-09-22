# AgentTraceLab

AgentTraceLab is an evidence-led evaluation backend for tool-using AI agents. It imports execution
traces, normalizes them into a framework-neutral model, runs deterministic safety and lifecycle
checks, and compares a candidate run with a baseline before promotion.

![AgentTraceLab paired promotion report](docs/assets/promotion-report.png)

The first real integration is WasmHatch/OfficeAgent. AgentTraceLab consumes the existing
`wasmhatch.run-journal.v1` export, while its OpenTelemetry-compatible receiver accepts OTLP/HTTP
trace exports. The importer accepts the original journal unchanged; the recorded candidate later
added decision linkage and post-commit verification to WasmHatch after the evaluator exposed those
gaps.

The second integration targets
[`SichengLong26/deepresearch_agent_harness`](https://github.com/SichengLong26/deepresearch_agent_harness).
AgentTraceLab can open its SQLite fact store read-only, export one privacy-reduced run snapshot, and
independently check the durable event chain, tool completion, decision-to-tool linkage, completion
contracts, citations, checkpoint hashes, source isolation, and tool-call accounting. See
[`docs/deepresearch-integration.md`](docs/deepresearch-integration.md) for the evidence boundary and
commands.

## Implemented in v2.3

- Adds a fail-closed evaluation-evidence readiness audit so a high Agent score cannot be mistaken
  for a trustworthy release result.
- Verifies the benchmark manifest digest, report/manifest consistency, complete task/trial matrix,
  completion and infrastructure-error rates, conservative pass-rate bound, semantic coverage,
  Judge separation, fresh trace IDs, and retained artifact integrity.
- Binds suite purpose, dataset review status, reference-solution coverage, outcome-state grading,
  transcript review, Judge calibration, environment isolation, environment fingerprint, and
  resource profile declarations to the immutable benchmark manifest.
- Can pin the Claim Judge provider, model, and prompt version; execution and readiness fail closed
  when the configured or calibrated Judge identity differs.
- Returns `ready`, `hold`, or `diagnostic_only`; missing release-critical evidence is `unknown` and
  blocks readiness rather than being silently counted as a pass.
- Reports evidence-supported pass-rate bounds: the lower bound includes only independently
  verified outcomes, while the upper bound also includes unverified potential passes.
- Candidate-release policy can require both answerable and safe-clarification cases, preventing a
  one-sided suite from looking complete.
- Adds versioned task-acceptance contracts for minimum report depth, required concepts, independent
  sources, and forbidden claims, plus a separate task-fulfillment Judge. A report that truthfully
  says “I cannot answer” or merely repeats the requested terms no longer passes as task-complete.
- Validates evidence with canonical HTTP(S) page URLs and enforces per-task allowed/excluded
  domains, so duplicate source labels, tracking-query variants, URL-less evidence, and off-domain
  citations cannot satisfy independent-source requirements.
- Can require independently published domain groups in addition to distinct pages; configured
  domain roots collapse their subdomains, preventing two pages from one allowed publisher from
  masquerading as two independent sources.
- Rejects malformed, duplicate, contradictory, insufficient, or parent/subdomain-overlapping domain
  policies before execution, instead of producing a misleading score from an ambiguous source set.
- Recomputes executable outcome-grader coverage from task definitions instead of trusting the
  manifest's declared percentage.
- Binds reference outputs by hash and runs their task contracts, preventing an untested reference
  coverage percentage from satisfying release readiness.
- Includes an explicitly `unreviewed` five-task, three-trial candidate suite; it must not be called
  a gold or release suite until human review, revision pinning, and cross-provider Judge setup are
  completed.
- Includes a source-linked `ai-researched-unreviewed` reference draft for all five tasks. Its three
  answerable outputs pass the executable contracts and its two clarification references remain
  empty, but the draft is deliberately not bound as human-approved evidence.
- Ships a strict candidate-release policy and machine-readable plus Markdown reports. See
  [`docs/evaluation-readiness.md`](docs/evaluation-readiness.md).

## Implemented in v2.2

- Runs a versioned DeepResearch task manifest against the real target API instead of evaluating
  only prebuilt snapshots.
- Creates a fresh Session for every trial, repeats each task, polls each Run to a terminal state,
  and retains the source snapshot, normalized trace, and deterministic evaluation for every run ID.
- Reports completion rate, empirical pass rate, a 95% Wilson lower bound, Pass@1, Pass^k
  reliability, latency percentiles, tokens, tool calls, tool-failure/recovery rates, claim citation
  gates, semantic-grading coverage, and failure categories.
- Refuses to overwrite an existing benchmark report and records the manifest digest, target
  revision, evidence class, and explicit evidence limitations.
- Keeps live-system, integration, and synthetic evidence distinct. A mocked integration test proves
  orchestration and aggregation; only a configured target with real model/search credentials can
  produce model-quality evidence.
- Splits generated reports into auditable claims, resolves evidence IDs, measures citation coverage,
  and computes an explicitly labelled lexical-support proxy. An optional strict Judge marks each
  claim supported, unsupported, or unverifiable and separately decides whether the complete report
  fulfills the original task. The task Judge receives the resolved citation catalog, so internal
  evidence IDs count as links only when they map to retained source URLs.
- Separates deterministic process pass, claim citation-gate pass, claim semantic support, and task
  fulfillment. Retained snapshots can be rescored with newer evaluator rules without rerunning the
  target Agent or overwriting the original report.
- Scores retained tool trajectories against versioned reference policies: tool selection, bounded
  call count, source mode, argument-record observability, result-to-evidence linkage, duplicate
  calls, and failure recovery. Generated reference policies are explicitly ineligible for promotion
  until humans review them.
- Ships a 20-case, deliberately `unreviewed` annotation queue covering official-source research,
  conflicting or insufficient evidence, recovery, efficiency, clarification, and safety. It is a
  candidate for a future gold set, not a claimed gold set.
- Separates that queue into 16 live-safe cases, three controlled-fault cases, and one manual-only
  security case; a reproducible builder creates the live benchmark and a second command binds its
  retained snapshots back to the reviewed expectations.
- Adds a 10-case mutation suite for citation, source, checkpoint, contract, accounting, tool-result,
  terminal-state, decision-link, secret, and duplicate-call faults. The included deterministic
  fixture detects 10/10 injected faults; this is evaluator-sensitivity evidence, not target recovery.
- Clusters baseline failure signatures into an immutable challenge manifest, re-runs a candidate on
  those tasks, pairs task/trial results, and applies pass-rate, score, latency, safety, trace-freshness,
  evidence-class, and optional claim-Judge gates before returning `promote` or `hold`.
- Never deploys a candidate automatically. Promotion remains a versioned evidence-backed
  recommendation.

Tool-trajectory evaluation and annotation instructions are documented in
[`docs/tool-trajectory-evaluation.md`](docs/tool-trajectory-evaluation.md).

## Implemented in v2.1

- Adds a strict `deepresearch-agent.run-snapshot.v1` adapter for durable Run events, tool calls,
  evidence, completion contracts, and checkpoint-integrity facts.
- Exports directly from a DeepResearch SQLite database in read-only mode and omits raw tool
  arguments/results plus checkpoint state; only hashes, status, identifiers, and bounded metadata
  cross the integration boundary.
- Adds five target-specific deterministic checks: required contracts, citation resolution,
  checkpoint integrity, source-mode isolation, and recorded-versus-reported tool-call consistency.
- Supports DeepResearch snapshots through the Python service, HTTP import endpoint, CLI, replay
  manifests, and offline datasets.
- Records a 100/100 integration fixture generated with the target repository's own persistence
  repositories. This proves the adapter and check wiring, not live search or model quality.

## Implemented in v2.0

- Validates a versioned adjudication set against the mined hard-case queue: source ID/version must
  match, case IDs must be known and unique, and complete review cannot omit a selected case.
- Enforces decision semantics: only confirmed fixes of resolved baseline failures become regression
  guards; training candidates require a confirmed-failure rationale; rejected cases remain auditable.
- Keeps all trials for one `task_id` in the same deterministic train/validation/test split, reducing
  direct task leakage across dataset partitions. Regression guards use a separate regression split.
- Refuses to overwrite an existing curated output, forcing a new immutable dataset version instead
  of silently changing labels.
- The generated-synthetic smoke explicitly records `review_mode=generated-synthetic`, turns three
  selected fixes into three regression guards, and never represents them as human labels or
  materialized training examples.

## Implemented in v1.9

- Mines candidate failures, safety regressions, paired score regressions, and resolved baseline
  failures from repeated stability evidence.
- Converts each case into a deterministic, content-free pointer containing task/trial identity,
  trace IDs, failed check IDs, scores, and triage reasons without copying prompts or tool payloads.
- Groups cases by failure signature, prioritizes critical safety evidence, and round-robins across
  signatures before taking repeats. Per-signature and total budgets prevent one common failure from
  consuming the complete review queue.
- Preserves resolved baseline failures as regression-guard candidates, connecting successful
  optimization back to future evaluation data instead of discarding the original bad cases.
- The generated-synthetic smoke finds 10 repeated resolved baseline cases, deduplicates them to one
  failure signature, and selects three bounded review entries with 100% signature coverage. This is
  a queue-generation demonstration; human labeling remains required.

## Implemented in v1.8

- Repeats every held-out task for the configured number of trials and evaluates the GEPA seed and
  selected candidate as paired samples instead of trusting one stochastic run.
- Alternates baseline/candidate execution order across trials to reduce fixed ordering bias while
  preserving a shared fresh-trace set seeded from training evidence.
- Reports baseline and candidate mean scores, paired mean delta, candidate success rate, a 90/95/99%
  Wilson lower confidence bound, safety regressions, and all trial trace IDs.
- Blocks promotion when the candidate mean, paired delta, confidence lower bound, safety-regression
  budget, or trace-freshness gate fails. CLI failure uses a non-zero exit code for CI.
- Extends the generated-synthetic smoke to 10 paired held-out samples: baseline mean `0.0`, candidate
  mean `1.0`, paired delta `1.0`, 95% success lower bound `0.7225`, zero safety regressions, and 20
  new stability trace IDs. These repeated fixture responses test the algorithm, not production
  independence or model quality.

## Implemented in v1.7

- Replays the selected GEPA candidate on a separate held-out manifest before recommending
  promotion; training score alone is no longer sufficient.
- Seeds the held-out freshness gate with every training trace ID, so a cached or reused training
  trace forces validation score `0.0` and a `hold` recommendation.
- Applies explicit minimum validation score, minimum task count, maximum train-validation gap,
  safety, expected-outcome, and fresh-trace gates.
- Emits a versioned candidate-validation report with training/validation scores, objective scores,
  trace IDs, failed checks, decision reasons, and an honest `promote|hold` recommendation. It never
  deploys a candidate automatically.
- Extends the generated-synthetic official GEPA smoke with two held-out requests: validation score
  `1.0`, train-validation gap `0.0`, two new trace IDs, and no training-trace reuse. This proves the
  gate wiring, not real-world generalization.

## Implemented in v1.6

- Runs the official `gepa.optimize_anything` 0.1.4 search loop as an optional integration rather
  than only exporting evaluator data.
- Binds named text candidates to existing fields in bounded Replay requests; missing paths fail
  closed instead of silently changing the target contract.
- Executes every candidate against the Agent endpoint and requires a new trace ID for every task and
  evaluation. Reused or missing trace evidence forces that candidate's primary score to `0.0`.
- Tracks deterministic gate, process quality, safety compliance, and fresh-trace integrity as
  higher-is-better Pareto objectives, with complete candidate fingerprints and evaluation evidence.
- Includes an OpenAI-compatible reflection client plus an official-engine, deterministic-proposer
  smoke experiment that moves `0.0 → 1.0` across two candidates and four fresh metric calls. The
  smoke remains generated-synthetic integration evidence, not a model-quality claim.

## Implemented in v1.5

- Exports the current GEPA `optimize_anything` evaluator result shape: a safety-gated primary
  `score` plus actionable `info`, pinned to GEPA commit
  `15ee314f9c7d34ec153b809d401f42f55c4dcd76`.
- Supplies higher-is-better objective scores for deterministic gate, process quality, and safety
  compliance so a caller can use GEPA's multi-objective Pareto tracking.
- Includes failed check IDs, the earliest evidence-linked span, bounded diagnostics, and explicit
  preservation constraints that reject candidates which weaken approval or privacy behavior.
- Exposes the derived record through Python, CLI, and a read-only HTTP endpoint. The candidate
  runner must execute every candidate and ingest a fresh trace; the v1.5 derivation itself does not
  run GEPA or mutate an Agent.

## Implemented in v1.4

- Exports an Agent Lightning v1.0 `EventCreate` reward payload from a stored trace and its latest
  deterministic evaluation.
- Uses the safety-gated `recommended_reward`; approval or secret-redaction failures therefore emit
  reward `0.0` even when other process checks pass.
- Requires an explicit Agent Lightning rollout mapping, validates rollout/attempt IDs before
  constructing the endpoint, and never posts to a training server automatically.
- Exposes the integration through Python, CLI, and a read-only HTTP derivation endpoint without
  adding Agent Lightning, VERL, vLLM, CUDA, or GPU runtime dependencies.

## Implemented in v1.3

- Batch evaluation embeds content-free optimization feedback and training reward records for every
  WasmHatch scenario.
- Paired comparison v2 reports process/recommended reward deltas and safety-block transitions, in
  addition to deterministic score and check changes.
- A generated five-scenario WasmHatch optimization experiment covers decision evidence, terminal
  tool results, approval-before-commit, post-commit validation, and metric integrity.
- Separately captured browser runs show the WasmHatch local workflow moving from 70/100 to 100/100
  after three trace-instrumentation gaps were closed, without weakening approval or privacy gates.
- A three-scenario recorded-local campaign then verifies the corrected evidence contract across
  normalization, reconciliation, and imported-CSV workflows; all three score 100/100 with distinct
  exported run IDs. This remains workflow evidence, not a model-quality benchmark.

## Implemented in v1.2

- Evidence-linked failure localization that identifies the earliest implicated trace span without
  claiming causal certainty or copying raw Agent content.
- Framework-neutral `score + feedback` artifacts for GEPA/DSPy reflective optimization.
- Privacy-safe training reward records with separate outcome, process, and safety-gated recommended
  rewards for external Agent Lightning/ART adapters.

- WasmHatch run-journal v1 importer with strict sequence validation.
- OpenInference-style OTLP importer with resource, scope, span, status, and attribute decoding.
- Native `POST /v1/traces` OTLP/HTTP receiver for JSON and Protobuf exports.
- Streaming wire-size enforcement, bounded gzip decompression, and media-type validation.
- Real OpenTelemetry Python SDK HTTP/Protobuf smoke path and end-to-end regression test.
- Separate OpenAI-compatible LLM-as-Judge reports with a versioned five-dimension trajectory rubric.
- Metadata-only default disclosure, explicit content opt-in, privacy fail-closed behavior, strict
  response validation, bounded retries, span-evidence verification, and persisted judge reports.
- Versioned human-labelled Judge calibration datasets with coverage, exact agreement, confusion
  matrices, Cohen's kappa, review rate, dimension MAE, pairwise agreement, and policy gates.
- Automatic human-review selection for deterministic/Judge conflicts, Judge deferrals, and low
  rubric scores, plus explicit audit sampling, immutable resolutions, and calibration export.
- Framework-neutral Trace / Span / DecisionEvidence models inspired by OpenInference terminology.
- Deterministic checks for:
  - contiguous trace order;
  - terminal disposition;
  - tool-result recording;
  - approval before committed effects;
  - validation after committed effects;
  - structured observation → action → validation → disposition evidence;
  - decision-to-tool linkage;
  - credential-shaped data leakage;
  - event budgets.
- Baseline/candidate regression comparison.
- FastAPI ingestion, retrieval, evaluation, and comparison endpoints.
- SQLite development storage and PostgreSQL Docker profile.
- CLI import, comparison, and versioned offline dataset commands.
- A versioned smoke dataset containing passing and intentional bad-case expectations.
- Bounded HTTP replay runner for WasmHatch, DeepResearch, normalized, and OpenInference responses.
- Failure taxonomy for transport, trace contract, lifecycle, tool, authorization, validation,
  decision-evidence, privacy, and budget failures.
- WasmHatch export-batch evaluation with explicit synthetic/recorded/live evidence labels.
- Event-derived metric reconciliation and versioned promotion-policy gates.
- Scenario-paired baseline/candidate comparison with coverage-loss and check-regression detection.
- Standalone HTML experiment reports and JUnit XML regression artifacts for CI.

This release deliberately does **not** claim an OTLP exporter, OTLP/gRPC receiver, distributed
replay workers, authenticated multi-reviewer adjudication, or a production dashboard. Replay is
currently a bounded synchronous CLI runner; the OTLP/HTTP receiver stores and evaluates trace data.

WasmHatch itself is browser-only and deliberately exports journals through an explicit user
download. AgentTraceLab respects that boundary: batch evaluation consumes those exported files and
does not add a hidden upload or claim unattended WasmHatch replay.

## Architecture

```text
WasmHatch run-journal.v1 ──► WasmHatch adapter ──┐
                                                  ├──► normalized Trace/Span model
OTLP/HTTP + legacy OTLP JSON ──► OTLP adapter ────┘              │
                                                                 ├──► deterministic evaluator
versioned dataset manifest ──► offline runner ────────────────────┤
HTTP replay manifest ──► bounded target request ──► trace adapter ┤
                                                                 ├──► SQL trace/evaluation store
                                                                 ├──► deterministic + LLM Judge disagreement
                                                                 │           │
                                                                 │           └──► human review ──► new dataset version
                                                                 └──► baseline/candidate comparator
```

## Local development

```bash
uv sync --extra dev
uv run pytest
uv run uvicorn agenttracelab.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

Install the optional GEPA runtime only when running candidate search:

```bash
uv sync --extra dev --extra optimization
```

Send standards-shaped OTLP/HTTP JSON traces to the native receiver:

```bash
curl -X POST http://127.0.0.1:8000/v1/traces \
  -H 'Content-Type: application/json' \
  --data-binary @tests/fixtures/openinference_otlp.json
```

Protobuf exporters use the same path with `Content-Type: application/x-protobuf`. See
[docs/otlp-http.md](docs/otlp-http.md) for compression, limits, responses, and current boundaries.

With the API running, send a real batched Protobuf export from the OpenTelemetry Python SDK:

```bash
uv run python examples/send_otel_trace.py
```

The command prints the generated trace ID. Retrieve it through `GET /v1/traces/{trace_id}` or inspect
its deterministic result through `GET /v1/evaluations/{trace_id}/latest`.

Import the included synthetic WasmHatch trace:

```bash
uv run agenttracelab import-wasmhatch tests/fixtures/wasmhatch_complete.json
```

The command exits with code `0` only when all required deterministic checks pass. A failed
evaluation exits with code `2`, making it usable as a CI regression gate.

Run the included repeatable smoke dataset:

```bash
uv run agenttracelab run-dataset evaluation/datasets/smoke/v1/manifest.json
```

The dataset runner distinguishes “the evaluator passed this trace” from “the observed result
matched the fixture's expectation.” This means intentional bad cases can pass the regression suite
without being misreported as successful Agent runs.

Audit whether a DeepResearch result contains enough evidence for a candidate-release claim:

```bash
uv run agenttracelab audit-deepresearch-evidence \
  evaluation/benchmarks/deepresearch/v1/manifest.json \
  evaluation-results/baseline-live-001/benchmark-report.json \
  --policy evaluation/policies/deepresearch-candidate-readiness.v1.json \
  --output evaluation-results/baseline-live-001/readiness-audit.json \
  --summary evaluation-results/baseline-live-001/readiness-audit.md
```

The audit exits successfully only for `ready`. See
[`docs/evaluation-readiness.md`](docs/evaluation-readiness.md) for the evidence model and limits.

Run the HTTP replay contract against an Agent endpoint:

```bash
uv run agenttracelab run-replay evaluation/replays/example/v1/manifest.json \
  --output evaluation-results/office-agent.json
```

The checked-in endpoint is an example, not a running dependency. See
[docs/replay-contract.md](docs/replay-contract.md) before wiring a real target.

Evaluate an authorized set of exported WasmHatch journals and apply a release policy:

```bash
uv run agenttracelab evaluate-wasmhatch-batch path/to/manifest.json \
  --policy evaluation/policies/recorded-release.v1.json \
  --output evaluation-results/batch.json \
  --summary evaluation-results/summary.md
```

See [docs/wasmhatch-batch.md](docs/wasmhatch-batch.md) for the manifest and evidence-mode rules.

The checked-in synthetic smoke gate can be exercised without a running Agent:

```bash
uv run agenttracelab evaluate-wasmhatch-batch \
  evaluation/batches/smoke/v1/manifest.json \
  --policy evaluation/policies/synthetic-smoke.v1.json
```

Compare two batches without allowing aggregate scores to hide a scenario regression:

```bash
uv run agenttracelab compare-wasmhatch-batches \
  path/to/baseline/manifest.json path/to/candidate/manifest.json \
  --output evaluation-results/comparison.json \
  --summary evaluation-results/comparison.md \
  --html evaluation-results/comparison.html \
  --junit evaluation-results/comparison.junit.xml
```

See [docs/batch-comparison.md](docs/batch-comparison.md) for the paired-denominator rules and
[docs/comparison-artifacts.md](docs/comparison-artifacts.md) for the generated artifacts.

Reproduce the report shown above from the checked-in synthetic baseline and candidate:

```bash
uv run agenttracelab compare-wasmhatch-batches \
  evaluation/datasets/smoke/v1/baseline-batch.json \
  evaluation/datasets/smoke/v1/candidate-batch.json \
  --output evaluation-results/promotion-demo.json \
  --summary evaluation-results/promotion-demo.md \
  --html evaluation-results/promotion-demo.html \
  --junit evaluation-results/promotion-demo.junit.xml
```

The image is demonstration evidence from synthetic fixtures, not a production benchmark.

Run an optional qualitative judge against a stored trace without changing its deterministic gate:

```bash
uv run agenttracelab judge-trace TRACE_ID \
  --base-url https://api.deepseek.com \
  --model deepseek-flash \
  --output evaluation-results/judge.json
```

The API key is read from `AGENTTRACELAB_JUDGE_API_KEY`. Content is omitted by default. See
[docs/llm-judge.md](docs/llm-judge.md) before enabling `--include-content`.

Evaluate Judge/Prompt configurations against human labels without making network calls:

```bash
uv run agenttracelab analyze-judge-calibration \
  evaluation/judge-calibration/smoke/v1/manifest.json \
  --output evaluation-results/judge-calibration.json \
  --summary evaluation-results/judge-calibration.md
```

The smoke dataset deliberately fails one weak judge to prove the gate catches missing coverage and
poor agreement. See [docs/judge-calibration.md](docs/judge-calibration.md).

Inspect, resolve, and promote automatically selected disagreements into a new labelled dataset:

```bash
uv run agenttracelab list-reviews --state pending
uv run agenttracelab resolve-review REVIEW_ID path/to/resolution.json
uv run agenttracelab export-review-calibration production-disagreements v1 \
  evaluation/judge-calibration/production-disagreements/v1/manifest.json
```

The exporter refuses to overwrite an existing file, so corrected labels require a new dataset
version. See [docs/human-review.md](docs/human-review.md) for selection, priority, payload, and
security boundaries.

Build actionable feedback for a reflective optimizer from a stored trace:

```bash
uv run agenttracelab optimization-feedback TRACE_ID \
  --output evaluation-results/optimization-feedback.json
```

Build an evidence-linked, content-free reward record for an external trainer:

```bash
uv run agenttracelab training-reward TRACE_ID \
  --output evaluation-results/training-reward.json
```

Map that reward to the current Agent Lightning v1.0 reward-event contract:

```bash
uv run agenttracelab agent-lightning-reward-event TRACE_ID \
  --rollout-id EXISTING_AGENT_LIGHTNING_ROLLOUT_ID \
  --attempt-id 0 \
  --output evaluation-results/agent-lightning-reward-event.json
```

The export includes the exact Agent Lightning event endpoint and request body, but does not send the
request. The rollout must already exist in the target Agent Lightning Gateway.

Build the current GEPA `optimize_anything` `(score, info)` evaluator data for a stored trace:

```bash
uv run agenttracelab gepa-evaluation TRACE_ID \
  --output evaluation-results/gepa-evaluation.json
```

The primary score uses the safety-gated recommended reward. `info.scores` keeps deterministic gate,
process quality, and safety compliance separate, while `diagnostics` and failed check IDs give the
reflection model actionable evidence. Every candidate still needs a fresh execution and trace.
The checked-in content-free sample under `evaluation/integrations/gepa/v1` was derived from the
recorded-local WasmHatch normalization run and is labelled as adapter evidence, not a search result.

These commands do not run GEPA, mutate an Agent, or update model weights. See
[docs/optimization-feedback.md](docs/optimization-feedback.md) for the DSPy/GEPA wrapper, reward
policy, and training-system boundary.

Run a real GEPA search loop against a configured trace-returning Agent endpoint:

```bash
export AGENTTRACELAB_GEPA_API_KEY=your-key
uv run agenttracelab run-gepa-experiment path/to/experiment.json \
  --reflection-base-url https://provider.example/v1 \
  --reflection-model reflection-model \
  --output evaluation-results/gepa-experiment.json
```

The experiment manifest controls candidate-to-request bindings and budgets; the referenced Replay
manifest controls the Agent endpoint, tasks, response adapter, timeout, and response-size limit. See
[docs/gepa-runtime.md](docs/gepa-runtime.md). Reproduce the no-key official-engine smoke with:

```bash
uv run python examples/run_gepa_runtime_smoke.py \
  --output evaluation-results/gepa-runtime-smoke-v1.json \
  --validation-output evaluation-results/gepa-validation-smoke-v1.json \
  --stability-output evaluation-results/gepa-stability-smoke-v1.json \
  --hard-cases-output evaluation-results/gepa-hard-cases-smoke-v1.json \
  --resolutions-output evaluation-results/gepa-hard-case-resolutions-smoke-v1.json \
  --curated-output evaluation-results/gepa-curated-challenge-set-smoke-v1.json
```

After a live search report exists, run the selected candidate against a separately versioned
held-out Replay manifest:

```bash
uv run agenttracelab validate-gepa-candidate path/to/validation.json \
  --output evaluation-results/gepa-candidate-validation.json
```

The command exits successfully only for `promote`. A low score, excessive generalization gap,
failed safety check, expectation mismatch, missing trace, or training-trace reuse returns `hold`
and a non-zero exit code. See [docs/gepa-runtime.md](docs/gepa-runtime.md) and
[docs/specs/gepa-validation-v1.md](docs/specs/gepa-validation-v1.md).

For a stochastic Agent, require repeated paired evidence rather than relying on one held-out run:

```bash
uv run agenttracelab run-gepa-stability-gate path/to/stability.json \
  --output evaluation-results/gepa-stability.json
```

See [docs/specs/gepa-stability-v1.md](docs/specs/gepa-stability-v1.md) for the confidence-bound
calculation, policy fields, and limitations.

Mine a bounded, diverse annotation/regression queue from the stability report:

```bash
uv run agenttracelab mine-gepa-hard-cases \
  evaluation-results/gepa-stability.json \
  --dataset-id office-agent-challenge-set \
  --dataset-version 1.0.0 \
  --max-cases 50 \
  --max-per-signature 5 \
  --output evaluation-results/gepa-hard-cases.json
```

The output contains evidence pointers and check IDs, not raw task text. See
[docs/specs/hard-case-mining-v1.md](docs/specs/hard-case-mining-v1.md).

After a human reviewer produces a resolution set, validate it and write a new immutable challenge
set version:

```bash
uv run agenttracelab curate-gepa-hard-cases \
  evaluation-results/gepa-hard-cases.json \
  path/to/resolutions.json \
  --output evaluation-results/curated-challenge-set-v1.json
```

Existing outputs are never overwritten. See
[docs/specs/hard-case-curation-v1.md](docs/specs/hard-case-curation-v1.md).

Run the checked-in paired optimization experiment:

```bash
uv run agenttracelab compare-wasmhatch-batches \
  evaluation/experiments/wasmhatch-optimization/v1/baseline.json \
  evaluation/experiments/wasmhatch-optimization/v1/candidate.json \
  --output evaluation-results/wasmhatch-optimization-v1.json \
  --summary evaluation-results/wasmhatch-optimization-v1.md \
  --html evaluation-results/wasmhatch-optimization-v1.html \
  --junit evaluation-results/wasmhatch-optimization-v1.junit.xml
```

The experiment uses generated synthetic failure injections. Its deltas validate the evaluation and
reward pipeline; they are not live-model quality claims.

## HTTP API

- `POST /v1/traces` (native OTLP/HTTP JSON or Protobuf receiver)
- `POST /v1/traces/import/wasmhatch`
- `POST /v1/traces/import/otlp` (legacy JSON convenience endpoint)
- `POST /v1/traces/import/normalized`
- `GET /v1/traces/{trace_id}`
- `GET /v1/evaluations/{trace_id}/latest`
- `GET /v1/optimization-feedback/{trace_id}`
- `GET /v1/training-rewards/{trace_id}`
- `GET /v1/integrations/agent-lightning/reward-events/{trace_id}?rollout_id=...&attempt_id=0`
- `GET /v1/integrations/gepa/evaluations/{trace_id}`
- `GET /v1/judgments/{trace_id}/latest`
- `POST /v1/reviews/queue/{trace_id}` (forced audit sampling or idempotent re-check)
- `GET /v1/reviews?state=pending|resolved`
- `GET /v1/reviews/{review_id}`
- `POST /v1/reviews/{review_id}/resolve`
- `POST /v1/comparisons`
- `GET /health`

## Evidence policy

AgentTraceLab distinguishes trace presence from Agent correctness. A model/tool log alone does not
prove an Agent loop. Passing `agent.decision_evidence` requires structured evidence that connects an
observation to a selected action, optional tool call, result validation, and a continue/replan/stop
disposition.

Deterministic checks and LLM-judge checks remain separate in reports. Scores from different modes
must never be merged into one unexplained percentage.

## Roadmap

1. OTLP exporter, OTLP/gRPC receiver, and broader semantic-convention coverage.
2. Async/distributed replay workers with explicit provider/model configuration.
3. Authenticated reviewer assignment, multi-annotator adjudication, and uncertainty-based sampling.
4. Multi-experiment dashboard built on the current standalone comparison report.
5. Live held-out GEPA suites against a deployed tunable Agent, with independent task curation and
   human promotion approval; authenticated Agent Lightning delivery after a real training Gateway
   is available.
6. GitHub check annotations and policy support beyond WasmHatch batch reports.

See [docs/references.md](docs/references.md) for the open-source projects studied and the reuse
boundary.
