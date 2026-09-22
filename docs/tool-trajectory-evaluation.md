# Tool-trajectory evaluation

AgentTraceLab evaluates both the final research artifact and the path used to produce it. The
trajectory scorer is deterministic: it reads retained, privacy-reduced snapshots and never calls or
edits the target Agent.

## What is scored

- **Tool selection:** required tools are present and unexpected tools are absent when the policy
  forbids them.
- **Argument observability:** every call contains a stable argument hash. This proves recording, not
  that the hidden argument value was semantically correct.
- **Call budget:** total calls stay inside a declared range.
- **Source policy:** calls use the required source mode.
- **Evidence-domain policy:** enough retrieved evidence comes from one of the accepted official
  domains. Subdomains match their declared parent domain.
- **Result grounding:** completed calls have a recorded result and linked evidence.
- **Recovery safety:** observed failures must meet the declared minimum and maximum; recovered and
  unrecovered counts have separate gates; recovery requires either a later successful call in the
  same task or a verified terminal task outcome backed by evidence from a successful sibling call;
  an explicit `tool.recovery_confirmed` edge is accepted only when those relationships resolve in
  the retained snapshot; and exact duplicate calls remain within policy. Legacy snapshots can still
  use the deterministic outcome fallback. The raw failure rate remains visible even when a task-level
  fallback succeeds. A recovery case with no observed fault fails.

The scorer does **not** establish final-answer factual correctness. That remains a separate claim and
evidence evaluation, ideally with a genuinely independent Judge plus human calibration.

## Run the retained live-snapshot example

```powershell
agenttracelab evaluate-tool-trajectory `
  evaluation-results\deepresearch-zero-key-stability-20260921-v1\tool-trajectory-manifest.json `
  --output evaluation-results\deepresearch-zero-key-stability-20260921-v1\tool-trajectory-report.json `
  --summary evaluation-results\deepresearch-zero-key-stability-20260921-v1\tool-trajectory-report.md
```

Its reference policies are `generated-synthetic`, so `promotion_eligible` is always false even when
every case passes. This prevents evaluator-authored expectations from being presented as human gold
labels.

## Turn the 20-case queue into a gold set

The candidate queue is
`evaluation/benchmarks/deepresearch/v1/tool-trajectory-annotation-queue.json`. It starts with
`status=unreviewed`, `annotator_count=0`, and no reviewed cases.

Cases also declare an execution lane. The current queue contains 16 `live-safe` cases, three
`controlled-fault` recovery cases, and one `manual-only` secret-exfiltration case. Only the
`live-safe` lane is converted into an ordinary benchmark. This prevents prompts that merely mention
a timeout from being presented as evidence that a timeout was actually injected.

Build the executable lane without contacting the target:

```powershell
agenttracelab build-tool-trajectory-benchmark `
  evaluation\benchmarks\deepresearch\v1\tool-trajectory-annotation-queue.json `
  --output evaluation\benchmarks\deepresearch\v1\tool-trajectory-live-safe.json `
  --benchmark-id deepresearch-tool-trajectory-live-safe `
  --target-name deepresearch-agent-harness `
  --target-provider deepseek `
  --target-model deepseek-flash
```

After running that benchmark against a configured target, bind each retained snapshot back to its
draft or human-reviewed expectation:

```powershell
agenttracelab materialize-tool-trajectory-manifest `
  evaluation\benchmarks\deepresearch\v1\tool-trajectory-annotation-queue.json `
  evaluation-results\LIVE_RUN\benchmark-report.json `
  --output evaluation-results\LIVE_RUN\tool-trajectory-manifest.json
```

The materialized manifest inherits `human-approved` only when the complete source queue has that
status. Otherwise it remains `generated-synthetic` and cannot become promotion evidence.

For every case, a reviewer should:

1. Execute the prompt against a pinned target revision and retain the full privacy-reduced trace.
2. Confirm or edit the expected answer behavior, allowed tools, call range, source mode, domain
   expectations, failure budget, and duplicate budget.
3. Inspect whether the call arguments match the intended task. A hash alone is insufficient.
4. Set the case to `human-reviewed` and record a short rationale in `review_notes`.
5. Resolve disagreements before changing the queue to `human-approved`; set `annotator_count` to the
   actual number of reviewers.

The schema refuses a `human-approved` queue unless at least one annotator is declared and every case
is marked `human-reviewed`. For meaningful calibration, use two reviewers on a subset and report
agreement instead of silently merging disagreements.

Validate the queue after each annotation pass:

```powershell
agenttracelab validate-tool-trajectory-annotations `
  evaluation\benchmarks\deepresearch\v1\tool-trajectory-annotation-queue.json
```

## Independent Judge identity

Benchmark manifests may declare `target_provider`, `target_model`, and
`minimum_judge_separation`. A run configured with `different_model` or `different_provider` fails
before execution when the Judge does not satisfy that separation. A same-model call can still be
useful as an auxiliary score, but it must not be described as independent verification.
`--claim-judge-provider` must name the actual inference provider (for example, `deepseek`), not the
wire protocol (`openai-compatible`).
