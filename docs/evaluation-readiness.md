# Evaluation evidence readiness

AgentTraceLab keeps Agent performance separate from confidence in the evaluation itself. A high
process score, one successful trial, or a complete trace can still be insufficient evidence for a
release claim. The readiness audit applies a versioned policy to the exact benchmark manifest,
report, and retained artifacts before labeling the result suitable for its intended use.

This follows four measurement principles:

1. A trial should verify the outcome, not only the Agent's final statement or preferred path.
2. Stochastic Agents require repeated, fresh trials and conservative confidence bounds.
3. Task design, human review, Judge calibration, environment isolation, and resource constraints
   are part of the measurement instrument.
4. Missing evidence remains `unknown`; it is never silently treated as a pass.
5. A single native pass rate is not sufficient when the retained evidence cannot verify every
   outcome. The report therefore exposes evidence-supported lower and upper bounds.

## Version-bound design evidence

`agenttracelab.deepresearch-benchmark.v1` supports an optional `design_evidence` block:

```json
{
  "design_evidence": {
    "suite_kind": "regression",
    "dataset_review_status": "human-reviewed",
    "task_source": "recorded-production",
    "reference_solution_coverage": 1.0,
    "outcome_grader_coverage": 1.0,
    "transcript_review_count": 20,
    "production_case_count": 12,
    "reference_solution_artifact_path": "reference-solutions.json",
    "reference_solution_artifact_sha256": "...",
    "dataset_review_artifact_path": "dataset-review.json",
    "dataset_review_artifact_sha256": "...",
    "judge_calibration_id": "claim-judge-calibration-2026-09-v1",
    "judge_calibration_artifact_path": "judge-calibration.json",
    "judge_calibration_artifact_sha256": "...",
    "environment_isolation": "container",
    "environment_fingerprint": "sha256:...",
    "resource_profile": "cpu=4,memory=8GiB,concurrency=1,timeout=300s"
  }
}
```

The benchmark manifest can also pin `expected_claim_judge_provider`,
`expected_claim_judge_model`, and `expected_claim_judge_prompt_version`. All three must be declared
together. The runner refuses to start when its configured Judge differs, and strict readiness
policies compare the same identity against both the benchmark report and calibration artifact.

These declarations are protected by the manifest digest recorded in the benchmark report. Strict
policies additionally require reference-solution, dataset-review, and Judge-calibration JSON
artifacts below the manifest directory; the audit rejects missing files, path traversal, unreadable
JSON, and SHA-256 drift. It executes task-acceptance contracts against answerable reference outputs,
requires empty output/evidence for safe-clarification references, verifies that the review approves
the canonical task digest and every task-level field, and requires a passing calibration row for
the exact Judge provider, model, and prompt version. The records still remain attestations:
cryptographic binding proves which exact record was used, not that a reviewer or environment
operator told the truth.

Older manifests remain valid and load with `unspecified` design evidence. A strict policy reports
those fields as `unknown` and fails closed.

## Run the audit

```powershell
agenttracelab audit-deepresearch-evidence `
  evaluation/benchmarks/deepresearch/v1/manifest.json `
  evaluation-results/baseline-live-001/benchmark-report.json `
  --policy evaluation/policies/deepresearch-candidate-readiness.v1.json `
  --output evaluation-results/baseline-live-001/readiness-audit.json `
  --summary evaluation-results/baseline-live-001/readiness-audit.md
```

The command exits with `0` only for `ready`; `hold` and `diagnostic_only` return `19`. Existing
outputs are never overwritten.

## Evidence-supported score bounds

For newly generated reports, AgentTraceLab separates successful trials into:

- **verified passes**: answerable tasks whose claim-semantic, executable acceptance, and
  task-fulfillment gates passed, plus deterministic no-retrieval clarification tasks that satisfied
  their locked outcome checks;
- **unverified potential passes**: tasks that passed process and citation checks but did not receive
  an independent semantic verdict.

The evidence-supported lower bound is `verified_passes / all_planned_trials`. The upper bound is
`(verified_passes + unverified_potential_passes) / all_planned_trials`. Execution failures and
verified failures are included in neither numerator. This is an evidence interval, not a statistical
confidence interval; the separate Wilson lower bound handles sampling uncertainty.

Legacy reports do not fabricate these fields. A policy that requires the lower bound receives an
`unknown` result until the retained trials are re-scored or the benchmark is rerun.

## Executable task-acceptance contracts

Claim verification answers “are the claims supported by their cited evidence?” It does not answer
“did the report cover the requested task?” An answer can be completely factual and still omit most
of the requested work. Answerable benchmark tasks can therefore include a deterministic contract:

```json
{
  "acceptance_contract": {
    "contract_id": "agent-eval-concepts-v1",
    "min_report_chars": 500,
    "min_independent_sources": 2,
    "min_independent_domains": 2,
    "required_concepts": [
      {"criterion_id": "outcome", "aliases": ["outcome", "最终结果"]},
      {"criterion_id": "trajectory", "aliases": ["trajectory", "轨迹"]}
    ],
    "forbidden_phrases": ["所有 Agent 都应采用同一种架构"]
  }
}
```

When source checks are enabled, every retained evidence item must provide a valid HTTP(S) `url`.
`min_independent_sources` counts distinct canonical page URLs, not caller-controlled `source_id`
labels; query strings, fragments, default ports, and a trailing slash cannot turn one page into
several sources. `min_independent_domains` separately requires distinct publication roots. When a
task declares `include_domains`, matching subdomains collapse to the longest configured root, so
`docs.example.com` and `blog.example.com` both count as `example.com`; reviewers therefore control
which domains represent distinct publishers. Without an allowlist, exact hostnames are used. The
runner derives the hostname from the URL and enforces `include_domains` and `exclude_domains` with
exact-or-subdomain matching. Missing or malformed URLs and domain-policy violations fail closed
with separate criterion IDs.

Domain policy is validated before a run starts. Entries must be plain DNS hostnames rather than
URLs, wildcards, ports, or paths; normalized duplicates and exact allow/deny conflicts are rejected.
An independent-domain requirement must declare enough explicit allowed roots, and those roots may
not overlap through a parent/subdomain relationship. This prevents an ambiguous policy such as
`example.com` plus `docs.example.com` from counting one publisher twice.

The runner records the contract ID, canonical SHA-256 digest, per-trial coverage, verdict, and failed
criterion IDs. The semantic Judge also returns a separate task-fulfillment verdict after comparing
the full report with the original prompt and the resolved citation catalog. This closes a concrete
loophole where a response could mention every required keyword and cite evidence for “the evidence
is insufficient” without actually answering the task. A verified quality pass requires claim
support, the executable acceptance contract, and task fulfillment. If a required semantic layer is
missing, a native gate pass remains an unverified potential pass and contributes only to the
evidence-supported upper bound.

Claim verdicts are requested in bounded batches (`--claim-judge-max-claims-per-request`) so smaller
or local OpenAI-compatible Judges do not silently omit claims from long reports. Every batch still
fails closed unless it returns every requested claim exactly once and cites only evidence supplied
for that claim. Batching changes transport size, not the Judge rubric or prompt version.

The readiness audit independently recalculates manifest contract digests and checks every trial's
binding. Alias matching is intentionally only a deterministic completeness floor: it can detect an
omitted required concept, but cannot prove that the concept was explained correctly. Task
fulfillment, claim support, Judge calibration, and human review remain distinct requirements for
release evidence.

## Recommendation semantics

- `ready`: every required check is present and passes the selected policy.
- `hold`: the measurement evidence is structurally adequate, but a performance or quality gate
  fails.
- `diagnostic_only`: the run is partial, artifacts or provenance are missing, the suite is too
  small, or required design/environment evidence is unknown. The result can guide debugging but
  cannot support the requested release claim.

## Machine-verified checks

The audit independently verifies:

- manifest digest and report/manifest metadata consistency;
- complete, non-duplicated task/trial matrix;
- completion and infrastructure-error rates;
- task count, repeated trials, and Wilson lower confidence bound;
- evidence-supported pass-rate bounds that preserve unknown outcomes;
- live/integration/synthetic evidence class;
- pinned target revision, exact Claim Judge identity, and Judge separation;
- semantic grading coverage and fresh trace IDs;
- executable task-acceptance coverage and manifest-to-trial contract digest binding;
- independently recomputed executable outcome-grader coverage, so declared coverage cannot exceed
  the tasks that actually have acceptance contracts or safe-clarification terminal checks;
- safe relative artifact paths and readable retained JSON;
- version-bound suite, dataset review, reference solution, outcome grading, transcript review,
  Judge calibration, isolation, environment fingerprint, and resource declarations.
- content-addressed reference-solution, dataset-review, and Judge-calibration artifacts;
- answerable and safe-clarification outcome classes when the selected policy requires both.

The readiness audit deliberately does not compress these checks into an average score. One missing
required artifact or unknown release-critical field blocks readiness even if every other dimension
passes.

## Boundaries

- A `ready` result is scoped to one policy and intended use. It is not a universal safety claim.
- Reference-solution coverage is recomputed from bound reference outputs that pass executable task
  contracts. This proves contract compatibility, not that the reference answer is factually ideal;
  human review and semantic evidence grading remain necessary.
- The release-candidate directory contains separate reference template and reference draft
  artifacts. The source-linked draft is machine-tested but marked `ai-researched-unreviewed`; it
  must remain unbound until a reviewer verifies every claim and source.
- Offline readiness does not replace production monitoring, user feedback, A/B testing, or periodic
  human transcript review.
- Infrastructure profiles must describe the actual enforced environment. Merely writing a larger
  resource allowance in the manifest does not make trials comparable.
