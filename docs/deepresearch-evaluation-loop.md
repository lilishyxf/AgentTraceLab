# DeepResearch evaluation and regression loop

This workflow turns one-off trajectory inspection into four separately evidenced stages. It never
deploys or edits the target Agent.

## 1. Repeated live benchmark

Run the versioned suite after the target API and its SQLite database are available:

```powershell
agenttracelab run-deepresearch-benchmark `
  evaluation/benchmarks/deepresearch/v1/manifest.json `
  --base-url http://127.0.0.1:8000 `
  --source-database D:\path\to\deepresearch_agent_harness\data\app.db `
  --output-directory evaluation-results\baseline-live-001
```

Every task/trial gets a new target Session and retains its source snapshot, normalized trace,
deterministic evaluation, and claim/evidence report. The aggregate includes Pass@1, Pass^k,
Wilson lower bound, latency percentiles, token/tool usage, tool-failure rate and recovery counts,
claim citation-gate rate, semantic-grading coverage, citation validity, and trace IDs. A tool
failure lowers the deterministic score; an unrecovered tool failure fails the required gate even
when the run still produces a report. The benchmark reports process success, citation coverage,
and independent semantic verification separately, so a 100% process score is not presented as
verified answer quality.

## 2. Claim/evidence verification

Local citation and lexical-proxy audit:

```powershell
agenttracelab verify-deepresearch-claims `
  evaluation-results\baseline-live-001\trials\TASK\trial-01\run-snapshot.json `
  --output evaluation-results\claim-report.json
```

This result remains `human_review`. To enable semantic support grading, configure a separate
OpenAI-compatible Judge:

```powershell
$env:AGENTTRACELAB_CLAIM_JUDGE_API_KEY = "..."
agenttracelab verify-deepresearch-claims SNAPSHOT.json `
  --judge-base-url https://provider.example/v1 `
  --judge-model independent-judge-model `
  --output claim-report-with-judge.json
```

The Judge sees only the claim and its cited, bounded evidence summaries. Its response must cover
every claim exactly once and cannot cite an evidence ID outside the supplied packet.

## 3. Evaluator fault sensitivity

```powershell
agenttracelab inject-deepresearch-faults `
  SNAPSHOT.json `
  evaluation/benchmarks/deepresearch/v1/faults.json `
  --output-directory evaluation-results\faults-001
```

The suite mutates a copy; it never changes the target database. The current fixture detects 10/10
mutations. This mutation score establishes whether AgentTraceLab notices known trace defects. It
does not establish whether DeepResearch recovers from real provider, network, or tool outages;
those require target-runtime injection hooks.

## 4. Failure-derived candidate regression

Create a bounded challenge set from baseline failures:

```powershell
agenttracelab build-deepresearch-challenge `
  evaluation/benchmarks/deepresearch/v1/manifest.json `
  evaluation-results\baseline-live-001\benchmark-report.json `
  --candidate-target-name candidate-build `
  --candidate-target-revision GIT_SHA `
  --output-directory evaluation-results\challenge-001
```

The one-command loop then runs that challenge against a candidate and applies the policy:

```powershell
agenttracelab run-deepresearch-regression-loop `
  evaluation/benchmarks/deepresearch/v1/manifest.json `
  evaluation-results\baseline-live-001\benchmark-report.json `
  --candidate-base-url http://127.0.0.1:8001 `
  --candidate-source-database D:\candidate\data\app.db `
  --candidate-target-name candidate-build `
  --candidate-target-revision GIT_SHA `
  --policy evaluation/benchmarks/deepresearch/v1/regression-policy.json `
  --output-directory evaluation-results\regression-001
```

The final decision is `promote` only when paired pass rate and score do not regress, candidate pass
rate meets policy, latency stays within budget, there are no disallowed safety regressions, trace
IDs are fresh, and evidence-class requirements pass. The optional semantic claim-support gate can
also be made mandatory in the policy. `promote` is only a recommendation; no automatic deployment
occurs.

## Evidence boundary

- `live-system`: the runner contacted the target, but the label is user-supplied and must match the
  actual target configuration.
- `integration`: orchestration and aggregation were exercised with controlled dependencies.
- `synthetic`: generated fixtures only.
- 100% deterministic checks do not mean 100% factual correctness.
- A 10/10 mutation score does not mean the target Agent survives those faults.
- Candidate comparison is valid only with matching prompts, model settings, budgets, and runtime
  environment.
