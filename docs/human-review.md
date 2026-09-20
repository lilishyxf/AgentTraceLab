# Human review and dataset promotion

AgentTraceLab v1.1 turns evaluator uncertainty into labelled evidence. Each new LLM Judge report is
kept separate from the deterministic result and checked for conditions that need a person. A
selected pair becomes one immutable review item bound to exact evaluation and Judge report IDs.

This follows the online-to-offline evaluation pattern described in
[LangSmith's evaluation types](https://docs.langchain.com/langsmith/evaluation-types): production
signals and human feedback can identify useful cases, but stable offline datasets remain the basis
for reproducible comparison.

## Selection and priority

A review item is created automatically after a Judge report is stored when at least one condition
is true:

| Condition | Stored reason | Priority |
| --- | --- | --- |
| Deterministic gate fails while Judge says pass | `deterministic_judge_conflict` | critical |
| Deterministic gate passes while Judge says fail | `deterministic_judge_conflict` | high |
| Judge returns review | `judge_deferred` | medium |
| Any Judge dimension is 0 or 1 | `low_dimension_score` | medium unless conflict raises it |
| Operator explicitly requests an audit sample | `forced_sample` | medium |

The identity tuple is `(trace_id, deterministic_evaluation_id, judge_report_id)`. Rechecking that
same tuple returns the existing review instead of creating another task. A new evaluator run,
Judge report, model, or prompt version can create a new review because it represents new evidence.

## Review workflow

Judge execution automatically applies selection. Operators can list the pending queue:

```bash
uv run agenttracelab list-reviews --state pending
```

Forced audit samples are explicit and still require an existing deterministic result and Judge
report:

```bash
uv run agenttracelab queue-review TRACE_ID --force
```

A resolution JSON file must name the reviewer, choose `pass`, `review`, or `fail`, score every
dimension from 0 through 4, and may include notes:

```json
{
  "reviewer_id": "reviewer@example",
  "verdict": "fail",
  "dimensions": {
    "trajectory_coherence": 2,
    "tool_choice": 1,
    "result_grounding": 1,
    "safety_awareness": 3,
    "efficiency": 2
  },
  "notes": "The final answer was not grounded in the recorded tool result."
}
```

Resolve exactly once:

```bash
uv run agenttracelab resolve-review REVIEW_ID resolution.json
```

The HTTP API exposes the same local service workflow. A second resolution attempt returns conflict
instead of replacing the original human decision.

## Promote reviewed cases into calibration

Resolved reviews can become a new offline calibration manifest:

```bash
uv run agenttracelab export-review-calibration production-disagreements v1 \
  evaluation/judge-calibration/production-disagreements/v1/manifest.json
```

Each case includes the human label, original recorded Judge rating, provider/model/prompt version,
Judge report ID, deterministic evaluation ID, review reasons, and resolution time. The command
refuses to overwrite an existing path. Correcting or extending labels therefore produces `v2` (or
another new version) rather than silently mutating `v1`.

The default calibration policy requires at least three cases. A one-case export is valid data but
will deliberately fail the calibration gate until more reviewed examples are collected.

## Security and product boundaries

- Review items contain references, scores, reasons, and reviewer decisions; they do not duplicate
  raw prompt or tool content.
- The current endpoints have no authentication, reviewer assignment, or role enforcement. Expose
  them only behind a trusted development/admin boundary.
- `reviewer_id` is provenance supplied by that trusted boundary, not independently verified
  identity.
- One resolution represents one annotator. Multi-annotator agreement and adjudication remain
  roadmap work.
- Automatic selection is rule-based active-data collection, not autonomous model retraining or
  self-improvement.
