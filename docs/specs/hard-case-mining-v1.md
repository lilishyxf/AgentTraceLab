# Hard-case mining v1

`agenttracelab.hard-case-mining.v1` is a content-free triage artifact derived from a GEPA stability
report. It turns repeated evaluation evidence into a bounded annotation and regression-review queue.
It does not label examples or copy raw Agent inputs and outputs.

## Eligible case kinds

- `safety_regression`: the baseline was safe and the paired candidate was not;
- `candidate_failure`: the candidate failed Replay expectations or deterministic evaluation;
- `score_regression`: the candidate scored below its paired baseline;
- `resolved_baseline_failure`: the baseline failed and the candidate passed, making the original
  failure valuable as a future regression guard.

Classification uses that precedence order. A safety regression therefore cannot be hidden inside a
generic score regression.

## Privacy and provenance

Each selected entry contains a deterministic case ID, source task/trial identity, baseline and
candidate trace IDs, scores, failed check signature, severity, and triage reason. It intentionally
omits request bodies, system prompts, observations, tool arguments/results, and model output. An
authorized reviewer follows the trace references to inspect source evidence.

Case IDs hash the source stability ID/version, task/trial, and trace IDs. Running the miner again on
the same immutable report produces the same IDs.

## Diversity selection

Eligible cases are grouped by `(case_kind, failure_signature)`. Within each group, cases are ordered
by deterministic priority:

1. critical safety evidence;
2. high candidate or score failures;
3. medium resolved baseline failures;
4. regression magnitude and low candidate score;
5. stable case ID tie-break.

Selection round-robins across groups before taking a second case from the same group. `max_cases`
bounds total review cost and `max_per_signature` prevents a frequent duplicate failure from filling
the queue.

## Command

```bash
uv run agenttracelab mine-gepa-hard-cases STABILITY_REPORT \
  --dataset-id challenge-set \
  --dataset-version 1.0.0 \
  --max-cases 50 \
  --max-per-signature 5 \
  --output hard-cases.json
```

The queue is an input to human review. It does not establish root cause, training suitability, or
permission to expose the referenced trace contents. Approved examples should be copied into a new,
immutable dataset version through a separate authorized workflow.
