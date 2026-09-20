# Judge calibration datasets

AgentTraceLab v1.0 evaluates the evaluator. A versioned calibration manifest combines human labels
with recorded ratings from one or more Judge/Prompt configurations. The analyzer is fully offline:
it makes no model calls and produces the same metrics from the same manifest.

## Why this exists

An LLM Judge returning valid JSON does not establish that its labels are useful. Before a judge is
used as a release or monitoring signal, it should be compared with a representative human-labelled
set. Missing model ratings must remain visible; calculating agreement only on the easiest returned
samples creates survivorship bias.

The calibration report keeps these questions separate:

- Did the judge cover every labelled case?
- How often did its pass/review/fail verdict exactly match the human label?
- How much agreement remains after accounting for chance (Cohen's kappa)?
- How often did it defer to review?
- How far were its five dimension scores from the human scores (mean absolute error)?
- Do two Judge/Prompt configurations agree with each other on their overlapping cases?

## Manifest contract

`agenttracelab.judge-calibration-manifest.v1` contains:

- a dataset ID and immutable version;
- explicit policy thresholds;
- unique case and trace IDs;
- a human consensus label with annotator count and all five dimension scores;
- zero or one recorded rating per `judge_key` in each case;
- provider, model, prompt version, optional source report ID, verdict, and five dimension scores for
  each recorded judge rating.

A `judge_key` must resolve to one consistent provider/model/prompt tuple throughout the manifest.
Changing a model snapshot or rubric version requires a new key. Missing ratings are allowed in the
data but reduce coverage and can fail the gate.

The included synthetic smoke dataset intentionally contains one well-calibrated judge and one
under-covered, over-deferring judge:

```text
evaluation/judge-calibration/smoke/v1/manifest.json
```

It validates metric and gate behavior; it is not evidence that any commercial model is calibrated.

## Run the analyzer

```bash
uv run agenttracelab analyze-judge-calibration \
  evaluation/judge-calibration/smoke/v1/manifest.json \
  --output evaluation-results/judge-calibration.json \
  --summary evaluation-results/judge-calibration.md
```

Exit code `0` means every discovered judge passed every policy threshold. Exit code `9` means at
least one judge failed. JSON contains confusion matrices and exact values; Markdown provides a
reviewable summary and each violation.

## Interpretation boundaries

- Cohen's kappa is an agreement statistic, not proof of factual correctness. Small or imbalanced
  datasets can make it unstable.
- Human labels require a documented rubric and adjudication process; annotator count alone does not
  prove label quality.
- Pairwise Judge agreement does not replace comparison with human labels. Two judges can agree and
  both be wrong.
- Thresholds should be chosen before inspecting candidate results and should be versioned with the
  dataset.
- Production disagreements and user feedback should be reviewed and promoted into a future dataset
  version instead of silently changing old labels.
