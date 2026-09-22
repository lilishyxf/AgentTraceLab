# Hard-case curation v1

Hard-case curation converts a mined pointer queue into a reviewed, immutable challenge-set version.
It validates review metadata and split assignment but does not retrieve or copy raw trace content.

## Resolution set

`agenttracelab.hard-case-resolutions.v1` declares:

- adjudication ID and version;
- exact source queue dataset ID and version;
- `human` or `generated-synthetic` review mode;
- reviewer identity and timestamp;
- completeness requirement and deterministic split seed/ratios;
- one resolution per reviewed case.

Each resolution selects `regression_guard`, `training_candidate`, or `reject`, with a fixed
rationale code. Free-form notes are intentionally excluded so the content-free artifact cannot
silently copy prompts or tool data. Complete adjudication rejects omitted cases. Unknown and
duplicate case IDs always fail closed.

## Decision invariants

- `regression_guard` is accepted only for `resolved_baseline_failure` with `confirmed_fix`;
- `training_candidate` requires `confirmed_failure`;
- `reject` preserves the decision in aggregate counts and the rejected case ID list;
- source dataset identity and version must match the queue exactly.

These rules prevent an unresolved candidate failure from being silently described as a fixed
regression case.

## Leakage-aware split

Regression guards go to the dedicated `regression` split. Training candidates are assigned by
hashing `(split_seed, task_id)` into the configured train/validation/test ratios. Because assignment
uses `task_id`, repeated trials of the same task cannot cross these splits.

This controls exact task leakage only. Semantically duplicated tasks with different IDs still need
separate similarity or provenance review.

## Immutability and evidence status

The CLI refuses to overwrite an existing output:

```bash
uv run agenttracelab curate-gepa-hard-cases HARD_CASE_QUEUE RESOLUTIONS \
  --output curated-challenge-set-v1.json
```

A changed decision requires a new adjudication or dataset version. `generated-synthetic` review is
test evidence and must never be reported as human labeling. Curated entries remain trace pointers;
materializing prompts, targets, or tool payloads requires a separate authorized privacy workflow.
