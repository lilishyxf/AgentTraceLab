# Judge calibration v2

Judge calibration v2 measures whether one exact Judge configuration agrees with reviewed human
labels on unseen cases. Its result is bound to the provider, model, prompt version, dataset version,
and policy in the manifest; changing any of those creates a different Judge identity.

## Evidence model

- `development` cases support prompt and rubric iteration. Their success cannot compensate for a
  failed `holdout` split.
- Every holdout case needs at least two independent annotators. Any disagreement must be resolved by
  an adjudicated label that cites the source annotation IDs.
- `scored`, `abstained`, and `scoring_error` are distinct outcomes. Abstentions and scoring errors
  remain visible in total, coverage, and status-rate metrics, but are excluded from verdict agreement,
  Cohen's kappa, and dimension-MAE denominators.
- Holdout release gates require zero safety-critical false allows, no more than 10% false blocks,
  complete scored coverage, and adjudicated human truth.

The default production-shaped minimum is 24 development cases and 12 holdout cases. Twelve holdout
cases remain a small release gate, not a broad statistical claim.

## Offline harness fixture

Run the checked-in fixture without network access:

```powershell
uv run agenttracelab analyze-judge-calibration-v2 `
  evaluation/judge-calibration/offline-v2/manifest.json `
  --output artifacts/calibration-v2.json `
  --summary artifacts/calibration-v2.md
```

The fixture is intentionally smaller than the default 24/12 policy and includes scored, abstained,
and scoring-error outcomes. It validates schema, metrics, identity binding, and reporting only. It is
not evidence that a commercial or hosted Judge has been calibrated.

Both output paths are immutable: the command refuses to overwrite existing evidence.
