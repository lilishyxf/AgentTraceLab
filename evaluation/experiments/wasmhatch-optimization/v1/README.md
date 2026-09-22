# WasmHatch optimization experiment v1

This deterministic, generated experiment pairs five failure-injection baselines with corrected
candidate run journals using the real `wasmhatch.run-journal.v1` export shape. It covers missing
decision evidence, an unterminated tool result, approval-before-commit, post-commit validation, and
aggregate metric integrity.

Regenerate the fixtures from the checked-in complete WasmHatch journal:

```bash
uv run python evaluation/experiments/wasmhatch-optimization/v1/generate_fixtures.py
```

Run the paired comparison:

```bash
uv run agenttracelab compare-wasmhatch-batches \
  evaluation/experiments/wasmhatch-optimization/v1/baseline.json \
  evaluation/experiments/wasmhatch-optimization/v1/candidate.json \
  --output evaluation-results/wasmhatch-optimization-v1.json \
  --summary evaluation-results/wasmhatch-optimization-v1.md \
  --html evaluation-results/wasmhatch-optimization-v1.html \
  --junit evaluation-results/wasmhatch-optimization-v1.junit.xml
```

The fixtures are synthetic failure injections, not production traffic and not live model runs.
Their purpose is to prove that localization, reward shaping, safety gates, paired denominators, and
report generation behave deterministically. They cannot establish real-world Agent quality or
statistical significance.
