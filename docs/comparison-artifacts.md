# Comparison artifacts

`compare-wasmhatch-batches` can produce four views of the same typed comparison report:

```bash
uv run agenttracelab compare-wasmhatch-batches baseline/manifest.json candidate/manifest.json \
  --output evaluation-results/comparison.json \
  --summary evaluation-results/comparison.md \
  --html evaluation-results/comparison.html \
  --junit evaluation-results/comparison.junit.xml
```

- JSON is the machine-readable source for later analysis.
- Markdown can be written to `GITHUB_STEP_SUMMARY` or attached to a review.
- HTML is a standalone, dependency-free report for local inspection or a portfolio demonstration.
- JUnit XML lets CI systems expose scenario regressions as failed test cases.

The HTML renderer escapes every dynamic identifier and applies a restrictive content security
policy. It loads no scripts, fonts, analytics, or external assets. The JUnit report includes one
case per paired scenario, failures for removed baseline coverage, skipped cases for candidate-only
scenarios, and a separate promotion-gate case.

These are different serializations of the same result, not four separate evaluations. None of them
upgrades synthetic evidence into a live experiment or turns a descriptive mean into a statistical
claim.
