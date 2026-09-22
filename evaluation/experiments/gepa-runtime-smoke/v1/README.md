# Official GEPA runtime smoke v1

This generated-synthetic experiment invokes `gepa.optimize_anything` 0.1.4 rather than emulating its
search loop. A deterministic custom proposer changes one bound system-prompt candidate. The mock
Agent returns a fresh WasmHatch journal on every request, allowing AgentTraceLab to evaluate the
complete candidate execution path.

The seed candidate fails approval, validation, decision-evidence, terminal-state, and metric checks.
Its safety-gated score is `0.0`. The proposed candidate scores `1.0`; GEPA selects it after two
candidates and four metric calls. All four evaluations use distinct trace IDs.

The selected candidate is then replayed on two separately declared held-out requests. Both pass,
the validation score is `1.0`, the train-validation gap is `0.0`, and both validation trace IDs are
new relative to training. Because the mock transport returns fixture-derived journals, this
validates promotion-gate behavior rather than semantic generalization.

Finally, the same two held-out requests run for five paired baseline/candidate trials each. Across
10 pairs, baseline mean is `0.0`, candidate mean is `1.0`, paired mean delta is `1.0`, the candidate
passes 10/10 with a 95% Wilson lower bound of `0.7225`, and all 20 traces are fresh. Repetition here
validates the stability algorithm only because every response still comes from fixtures.

The hard-case miner treats the 10 repaired baseline failures as regression-guard candidates. Their
identical failed-check set becomes one failure signature, and the configured cap selects three
trace-reference entries instead of repeating all 10. No task prompt or tool payload is copied into
the queue.

The smoke generator then writes an explicit `generated-synthetic` resolution set and curates all
three selected fixes into the regression split. This verifies source matching, decision validation,
and immutable challenge-set construction; it is not human review.

Run:

```bash
uv sync --extra dev --extra optimization
uv run python examples/run_gepa_runtime_smoke.py \
  --output evaluation-results/gepa-runtime-smoke-v1.json \
  --validation-output evaluation-results/gepa-validation-smoke-v1.json \
  --stability-output evaluation-results/gepa-stability-smoke-v1.json \
  --hard-cases-output evaluation-results/gepa-hard-cases-smoke-v1.json \
  --resolutions-output evaluation-results/gepa-hard-case-resolutions-smoke-v1.json \
  --curated-output evaluation-results/gepa-curated-challenge-set-smoke-v1.json
```

This is runtime-integration evidence only. It does not use an LLM reflection model, a deployed Agent,
or held-out production tasks, and therefore is not a self-evolution or model-quality claim. A
`promote` result here means the synthetic gate passed; it is not deployment approval.
