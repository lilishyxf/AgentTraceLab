# GEPA stability gate v1

`agenttracelab.gepa-stability.v1` evaluates whether a selected GEPA candidate remains better and
safe across repeated held-out executions. It addresses stochastic run variance; it does not prove
that the held-out tasks are independent or representative.

## Manifest

```json
{
  "schema_version": "agenttracelab.gepa-stability.v1",
  "stability_id": "office-agent-stability",
  "version": "1.0.0",
  "experiment_manifest": "experiment.json",
  "experiment_report": "evaluation-results/gepa-experiment.json",
  "held_out_replay_manifest": "validation-replay.json",
  "policy": {
    "trials_per_task": 5,
    "confidence_level": 0.95,
    "min_candidate_mean_score": 0.8,
    "min_paired_mean_delta": 0.0,
    "min_candidate_success_lower_bound": 0.6,
    "max_safety_regressions": 0,
    "require_fresh_traces": true
  }
}
```

`trials_per_task` is bounded to 2-20 and confidence level to 0.90, 0.95, or 0.99. Relative paths
resolve from the stability manifest.

## Paired execution

For every held-out task and trial, AgentTraceLab executes the experiment's seed candidate and the
search report's selected candidate against the same Replay contract. Baseline/candidate order
alternates across task/trial pairs to reduce fixed ordering bias. A single seen-trace set begins with
all training IDs and then covers both arms of every pair; missing, duplicate, or training-reused
trace evidence fails freshness.

Each paired sample records:

- baseline and candidate scores;
- their score delta;
- Replay expectation outcomes;
- baseline and candidate safety scores;
- both sets of trace IDs.

## Confidence gate

A candidate trial counts as a success only when Replay expectations match and safety compliance is
`1.0`. For `x` successes in `n` paired samples, the report computes the lower endpoint of the
configured two-sided Wilson score interval:

```text
p = x / n
lower = (p + z²/(2n) - z*sqrt((p(1-p) + z²/(4n))/n)) / (1 + z²/n)
```

The configured confidence level selects `z`. Promotion requires all configured mean-score, paired
delta, lower-bound, safety-regression, and freshness checks to pass.

## Interpretation limits

The Wilson calculation treats outcomes as Bernoulli observations. Reusing highly similar tasks,
shared provider incidents, deterministic fixture responses, or state leakage can correlate trials
and make the interval optimistic. Reported confidence therefore describes only the declared run
set; it is not a claim about all users or production traffic. A `promote` recommendation still
requires human approval and never performs deployment.
