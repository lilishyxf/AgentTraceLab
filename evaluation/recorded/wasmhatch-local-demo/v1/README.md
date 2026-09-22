# Recorded WasmHatch local demo v1

This journal was downloaded from the actual WasmHatch browser UI during its local normalization
demo. Playwright opened the built application, ran the QuickJS/Wasm transformation, approved the
staged proposal, committed the local spreadsheet effect, and used the product's **Export JSON**
action. No provider call or external account was used.

AgentTraceLab observes a deterministic score of `70.0` and a recommended reward of `0.7`. Approval,
metric integrity, terminal state, secret detection, and event budget checks pass. Three checks fail:

- `agent.decision_evidence`: the script event has no structured observation/action/validation record;
- `agent.tool_decision_link`: the script span is not linked from decision evidence;
- `effect.post_commit_validation`: the journal commits the effect but records no later readback.

This is a recorded integration baseline, not a passing quality claim. The raw source contents and
credentials are excluded by WasmHatch's exporter.

`candidate-journal.json` is a second, separately captured browser run after WasmHatch added:

- structured observation/action/validation/disposition evidence to the script event;
- an exact `tool_call_id` link from that decision to the script span;
- an actual post-commit connector readback, with an explicit validated or uncertain journal event.

The decision evidence is explicitly attributed to `host-workflow`, not to an LLM or autonomous
planner.

The paired recorded-local comparison is `70.0 → 100.0`, process and recommended reward
`0.7 → 1.0`, with three deterministic check improvements and no regressions. This measures trace
completeness and post-commit verification for one local workflow. It does not measure model quality,
task accuracy across a dataset, or production reliability.
