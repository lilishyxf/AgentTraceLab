# Recorded WasmHatch local campaign v1

This campaign contains three run journals exported through the built WasmHatch browser UI after
separate executions of its local spreadsheet workflows:

- bundled local normalization;
- bundled invoice reconciliation;
- normalization of a CSV uploaded during the browser run.

Each run executed the QuickJS/Wasm transformation, staged a typed mutation proposal, required an
explicit foreground-user approval, committed the local effect, read the target back, and exported
the product run journal. The traces have distinct run IDs and were not copied from one template.

The campaign is intentionally provider-free. Structured decision evidence is attributed to
`host-workflow`; these records do not claim an LLM planned the actions. The uploaded CSV used generic
test data, and the exported journals declare that source contents and credential fields are absent.

AgentTraceLab evaluates all three runs at `100.0`, with process and recommended reward `1.0`. This
shows that the deterministic evidence contract holds across three local workflows. It is still a
small engineering validation set, not evidence of model quality, statistical significance, or
production reliability.
