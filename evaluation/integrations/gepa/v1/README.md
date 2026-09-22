# GEPA v1 integration evidence

`wasmhatch-recorded-local-evaluation.json` is the content-free output of AgentTraceLab 1.5's
`gepa-evaluation` command for the recorded-local WasmHatch normalization trace at:

`evaluation/recorded/wasmhatch-local-campaign/v1/local-normalization.json`

The source trace ID is `run_journal_16bcb21277afbb8514017c8f4ee1b3d7`. The export scored `1.0`
because that recorded workflow passed every applicable deterministic check. It proves the adapter
can derive GEPA-compatible `(score, info)` data from stored evidence.

It does **not** prove that GEPA ran, that a candidate was mutated, that a reflection model was
called, or that Agent/model quality improved. A genuine optimization experiment must execute every
candidate and ingest a fresh trace before scoring it.
