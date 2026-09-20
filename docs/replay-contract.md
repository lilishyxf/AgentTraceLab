# Replay contract

AgentTraceLab v0.3 can execute a bounded set of JSON tasks against one HTTP endpoint. The target
must return a WasmHatch journal, a normalized AgentTraceLab trace, or OpenInference-style OTLP JSON.
The replay runner is intentionally CLI-only: exposing arbitrary target URLs through the public API
would create a server-side request forgery surface unless an allowlist and tenant policy were added.

## Response envelope

`response_path` is an optional dot-separated object path. For example, `trace` reads the trace from
`{"trace": {...}}`; `result.trace` reads it from `{"result": {"trace": {...}}}`. Arrays and general
JSONPath expressions are not accepted.

## Bounds and credentials

- One manifest may contain 1–100 tasks.
- A request body is limited to 256 KiB.
- A manifest is limited to 1 MiB.
- A response is streamed and stopped at the configured limit (1 KiB–5 MiB).
- Timeouts are bounded to 0.1–60 seconds per request.
- Redirects are disabled.
- Credentials cannot be embedded in the endpoint URL.
- Optional bearer credentials are read from an environment variable and never written to reports.

## Running

```bash
uv run agenttracelab run-replay evaluation/replays/example/v1/manifest.json \
  --output evaluation-results/office-agent.json
```

The example endpoint is illustrative and must be replaced by an actual local or deployed Agent
endpoint. AgentTraceLab exits with code `0` only when every completed result matches its declared
expectation. Transport and trace-contract failures never count as expected Agent failures.
