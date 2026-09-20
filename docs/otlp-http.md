# OTLP/HTTP trace receiver

AgentTraceLab v0.8 exposes the standard OTLP/HTTP trace endpoint:

```text
POST /v1/traces
```

The receiver exists to ingest real Agent traces into the same deterministic evaluator used by the
offline datasets and WasmHatch batch gates. It is separate from the legacy
`POST /v1/traces/import/otlp` convenience endpoint, which returns AgentTraceLab-specific result
objects and remains available for compatibility.

## Supported request contract

| Concern | Supported behavior |
| --- | --- |
| Method and path | `POST /v1/traces` |
| Encodings | OTLP JSON and OTLP Protobuf |
| Content types | `application/json`, `application/x-protobuf` |
| Compression | identity/no compression and `Content-Encoding: gzip` |
| Successful response | HTTP 200 with an empty `ExportTraceServiceResponse` in the request media type |
| Empty export | Accepted as a successful no-op |

JSON trace and span identifiers stay in the OTLP hexadecimal representation. Protobuf `bytes`
identifiers are converted to the same hexadecimal representation before normalization, so both
encodings produce the same stored identity.

Unknown OTLP JSON fields are ignored by the adapter. OpenInference-style attributes such as
`openinference.span.kind` and the AgentTraceLab decision-evidence attributes are decoded from the
standard OTLP attribute list.

## Resource bounds

The body is read as a bounded stream before parsing. Gzip payloads are then decompressed with a
second independent limit, so a small compressed request cannot expand without bound.

| Limit | Default | Environment variable |
| --- | ---: | --- |
| Bytes received over HTTP | 4 MiB | `AGENTTRACELAB_OTLP_MAX_WIRE_BYTES` |
| Bytes after decompression | 64 MiB | `AGENTTRACELAB_OTLP_MAX_DECODED_BYTES` |

Both values must be positive integers. `create_app` also accepts `otlp_max_wire_bytes` and
`otlp_max_decoded_bytes` for tests or embedded deployments. Wire or decoded overflow returns HTTP
413. Malformed JSON, Protobuf, or gzip returns HTTP 400; unsupported media types or content
encodings return HTTP 415. Error bodies use the OTLP `google.rpc.Status` shape in the selected
response media type.

## Current boundaries

- The receiver is OTLP/HTTP only; it does not implement OTLP/gRPC.
- Only trace exports are accepted. Metrics and logs are outside this endpoint.
- A non-empty export is handled atomically by the current adapter. AgentTraceLab does not yet emit
  an OTLP `partial_success` response.
- Successful ingestion means the trace was decoded, normalized, stored, and evaluated. It does not
  mean the Agent passed every deterministic check; retrieve the stored evaluation for that result.
- Export, retry, authentication, multi-tenant quotas, and collector-grade backpressure remain
  deployment responsibilities.

The protocol behavior follows the OpenTelemetry OTLP specification and the canonical trace service
Protobuf definition. See [references.md](references.md) for source links and reuse boundaries.

## Real SDK smoke path

Start AgentTraceLab, then run:

```bash
uv sync --extra otel
uv run python examples/send_otel_trace.py
```

This example uses the official OpenTelemetry Python SDK, `BatchSpanProcessor`, and the OTLP
HTTP/Protobuf exporter. It emits one Agent span and one linked Tool span with structured
observation, action, validation, and stop evidence. The automated end-to-end test starts a real
HTTP server, runs the same exporter, retrieves the stored trace, and checks the evaluation result.

Override the full signal-specific endpoint with either `--endpoint` or
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
