from __future__ import annotations

import gzip
import json

from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from agenttracelab.api import create_app


def _client(*, wire_limit: int = 4 * 1024 * 1024, decoded_limit: int = 64 * 1024 * 1024):
    return TestClient(
        create_app(
            "sqlite+pysqlite:///:memory:",
            otlp_max_wire_bytes=wire_limit,
            otlp_max_decoded_bytes=decoded_limit,
        )
    )


def _protobuf_request() -> bytes:
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    resource_attribute = resource_spans.resource.attributes.add()
    resource_attribute.key = "service.name"
    resource_attribute.value.string_value = "protobuf-agent"
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "agenttracelab.protobuf-test"
    span = scope_spans.spans.add()
    span.trace_id = bytes.fromhex("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    span.span_id = bytes.fromhex("3333333333333333")
    span.name = "protobuf-agent-run"
    span.start_time_unix_nano = 1_789_862_400_000_000_000
    span.end_time_unix_nano = 1_789_862_401_000_000_000
    span.status.code = 1
    for key, value in {
        "openinference.span.kind": "AGENT",
        "agent.observation": "The request needs no tool.",
        "agent.selected_action": "Answer directly.",
        "agent.validation": "The answer matches the request.",
        "agent.disposition": "stop",
    }.items():
        attribute = span.attributes.add()
        attribute.key = key
        attribute.value.string_value = value
    return request.SerializeToString()


def test_otlp_http_accepts_json_and_persists_trace(otlp_payload: dict) -> None:
    client = _client()

    response = client.post("/v1/traces", json=otlp_payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {}
    stored = client.get("/v1/traces/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert stored.status_code == 200
    assert stored.json()["source"]["kind"] == "openinference"


def test_otlp_http_accepts_protobuf_and_preserves_hex_trace_id() -> None:
    client = _client()

    response = client.post(
        "/v1/traces",
        content=_protobuf_request(),
        headers={"content-type": "application/x-protobuf"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-protobuf")
    assert response.content == b""
    stored = client.get("/v1/traces/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    assert stored.status_code == 200
    assert stored.json()["spans"][0]["span_id"] == "3333333333333333"


def test_otlp_http_accepts_gzip_json(otlp_payload: dict) -> None:
    client = _client()
    body = gzip.compress(json.dumps(otlp_payload).encode())

    response = client.post(
        "/v1/traces",
        content=body,
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )

    assert response.status_code == 200
    assert client.get("/v1/traces/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").status_code == 200


def test_otlp_http_treats_empty_export_as_success() -> None:
    client = _client()

    json_response = client.post("/v1/traces", json={})
    protobuf_response = client.post(
        "/v1/traces",
        content=b"",
        headers={"content-type": "application/x-protobuf"},
    )

    assert json_response.status_code == 200
    assert protobuf_response.status_code == 200


def test_otlp_http_rejects_unsupported_media_type_and_encoding() -> None:
    client = _client()

    media_type = client.post(
        "/v1/traces",
        content=b"{}",
        headers={"content-type": "text/plain"},
    )
    encoding = client.post(
        "/v1/traces",
        content=b"{}",
        headers={"content-type": "application/json", "content-encoding": "br"},
    )

    assert media_type.status_code == 415
    assert media_type.json()["code"] == 3
    assert encoding.status_code == 415


def test_otlp_http_rejects_malformed_json_protobuf_and_gzip() -> None:
    client = _client()

    malformed_json = client.post(
        "/v1/traces",
        content=b"{",
        headers={"content-type": "application/json"},
    )
    malformed_protobuf = client.post(
        "/v1/traces",
        content=b"\x0a\xff",
        headers={"content-type": "application/x-protobuf"},
    )
    malformed_gzip = client.post(
        "/v1/traces",
        content=b"not-gzip",
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )

    assert malformed_json.status_code == 400
    assert malformed_protobuf.status_code == 400
    assert malformed_gzip.status_code == 400


def test_otlp_http_enforces_wire_and_decoded_limits() -> None:
    wire_client = _client(wire_limit=8, decoded_limit=128)
    decoded_client = _client(wire_limit=128, decoded_limit=8)

    wire_response = wire_client.post(
        "/v1/traces",
        content=b'{"padding":"too-long"}',
        headers={"content-type": "application/json"},
    )
    decoded_response = decoded_client.post(
        "/v1/traces",
        content=gzip.compress(b'{"padding":"too-long"}'),
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )

    assert wire_response.status_code == 413
    assert wire_response.json()["code"] == 8
    assert decoded_response.status_code == 413
