from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from google.protobuf.json_format import MessageToDict, MessageToJson
from google.protobuf.message import DecodeError
from google.rpc.status_pb2 import Status
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

OTLP_JSON_CONTENT_TYPE = "application/json"
OTLP_PROTOBUF_CONTENT_TYPE = "application/x-protobuf"
SUPPORTED_CONTENT_TYPES = frozenset({OTLP_JSON_CONTENT_TYPE, OTLP_PROTOBUF_CONTENT_TYPE})

DEFAULT_MAX_WIRE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_DECODED_BYTES = 64 * 1024 * 1024


class OtlpHttpError(ValueError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DecodedOtlpRequest:
    content_type: str
    payload: dict[str, Any]
    span_count: int


def _normalize_content_type(value: str | None) -> str:
    content_type = (value or "").split(";", maxsplit=1)[0].strip().lower()
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise OtlpHttpError(
            415,
            "OTLP/HTTP traces require Content-Type application/json or application/x-protobuf",
        )
    return content_type


async def _read_limited_body(request: Request, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise OtlpHttpError(400, "Content-Length must be an integer") from exc
        if declared_length < 0:
            raise OtlpHttpError(400, "Content-Length must not be negative")
        if declared_length > max_bytes:
            raise OtlpHttpError(413, f"OTLP request exceeds the {max_bytes}-byte wire limit")

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise OtlpHttpError(413, f"OTLP request exceeds the {max_bytes}-byte wire limit")
    return bytes(body)


def _decompress_gzip_limited(body: bytes, max_bytes: int) -> bytes:
    try:
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoded = bytearray(decoder.decompress(body, max_bytes + 1))
        if len(decoded) > max_bytes or decoder.unconsumed_tail:
            raise OtlpHttpError(413, f"Decoded OTLP request exceeds the {max_bytes}-byte limit")
        decoded.extend(decoder.flush(max_bytes + 1 - len(decoded)))
    except OtlpHttpError:
        raise
    except zlib.error as exc:
        raise OtlpHttpError(400, "OTLP request has an invalid gzip body") from exc

    if len(decoded) > max_bytes:
        raise OtlpHttpError(413, f"Decoded OTLP request exceeds the {max_bytes}-byte limit")
    if not decoder.eof or decoder.unused_data:
        raise OtlpHttpError(400, "OTLP request has an incomplete or trailing gzip body")
    return bytes(decoded)


def _decode_body(body: bytes, content_encoding: str | None, max_decoded_bytes: int) -> bytes:
    encoding = (content_encoding or "identity").strip().lower()
    if encoding in {"", "identity"}:
        if len(body) > max_decoded_bytes:
            raise OtlpHttpError(
                413,
                f"Decoded OTLP request exceeds the {max_decoded_bytes}-byte limit",
            )
        return body
    if encoding == "gzip":
        return _decompress_gzip_limited(body, max_decoded_bytes)
    raise OtlpHttpError(415, "OTLP/HTTP traces support only identity or gzip Content-Encoding")


def _restore_hex_ids(
    message: ExportTraceServiceRequest,
    payload: dict[str, Any],
) -> None:
    resource_payloads = payload.get("resourceSpans", [])
    if not isinstance(resource_payloads, list):
        return
    for resource_message, resource_payload in zip(
        message.resource_spans,
        resource_payloads,
        strict=False,
    ):
        if not isinstance(resource_payload, dict):
            continue
        scope_payloads = resource_payload.get("scopeSpans", [])
        if not isinstance(scope_payloads, list):
            continue
        for scope_message, scope_payload in zip(
            resource_message.scope_spans,
            scope_payloads,
            strict=False,
        ):
            if not isinstance(scope_payload, dict):
                continue
            span_payloads = scope_payload.get("spans", [])
            if not isinstance(span_payloads, list):
                continue
            for span_message, span_payload in zip(scope_message.spans, span_payloads, strict=False):
                if not isinstance(span_payload, dict):
                    continue
                span_payload["traceId"] = span_message.trace_id.hex()
                span_payload["spanId"] = span_message.span_id.hex()
                if span_message.parent_span_id:
                    span_payload["parentSpanId"] = span_message.parent_span_id.hex()


def _decode_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OtlpHttpError(400, "OTLP JSON request body is malformed") from exc
    if not isinstance(payload, dict):
        raise OtlpHttpError(400, "OTLP JSON request body must be an object")
    return payload


def _decode_protobuf(body: bytes) -> dict[str, Any]:
    request_message = ExportTraceServiceRequest()
    try:
        request_message.ParseFromString(body)
    except DecodeError as exc:
        raise OtlpHttpError(400, "OTLP Protobuf request body is malformed") from exc
    payload = MessageToDict(request_message, preserving_proto_field_name=False)
    _restore_hex_ids(request_message, payload)
    return payload


def count_otlp_spans(payload: dict[str, Any]) -> int:
    count = 0
    resource_spans = payload.get("resourceSpans", payload.get("resource_spans", []))
    if not isinstance(resource_spans, list):
        return 0
    for resource_entry in resource_spans:
        if not isinstance(resource_entry, dict):
            continue
        scope_spans = resource_entry.get("scopeSpans", resource_entry.get("scope_spans", []))
        if not isinstance(scope_spans, list):
            continue
        for scope_entry in scope_spans:
            if not isinstance(scope_entry, dict):
                continue
            spans = scope_entry.get("spans", [])
            if isinstance(spans, list):
                count += len(spans)
    return count


async def decode_otlp_http_request(
    request: Request,
    *,
    max_wire_bytes: int = DEFAULT_MAX_WIRE_BYTES,
    max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES,
) -> DecodedOtlpRequest:
    content_type = _normalize_content_type(request.headers.get("content-type"))
    wire_body = await _read_limited_body(request, max_wire_bytes)
    body = _decode_body(wire_body, request.headers.get("content-encoding"), max_decoded_bytes)
    payload = _decode_json(body) if content_type == OTLP_JSON_CONTENT_TYPE else _decode_protobuf(body)
    return DecodedOtlpRequest(
        content_type=content_type,
        payload=payload,
        span_count=count_otlp_spans(payload),
    )


def otlp_success_response(content_type: str) -> Response:
    if content_type == OTLP_PROTOBUF_CONTENT_TYPE:
        body = ExportTraceServiceResponse().SerializeToString()
    else:
        body = b"{}"
    return Response(content=body, status_code=200, media_type=content_type)


def otlp_error_response(content_type: str | None, error: OtlpHttpError) -> Response:
    normalized = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    grpc_code = 8 if error.status_code == 413 else 3
    status = Status(code=grpc_code, message=str(error))
    if normalized == OTLP_PROTOBUF_CONTENT_TYPE:
        return Response(
            content=status.SerializeToString(),
            status_code=error.status_code,
            media_type=normalized,
        )
    body = MessageToJson(status, indent=None).encode()
    return Response(content=body, status_code=error.status_code, media_type=OTLP_JSON_CONTENT_TYPE)
