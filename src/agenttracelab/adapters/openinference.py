from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from heapq import heappop, heappush
from typing import Any

from agenttracelab.models import (
    DecisionEvidence,
    Disposition,
    SpanKind,
    SpanStatus,
    TraceEnvelope,
    TraceSource,
    TraceSpan,
)


def _field(value: dict[str, Any], camel: str, snake: str) -> Any:
    return value.get(camel, value.get(snake))


def _decode_any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    scalar_keys = ("stringValue", "boolValue", "intValue", "doubleValue", "bytesValue")
    snake_scalar_keys = ("string_value", "bool_value", "int_value", "double_value", "bytes_value")
    for key in scalar_keys + snake_scalar_keys:
        if key in value:
            return value[key]
    array = _field(value, "arrayValue", "array_value")
    if isinstance(array, dict):
        return [_decode_any_value(item) for item in array.get("values", [])]
    pairs = _field(value, "kvlistValue", "kvlist_value")
    if isinstance(pairs, dict):
        return _decode_attributes(pairs.get("values", []))
    return value


def _decode_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, list):
        return {}
    attributes: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            continue
        attributes[item["key"]] = _decode_any_value(item.get("value"))
    return attributes


def _span_kind(attributes: dict[str, Any], name: str) -> SpanKind:
    raw = attributes.get("openinference.span.kind")
    if isinstance(raw, str):
        try:
            return SpanKind(raw.upper())
        except ValueError:
            pass
    lowered = name.lower()
    if "tool" in lowered:
        return SpanKind.TOOL
    if "retriev" in lowered or "search" in lowered:
        return SpanKind.RETRIEVER
    if "model" in lowered or "llm" in lowered or "completion" in lowered:
        return SpanKind.LLM
    return SpanKind.CHAIN


def _span_status(status: Any) -> SpanStatus:
    if not isinstance(status, dict):
        return SpanStatus.UNSET
    code = status.get("code", status.get("statusCode", status.get("status_code", 0)))
    if code in (1, "1", "OK", "STATUS_CODE_OK"):
        return SpanStatus.OK
    if code in (2, "2", "ERROR", "STATUS_CODE_ERROR"):
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def _unix_nanos(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer Unix-nanosecond value") from exc
    if result < 0:
        raise ValueError(f"{label} must not be negative")
    return result


def _as_datetime(nanos: int) -> datetime:
    return datetime.fromtimestamp(nanos / 1_000_000_000, tz=UTC)


def _decision_from_span(
    *,
    trace_id: str,
    span_id: str,
    ended_at: datetime,
    attributes: dict[str, Any],
) -> DecisionEvidence | None:
    observation = attributes.get("agent.observation")
    selected_action = attributes.get("agent.selected_action", attributes.get("agent.action"))
    validation = attributes.get("agent.validation", attributes.get("agent.result_validation"))
    disposition_raw = attributes.get("agent.disposition")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (
            observation,
            selected_action,
            validation,
            disposition_raw,
        )
    ):
        return None
    try:
        disposition = Disposition(disposition_raw)
    except ValueError:
        return None
    tool_call_id = attributes.get("agent.tool_call_id")
    role = attributes.get("agent.role")
    reason = attributes.get("agent.reason")
    return DecisionEvidence(
        evidence_id=f"{trace_id}:decision:{span_id}",
        role=role if isinstance(role, str) and role else "agent",
        observation=observation,
        selected_action=selected_action,
        reason=reason if isinstance(reason, str) else "",
        tool_call_id=tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None,
        validation=validation,
        disposition=disposition,
        created_at=ended_at,
    )


def _iter_otlp_spans(payload: dict[str, Any]):
    resource_spans = _field(payload, "resourceSpans", "resource_spans")
    if not isinstance(resource_spans, list) or not resource_spans:
        raise ValueError("OTLP JSON payload must contain resourceSpans")
    for resource_entry in resource_spans:
        if not isinstance(resource_entry, dict):
            continue
        resource = resource_entry.get("resource", {})
        resource_attributes = (
            _decode_attributes(resource.get("attributes", [])) if isinstance(resource, dict) else {}
        )
        scope_spans = _field(resource_entry, "scopeSpans", "scope_spans")
        if not isinstance(scope_spans, list):
            continue
        for scope_entry in scope_spans:
            if not isinstance(scope_entry, dict):
                continue
            scope = scope_entry.get("scope", {}) if isinstance(scope_entry.get("scope", {}), dict) else {}
            spans = scope_entry.get("spans", [])
            if not isinstance(spans, list):
                continue
            for span in spans:
                if isinstance(span, dict):
                    yield resource_attributes, scope, span


def _order_spans_parent_first(
    prepared: list[tuple[int, int, str, dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> list[tuple[int, int, str, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    by_id = {item[2]: item for item in prepared}
    if len(by_id) != len(prepared):
        raise ValueError("OTLP trace contains duplicate spanId values")

    children: dict[str, list[str]] = defaultdict(list)
    indegree = {span_id: 0 for span_id in by_id}
    for span_id, item in by_id.items():
        parent = _field(item[5], "parentSpanId", "parent_span_id")
        if isinstance(parent, str) and parent in by_id:
            children[parent].append(span_id)
            indegree[span_id] = 1

    ready: list[tuple[int, int, str]] = []
    for span_id, degree in indegree.items():
        if degree == 0:
            start, end = by_id[span_id][:2]
            heappush(ready, (start, end, span_id))

    ordered = []
    while ready:
        _, _, span_id = heappop(ready)
        ordered.append(by_id[span_id])
        for child_id in children[span_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                start, end = by_id[child_id][:2]
                heappush(ready, (start, end, child_id))
    if len(ordered) != len(prepared):
        raise ValueError("OTLP trace contains a cycle in parent span relationships")
    return ordered


def adapt_otlp_json(payload: dict[str, Any]) -> tuple[TraceEnvelope, ...]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for resource, scope, span in _iter_otlp_spans(payload):
        trace_id = _field(span, "traceId", "trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("Every OTLP span must contain traceId")
        grouped[trace_id].append((resource, scope, span))
    if not grouped:
        raise ValueError("OTLP JSON payload contains no spans")

    traces: list[TraceEnvelope] = []
    for trace_id, raw_spans in sorted(grouped.items()):
        prepared: list[tuple[int, int, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for resource, scope, span in raw_spans:
            span_id = _field(span, "spanId", "span_id")
            if not isinstance(span_id, str) or not span_id:
                raise ValueError("Every OTLP span must contain spanId")
            start = _unix_nanos(
                _field(span, "startTimeUnixNano", "start_time_unix_nano"),
                "startTimeUnixNano",
            )
            end = _unix_nanos(_field(span, "endTimeUnixNano", "end_time_unix_nano"), "endTimeUnixNano")
            if end < start:
                raise ValueError("OTLP span end time must not precede start time")
            prepared.append((start, end, span_id, resource, scope, span))
        prepared = _order_spans_parent_first(prepared)

        normalized_spans: list[TraceSpan] = []
        decisions: list[DecisionEvidence] = []
        for sequence, (start, end, span_id, _resource, _scope, span) in enumerate(prepared, start=1):
            attributes = _decode_attributes(span.get("attributes", []))
            name = span.get("name") if isinstance(span.get("name"), str) else "unnamed span"
            parent_span_id = _field(span, "parentSpanId", "parent_span_id")
            parent = parent_span_id if isinstance(parent_span_id, str) and parent_span_id else None
            normalized = TraceSpan(
                span_id=span_id,
                parent_span_id=parent,
                sequence=sequence,
                kind=_span_kind(attributes, name),
                name=name,
                status=_span_status(span.get("status", {})),
                started_at=_as_datetime(start),
                ended_at=_as_datetime(end),
                duration_ms=(end - start) // 1_000_000,
                attributes=attributes,
            )
            normalized_spans.append(normalized)
            decision = _decision_from_span(
                trace_id=trace_id,
                span_id=span_id,
                ended_at=normalized.ended_at,
                attributes=attributes,
            )
            if decision:
                decisions.append(decision)

        first_resource, first_scope, _ = prepared[0][3:]
        root = next((span for span in normalized_spans if span.parent_span_id is None), normalized_spans[0])
        task_value = root.attributes.get("input.value")
        task = task_value[:4_096] if isinstance(task_value, str) else None
        state = (
            "needs_attention"
            if any(span.status == SpanStatus.ERROR for span in normalized_spans)
            else "completed"
        )
        traces.append(
            TraceEnvelope(
                trace_id=trace_id,
                source=TraceSource(kind="openinference", schema_version="otlp-json.v1"),
                state=state,
                task=task,
                started_at=normalized_spans[0].started_at,
                ended_at=max(span.ended_at for span in normalized_spans),
                spans=tuple(normalized_spans),
                decision_evidence=tuple(decisions),
                metadata={
                    "otel.resource": first_resource,
                    "otel.scope": first_scope,
                    "otel.span_count": len(normalized_spans),
                },
            )
        )
    return tuple(traces)
