from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agenttracelab.adapters import adapt_otlp_json
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.models import SpanKind


def otlp_fixture() -> dict:
    path = Path(__file__).parent / "fixtures" / "openinference_otlp.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_imports_openinference_otlp_trace() -> None:
    traces = adapt_otlp_json(otlp_fixture())

    assert len(traces) == 1
    trace = traces[0]
    assert trace.source.kind == "openinference"
    assert trace.task == "Inspect the table before answering."
    assert [span.kind for span in trace.spans] == [SpanKind.AGENT, SpanKind.TOOL]
    assert trace.spans[1].duration_ms == 1000
    assert trace.decision_evidence[0].tool_call_id == trace.spans[1].span_id
    assert evaluate_trace(trace).passed is True


def test_groups_multiple_otlp_trace_ids() -> None:
    payload = otlp_fixture()
    second = copy.deepcopy(payload["resourceSpans"][0]["scopeSpans"][0]["spans"][1])
    second["traceId"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    second["spanId"] = "3333333333333333"
    second.pop("parentSpanId")
    payload["resourceSpans"][0]["scopeSpans"][0]["spans"].append(second)

    traces = adapt_otlp_json(payload)

    assert [trace.trace_id for trace in traces] == [
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ]


def test_parent_precedes_child_when_clock_resolution_makes_start_times_equal() -> None:
    payload = otlp_fixture()
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    spans[1]["startTimeUnixNano"] = spans[0]["startTimeUnixNano"]

    trace = adapt_otlp_json(payload)[0]

    assert [span.kind for span in trace.spans] == [SpanKind.AGENT, SpanKind.TOOL]
    assert [span.sequence for span in trace.spans] == [1, 2]


def test_rejects_otlp_span_without_trace_id() -> None:
    payload = otlp_fixture()
    del payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]

    with pytest.raises(ValueError, match="traceId"):
        adapt_otlp_json(payload)
