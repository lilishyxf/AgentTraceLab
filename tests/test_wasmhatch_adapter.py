from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from agenttracelab.adapters import adapt_wasmhatch_journal
from agenttracelab.models import SpanKind, SpanStatus


def test_maps_wasmhatch_events_and_decision_evidence(complete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)

    assert trace.trace_id == complete_journal["runId"]
    assert trace.source.schema_version == "wasmhatch.run-journal.v1"
    assert trace.task == complete_journal["context"]["task"]
    assert [span.sequence for span in trace.spans] == [1, 2, 3, 4, 5, 6]
    assert trace.spans[0].kind == SpanKind.LLM
    assert trace.spans[1].kind == SpanKind.TOOL
    assert trace.spans[4].status == SpanStatus.OK
    assert len(trace.decision_evidence) == 1
    assert trace.decision_evidence[0].tool_call_id == trace.spans[1].span_id


def test_rejects_non_contiguous_event_sequence(complete_journal: dict) -> None:
    payload = copy.deepcopy(complete_journal)
    payload["events"][2]["sequence"] = 9

    with pytest.raises(ValidationError, match="contiguous sequence"):
        adapt_wasmhatch_journal(payload)


def test_does_not_invent_missing_decision_evidence(incomplete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(incomplete_journal)

    assert trace.decision_evidence == ()
