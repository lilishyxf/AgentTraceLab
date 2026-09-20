from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agenttracelab.models import (
    DecisionEvidence,
    Disposition,
    SpanKind,
    SpanStatus,
    TraceEnvelope,
    TraceSource,
    TraceSpan,
)


class WasmHatchModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class WasmHatchEvent(WasmHatchModel):
    sequence: int = Field(ge=1)
    occurredAt: datetime
    elapsedMs: int = Field(ge=0)
    category: str
    outcome: str
    summary: str
    detail: str = ""
    evidence: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class WasmHatchJournal(WasmHatchModel):
    schemaVersion: Literal[1]
    runId: str
    startedAt: datetime
    updatedAt: datetime
    state: str
    events: tuple[WasmHatchEvent, ...]
    metrics: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] | None = None
    privacy: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_sequence(self) -> WasmHatchJournal:
        expected = list(range(1, len(self.events) + 1))
        actual = [event.sequence for event in self.events]
        if actual != expected:
            raise ValueError("WasmHatch events must use contiguous sequence numbers starting at 1")
        return self


CATEGORY_KIND = {
    "system": SpanKind.CHAIN,
    "source": SpanKind.RETRIEVER,
    "model": SpanKind.LLM,
    "tool": SpanKind.TOOL,
    "script": SpanKind.TOOL,
    "policy": SpanKind.GUARDRAIL,
    "proposal": SpanKind.CHAIN,
    "approval": SpanKind.GUARDRAIL,
    "effect": SpanKind.CHAIN,
    "export": SpanKind.CHAIN,
}

OK_OUTCOMES = {"allowed", "completed", "prepared", "approved", "rejected", "committed"}
ERROR_OUTCOMES = {"denied", "conflict", "failed"}


def _status_for(outcome: str) -> SpanStatus:
    if outcome in OK_OUTCOMES:
        return SpanStatus.OK
    if outcome in ERROR_OUTCOMES:
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _decision_evidence(event: WasmHatchEvent, run_id: str) -> DecisionEvidence | None:
    evidence = event.evidence
    observation = _text(evidence.get("observation"))
    selected_action = _text(evidence.get("selected_action"))
    validation = _text(evidence.get("validation"))
    disposition_raw = _text(evidence.get("disposition"))
    if not all((observation, selected_action, validation, disposition_raw)):
        return None
    try:
        disposition = Disposition(disposition_raw)
    except ValueError:
        return None
    return DecisionEvidence(
        evidence_id=f"{run_id}:decision:{event.sequence}",
        role=_text(evidence.get("role")) or "agent",
        observation=observation,
        selected_action=selected_action,
        reason=_text(evidence.get("reason")) or "",
        tool_call_id=_text(evidence.get("tool_call_id")),
        validation=validation,
        disposition=disposition,
        created_at=event.occurredAt,
    )


def adapt_wasmhatch_journal(payload: dict[str, Any] | WasmHatchJournal) -> TraceEnvelope:
    journal = payload if isinstance(payload, WasmHatchJournal) else WasmHatchJournal.model_validate(payload)
    spans = tuple(
        TraceSpan(
            span_id=f"{journal.runId}:event:{event.sequence}",
            parent_span_id=journal.runId,
            sequence=event.sequence,
            kind=CATEGORY_KIND.get(event.category, SpanKind.CHAIN),
            name=event.summary,
            status=_status_for(event.outcome),
            started_at=event.occurredAt,
            ended_at=event.occurredAt,
            duration_ms=0,
            attributes={
                "wasmhatch.category": event.category,
                "wasmhatch.outcome": event.outcome,
                "wasmhatch.elapsed_ms": event.elapsedMs,
                "wasmhatch.detail": event.detail,
                **{f"wasmhatch.evidence.{key}": value for key, value in event.evidence.items()},
            },
        )
        for event in journal.events
    )
    decisions = tuple(
        decision
        for event in journal.events
        if (decision := _decision_evidence(event, journal.runId)) is not None
    )
    context = journal.context or {}
    task = context.get("task") if isinstance(context.get("task"), str) else None
    return TraceEnvelope(
        trace_id=journal.runId,
        source=TraceSource(kind="wasmhatch", schema_version="wasmhatch.run-journal.v1"),
        state=journal.state,
        task=task,
        started_at=journal.startedAt,
        ended_at=journal.updatedAt,
        spans=spans,
        decision_evidence=decisions,
        metadata={
            "wasmhatch.metrics": journal.metrics,
            "wasmhatch.context": context,
            "wasmhatch.privacy": journal.privacy or {},
        },
    )
