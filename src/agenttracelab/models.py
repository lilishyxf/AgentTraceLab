from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpanKind(StrEnum):
    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"
    CHAIN = "CHAIN"


class SpanStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class Disposition(StrEnum):
    CONTINUE = "continue"
    REPLAN = "replan"
    STOP = "stop"
    HUMAN_REQUIRED = "human-required"


class TraceSource(FrozenModel):
    kind: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=128)


class TraceSpan(FrozenModel):
    span_id: str = Field(min_length=1, max_length=256)
    parent_span_id: str | None = Field(default=None, max_length=256)
    sequence: int = Field(ge=1)
    kind: SpanKind
    name: str = Field(min_length=1, max_length=256)
    status: SpanStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: int = Field(ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_range(self) -> TraceSpan:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        return self


class DecisionEvidence(FrozenModel):
    evidence_id: str = Field(min_length=1, max_length=256)
    role: str = Field(default="agent", min_length=1, max_length=64)
    observation: str = Field(min_length=1, max_length=2_000)
    selected_action: str = Field(min_length=1, max_length=2_000)
    reason: str = Field(default="", max_length=2_000)
    tool_call_id: str | None = Field(default=None, max_length=256)
    validation: str = Field(min_length=1, max_length=2_000)
    disposition: Disposition
    created_at: datetime


class TraceEnvelope(FrozenModel):
    schema_version: Literal["agenttracelab.trace.v1"] = "agenttracelab.trace.v1"
    trace_id: str = Field(min_length=1, max_length=256)
    source: TraceSource
    state: str = Field(min_length=1, max_length=64)
    task: str | None = Field(default=None, max_length=4_096)
    started_at: datetime
    ended_at: datetime
    spans: tuple[TraceSpan, ...]
    decision_evidence: tuple[DecisionEvidence, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trace(self) -> TraceEnvelope:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        sequences = [span.sequence for span in self.spans]
        if len(sequences) != len(set(sequences)):
            raise ValueError("span sequences must be unique")
        if sequences != sorted(sequences):
            raise ValueError("spans must be ordered by sequence")
        return self


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class CheckSeverity(StrEnum):
    REQUIRED = "required"
    ADVISORY = "advisory"


class EvaluationCheck(FrozenModel):
    check_id: str = Field(min_length=1, max_length=128)
    status: CheckStatus
    severity: CheckSeverity
    summary: str = Field(min_length=1, max_length=512)
    evidence: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(FrozenModel):
    schema_version: Literal["agenttracelab.evaluation.v1"] = "agenttracelab.evaluation.v1"
    evaluation_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=256)
    evaluator_version: str = Field(min_length=1, max_length=128)
    created_at: datetime
    passed: bool
    score: float = Field(ge=0, le=100)
    checks: tuple[EvaluationCheck, ...]
    failure_categories: tuple[str, ...] = ()


class ComparisonReport(FrozenModel):
    schema_version: Literal["agenttracelab.comparison.v1"] = "agenttracelab.comparison.v1"
    baseline_trace_id: str
    candidate_trace_id: str
    baseline_score: float
    candidate_score: float
    score_delta: float
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    recommendation: Literal["promote", "hold"]
