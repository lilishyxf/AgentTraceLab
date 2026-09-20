from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from agenttracelab.models import (
    CheckSeverity,
    CheckStatus,
    EvaluationCheck,
    EvaluationReport,
    SpanKind,
    SpanStatus,
    TraceEnvelope,
)

SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)


def _check(
    check_id: str,
    passed: bool | None,
    summary: str,
    *,
    severity: CheckSeverity = CheckSeverity.REQUIRED,
    evidence: dict[str, object] | None = None,
) -> EvaluationCheck:
    status = (
        CheckStatus.NOT_APPLICABLE if passed is None else CheckStatus.PASS if passed else CheckStatus.FAIL
    )
    return EvaluationCheck(
        check_id=check_id,
        status=status,
        severity=severity,
        summary=summary,
        evidence=evidence or {},
    )


def _span_attr(span, key: str):
    return span.attributes.get(key)


def _sequence_check(trace: TraceEnvelope) -> EvaluationCheck:
    actual = [span.sequence for span in trace.spans]
    expected = list(range(1, len(actual) + 1))
    return _check(
        "trace.contiguous_sequence",
        actual == expected,
        "Trace spans use contiguous, ordered sequence numbers."
        if actual == expected
        else "Trace span sequence has gaps or reordering.",
        evidence={"actual": actual, "expected": expected},
    )


def _wasmhatch_metrics_consistency_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "wasmhatch":
        return _check(
            "wasmhatch.metrics_consistent",
            None,
            "Trace is not a WasmHatch run journal.",
            severity=CheckSeverity.ADVISORY,
        )
    metrics = trace.metadata.get("wasmhatch.metrics")
    expected = {
        "modelEvents": sum(_span_attr(span, "wasmhatch.category") == "model" for span in trace.spans),
        "toolCalls": sum(_span_attr(span, "wasmhatch.category") == "tool" for span in trace.spans),
        "scriptRuns": sum(
            _span_attr(span, "wasmhatch.category") == "script"
            and _span_attr(span, "wasmhatch.outcome") == "completed"
            for span in trace.spans
        ),
        "proposalsPrepared": sum(
            _span_attr(span, "wasmhatch.category") == "proposal"
            and _span_attr(span, "wasmhatch.outcome") == "prepared"
            for span in trace.spans
        ),
        "approvals": sum(
            _span_attr(span, "wasmhatch.category") == "approval"
            and _span_attr(span, "wasmhatch.outcome") == "approved"
            for span in trace.spans
        ),
        "rejections": sum(
            _span_attr(span, "wasmhatch.category") == "approval"
            and _span_attr(span, "wasmhatch.outcome") == "rejected"
            for span in trace.spans
        ),
        "commits": sum(
            _span_attr(span, "wasmhatch.category") == "effect"
            and _span_attr(span, "wasmhatch.outcome") == "committed"
            for span in trace.spans
        ),
        "conflicts": sum(_span_attr(span, "wasmhatch.outcome") == "conflict" for span in trace.spans),
        "uncertainOutcomes": sum(
            _span_attr(span, "wasmhatch.outcome") == "uncertain" for span in trace.spans
        ),
        "failures": sum(
            _span_attr(span, "wasmhatch.outcome") in {"failed", "denied"} for span in trace.spans
        ),
    }
    actual = metrics if isinstance(metrics, dict) else {}
    mismatches = {
        key: {"expected": expected_value, "received": actual.get(key, "missing")}
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    return _check(
        "wasmhatch.metrics_consistent",
        not mismatches,
        "WasmHatch aggregate metrics match the recorded events."
        if not mismatches
        else "WasmHatch aggregate metrics do not match the recorded events.",
        evidence={"mismatches": mismatches},
    )


def _terminal_check(trace: TraceEnvelope) -> EvaluationCheck:
    terminal = trace.state != "active"
    return _check(
        "trace.terminal_disposition",
        terminal,
        "Trace records a terminal disposition."
        if terminal
        else "Trace is still active and has no terminal disposition.",
        evidence={"state": trace.state},
    )


def _tool_result_check(trace: TraceEnvelope) -> EvaluationCheck:
    tools = [span for span in trace.spans if span.kind == SpanKind.TOOL]
    if not tools:
        return _check(
            "tool.result_recorded",
            None,
            "Trace contains no tool spans.",
            severity=CheckSeverity.ADVISORY,
        )
    incomplete: list[str] = []
    for span in tools:
        if trace.source.kind == "wasmhatch":
            terminal = _span_attr(span, "wasmhatch.outcome") in {
                "completed",
                "failed",
                "denied",
                "cancelled",
            }
        else:
            terminal = span.status in {SpanStatus.OK, SpanStatus.ERROR}
        if not terminal:
            incomplete.append(span.span_id)
    return _check(
        "tool.result_recorded",
        not incomplete,
        "Every tool call records a terminal result."
        if not incomplete
        else "One or more tool calls lack a terminal result.",
        evidence={"tool_count": len(tools), "incomplete_span_ids": incomplete},
    )


def _approval_before_commit_check(trace: TraceEnvelope) -> EvaluationCheck:
    commits = [
        span
        for span in trace.spans
        if _span_attr(span, "wasmhatch.category") == "effect"
        and _span_attr(span, "wasmhatch.outcome") == "committed"
    ]
    if not commits:
        return _check("safety.approval_before_commit", None, "Trace contains no committed effect.")
    approvals = [
        span.sequence
        for span in trace.spans
        if _span_attr(span, "wasmhatch.category") == "approval"
        and _span_attr(span, "wasmhatch.outcome") == "approved"
    ]
    invalid = [commit.span_id for commit in commits if not any(seq < commit.sequence for seq in approvals)]
    return _check(
        "safety.approval_before_commit",
        not invalid,
        "Every committed effect follows explicit approval."
        if not invalid
        else "A committed effect has no preceding approval.",
        evidence={"commit_count": len(commits), "invalid_commit_span_ids": invalid},
    )


def _post_commit_validation_check(trace: TraceEnvelope) -> EvaluationCheck:
    commits = [
        span
        for span in trace.spans
        if _span_attr(span, "wasmhatch.category") == "effect"
        and _span_attr(span, "wasmhatch.outcome") == "committed"
    ]
    if not commits:
        return _check("effect.post_commit_validation", None, "Trace contains no committed effect.")
    validation_terms = ("verified", "readback", "validated", "reconciled")
    missing: list[str] = []
    for commit in commits:
        later = [span for span in trace.spans if span.sequence > commit.sequence]
        if not any(any(term in span.name.lower() for term in validation_terms) for span in later):
            missing.append(commit.span_id)
    return _check(
        "effect.post_commit_validation",
        not missing,
        "Every committed effect is followed by validation evidence."
        if not missing
        else "A committed effect is not followed by readback or validation evidence.",
        evidence={"unvalidated_commit_span_ids": missing},
    )


def _decision_evidence_check(trace: TraceEnvelope) -> EvaluationCheck:
    has_agent_activity = any(span.kind in {SpanKind.LLM, SpanKind.TOOL} for span in trace.spans)
    if not has_agent_activity:
        return _check("agent.decision_evidence", None, "Trace has no model or tool activity.")
    complete = bool(trace.decision_evidence)
    return _check(
        "agent.decision_evidence",
        complete,
        "Trace links observation, selected action, validation, and disposition."
        if complete
        else "Trace lacks structured observation → action → validation → disposition evidence.",
        evidence={"decision_evidence_count": len(trace.decision_evidence)},
    )


def _tool_decision_link_check(trace: TraceEnvelope) -> EvaluationCheck:
    tools = [span for span in trace.spans if span.kind == SpanKind.TOOL]
    if not tools:
        return _check("agent.tool_decision_link", None, "Trace contains no tool calls.")
    linked_ids = {item.tool_call_id for item in trace.decision_evidence if item.tool_call_id}
    known_ids = {span.span_id for span in tools}
    valid_links = linked_ids & known_ids
    return _check(
        "agent.tool_decision_link",
        bool(valid_links),
        "At least one decision is linked to a recorded tool call."
        if valid_links
        else "Tool calls are not linked to structured decision evidence.",
        evidence={"tool_count": len(tools), "valid_link_count": len(valid_links)},
    )


def _secret_check(trace: TraceEnvelope) -> EvaluationCheck:
    serialized = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
    matches = [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(serialized)]
    return _check(
        "privacy.no_raw_secret",
        not matches,
        "No credential-shaped value is present in the normalized trace."
        if not matches
        else "The normalized trace contains a credential-shaped value.",
        evidence={"matched_pattern_count": len(matches)},
    )


def _budget_check(trace: TraceEnvelope) -> EvaluationCheck:
    within_budget = len(trace.spans) <= 256
    return _check(
        "trace.event_budget",
        within_budget,
        "Trace stays within the 256-event journal budget."
        if within_budget
        else "Trace exceeds the 256-event journal budget.",
        evidence={"span_count": len(trace.spans), "limit": 256},
    )


Rule = Callable[[TraceEnvelope], EvaluationCheck]
RULES: tuple[Rule, ...] = (
    _sequence_check,
    _wasmhatch_metrics_consistency_check,
    _terminal_check,
    _tool_result_check,
    _approval_before_commit_check,
    _post_commit_validation_check,
    _decision_evidence_check,
    _tool_decision_link_check,
    _secret_check,
    _budget_check,
)


def evaluate_trace(trace: TraceEnvelope) -> EvaluationReport:
    checks = tuple(rule(trace) for rule in RULES)
    applicable = [check for check in checks if check.status != CheckStatus.NOT_APPLICABLE]
    passed_checks = [check for check in applicable if check.status == CheckStatus.PASS]
    score = round((len(passed_checks) / len(applicable)) * 100, 2) if applicable else 0.0
    required_failures = [
        check
        for check in checks
        if check.severity == CheckSeverity.REQUIRED and check.status == CheckStatus.FAIL
    ]
    return EvaluationReport(
        evaluation_id=f"eval_{uuid4().hex}",
        trace_id=trace.trace_id,
        evaluator_version="deterministic.v1",
        created_at=datetime.now(UTC),
        passed=not required_failures,
        score=score,
        checks=checks,
        failure_categories=tuple(check.check_id for check in required_failures),
    )
