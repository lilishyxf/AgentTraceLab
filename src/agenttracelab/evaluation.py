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


def _deepresearch_contract_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.required_contracts_pass",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    metrics = trace.metadata.get("deepresearch.contracts", {})
    total = int(metrics.get("required_total", 0) or 0)
    passed = int(metrics.get("required_passed", 0) or 0)
    complete = total > 0 and passed == total
    return _check(
        "deepresearch.required_contracts_pass",
        complete,
        "Every required completion contract passed."
        if complete
        else "Required completion contracts are missing, unresolved, or failed.",
        evidence={"required_total": total, "required_passed": passed},
    )


def _deepresearch_citation_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.citations_valid",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    metrics = trace.metadata.get("deepresearch.citations", {})
    count = int(metrics.get("count", 0) or 0)
    valid_count = int(metrics.get("valid_count", 0) or 0)
    evidence_count = int(trace.metadata.get("deepresearch.evidence_count", 0) or 0)
    passed = count > 0 and count == valid_count if evidence_count else count == 0
    return _check(
        "deepresearch.citations_valid",
        passed,
        "Report citations resolve to exported evidence."
        if passed
        else "The report has missing or unresolved evidence citations.",
        evidence={
            "citation_count": count,
            "valid_citation_count": valid_count,
            "evidence_count": evidence_count,
        },
    )


def _deepresearch_checkpoint_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.checkpoints_intact",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    count = int(trace.metadata.get("deepresearch.checkpoint_count", 0) or 0)
    rate = trace.metadata.get("deepresearch.checkpoint_integrity_rate")
    passed = count > 0 and rate == 1.0
    return _check(
        "deepresearch.checkpoints_intact",
        passed,
        "All exported checkpoint hashes are valid."
        if passed
        else "Checkpoints are missing or at least one checkpoint hash is invalid.",
        evidence={"checkpoint_count": count, "integrity_rate": rate},
    )


def _deepresearch_source_isolation_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.source_isolation",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    leakage = bool(trace.metadata.get("deepresearch.source_leakage_detected", False))
    return _check(
        "deepresearch.source_isolation",
        not leakage,
        "Evidence source modes match the run source mode."
        if not leakage
        else "The run contains evidence from a disallowed source mode.",
        evidence={
            "source_mode": trace.metadata.get("deepresearch.source_mode"),
            "leakage_detected": leakage,
        },
    )


def _deepresearch_tool_accounting_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.tool_accounting_consistent",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    usage_container = trace.metadata.get("deepresearch.usage", {})
    usage = usage_container.get("usage", usage_container) if isinstance(usage_container, dict) else {}
    reported = int(usage.get("tool_calls", 0) or 0)
    recorded = int(trace.metadata.get("deepresearch.tool_call_count", 0) or 0)
    passed = reported == recorded
    return _check(
        "deepresearch.tool_accounting_consistent",
        passed,
        "Reported tool-call usage matches durable tool-call records."
        if passed
        else "Reported tool-call usage does not match durable tool-call records.",
        evidence={"reported_tool_calls": reported, "recorded_tool_calls": recorded},
    )


def _deepresearch_duplicate_tool_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.duplicate_tool_calls_absent",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    duplicates = int(trace.metadata.get("deepresearch.duplicate_tool_call_fingerprints", 0) or 0)
    return _check(
        "deepresearch.duplicate_tool_calls_absent",
        duplicates == 0,
        "No duplicate task/tool/argument fingerprints were recorded."
        if duplicates == 0
        else "The run repeated an identical tool call for the same task and source mode.",
        evidence={"duplicate_tool_call_fingerprints": duplicates},
    )


def _deepresearch_verification_recovery_routing_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.verification_recovery_routing",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    misrouted = int(trace.metadata.get("deepresearch.misrouted_report_replan_count", 0) or 0)
    evidence = {
        "verification_failure_count": int(
            trace.metadata.get("deepresearch.verification_failure_count", 0) or 0
        ),
        "repeated_verification_failure_count": int(
            trace.metadata.get("deepresearch.repeated_verification_failure_count", 0) or 0
        ),
        "misrouted_report_replan_count": misrouted,
        "tool_calls_after_first_misrouted_replan": int(
            trace.metadata.get("deepresearch.tool_calls_after_first_misrouted_replan", 0) or 0
        ),
        "elapsed_after_first_misrouted_replan_ms": int(
            trace.metadata.get("deepresearch.elapsed_after_first_misrouted_replan_ms", 0) or 0
        ),
        "tool_call_amplification_factor": trace.metadata.get("deepresearch.tool_call_amplification_factor"),
        "repeated_failure_signatures": trace.metadata.get(
            "deepresearch.repeated_verification_failure_signatures", []
        ),
    }
    return _check(
        "deepresearch.verification_recovery_routing",
        misrouted == 0,
        "Report-repairable verification failures did not trigger full research replans."
        if misrouted == 0
        else "A report-repairable verification failure triggered a full research replan.",
        evidence=evidence,
    )


def _deepresearch_tool_failure_free_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.tool_failure_free",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    total = int(trace.metadata.get("deepresearch.tool_call_count", 0) or 0)
    if total == 0:
        return _check(
            "deepresearch.tool_failure_free",
            None,
            "Trace contains no DeepResearch tool calls.",
            severity=CheckSeverity.ADVISORY,
        )
    failed = int(trace.metadata.get("deepresearch.failed_tool_call_count", 0) or 0)
    failure_rate = float(trace.metadata.get("deepresearch.tool_failure_rate", 0.0) or 0.0)
    return _check(
        "deepresearch.tool_failure_free",
        failed == 0,
        "Every DeepResearch tool call completed successfully."
        if failed == 0
        else "At least one DeepResearch tool call failed, was cancelled, or was denied.",
        severity=CheckSeverity.ADVISORY,
        evidence={
            "tool_call_count": total,
            "failed_tool_call_count": failed,
            "tool_failure_rate": failure_rate,
        },
    )


def _deepresearch_tool_recovery_check(trace: TraceEnvelope) -> EvaluationCheck:
    if trace.source.kind != "deepresearch_agent_harness":
        return _check(
            "deepresearch.tool_failures_recovered",
            None,
            "Trace is not a DeepResearch Agent Harness run.",
            severity=CheckSeverity.ADVISORY,
        )
    failed = int(trace.metadata.get("deepresearch.failed_tool_call_count", 0) or 0)
    if failed == 0:
        return _check(
            "deepresearch.tool_failures_recovered",
            None,
            "Trace contains no failed DeepResearch tool calls to recover.",
            severity=CheckSeverity.ADVISORY,
        )
    recovered = int(trace.metadata.get("deepresearch.recovered_tool_failure_count", 0) or 0)
    unrecovered = int(trace.metadata.get("deepresearch.unrecovered_tool_failure_count", 0) or 0)
    return _check(
        "deepresearch.tool_failures_recovered",
        unrecovered == 0,
        "Every failed DeepResearch tool call was followed by a successful call for the same task."
        if unrecovered == 0
        else "At least one failed DeepResearch tool call had no later successful call for the same task.",
        evidence={
            "failed_tool_call_count": failed,
            "recovered_tool_failure_count": recovered,
            "unrecovered_tool_failure_count": unrecovered,
        },
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
    _deepresearch_contract_check,
    _deepresearch_citation_check,
    _deepresearch_checkpoint_check,
    _deepresearch_source_isolation_check,
    _deepresearch_tool_accounting_check,
    _deepresearch_duplicate_tool_check,
    _deepresearch_verification_recovery_routing_check,
    _deepresearch_tool_failure_free_check,
    _deepresearch_tool_recovery_check,
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
