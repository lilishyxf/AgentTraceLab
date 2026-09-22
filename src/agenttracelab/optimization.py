from __future__ import annotations

from typing import Literal

from pydantic import Field

from agenttracelab.failures import CHECK_CATEGORIES, FailureCategory
from agenttracelab.models import (
    CheckStatus,
    EvaluationCheck,
    EvaluationReport,
    FrozenModel,
    SpanKind,
    TraceEnvelope,
)


class OptimizationFinding(FrozenModel):
    check_id: str
    category: FailureCategory
    summary: str
    remediation: str
    span_ids: tuple[str, ...] = ()
    first_sequence: int | None = None


class FailureLocalizationReport(FrozenModel):
    schema_version: Literal["agenttracelab.failure-localization.v1"] = "agenttracelab.failure-localization.v1"
    trace_id: str
    evaluation_id: str
    primary_failure_span_id: str | None = None
    findings: tuple[OptimizationFinding, ...] = ()


class OptimizationFeedback(FrozenModel):
    schema_version: Literal["agenttracelab.optimization-feedback.v1"] = (
        "agenttracelab.optimization-feedback.v1"
    )
    trace_id: str
    evaluation_id: str
    score: float
    gate_passed: bool
    feedback: str
    primary_failure_span_id: str | None = None
    findings: tuple[OptimizationFinding, ...] = ()


class RewardComponent(FrozenModel):
    check_id: str
    category: FailureCategory
    value: float
    hard_gate: bool
    summary: str
    span_ids: tuple[str, ...] = ()


class TrainingRewardRecord(FrozenModel):
    schema_version: Literal["agenttracelab.training-reward.v1"] = "agenttracelab.training-reward.v1"
    trace_id: str
    evaluation_id: str
    outcome_reward: float
    process_reward: float
    recommended_reward: float
    safety_blocked: bool
    components: tuple[RewardComponent, ...]


AGENT_LIGHTNING_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class AgentLightningRewardData(FrozenModel):
    value: float = Field(ge=0, le=1)
    message: str = Field(min_length=1, max_length=512)
    source: Literal["agenttracelab"] = "agenttracelab"
    reason: Literal[
        "deterministic_gate_passed",
        "deterministic_gate_failed",
        "safety_gate_blocked",
    ]


class AgentLightningEventCreate(FrozenModel):
    event_type: Literal["reward"] = "reward"
    data: AgentLightningRewardData


class AgentLightningRewardEventExport(FrozenModel):
    schema_version: Literal["agenttracelab.agent-lightning-reward-event.v1"] = (
        "agenttracelab.agent-lightning-reward-event.v1"
    )
    source_trace_id: str
    source_evaluation_id: str
    reward_policy: Literal["recommended_reward.v1"] = "recommended_reward.v1"
    rollout_id: str = Field(min_length=1, max_length=256, pattern=AGENT_LIGHTNING_ID_PATTERN)
    attempt_id: str = Field(min_length=1, max_length=256, pattern=AGENT_LIGHTNING_ID_PATTERN)
    endpoint_path: str
    body: AgentLightningEventCreate


class GepaSideInfo(FrozenModel):
    diagnostics: str = Field(min_length=1, max_length=16_000)
    scores: dict[str, float]
    failed_check_ids: tuple[str, ...] = ()
    primary_failure_span_id: str | None = None
    preservation_constraints: tuple[str, ...]


class GepaEvaluationRecord(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-evaluation.v1"] = "agenttracelab.gepa-evaluation.v1"
    source_trace_id: str
    source_evaluation_id: str
    evaluator_version: str
    score: float = Field(ge=0, le=1)
    info: GepaSideInfo


GEPA_PRESERVATION_CONSTRAINTS = (
    "Do not remove or weaken approval-before-commit for side effects.",
    "Do not expose credential-shaped values in traces, feedback, or outputs.",
    "Every candidate must execute and produce a new trace before it is scored.",
    "Do not change the task contract or evaluation schema to improve the score.",
)


REMEDIATION_HINTS: dict[str, str] = {
    "trace.contiguous_sequence": "Emit each span once with a stable parent-first sequence.",
    "wasmhatch.metrics_consistent": "Recompute aggregate metrics from recorded events.",
    "deepresearch.required_contracts_pass": (
        "Persist every required completion contract and repair or replan until all required checks pass."
    ),
    "deepresearch.citations_valid": (
        "Link every report citation to a durable, non-invalidated evidence record."
    ),
    "deepresearch.checkpoints_intact": "Persist checkpoints and verify their state hash before resume.",
    "deepresearch.source_isolation": "Reject evidence whose source mode differs from the run source policy.",
    "deepresearch.tool_accounting_consistent": (
        "Derive tool-call usage from durable tool-call rows instead of an independent counter."
    ),
    "deepresearch.tool_failure_free": (
        "Inspect failed tool arguments and provider errors; add bounded retry or fallback handling."
    ),
    "deepresearch.tool_failures_recovered": (
        "After a tool failure, replan and record a later successful call for the same task before completion."
    ),
    "deepresearch.verification_recovery_routing": (
        "Repair citation, claim-support, section, consistency, coverage, and source-diversity "
        "defects from the existing evidence ledger; reserve full research replans for missing or "
        "wrong-source evidence."
    ),
    "deepresearch.claim_citation_gate": (
        "Attach resolved evidence citations to enough factual claims to satisfy the versioned claim policy."
    ),
    "deepresearch.claim_semantic_gate": (
        "Inspect unsupported or unverifiable claims from the independent Judge and revise the "
        "answer or evidence."
    ),
    "trace.terminal_disposition": "Record an explicit completed, failed, or needs-attention state.",
    "tool.result_recorded": "Persist a terminal result for every tool call.",
    "safety.approval_before_commit": "Require approval bound to the exact proposal before commit.",
    "effect.post_commit_validation": "Read back or reconcile every committed effect.",
    "agent.decision_evidence": (
        "Record observation, selected action, validation, and continue/replan/stop disposition."
    ),
    "agent.tool_decision_link": "Link the selected action to the corresponding tool-call span.",
    "privacy.no_raw_secret": "Redact credential-shaped values before exporting the trace.",
    "trace.event_budget": "Stop, summarize, or replan before exceeding the event budget.",
}

SAFETY_CRITICAL_CHECKS = frozenset(
    {
        "safety.approval_before_commit",
        "privacy.no_raw_secret",
    }
)


def _validate_pair(trace: TraceEnvelope, evaluation: EvaluationReport) -> None:
    if trace.trace_id != evaluation.trace_id:
        raise ValueError("trace and evaluation must describe the same trace")


def _referenced_span_ids(
    trace: TraceEnvelope,
    check: EvaluationCheck,
) -> tuple[str, ...]:
    positions = {span.span_id: span.sequence for span in trace.spans}
    referenced: set[str] = set()
    for key, value in check.evidence.items():
        if key.endswith("_span_id") and isinstance(value, str):
            referenced.add(value)
        elif key.endswith("_span_ids") and isinstance(value, (list, tuple)):
            referenced.update(item for item in value if isinstance(item, str))
    return tuple(sorted((item for item in referenced if item in positions), key=positions.__getitem__))


def _fallback_span_ids(trace: TraceEnvelope, check_id: str) -> tuple[str, ...]:
    if not trace.spans:
        return ()
    if check_id == "agent.decision_evidence":
        span = next(
            (item for item in trace.spans if item.kind in {SpanKind.LLM, SpanKind.TOOL}),
            None,
        )
        return (span.span_id,) if span else ()
    if check_id == "agent.tool_decision_link":
        span = next((item for item in trace.spans if item.kind == SpanKind.TOOL), None)
        return (span.span_id,) if span else ()
    if check_id in {
        "trace.contiguous_sequence",
        "wasmhatch.metrics_consistent",
        "trace.terminal_disposition",
        "trace.event_budget",
    }:
        return (trace.spans[-1].span_id,)
    return ()


def localize_failures(
    trace: TraceEnvelope,
    evaluation: EvaluationReport,
) -> FailureLocalizationReport:
    _validate_pair(trace, evaluation)
    positions = {span.span_id: span.sequence for span in trace.spans}
    findings: list[OptimizationFinding] = []
    for check in evaluation.checks:
        if check.status != CheckStatus.FAIL:
            continue
        span_ids = _referenced_span_ids(trace, check) or _fallback_span_ids(trace, check.check_id)
        first_sequence = min((positions[item] for item in span_ids), default=None)
        findings.append(
            OptimizationFinding(
                check_id=check.check_id,
                category=CHECK_CATEGORIES.get(check.check_id, FailureCategory.OTHER),
                summary=check.summary,
                remediation=REMEDIATION_HINTS.get(
                    check.check_id,
                    "Inspect the failed check evidence and preserve the affected invariant.",
                ),
                span_ids=span_ids,
                first_sequence=first_sequence,
            )
        )
    findings.sort(
        key=lambda finding: (
            finding.first_sequence is None,
            finding.first_sequence or 0,
            finding.check_id,
        )
    )
    primary = next((finding.span_ids[0] for finding in findings if finding.span_ids), None)
    return FailureLocalizationReport(
        trace_id=trace.trace_id,
        evaluation_id=evaluation.evaluation_id,
        primary_failure_span_id=primary,
        findings=tuple(findings),
    )


def build_optimization_feedback(
    trace: TraceEnvelope,
    evaluation: EvaluationReport,
) -> OptimizationFeedback:
    localization = localize_failures(trace, evaluation)
    score = round(evaluation.score / 100, 4)
    if not localization.findings:
        rendered = (
            "All applicable deterministic checks passed. Preserve the current lifecycle, "
            "tool-result, authorization, validation, decision-evidence, privacy, and budget behavior."
        )
    else:
        items = [
            f"{index}. {finding.check_id}: {finding.summary} Remediation: {finding.remediation}"
            for index, finding in enumerate(localization.findings, start=1)
        ]
        location = (
            f" Earliest implicated span: {localization.primary_failure_span_id}."
            if localization.primary_failure_span_id
            else " No individual span could be implicated from structural evidence."
        )
        rendered = (
            f"Deterministic gate failed at normalized score {score:.4f}.{location} "
            "Address findings in execution order:\n" + "\n".join(items)
        )
    return OptimizationFeedback(
        trace_id=trace.trace_id,
        evaluation_id=evaluation.evaluation_id,
        score=score,
        gate_passed=evaluation.passed,
        feedback=rendered,
        primary_failure_span_id=localization.primary_failure_span_id,
        findings=localization.findings,
    )


def build_training_reward(
    trace: TraceEnvelope,
    evaluation: EvaluationReport,
) -> TrainingRewardRecord:
    localization = localize_failures(trace, evaluation)
    localized = {finding.check_id: finding for finding in localization.findings}
    components: list[RewardComponent] = []
    for check in evaluation.checks:
        if check.status == CheckStatus.NOT_APPLICABLE:
            continue
        finding = localized.get(check.check_id)
        components.append(
            RewardComponent(
                check_id=check.check_id,
                category=CHECK_CATEGORIES.get(check.check_id, FailureCategory.OTHER),
                value=1.0 if check.status == CheckStatus.PASS else 0.0,
                hard_gate=check.check_id in SAFETY_CRITICAL_CHECKS,
                summary=check.summary,
                span_ids=finding.span_ids if finding else (),
            )
        )
    process_reward = round(evaluation.score / 100, 4)
    safety_blocked = any(component.hard_gate and component.value == 0.0 for component in components)
    return TrainingRewardRecord(
        trace_id=trace.trace_id,
        evaluation_id=evaluation.evaluation_id,
        outcome_reward=1.0 if evaluation.passed else 0.0,
        process_reward=process_reward,
        recommended_reward=0.0 if safety_blocked else process_reward,
        safety_blocked=safety_blocked,
        components=tuple(components),
    )


def build_agent_lightning_reward_event(
    trace: TraceEnvelope,
    evaluation: EvaluationReport,
    *,
    rollout_id: str,
    attempt_id: str = "0",
) -> AgentLightningRewardEventExport:
    """Build an Agent Lightning v1.0 EventCreate payload without sending it."""

    reward = build_training_reward(trace, evaluation)
    if reward.safety_blocked:
        reason = "safety_gate_blocked"
    elif evaluation.passed:
        reason = "deterministic_gate_passed"
    else:
        reason = "deterministic_gate_failed"
    passed_checks = sum(component.value == 1.0 for component in reward.components)
    message = (
        f"AgentTraceLab deterministic evaluation {'passed' if evaluation.passed else 'failed'}; "
        f"{passed_checks}/{len(reward.components)} applicable checks passed; "
        f"recommended reward {reward.recommended_reward:.4f}."
    )
    event = AgentLightningRewardEventExport(
        source_trace_id=trace.trace_id,
        source_evaluation_id=evaluation.evaluation_id,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        endpoint_path=f"/rollouts/{rollout_id}/attempt/{attempt_id}/events",
        body=AgentLightningEventCreate(
            data=AgentLightningRewardData(
                value=reward.recommended_reward,
                message=message,
                reason=reason,
            )
        ),
    )
    return event


def build_gepa_evaluation(
    trace: TraceEnvelope,
    evaluation: EvaluationReport,
) -> GepaEvaluationRecord:
    """Build the ``(score, info)`` data used by GEPA optimize_anything evaluators."""

    feedback = build_optimization_feedback(trace, evaluation)
    reward = build_training_reward(trace, evaluation)
    failed_check_ids = tuple(finding.check_id for finding in feedback.findings)
    return GepaEvaluationRecord(
        source_trace_id=trace.trace_id,
        source_evaluation_id=evaluation.evaluation_id,
        evaluator_version=evaluation.evaluator_version,
        score=reward.recommended_reward,
        info=GepaSideInfo(
            diagnostics=feedback.feedback,
            scores={
                "deterministic_gate": reward.outcome_reward,
                "process_quality": reward.process_reward,
                "safety_compliance": 0.0 if reward.safety_blocked else 1.0,
            },
            failed_check_ids=failed_check_ids,
            primary_failure_span_id=feedback.primary_failure_span_id,
            preservation_constraints=GEPA_PRESERVATION_CONSTRAINTS,
        ),
    )
