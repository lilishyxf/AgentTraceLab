from __future__ import annotations

from collections import defaultdict
from enum import StrEnum

from agenttracelab.models import CheckStatus, EvaluationReport, FrozenModel


class FailureCategory(StrEnum):
    TRANSPORT = "transport"
    TRACE_CONTRACT = "trace_contract"
    LIFECYCLE = "lifecycle"
    TOOL_EXECUTION = "tool_execution"
    SAFETY_AUTHORIZATION = "safety_authorization"
    EFFECT_VALIDATION = "effect_validation"
    DECISION_EVIDENCE = "decision_evidence"
    TOOL_LINKAGE = "tool_linkage"
    PRIVACY = "privacy"
    BUDGET = "budget"
    EXPECTATION = "expectation"
    JUDGE_PROVIDER = "judge_provider"
    JUDGE_PROTOCOL = "judge_protocol"
    OTHER = "other"


class FailureFinding(FrozenModel):
    category: FailureCategory
    check_ids: tuple[str, ...] = ()
    summary: str


CHECK_CATEGORIES: dict[str, FailureCategory] = {
    "trace.contiguous_sequence": FailureCategory.TRACE_CONTRACT,
    "wasmhatch.metrics_consistent": FailureCategory.TRACE_CONTRACT,
    "deepresearch.required_contracts_pass": FailureCategory.TRACE_CONTRACT,
    "deepresearch.citations_valid": FailureCategory.EFFECT_VALIDATION,
    "deepresearch.checkpoints_intact": FailureCategory.TRACE_CONTRACT,
    "deepresearch.source_isolation": FailureCategory.SAFETY_AUTHORIZATION,
    "deepresearch.tool_accounting_consistent": FailureCategory.TRACE_CONTRACT,
    "deepresearch.tool_failure_free": FailureCategory.TOOL_EXECUTION,
    "deepresearch.tool_failures_recovered": FailureCategory.TOOL_EXECUTION,
    "deepresearch.verification_recovery_routing": FailureCategory.BUDGET,
    "deepresearch.claim_citation_gate": FailureCategory.EFFECT_VALIDATION,
    "deepresearch.claim_semantic_gate": FailureCategory.JUDGE_PROTOCOL,
    "trace.terminal_disposition": FailureCategory.LIFECYCLE,
    "tool.result_recorded": FailureCategory.TOOL_EXECUTION,
    "safety.approval_before_commit": FailureCategory.SAFETY_AUTHORIZATION,
    "effect.post_commit_validation": FailureCategory.EFFECT_VALIDATION,
    "agent.decision_evidence": FailureCategory.DECISION_EVIDENCE,
    "agent.tool_decision_link": FailureCategory.TOOL_LINKAGE,
    "privacy.no_raw_secret": FailureCategory.PRIVACY,
    "trace.event_budget": FailureCategory.BUDGET,
}


def classify_evaluation(report: EvaluationReport) -> tuple[FailureFinding, ...]:
    grouped: dict[FailureCategory, list[str]] = defaultdict(list)
    for check in report.checks:
        if check.status == CheckStatus.FAIL:
            grouped[CHECK_CATEGORIES.get(check.check_id, FailureCategory.OTHER)].append(check.check_id)
    return tuple(
        FailureFinding(
            category=category,
            check_ids=tuple(check_ids),
            summary=f"{len(check_ids)} failed deterministic check(s) in {category.value}.",
        )
        for category, check_ids in sorted(grouped.items(), key=lambda item: item[0].value)
    )


def runtime_failure(category: FailureCategory, summary: str) -> FailureFinding:
    return FailureFinding(category=category, summary=summary)
