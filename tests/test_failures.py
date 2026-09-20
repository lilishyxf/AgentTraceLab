from __future__ import annotations

from agenttracelab.evaluation import evaluate_trace
from agenttracelab.failures import FailureCategory, classify_evaluation
from agenttracelab.models import TraceEnvelope


def test_classifies_failed_checks_by_engineering_category(incomplete_journal: dict) -> None:
    from agenttracelab.adapters import adapt_wasmhatch_journal

    trace: TraceEnvelope = adapt_wasmhatch_journal(incomplete_journal)

    findings = classify_evaluation(evaluate_trace(trace))

    categories = {finding.category for finding in findings}
    assert categories == {
        FailureCategory.DECISION_EVIDENCE,
        FailureCategory.EFFECT_VALIDATION,
        FailureCategory.LIFECYCLE,
        FailureCategory.SAFETY_AUTHORIZATION,
        FailureCategory.TOOL_LINKAGE,
        FailureCategory.TRACE_CONTRACT,
    }
