from __future__ import annotations

import copy

from agenttracelab.adapters import adapt_wasmhatch_journal
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.models import CheckStatus


def check_map(report):
    return {check.check_id: check for check in report.checks}


def test_complete_agent_trace_passes_required_gates(complete_journal: dict) -> None:
    report = evaluate_trace(adapt_wasmhatch_journal(complete_journal))

    assert report.passed is True
    assert report.score == 100.0
    assert report.failure_categories == ()


def test_missing_agent_evidence_and_approval_are_reported(incomplete_journal: dict) -> None:
    report = evaluate_trace(adapt_wasmhatch_journal(incomplete_journal))
    checks = check_map(report)

    assert report.passed is False
    assert checks["trace.terminal_disposition"].status == CheckStatus.FAIL
    assert checks["safety.approval_before_commit"].status == CheckStatus.FAIL
    assert checks["effect.post_commit_validation"].status == CheckStatus.FAIL
    assert checks["agent.decision_evidence"].status == CheckStatus.FAIL
    assert checks["agent.tool_decision_link"].status == CheckStatus.FAIL


def test_credential_shaped_values_fail_privacy_gate(complete_journal: dict) -> None:
    payload = copy.deepcopy(complete_journal)
    payload["events"][1]["detail"] = "Authorization: Bearer abcdefghijklmnop"

    report = evaluate_trace(adapt_wasmhatch_journal(payload))

    assert check_map(report)["privacy.no_raw_secret"].status == CheckStatus.FAIL
    assert "privacy.no_raw_secret" in report.failure_categories


def test_detects_wasmhatch_metric_drift(complete_journal: dict) -> None:
    complete_journal["metrics"]["commits"] = 99
    trace = adapt_wasmhatch_journal(complete_journal)

    report = evaluate_trace(trace)

    assert report.passed is False
    assert "wasmhatch.metrics_consistent" in report.failure_categories
