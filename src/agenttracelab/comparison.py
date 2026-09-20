from __future__ import annotations

from agenttracelab.models import CheckStatus, ComparisonReport, EvaluationReport


def compare_evaluations(baseline: EvaluationReport, candidate: EvaluationReport) -> ComparisonReport:
    baseline_checks = {check.check_id: check for check in baseline.checks}
    candidate_checks = {check.check_id: check for check in candidate.checks}
    shared_ids = sorted(baseline_checks.keys() & candidate_checks.keys())
    regressions = tuple(
        check_id
        for check_id in shared_ids
        if baseline_checks[check_id].status == CheckStatus.PASS
        and candidate_checks[check_id].status == CheckStatus.FAIL
    )
    improvements = tuple(
        check_id
        for check_id in shared_ids
        if baseline_checks[check_id].status == CheckStatus.FAIL
        and candidate_checks[check_id].status == CheckStatus.PASS
    )
    delta = round(candidate.score - baseline.score, 2)
    recommendation = "promote" if not regressions and candidate.passed and delta >= 0 else "hold"
    return ComparisonReport(
        baseline_trace_id=baseline.trace_id,
        candidate_trace_id=candidate.trace_id,
        baseline_score=baseline.score,
        candidate_score=candidate.score,
        score_delta=delta,
        regressions=regressions,
        improvements=improvements,
        recommendation=recommendation,
    )
