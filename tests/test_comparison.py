from __future__ import annotations

from agenttracelab.adapters import adapt_wasmhatch_journal
from agenttracelab.comparison import compare_evaluations
from agenttracelab.evaluation import evaluate_trace


def test_candidate_improvements_can_be_promoted(complete_journal: dict, incomplete_journal: dict) -> None:
    baseline = evaluate_trace(adapt_wasmhatch_journal(incomplete_journal))
    candidate = evaluate_trace(adapt_wasmhatch_journal(complete_journal))

    report = compare_evaluations(baseline, candidate)

    assert report.recommendation == "promote"
    assert report.score_delta > 0
    assert "agent.decision_evidence" in report.improvements
    assert report.regressions == ()


def test_required_regression_blocks_promotion(complete_journal: dict, incomplete_journal: dict) -> None:
    baseline = evaluate_trace(adapt_wasmhatch_journal(complete_journal))
    candidate = evaluate_trace(adapt_wasmhatch_journal(incomplete_journal))

    report = compare_evaluations(baseline, candidate)

    assert report.recommendation == "hold"
    assert "agent.decision_evidence" in report.regressions
