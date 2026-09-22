from __future__ import annotations

from datetime import UTC, datetime
from xml.etree.ElementTree import fromstring

from agenttracelab.batch_comparison import BatchComparisonReport, ScenarioComparison
from agenttracelab.reporting import render_comparison_html, render_comparison_junit


def _report(*, hold: bool = False, unsafe_id: bool = False) -> BatchComparisonReport:
    scenario_id = "<script>alert(1)</script>" if unsafe_id else "safe-scenario"
    scenario = ScenarioComparison(
        scenario_id=scenario_id,
        baseline_trace_id="baseline-trace",
        candidate_trace_id="candidate-trace",
        baseline_passed=True,
        candidate_passed=not hold,
        baseline_score=100.0,
        candidate_score=50.0 if hold else 100.0,
        score_delta=-50.0 if hold else 0.0,
        check_regressions=("agent.decision_evidence",) if hold else (),
        check_improvements=(),
    )
    return BatchComparisonReport(
        baseline_batch_id="baseline",
        baseline_version="1.0.0",
        candidate_batch_id="candidate",
        candidate_version="2.0.0",
        created_at=datetime.now(UTC),
        paired_scenario_count=1,
        baseline_only_scenarios=("removed-scenario",) if hold else (),
        candidate_only_scenarios=("new-scenario",) if hold else (),
        run_regressions=(scenario_id,) if hold else (),
        run_improvements=(),
        check_regression_count=1 if hold else 0,
        check_improvement_count=0,
        mean_paired_score_delta=scenario.score_delta,
        evidence_modes_match=True,
        recommendation="hold" if hold else "promote",
        reasons=("candidate regressed",) if hold else (),
        comparison_notice="Deterministic paired comparison only.",
        scenarios=(scenario,),
    )


def test_html_report_is_standalone_and_escapes_dynamic_values() -> None:
    rendered = render_comparison_html(_report(unsafe_id=True))

    assert "<!doctype html>" in rendered
    assert "Content-Security-Policy" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered
    assert "PROMOTE" in rendered
    assert "mean process reward delta" in rendered


def test_junit_report_encodes_regressions_coverage_and_gate() -> None:
    root = fromstring(render_comparison_junit(_report(hold=True)))

    assert root.tag == "testsuite"
    assert root.attrib == {
        "name": "AgentTraceLab paired comparison",
        "tests": "4",
        "failures": "3",
        "errors": "0",
        "skipped": "1",
    }
    failures = root.findall(".//failure")
    assert {failure.attrib["type"] for failure in failures} == {
        "deterministic-regression",
        "missing-baseline-scenario",
        "promotion-hold",
    }


def test_junit_promote_report_has_no_failures() -> None:
    root = fromstring(render_comparison_junit(_report()))

    assert root.attrib["failures"] == "0"
    assert root.find(".//failure") is None
    assert "process_reward_delta=" in root.find(".//testcase/system-out").text
