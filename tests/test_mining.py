from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agenttracelab.mining import (
    HardCaseMiningPolicy,
    mine_gepa_hard_cases,
    mine_gepa_hard_cases_file,
)
from agenttracelab.stability import (
    GepaStabilityPolicy,
    GepaStabilityReport,
    GepaStabilitySample,
)


def _sample(
    *,
    task_id: str,
    trial: int,
    baseline_score: float,
    candidate_score: float,
    baseline_passed: bool,
    candidate_passed: bool,
    baseline_safety: float,
    candidate_safety: float,
    baseline_checks: tuple[str, ...] = (),
    candidate_checks: tuple[str, ...] = (),
) -> GepaStabilitySample:
    return GepaStabilitySample(
        task_id=task_id,
        trial=trial,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        score_delta=round(candidate_score - baseline_score, 4),
        baseline_passed=baseline_passed,
        candidate_passed=candidate_passed,
        baseline_safety_score=baseline_safety,
        candidate_safety_score=candidate_safety,
        baseline_failed_check_ids=baseline_checks,
        candidate_failed_check_ids=candidate_checks,
        baseline_trace_ids=(f"baseline-{task_id}-{trial}",),
        candidate_trace_ids=(f"candidate-{task_id}-{trial}",),
    )


def _stability_report() -> GepaStabilityReport:
    samples = (
        _sample(
            task_id="safety-one",
            trial=1,
            baseline_score=1.0,
            candidate_score=0.0,
            baseline_passed=True,
            candidate_passed=False,
            baseline_safety=1.0,
            candidate_safety=0.0,
            candidate_checks=("safety.approval_before_commit",),
        ),
        _sample(
            task_id="safety-two",
            trial=1,
            baseline_score=1.0,
            candidate_score=0.0,
            baseline_passed=True,
            candidate_passed=False,
            baseline_safety=1.0,
            candidate_safety=0.0,
            candidate_checks=("safety.approval_before_commit",),
        ),
        _sample(
            task_id="tool-result",
            trial=1,
            baseline_score=0.8,
            candidate_score=0.7,
            baseline_passed=True,
            candidate_passed=False,
            baseline_safety=1.0,
            candidate_safety=1.0,
            candidate_checks=("agent.tool_result_recorded",),
        ),
        _sample(
            task_id="resolved-validation",
            trial=1,
            baseline_score=0.0,
            candidate_score=1.0,
            baseline_passed=False,
            candidate_passed=True,
            baseline_safety=0.0,
            candidate_safety=1.0,
            baseline_checks=("effect.post_commit_validation",),
        ),
    )
    trace_ids = tuple(
        trace_id
        for sample in samples
        for trace_id in (*sample.baseline_trace_ids, *sample.candidate_trace_ids)
    )
    return GepaStabilityReport(
        stability_id="mining-source",
        stability_version="1.0.0",
        experiment_id="experiment",
        experiment_version="1.0.0",
        created_at=datetime.now(UTC),
        policy=GepaStabilityPolicy(trials_per_task=2),
        task_count=4,
        trials_per_task=2,
        paired_sample_count=4,
        baseline_mean_score=0.7,
        candidate_mean_score=0.425,
        paired_mean_delta=-0.275,
        candidate_successes=1,
        candidate_success_rate=0.25,
        candidate_success_lower_bound=0.0456,
        safety_regressions=2,
        trace_ids=trace_ids,
        fresh_trace_passed=True,
        recommendation="hold",
        reasons=("candidate is unstable",),
        samples=samples,
        limitations=("test fixture",),
    )


def test_hard_case_mining_prioritizes_and_diversifies_signatures() -> None:
    report = mine_gepa_hard_cases(
        _stability_report(),
        dataset_id="challenge-set",
        dataset_version="1.0.0",
        policy=HardCaseMiningPolicy(max_cases=3, max_per_signature=1),
    )

    assert report.eligible_case_count == 4
    assert report.selected_case_count == 3
    assert report.distinct_signature_count == 3
    assert report.selected_signature_count == 3
    assert report.signature_coverage == 1.0
    assert {case.case_kind for case in report.cases} == {
        "safety_regression",
        "candidate_failure",
        "resolved_baseline_failure",
    }
    assert report.cases[0].severity == "critical"
    safety_cases = [case for case in report.cases if case.case_kind == "safety_regression"]
    assert len(safety_cases) == 1


def test_hard_case_ids_are_deterministic_and_output_is_content_free() -> None:
    first = mine_gepa_hard_cases(
        _stability_report(),
        dataset_id="challenge-set",
        dataset_version="1.0.0",
    )
    second = mine_gepa_hard_cases(
        _stability_report(),
        dataset_id="challenge-set",
        dataset_version="1.0.0",
    )

    assert [case.case_id for case in first.cases] == [case.case_id for case in second.cases]
    rendered = first.model_dump_json()
    assert "system_prompt" not in rendered
    assert "tool_payload" not in rendered
    assert "safety.approval_before_commit" in rendered


def test_hard_case_file_loader_reads_versioned_stability_report(tmp_path: Path) -> None:
    path = tmp_path / "stability.json"
    path.write_text(_stability_report().model_dump_json(indent=2), encoding="utf-8")

    report = mine_gepa_hard_cases_file(
        path,
        dataset_id="from-file",
        dataset_version="2.0.0",
        policy=HardCaseMiningPolicy(max_cases=2),
    )

    assert report.dataset_id == "from-file"
    assert report.dataset_version == "2.0.0"
    assert report.selected_case_count == 2


def test_hard_case_mining_can_disable_all_case_kinds() -> None:
    report = mine_gepa_hard_cases(
        _stability_report(),
        dataset_id="empty-policy",
        dataset_version="1.0.0",
        policy=HardCaseMiningPolicy(
            include_candidate_failures=False,
            include_score_regressions=False,
            include_resolved_baseline_failures=False,
        ),
    )

    assert report.eligible_case_count == 0
    assert report.selected_case_count == 0
    assert report.signature_coverage == 0.0
