from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agenttracelab.curation import (
    HardCaseResolution,
    HardCaseResolutionSet,
    curate_hard_cases,
    curate_hard_cases_files,
)
from agenttracelab.mining import (
    HardCaseEntry,
    HardCaseMiningPolicy,
    HardCaseMiningReport,
)


def _entry(
    suffix: str,
    *,
    kind: str,
    task_id: str,
    trial: int,
) -> HardCaseEntry:
    return HardCaseEntry(
        case_id=f"hardcase_{suffix * 24}",
        case_kind=kind,
        severity="medium" if kind == "resolved_baseline_failure" else "high",
        priority_score=50.0,
        task_id=task_id,
        trial=trial,
        failure_signature=("effect.post_commit_validation",),
        baseline_score=0.0 if kind == "resolved_baseline_failure" else 1.0,
        candidate_score=1.0 if kind == "resolved_baseline_failure" else 0.0,
        score_delta=1.0 if kind == "resolved_baseline_failure" else -1.0,
        baseline_trace_ids=(f"baseline-{suffix}",),
        candidate_trace_ids=(f"candidate-{suffix}",),
        reasons=("review this case",),
    )


def _queue() -> HardCaseMiningReport:
    cases = (
        _entry("a", kind="resolved_baseline_failure", task_id="fixed-task", trial=1),
        _entry("b", kind="candidate_failure", task_id="repeated-task", trial=1),
        _entry("c", kind="candidate_failure", task_id="repeated-task", trial=2),
        _entry("d", kind="candidate_failure", task_id="other-task", trial=1),
        _entry("e", kind="candidate_failure", task_id="rejected-task", trial=1),
    )
    return HardCaseMiningReport(
        dataset_id="hard-cases",
        dataset_version="1.0.0",
        created_at=datetime.now(UTC),
        source_stability_id="stability",
        source_stability_version="1.0.0",
        policy=HardCaseMiningPolicy(),
        source_sample_count=5,
        eligible_case_count=5,
        selected_case_count=5,
        distinct_signature_count=2,
        selected_signature_count=2,
        signature_coverage=1.0,
        cases=cases,
        limitations=("test fixture",),
    )


def _resolutions(queue: HardCaseMiningReport) -> HardCaseResolutionSet:
    values = []
    for case in queue.cases:
        if case.case_kind == "resolved_baseline_failure":
            decision = "regression_guard"
            rationale = "confirmed_fix"
        elif case.task_id == "rejected-task":
            decision = "reject"
            rationale = "insufficient_evidence"
        else:
            decision = "training_candidate"
            rationale = "confirmed_failure"
        values.append(
            HardCaseResolution(
                case_id=case.case_id,
                decision=decision,
                rationale_code=rationale,
            )
        )
    return HardCaseResolutionSet(
        schema_version="agenttracelab.hard-case-resolutions.v1",
        adjudication_id="curated-set",
        version="1.0.0",
        source_dataset_id=queue.dataset_id,
        source_dataset_version=queue.dataset_version,
        review_mode="human",
        reviewer_id="reviewer-one",
        reviewed_at=datetime.now(UTC),
        split_seed="fixed-seed",
        resolutions=tuple(values),
    )


def test_curation_validates_decisions_and_groups_task_splits() -> None:
    queue = _queue()
    curated = curate_hard_cases(queue, _resolutions(queue))

    assert curated.included_case_count == 4
    assert curated.decision_counts == {
        "regression_guard": 1,
        "reject": 1,
        "training_candidate": 3,
    }
    assert curated.rejected_case_ids == ("hardcase_" + "e" * 24,)
    regression = [case for case in curated.cases if case.decision == "regression_guard"]
    assert len(regression) == 1
    assert regression[0].split == "regression"
    repeated = [case for case in curated.cases if case.source.task_id == "repeated-task"]
    assert len(repeated) == 2
    assert len({case.split for case in repeated}) == 1


def test_curation_rejects_unknown_duplicate_and_missing_resolutions() -> None:
    queue = _queue()
    base = _resolutions(queue)
    unknown = HardCaseResolution(
        case_id="hardcase_" + "f" * 24,
        decision="reject",
        rationale_code="out_of_scope",
    )
    with pytest.raises(ValueError, match="unknown cases"):
        curate_hard_cases(
            queue,
            base.model_copy(update={"resolutions": (*base.resolutions, unknown)}),
        )
    with pytest.raises(ValueError, match="duplicate"):
        curate_hard_cases(
            queue,
            base.model_copy(update={"resolutions": (*base.resolutions, base.resolutions[0])}),
        )
    with pytest.raises(ValueError, match="missing cases"):
        curate_hard_cases(
            queue,
            base.model_copy(update={"resolutions": base.resolutions[:-1]}),
        )


def test_curation_enforces_decision_semantics() -> None:
    queue = _queue()
    base = _resolutions(queue)
    bad = base.resolutions[1].model_copy(
        update={"decision": "regression_guard", "rationale_code": "confirmed_fix"}
    )
    with pytest.raises(ValueError, match="resolved baseline failure"):
        curate_hard_cases(
            queue,
            base.model_copy(update={"resolutions": (base.resolutions[0], bad, *base.resolutions[2:])}),
        )
    bad_rationale = base.resolutions[1].model_copy(update={"rationale_code": "confirmed_fix"})
    with pytest.raises(ValueError, match="confirmed_failure"):
        curate_hard_cases(
            queue,
            base.model_copy(
                update={
                    "resolutions": (
                        base.resolutions[0],
                        bad_rationale,
                        *base.resolutions[2:],
                    )
                }
            ),
        )


def test_file_curation_preserves_explicit_synthetic_review_mode(tmp_path: Path) -> None:
    queue = _queue()
    resolutions = _resolutions(queue).model_copy(
        update={"review_mode": "generated-synthetic", "reviewer_id": "smoke-generator"}
    )
    queue_path = tmp_path / "queue.json"
    resolution_path = tmp_path / "resolutions.json"
    queue_path.write_text(queue.model_dump_json(indent=2), encoding="utf-8")
    resolution_path.write_text(resolutions.model_dump_json(indent=2), encoding="utf-8")

    curated = curate_hard_cases_files(queue_path, resolution_path)

    assert curated.review_mode == "generated-synthetic"
    assert curated.reviewer_id == "smoke-generator"
    assert "system_prompt" not in curated.model_dump_json()
