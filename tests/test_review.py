from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agenttracelab.api import create_app
from agenttracelab.calibration import analyze_judge_calibration
from agenttracelab.cli import _export_review_calibration
from agenttracelab.judging import (
    JUDGE_PROMPT_VERSION,
    ContentPolicy,
    JudgeDimensionName,
    JudgeDimensionScore,
    JudgeVerdict,
    LlmJudgeReport,
)
from agenttracelab.models import EvaluationReport, TraceEnvelope
from agenttracelab.review import (
    ReviewPriority,
    ReviewReason,
    ReviewResolutionInput,
    ReviewState,
)
from agenttracelab.service import AgentTraceService
from agenttracelab.storage import Database


def _service() -> AgentTraceService:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    return AgentTraceService(database)


def _judge_report(
    deterministic: EvaluationReport,
    verdict: JudgeVerdict,
    *,
    score: int,
    suffix: str = "one",
) -> LlmJudgeReport:
    dimensions = tuple(
        JudgeDimensionScore(
            name=name,
            score=score,
            rationale=f"Recorded {name.value} assessment.",
        )
        for name in JudgeDimensionName
    )
    return LlmJudgeReport(
        judge_report_id=f"judge_{suffix}",
        trace_id=deterministic.trace_id,
        deterministic_evaluation_id=deterministic.evaluation_id,
        provider="recorded",
        model="review-fixture",
        prompt_version=JUDGE_PROMPT_VERSION,
        content_policy=ContentPolicy.METADATA_ONLY,
        created_at=datetime.now(UTC),
        verdict=verdict,
        qualitative_score=score / 4 * 100,
        summary="Recorded review fixture.",
        dimensions=dimensions,
        request_attempts=1,
    )


def _resolution(verdict: JudgeVerdict = JudgeVerdict.FAIL) -> ReviewResolutionInput:
    return ReviewResolutionInput(
        reviewer_id="reviewer@example",
        verdict=verdict,
        dimensions={name: 2 for name in JudgeDimensionName},
        notes="Reviewed against the stored trace and evaluation versions.",
    )


class _StaticJudge:
    def __init__(self, verdict: JudgeVerdict, score: int) -> None:
        self.verdict = verdict
        self.score = score

    def evaluate(
        self,
        trace: TraceEnvelope,
        deterministic_report: EvaluationReport,
    ) -> LlmJudgeReport:
        assert trace.trace_id == deterministic_report.trace_id
        return _judge_report(deterministic_report, self.verdict, score=self.score)


def test_judge_conflict_automatically_creates_deduplicated_review(complete_journal: dict) -> None:
    service = _service()
    trace, _ = service.import_wasmhatch(complete_journal)

    service.judge_trace(trace.trace_id, _StaticJudge(JudgeVerdict.FAIL, 1))

    items = service.list_review_items(ReviewState.PENDING)
    assert len(items) == 1
    assert items[0].priority == ReviewPriority.HIGH
    assert items[0].reasons == (
        ReviewReason.DETERMINISTIC_JUDGE_CONFLICT,
        ReviewReason.LOW_DIMENSION_SCORE,
    )
    assert service.queue_review(trace.trace_id).review_id == items[0].review_id  # type: ignore[union-attr]


def test_deterministic_failure_judge_pass_is_critical(incomplete_journal: dict) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(incomplete_journal)
    assert deterministic.passed is False
    service.database.save_judge_report(_judge_report(deterministic, JudgeVerdict.PASS, score=4))

    item = service.queue_review(trace.trace_id)

    assert item is not None
    assert item.priority == ReviewPriority.CRITICAL
    assert item.reasons == (ReviewReason.DETERMINISTIC_JUDGE_CONFLICT,)


def test_deferred_and_forced_samples_use_explicit_reasons(complete_journal: dict) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(complete_journal)
    service.database.save_judge_report(
        _judge_report(deterministic, JudgeVerdict.REVIEW, score=3, suffix="review")
    )
    deferred = service.queue_review(trace.trace_id)
    assert deferred is not None
    assert deferred.reasons == (ReviewReason.JUDGE_DEFERRED,)

    other_service = _service()
    other_trace, other_deterministic = other_service.import_wasmhatch(complete_journal)
    other_service.database.save_judge_report(
        _judge_report(other_deterministic, JudgeVerdict.PASS, score=4, suffix="pass")
    )
    assert other_service.queue_review(other_trace.trace_id) is None
    sampled = other_service.queue_review(other_trace.trace_id, force=True)
    assert sampled is not None
    assert sampled.reasons == (ReviewReason.FORCED_SAMPLE,)


def test_resolution_exports_new_calibration_case_with_version_provenance(
    complete_journal: dict,
) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(complete_journal)
    report = _judge_report(deterministic, JudgeVerdict.FAIL, score=1)
    service.database.save_judge_report(report)
    queued = service.queue_review(trace.trace_id)
    assert queued is not None

    resolved = service.resolve_review_item(queued.review_id, _resolution())
    assert resolved.state == ReviewState.RESOLVED
    with pytest.raises(ValueError, match="already resolved"):
        service.resolve_review_item(queued.review_id, _resolution())

    manifest = service.export_resolved_reviews(dataset_id="production-disagreements", version="v1")
    case = manifest.cases[0]
    assert case.human.verdict == JudgeVerdict.FAIL
    assert case.judges[0].prompt_version == JUDGE_PROMPT_VERSION
    assert case.provenance is not None
    assert case.provenance.deterministic_evaluation_id == deterministic.evaluation_id
    assert case.provenance.judge_report_id == report.judge_report_id
    assert case.provenance.review_reasons == (
        ReviewReason.DETERMINISTIC_JUDGE_CONFLICT.value,
        ReviewReason.LOW_DIMENSION_SCORE.value,
    )
    calibration = analyze_judge_calibration(manifest)
    assert calibration.total_cases == 1
    assert calibration.passed is False  # Default policy requires at least three labelled cases.


def test_review_api_queues_lists_and_resolves(complete_journal: dict) -> None:
    app = create_app("sqlite+pysqlite:///:memory:")
    client = TestClient(app)
    imported = client.post("/v1/traces/import/wasmhatch", json=complete_journal)
    deterministic = EvaluationReport.model_validate(imported.json()["evaluation"])
    app.state.database.save_judge_report(_judge_report(deterministic, JudgeVerdict.FAIL, score=1))

    queued = client.post(
        f"/v1/reviews/queue/{deterministic.trace_id}",
        json={"force": False},
    )
    assert queued.status_code == 200
    review_id = queued.json()["review_id"]

    pending = client.get("/v1/reviews", params={"state": "pending"})
    assert pending.status_code == 200
    assert [item["review_id"] for item in pending.json()["items"]] == [review_id]

    resolution = _resolution().model_dump(mode="json")
    resolved = client.post(f"/v1/reviews/{review_id}/resolve", json=resolution)
    assert resolved.status_code == 200
    assert resolved.json()["state"] == "resolved"
    assert client.post(f"/v1/reviews/{review_id}/resolve", json=resolution).status_code == 409
    assert len(client.get("/v1/reviews", params={"state": "resolved"}).json()["items"]) == 1


def test_review_api_rejects_non_selected_pair_without_force(complete_journal: dict) -> None:
    app = create_app("sqlite+pysqlite:///:memory:")
    client = TestClient(app)
    imported = client.post("/v1/traces/import/wasmhatch", json=complete_journal)
    deterministic = EvaluationReport.model_validate(imported.json()["evaluation"])
    app.state.database.save_judge_report(_judge_report(deterministic, JudgeVerdict.PASS, score=4))

    rejected = client.post(
        f"/v1/reviews/queue/{deterministic.trace_id}",
        json={"force": False},
    )
    sampled = client.post(
        f"/v1/reviews/queue/{deterministic.trace_id}",
        json={"force": True},
    )

    assert rejected.status_code == 409
    assert sampled.status_code == 200
    assert sampled.json()["reasons"] == ["forced_sample"]


def test_review_calibration_cli_refuses_to_overwrite_existing_version(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'review.db').as_posix()}"
    service = AgentTraceService(Database(database_url))
    service.database.create_schema()
    trace, deterministic = service.import_wasmhatch(complete_journal)
    service.database.save_judge_report(_judge_report(deterministic, JudgeVerdict.FAIL, score=1))
    queued = service.queue_review(trace.trace_id)
    assert queued is not None
    service.resolve_review_item(queued.review_id, _resolution())
    output = tmp_path / "calibration" / "v1" / "manifest.json"
    args = Namespace(
        database_url=database_url,
        dataset_id="review-export",
        version="v1",
        output=str(output),
        description="Reviewed production disagreements.",
    )

    assert _export_review_calibration(args) == 0
    first_payload = output.read_text(encoding="utf-8")
    assert _export_review_calibration(args) == 10
    assert output.read_text(encoding="utf-8") == first_payload
