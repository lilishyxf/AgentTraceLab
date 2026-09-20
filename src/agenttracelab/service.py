from __future__ import annotations

from typing import Any

from agenttracelab.adapters import adapt_otlp_json, adapt_wasmhatch_journal
from agenttracelab.calibration import (
    CalibrationCase,
    CalibrationCaseProvenance,
    CalibrationManifest,
    HumanCalibrationLabel,
    RecordedJudgeRating,
)
from agenttracelab.comparison import compare_evaluations
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.judging import LlmJudgeReport, TraceJudge
from agenttracelab.models import ComparisonReport, EvaluationReport, TraceEnvelope
from agenttracelab.review import (
    ReviewItem,
    ReviewResolutionInput,
    ReviewState,
    resolve_review,
    select_for_review,
)
from agenttracelab.storage import Database


class TraceNotFoundError(LookupError):
    pass


class AgentTraceService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def import_wasmhatch(self, payload: dict[str, Any]) -> tuple[TraceEnvelope, EvaluationReport]:
        trace = adapt_wasmhatch_journal(payload)
        return self._save_and_evaluate(trace)

    def import_normalized(self, trace: TraceEnvelope) -> tuple[TraceEnvelope, EvaluationReport]:
        return self._save_and_evaluate(trace)

    def import_otlp(self, payload: dict[str, Any]) -> tuple[tuple[TraceEnvelope, EvaluationReport], ...]:
        return tuple(self._save_and_evaluate(trace) for trace in adapt_otlp_json(payload))

    def _save_and_evaluate(self, trace: TraceEnvelope) -> tuple[TraceEnvelope, EvaluationReport]:
        self.database.save_trace(trace)
        report = evaluate_trace(trace)
        self.database.save_evaluation(report)
        return trace, report

    def get_trace(self, trace_id: str) -> TraceEnvelope:
        trace = self.database.get_trace(trace_id)
        if trace is None:
            raise TraceNotFoundError(trace_id)
        return trace

    def get_latest_evaluation(self, trace_id: str) -> EvaluationReport:
        report = self.database.get_latest_evaluation(trace_id)
        if report is None:
            raise TraceNotFoundError(trace_id)
        return report

    def compare(self, baseline_trace_id: str, candidate_trace_id: str) -> ComparisonReport:
        baseline = self.get_latest_evaluation(baseline_trace_id)
        candidate = self.get_latest_evaluation(candidate_trace_id)
        return compare_evaluations(baseline, candidate)

    def judge_trace(self, trace_id: str, judge: TraceJudge) -> LlmJudgeReport:
        trace = self.get_trace(trace_id)
        deterministic_report = self.get_latest_evaluation(trace_id)
        report = judge.evaluate(trace, deterministic_report)
        self.database.save_judge_report(report)
        self.queue_review(trace_id)
        return report

    def get_latest_judge_report(self, trace_id: str) -> LlmJudgeReport:
        report = self.database.get_latest_judge_report(trace_id)
        if report is None:
            raise TraceNotFoundError(trace_id)
        return report

    def queue_review(self, trace_id: str, *, force: bool = False) -> ReviewItem | None:
        deterministic = self.get_latest_evaluation(trace_id)
        judge = self.get_latest_judge_report(trace_id)
        existing = self.database.find_review_item(
            trace_id,
            deterministic.evaluation_id,
            judge.judge_report_id,
        )
        if existing is not None:
            return existing
        item = select_for_review(deterministic, judge, force=force)
        if item is not None:
            self.database.save_review_item(item)
        return item

    def get_review_item(self, review_id: str) -> ReviewItem:
        item = self.database.get_review_item(review_id)
        if item is None:
            raise TraceNotFoundError(review_id)
        return item

    def list_review_items(self, state: ReviewState | None = None) -> tuple[ReviewItem, ...]:
        return self.database.list_review_items(state)

    def resolve_review_item(
        self,
        review_id: str,
        resolution: ReviewResolutionInput,
    ) -> ReviewItem:
        resolved = resolve_review(self.get_review_item(review_id), resolution)
        self.database.save_review_item(resolved)
        return resolved

    def export_resolved_reviews(
        self,
        *,
        dataset_id: str,
        version: str,
        description: str = "Human-resolved AgentTraceLab review items.",
    ) -> CalibrationManifest:
        cases: list[CalibrationCase] = []
        for item in self.list_review_items(ReviewState.RESOLVED):
            if item.resolution is None:
                continue
            judge = self.database.get_judge_report(item.judge_report_id)
            if judge is None:
                raise TraceNotFoundError(item.judge_report_id)
            cases.append(
                CalibrationCase(
                    case_id=item.review_id,
                    trace_id=item.trace_id,
                    human=HumanCalibrationLabel(
                        verdict=item.resolution.verdict,
                        dimensions=item.resolution.dimensions,
                        annotator_count=1,
                        notes=item.resolution.notes,
                    ),
                    judges=(
                        RecordedJudgeRating(
                            judge_key=(f"{judge.provider}:{judge.model}@{judge.prompt_version}"),
                            provider=judge.provider,
                            model=judge.model,
                            prompt_version=judge.prompt_version,
                            judge_report_id=judge.judge_report_id,
                            verdict=judge.verdict,
                            dimensions={entry.name: entry.score for entry in judge.dimensions},
                        ),
                    ),
                    provenance=CalibrationCaseProvenance(
                        review_id=item.review_id,
                        deterministic_evaluation_id=item.deterministic_evaluation_id,
                        judge_report_id=item.judge_report_id,
                        review_reasons=tuple(reason.value for reason in item.reasons),
                        resolved_at=item.resolution.resolved_at,
                    ),
                )
            )
        if not cases:
            raise ValueError("no resolved review items are available for export")
        return CalibrationManifest(
            dataset_id=dataset_id,
            version=version,
            description=description,
            cases=tuple(cases),
        )
