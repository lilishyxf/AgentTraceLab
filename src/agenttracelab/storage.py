from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from agenttracelab.judging import LlmJudgeReport
from agenttracelab.models import EvaluationReport, TraceEnvelope
from agenttracelab.review import ReviewItem, ReviewState


class Base(DeclarativeBase):
    pass


class TraceRow(Base):
    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    source_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EvaluationRow(Base):
    __tablename__ = "evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.trace_id"), nullable=False, index=True)
    evaluator_version: Mapped[str] = mapped_column(String(128), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class JudgeReportRow(Base):
    __tablename__ = "judge_reports"

    judge_report_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.trace_id"), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class ReviewItemRow(Base):
    __tablename__ = "review_items"

    review_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.trace_id"), nullable=False, index=True)
    deterministic_evaluation_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    judge_report_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    priority: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class Database:
    def __init__(self, url: str) -> None:
        options: dict = {"future": True}
        if url == "sqlite+pysqlite:///:memory:":
            options.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
        elif url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **options)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine) as session:
            yield session

    def save_trace(self, trace: TraceEnvelope) -> None:
        now = datetime.now(UTC)
        payload = trace.model_dump(mode="json")
        with self.session() as session:
            row = session.get(TraceRow, trace.trace_id)
            if row is None:
                row = TraceRow(
                    trace_id=trace.trace_id,
                    source_kind=trace.source.kind,
                    schema_version=trace.schema_version,
                    payload=payload,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.source_kind = trace.source.kind
                row.schema_version = trace.schema_version
                row.payload = payload
                row.updated_at = now
            session.commit()

    def get_trace(self, trace_id: str) -> TraceEnvelope | None:
        with self.session() as session:
            row = session.get(TraceRow, trace_id)
            return TraceEnvelope.model_validate(row.payload) if row else None

    def save_evaluation(self, report: EvaluationReport) -> None:
        with self.session() as session:
            session.add(
                EvaluationRow(
                    evaluation_id=report.evaluation_id,
                    trace_id=report.trace_id,
                    evaluator_version=report.evaluator_version,
                    passed=report.passed,
                    score=report.score,
                    payload=report.model_dump(mode="json"),
                    created_at=report.created_at,
                )
            )
            session.commit()

    def get_latest_evaluation(self, trace_id: str) -> EvaluationReport | None:
        query = (
            select(EvaluationRow)
            .where(EvaluationRow.trace_id == trace_id)
            .order_by(EvaluationRow.created_at.desc())
            .limit(1)
        )
        with self.session() as session:
            row = session.scalar(query)
            return EvaluationReport.model_validate(row.payload) if row else None

    def save_judge_report(self, report: LlmJudgeReport) -> None:
        with self.session() as session:
            session.add(
                JudgeReportRow(
                    judge_report_id=report.judge_report_id,
                    trace_id=report.trace_id,
                    model=report.model,
                    prompt_version=report.prompt_version,
                    verdict=report.verdict.value,
                    payload=report.model_dump(mode="json"),
                    created_at=report.created_at,
                )
            )
            session.commit()

    def get_latest_judge_report(self, trace_id: str) -> LlmJudgeReport | None:
        query = (
            select(JudgeReportRow)
            .where(JudgeReportRow.trace_id == trace_id)
            .order_by(JudgeReportRow.created_at.desc())
            .limit(1)
        )
        with self.session() as session:
            row = session.scalar(query)
            return LlmJudgeReport.model_validate_json(json.dumps(row.payload)) if row else None

    def get_judge_report(self, judge_report_id: str) -> LlmJudgeReport | None:
        with self.session() as session:
            row = session.get(JudgeReportRow, judge_report_id)
            return LlmJudgeReport.model_validate_json(json.dumps(row.payload)) if row else None

    def save_review_item(self, item: ReviewItem) -> None:
        payload = item.model_dump(mode="json")
        with self.session() as session:
            row = session.get(ReviewItemRow, item.review_id)
            if row is None:
                row = ReviewItemRow(
                    review_id=item.review_id,
                    trace_id=item.trace_id,
                    deterministic_evaluation_id=item.deterministic_evaluation_id,
                    judge_report_id=item.judge_report_id,
                    state=item.state.value,
                    priority=item.priority.value,
                    payload=payload,
                    created_at=item.created_at,
                )
                session.add(row)
            else:
                row.state = item.state.value
                row.priority = item.priority.value
                row.payload = payload
            session.commit()

    def get_review_item(self, review_id: str) -> ReviewItem | None:
        with self.session() as session:
            row = session.get(ReviewItemRow, review_id)
            return ReviewItem.model_validate(row.payload) if row else None

    def find_review_item(
        self,
        trace_id: str,
        deterministic_evaluation_id: str,
        judge_report_id: str,
    ) -> ReviewItem | None:
        query = (
            select(ReviewItemRow)
            .where(
                ReviewItemRow.trace_id == trace_id,
                ReviewItemRow.deterministic_evaluation_id == deterministic_evaluation_id,
                ReviewItemRow.judge_report_id == judge_report_id,
            )
            .limit(1)
        )
        with self.session() as session:
            row = session.scalar(query)
            return ReviewItem.model_validate(row.payload) if row else None

    def list_review_items(self, state: ReviewState | None = None) -> tuple[ReviewItem, ...]:
        query = select(ReviewItemRow).order_by(ReviewItemRow.created_at.asc())
        if state is not None:
            query = query.where(ReviewItemRow.state == state.value)
        with self.session() as session:
            rows = session.scalars(query).all()
            return tuple(ReviewItem.model_validate(row.payload) for row in rows)
