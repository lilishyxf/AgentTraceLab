from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, ValidationError

from agenttracelab import __version__
from agenttracelab.judging import LlmJudgeReport
from agenttracelab.models import ComparisonReport, EvaluationReport, TraceEnvelope
from agenttracelab.otlp_http import (
    DEFAULT_MAX_DECODED_BYTES,
    DEFAULT_MAX_WIRE_BYTES,
    OtlpHttpError,
    decode_otlp_http_request,
    otlp_error_response,
    otlp_success_response,
)
from agenttracelab.review import ReviewItem, ReviewResolutionInput, ReviewState
from agenttracelab.service import AgentTraceService, TraceNotFoundError
from agenttracelab.storage import Database


class ImportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace: TraceEnvelope
    evaluation: EvaluationReport


class BatchImportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: tuple[ImportResponse, ...]


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_trace_id: str
    candidate_trace_id: str


class ReviewQueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: bool = False


class ReviewListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[ReviewItem, ...]


def _configured_positive_int(explicit: int | None, environment_name: str, default: int) -> int:
    raw: int | str = explicit if explicit is not None else os.getenv(environment_name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{environment_name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{environment_name} must be a positive integer")
    return value


def create_app(
    database_url: str | None = None,
    *,
    otlp_max_wire_bytes: int | None = None,
    otlp_max_decoded_bytes: int | None = None,
) -> FastAPI:
    url = database_url or os.getenv("DATABASE_URL", "sqlite+pysqlite:///./agenttracelab.db")
    max_wire_bytes = _configured_positive_int(
        otlp_max_wire_bytes,
        "AGENTTRACELAB_OTLP_MAX_WIRE_BYTES",
        DEFAULT_MAX_WIRE_BYTES,
    )
    max_decoded_bytes = _configured_positive_int(
        otlp_max_decoded_bytes,
        "AGENTTRACELAB_OTLP_MAX_DECODED_BYTES",
        DEFAULT_MAX_DECODED_BYTES,
    )
    database = Database(url)
    database.create_schema()
    service = AgentTraceService(database)

    app = FastAPI(
        title="AgentTraceLab",
        version=__version__,
        description="Trace ingestion and evidence-led regression evaluation for AI agents.",
    )
    app.state.database = database
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/traces/import/wasmhatch", response_model=ImportResponse)
    def import_wasmhatch(payload: dict[str, Any]) -> ImportResponse:
        try:
            trace, report = service.import_wasmhatch(payload)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ImportResponse(trace=trace, evaluation=report)

    @app.post("/v1/traces/import/normalized", response_model=ImportResponse)
    def import_normalized(trace: TraceEnvelope) -> ImportResponse:
        stored, report = service.import_normalized(trace)
        return ImportResponse(trace=stored, evaluation=report)

    @app.post("/v1/traces/import/otlp", response_model=BatchImportResponse)
    def import_otlp(payload: dict[str, Any]) -> BatchImportResponse:
        try:
            imported = service.import_otlp(payload)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BatchImportResponse(
            results=tuple(ImportResponse(trace=trace, evaluation=report) for trace, report in imported)
        )

    @app.post("/v1/traces", response_class=Response)
    async def receive_otlp_traces(request: Request) -> Response:
        try:
            decoded = await decode_otlp_http_request(
                request,
                max_wire_bytes=max_wire_bytes,
                max_decoded_bytes=max_decoded_bytes,
            )
            if decoded.span_count:
                service.import_otlp(decoded.payload)
        except OtlpHttpError as exc:
            return otlp_error_response(request.headers.get("content-type"), exc)
        except (ValidationError, ValueError) as exc:
            error = OtlpHttpError(400, f"OTLP trace payload is invalid: {exc}")
            return otlp_error_response(request.headers.get("content-type"), error)
        return otlp_success_response(decoded.content_type)

    @app.get("/v1/traces/{trace_id}", response_model=TraceEnvelope)
    def get_trace(trace_id: str) -> TraceEnvelope:
        try:
            return service.get_trace(trace_id)
        except TraceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Trace not found") from exc

    @app.get("/v1/evaluations/{trace_id}/latest", response_model=EvaluationReport)
    def get_latest_evaluation(trace_id: str) -> EvaluationReport:
        try:
            return service.get_latest_evaluation(trace_id)
        except TraceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Evaluation not found") from exc

    @app.get("/v1/judgments/{trace_id}/latest", response_model=LlmJudgeReport)
    def get_latest_judgment(trace_id: str) -> LlmJudgeReport:
        try:
            return service.get_latest_judge_report(trace_id)
        except TraceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="LLM judge report not found") from exc

    @app.post("/v1/reviews/queue/{trace_id}", response_model=ReviewItem)
    def queue_review(trace_id: str, request: ReviewQueueRequest) -> ReviewItem:
        try:
            item = service.queue_review(trace_id, force=request.force)
        except TraceNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Trace, evaluation, or LLM judge report not found",
            ) from exc
        if item is None:
            raise HTTPException(
                status_code=409,
                detail="The latest evaluation and judge report do not meet review selection rules",
            )
        return item

    @app.get("/v1/reviews", response_model=ReviewListResponse)
    def list_reviews(state: ReviewState | None = None) -> ReviewListResponse:
        return ReviewListResponse(items=service.list_review_items(state))

    @app.get("/v1/reviews/{review_id}", response_model=ReviewItem)
    def get_review(review_id: str) -> ReviewItem:
        try:
            return service.get_review_item(review_id)
        except TraceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Review item not found") from exc

    @app.post("/v1/reviews/{review_id}/resolve", response_model=ReviewItem)
    def resolve_review(review_id: str, resolution: ReviewResolutionInput) -> ReviewItem:
        try:
            return service.resolve_review_item(review_id, resolution)
        except TraceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Review item not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/comparisons", response_model=ComparisonReport)
    def compare(request: ComparisonRequest) -> ComparisonReport:
        try:
            return service.compare(request.baseline_trace_id, request.candidate_trace_id)
        except TraceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Trace or evaluation not found: {exc}") from exc

    return app


app = create_app()
