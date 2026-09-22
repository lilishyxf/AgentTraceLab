from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import Field, HttpUrl, model_validator

from agenttracelab.adapters import (
    adapt_deepresearch_snapshot,
    adapt_otlp_json,
    adapt_wasmhatch_journal,
)
from agenttracelab.datasets import match_expectations
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.failures import (
    FailureCategory,
    FailureFinding,
    classify_evaluation,
    runtime_failure,
)
from agenttracelab.models import CheckStatus, EvaluationReport, FrozenModel, TraceEnvelope


class ReplayTarget(FrozenModel):
    endpoint: HttpUrl
    timeout_seconds: float = Field(default=15.0, ge=0.1, le=60.0)
    max_response_bytes: int = Field(default=2_097_152, ge=1_024, le=5_242_880)
    bearer_token_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,127}$")

    @model_validator(mode="after")
    def reject_embedded_credentials(self) -> ReplayTarget:
        if self.endpoint.username or self.endpoint.password:
            raise ValueError("replay endpoint must not contain embedded credentials")
        return self


class ReplayTask(FrozenModel):
    task_id: str = Field(min_length=1, max_length=128)
    request: dict[str, Any]
    adapter: Literal["wasmhatch", "deepresearch", "normalized", "otlp"]
    response_path: str | None = Field(default=None, max_length=256)
    expected_pass: bool = True
    expected_check_statuses: dict[str, CheckStatus] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_request_size(self) -> ReplayTask:
        encoded = json.dumps(self.request, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 262_144:
            raise ValueError("replay task request exceeds the 256 KiB limit")
        return self


class ReplayManifest(FrozenModel):
    schema_version: Literal["agenttracelab.replay.v1"]
    replay_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    target: ReplayTarget
    tasks: tuple[ReplayTask, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_task_ids(self) -> ReplayManifest:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("replay task IDs must be unique")
        return self


class ReplayTaskResult(FrozenModel):
    task_id: str
    status: Literal["completed", "transport_error", "contract_error"]
    http_status: int | None = None
    elapsed_ms: int = Field(ge=0)
    trace_id: str | None = None
    evaluation: EvaluationReport | None = None
    matched_expectation: bool = False
    mismatches: tuple[str, ...] = ()
    failures: tuple[FailureFinding, ...] = ()
    error: str | None = None


class ReplayRunReport(FrozenModel):
    schema_version: Literal["agenttracelab.replay-run.v1"] = "agenttracelab.replay-run.v1"
    replay_id: str
    replay_version: str
    created_at: datetime
    passed: bool
    matched_tasks: int
    task_count: int
    completed_tasks: int
    evaluation_pass_rate: float = Field(ge=0, le=100)
    results: tuple[ReplayTaskResult, ...]


TraceEvaluationObserver = Callable[[TraceEnvelope, EvaluationReport], None]


def load_replay_manifest(path: str | Path) -> ReplayManifest:
    manifest_path = Path(path).resolve()
    if manifest_path.stat().st_size > 1_048_576:
        raise ValueError("replay manifest exceeds the 1 MiB limit")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("replay manifest must contain a JSON object")
    return ReplayManifest.model_validate(value)


def _extract_payload(payload: Any, response_path: str | None) -> dict[str, Any]:
    value = payload
    if response_path:
        for segment in response_path.split("."):
            if not segment or not isinstance(value, dict) or segment not in value:
                raise ValueError(f"response_path does not exist: {response_path}")
            value = value[segment]
    if not isinstance(value, dict):
        raise ValueError("replay response trace payload must be a JSON object")
    return value


def _adapt_trace(task: ReplayTask, payload: dict[str, Any]) -> TraceEnvelope:
    if task.adapter == "wasmhatch":
        return adapt_wasmhatch_journal(payload)
    if task.adapter == "deepresearch":
        return adapt_deepresearch_snapshot(payload)
    if task.adapter == "normalized":
        return TraceEnvelope.model_validate(payload)
    traces = adapt_otlp_json(payload)
    if len(traces) != 1:
        raise ValueError(f"replay OTLP task {task.task_id} must return exactly one trace")
    return traces[0]


def _error_result(
    task: ReplayTask,
    *,
    status: Literal["transport_error", "contract_error"],
    elapsed_ms: int,
    category: FailureCategory,
    error: str,
    http_status: int | None = None,
) -> ReplayTaskResult:
    return ReplayTaskResult(
        task_id=task.task_id,
        status=status,
        http_status=http_status,
        elapsed_ms=elapsed_ms,
        failures=(runtime_failure(category, error),),
        error=error,
    )


def _run_task(
    client: httpx.Client,
    target: ReplayTarget,
    task: ReplayTask,
    headers: dict[str, str],
    on_evaluated: TraceEvaluationObserver | None = None,
) -> ReplayTaskResult:
    started = perf_counter()
    try:
        request = client.build_request(
            "POST",
            str(target.endpoint),
            json=task.request,
            headers=headers,
        )
        response = client.send(request, stream=True)
        try:
            if response.status_code < 200 or response.status_code >= 300:
                elapsed_ms = round((perf_counter() - started) * 1_000)
                return _error_result(
                    task,
                    status="transport_error",
                    elapsed_ms=elapsed_ms,
                    category=FailureCategory.TRANSPORT,
                    error=f"target returned HTTP {response.status_code}",
                    http_status=response.status_code,
                )
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > target.max_response_bytes:
                    elapsed_ms = round((perf_counter() - started) * 1_000)
                    return _error_result(
                        task,
                        status="contract_error",
                        elapsed_ms=elapsed_ms,
                        category=FailureCategory.TRACE_CONTRACT,
                        error="target response exceeds configured size limit",
                        http_status=response.status_code,
                    )
        finally:
            response.close()
    except httpx.RequestError as exc:
        elapsed_ms = round((perf_counter() - started) * 1_000)
        return _error_result(
            task,
            status="transport_error",
            elapsed_ms=elapsed_ms,
            category=FailureCategory.TRANSPORT,
            error=f"target request failed: {type(exc).__name__}",
        )
    elapsed_ms = round((perf_counter() - started) * 1_000)
    try:
        response_payload = json.loads(body)
        trace_payload = _extract_payload(response_payload, task.response_path)
        trace = _adapt_trace(task, trace_payload)
    except ValueError as exc:
        return _error_result(
            task,
            status="contract_error",
            elapsed_ms=elapsed_ms,
            category=FailureCategory.TRACE_CONTRACT,
            error=f"invalid target trace contract: {exc}",
            http_status=response.status_code,
        )

    evaluation = evaluate_trace(trace)
    if on_evaluated is not None:
        on_evaluated(trace, evaluation)
    matched, mismatches = match_expectations(
        expected_pass=task.expected_pass,
        expected_check_statuses=task.expected_check_statuses,
        report=evaluation,
    )
    failures = list(classify_evaluation(evaluation))
    if mismatches:
        failures.append(
            runtime_failure(
                FailureCategory.EXPECTATION,
                f"{len(mismatches)} replay expectation mismatch(es).",
            )
        )
    return ReplayTaskResult(
        task_id=task.task_id,
        status="completed",
        http_status=response.status_code,
        elapsed_ms=elapsed_ms,
        trace_id=trace.trace_id,
        evaluation=evaluation,
        matched_expectation=matched,
        mismatches=mismatches,
        failures=tuple(failures),
    )


def run_replay_manifest(
    manifest: ReplayManifest,
    *,
    client: httpx.Client | None = None,
    on_evaluated: TraceEvaluationObserver | None = None,
) -> ReplayRunReport:
    headers = {"Accept": "application/json"}
    if manifest.target.bearer_token_env:
        token = os.getenv(manifest.target.bearer_token_env)
        if not token:
            raise ValueError(
                f"missing replay bearer token environment variable: {manifest.target.bearer_token_env}"
            )
        headers["Authorization"] = f"Bearer {token}"

    owns_client = client is None
    active_client = client or httpx.Client(
        timeout=manifest.target.timeout_seconds,
        follow_redirects=False,
    )
    try:
        results = tuple(
            _run_task(
                active_client,
                manifest.target,
                task,
                headers,
                on_evaluated,
            )
            for task in manifest.tasks
        )
    finally:
        if owns_client:
            active_client.close()

    matched_tasks = sum(result.matched_expectation for result in results)
    completed = [result for result in results if result.evaluation is not None]
    evaluation_passes = sum(result.evaluation.passed for result in completed if result.evaluation)
    pass_rate = round((evaluation_passes / len(completed)) * 100, 2) if completed else 0.0
    return ReplayRunReport(
        replay_id=manifest.replay_id,
        replay_version=manifest.version,
        created_at=datetime.now(UTC),
        passed=matched_tasks == len(results),
        matched_tasks=matched_tasks,
        task_count=len(results),
        completed_tasks=len(completed),
        evaluation_pass_rate=pass_rate,
        results=results,
    )


def run_replay(
    manifest_path: str | Path,
    *,
    client: httpx.Client | None = None,
    on_evaluated: TraceEvaluationObserver | None = None,
) -> ReplayRunReport:
    return run_replay_manifest(
        load_replay_manifest(manifest_path),
        client=client,
        on_evaluated=on_evaluated,
    )
