from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agenttracelab.adapters import (
    adapt_deepresearch_snapshot,
    adapt_otlp_json,
    adapt_wasmhatch_journal,
)
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.models import CheckStatus, EvaluationReport, FrozenModel, TraceEnvelope


class DatasetTask(FrozenModel):
    task_id: str = Field(min_length=1, max_length=128)
    adapter: Literal["wasmhatch", "deepresearch", "normalized", "otlp"]
    fixture: str = Field(min_length=1, max_length=512)
    expected_pass: bool
    expected_check_statuses: dict[str, CheckStatus] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()


class DatasetManifest(FrozenModel):
    schema_version: Literal["agenttracelab.dataset.v1"]
    dataset_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    tasks: tuple[DatasetTask, ...]

    @model_validator(mode="after")
    def validate_task_ids(self) -> DatasetManifest:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("dataset task IDs must be unique")
        if not task_ids:
            raise ValueError("dataset must contain at least one task")
        return self


class DatasetTaskResult(FrozenModel):
    task_id: str
    trace_id: str
    evaluation: EvaluationReport
    matched_expectation: bool
    mismatches: tuple[str, ...]


class DatasetRunReport(FrozenModel):
    schema_version: Literal["agenttracelab.dataset-run.v1"] = "agenttracelab.dataset-run.v1"
    dataset_id: str
    dataset_version: str
    created_at: datetime
    passed: bool
    matched_tasks: int
    task_count: int
    evaluation_pass_rate: float = Field(ge=0, le=100)
    results: tuple[DatasetTaskResult, ...]


def _safe_fixture_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"fixture path escapes the dataset directory: {relative_path}") from exc
    if not candidate.is_file():
        raise ValueError(f"fixture does not exist: {relative_path}")
    if candidate.stat().st_size > 1_048_576:
        raise ValueError(f"fixture exceeds the 1 MiB dataset limit: {relative_path}")
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"fixture must contain a JSON object: {path.name}")
    return value


def _adapt_task(task: DatasetTask, payload: dict[str, Any]) -> TraceEnvelope:
    if task.adapter == "wasmhatch":
        return adapt_wasmhatch_journal(payload)
    if task.adapter == "deepresearch":
        return adapt_deepresearch_snapshot(payload)
    if task.adapter == "normalized":
        return TraceEnvelope.model_validate(payload)
    traces = adapt_otlp_json(payload)
    if len(traces) != 1:
        raise ValueError(f"dataset OTLP task {task.task_id} must contain exactly one trace")
    return traces[0]


def match_expectations(
    *,
    expected_pass: bool,
    expected_check_statuses: dict[str, CheckStatus],
    report: EvaluationReport,
) -> tuple[bool, tuple[str, ...]]:
    mismatches: list[str] = []
    if report.passed != expected_pass:
        mismatches.append(f"expected passed={expected_pass}, received {report.passed}")
    actual = {check.check_id: check.status for check in report.checks}
    for check_id, expected_status in expected_check_statuses.items():
        received = actual.get(check_id)
        if received != expected_status:
            rendered = received.value if received else "missing"
            mismatches.append(f"{check_id}: expected {expected_status.value}, received {rendered}")
    return not mismatches, tuple(mismatches)


def run_dataset(manifest_path: str | Path) -> DatasetRunReport:
    path = Path(manifest_path).resolve()
    manifest = DatasetManifest.model_validate(_load_json(path))
    root = path.parent.resolve()
    results: list[DatasetTaskResult] = []
    for task in manifest.tasks:
        fixture_path = _safe_fixture_path(root, task.fixture)
        trace = _adapt_task(task, _load_json(fixture_path))
        evaluation = evaluate_trace(trace)
        matched, mismatches = match_expectations(
            expected_pass=task.expected_pass,
            expected_check_statuses=task.expected_check_statuses,
            report=evaluation,
        )
        results.append(
            DatasetTaskResult(
                task_id=task.task_id,
                trace_id=trace.trace_id,
                evaluation=evaluation,
                matched_expectation=matched,
                mismatches=mismatches,
            )
        )
    matched_tasks = sum(result.matched_expectation for result in results)
    evaluation_passes = sum(result.evaluation.passed for result in results)
    return DatasetRunReport(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        created_at=datetime.now(UTC),
        passed=matched_tasks == len(results),
        matched_tasks=matched_tasks,
        task_count=len(results),
        evaluation_pass_rate=round((evaluation_passes / len(results)) * 100, 2),
        results=tuple(results),
    )
