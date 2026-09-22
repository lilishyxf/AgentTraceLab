from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenttracelab.adapters import adapt_wasmhatch_journal
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.failures import FailureCategory, FailureFinding, classify_evaluation
from agenttracelab.models import EvaluationReport, FrozenModel, TraceEnvelope
from agenttracelab.optimization import (
    OptimizationFeedback,
    TrainingRewardRecord,
    build_optimization_feedback,
    build_training_reward,
)

EvidenceMode = Literal["synthetic_fixture", "recorded_local", "live_provider"]
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
DERIVED_METRIC_KEYS = (
    "modelEvents",
    "toolCalls",
    "scriptRuns",
    "proposalsPrepared",
    "approvals",
    "rejections",
    "commits",
    "conflicts",
    "uncertainOutcomes",
    "failures",
)


class WasmHatchBatchEntry(FrozenModel):
    scenario_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    journal: str = Field(min_length=1, max_length=512)
    tags: tuple[str, ...] = ()


class WasmHatchBatchManifest(FrozenModel):
    schema_version: Literal["agenttracelab.wasmhatch-batch.v1"]
    batch_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    version: str = Field(min_length=1, max_length=64, pattern=ID_PATTERN)
    evidence_mode: EvidenceMode
    entries: tuple[WasmHatchBatchEntry, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_scenario_ids(self) -> WasmHatchBatchManifest:
        scenario_ids = [entry.scenario_id for entry in self.entries]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("batch scenario IDs must be unique")
        return self


class WasmHatchBatchItemResult(FrozenModel):
    scenario_id: str
    trace_id: str
    state: str
    evaluation: EvaluationReport
    failures: tuple[FailureFinding, ...]
    optimization_feedback: OptimizationFeedback
    training_reward: TrainingRewardRecord


class WasmHatchBatchReport(FrozenModel):
    schema_version: Literal["agenttracelab.wasmhatch-batch-report.v2"] = (
        "agenttracelab.wasmhatch-batch-report.v2"
    )
    batch_id: str
    batch_version: str
    evidence_mode: EvidenceMode
    created_at: datetime
    passed: bool
    run_count: int = Field(ge=1)
    passed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=100)
    state_counts: dict[str, int]
    failure_counts: dict[FailureCategory, int]
    derived_metric_totals: dict[str, int]
    evidence_notice: str
    results: tuple[WasmHatchBatchItemResult, ...]


def _load_json_object(path: Path, *, size_limit: int, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path.name}")
    if path.stat().st_size > size_limit:
        raise ValueError(f"{label} exceeds its size limit: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path.name}")
    return value


def _safe_journal_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"journal path escapes the batch directory: {relative_path}") from exc
    return candidate


def _derived_metrics(trace: TraceEnvelope) -> dict[str, int]:
    spans = trace.spans
    category = lambda span: span.attributes.get("wasmhatch.category")  # noqa: E731
    outcome = lambda span: span.attributes.get("wasmhatch.outcome")  # noqa: E731
    return {
        "modelEvents": sum(category(span) == "model" for span in spans),
        "toolCalls": sum(category(span) == "tool" for span in spans),
        "scriptRuns": sum(category(span) == "script" and outcome(span) == "completed" for span in spans),
        "proposalsPrepared": sum(
            category(span) == "proposal" and outcome(span) == "prepared" for span in spans
        ),
        "approvals": sum(category(span) == "approval" and outcome(span) == "approved" for span in spans),
        "rejections": sum(category(span) == "approval" and outcome(span) == "rejected" for span in spans),
        "commits": sum(category(span) == "effect" and outcome(span) == "committed" for span in spans),
        "conflicts": sum(outcome(span) == "conflict" for span in spans),
        "uncertainOutcomes": sum(outcome(span) == "uncertain" for span in spans),
        "failures": sum(outcome(span) in {"failed", "denied"} for span in spans),
    }


def evaluate_wasmhatch_batch(manifest_path: str | Path) -> WasmHatchBatchReport:
    path = Path(manifest_path).resolve()
    manifest = WasmHatchBatchManifest.model_validate(
        _load_json_object(path, size_limit=1_048_576, label="batch manifest")
    )
    root = path.parent.resolve()
    results: list[WasmHatchBatchItemResult] = []
    trace_ids: set[str] = set()
    metric_totals = dict.fromkeys(DERIVED_METRIC_KEYS, 0)
    for entry in manifest.entries:
        journal_path = _safe_journal_path(root, entry.journal)
        payload = _load_json_object(journal_path, size_limit=524_288, label="run journal")
        trace = adapt_wasmhatch_journal(payload)
        if trace.trace_id in trace_ids:
            raise ValueError(f"batch contains duplicate trace ID: {trace.trace_id}")
        trace_ids.add(trace.trace_id)
        evaluation = evaluate_trace(trace)
        for key, value in _derived_metrics(trace).items():
            metric_totals[key] += value
        results.append(
            WasmHatchBatchItemResult(
                scenario_id=entry.scenario_id,
                trace_id=trace.trace_id,
                state=trace.state,
                evaluation=evaluation,
                failures=classify_evaluation(evaluation),
                optimization_feedback=build_optimization_feedback(trace, evaluation),
                training_reward=build_training_reward(trace, evaluation),
            )
        )

    passed_runs = sum(result.evaluation.passed for result in results)
    failure_counts: Counter[FailureCategory] = Counter(
        finding.category for result in results for finding in result.failures
    )
    state_counts = Counter(result.state for result in results)
    run_count = len(results)
    return WasmHatchBatchReport(
        batch_id=manifest.batch_id,
        batch_version=manifest.version,
        evidence_mode=manifest.evidence_mode,
        created_at=datetime.now(UTC),
        passed=passed_runs == run_count,
        run_count=run_count,
        passed_runs=passed_runs,
        failed_runs=run_count - passed_runs,
        pass_rate=round((passed_runs / run_count) * 100, 2),
        state_counts=dict(sorted(state_counts.items())),
        failure_counts=dict(sorted(failure_counts.items(), key=lambda item: item[0].value)),
        derived_metric_totals=metric_totals,
        evidence_notice=(
            "Results describe exported run journals in the declared evidence mode; "
            "they do not independently prove provider execution or business correctness."
        ),
        results=tuple(results),
    )
