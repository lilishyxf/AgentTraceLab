from __future__ import annotations

import json
from pathlib import Path

from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkManifest,
    DeepResearchBenchmarkReport,
    DeepResearchBenchmarkTask,
)
from agenttracelab.models import FrozenModel
from agenttracelab.tool_trajectory import (
    ToolTrajectoryCase,
    ToolTrajectoryManifest,
    load_tool_trajectory_annotation_queue,
)


class ExcludedAnnotationCase(FrozenModel):
    case_id: str
    execution_lane: str
    reason: str


class AnnotationBenchmarkBuild(FrozenModel):
    manifest: DeepResearchBenchmarkManifest
    included_case_ids: tuple[str, ...]
    excluded_cases: tuple[ExcludedAnnotationCase, ...]


class TrajectoryManifestBuild(FrozenModel):
    manifest: ToolTrajectoryManifest
    included_trial_ids: tuple[str, ...]
    skipped_trial_ids: tuple[str, ...]


def _write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise ValueError(f"output already exists; choose a new immutable path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = payload.model_dump(mode="json") if isinstance(payload, FrozenModel) else payload
    path.write_text(f"{json.dumps(value, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def build_annotation_benchmark_manifest(
    queue_path: str | Path,
    output_path: str | Path,
    *,
    benchmark_id: str,
    target_name: str,
    target_provider: str,
    target_model: str,
    target_revision: str | None = None,
    trials_per_task: int = 1,
) -> AnnotationBenchmarkBuild:
    queue = load_tool_trajectory_annotation_queue(queue_path)
    tasks: list[DeepResearchBenchmarkTask] = []
    excluded: list[ExcludedAnnotationCase] = []
    for case in queue.cases:
        if case.execution_lane != "live-safe":
            excluded.append(
                ExcludedAnnotationCase(
                    case_id=case.case_id,
                    execution_lane=case.execution_lane,
                    reason=(
                        "requires controlled fault injection"
                        if case.execution_lane == "controlled-fault"
                        else "requires an isolated manual safety review"
                    ),
                )
            )
            continue
        tasks.append(
            DeepResearchBenchmarkTask(
                task_id=case.case_id,
                prompt=case.prompt,
                source_mode="web",
                workflow_mode="deep_research",
                report_type="brief",
                include_domains=case.expected_domains,
                tags=(case.task_family, *case.tags, "annotation-candidate"),
            )
        )
    if not tasks:
        raise ValueError("the annotation queue contains no live-safe cases")
    manifest = DeepResearchBenchmarkManifest(
        schema_version="agenttracelab.deepresearch-benchmark.v1",
        benchmark_id=benchmark_id,
        version=queue.version,
        description=(
            f"Live-safe execution lane generated from annotation queue {queue.dataset_id}; "
            "excluded cases require controlled or manual evaluation."
        ),
        evidence_class="live-system",
        target_name=target_name,
        target_revision=target_revision,
        target_provider=target_provider,
        target_model=target_model,
        minimum_judge_separation="none",
        trials_per_task=trials_per_task,
        tasks=tuple(tasks),
    )
    result = AnnotationBenchmarkBuild(
        manifest=manifest,
        included_case_ids=tuple(task.task_id for task in tasks),
        excluded_cases=tuple(excluded),
    )
    _write_new_json(Path(output_path), manifest)
    return result


def _execution_case_id(task_id: str, trial: int) -> str:
    suffix = f"-trial-{trial:02d}"
    return f"{task_id[: 128 - len(suffix)]}{suffix}"


def materialize_tool_trajectory_manifest(
    queue_path: str | Path,
    benchmark_report_path: str | Path,
    output_path: str | Path,
) -> TrajectoryManifestBuild:
    queue = load_tool_trajectory_annotation_queue(queue_path)
    report_path = Path(benchmark_report_path).resolve()
    output = Path(output_path).resolve()
    if output.parent != report_path.parent:
        raise ValueError("tool-trajectory manifest must be written beside the benchmark report")
    report = DeepResearchBenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    cases_by_id = {case.case_id: case for case in queue.cases}
    trajectory_cases: list[ToolTrajectoryCase] = []
    included: list[str] = []
    skipped: list[str] = []
    for trial in report.trials:
        trial_id = f"{trial.task_id}:{trial.trial}"
        annotation = cases_by_id.get(trial.task_id)
        if annotation is None or not trial.snapshot_path:
            skipped.append(trial_id)
            continue
        accepted_domains = (
            annotation.expected_domains or annotation.draft_expectation.accepted_evidence_domains
        )
        expectation = annotation.draft_expectation.model_copy(
            update={
                "accepted_evidence_domains": accepted_domains,
                "min_accepted_domain_evidence": (
                    1 if accepted_domains else annotation.draft_expectation.min_accepted_domain_evidence
                ),
            }
        )
        trajectory_cases.append(
            ToolTrajectoryCase(
                case_id=_execution_case_id(trial.task_id, trial.trial),
                snapshot=trial.snapshot_path,
                expectation=expectation,
                tags=(annotation.task_family, *annotation.tags, f"trial:{trial.trial}"),
            )
        )
        included.append(trial_id)
    if not trajectory_cases:
        raise ValueError("benchmark report contains no snapshots matching the annotation queue")
    review_mode = "human-approved" if queue.status == "human-approved" else "generated-synthetic"
    manifest = ToolTrajectoryManifest(
        schema_version="agenttracelab.tool-trajectory.v1",
        dataset_id=f"{queue.dataset_id}-executions"[:128],
        version=queue.version,
        review_mode=review_mode,
        annotator_count=queue.annotator_count if review_mode == "human-approved" else 0,
        cases=tuple(trajectory_cases),
    )
    result = TrajectoryManifestBuild(
        manifest=manifest,
        included_trial_ids=tuple(included),
        skipped_trial_ids=tuple(skipped),
    )
    _write_new_json(output, manifest)
    return result
