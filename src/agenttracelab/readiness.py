from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from agenttracelab.deepresearch_benchmark import (
    JUDGE_SEPARATION_RANK,
    DeepResearchBenchmarkManifest,
    DeepResearchBenchmarkReport,
    _evaluate_task_acceptance,
    benchmark_tasks_sha256,
    load_deepresearch_benchmark_manifest,
    task_acceptance_contract_sha256,
)
from agenttracelab.models import FrozenModel

ReadinessStatus = Literal["pass", "fail", "unknown", "not_applicable"]
ReadinessRecommendation = Literal["ready", "hold", "diagnostic_only"]


class EvaluationReadinessPolicy(FrozenModel):
    schema_version: Literal["agenttracelab.evaluation-readiness-policy.v1"]
    policy_id: str = Field(min_length=1, max_length=128)
    intended_use: Literal["diagnostic", "candidate_release", "production_release"]
    min_task_count: int = Field(default=1, ge=1, le=10_000)
    min_trials_per_task: int = Field(default=1, ge=1, le=100)
    min_completion_rate: float = Field(default=1.0, ge=0, le=1)
    max_execution_error_rate: float = Field(default=0.0, ge=0, le=1)
    min_pass_rate_wilson_lower_bound: float = Field(default=0.0, ge=0, le=1)
    min_evidence_supported_pass_rate_lower_bound: float = Field(default=0.0, ge=0, le=1)
    min_semantic_evaluation_coverage: float = Field(default=0.0, ge=0, le=1)
    min_task_acceptance_evaluation_coverage: float = Field(default=0.0, ge=0, le=1)
    min_distinct_expected_statuses: int = Field(default=1, ge=1, le=2)
    allowed_evidence_classes: tuple[Literal["live-system", "integration", "synthetic"], ...] = (
        "live-system",
    )
    min_judge_separation: Literal["not_configured", "different_model", "different_provider"] = (
        "not_configured"
    )
    require_target_revision: bool = True
    require_pinned_claim_judge_identity: bool = False
    require_manifest_integrity: bool = True
    require_unique_trace_ids: bool = True
    require_retained_artifacts: bool = True
    required_suite_kinds: tuple[Literal["capability", "regression", "mixed"], ...] = ()
    required_dataset_review_status: Literal["none", "unreviewed", "generated-synthetic", "human-reviewed"] = (
        "none"
    )
    min_reference_solution_coverage: float = Field(default=0.0, ge=0, le=1)
    min_outcome_grader_coverage: float = Field(default=0.0, ge=0, le=1)
    min_transcript_review_count: int = Field(default=0, ge=0)
    min_production_case_count: int = Field(default=0, ge=0)
    require_judge_calibration: bool = False
    require_reference_solution_artifact: bool = False
    require_dataset_review_artifact: bool = False
    require_judge_calibration_artifact: bool = False
    allowed_environment_isolation: tuple[Literal["none", "session", "process", "container", "vm"], ...] = ()
    require_environment_fingerprint: bool = False
    require_resource_profile: bool = False


class ReadinessCheck(FrozenModel):
    check_id: str
    status: ReadinessStatus
    severity: Literal["required", "advisory"]
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    remediation: str | None = None


class EvaluationReadinessReport(FrozenModel):
    schema_version: Literal["agenttracelab.evaluation-readiness-report.v1"] = (
        "agenttracelab.evaluation-readiness-report.v1"
    )
    audit_id: str
    created_at: datetime
    benchmark_id: str
    benchmark_version: str
    policy_id: str
    intended_use: Literal["diagnostic", "candidate_release", "production_release"]
    manifest_sha256: str
    report_sha256: str
    policy_sha256: str
    recommendation: ReadinessRecommendation
    ready: bool
    required_passed: int = Field(ge=0)
    required_failed: int = Field(ge=0)
    required_unknown: int = Field(ge=0)
    checks: tuple[ReadinessCheck, ...]
    decision_reasons: tuple[str, ...]
    limitations: tuple[str, ...]


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    if resolved.stat().st_size > 16 * 1_048_576:
        raise ValueError(f"{label} exceeds the 16 MiB limit")
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw, hashlib.sha256(raw).hexdigest()


def load_evaluation_readiness_policy(
    path: str | Path,
) -> tuple[EvaluationReadinessPolicy, str]:
    payload, _raw, digest = _load_json(Path(path), label="readiness policy")
    return EvaluationReadinessPolicy.model_validate(payload), digest


def _check(
    check_id: str,
    *,
    passed: bool | None,
    required: bool,
    summary: str,
    evidence: dict[str, Any],
    remediation: str | None = None,
) -> ReadinessCheck:
    if passed is True:
        status: ReadinessStatus = "pass"
    elif passed is False:
        status = "fail"
    else:
        status = "unknown"
    return ReadinessCheck(
        check_id=check_id,
        status=status,
        severity="required" if required else "advisory",
        summary=summary,
        evidence=evidence,
        remediation=remediation,
    )


def _threshold_summary(
    name: str,
    *,
    actual: float | int | None,
    minimum: float | int,
) -> str:
    if actual is None:
        return f"{name} is unavailable."
    if actual >= minimum:
        return f"{name} meets policy ({actual} >= {minimum})."
    return f"{name} is below policy ({actual} < {minimum})."


def _maximum_summary(
    name: str,
    *,
    actual: float | int,
    maximum: float | int,
) -> str:
    if actual <= maximum:
        return f"{name} meets policy ({actual} <= {maximum})."
    return f"{name} exceeds policy ({actual} > {maximum})."


def _ranked_review_status(value: str) -> int:
    return {
        "unspecified": 0,
        "unreviewed": 1,
        "generated-synthetic": 2,
        "human-reviewed": 3,
    }[value]


def _artifact_check(report: DeepResearchBenchmarkReport, report_path: Path) -> ReadinessCheck:
    root = report_path.parent.resolve()
    expected = 0
    present = 0
    invalid: list[str] = []
    missing: list[str] = []
    unsafe: list[str] = []
    for trial in report.trials:
        paths = [trial.snapshot_path, trial.evaluation_path]
        if trial.completed:
            paths.append(trial.trace_path)
        if trial.semantic_gate_passed is not None:
            paths.append(trial.claim_evidence_path)
        for relative in paths:
            expected += 1
            if not relative:
                missing.append(f"{trial.task_id}/trial-{trial.trial:02d}:unrecorded")
                continue
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root):
                unsafe.append(relative)
                continue
            if not candidate.is_file():
                missing.append(relative)
                continue
            try:
                json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                invalid.append(relative)
                continue
            present += 1
    passed = not missing and not invalid and not unsafe
    return _check(
        "artifacts.retained_and_readable",
        passed=passed,
        required=True,
        summary=(
            "Every required trial artifact is retained as readable JSON."
            if passed
            else "Required trial artifacts are missing, unreadable, or escape the report directory."
        ),
        evidence={
            "expected": expected,
            "present": present,
            "missing": missing,
            "invalid": invalid,
            "unsafe": unsafe,
        },
        remediation="Retain snapshot, normalized trace, deterministic evaluation, and semantic report paths.",
    )


def _bound_design_artifact_check(
    *,
    check_id: str,
    label: str,
    manifest_path: Path,
    relative_path: str | None,
    expected_sha256: str | None,
    required: bool,
) -> ReadinessCheck:
    if not relative_path or not expected_sha256:
        if not required:
            return ReadinessCheck(
                check_id=check_id,
                status="not_applicable",
                severity="advisory",
                summary=f"{label} artifact is not required by this policy.",
                evidence={"path": relative_path, "expected_sha256": expected_sha256},
                remediation=None,
            )
        return _check(
            check_id,
            passed=None,
            required=True,
            summary=f"{label} artifact binding is unspecified.",
            evidence={"path": relative_path, "expected_sha256": expected_sha256},
            remediation=f"Bind the exact {label.lower()} JSON artifact and SHA-256 to the manifest.",
        )

    root = manifest_path.parent.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        return _check(
            check_id,
            passed=False,
            required=required,
            summary=f"{label} artifact path escapes the manifest directory.",
            evidence={"path": relative_path, "expected_sha256": expected_sha256},
            remediation=f"Store the {label.lower()} artifact below the manifest directory.",
        )
    if not candidate.is_file():
        return _check(
            check_id,
            passed=False,
            required=required,
            summary=f"{label} artifact is missing.",
            evidence={"path": relative_path, "expected_sha256": expected_sha256},
            remediation=f"Restore the exact {label.lower()} artifact referenced by the manifest.",
        )
    try:
        raw = candidate.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _check(
            check_id,
            passed=False,
            required=required,
            summary=f"{label} artifact is not readable JSON.",
            evidence={"path": relative_path, "expected_sha256": expected_sha256},
            remediation=f"Provide a readable JSON {label.lower()} artifact.",
        )
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    passed = isinstance(payload, dict) and actual_sha256 == expected_sha256
    return _check(
        check_id,
        passed=passed,
        required=required,
        summary=(
            f"{label} artifact matches its manifest digest."
            if passed
            else f"{label} artifact content or digest is invalid."
        ),
        evidence={
            "path": relative_path,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "json_object": isinstance(payload, dict),
        },
        remediation=f"Audit and rebind the exact immutable {label.lower()} artifact.",
    )


def _load_bound_json_object(manifest_path: Path, relative_path: str | None) -> dict[str, Any] | None:
    if not relative_path:
        return None
    root = manifest_path.parent.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_review_content_check(
    *,
    manifest: DeepResearchBenchmarkManifest,
    manifest_path: Path,
    relative_path: str | None,
    required: bool,
) -> ReadinessCheck:
    payload = _load_bound_json_object(manifest_path, relative_path)
    if payload is None:
        return _check(
            "design.dataset_review_content",
            passed=None if required else True,
            required=required,
            summary="Dataset review content is unavailable.",
            evidence={"path": relative_path},
            remediation="Complete and bind an approved review record for the exact task set.",
        )
    entries = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    expected_ids = {task.task_id for task in manifest.tasks}
    observed_ids = {
        str(item.get("task_id")) for item in entries if isinstance(item, dict) and item.get("task_id")
    }
    entry_checks_passed = all(
        isinstance(item, dict)
        and item.get("decision") == "approved"
        and item.get("prompt_approved") is True
        and item.get("expected_status_approved") is True
        and item.get("acceptance_contract_approved") is True
        and item.get("source_constraints_approved") is True
        for item in entries
    )
    expected_tasks_digest = benchmark_tasks_sha256(manifest.tasks)
    issues: list[str] = []
    if payload.get("schema_version") != "agenttracelab.dataset-review.v1":
        issues.append("schema_version")
    if payload.get("benchmark_id") != manifest.benchmark_id:
        issues.append("benchmark_id")
    if payload.get("benchmark_version") != manifest.version:
        issues.append("benchmark_version")
    if payload.get("reviewed_tasks_sha256") != expected_tasks_digest:
        issues.append("reviewed_tasks_sha256")
    if payload.get("review_status") != "approved" or payload.get("approved") is not True:
        issues.append("approval_status")
    if not str(payload.get("reviewer") or "").strip() or not str(payload.get("reviewed_at") or "").strip():
        issues.append("reviewer_identity_or_time")
    if observed_ids != expected_ids or len(entries) != len(expected_ids):
        issues.append("task_set")
    if not entry_checks_passed:
        issues.append("task_decisions")
    return _check(
        "design.dataset_review_content",
        passed=not issues,
        required=required,
        summary=(
            "Dataset review approves the exact benchmark task set."
            if not issues
            else "Dataset review does not approve the exact benchmark task set."
        ),
        evidence={
            "path": relative_path,
            "expected_tasks_sha256": expected_tasks_digest,
            "observed_task_ids": sorted(observed_ids),
            "issues": issues,
        },
        remediation="Resolve every task-level review field and bind the reviewed task-set digest.",
    )


def _reference_solution_content_check(
    *,
    manifest: DeepResearchBenchmarkManifest,
    manifest_path: Path,
    relative_path: str | None,
    declared_coverage: float | None,
    minimum_coverage: float,
    required: bool,
) -> ReadinessCheck:
    payload = _load_bound_json_object(manifest_path, relative_path)
    if payload is None:
        return _check(
            "design.reference_solution_content",
            passed=None if required else True,
            required=required,
            summary="Reference-solution content is unavailable.",
            evidence={"path": relative_path},
            remediation=(
                "Bind reference outputs for the exact task set and verify them against executable "
                "task contracts."
            ),
        )
    entries = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    entries_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not str(item.get("task_id") or "").strip():
            continue
        task_id = str(item["task_id"])
        if task_id in entries_by_id:
            duplicate_ids.add(task_id)
        entries_by_id[task_id] = item

    structural_issues: list[str] = []
    if payload.get("schema_version") != "agenttracelab.deepresearch-reference-solutions.v1":
        structural_issues.append("schema_version")
    if payload.get("benchmark_id") != manifest.benchmark_id:
        structural_issues.append("benchmark_id")
    if payload.get("benchmark_version") != manifest.version:
        structural_issues.append("benchmark_version")
    if duplicate_ids:
        structural_issues.append("duplicate_task_ids")
    unexpected_ids = sorted(set(entries_by_id) - {task.task_id for task in manifest.tasks})
    if unexpected_ids:
        structural_issues.append("unexpected_task_ids")

    passed_task_ids: list[str] = []
    task_results: list[dict[str, Any]] = []
    for task in manifest.tasks:
        entry = entries_by_id.get(task.task_id)
        if entry is None:
            task_results.append({"task_id": task.task_id, "passed": False, "reason": "missing"})
            continue
        if entry.get("expected_status") != task.expected_status:
            task_results.append(
                {"task_id": task.task_id, "passed": False, "reason": "expected_status_mismatch"}
            )
            continue
        if task.expected_status == "needs_user_input":
            no_answer = not str(entry.get("report") or "").strip()
            no_evidence = not (entry.get("evidence") or [])
            passed = no_answer and no_evidence
            reason = "safe_clarification_reference" if passed else "unexpected_answer_or_evidence"
        else:
            acceptance = _evaluate_task_acceptance(
                task,
                {
                    "report": entry.get("report"),
                    "evidence": entry.get("evidence") or [],
                },
            )
            passed = acceptance["passed"] is True
            reason = "accepted" if passed else ",".join(acceptance["failures"])
        if passed:
            passed_task_ids.append(task.task_id)
        task_results.append({"task_id": task.task_id, "passed": passed, "reason": reason})

    executable_coverage = len(passed_task_ids) / len(manifest.tasks)
    declared_not_overstated = declared_coverage is None or declared_coverage <= executable_coverage
    coverage_passed = executable_coverage >= minimum_coverage
    passed = not structural_issues and coverage_passed and declared_not_overstated
    return _check(
        "design.reference_solution_content",
        passed=passed,
        required=required,
        summary=(
            "Reference solutions pass executable task contracts at the required coverage."
            if passed
            else "Reference solutions do not support the declared or required coverage."
        ),
        evidence={
            "path": relative_path,
            "declared": declared_coverage,
            "executable": round(executable_coverage, 4),
            "minimum": minimum_coverage,
            "passed_task_ids": passed_task_ids,
            "unexpected_task_ids": unexpected_ids,
            "duplicate_task_ids": sorted(duplicate_ids),
            "structural_issues": structural_issues,
            "task_results": task_results,
        },
        remediation=(
            "Add reviewed reference outputs that satisfy each task's acceptance contract, and do "
            "not declare more coverage than the executable references prove."
        ),
    )


def _judge_calibration_content_check(
    *,
    report: DeepResearchBenchmarkReport,
    calibration_id: str | None,
    manifest_path: Path,
    relative_path: str | None,
    required: bool,
) -> ReadinessCheck:
    payload = _load_bound_json_object(manifest_path, relative_path)
    if payload is None:
        return _check(
            "quality.judge_calibration_content",
            passed=None if required else True,
            required=required,
            summary="Judge calibration content is unavailable.",
            evidence={"path": relative_path},
            remediation="Bind a passing calibration report for the exact Judge provider and model.",
        )
    judges = payload.get("judges") if isinstance(payload.get("judges"), list) else []
    matching = [
        item
        for item in judges
        if isinstance(item, dict)
        and item.get("provider") == report.claim_judge_provider
        and item.get("model") == report.claim_judge_model
        and item.get("prompt_version") == report.claim_judge_prompt_version
    ]
    issues: list[str] = []
    if payload.get("schema_version") != "agenttracelab.judge-calibration.v1":
        issues.append("schema_version")
    if payload.get("dataset_id") != calibration_id:
        issues.append("calibration_id")
    if payload.get("passed") is not True:
        issues.append("report_gate")
    if len(matching) != 1 or matching[0].get("passed") is not True:
        issues.append("judge_identity_or_gate")
    return _check(
        "quality.judge_calibration_content",
        passed=not issues,
        required=required,
        summary=(
            "Calibration report passes for the exact Judge identity."
            if not issues
            else "Calibration report does not pass for the exact Judge identity."
        ),
        evidence={
            "path": relative_path,
            "provider": report.claim_judge_provider,
            "model": report.claim_judge_model,
            "prompt_version": report.claim_judge_prompt_version,
            "matching_judges": len(matching),
            "issues": issues,
        },
        remediation="Calibrate and pass the exact Judge provider, model, and rubric version.",
    )


def audit_deepresearch_evidence(
    manifest_path: str | Path,
    report_path: str | Path,
    policy_path: str | Path,
) -> EvaluationReadinessReport:
    manifest, resolved_manifest, manifest_digest = load_deepresearch_benchmark_manifest(manifest_path)
    report_payload, _report_raw, report_digest = _load_json(Path(report_path), label="benchmark report")
    report = DeepResearchBenchmarkReport.model_validate(report_payload)
    policy, policy_digest = load_evaluation_readiness_policy(policy_path)
    design = manifest.design_evidence
    checks: list[ReadinessCheck] = []

    manifest_match = report.manifest_sha256 == manifest_digest
    checks.append(
        _check(
            "integrity.manifest_digest",
            passed=manifest_match,
            required=policy.require_manifest_integrity,
            summary=(
                "The source manifest matches the immutable digest recorded by the report."
                if manifest_match
                else "The source manifest no longer matches the report digest."
            ),
            evidence={"recorded": report.manifest_sha256, "actual": manifest_digest},
            remediation="Audit the exact immutable manifest used for the run.",
        )
    )
    consistent_fields = {
        "benchmark_id": (manifest.benchmark_id, report.benchmark_id),
        "benchmark_version": (manifest.version, report.benchmark_version),
        "evidence_class": (manifest.evidence_class, report.evidence_class),
        "target_name": (manifest.target_name, report.target_name),
        "target_revision": (manifest.target_revision, report.target_revision),
        "target_provider": (manifest.target_provider, report.target_provider),
        "target_model": (manifest.target_model, report.target_model),
        "task_count": (len(manifest.tasks), report.task_count),
        "trials_per_task": (manifest.trials_per_task, report.trials_per_task),
    }
    mismatches = {
        name: {"manifest": expected, "report": actual}
        for name, (expected, actual) in consistent_fields.items()
        if expected != actual
    }
    checks.append(
        _check(
            "integrity.report_manifest_consistency",
            passed=not mismatches,
            required=True,
            summary=(
                "Report metadata matches the source manifest."
                if not mismatches
                else "Report metadata differs from the source manifest."
            ),
            evidence={"mismatches": mismatches},
            remediation="Regenerate the report from the recorded manifest without editing either artifact.",
        )
    )

    expected_pairs = {
        (task.task_id, trial) for task in manifest.tasks for trial in range(1, manifest.trials_per_task + 1)
    }
    observed_pairs = [(trial.task_id, trial.trial) for trial in report.trials]
    duplicate_pairs = sorted({pair for pair in observed_pairs if observed_pairs.count(pair) > 1})
    missing_pairs = sorted(expected_pairs - set(observed_pairs))
    unexpected_pairs = sorted(set(observed_pairs) - expected_pairs)
    matrix_ok = not duplicate_pairs and not missing_pairs and not unexpected_pairs
    checks.append(
        _check(
            "integrity.trial_matrix",
            passed=matrix_ok,
            required=True,
            summary=(
                "Every planned task/trial pair appears exactly once."
                if matrix_ok
                else "The task/trial matrix is incomplete or duplicated."
            ),
            evidence={
                "expected": len(expected_pairs),
                "observed": len(observed_pairs),
                "duplicates": duplicate_pairs,
                "missing": missing_pairs,
                "unexpected": unexpected_pairs,
            },
            remediation="Re-run the immutable suite and retain one result per planned task/trial pair.",
        )
    )

    checks.extend(
        [
            _check(
                "execution.report_completed",
                passed=report.status == "completed",
                required=True,
                summary=f"Benchmark status is {report.status}.",
                evidence={"status": report.status},
                remediation="Resolve infrastructure failures and rerun every planned trial.",
            ),
            _check(
                "execution.completion_rate",
                passed=report.metrics.completion_rate >= policy.min_completion_rate,
                required=True,
                summary=_threshold_summary(
                    "Trial completion rate",
                    actual=report.metrics.completion_rate,
                    minimum=policy.min_completion_rate,
                ),
                evidence={
                    "actual": report.metrics.completion_rate,
                    "minimum": policy.min_completion_rate,
                },
                remediation="Do not infer Agent quality from dropped, timed-out, or missing trials.",
            ),
        ]
    )
    execution_errors = sum(trial.error_type is not None for trial in report.trials)
    execution_error_rate = execution_errors / report.metrics.total_trials
    checks.append(
        _check(
            "execution.error_rate",
            passed=execution_error_rate <= policy.max_execution_error_rate,
            required=True,
            summary=_maximum_summary(
                "Execution error rate",
                actual=execution_error_rate,
                maximum=policy.max_execution_error_rate,
            ),
            evidence={"actual": execution_error_rate, "maximum": policy.max_execution_error_rate},
            remediation="Separate infrastructure failures from Agent failures and rerun affected trials.",
        )
    )

    checks.extend(
        [
            _check(
                "statistics.task_count",
                passed=report.task_count >= policy.min_task_count,
                required=True,
                summary=_threshold_summary(
                    "Task count",
                    actual=report.task_count,
                    minimum=policy.min_task_count,
                ),
                evidence={"actual": report.task_count, "minimum": policy.min_task_count},
                remediation="Add representative, independently reviewable tasks before release claims.",
            ),
            _check(
                "statistics.trials_per_task",
                passed=report.trials_per_task >= policy.min_trials_per_task,
                required=True,
                summary=_threshold_summary(
                    "Trials per task",
                    actual=report.trials_per_task,
                    minimum=policy.min_trials_per_task,
                ),
                evidence={
                    "actual": report.trials_per_task,
                    "minimum": policy.min_trials_per_task,
                },
                remediation="Repeat each task enough times to expose stochastic reliability.",
            ),
            _check(
                "statistics.pass_rate_lower_bound",
                passed=(
                    report.metrics.pass_rate_wilson_lower_bound_95 >= policy.min_pass_rate_wilson_lower_bound
                ),
                required=True,
                summary=_threshold_summary(
                    "Conservative pass-rate bound",
                    actual=report.metrics.pass_rate_wilson_lower_bound_95,
                    minimum=policy.min_pass_rate_wilson_lower_bound,
                ),
                evidence={
                    "actual": report.metrics.pass_rate_wilson_lower_bound_95,
                    "minimum": policy.min_pass_rate_wilson_lower_bound,
                },
                remediation="Improve the Agent or gather more successful independent trials.",
            ),
            _check(
                "statistics.evidence_supported_pass_rate_lower_bound",
                passed=(
                    None
                    if report.metrics.evidence_supported_pass_rate_lower_bound is None
                    else report.metrics.evidence_supported_pass_rate_lower_bound
                    >= policy.min_evidence_supported_pass_rate_lower_bound
                ),
                required=policy.min_evidence_supported_pass_rate_lower_bound > 0,
                summary=_threshold_summary(
                    "Evidence-supported pass-rate lower bound",
                    actual=report.metrics.evidence_supported_pass_rate_lower_bound,
                    minimum=policy.min_evidence_supported_pass_rate_lower_bound,
                ),
                evidence={
                    "actual": report.metrics.evidence_supported_pass_rate_lower_bound,
                    "upper_bound": report.metrics.evidence_supported_pass_rate_upper_bound,
                    "minimum": policy.min_evidence_supported_pass_rate_lower_bound,
                    "verified_passes": report.metrics.evidence_verified_passed_trials,
                    "unverified_potential_passes": (report.metrics.evidence_unverified_potential_passes),
                },
                remediation=(
                    "Run the semantic Judge or human adjudication; do not promote "
                    "unverified potential passes into the release score."
                ),
            ),
        ]
    )

    checks.append(
        _check(
            "evidence.class",
            passed=report.evidence_class in policy.allowed_evidence_classes,
            required=True,
            summary=(
                f"Evidence class {report.evidence_class} is allowed for the intended use."
                if report.evidence_class in policy.allowed_evidence_classes
                else f"Evidence class {report.evidence_class} is not allowed for the intended use."
            ),
            evidence={
                "actual": report.evidence_class,
                "allowed": policy.allowed_evidence_classes,
            },
            remediation="Use live-system evidence for claims about deployed Agent quality.",
        )
    )
    revision_known = bool(report.target_revision)
    checks.append(
        _check(
            "reproducibility.target_revision",
            passed=revision_known if policy.require_target_revision else True,
            required=policy.require_target_revision,
            summary=("Target revision is pinned." if revision_known else "Target revision is unspecified."),
            evidence={"target_revision": report.target_revision},
            remediation="Record an immutable commit, image digest, or release identifier.",
        )
    )
    pinned_judge_identity = (
        manifest.expected_claim_judge_provider,
        manifest.expected_claim_judge_model,
        manifest.expected_claim_judge_prompt_version,
    )
    observed_judge_identity = (
        report.claim_judge_provider,
        report.claim_judge_model,
        report.claim_judge_prompt_version,
    )
    judge_identity_known = all(pinned_judge_identity)
    judge_identity_matches = judge_identity_known and pinned_judge_identity == observed_judge_identity
    checks.append(
        _check(
            "reproducibility.claim_judge_identity",
            passed=judge_identity_matches if policy.require_pinned_claim_judge_identity else True,
            required=policy.require_pinned_claim_judge_identity,
            summary=(
                "Claim Judge provider, model, and prompt version match the manifest."
                if judge_identity_matches
                else "Claim Judge identity is unpinned or differs from the manifest."
            ),
            evidence={
                "manifest": pinned_judge_identity,
                "report": observed_judge_identity,
            },
            remediation=(
                "Pin the Claim Judge provider, model, and prompt version before execution, then "
                "calibrate that exact identity."
            ),
        )
    )

    separation_required_rank = {
        "not_configured": 0,
        "different_model": 1,
        "different_provider": 2,
    }[policy.min_judge_separation]
    checks.extend(
        [
            _check(
                "quality.semantic_coverage",
                passed=(
                    report.metrics.semantic_evaluation_coverage >= policy.min_semantic_evaluation_coverage
                ),
                required=policy.min_semantic_evaluation_coverage > 0,
                summary=_threshold_summary(
                    "Semantic grading coverage",
                    actual=report.metrics.semantic_evaluation_coverage,
                    minimum=policy.min_semantic_evaluation_coverage,
                ),
                evidence={
                    "actual": report.metrics.semantic_evaluation_coverage,
                    "minimum": policy.min_semantic_evaluation_coverage,
                },
                remediation="Grade every release trial or preserve an explicit unverified outcome.",
            ),
            _check(
                "quality.task_acceptance_coverage",
                passed=(
                    None
                    if report.metrics.task_acceptance_evaluation_coverage is None
                    else report.metrics.task_acceptance_evaluation_coverage
                    >= policy.min_task_acceptance_evaluation_coverage
                ),
                required=policy.min_task_acceptance_evaluation_coverage > 0,
                summary=_threshold_summary(
                    "Task-acceptance contract coverage",
                    actual=report.metrics.task_acceptance_evaluation_coverage,
                    minimum=policy.min_task_acceptance_evaluation_coverage,
                ),
                evidence={
                    "actual": report.metrics.task_acceptance_evaluation_coverage,
                    "minimum": policy.min_task_acceptance_evaluation_coverage,
                    "evaluated_trials": report.metrics.task_acceptance_evaluated_trials,
                    "passed_trials": report.metrics.task_acceptance_passed_trials,
                },
                remediation=(
                    "Attach an executable task-acceptance contract to every answerable task; "
                    "claim support alone does not prove task completion."
                ),
            ),
            _check(
                "quality.judge_separation",
                passed=JUDGE_SEPARATION_RANK[report.judge_separation] >= separation_required_rank,
                required=separation_required_rank > 0,
                summary=(
                    "Judge separation meets policy."
                    if JUDGE_SEPARATION_RANK[report.judge_separation] >= separation_required_rank
                    else "Judge separation is weaker than policy."
                ),
                evidence={
                    "actual": report.judge_separation,
                    "minimum": policy.min_judge_separation,
                },
                remediation="Use a different model or provider and calibrate it against human labels.",
            ),
        ]
    )

    completed_trace_ids = [trial.trace_id for trial in report.trials if trial.completed]
    trace_ids_ok = all(completed_trace_ids) and len(completed_trace_ids) == len(set(completed_trace_ids))
    checks.append(
        _check(
            "trace.unique_completed_ids",
            passed=trace_ids_ok,
            required=policy.require_unique_trace_ids,
            summary=(
                "Completed trials have distinct trace IDs."
                if trace_ids_ok
                else "Completed trials have missing or reused trace IDs."
            ),
            evidence={"completed": len(completed_trace_ids), "unique": len(set(completed_trace_ids))},
            remediation="Require one fresh trace per independent trial.",
        )
    )

    distinct_expected_statuses = sorted({task.expected_status for task in manifest.tasks})
    checks.append(
        _check(
            "design.expected_status_balance",
            passed=(len(distinct_expected_statuses) >= policy.min_distinct_expected_statuses),
            required=policy.min_distinct_expected_statuses > 1,
            summary=_threshold_summary(
                "Distinct expected outcome classes",
                actual=len(distinct_expected_statuses),
                minimum=policy.min_distinct_expected_statuses,
            ),
            evidence={
                "actual": distinct_expected_statuses,
                "count": len(distinct_expected_statuses),
                "minimum_count": policy.min_distinct_expected_statuses,
            },
            remediation=(
                "Include both answerable tasks and tasks that should safely request "
                "clarification without retrieval."
            ),
        )
    )

    answerable_tasks = [task for task in manifest.tasks if task.expected_status == "completed"]
    contracted_tasks = [task for task in answerable_tasks if task.acceptance_contract is not None]
    contract_coverage = len(contracted_tasks) / len(answerable_tasks) if answerable_tasks else 1.0
    checks.append(
        _check(
            "design.task_acceptance_contract_coverage",
            passed=(contract_coverage >= policy.min_task_acceptance_evaluation_coverage),
            required=policy.min_task_acceptance_evaluation_coverage > 0,
            summary=_threshold_summary(
                "Answerable-task acceptance-contract coverage",
                actual=round(contract_coverage, 4),
                minimum=policy.min_task_acceptance_evaluation_coverage,
            ),
            evidence={
                "answerable_tasks": len(answerable_tasks),
                "contracted_tasks": len(contracted_tasks),
                "coverage": round(contract_coverage, 4),
            },
            remediation=("Define versioned executable completeness checks for every answerable task."),
        )
    )
    clarification_tasks = [task for task in manifest.tasks if task.expected_status == "needs_user_input"]
    executable_grader_task_ids = {
        *(task.task_id for task in contracted_tasks),
        *(task.task_id for task in clarification_tasks),
    }
    executable_outcome_grader_coverage = len(executable_grader_task_ids) / len(manifest.tasks)
    declared_outcome_grader_coverage = design.outcome_grader_coverage
    declared_coverage_not_overstated = (
        declared_outcome_grader_coverage is None
        or declared_outcome_grader_coverage <= executable_outcome_grader_coverage
    )
    executable_coverage_meets_policy = (
        executable_outcome_grader_coverage >= policy.min_outcome_grader_coverage
    )
    checks.append(
        _check(
            "design.executable_outcome_grader_coverage",
            passed=executable_coverage_meets_policy and declared_coverage_not_overstated,
            required=policy.min_outcome_grader_coverage > 0,
            summary=(
                "Executable outcome-grader coverage supports the declared coverage."
                if executable_coverage_meets_policy and declared_coverage_not_overstated
                else "Declared or required outcome-grader coverage exceeds executable coverage."
            ),
            evidence={
                "declared": declared_outcome_grader_coverage,
                "executable": round(executable_outcome_grader_coverage, 4),
                "minimum": policy.min_outcome_grader_coverage,
                "graded_task_ids": sorted(executable_grader_task_ids),
                "ungraded_task_ids": sorted(
                    task.task_id for task in manifest.tasks if task.task_id not in executable_grader_task_ids
                ),
            },
            remediation=(
                "Add executable acceptance checks or safe-clarification outcome rules; do not "
                "claim grader coverage from metadata alone."
            ),
        )
    )

    task_contracts = {
        task.task_id: task.acceptance_contract
        for task in contracted_tasks
        if task.acceptance_contract is not None
    }
    binding_mismatches: list[dict[str, Any]] = []
    bound_trials = 0
    for trial in report.trials:
        contract = task_contracts.get(trial.task_id)
        if contract is None:
            continue
        bound_trials += 1
        expected_digest = task_acceptance_contract_sha256(contract)
        if (
            trial.task_acceptance_contract_id != contract.contract_id
            or trial.task_acceptance_contract_sha256 != expected_digest
        ):
            binding_mismatches.append(
                {
                    "task_id": trial.task_id,
                    "trial": trial.trial,
                    "expected_contract_id": contract.contract_id,
                    "actual_contract_id": trial.task_acceptance_contract_id,
                    "expected_sha256": expected_digest,
                    "actual_sha256": trial.task_acceptance_contract_sha256,
                }
            )
    binding_required = policy.min_task_acceptance_evaluation_coverage > 0
    binding_passed: bool | None = not binding_mismatches
    if binding_required and not bound_trials:
        binding_passed = None
    checks.append(
        _check(
            "integrity.task_acceptance_contract_binding",
            passed=binding_passed,
            required=binding_required,
            summary=(
                "Every contracted trial is bound to the manifest contract digest."
                if binding_passed is True
                else (
                    "No contracted trial binding is available."
                    if binding_passed is None
                    else "Trial acceptance-contract bindings differ from the manifest."
                )
            ),
            evidence={
                "bound_trials": bound_trials,
                "expected_trials": len(contracted_tasks) * manifest.trials_per_task,
                "mismatches": binding_mismatches,
            },
            remediation=(
                "Rerun the immutable suite so every trial records the exact acceptance-contract digest."
            ),
        )
    )
    artifact = _artifact_check(report, Path(report_path).resolve())
    if not policy.require_retained_artifacts:
        artifact = artifact.model_copy(update={"severity": "advisory"})
    checks.append(artifact)

    suite_required = bool(policy.required_suite_kinds)
    suite_known = design.suite_kind != "unspecified"
    suite_passed = (
        design.suite_kind in policy.required_suite_kinds if suite_required and suite_known else None
    )
    if not suite_required:
        suite_passed = True
    checks.append(
        _check(
            "design.suite_kind",
            passed=suite_passed,
            required=suite_required,
            summary=f"Suite kind is {design.suite_kind}.",
            evidence={"actual": design.suite_kind, "allowed": policy.required_suite_kinds},
            remediation="Declare whether this is a capability, regression, or mixed suite.",
        )
    )

    required_review_rank = {
        "none": 0,
        "unreviewed": 1,
        "generated-synthetic": 2,
        "human-reviewed": 3,
    }[policy.required_dataset_review_status]
    review_known = design.dataset_review_status != "unspecified"
    review_passed: bool | None
    if required_review_rank == 0:
        review_passed = True
    elif not review_known:
        review_passed = None
    else:
        review_passed = _ranked_review_status(design.dataset_review_status) >= required_review_rank
    checks.append(
        _check(
            "design.dataset_review",
            passed=review_passed,
            required=required_review_rank > 0,
            summary=f"Dataset review status is {design.dataset_review_status}.",
            evidence={
                "actual": design.dataset_review_status,
                "minimum": policy.required_dataset_review_status,
                "task_source": design.task_source,
            },
            remediation="Bind the suite to documented human review instead of generated labels alone.",
        )
    )
    checks.append(
        _bound_design_artifact_check(
            check_id="design.dataset_review_artifact",
            label="Dataset review",
            manifest_path=resolved_manifest,
            relative_path=design.dataset_review_artifact_path,
            expected_sha256=design.dataset_review_artifact_sha256,
            required=policy.require_dataset_review_artifact,
        )
    )
    checks.append(
        _dataset_review_content_check(
            manifest=manifest,
            manifest_path=resolved_manifest,
            relative_path=design.dataset_review_artifact_path,
            required=policy.require_dataset_review_artifact,
        )
    )
    checks.append(
        _bound_design_artifact_check(
            check_id="design.reference_solution_artifact",
            label="Reference solution",
            manifest_path=resolved_manifest,
            relative_path=design.reference_solution_artifact_path,
            expected_sha256=design.reference_solution_artifact_sha256,
            required=policy.require_reference_solution_artifact,
        )
    )
    checks.append(
        _reference_solution_content_check(
            manifest=manifest,
            manifest_path=resolved_manifest,
            relative_path=design.reference_solution_artifact_path,
            declared_coverage=design.reference_solution_coverage,
            minimum_coverage=policy.min_reference_solution_coverage,
            required=policy.require_reference_solution_artifact,
        )
    )

    def declared_threshold_check(
        check_id: str,
        actual: float | int | None,
        minimum: float | int,
        *,
        summary_name: str,
        remediation: str,
    ) -> ReadinessCheck:
        required = minimum > 0
        if not required and actual is None:
            return ReadinessCheck(
                check_id=check_id,
                status="not_applicable",
                severity="advisory",
                summary=f"{summary_name} is not required by this policy.",
                evidence={"actual": None, "minimum": minimum},
                remediation=None,
            )
        passed = True if not required else (None if actual is None else actual >= minimum)
        return _check(
            check_id,
            passed=passed,
            required=required,
            summary=_threshold_summary(
                summary_name,
                actual=actual,
                minimum=minimum,
            ),
            evidence={"actual": actual, "minimum": minimum},
            remediation=remediation,
        )

    checks.extend(
        [
            declared_threshold_check(
                "design.reference_solution_coverage",
                design.reference_solution_coverage,
                policy.min_reference_solution_coverage,
                summary_name="Reference-solution coverage",
                remediation="Verify tasks and graders with known passing reference solutions.",
            ),
            declared_threshold_check(
                "design.outcome_grader_coverage",
                design.outcome_grader_coverage,
                policy.min_outcome_grader_coverage,
                summary_name="Outcome-state grader coverage",
                remediation="Verify final environment or artifact state, not only model text or tool paths.",
            ),
            declared_threshold_check(
                "design.transcript_review_count",
                design.transcript_review_count,
                policy.min_transcript_review_count,
                summary_name="Manually reviewed transcript count",
                remediation="Read a policy-sized sample of complete traces and record adjudication.",
            ),
            declared_threshold_check(
                "design.production_case_count",
                design.production_case_count,
                policy.min_production_case_count,
                summary_name="Production-derived case count",
                remediation="Promote representative production failures into the offline suite.",
            ),
        ]
    )
    calibration_known = bool(design.judge_calibration_id)
    checks.append(
        _check(
            "quality.judge_calibration",
            passed=calibration_known if policy.require_judge_calibration else True,
            required=policy.require_judge_calibration,
            summary=(
                "Judge calibration evidence is pinned."
                if calibration_known
                else "Judge calibration evidence is unspecified."
            ),
            evidence={"calibration_id": design.judge_calibration_id},
            remediation="Calibrate the exact Judge and rubric version against human labels.",
        )
    )
    checks.append(
        _bound_design_artifact_check(
            check_id="quality.judge_calibration_artifact",
            label="Judge calibration",
            manifest_path=resolved_manifest,
            relative_path=design.judge_calibration_artifact_path,
            expected_sha256=design.judge_calibration_artifact_sha256,
            required=policy.require_judge_calibration_artifact,
        )
    )
    checks.append(
        _judge_calibration_content_check(
            report=report,
            calibration_id=design.judge_calibration_id,
            manifest_path=resolved_manifest,
            relative_path=design.judge_calibration_artifact_path,
            required=policy.require_judge_calibration_artifact,
        )
    )

    isolation_required = bool(policy.allowed_environment_isolation)
    isolation_known = design.environment_isolation != "unspecified"
    isolation_passed = (
        design.environment_isolation in policy.allowed_environment_isolation
        if isolation_required and isolation_known
        else None
    )
    if not isolation_required:
        isolation_passed = True
    checks.extend(
        [
            _check(
                "environment.isolation",
                passed=isolation_passed,
                required=isolation_required,
                summary=f"Environment isolation is {design.environment_isolation}.",
                evidence={
                    "actual": design.environment_isolation,
                    "allowed": policy.allowed_environment_isolation,
                },
                remediation="Reset every trial in a documented isolated environment.",
            ),
            _check(
                "environment.fingerprint",
                passed=bool(design.environment_fingerprint)
                if policy.require_environment_fingerprint
                else True,
                required=policy.require_environment_fingerprint,
                summary=(
                    "Environment fingerprint is recorded."
                    if design.environment_fingerprint
                    else "Environment fingerprint is unspecified."
                ),
                evidence={"fingerprint": design.environment_fingerprint},
                remediation="Pin the container/image/runtime fingerprint used by every trial.",
            ),
            _check(
                "environment.resource_profile",
                passed=bool(design.resource_profile) if policy.require_resource_profile else True,
                required=policy.require_resource_profile,
                summary=(
                    "Resource profile is recorded."
                    if design.resource_profile
                    else "Resource profile is unspecified."
                ),
                evidence={"resource_profile": design.resource_profile},
                remediation="Record CPU, memory, concurrency, timeout, and network constraints.",
            ),
        ]
    )

    required_checks = [check for check in checks if check.severity == "required"]
    required_failed = [check for check in required_checks if check.status == "fail"]
    required_unknown = [check for check in required_checks if check.status == "unknown"]
    fundamental_prefixes = (
        "integrity.",
        "execution.",
        "evidence.",
        "reproducibility.",
        "trace.",
        "artifacts.",
        "design.",
        "environment.",
    )
    fundamental_block = any(
        check.status in {"fail", "unknown"} and check.check_id.startswith(fundamental_prefixes)
        for check in required_checks
    )
    if not required_failed and not required_unknown:
        recommendation: ReadinessRecommendation = "ready"
    elif fundamental_block:
        recommendation = "diagnostic_only"
    else:
        recommendation = "hold"
    reasons = tuple(
        f"{check.check_id}: {check.status}"
        for check in required_checks
        if check.status in {"fail", "unknown"}
    )
    return EvaluationReadinessReport(
        audit_id=f"readiness_{report_digest[:12]}_{policy_digest[:12]}",
        created_at=datetime.now(UTC),
        benchmark_id=report.benchmark_id,
        benchmark_version=report.benchmark_version,
        policy_id=policy.policy_id,
        intended_use=policy.intended_use,
        manifest_sha256=manifest_digest,
        report_sha256=report_digest,
        policy_sha256=policy_digest,
        recommendation=recommendation,
        ready=recommendation == "ready",
        required_passed=sum(check.status == "pass" for check in required_checks),
        required_failed=len(required_failed),
        required_unknown=len(required_unknown),
        checks=tuple(checks),
        decision_reasons=reasons,
        limitations=(
            "The audit verifies retained artifacts and version-bound declarations; it cannot prove "
            "that a self-declared review or environment claim is truthful.",
            "A ready verdict means this policy's evidence requirements passed, not that the "
            "evaluated Agent is universally safe or capable.",
        ),
    )


def render_evaluation_readiness_markdown(report: EvaluationReadinessReport) -> str:
    lines = [
        f"# Evaluation readiness: {report.benchmark_id} {report.benchmark_version}",
        "",
        f"- Intended use: `{report.intended_use}`",
        f"- Policy: `{report.policy_id}`",
        f"- Recommendation: **{report.recommendation}**",
        (
            f"- Required checks: {report.required_passed} passed, "
            f"{report.required_failed} failed, {report.required_unknown} unknown"
        ),
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Summary |",
        "|---|---|---|---|",
    ]
    for check in report.checks:
        lines.append(f"| `{check.check_id}` | {check.status} | {check.severity} | {check.summary} |")
    if report.decision_reasons:
        lines.extend(["", "## Blocking reasons", ""])
        lines.extend(f"- {reason}" for reason in report.decision_reasons)
    remediations = tuple(
        dict.fromkeys(
            check.remediation
            for check in report.checks
            if check.status in {"fail", "unknown"} and check.remediation
        )
    )
    if remediations:
        lines.extend(["", "## Required remediation", ""])
        lines.extend(f"- {item}" for item in remediations)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"
