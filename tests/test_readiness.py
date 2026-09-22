from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from agenttracelab.cli import main
from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkMetrics,
    DeepResearchBenchmarkReport,
    DeepResearchBenchmarkTask,
    DeepResearchTaskSummary,
    DeepResearchTrialResult,
    TaskAcceptanceContract,
    benchmark_tasks_sha256,
    task_acceptance_contract_sha256,
)
from agenttracelab.readiness import audit_deepresearch_evidence


def _write_inputs(
    tmp_path: Path,
    *,
    partial: bool = False,
    complete_design: bool = True,
) -> tuple[Path, Path, Path]:
    acceptance_contract = {
        "contract_id": "case-1-v1",
        "min_report_chars": 10,
        "min_independent_sources": 1,
        "required_concepts": [{"criterion_id": "grounded-answer", "aliases": ["grounded answer"]}],
    }
    acceptance_digest = task_acceptance_contract_sha256(
        TaskAcceptanceContract.model_validate(acceptance_contract)
    )
    task_payload = {
        "task_id": "case-1",
        "prompt": "Produce an evidence-grounded answer.",
        "tags": ["grounding"],
        "acceptance_contract": acceptance_contract,
    }
    reviewed_tasks_digest = benchmark_tasks_sha256((DeepResearchBenchmarkTask.model_validate(task_payload),))
    reference_artifact = tmp_path / "reference-solutions.json"
    reference_artifact.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.deepresearch-reference-solutions.v1",
                "benchmark_id": "readiness-smoke",
                "benchmark_version": "1.0.0",
                "tasks": [
                    {
                        "task_id": "case-1",
                        "expected_status": "completed",
                        "report": "This is a grounded answer.",
                        "evidence": [
                            {
                                "source_id": "source-1",
                                "url": "https://example.com/grounded-answer",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reference_digest = hashlib.sha256(reference_artifact.read_bytes()).hexdigest()
    review_artifact = tmp_path / "dataset-review.json"
    review_artifact.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.dataset-review.v1",
                "review_id": "review-v1",
                "benchmark_id": "readiness-smoke",
                "benchmark_version": "1.0.0",
                "reviewed_tasks_sha256": reviewed_tasks_digest,
                "review_status": "approved",
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-09-22T00:00:00Z",
                "tasks": [
                    {
                        "task_id": "case-1",
                        "prompt_approved": True,
                        "expected_status_approved": True,
                        "acceptance_contract_approved": True,
                        "source_constraints_approved": True,
                        "decision": "approved",
                        "notes": "",
                    }
                ],
                "approved": True,
            }
        ),
        encoding="utf-8",
    )
    review_digest = hashlib.sha256(review_artifact.read_bytes()).hexdigest()
    calibration_artifact = tmp_path / "judge-calibration.json"
    calibration_artifact.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.judge-calibration.v1",
                "dataset_id": "calibration-v1",
                "passed": True,
                "judges": [
                    {
                        "provider": "provider-b",
                        "model": "model-b",
                        "prompt_version": "claim-evidence.v1",
                        "passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration_digest = hashlib.sha256(calibration_artifact.read_bytes()).hexdigest()
    design = (
        {
            "suite_kind": "regression",
            "dataset_review_status": "human-reviewed",
            "task_source": "recorded-production",
            "reference_solution_coverage": 1.0,
            "outcome_grader_coverage": 1.0,
            "transcript_review_count": 5,
            "production_case_count": 5,
            "reference_solution_artifact_path": "reference-solutions.json",
            "reference_solution_artifact_sha256": reference_digest,
            "dataset_review_artifact_path": "dataset-review.json",
            "dataset_review_artifact_sha256": review_digest,
            "judge_calibration_id": "calibration-v1",
            "judge_calibration_artifact_path": "judge-calibration.json",
            "judge_calibration_artifact_sha256": calibration_digest,
            "environment_isolation": "container",
            "environment_fingerprint": "sha256:container-image",
            "resource_profile": "cpu=4,memory=8GiB,concurrency=1,timeout=300s",
        }
        if complete_design
        else {}
    )
    manifest_payload = {
        "schema_version": "agenttracelab.deepresearch-benchmark.v1",
        "benchmark_id": "readiness-smoke",
        "version": "1.0.0",
        "evidence_class": "live-system",
        "target_name": "target-agent",
        "target_revision": "commit-abc",
        "target_provider": "provider-a",
        "target_model": "model-a",
        "expected_claim_judge_provider": "provider-b",
        "expected_claim_judge_model": "model-b",
        "expected_claim_judge_prompt_version": "claim-evidence.v1",
        "minimum_judge_separation": "different_provider",
        "trials_per_task": 1,
        "design_evidence": design,
        "tasks": [task_payload],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    report_root = tmp_path / "result"
    artifact_root = report_root / "trials" / "case-1" / "trial-01"
    artifact_root.mkdir(parents=True)
    relative_artifacts = {
        "snapshot_path": "trials/case-1/trial-01/run-snapshot.json",
        "trace_path": "trials/case-1/trial-01/normalized-trace.json",
        "evaluation_path": "trials/case-1/trial-01/evaluation-report.json",
        "claim_evidence_path": "trials/case-1/trial-01/claim-evidence-report.json",
    }
    for relative in relative_artifacts.values():
        path = report_root / relative
        path.write_text("{}\n", encoding="utf-8")

    completed = not partial
    trial = DeepResearchTrialResult(
        task_id="case-1",
        trial=1,
        run_id="run-1",
        run_status="completed" if completed else "cancelled",
        trace_id="trace-1" if completed else None,
        completed=completed,
        passed=completed,
        process_passed=completed,
        citation_gate_passed=completed,
        semantic_gate_passed=True if completed else None,
        task_acceptance_contract_id="case-1-v1" if completed else None,
        task_acceptance_contract_sha256=acceptance_digest if completed else None,
        task_acceptance_passed=True if completed else None,
        task_acceptance_coverage=1.0 if completed else None,
        quality_status="verified_pass" if completed else "unverified",
        score=100.0 if completed else None,
        failure_categories=() if completed else ("benchmark.execution_error",),
        error_type=None if completed else "TimeoutError",
        error_message=None if completed else "timed out",
        **relative_artifacts,
    )
    metrics = DeepResearchBenchmarkMetrics(
        total_trials=1,
        completed_trials=1 if completed else 0,
        passed_trials=1 if completed else 0,
        process_passed_trials=1 if completed else 0,
        citation_gate_passed_trials=1 if completed else 0,
        semantic_evaluated_trials=1 if completed else 0,
        semantic_gate_passed_trials=1 if completed else 0,
        completion_rate=1.0 if completed else 0.0,
        pass_rate=1.0 if completed else 0.0,
        process_pass_rate=1.0 if completed else 0.0,
        citation_gate_pass_rate=1.0 if completed else 0.0,
        semantic_evaluation_coverage=1.0 if completed else 0.0,
        semantic_gate_pass_rate=1.0 if completed else None,
        task_acceptance_evaluated_trials=1 if completed else 0,
        task_acceptance_passed_trials=1 if completed else 0,
        task_acceptance_evaluation_coverage=1.0 if completed else 0.0,
        task_acceptance_pass_rate=1.0 if completed else None,
        evidence_verified_passed_trials=1 if completed else 0,
        evidence_unverified_potential_passes=0,
        evidence_supported_pass_rate_lower_bound=1.0 if completed else 0.0,
        evidence_supported_pass_rate_upper_bound=1.0 if completed else 0.0,
        pass_rate_wilson_lower_bound_95=1.0 if completed else 0.0,
        pass_at_1_rate=1.0 if completed else 0.0,
        pass_power_k_rate=1.0 if completed else 0.0,
    )
    task = DeepResearchTaskSummary(
        task_id="case-1",
        trials=1,
        completed=1 if completed else 0,
        passed=1 if completed else 0,
        pass_rate=1.0 if completed else 0.0,
        pass_at_1=completed,
        pass_power_k=completed,
        mean_score=100.0 if completed else None,
    )
    now = datetime.now(UTC)
    report = DeepResearchBenchmarkReport(
        benchmark_id="readiness-smoke",
        benchmark_version="1.0.0",
        manifest_sha256=manifest_digest,
        evidence_class="live-system",
        target_name="target-agent",
        target_revision="commit-abc",
        target_provider="provider-a",
        target_model="model-a",
        claim_judge_provider="provider-b",
        claim_judge_model="model-b",
        claim_judge_prompt_version="claim-evidence.v1",
        judge_separation="different_provider",
        target_base_url="http://target.test",
        source_database_file="target.db",
        created_at=now,
        finished_at=now,
        status="completed" if completed else "failed",
        task_count=1,
        trials_per_task=1,
        metrics=metrics,
        task_summaries=(task,),
        trials=(trial,),
        limitations=(),
    )
    report_path = report_root / "benchmark-report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json"), default=str), encoding="utf-8")
    policy_payload = {
        "schema_version": "agenttracelab.evaluation-readiness-policy.v1",
        "policy_id": "strict-smoke",
        "intended_use": "production_release",
        "min_task_count": 1,
        "min_trials_per_task": 1,
        "min_completion_rate": 1.0,
        "max_execution_error_rate": 0.0,
        "min_pass_rate_wilson_lower_bound": 0.9,
        "min_evidence_supported_pass_rate_lower_bound": 0.9,
        "min_semantic_evaluation_coverage": 1.0,
        "min_task_acceptance_evaluation_coverage": 1.0,
        "min_distinct_expected_statuses": 1,
        "allowed_evidence_classes": ["live-system"],
        "min_judge_separation": "different_provider",
        "require_target_revision": True,
        "require_pinned_claim_judge_identity": True,
        "require_manifest_integrity": True,
        "require_unique_trace_ids": True,
        "require_retained_artifacts": True,
        "required_suite_kinds": ["regression"],
        "required_dataset_review_status": "human-reviewed",
        "min_reference_solution_coverage": 1.0,
        "min_outcome_grader_coverage": 1.0,
        "min_transcript_review_count": 1,
        "min_production_case_count": 1,
        "require_judge_calibration": True,
        "require_reference_solution_artifact": True,
        "require_dataset_review_artifact": True,
        "require_judge_calibration_artifact": True,
        "allowed_environment_isolation": ["container", "vm"],
        "require_environment_fingerprint": True,
        "require_resource_profile": True,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
    return manifest_path, report_path, policy_path


def test_readiness_audit_accepts_complete_version_bound_evidence(tmp_path: Path) -> None:
    manifest, report, policy = _write_inputs(tmp_path)

    result = audit_deepresearch_evidence(manifest, report, policy)

    assert result.ready is True
    assert result.recommendation == "ready"
    assert result.required_failed == 0
    assert result.required_unknown == 0
    assert all(check.status in {"pass", "not_applicable"} for check in result.checks)


def test_readiness_audit_marks_partial_unreviewed_evidence_diagnostic_only(
    tmp_path: Path,
) -> None:
    manifest, report, policy = _write_inputs(tmp_path, partial=True, complete_design=False)

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    assert result.ready is False
    assert result.recommendation == "diagnostic_only"
    assert by_id["execution.report_completed"].status == "fail"
    assert by_id["design.dataset_review"].status == "unknown"
    assert by_id["environment.isolation"].status == "unknown"


def test_readiness_audit_detects_manifest_drift(tmp_path: Path) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["description"] = "edited after the benchmark"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_deepresearch_evidence(manifest, report, policy)

    by_id = {check.check_id: check for check in result.checks}
    assert by_id["integrity.manifest_digest"].status == "fail"
    assert result.recommendation == "diagnostic_only"


def test_readiness_audit_detects_unpinned_runtime_judge_change(tmp_path: Path) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["claim_judge_prompt_version"] = "claim-evidence.v2"
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    assert by_id["reproducibility.claim_judge_identity"].status == "fail"
    assert result.recommendation == "diagnostic_only"


def test_readiness_audit_detects_bound_review_artifact_drift(tmp_path: Path) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    (tmp_path / "dataset-review.json").write_text(
        json.dumps({"review_id": "edited-after-review"}), encoding="utf-8"
    )

    result = audit_deepresearch_evidence(manifest, report, policy)

    by_id = {check.check_id: check for check in result.checks}
    assert by_id["design.dataset_review_artifact"].status == "fail"
    assert result.recommendation == "diagnostic_only"


def _rebind_design_artifact(
    manifest: Path,
    report: Path,
    artifact: Path,
    digest_field: str,
) -> None:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["design_evidence"][digest_field] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_payload), encoding="utf-8")


def test_readiness_audit_rejects_rebound_but_unapproved_task_review(tmp_path: Path) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    artifact = tmp_path / "dataset-review.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["tasks"][0]["acceptance_contract_approved"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    _rebind_design_artifact(
        manifest,
        report,
        artifact,
        "dataset_review_artifact_sha256",
    )

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    assert by_id["design.dataset_review_artifact"].status == "pass"
    assert by_id["design.dataset_review_content"].status == "fail"


def test_readiness_audit_rejects_rebound_calibration_for_wrong_prompt_version(
    tmp_path: Path,
) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    artifact = tmp_path / "judge-calibration.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["judges"][0]["prompt_version"] = "claim-evidence.v2"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    _rebind_design_artifact(
        manifest,
        report,
        artifact,
        "judge_calibration_artifact_sha256",
    )

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    assert by_id["quality.judge_calibration_artifact"].status == "pass"
    assert by_id["quality.judge_calibration_content"].status == "fail"


def test_readiness_audit_rejects_rebound_reference_that_fails_task_contract(
    tmp_path: Path,
) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    artifact = tmp_path / "reference-solutions.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["tasks"][0]["report"] = "This answer omits the required concept."
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    _rebind_design_artifact(
        manifest,
        report,
        artifact,
        "reference_solution_artifact_sha256",
    )

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    assert by_id["design.reference_solution_artifact"].status == "pass"
    assert by_id["design.reference_solution_content"].status == "fail"


def test_readiness_audit_blocks_one_sided_suite_and_legacy_score_bounds(
    tmp_path: Path,
) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    policy_payload = json.loads(policy.read_text(encoding="utf-8"))
    policy_payload["min_distinct_expected_statuses"] = 2
    policy.write_text(json.dumps(policy_payload), encoding="utf-8")

    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["metrics"].pop("evidence_supported_pass_rate_lower_bound")
    report_payload["metrics"].pop("evidence_supported_pass_rate_upper_bound")
    report_payload["metrics"].pop("evidence_verified_passed_trials")
    report_payload["metrics"].pop("evidence_unverified_potential_passes")
    report.write_text(json.dumps(report_payload), encoding="utf-8")

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    assert by_id["design.expected_status_balance"].status == "fail"
    assert by_id["statistics.evidence_supported_pass_rate_lower_bound"].status == "unknown"
    assert result.recommendation == "diagnostic_only"


def test_readiness_recomputes_executable_outcome_grader_coverage(tmp_path: Path) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["tasks"][0].pop("acceptance_contract")
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_payload), encoding="utf-8")

    result = audit_deepresearch_evidence(manifest, report, policy)
    by_id = {check.check_id: check for check in result.checks}

    check = by_id["design.executable_outcome_grader_coverage"]
    assert check.status == "fail"
    assert check.evidence["declared"] == 1.0
    assert check.evidence["executable"] == 0.0
    assert check.evidence["ungraded_task_ids"] == ["case-1"]


def test_readiness_cli_writes_machine_and_human_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest, report, policy = _write_inputs(tmp_path)
    output = tmp_path / "readiness.json"
    summary = tmp_path / "readiness.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agenttracelab",
            "audit-deepresearch-evidence",
            str(manifest),
            str(report),
            "--policy",
            str(policy),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["recommendation"] == "ready"
    assert "Recommendation: **ready**" in summary.read_text(encoding="utf-8")
