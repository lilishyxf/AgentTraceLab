from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from agenttracelab.calibration_v2 import (
    AdjudicatedLabel,
    CalibrationCaseV2,
    CalibrationManifestV2,
    CalibrationPolicyV2,
    CalibrationSplit,
    HumanAnnotation,
    JudgeIdentity,
    RecordedJudgeOutcomeV2,
    analyze_judge_calibration_v2,
)
from agenttracelab.cli import main
from agenttracelab.final_state import (
    EnvironmentSnapshot,
    FileState,
    FinalStateContract,
    verify_final_state,
)
from agenttracelab.judging import JudgeDimensionName, JudgeVerdict
from agenttracelab.runtime_faults import (
    OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256,
    OfflineRuntimeFaultRunner,
    RuntimeFaultManifest,
    RuntimeFaultType,
    run_runtime_fault_campaign,
)
from agenttracelab.trust_readiness import audit_trust_evidence


def _dimensions() -> dict[JudgeDimensionName, int]:
    return {dimension: 3 for dimension in JudgeDimensionName}


def _write_json(path: Path, payload: object) -> str:
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return sha256(path.read_bytes()).hexdigest()


def _write_complete_evidence(
    root: Path,
    *,
    scope: str = "offline_harness",
    review_status: str = "human_reviewed",
) -> Path:
    root.mkdir(parents=True)
    identity = JudgeIdentity(
        provider="offline-fixture",
        model="deterministic-rules-v1",
        prompt_version="trajectory-rubric.v2",
    )
    now = datetime(2026, 10, 5, tzinfo=UTC)

    def calibration_case(case_id: str, split: CalibrationSplit) -> CalibrationCaseV2:
        annotations = (
            HumanAnnotation(
                annotation_id=f"{case_id}-a1",
                annotator_id="annotator-1",
                verdict=JudgeVerdict.PASS,
                dimensions=_dimensions(),
                created_at=now,
            ),
            HumanAnnotation(
                annotation_id=f"{case_id}-a2",
                annotator_id="annotator-2",
                verdict=JudgeVerdict.PASS,
                dimensions=_dimensions(),
                created_at=now,
            ),
        )
        return CalibrationCaseV2(
            case_id=case_id,
            trace_id=f"trace-{case_id}",
            split=split,
            annotations=annotations,
            adjudicated=AdjudicatedLabel(
                adjudicator_id="adjudicator-1",
                source_annotation_ids=tuple(item.annotation_id for item in annotations),
                verdict=JudgeVerdict.PASS,
                dimensions=_dimensions(),
                rationale="fixture truth",
                adjudicated_at=now,
            ),
            judge_outcomes=(
                RecordedJudgeOutcomeV2(
                    judge_key="judge-primary",
                    identity=identity,
                    status="scored",
                    verdict=JudgeVerdict.PASS,
                    dimensions=_dimensions(),
                ),
            ),
        )

    calibration_manifest = CalibrationManifestV2(
        dataset_id="trust-fixture",
        version="v1",
        policy=CalibrationPolicyV2(
            min_development_cases=1,
            min_holdout_cases=1,
            min_holdout_annotators=2,
        ),
        judge_identities={"judge-primary": identity},
        cases=(
            calibration_case("development", CalibrationSplit.DEVELOPMENT),
            calibration_case("holdout", CalibrationSplit.HOLDOUT),
        ),
    )
    calibration_report = analyze_judge_calibration_v2(calibration_manifest)

    file_state = FileState(sha256="b" * 64, size_bytes=4, text="seed")
    snapshot = EnvironmentSnapshot(
        adapter_kind="filesystem",
        environment_id="environment-1",
        seed="seed-1",
        fixture_sha256="c" * 64,
        snapshot_sha256="d" * 64,
        workspace_path=str(root / "workspace"),
        files={"protected.txt": file_state},
    )
    final_contract = FinalStateContract(
        task_id="no-write",
        version="v1",
        adapter_kind="filesystem",
        fixture_reference="fixture/files",
        valid_alternatives=(
            {"alternative_id": "unchanged", "required_predicates": []},
        ),
        protected_predicates=(
            {"operation": "protected_equals", "path": "protected.txt"},
        ),
        max_changed_resources=0,
    )
    final_report = verify_final_state(final_contract, snapshot, snapshot)

    runtime_manifest = RuntimeFaultManifest(
        campaign_id="trust-runtime",
        version="v1",
        target_runner_version="offline-kv-v1",
        environment_snapshot_sha256=OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256,
        repetitions=1,
        faults=tuple(RuntimeFaultType),
        operation_key="status",
        operation_value="done",
    )
    runtime_report = run_runtime_fault_campaign(
        runtime_manifest,
        OfflineRuntimeFaultRunner(runtime_manifest),
    )

    payloads = {
        "calibration_manifest": calibration_manifest.model_dump(mode="json"),
        "calibration_report": calibration_report.model_dump(mode="json"),
        "final_state_contract": final_contract.model_dump(mode="json"),
        "final_state_initial": snapshot.model_dump(mode="json"),
        "final_state_final": snapshot.model_dump(mode="json"),
        "final_state_report": final_report.model_dump(mode="json"),
        "runtime_fault_manifest": runtime_manifest.model_dump(mode="json"),
        "runtime_fault_report": runtime_report.model_dump(mode="json"),
    }
    artifacts: dict[str, dict[str, str]] = {}
    for role, payload in payloads.items():
        path = root / f"{role}.json"
        artifacts[role] = {"path": path.name, "sha256": _write_json(path, payload)}

    manifest = {
        "schema_version": "agenttracelab.trust-evidence-manifest.v1",
        "evidence_id": "offline-trust-fixture",
        "version": "v1",
        "evidence_scope": scope,
        "review_status": review_status,
        "expected_judge": {
            "judge_key": "judge-primary",
            **identity.model_dump(mode="json"),
        },
        "artifacts": artifacts,
        "limitations": ["Offline deterministic harness; not third-party Agent evidence."],
    }
    manifest_path = root / "trust-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_trust_readiness_passes_complete_offline_evidence(tmp_path: Path) -> None:
    manifest = _write_complete_evidence(tmp_path / "evidence")

    report = audit_trust_evidence(manifest)

    assert report.evidence_complete is True
    assert report.required_failed == 0
    assert report.required_unknown == 0
    assert report.recommendation == "diagnostic_only"
    assert any("offline" in item.lower() for item in report.limitations)


def test_trust_readiness_rejects_missing_artifact(tmp_path: Path) -> None:
    manifest = _write_complete_evidence(tmp_path / "evidence")
    (manifest.parent / "final_state_report.json").unlink()

    report = audit_trust_evidence(manifest)

    check = {item.check_id: item for item in report.checks}["artifact.final_state_report"]
    assert check.status == "fail"
    assert report.evidence_complete is False


def test_trust_readiness_rejects_changed_artifact(tmp_path: Path) -> None:
    manifest = _write_complete_evidence(tmp_path / "evidence")
    artifact = manifest.parent / "runtime_fault_report.json"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = audit_trust_evidence(manifest)

    check = {item.check_id: item for item in report.checks}["artifact.runtime_fault_report"]
    assert check.status == "fail"
    assert check.evidence["hash_matches"] is False


def test_trust_readiness_rejects_identity_mismatch(tmp_path: Path) -> None:
    manifest = _write_complete_evidence(tmp_path / "evidence")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["expected_judge"]["model"] = "different-model"
    _write_json(manifest, payload)

    report = audit_trust_evidence(manifest)

    check = {item.check_id: item for item in report.checks}["binding.judge_identity"]
    assert check.status == "fail"
    assert report.recommendation == "diagnostic_only"


def test_trust_readiness_keeps_unreviewed_evidence_diagnostic_only(tmp_path: Path) -> None:
    manifest = _write_complete_evidence(
        tmp_path / "evidence",
        scope="target_integration",
        review_status="unreviewed",
    )

    report = audit_trust_evidence(manifest)

    assert report.evidence_complete is True
    assert report.recommendation == "diagnostic_only"
    assert any("human review" in item.lower() for item in report.decision_reasons)


def test_trust_readiness_cli_writes_immutable_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_complete_evidence(tmp_path / "evidence")
    output = tmp_path / "trust.json"
    summary = tmp_path / "trust.md"
    argv = [
        "agenttracelab",
        "audit-trust-evidence",
        str(manifest),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 19
    assert json.loads(output.read_text(encoding="utf-8"))["evidence_complete"] is True
    assert "Recommendation: **diagnostic_only**" in summary.read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 18
