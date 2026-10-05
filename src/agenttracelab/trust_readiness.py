from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agenttracelab.calibration_v2 import CalibrationManifestV2, CalibrationReportV2
from agenttracelab.final_state import (
    EnvironmentSnapshot,
    FinalStateContract,
    FinalStateReport,
    verify_final_state,
)
from agenttracelab.models import FrozenModel
from agenttracelab.runtime_faults import RuntimeFaultCampaignReport, RuntimeFaultManifest

ArtifactRole = Literal[
    "calibration_manifest",
    "calibration_report",
    "final_state_contract",
    "final_state_initial",
    "final_state_final",
    "final_state_report",
    "runtime_fault_manifest",
    "runtime_fault_report",
]
EXPECTED_ROLES: tuple[ArtifactRole, ...] = (
    "calibration_manifest",
    "calibration_report",
    "final_state_contract",
    "final_state_initial",
    "final_state_final",
    "final_state_report",
    "runtime_fault_manifest",
    "runtime_fault_report",
)


class BoundArtifact(FrozenModel):
    path: str = Field(min_length=1, max_length=1_024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExpectedJudgeIdentity(FrozenModel):
    judge_key: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=128)


class TrustEvidenceManifest(FrozenModel):
    schema_version: Literal["agenttracelab.trust-evidence-manifest.v1"] = (
        "agenttracelab.trust-evidence-manifest.v1"
    )
    evidence_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    evidence_scope: Literal["offline_harness", "target_integration"]
    review_status: Literal["unreviewed", "human_reviewed"]
    expected_judge: ExpectedJudgeIdentity
    artifacts: dict[ArtifactRole, BoundArtifact]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_artifacts(self) -> TrustEvidenceManifest:
        if set(self.artifacts) != set(EXPECTED_ROLES):
            missing = sorted(set(EXPECTED_ROLES) - set(self.artifacts))
            extra = sorted(set(self.artifacts) - set(EXPECTED_ROLES))
            raise ValueError(
                f"trust evidence artifacts must contain exact required roles; "
                f"missing={missing}, extra={extra}"
            )
        return self


class TrustReadinessCheck(FrozenModel):
    check_id: str
    status: Literal["pass", "fail", "unknown"]
    severity: Literal["required", "advisory"]
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class TrustReadinessReport(FrozenModel):
    schema_version: Literal["agenttracelab.trust-readiness-report.v1"] = (
        "agenttracelab.trust-readiness-report.v1"
    )
    evidence_id: str
    evidence_version: str
    manifest_sha256: str
    evidence_complete: bool
    ready: bool
    recommendation: Literal["ready", "diagnostic_only"]
    required_passed: int = Field(ge=0)
    required_failed: int = Field(ge=0)
    required_unknown: int = Field(ge=0)
    checks: tuple[TrustReadinessCheck, ...]
    decision_reasons: tuple[str, ...]
    limitations: tuple[str, ...]


def _canonical_hash(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _check(
    check_id: str,
    status: Literal["pass", "fail", "unknown"],
    summary: str,
    *,
    severity: Literal["required", "advisory"] = "required",
    evidence: dict[str, Any] | None = None,
) -> TrustReadinessCheck:
    return TrustReadinessCheck(
        check_id=check_id,
        status=status,
        severity=severity,
        summary=summary,
        evidence=evidence or {},
    )


def _load_manifest(path: Path) -> tuple[TrustEvidenceManifest, bytes]:
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("trust evidence manifest exceeds 1048576 bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("trust evidence manifest must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("trust evidence manifest root must be a JSON object")
    return TrustEvidenceManifest.model_validate(payload), raw


def _load_bound_artifacts(
    manifest_path: Path,
    manifest: TrustEvidenceManifest,
) -> tuple[list[TrustReadinessCheck], dict[ArtifactRole, dict[str, Any]]]:
    root = manifest_path.parent.resolve()
    checks: list[TrustReadinessCheck] = []
    payloads: dict[ArtifactRole, dict[str, Any]] = {}
    for role in EXPECTED_ROLES:
        binding = manifest.artifacts[role]
        candidate = (root / binding.path).resolve()
        if not candidate.is_relative_to(root):
            checks.append(
                _check(
                    f"artifact.{role}",
                    "fail",
                    "Artifact path escapes the trust manifest directory.",
                    evidence={"path": binding.path, "hash_matches": False},
                )
            )
            continue
        if not candidate.is_file():
            checks.append(
                _check(
                    f"artifact.{role}",
                    "fail",
                    "Bound artifact is missing.",
                    evidence={"path": binding.path, "hash_matches": False},
                )
            )
            continue
        if candidate.stat().st_size > 16 * 1024 * 1024:
            checks.append(
                _check(
                    f"artifact.{role}",
                    "fail",
                    "Bound artifact exceeds the 16 MiB limit.",
                    evidence={"path": binding.path, "hash_matches": False},
                )
            )
            continue
        raw = candidate.read_bytes()
        actual_hash = sha256(raw).hexdigest()
        if actual_hash != binding.sha256:
            checks.append(
                _check(
                    f"artifact.{role}",
                    "fail",
                    "Bound artifact changed after hashing.",
                    evidence={
                        "path": binding.path,
                        "expected_sha256": binding.sha256,
                        "actual_sha256": actual_hash,
                        "hash_matches": False,
                    },
                )
            )
            continue
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            checks.append(
                _check(
                    f"artifact.{role}",
                    "fail",
                    "Bound artifact is not valid UTF-8 JSON.",
                    evidence={"path": binding.path, "hash_matches": True},
                )
            )
            continue
        if not isinstance(payload, dict):
            checks.append(
                _check(
                    f"artifact.{role}",
                    "fail",
                    "Bound artifact root is not a JSON object.",
                    evidence={"path": binding.path, "hash_matches": True},
                )
            )
            continue
        checks.append(
            _check(
                f"artifact.{role}",
                "pass",
                "Bound artifact exists and matches its declared SHA-256.",
                evidence={"path": binding.path, "hash_matches": True},
            )
        )
        payloads[role] = payload
    return checks, payloads


def _validated_artifacts(
    payloads: dict[ArtifactRole, dict[str, Any]],
) -> tuple[list[TrustReadinessCheck], dict[ArtifactRole, Any]]:
    models: dict[ArtifactRole, type[FrozenModel]] = {
        "calibration_manifest": CalibrationManifestV2,
        "calibration_report": CalibrationReportV2,
        "final_state_contract": FinalStateContract,
        "final_state_initial": EnvironmentSnapshot,
        "final_state_final": EnvironmentSnapshot,
        "final_state_report": FinalStateReport,
        "runtime_fault_manifest": RuntimeFaultManifest,
        "runtime_fault_report": RuntimeFaultCampaignReport,
    }
    checks: list[TrustReadinessCheck] = []
    validated: dict[ArtifactRole, Any] = {}
    for role in EXPECTED_ROLES:
        if role not in payloads:
            checks.append(
                _check(
                    f"schema.{role}",
                    "unknown",
                    "Artifact schema cannot be checked because its bound content is unavailable.",
                )
            )
            continue
        try:
            validated[role] = models[role].model_validate(payloads[role])
        except ValueError as exc:
            checks.append(
                _check(
                    f"schema.{role}",
                    "fail",
                    "Artifact does not satisfy its required schema.",
                    evidence={"error": str(exc)},
                )
            )
        else:
            checks.append(
                _check(
                    f"schema.{role}",
                    "pass",
                    "Artifact satisfies its exact required schema.",
                )
            )
    return checks, validated


def _binding_check(
    check_id: str,
    available: bool,
    passed: bool,
    pass_summary: str,
    fail_summary: str,
) -> TrustReadinessCheck:
    if not available:
        return _check(check_id, "unknown", "Required artifacts are unavailable for this binding.")
    return _check(check_id, "pass" if passed else "fail", pass_summary if passed else fail_summary)


def audit_trust_evidence(manifest_path: str | Path) -> TrustReadinessReport:
    path = Path(manifest_path).resolve(strict=True)
    manifest, raw = _load_manifest(path)
    checks, payloads = _load_bound_artifacts(path, manifest)
    schema_checks, artifacts = _validated_artifacts(payloads)
    checks.extend(schema_checks)

    calibration_available = {
        "calibration_manifest",
        "calibration_report",
    }.issubset(artifacts)
    if calibration_available:
        calibration_manifest: CalibrationManifestV2 = artifacts["calibration_manifest"]
        calibration_report: CalibrationReportV2 = artifacts["calibration_report"]
        calibration_bound = calibration_report.manifest_sha256 == _canonical_hash(
            calibration_manifest.model_dump(mode="json")
        )
        expected = manifest.expected_judge
        bound_identity = calibration_manifest.judge_identities.get(expected.judge_key)
        report_judge = next(
            (item for item in calibration_report.judges if item.judge_key == expected.judge_key),
            None,
        )
        judge_matches = (
            bound_identity is not None
            and report_judge is not None
            and bound_identity.provider == expected.provider
            and bound_identity.model == expected.model
            and bound_identity.prompt_version == expected.prompt_version
            and report_judge.identity == bound_identity
        )
        calibration_passed = calibration_report.passed
    else:
        calibration_bound = judge_matches = calibration_passed = False
    checks.extend(
        [
            _binding_check(
                "binding.calibration_manifest",
                calibration_available,
                calibration_bound,
                "Calibration report binds the exact calibration manifest content.",
                "Calibration report manifest hash does not match the bound manifest.",
            ),
            _binding_check(
                "binding.judge_identity",
                calibration_available,
                judge_matches,
                "Calibration manifest and report match the expected Judge identity.",
                "Judge key, provider, model, or prompt identity does not match.",
            ),
            _binding_check(
                "gate.calibration_holdout",
                calibration_available,
                calibration_passed,
                "Held-out calibration gates passed.",
                "Held-out calibration gates failed.",
            ),
        ]
    )

    final_roles = {
        "final_state_contract",
        "final_state_initial",
        "final_state_final",
        "final_state_report",
    }
    final_available = final_roles.issubset(artifacts)
    if final_available:
        contract: FinalStateContract = artifacts["final_state_contract"]
        initial: EnvironmentSnapshot = artifacts["final_state_initial"]
        final: EnvironmentSnapshot = artifacts["final_state_final"]
        final_report: FinalStateReport = artifacts["final_state_report"]
        recomputed_final_report = verify_final_state(contract, initial, final)
        final_binding = (
            final_report.contract_sha256 == _canonical_hash(contract.model_dump(mode="json"))
            and final_report.initial_snapshot_sha256 == initial.snapshot_sha256
            and final_report.final_snapshot_sha256 == final.snapshot_sha256
            and initial.environment_id == final.environment_id == final_report.environment_id
            and initial.fixture_sha256 == final.fixture_sha256
            and final_report == recomputed_final_report
        )
        final_passed = final_report.passed and final_report.protected_state_passed
    else:
        final_binding = final_passed = False
    checks.extend(
        [
            _binding_check(
                "binding.final_state",
                final_available,
                final_binding,
                "Final-state contract, fixture identity, snapshots, and report are bound.",
                "Final-state contract, fixture identity, snapshot, or report binding differs.",
            ),
            _binding_check(
                "gate.final_state",
                final_available,
                final_passed,
                "Independent final-state and protected-state checks passed.",
                "Independent final-state or protected-state checks failed.",
            ),
        ]
    )

    runtime_available = {
        "runtime_fault_manifest",
        "runtime_fault_report",
    }.issubset(artifacts)
    if runtime_available:
        runtime_manifest: RuntimeFaultManifest = artifacts["runtime_fault_manifest"]
        runtime_report: RuntimeFaultCampaignReport = artifacts["runtime_fault_report"]
        runtime_binding = (
            runtime_report.manifest_sha256
            == _canonical_hash(runtime_manifest.model_dump(mode="json"))
            and runtime_report.target_runner_version == runtime_manifest.target_runner_version
            and runtime_report.environment_snapshot_sha256
            == runtime_manifest.environment_snapshot_sha256
        )
        runtime_passed = runtime_report.passed
    else:
        runtime_binding = runtime_passed = False
    checks.extend(
        [
            _binding_check(
                "binding.runtime_faults",
                runtime_available,
                runtime_binding,
                "Runtime fault report binds the exact manifest, runner, and environment.",
                "Runtime fault manifest, runner, or environment binding differs.",
            ),
            _binding_check(
                "gate.runtime_faults",
                runtime_available,
                runtime_passed,
                "Runtime fault campaign gates passed.",
                "Runtime fault campaign gates failed.",
            ),
        ]
    )

    required = [item for item in checks if item.severity == "required"]
    required_passed = sum(item.status == "pass" for item in required)
    required_failed = sum(item.status == "fail" for item in required)
    required_unknown = sum(item.status == "unknown" for item in required)
    evidence_complete = required_failed == 0 and required_unknown == 0
    promotable = (
        evidence_complete
        and manifest.evidence_scope == "target_integration"
        and manifest.review_status == "human_reviewed"
    )
    decision_reasons: list[str] = []
    if not evidence_complete:
        decision_reasons.append("Required trust evidence is failed or unknown.")
    if manifest.evidence_scope != "target_integration":
        decision_reasons.append("Offline harness evidence cannot establish target-Agent readiness.")
    if manifest.review_status != "human_reviewed":
        decision_reasons.append("Evidence has not completed human review.")
    if promotable:
        decision_reasons.append("All trust gates, target integration, and human review are complete.")
    limitations = list(manifest.limitations)
    if manifest.evidence_scope == "offline_harness" and not any(
        "offline" in item.lower() for item in limitations
    ):
        limitations.append("Offline harness evidence is not proof of third-party Agent behavior.")
    return TrustReadinessReport(
        evidence_id=manifest.evidence_id,
        evidence_version=manifest.version,
        manifest_sha256=sha256(raw).hexdigest(),
        evidence_complete=evidence_complete,
        ready=promotable,
        recommendation="ready" if promotable else "diagnostic_only",
        required_passed=required_passed,
        required_failed=required_failed,
        required_unknown=required_unknown,
        checks=tuple(checks),
        decision_reasons=tuple(decision_reasons),
        limitations=tuple(limitations),
    )


def render_trust_readiness_markdown(report: TrustReadinessReport) -> str:
    lines = [
        f"# Evaluation trust readiness: {report.evidence_id} ({report.evidence_version})",
        "",
        f"Recommendation: **{report.recommendation}**",
        "",
        f"Required checks: {report.required_passed} pass, {report.required_failed} fail, "
        f"{report.required_unknown} unknown",
        "",
        "| Check | Status | Summary |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {item.check_id} | {item.status} | {item.summary} |" for item in report.checks)
    lines.extend(["", "## Decision reasons", ""])
    lines.extend(f"- {item}" for item in report.decision_reasons)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"
