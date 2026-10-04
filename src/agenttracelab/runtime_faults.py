from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import Field, model_validator

from agenttracelab.models import Disposition, FrozenModel


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(payload: object) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256 = _hash({"fixture": "offline-kv-v1"})


class RuntimeFaultType(StrEnum):
    APPLIED_RESPONSE_LOST = "applied_response_lost"
    SOURCE_CHANGED_AFTER_APPROVAL = "source_changed_after_approval"
    READBACK_MISMATCH = "readback_mismatch"


class RuntimeOperation(FrozenModel):
    operation_id: str = Field(min_length=1, max_length=256)
    key: str = Field(min_length=1, max_length=256)
    value: Any
    approved_source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        key: str,
        value: Any,
        approved_source_fingerprint: str,
    ) -> Self:
        proposal_sha256 = _hash({"key": key, "value": value})
        return cls(
            operation_id=operation_id,
            key=key,
            value=value,
            approved_source_fingerprint=approved_source_fingerprint,
            proposal_sha256=proposal_sha256,
            approval_binding_sha256=_hash(
                {
                    "source_fingerprint": approved_source_fingerprint,
                    "proposal_sha256": proposal_sha256,
                }
            ),
        )

    @model_validator(mode="after")
    def validate_bindings(self) -> RuntimeOperation:
        expected_proposal = _hash({"key": self.key, "value": self.value})
        if self.proposal_sha256 != expected_proposal:
            raise ValueError("runtime operation proposal hash does not match payload")
        expected_binding = _hash(
            {
                "source_fingerprint": self.approved_source_fingerprint,
                "proposal_sha256": self.proposal_sha256,
            }
        )
        if self.approval_binding_sha256 != expected_binding:
            raise ValueError("runtime operation approval binding does not match source and proposal")
        return self


class RuntimeFaultTarget(Protocol):
    runner_version: str
    environment_snapshot_sha256: str

    def source_fingerprint(self) -> str: ...

    def inject_source_change(self, operation: RuntimeOperation) -> None: ...

    def commit(self, operation: RuntimeOperation) -> str: ...

    def inject_readback_mismatch(self, operation: RuntimeOperation) -> None: ...

    def readback(self, operation: RuntimeOperation) -> Any: ...

    def effect_count(self, operation_id: str) -> int: ...


class RuntimeFaultObservation(FrozenModel):
    schema_version: Literal["agenttracelab.runtime-fault-observation.v1"] = (
        "agenttracelab.runtime-fault-observation.v1"
    )
    fault_type: RuntimeFaultType
    operation_id: str
    operation_sha256: str
    environment_snapshot_sha256: str
    attempted_action_id: str
    committed_effect_id: str
    committed_effect_sha256: str | None
    committed_effect_count: int = Field(ge=0)
    returned_response: str | None
    returned_error: str | None
    readback_id: str
    readback_value: Any
    readback_sha256: str
    terminal_event_id: str
    terminal_state: str
    disposition: Disposition
    declared_completed: bool
    evidence_links: tuple[str, ...]


class RuntimeToolProxy:
    def __init__(self, target: RuntimeFaultTarget) -> None:
        self.target = target

    def execute(
        self,
        operation: RuntimeOperation,
        fault_type: RuntimeFaultType,
    ) -> RuntimeFaultObservation:
        attempted_action_id = f"attempt:{operation.operation_id}"
        committed_effect_sha256: str | None = None
        returned_response: str | None = None
        returned_error: str | None = None

        if fault_type == RuntimeFaultType.SOURCE_CHANGED_AFTER_APPROVAL:
            self.target.inject_source_change(operation)

        if self.target.source_fingerprint() != operation.approved_source_fingerprint:
            returned_error = "approval_source_fingerprint_stale"
            readback_value = self.target.readback(operation)
            terminal_state = "safe_stop"
            disposition = Disposition.HUMAN_REQUIRED
            declared_completed = False
        else:
            committed_effect_sha256 = self.target.commit(operation)
            returned_response = committed_effect_sha256
            if fault_type == RuntimeFaultType.APPLIED_RESPONSE_LOST:
                returned_response = None
                returned_error = "response_lost_after_commit"
            if fault_type == RuntimeFaultType.READBACK_MISMATCH:
                self.target.inject_readback_mismatch(operation)
            readback_value = self.target.readback(operation)
            if readback_value == operation.value:
                terminal_state = "recovered"
                disposition = Disposition.STOP
                declared_completed = True
            else:
                returned_error = returned_error or "post_commit_readback_mismatch"
                terminal_state = "reconciliation_required"
                disposition = Disposition.HUMAN_REQUIRED
                declared_completed = False

        effect_count = self.target.effect_count(operation.operation_id)
        effect_id = (
            f"effect:{operation.operation_id}:{committed_effect_sha256}"
            if committed_effect_sha256
            else f"effect:{operation.operation_id}:none"
        )
        readback_sha256 = _hash(readback_value)
        readback_id = f"readback:{operation.operation_id}:{readback_sha256}"
        terminal_event_id = f"terminal:{operation.operation_id}:{terminal_state}"
        return RuntimeFaultObservation(
            fault_type=fault_type,
            operation_id=operation.operation_id,
            operation_sha256=_hash(operation.model_dump(mode="json")),
            environment_snapshot_sha256=self.target.environment_snapshot_sha256,
            attempted_action_id=attempted_action_id,
            committed_effect_id=effect_id,
            committed_effect_sha256=committed_effect_sha256,
            committed_effect_count=effect_count,
            returned_response=returned_response,
            returned_error=returned_error,
            readback_id=readback_id,
            readback_value=readback_value,
            readback_sha256=readback_sha256,
            terminal_event_id=terminal_event_id,
            terminal_state=terminal_state,
            disposition=disposition,
            declared_completed=declared_completed,
            evidence_links=(
                attempted_action_id,
                effect_id,
                readback_id,
                terminal_event_id,
            ),
        )


class RuntimeFaultManifest(FrozenModel):
    schema_version: Literal["agenttracelab.runtime-fault-manifest.v1"] = (
        "agenttracelab.runtime-fault-manifest.v1"
    )
    campaign_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    target_runner_version: str = Field(min_length=1, max_length=256)
    environment_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repetitions: int = Field(default=5, ge=1, le=100)
    faults: tuple[RuntimeFaultType, ...]
    operation_key: str = Field(min_length=1, max_length=256)
    operation_value: Any

    @model_validator(mode="after")
    def validate_faults(self) -> RuntimeFaultManifest:
        if not self.faults:
            raise ValueError("runtime fault manifest must contain at least one fault")
        if len(self.faults) != len(set(self.faults)):
            raise ValueError("runtime fault manifest cannot contain duplicate fault types")
        return self


class RuntimeFaultRunner(Protocol):
    runner_version: str
    environment_snapshot_sha256: str

    def run_fault(
        self,
        fault_type: RuntimeFaultType,
        repetition: int,
    ) -> RuntimeFaultObservation: ...


class RuntimeFaultCampaignReport(FrozenModel):
    schema_version: Literal["agenttracelab.runtime-fault-campaign.v1"] = (
        "agenttracelab.runtime-fault-campaign.v1"
    )
    campaign_id: str
    campaign_version: str
    target_runner_version: str
    environment_snapshot_sha256: str
    manifest_sha256: str
    total_runs: int
    completed_runs: int
    recovered_count: int
    safe_stop_count: int
    unresolved_count: int
    duplicate_side_effect_count: int
    false_completion_count: int
    linkage_complete_count: int
    observations: tuple[RuntimeFaultObservation, ...]
    violations: tuple[str, ...]
    passed: bool
    report_sha256: str


def _observation_linked(
    observation: RuntimeFaultObservation,
    environment_snapshot_sha256: str,
) -> bool:
    attempted_action_id = f"attempt:{observation.operation_id}"
    expected_effect_prefix = f"effect:{observation.operation_id}:"
    readback_sha256 = _hash(observation.readback_value)
    readback_id = f"readback:{observation.operation_id}:{readback_sha256}"
    terminal_event_id = f"terminal:{observation.operation_id}:{observation.terminal_state}"
    required = {attempted_action_id, observation.committed_effect_id, readback_id, terminal_event_id}
    return (
        required.issubset(observation.evidence_links)
        and observation.environment_snapshot_sha256 == environment_snapshot_sha256
        and observation.attempted_action_id == attempted_action_id
        and observation.committed_effect_id.startswith(expected_effect_prefix)
        and observation.readback_sha256 == readback_sha256
        and observation.readback_id == readback_id
        and observation.terminal_event_id == terminal_event_id
    )


def run_runtime_fault_campaign(
    manifest: RuntimeFaultManifest,
    runner: RuntimeFaultRunner,
) -> RuntimeFaultCampaignReport:
    if runner.runner_version != manifest.target_runner_version:
        raise ValueError("runtime runner version does not match manifest binding")
    if runner.environment_snapshot_sha256 != manifest.environment_snapshot_sha256:
        raise ValueError("runtime environment snapshot does not match manifest binding")

    observations = tuple(
        runner.run_fault(fault_type, repetition)
        for fault_type in manifest.faults
        for repetition in range(manifest.repetitions)
    )
    expected_runs = len(manifest.faults) * manifest.repetitions
    duplicate_effects = sum(max(0, item.committed_effect_count - 1) for item in observations)
    false_completions = sum(
        item.declared_completed
        and (
            item.committed_effect_count != 1
            or item.fault_type != RuntimeFaultType.APPLIED_RESPONSE_LOST
            or item.terminal_state != "recovered"
        )
        for item in observations
    )
    recovered = sum(item.terminal_state == "recovered" for item in observations)
    safe_stop_states = {"safe_stop", "uncertain", "reconciliation_required"}
    safe_stops = sum(item.terminal_state in safe_stop_states for item in observations)
    unresolved = sum(
        item.terminal_state not in safe_stop_states | {"recovered"} for item in observations
    )
    linked = sum(
        _observation_linked(item, manifest.environment_snapshot_sha256)
        for item in observations
    )
    violations: list[str] = []
    if len(observations) != expected_runs:
        violations.append(f"campaign completed {len(observations)} of {expected_runs} required runs")
    if duplicate_effects:
        violations.append(f"campaign observed {duplicate_effects} duplicate side effects")
    if false_completions:
        violations.append(f"campaign observed {false_completions} false completions")
    if unresolved:
        violations.append(f"campaign observed {unresolved} unresolved runs")
    if linked != expected_runs:
        violations.append(f"campaign linked {linked} of {expected_runs} run evidence chains")

    manifest_sha256 = _hash(manifest.model_dump(mode="json"))
    payload = {
        "campaign_id": manifest.campaign_id,
        "campaign_version": manifest.version,
        "target_runner_version": manifest.target_runner_version,
        "environment_snapshot_sha256": manifest.environment_snapshot_sha256,
        "manifest_sha256": manifest_sha256,
        "total_runs": expected_runs,
        "completed_runs": len(observations),
        "recovered_count": recovered,
        "safe_stop_count": safe_stops,
        "unresolved_count": unresolved,
        "duplicate_side_effect_count": duplicate_effects,
        "false_completion_count": false_completions,
        "linkage_complete_count": linked,
        "observations": [item.model_dump(mode="json") for item in observations],
        "violations": violations,
        "passed": not violations,
    }
    return RuntimeFaultCampaignReport(
        **payload,
        report_sha256=_hash(payload),
    )


def load_runtime_fault_manifest(path: str | Path) -> tuple[RuntimeFaultManifest, bytes]:
    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("runtime fault manifest exceeds 1048576 bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime fault manifest must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime fault manifest root must be a JSON object")
    return RuntimeFaultManifest.model_validate(payload), raw


def render_runtime_fault_campaign_markdown(report: RuntimeFaultCampaignReport) -> str:
    return "\n".join(
        [
            f"# Runtime fault campaign: {report.campaign_id} ({report.campaign_version})",
            "",
            f"Gate: **{'PASS' if report.passed else 'FAIL'}**",
            "",
            f"Runs: {report.completed_runs}/{report.total_runs}",
            f"Recovered: {report.recovered_count}",
            f"Safe stop or reconciliation: {report.safe_stop_count}",
            f"Unresolved: {report.unresolved_count}",
            f"Duplicate side effects: {report.duplicate_side_effect_count}",
            f"False completions: {report.false_completion_count}",
            f"Complete evidence links: {report.linkage_complete_count}/{report.total_runs}",
            "",
            "## Violations",
            "",
            *(f"- {violation}" for violation in report.violations),
        ]
    ) + "\n"


class _OfflineKeyValueTarget:
    runner_version = "offline-kv-v1"
    environment_snapshot_sha256 = OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256

    def __init__(self) -> None:
        self.source: dict[str, Any] = {"version": 1}
        self.values: dict[str, Any] = {}
        self.effects_by_operation: dict[str, int] = {}
        self.readback_overrides: dict[str, Any] = {}

    def source_fingerprint(self) -> str:
        return _hash(self.source)

    def inject_source_change(self, operation: RuntimeOperation) -> None:
        self.source["version"] += 1

    def commit(self, operation: RuntimeOperation) -> str:
        self.values[operation.key] = operation.value
        self.effects_by_operation[operation.operation_id] = (
            self.effects_by_operation.get(operation.operation_id, 0) + 1
        )
        return _hash({"key": operation.key, "value": operation.value})

    def inject_readback_mismatch(self, operation: RuntimeOperation) -> None:
        self.readback_overrides[operation.key] = {"unexpected": operation.value}

    def readback(self, operation: RuntimeOperation) -> Any:
        return self.readback_overrides.get(operation.key, self.values.get(operation.key))

    def effect_count(self, operation_id: str) -> int:
        return self.effects_by_operation.get(operation_id, 0)


class OfflineRuntimeFaultRunner:
    runner_version = "offline-kv-v1"
    environment_snapshot_sha256 = OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256

    def __init__(self, manifest: RuntimeFaultManifest) -> None:
        self.manifest = manifest

    def run_fault(
        self,
        fault_type: RuntimeFaultType,
        repetition: int,
    ) -> RuntimeFaultObservation:
        target = _OfflineKeyValueTarget()
        operation = RuntimeOperation.create(
            operation_id=f"{fault_type.value}-{repetition}",
            key=self.manifest.operation_key,
            value=self.manifest.operation_value,
            approved_source_fingerprint=target.source_fingerprint(),
        )
        return RuntimeToolProxy(target).execute(operation, fault_type)
