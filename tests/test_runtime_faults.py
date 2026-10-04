from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from agenttracelab.cli import main
from agenttracelab.models import Disposition
from agenttracelab.runtime_faults import (
    OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256,
    RuntimeFaultManifest,
    RuntimeFaultObservation,
    RuntimeFaultType,
    RuntimeOperation,
    RuntimeToolProxy,
    run_runtime_fault_campaign,
)


def _hash(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class InMemoryTarget:
    runner_version = "in-memory-v1"
    environment_snapshot_sha256 = "a" * 64

    def __init__(self) -> None:
        self.source: dict[str, Any] = {"version": 1}
        self.values: dict[str, Any] = {}
        self.commit_calls = 0
        self.effects_by_operation: dict[str, int] = {}
        self.readback_overrides: dict[str, Any] = {}

    def source_fingerprint(self) -> str:
        return _hash(self.source)

    def inject_source_change(self, operation: RuntimeOperation) -> None:
        self.source["version"] += 1

    def commit(self, operation: RuntimeOperation) -> str:
        self.commit_calls += 1
        self.values[operation.key] = operation.value
        self.effects_by_operation[operation.operation_id] = (
            self.effects_by_operation.get(operation.operation_id, 0) + 1
        )
        return _hash({"key": operation.key, "value": operation.value})

    def inject_readback_mismatch(self, operation: RuntimeOperation) -> None:
        self.readback_overrides[operation.key] = "unexpected"

    def readback(self, operation: RuntimeOperation) -> Any:
        return self.readback_overrides.get(operation.key, self.values.get(operation.key))

    def effect_count(self, operation_id: str) -> int:
        return self.effects_by_operation.get(operation_id, 0)


def _operation(target: InMemoryTarget, operation_id: str = "operation-1") -> RuntimeOperation:
    return RuntimeOperation.create(
        operation_id=operation_id,
        key="status",
        value="done",
        approved_source_fingerprint=target.source_fingerprint(),
    )


def test_applied_response_lost_commits_once_without_blind_retry() -> None:
    target = InMemoryTarget()
    observation = RuntimeToolProxy(target).execute(
        _operation(target),
        RuntimeFaultType.APPLIED_RESPONSE_LOST,
    )

    assert target.commit_calls == 1
    assert target.effect_count("operation-1") == 1
    assert observation.committed_effect_count == 1
    assert observation.returned_error == "response_lost_after_commit"
    assert observation.readback_value == "done"
    assert observation.terminal_state == "recovered"
    assert observation.disposition == Disposition.STOP
    assert observation.declared_completed is True


def test_stale_approval_prevents_commit() -> None:
    target = InMemoryTarget()
    observation = RuntimeToolProxy(target).execute(
        _operation(target),
        RuntimeFaultType.SOURCE_CHANGED_AFTER_APPROVAL,
    )

    assert target.commit_calls == 0
    assert observation.committed_effect_count == 0
    assert observation.returned_error == "approval_source_fingerprint_stale"
    assert observation.terminal_state == "safe_stop"
    assert observation.disposition == Disposition.HUMAN_REQUIRED
    assert observation.declared_completed is False


def test_readback_mismatch_prevents_completion() -> None:
    target = InMemoryTarget()
    observation = RuntimeToolProxy(target).execute(
        _operation(target),
        RuntimeFaultType.READBACK_MISMATCH,
    )

    assert target.commit_calls == 1
    assert observation.committed_effect_count == 1
    assert observation.readback_value == "unexpected"
    assert observation.terminal_state == "reconciliation_required"
    assert observation.disposition == Disposition.HUMAN_REQUIRED
    assert observation.declared_completed is False


def _manifest(repetitions: int = 5) -> RuntimeFaultManifest:
    return RuntimeFaultManifest(
        campaign_id="offline-runtime-faults",
        version="v1",
        target_runner_version="in-memory-v1",
        environment_snapshot_sha256="a" * 64,
        repetitions=repetitions,
        faults=tuple(RuntimeFaultType),
        operation_key="status",
        operation_value="done",
    )


class FreshTargetRunner:
    runner_version = "in-memory-v1"
    environment_snapshot_sha256 = "a" * 64

    def run_fault(
        self,
        fault_type: RuntimeFaultType,
        repetition: int,
    ) -> RuntimeFaultObservation:
        target = InMemoryTarget()
        operation = _operation(target, f"{fault_type.value}-{repetition}")
        return RuntimeToolProxy(target).execute(operation, fault_type)


class MutatingRunner(FreshTargetRunner):
    def __init__(self, mutation: str) -> None:
        self.mutation = mutation

    def run_fault(
        self,
        fault_type: RuntimeFaultType,
        repetition: int,
    ) -> RuntimeFaultObservation:
        observation = super().run_fault(fault_type, repetition)
        if repetition != 0 or fault_type != RuntimeFaultType.READBACK_MISMATCH:
            return observation
        if self.mutation == "duplicate":
            return observation.model_copy(update={"committed_effect_count": 2})
        if self.mutation == "false_completion":
            return observation.model_copy(update={"declared_completed": True})
        if self.mutation == "unlinked":
            return observation.model_copy(
                update={
                    "evidence_links": tuple(
                        link
                        for link in observation.evidence_links
                        if link != observation.terminal_event_id
                    )
                }
            )
        if self.mutation == "forged_links":
            forged = f"readback:{observation.operation_id}:forged"
            return observation.model_copy(
                update={
                    "readback_id": forged,
                    "evidence_links": tuple(
                        forged if link == observation.readback_id else link
                        for link in observation.evidence_links
                    ),
                }
            )
        raise AssertionError(f"unknown mutation: {self.mutation}")


class MissingEffectRunner(FreshTargetRunner):
    def run_fault(
        self,
        fault_type: RuntimeFaultType,
        repetition: int,
    ) -> RuntimeFaultObservation:
        observation = super().run_fault(fault_type, repetition)
        if fault_type == RuntimeFaultType.APPLIED_RESPONSE_LOST:
            return observation.model_copy(update={"committed_effect_count": 0})
        return observation


def test_campaign_runs_three_faults_five_times() -> None:
    report = run_runtime_fault_campaign(_manifest(), FreshTargetRunner())

    assert report.total_runs == 15
    assert report.completed_runs == 15
    assert report.recovered_count == 5
    assert report.safe_stop_count == 10
    assert report.unresolved_count == 0
    assert report.duplicate_side_effect_count == 0
    assert report.false_completion_count == 0
    assert report.linkage_complete_count == 15
    assert report.passed is True


def test_campaign_detects_duplicate_side_effect() -> None:
    report = run_runtime_fault_campaign(_manifest(1), MutatingRunner("duplicate"))

    assert report.duplicate_side_effect_count == 1
    assert report.passed is False


def test_campaign_detects_false_completion() -> None:
    report = run_runtime_fault_campaign(_manifest(1), MutatingRunner("false_completion"))

    assert report.false_completion_count == 1
    assert report.passed is False


def test_campaign_rejects_unlinked_safe_stop() -> None:
    report = run_runtime_fault_campaign(_manifest(1), MutatingRunner("unlinked"))

    assert report.linkage_complete_count == 2
    assert report.passed is False


def test_campaign_rejects_forged_evidence_ids() -> None:
    report = run_runtime_fault_campaign(_manifest(1), MutatingRunner("forged_links"))

    assert report.linkage_complete_count == 2
    assert report.passed is False


def test_campaign_detects_completion_without_effect() -> None:
    report = run_runtime_fault_campaign(_manifest(1), MissingEffectRunner())

    assert report.false_completion_count == 1
    assert report.passed is False


def test_campaign_artifact_hashes_are_stable() -> None:
    first = run_runtime_fault_campaign(_manifest(1), FreshTargetRunner())
    second = run_runtime_fault_campaign(_manifest(1), FreshTargetRunner())

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.report_sha256 == second.report_sha256


def test_runtime_fault_campaign_cli_writes_immutable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    output_directory = tmp_path / "artifacts"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.runtime-fault-manifest.v1",
                "campaign_id": "offline-cli",
                "version": "v1",
                "target_runner_version": "offline-kv-v1",
                "environment_snapshot_sha256": OFFLINE_ENVIRONMENT_SNAPSHOT_SHA256,
                "repetitions": 5,
                "faults": [fault.value for fault in RuntimeFaultType],
                "operation_key": "status",
                "operation_value": "done",
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "agenttracelab",
        "run-runtime-faults",
        str(manifest),
        "--output-directory",
        str(output_directory),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    assert json.loads(
        (output_directory / "campaign-report.json").read_text(encoding="utf-8")
    )["passed"] is True
    assert "Runtime fault campaign" in (
        output_directory / "campaign-summary.md"
    ).read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 18
