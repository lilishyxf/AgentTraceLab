from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenttracelab.adapters import adapt_deepresearch_snapshot
from agenttracelab.adapters.deepresearch import DeepResearchSnapshot
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.models import FrozenModel


class DeepResearchFaultType(StrEnum):
    CITATION_CORRUPTION = "citation_corruption"
    SOURCE_MODE_LEAKAGE = "source_mode_leakage"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    REQUIRED_CONTRACT_FAILURE = "required_contract_failure"
    TOOL_ACCOUNTING_DRIFT = "tool_accounting_drift"
    TOOL_RESULT_LOSS = "tool_result_loss"
    NON_TERMINAL_RUN = "non_terminal_run"
    DECISION_TOOL_LINK_LOSS = "decision_tool_link_loss"
    SECRET_LEAKAGE = "secret_leakage"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"


EXPECTED_CHECKS: dict[DeepResearchFaultType, tuple[str, ...]] = {
    DeepResearchFaultType.CITATION_CORRUPTION: ("deepresearch.citations_valid",),
    DeepResearchFaultType.SOURCE_MODE_LEAKAGE: ("deepresearch.source_isolation",),
    DeepResearchFaultType.CHECKPOINT_CORRUPTION: ("deepresearch.checkpoints_intact",),
    DeepResearchFaultType.REQUIRED_CONTRACT_FAILURE: ("deepresearch.required_contracts_pass",),
    DeepResearchFaultType.TOOL_ACCOUNTING_DRIFT: ("deepresearch.tool_accounting_consistent",),
    DeepResearchFaultType.TOOL_RESULT_LOSS: ("tool.result_recorded",),
    DeepResearchFaultType.NON_TERMINAL_RUN: ("trace.terminal_disposition",),
    DeepResearchFaultType.DECISION_TOOL_LINK_LOSS: ("agent.tool_decision_link",),
    DeepResearchFaultType.SECRET_LEAKAGE: ("privacy.no_raw_secret",),
    DeepResearchFaultType.DUPLICATE_TOOL_CALL: ("deepresearch.duplicate_tool_calls_absent",),
}


class DeepResearchFaultCase(FrozenModel):
    fault_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    fault_type: DeepResearchFaultType
    description: str = Field(default="", max_length=1_000)
    expected_check_ids: tuple[str, ...] = ()


class DeepResearchFaultManifest(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-faults.v1"]
    suite_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    cases: tuple[DeepResearchFaultCase, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_cases(self) -> DeepResearchFaultManifest:
        ids = [item.fault_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("DeepResearch fault_id values must be unique")
        return self


class DeepResearchFaultResult(FrozenModel):
    fault_id: str
    fault_type: DeepResearchFaultType
    expected_check_ids: tuple[str, ...]
    observed_failure_categories: tuple[str, ...]
    detected: bool
    evaluator_score: float = Field(ge=0, le=100)
    mutant_path: str | None = None
    evaluation_path: str | None = None


class DeepResearchFaultReport(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-fault-report.v1"] = (
        "agenttracelab.deepresearch-fault-report.v1"
    )
    suite_id: str
    suite_version: str
    source_run_id: str
    source_snapshot_sha256: str
    created_at: datetime
    original_passed: bool
    original_score: float = Field(ge=0, le=100)
    case_count: int = Field(ge=1)
    detected_count: int = Field(ge=0)
    mutation_detection_rate: float = Field(ge=0, le=1)
    missed_fault_ids: tuple[str, ...]
    observed_failure_categories: dict[str, int]
    results: tuple[DeepResearchFaultResult, ...]
    limitations: tuple[str, ...]


def load_deepresearch_fault_manifest(
    path: str | Path,
) -> tuple[DeepResearchFaultManifest, Path]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"DeepResearch fault manifest does not exist: {manifest_path}")
    if manifest_path.stat().st_size > 1_048_576:
        raise ValueError("DeepResearch fault manifest exceeds the 1 MiB limit")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeepResearch fault manifest must contain a JSON object")
    return DeepResearchFaultManifest.model_validate(payload), manifest_path


def _require(items: list[dict], fault_type: DeepResearchFaultType, record: str) -> dict:
    if not items:
        raise ValueError(f"fault {fault_type.value} requires at least one {record} record")
    return items[0]


def inject_deepresearch_fault(
    payload: dict,
    fault_type: DeepResearchFaultType,
) -> dict:
    mutated = copy.deepcopy(payload)
    run = mutated["run"]
    if fault_type == DeepResearchFaultType.CITATION_CORRUPTION:
        mutated["report"] = f"{mutated.get('report') or ''} [ev_injected_missing]"
    elif fault_type == DeepResearchFaultType.SOURCE_MODE_LEAKAGE:
        evidence = _require(mutated.get("evidence") or [], fault_type, "evidence")
        evidence["source_mode"] = "graphrag" if run["source_mode"] == "web" else "web"
    elif fault_type == DeepResearchFaultType.CHECKPOINT_CORRUPTION:
        checkpoint = _require(mutated.get("checkpoints") or [], fault_type, "checkpoint")
        checkpoint["integrity_valid"] = False
    elif fault_type == DeepResearchFaultType.REQUIRED_CONTRACT_FAILURE:
        contracts = [item for item in mutated.get("contracts") or [] if item.get("required")]
        contract = _require(contracts, fault_type, "required contract")
        contract["passed"] = False
    elif fault_type == DeepResearchFaultType.TOOL_ACCOUNTING_DRIFT:
        usage = run.setdefault("usage", {}).setdefault("usage", {})
        usage["tool_calls"] = len(mutated.get("tool_calls") or []) + 7
    elif fault_type == DeepResearchFaultType.TOOL_RESULT_LOSS:
        tool = _require(mutated.get("tool_calls") or [], fault_type, "tool call")
        tool.update({"status": "running", "result_present": False, "completed_at": None})
    elif fault_type == DeepResearchFaultType.NON_TERMINAL_RUN:
        run.update({"status": "active", "current_stage": "executing", "completed_at": None})
    elif fault_type == DeepResearchFaultType.DECISION_TOOL_LINK_LOSS:
        mutated["events"] = [
            item for item in mutated.get("events") or [] if item.get("event_type") != "tool.completed"
        ]
    elif fault_type == DeepResearchFaultType.SECRET_LEAKAGE:
        events = mutated.setdefault("events", [])
        next_id = max((int(item["event_id"]) for item in events), default=0) + 1
        events.append(
            {
                "event_id": next_id,
                "event_type": "debug.recorded",
                "stage": "completed",
                "payload": {"authorization": "Bearer TESTTOKEN1234567890"},
                "created_at": run["updated_at"],
            }
        )
    elif fault_type == DeepResearchFaultType.DUPLICATE_TOOL_CALL:
        tool = copy.deepcopy(_require(mutated.get("tool_calls") or [], fault_type, "tool call"))
        tool["tool_call_id"] = f"{tool['tool_call_id']}_duplicate"
        mutated["tool_calls"].append(tool)
        usage = run.setdefault("usage", {}).setdefault("usage", {})
        usage["tool_calls"] = len(mutated["tool_calls"])
    return DeepResearchSnapshot.model_validate(mutated).model_dump(mode="json")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def run_deepresearch_fault_suite(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    *,
    output_directory: str | Path,
    retain_mutants: bool = True,
) -> DeepResearchFaultReport:
    source_path = Path(snapshot_path).resolve()
    raw = source_path.read_bytes()
    source = json.loads(raw)
    snapshot = DeepResearchSnapshot.model_validate(source)
    manifest, _ = load_deepresearch_fault_manifest(manifest_path)
    output = Path(output_directory).resolve()
    report_path = output / "fault-report.json"
    if report_path.exists():
        raise ValueError("fault output already contains fault-report.json; use a new output directory")
    original = evaluate_trace(adapt_deepresearch_snapshot(snapshot))
    results: list[DeepResearchFaultResult] = []
    for case in manifest.cases:
        mutant = inject_deepresearch_fault(source, case.fault_type)
        evaluation = evaluate_trace(adapt_deepresearch_snapshot(mutant))
        expected = case.expected_check_ids or EXPECTED_CHECKS[case.fault_type]
        detected = set(expected).issubset(evaluation.failure_categories)
        mutant_path = None
        evaluation_path = None
        if retain_mutants:
            case_directory = output / "mutants" / case.fault_id
            mutant_file = case_directory / "run-snapshot.json"
            evaluation_file = case_directory / "evaluation-report.json"
            _write_json(mutant_file, mutant)
            _write_json(evaluation_file, evaluation.model_dump(mode="json"))
            mutant_path = mutant_file.relative_to(output).as_posix()
            evaluation_path = evaluation_file.relative_to(output).as_posix()
        results.append(
            DeepResearchFaultResult(
                fault_id=case.fault_id,
                fault_type=case.fault_type,
                expected_check_ids=expected,
                observed_failure_categories=evaluation.failure_categories,
                detected=detected,
                evaluator_score=evaluation.score,
                mutant_path=mutant_path,
                evaluation_path=evaluation_path,
            )
        )
    failures = Counter(category for item in results for category in item.observed_failure_categories)
    detected_count = sum(item.detected for item in results)
    report = DeepResearchFaultReport(
        suite_id=manifest.suite_id,
        suite_version=manifest.version,
        source_run_id=snapshot.run.run_id,
        source_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        created_at=datetime.now(UTC),
        original_passed=original.passed,
        original_score=original.score,
        case_count=len(results),
        detected_count=detected_count,
        mutation_detection_rate=round(detected_count / len(results), 4),
        missed_fault_ids=tuple(item.fault_id for item in results if not item.detected),
        observed_failure_categories=dict(sorted(failures.items())),
        results=tuple(results),
        limitations=(
            "Snapshot mutation measures evaluator sensitivity, not target-Agent recovery behavior.",
            "Target-side network, provider, and tool faults require injection hooks in the target runtime.",
        ),
    )
    _write_json(report_path, report.model_dump(mode="json"))
    return report
