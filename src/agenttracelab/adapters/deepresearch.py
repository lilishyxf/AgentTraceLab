from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agenttracelab.models import (
    DecisionEvidence,
    Disposition,
    FrozenModel,
    SpanKind,
    SpanStatus,
    TraceEnvelope,
    TraceSource,
    TraceSpan,
)


class DeepResearchRun(FrozenModel):
    run_id: str
    session_id: str
    source_mode: str
    workflow_mode: str
    status: str
    current_stage: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime


class DeepResearchEvent(FrozenModel):
    event_id: int = Field(ge=1)
    event_type: str = Field(min_length=1)
    stage: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DeepResearchToolCall(FrozenModel):
    tool_call_id: str
    task_id: str | None = None
    tool_name: str
    source_mode: str
    status: str
    args_hash: str | None = None
    result_present: bool = False
    error_code: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class DeepResearchEvidence(FrozenModel):
    evidence_id: str
    tool_call_id: str | None = None
    source_mode: str
    provider: str
    source_id: str
    content_hash: str
    domain: str | None = None
    title: str | None = Field(default=None, max_length=1_000)
    summary: str | None = Field(default=None, max_length=8_000)
    url: str | None = Field(default=None, max_length=4_096)
    evidence_scope: str | None = Field(default=None, max_length=128)
    page_fetch_status: str | None = Field(default=None, max_length=128)
    page_fetch_error_code: str | None = Field(default=None, max_length=128)


class DeepResearchContract(FrozenModel):
    check_id: str
    kind: str
    required: bool
    passed: bool | None
    verifier: str
    verifier_version: str


class DeepResearchCheckpoint(FrozenModel):
    checkpoint_id: str
    version: int = Field(ge=1)
    stage: str
    state_hash: str
    integrity_valid: bool
    created_at: datetime


class DeepResearchSnapshot(FrozenModel):
    schema_version: Literal["deepresearch-agent.run-snapshot.v1"]
    run: DeepResearchRun
    task: str | None = None
    report: str | None = None
    events: tuple[DeepResearchEvent, ...] = ()
    tool_calls: tuple[DeepResearchToolCall, ...] = ()
    evidence: tuple[DeepResearchEvidence, ...] = ()
    contracts: tuple[DeepResearchContract, ...] = ()
    checkpoints: tuple[DeepResearchCheckpoint, ...] = ()
    plan_count: int = Field(default=0, ge=0)
    task_count: int = Field(default=0, ge=0)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> DeepResearchSnapshot:
        event_ids = [event.event_id for event in self.events]
        if event_ids != sorted(event_ids) or len(event_ids) != len(set(event_ids)):
            raise ValueError("DeepResearch events must have unique ascending event_id values")
        tool_ids = [call.tool_call_id for call in self.tool_calls]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("DeepResearch tool_call_id values must be unique")
        return self


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _span_kind(event_type: str) -> SpanKind:
    if event_type.startswith("evidence.") or event_type.startswith("source."):
        return SpanKind.RETRIEVER
    if event_type.startswith("verification."):
        return SpanKind.EVALUATOR
    if event_type.startswith(("plan.", "run.", "agent.", "iteration.")):
        return SpanKind.AGENT
    return SpanKind.CHAIN


def _event_status(event: DeepResearchEvent) -> SpanStatus:
    if event.event_type.endswith((".failed", ".error")) or event.payload.get("passed") is False:
        return SpanStatus.ERROR
    if event.event_type.endswith((".completed", ".created", ".resumed", ".added")):
        return SpanStatus.OK
    return SpanStatus.UNSET


def _tool_status(status: str) -> SpanStatus:
    if status == "completed":
        return SpanStatus.OK
    if status in {"failed", "cancelled", "denied"}:
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def _tool_failure_metrics(snapshot: DeepResearchSnapshot) -> dict[str, int | float]:
    ordered = sorted(snapshot.tool_calls, key=lambda item: (_aware(item.created_at), item.tool_call_id))
    failed_statuses = {"failed", "cancelled", "denied"}
    failed = [item for item in ordered if item.status in failed_statuses]
    required_contracts = [item for item in snapshot.contracts if item.required]
    verified_terminal = (
        snapshot.run.status == "completed"
        and bool(required_contracts)
        and all(item.passed is True for item in required_contracts)
    )
    evidence_call_ids = {item.tool_call_id for item in snapshot.evidence if item.tool_call_id}
    calls_by_id = {item.tool_call_id: item for item in ordered}
    explicit_recovered_ids: set[str] = set()
    if verified_terminal:
        for event in snapshot.events:
            if event.event_type != "tool.recovery_confirmed":
                continue
            failed_id = str(event.payload.get("failed_tool_call_id") or "")
            failed_call = calls_by_id.get(failed_id)
            recovered_by_ids = event.payload.get("recovered_by_tool_call_ids") or []
            if failed_call is None or failed_call.status not in failed_statuses:
                continue
            if any(
                (candidate := calls_by_id.get(str(candidate_id))) is not None
                and candidate.status == "completed"
                and candidate.task_id == failed_call.task_id
                and candidate.tool_call_id in evidence_call_ids
                for candidate_id in recovered_by_ids
            ):
                explicit_recovered_ids.add(failed_id)
    outcome_recovered_tasks = {
        item.task_id
        for item in ordered
        if verified_terminal
        and item.status == "completed"
        and item.result_present
        and item.tool_call_id in evidence_call_ids
    }
    recovered = 0
    for failed_call in failed:
        later_success = any(
            candidate.status == "completed"
            and candidate.task_id == failed_call.task_id
            and _aware(candidate.created_at) > _aware(failed_call.created_at)
            for candidate in ordered
        )
        if (
            failed_call.tool_call_id in explicit_recovered_ids
            or later_success
            or failed_call.task_id in outcome_recovered_tasks
        ):
            recovered += 1
    total = len(ordered)
    failed_count = len(failed)
    return {
        "successful": sum(item.status == "completed" for item in ordered),
        "failed": failed_count,
        "failure_rate": round(failed_count / total, 4) if total else 0.0,
        "recovered": recovered,
        "unrecovered": failed_count - recovered,
        "explicit_recovered": len(explicit_recovered_ids),
        "explicit_recovery_coverage": (
            round(len(explicit_recovered_ids) / failed_count, 4) if failed_count else 0.0
        ),
    }


def _contract_metrics(snapshot: DeepResearchSnapshot) -> dict[str, Any]:
    required = [item for item in snapshot.contracts if item.required]
    passed = sum(item.passed is True for item in required)
    return {
        "required_total": len(required),
        "required_passed": passed,
        "required_pass_rate": passed / len(required) if required else 0.0,
        "results": {item.kind: item.passed for item in snapshot.contracts},
    }


def _citation_metrics(snapshot: DeepResearchSnapshot) -> dict[str, Any]:
    citations = re.findall(r"\[\^?(ev_[A-Za-z0-9_-]+)\]", snapshot.report or "")
    evidence_ids = {item.evidence_id for item in snapshot.evidence}
    valid = [citation for citation in citations if citation in evidence_ids]
    return {
        "count": len(citations),
        "valid_count": len(valid),
        "validity": len(valid) / len(citations) if citations else None,
    }


_REPORT_REPAIRABLE_FAILURES = frozenset(
    {
        "citation_integrity",
        "claim_support",
        "evidence_card_coverage",
        "report_consistency",
        "required_section",
        "source_diversity",
    }
)


def _verification_recovery_metrics(snapshot: DeepResearchSnapshot) -> dict[str, Any]:
    failed = [
        event
        for event in snapshot.events
        if event.event_type == "verification.completed" and event.payload.get("passed") is False
    ]
    signatures: dict[tuple[str, ...], int] = {}
    misrouted: list[DeepResearchEvent] = []
    for event in failed:
        failures = tuple(sorted({str(item) for item in event.payload.get("failures") or [] if item}))
        if failures:
            signatures[failures] = signatures.get(failures, 0) + 1
        if (
            failures
            and set(failures).issubset(_REPORT_REPAIRABLE_FAILURES)
            and event.payload.get("recovery_action") == "replan"
        ):
            misrouted.append(event)

    repeated = sum(count - 1 for count in signatures.values() if count > 1)
    first_misroute = min((_aware(item.created_at) for item in misrouted), default=None)
    calls_before_or_at = 0
    calls_after = 0
    elapsed_after_ms = 0
    amplification = None
    if first_misroute is not None:
        calls_before_or_at = sum(_aware(call.created_at) <= first_misroute for call in snapshot.tool_calls)
        calls_after = len(snapshot.tool_calls) - calls_before_or_at
        terminal = _aware(snapshot.run.completed_at or snapshot.run.updated_at or snapshot.run.created_at)
        elapsed_after_ms = max(0, round((terminal - first_misroute).total_seconds() * 1_000))
        amplification = round(
            len(snapshot.tool_calls) / max(1, calls_before_or_at),
            4,
        )
    return {
        "verification_failure_count": len(failed),
        "repeated_verification_failure_count": repeated,
        "misrouted_report_replan_count": len(misrouted),
        "tool_calls_after_first_misrouted_replan": calls_after,
        "elapsed_after_first_misrouted_replan_ms": elapsed_after_ms,
        "tool_call_amplification_factor": amplification,
        "repeated_failure_signatures": [
            {"failures": list(signature), "count": count}
            for signature, count in sorted(signatures.items())
            if count > 1
        ],
    }


def _decision_evidence(snapshot: DeepResearchSnapshot) -> tuple[DecisionEvidence, ...]:
    decisions: list[DecisionEvidence] = []
    events = list(snapshot.events)
    for index, event in enumerate(events):
        if event.event_type not in {"agent.decision", "plan.replanning", "verification.completed"}:
            continue
        if event.event_type == "verification.completed" and event.payload.get("passed") is not True:
            continue
        later = events[index + 1 :]
        declared_tool_call_id = event.payload.get("tool_call_id")
        tool_event = next(
            (
                item
                for item in later
                if item.event_type in {"tool.completed", "tool.failed"}
                and (not declared_tool_call_id or item.payload.get("tool_call_id") == declared_tool_call_id)
            ),
            None,
        )
        terminal = next(
            (
                item
                for item in later
                if item.event_type in {"verification.completed", "run.completed", "run.failed"}
            ),
            None,
        )
        payload = event.payload
        if event.event_type == "agent.decision":
            observation = str(payload.get("observation") or "The agent selected a tool action.")
            selected_action = str(payload.get("selected_action") or "invoke_tool")
            disposition = Disposition.CONTINUE
        elif event.event_type == "plan.replanning":
            observation = str(
                payload.get("recovery_reason")
                or payload.get("failures")
                or "The run recorded a verification or evidence-coverage failure."
            )
            selected_action = str(payload.get("recovery_action") or "replan")
            disposition = Disposition.REPLAN
        else:
            observation = "The completion contract passed."
            selected_action = str(payload.get("recovery_action") or "complete")
            disposition = Disposition.STOP
        if event.event_type == "agent.decision":
            validation = (
                f"{tool_event.event_type}: {tool_event.payload}"
                if tool_event is not None
                else "No matching tool result event was exported."
            )
        else:
            validation = (
                f"{terminal.event_type}: {terminal.payload}"
                if terminal is not None
                else "No later validation event was exported."
            )
        if event.event_type == "agent.decision":
            tool_call_id = declared_tool_call_id if tool_event is not None else None
        else:
            tool_call_id = (
                declared_tool_call_id
                if declared_tool_call_id
                else None
                if tool_event is None
                else tool_event.payload.get("tool_call_id")
            )
        decisions.append(
            DecisionEvidence(
                evidence_id=f"decision_event_{event.event_id}",
                observation=observation[:2_000],
                selected_action=selected_action[:2_000],
                reason=str(
                    payload.get("reason") or payload.get("failures") or payload.get("recovery_reason") or ""
                )[:2_000],
                tool_call_id=str(tool_call_id) if tool_call_id else None,
                validation=validation[:2_000],
                disposition=disposition,
                created_at=_aware(event.created_at),
            )
        )
    return tuple(decisions)


def adapt_deepresearch_snapshot(
    payload: dict[str, Any] | DeepResearchSnapshot,
) -> TraceEnvelope:
    snapshot = (
        payload if isinstance(payload, DeepResearchSnapshot) else DeepResearchSnapshot.model_validate(payload)
    )
    records: list[tuple[datetime, int, str, Any]] = []
    for event in snapshot.events:
        records.append((_aware(event.created_at), event.event_id, "event", event))
    offset = max((event.event_id for event in snapshot.events), default=0) + 1
    for index, call in enumerate(snapshot.tool_calls):
        records.append((_aware(call.created_at), offset + index, "tool", call))
    records.sort(key=lambda item: (item[0], item[1], item[2]))

    spans: list[TraceSpan] = []
    for sequence, (_, _, record_type, record) in enumerate(records, start=1):
        if record_type == "event":
            event: DeepResearchEvent = record
            spans.append(
                TraceSpan(
                    span_id=f"event_{event.event_id}",
                    sequence=sequence,
                    kind=_span_kind(event.event_type),
                    name=event.event_type,
                    status=_event_status(event),
                    started_at=_aware(event.created_at),
                    ended_at=_aware(event.created_at),
                    duration_ms=0,
                    attributes={
                        "deepresearch.record_type": "event",
                        "deepresearch.event_id": event.event_id,
                        "deepresearch.stage": event.stage,
                        "deepresearch.payload": event.payload,
                    },
                )
            )
            continue
        call: DeepResearchToolCall = record
        started = _aware(call.created_at)
        ended = _aware(call.completed_at or call.created_at)
        spans.append(
            TraceSpan(
                span_id=call.tool_call_id,
                sequence=sequence,
                kind=SpanKind.TOOL,
                name=call.tool_name,
                status=_tool_status(call.status),
                started_at=started,
                ended_at=ended,
                duration_ms=max(0, round((ended - started).total_seconds() * 1_000)),
                attributes={
                    "deepresearch.record_type": "tool_call",
                    "deepresearch.task_id": call.task_id,
                    "deepresearch.source_mode": call.source_mode,
                    "deepresearch.status": call.status,
                    "deepresearch.args_hash": call.args_hash,
                    "deepresearch.result_present": call.result_present,
                    "deepresearch.error_code": call.error_code,
                },
            )
        )

    run = snapshot.run
    started_at = _aware(run.started_at or run.created_at)
    ended_at = _aware(run.completed_at or run.updated_at)
    checkpoint_rate = (
        sum(item.integrity_valid for item in snapshot.checkpoints) / len(snapshot.checkpoints)
        if snapshot.checkpoints
        else None
    )
    source_leakage = any(item.source_mode != run.source_mode for item in snapshot.evidence)
    duplicate_hashes: dict[str, int] = {}
    for call in snapshot.tool_calls:
        if call.args_hash:
            key = f"{call.task_id}|{call.tool_name}|{call.source_mode}|{call.args_hash}"
            duplicate_hashes[key] = duplicate_hashes.get(key, 0) + 1
    duplicate_calls = sum(count - 1 for count in duplicate_hashes.values() if count > 1)
    tool_health = _tool_failure_metrics(snapshot)
    full_page_evidence_count = sum(item.evidence_scope == "fetched_page" for item in snapshot.evidence)
    search_snippet_evidence_count = sum(item.evidence_scope == "search_snippet" for item in snapshot.evidence)
    page_fetch_attempt_count = sum(
        item.page_fetch_status in {"completed", "failed"} for item in snapshot.evidence
    )
    page_fetch_failed_count = sum(item.page_fetch_status == "failed" for item in snapshot.evidence)
    page_fetch_failure_categories: dict[str, int] = {}
    for item in snapshot.evidence:
        if item.page_fetch_status != "failed":
            continue
        category = item.page_fetch_error_code or "unspecified"
        page_fetch_failure_categories[category] = page_fetch_failure_categories.get(category, 0) + 1
    verification_recovery = _verification_recovery_metrics(snapshot)
    return TraceEnvelope(
        trace_id=f"deepresearch_{run.run_id}",
        source=TraceSource(
            kind="deepresearch_agent_harness",
            schema_version=snapshot.schema_version,
        ),
        state=run.status,
        task=snapshot.task,
        started_at=started_at,
        ended_at=max(started_at, ended_at),
        spans=tuple(spans),
        decision_evidence=_decision_evidence(snapshot),
        metadata={
            "deepresearch.run_id": run.run_id,
            "deepresearch.session_id": run.session_id,
            "deepresearch.source_mode": run.source_mode,
            "deepresearch.workflow_mode": run.workflow_mode,
            "deepresearch.current_stage": run.current_stage,
            "deepresearch.error_code": run.error_code,
            "deepresearch.usage": run.usage,
            "deepresearch.contracts": _contract_metrics(snapshot),
            "deepresearch.citations": _citation_metrics(snapshot),
            "deepresearch.evidence_count": len(snapshot.evidence),
            "deepresearch.full_page_evidence_count": full_page_evidence_count,
            "deepresearch.search_snippet_evidence_count": search_snippet_evidence_count,
            "deepresearch.full_page_evidence_rate": (
                full_page_evidence_count / len(snapshot.evidence) if snapshot.evidence else None
            ),
            "deepresearch.page_fetch_attempt_count": page_fetch_attempt_count,
            "deepresearch.page_fetch_failed_count": page_fetch_failed_count,
            "deepresearch.page_fetch_failure_categories": dict(sorted(page_fetch_failure_categories.items())),
            "deepresearch.page_fetch_success_rate": (
                (page_fetch_attempt_count - page_fetch_failed_count) / page_fetch_attempt_count
                if page_fetch_attempt_count
                else None
            ),
            "deepresearch.independent_source_count": len(
                {item.domain or item.source_id for item in snapshot.evidence if item.domain or item.source_id}
            ),
            "deepresearch.source_leakage_detected": source_leakage,
            "deepresearch.checkpoint_count": len(snapshot.checkpoints),
            "deepresearch.checkpoint_integrity_rate": checkpoint_rate,
            "deepresearch.plan_count": snapshot.plan_count,
            "deepresearch.task_count": snapshot.task_count,
            "deepresearch.tool_call_count": len(snapshot.tool_calls),
            "deepresearch.successful_tool_call_count": tool_health["successful"],
            "deepresearch.failed_tool_call_count": tool_health["failed"],
            "deepresearch.tool_failure_rate": tool_health["failure_rate"],
            "deepresearch.recovered_tool_failure_count": tool_health["recovered"],
            "deepresearch.unrecovered_tool_failure_count": tool_health["unrecovered"],
            "deepresearch.explicit_recovered_tool_failure_count": tool_health["explicit_recovered"],
            "deepresearch.explicit_recovery_coverage": tool_health["explicit_recovery_coverage"],
            "deepresearch.duplicate_tool_call_fingerprints": duplicate_calls,
            "deepresearch.verification_failure_count": verification_recovery["verification_failure_count"],
            "deepresearch.repeated_verification_failure_count": verification_recovery[
                "repeated_verification_failure_count"
            ],
            "deepresearch.misrouted_report_replan_count": verification_recovery[
                "misrouted_report_replan_count"
            ],
            "deepresearch.tool_calls_after_first_misrouted_replan": verification_recovery[
                "tool_calls_after_first_misrouted_replan"
            ],
            "deepresearch.elapsed_after_first_misrouted_replan_ms": verification_recovery[
                "elapsed_after_first_misrouted_replan_ms"
            ],
            "deepresearch.tool_call_amplification_factor": verification_recovery[
                "tool_call_amplification_factor"
            ],
            "deepresearch.repeated_verification_failure_signatures": verification_recovery[
                "repeated_failure_signatures"
            ],
            "deepresearch.provenance": snapshot.provenance,
        },
    )


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _evidence_metadata_field(value: str | None, key: str) -> Any:
    metadata = _json_object(value)
    if key in metadata:
        return metadata[key]
    extra = metadata.get("extra")
    return extra.get(key) if isinstance(extra, dict) else None


def export_deepresearch_sqlite_snapshot(database_path: str | Path, run_id: str) -> dict[str, Any]:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise ValueError(f"DeepResearch SQLite database does not exist: {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        run_row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise ValueError(f"DeepResearch run does not exist: {run_id}")
        run = dict(run_row)
        trigger = connection.execute(
            "SELECT content FROM messages WHERE message_id = ?", (run["trigger_message_id"],)
        ).fetchone()
        report = connection.execute(
            "SELECT content FROM messages WHERE run_id = ? AND role = 'assistant' LIMIT 1", (run_id,)
        ).fetchone()

        def rows(sql: str) -> list[dict[str, Any]]:
            return [dict(item) for item in connection.execute(sql, (run_id,)).fetchall()]

        events = rows("SELECT * FROM run_events WHERE run_id = ? ORDER BY event_id")
        tools = rows("SELECT * FROM tool_calls WHERE run_id = ? ORDER BY created_at, tool_call_id")
        evidence = rows(
            "SELECT * FROM evidence WHERE run_id = ? AND invalidated_at IS NULL ORDER BY created_at"
        )
        contracts = rows("SELECT * FROM contract_checks WHERE run_id = ? ORDER BY kind")
        checkpoints = rows("SELECT * FROM checkpoints WHERE run_id = ? ORDER BY version")
        plan_count = connection.execute("SELECT COUNT(*) FROM plans WHERE run_id = ?", (run_id,)).fetchone()[
            0
        ]
        task_count = connection.execute("SELECT COUNT(*) FROM tasks WHERE run_id = ?", (run_id,)).fetchone()[
            0
        ]
    finally:
        connection.close()

    return {
        "schema_version": "deepresearch-agent.run-snapshot.v1",
        "run": {
            "run_id": run["run_id"],
            "session_id": run["session_id"],
            "source_mode": run["source_mode"],
            "workflow_mode": run["workflow_mode"],
            "status": run["status"],
            "current_stage": run["current_stage"],
            "usage": _json_object(run["usage_json"]),
            "error_code": run["error_code"],
            "error_message": run["error_message"],
            "created_at": run["created_at"],
            "started_at": run["started_at"],
            "completed_at": run["completed_at"],
            "updated_at": run["updated_at"],
        },
        "task": trigger["content"] if trigger else None,
        "report": report["content"] if report else None,
        "events": [
            {
                "event_id": item["event_id"],
                "event_type": item["event_type"],
                "stage": item["stage"],
                "payload": _json_object(item["payload_json"]),
                "created_at": item["created_at"],
            }
            for item in events
        ],
        "tool_calls": [
            {
                "tool_call_id": item["tool_call_id"],
                "task_id": item["task_id"],
                "tool_name": item["tool_name"],
                "source_mode": item["source_mode"],
                "status": item["status"],
                "args_hash": hashlib.sha256((item["args_json"] or "{}").encode("utf-8")).hexdigest(),
                "result_present": item["result_json"] is not None,
                "error_code": item["error_code"],
                "created_at": item["created_at"],
                "completed_at": item["completed_at"],
            }
            for item in tools
        ],
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "tool_call_id": item["tool_call_id"],
                "source_mode": item["source_mode"],
                "provider": item["provider"],
                "source_id": item["source_id"],
                "content_hash": item["content_hash"],
                "domain": _json_object(item["metadata_json"]).get("domain"),
                "title": item.get("title"),
                "summary": (item.get("summary") or "")[:8_000] or None,
                "url": _json_object(item["metadata_json"]).get("url"),
                "evidence_scope": _evidence_metadata_field(item["metadata_json"], "evidence_scope"),
                "page_fetch_status": _evidence_metadata_field(item["metadata_json"], "page_fetch_status"),
                "page_fetch_error_code": _evidence_metadata_field(
                    item["metadata_json"], "page_fetch_error_code"
                ),
            }
            for item in evidence
        ],
        "contracts": [
            {
                "check_id": item["check_id"],
                "kind": item["kind"],
                "required": bool(item["required"]),
                "passed": None if item["passed"] is None else bool(item["passed"]),
                "verifier": item["verifier"],
                "verifier_version": item["verifier_version"],
            }
            for item in contracts
        ],
        "checkpoints": [
            {
                "checkpoint_id": item["checkpoint_id"],
                "version": item["version"],
                "stage": item["stage"],
                "state_hash": item["state_hash"],
                "integrity_valid": hashlib.sha256(item["state_json"].encode("utf-8")).hexdigest()
                == item["state_hash"],
                "created_at": item["created_at"],
            }
            for item in checkpoints
        ],
        "plan_count": plan_count,
        "task_count": task_count,
        "provenance": {
            "source": "deepresearch_agent_harness.sqlite",
            "database_file": path.name,
            "exported_at": datetime.now(UTC).isoformat(),
            "privacy": (
                "tool arguments/results and checkpoint state omitted; evidence title and bounded "
                "summary retained for claim verification"
            ),
        },
    }
