from __future__ import annotations

import hashlib
import json
import sqlite3

from fastapi.testclient import TestClient

from agenttracelab.adapters import (
    adapt_deepresearch_snapshot,
    export_deepresearch_sqlite_snapshot,
)
from agenttracelab.api import create_app
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.models import CheckStatus, SpanKind


def deepresearch_snapshot() -> dict:
    return {
        "schema_version": "deepresearch-agent.run-snapshot.v1",
        "run": {
            "run_id": "run_demo",
            "session_id": "session_demo",
            "source_mode": "web",
            "workflow_mode": "deep_research",
            "status": "completed",
            "current_stage": "completed",
            "usage": {"usage": {"llm_tokens": 120, "tool_calls": 1}},
            "error_code": None,
            "error_message": None,
            "created_at": "2026-09-21T00:00:00Z",
            "started_at": "2026-09-21T00:00:01Z",
            "completed_at": "2026-09-21T00:00:05Z",
            "updated_at": "2026-09-21T00:00:05Z",
        },
        "task": "Find evidence for the research question.",
        "report": "Supported conclusion [ev_gold]",
        "events": [
            {
                "event_id": 1,
                "event_type": "agent.decision",
                "stage": "executing",
                "payload": {
                    "tool_call_id": "tool_1",
                    "observation": "External evidence is required.",
                    "selected_action": "invoke:web_search",
                    "reason": "Ground the answer before generation.",
                },
                "created_at": "2026-09-21T00:00:01Z",
            },
            {
                "event_id": 2,
                "event_type": "plan.replanning",
                "stage": "replanning",
                "payload": {
                    "recovery_action": "replan",
                    "recovery_reason": "insufficient_source_evidence",
                },
                "created_at": "2026-09-21T00:00:01Z",
            },
            {
                "event_id": 3,
                "event_type": "tool.completed",
                "stage": "executing",
                "payload": {"tool_call_id": "tool_1", "tool_name": "web_search"},
                "created_at": "2026-09-21T00:00:03Z",
            },
            {
                "event_id": 4,
                "event_type": "verification.completed",
                "stage": "verifying",
                "payload": {"passed": True, "recovery_action": "complete"},
                "created_at": "2026-09-21T00:00:04Z",
            },
            {
                "event_id": 5,
                "event_type": "run.completed",
                "stage": "completed",
                "payload": {"verified": True},
                "created_at": "2026-09-21T00:00:05Z",
            },
        ],
        "tool_calls": [
            {
                "tool_call_id": "tool_1",
                "task_id": "task_1",
                "tool_name": "web_search",
                "source_mode": "web",
                "status": "completed",
                "args_hash": "a" * 64,
                "result_present": True,
                "error_code": None,
                "created_at": "2026-09-21T00:00:02Z",
                "completed_at": "2026-09-21T00:00:03Z",
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev_gold",
                "tool_call_id": "tool_1",
                "source_mode": "web",
                "provider": "web_search",
                "source_id": "https://example.com/source",
                "content_hash": "b" * 64,
                "domain": "example.com",
                "title": "Research source",
                "summary": "The source supports the research conclusion.",
                "url": "https://example.com/source",
                "evidence_scope": "fetched_page",
                "page_fetch_status": "completed",
            }
        ],
        "contracts": [
            {
                "check_id": "contract_source",
                "kind": "source_match",
                "required": True,
                "passed": True,
                "verifier": "deterministic",
                "verifier_version": "1",
            },
            {
                "check_id": "contract_citation",
                "kind": "citation_integrity",
                "required": True,
                "passed": True,
                "verifier": "deterministic",
                "verifier_version": "1",
            },
        ],
        "checkpoints": [
            {
                "checkpoint_id": "chk_1",
                "version": 1,
                "stage": "completed",
                "state_hash": "c" * 64,
                "integrity_valid": True,
                "created_at": "2026-09-21T00:00:05Z",
            }
        ],
        "plan_count": 2,
        "task_count": 1,
        "provenance": {"source": "test"},
    }


def test_adapts_deepresearch_durable_run_and_evaluates_it() -> None:
    trace = adapt_deepresearch_snapshot(deepresearch_snapshot())
    report = evaluate_trace(trace)

    assert trace.source.kind == "deepresearch_agent_harness"
    assert trace.trace_id == "deepresearch_run_demo"
    assert any(span.kind == SpanKind.TOOL and span.span_id == "tool_1" for span in trace.spans)
    assert any(item.tool_call_id == "tool_1" for item in trace.decision_evidence)
    assert report.passed is True
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["deepresearch.required_contracts_pass"] == CheckStatus.PASS
    assert statuses["deepresearch.citations_valid"] == CheckStatus.PASS
    assert statuses["deepresearch.checkpoints_intact"] == CheckStatus.PASS
    assert statuses["deepresearch.source_isolation"] == CheckStatus.PASS
    assert statuses["deepresearch.tool_accounting_consistent"] == CheckStatus.PASS
    assert statuses["deepresearch.tool_failure_free"] == CheckStatus.PASS
    assert statuses["deepresearch.tool_failures_recovered"] == CheckStatus.NOT_APPLICABLE
    assert trace.metadata["deepresearch.full_page_evidence_count"] == 1
    assert trace.metadata["deepresearch.search_snippet_evidence_count"] == 0
    assert trace.metadata["deepresearch.full_page_evidence_rate"] == 1.0
    assert trace.metadata["deepresearch.page_fetch_attempt_count"] == 1
    assert trace.metadata["deepresearch.page_fetch_failed_count"] == 0
    assert trace.metadata["deepresearch.page_fetch_success_rate"] == 1.0


def test_counts_privacy_safe_page_fetch_failure_categories() -> None:
    payload = deepresearch_snapshot()
    payload["evidence"].append(
        {
            "evidence_id": "ev_fallback",
            "tool_call_id": "tool_1",
            "source_mode": "web",
            "provider": "web_search",
            "source_id": "https://example.com/fallback",
            "content_hash": "f" * 64,
            "domain": "example.com",
            "title": "Fallback snippet",
            "summary": "A retained search snippet.",
            "url": "https://example.com/fallback",
            "evidence_scope": "search_snippet",
            "page_fetch_status": "failed",
            "page_fetch_error_code": "http_status",
        }
    )

    trace = adapt_deepresearch_snapshot(payload)

    assert trace.metadata["deepresearch.page_fetch_attempt_count"] == 2
    assert trace.metadata["deepresearch.page_fetch_failed_count"] == 1
    assert trace.metadata["deepresearch.page_fetch_success_rate"] == 0.5
    assert trace.metadata["deepresearch.page_fetch_failure_categories"] == {"http_status": 1}


def test_unrecovered_deepresearch_tool_failure_fails_the_required_gate() -> None:
    payload = deepresearch_snapshot()
    payload["run"].update({"status": "failed", "current_stage": "failed"})
    payload["run"]["usage"]["usage"]["tool_calls"] = 2
    payload["tool_calls"].append(
        {
            "tool_call_id": "tool_2",
            "task_id": "task_1",
            "tool_name": "web_search",
            "source_mode": "web",
            "status": "failed",
            "args_hash": "f" * 64,
            "result_present": False,
            "error_code": "TOOL_FAILED",
            "created_at": "2026-09-21T00:00:03.500Z",
            "completed_at": "2026-09-21T00:00:04Z",
        }
    )
    payload["events"].append(
        {
            "event_id": 6,
            "event_type": "tool.recovery_confirmed",
            "stage": "verifying",
            "payload": {
                "failed_tool_call_id": "tool_2",
                "recovered_by_tool_call_ids": ["tool_1"],
                "strategy": "verified_task_outcome",
            },
            "created_at": "2026-09-21T00:00:04.500Z",
        }
    )

    trace = adapt_deepresearch_snapshot(payload)
    report = evaluate_trace(trace)
    checks = {check.check_id: check for check in report.checks}

    assert trace.metadata["deepresearch.tool_failure_rate"] == 0.5
    assert trace.metadata["deepresearch.unrecovered_tool_failure_count"] == 1
    assert trace.metadata["deepresearch.explicit_recovered_tool_failure_count"] == 0
    assert checks["deepresearch.tool_failure_free"].status == CheckStatus.FAIL
    assert checks["deepresearch.tool_failures_recovered"].status == CheckStatus.FAIL
    assert report.passed is False
    assert "deepresearch.tool_failures_recovered" in report.failure_categories


def test_verified_task_outcome_recovers_a_late_alternative_query_failure() -> None:
    payload = deepresearch_snapshot()
    payload["run"]["usage"]["usage"]["tool_calls"] = 2
    payload["events"][-1]["event_id"] = 6
    payload["tool_calls"].append(
        {
            "tool_call_id": "tool_2",
            "task_id": "task_1",
            "tool_name": "web_search",
            "source_mode": "web",
            "status": "failed",
            "args_hash": "f" * 64,
            "result_present": False,
            "error_code": "TOOL_FAILED",
            "created_at": "2026-09-21T00:00:03.500Z",
            "completed_at": "2026-09-21T00:00:04Z",
        }
    )
    payload["events"].insert(
        -1,
        {
            "event_id": 5,
            "event_type": "tool.recovery_confirmed",
            "stage": "verifying",
            "payload": {
                "failed_tool_call_id": "tool_2",
                "recovered_by_tool_call_ids": ["tool_1"],
                "strategy": "verified_task_outcome",
            },
            "created_at": "2026-09-21T00:00:04.500Z",
        },
    )

    trace = adapt_deepresearch_snapshot(payload)
    report = evaluate_trace(trace)
    checks = {check.check_id: check for check in report.checks}

    assert trace.metadata["deepresearch.tool_failure_rate"] == 0.5
    assert trace.metadata["deepresearch.recovered_tool_failure_count"] == 1
    assert trace.metadata["deepresearch.unrecovered_tool_failure_count"] == 0
    assert trace.metadata["deepresearch.explicit_recovered_tool_failure_count"] == 1
    assert trace.metadata["deepresearch.explicit_recovery_coverage"] == 1.0
    assert checks["deepresearch.tool_failure_free"].status == CheckStatus.FAIL
    assert checks["deepresearch.tool_failures_recovered"].status == CheckStatus.PASS
    assert report.passed is True


def test_recovered_deepresearch_tool_failure_is_degraded_but_passes_required_gate() -> None:
    payload = deepresearch_snapshot()
    payload["run"]["usage"]["usage"]["tool_calls"] = 2
    payload["tool_calls"][0].update(
        {
            "status": "failed",
            "result_present": False,
            "error_code": "TOOL_FAILED",
            "created_at": "2026-09-21T00:00:01.500Z",
            "completed_at": "2026-09-21T00:00:02Z",
        }
    )
    payload["tool_calls"].append(
        {
            "tool_call_id": "tool_2",
            "task_id": "task_1",
            "tool_name": "web_search",
            "source_mode": "web",
            "status": "completed",
            "args_hash": "f" * 64,
            "result_present": True,
            "error_code": None,
            "created_at": "2026-09-21T00:00:02.500Z",
            "completed_at": "2026-09-21T00:00:03Z",
        }
    )

    trace = adapt_deepresearch_snapshot(payload)
    report = evaluate_trace(trace)
    checks = {check.check_id: check for check in report.checks}

    assert trace.metadata["deepresearch.recovered_tool_failure_count"] == 1
    assert trace.metadata["deepresearch.unrecovered_tool_failure_count"] == 0
    assert checks["deepresearch.tool_failure_free"].status == CheckStatus.FAIL
    assert checks["deepresearch.tool_failures_recovered"].status == CheckStatus.PASS
    assert report.passed is True
    assert report.score < 100


def test_detects_deepresearch_tool_accounting_drift() -> None:
    payload = deepresearch_snapshot()
    payload["run"]["usage"]["usage"]["tool_calls"] = 3
    report = evaluate_trace(adapt_deepresearch_snapshot(payload))

    assert report.passed is False
    assert "deepresearch.tool_accounting_consistent" in report.failure_categories


def test_detects_report_level_verification_failures_misrouted_to_full_research_replans() -> None:
    payload = deepresearch_snapshot()
    payload["run"].update(
        {
            "status": "cancelled",
            "current_stage": "cancelled",
            "completed_at": "2026-09-21T00:00:08Z",
            "updated_at": "2026-09-21T00:00:08Z",
        }
    )
    payload["run"]["usage"]["usage"].update({"tool_calls": 2, "replans": 2})
    payload["events"] = [
        payload["events"][0],
        {
            "event_id": 2,
            "event_type": "tool.completed",
            "stage": "executing",
            "payload": {"tool_call_id": "tool_1", "tool_name": "web_search"},
            "created_at": "2026-09-21T00:00:03Z",
        },
        {
            "event_id": 3,
            "event_type": "verification.completed",
            "stage": "verifying",
            "payload": {
                "passed": False,
                "failures": ["citation_integrity", "claim_support", "source_diversity"],
                "recovery_action": "replan",
            },
            "created_at": "2026-09-21T00:00:04Z",
        },
        {
            "event_id": 4,
            "event_type": "plan.replanning",
            "stage": "replanning",
            "payload": {
                "failures": ["citation_integrity", "claim_support", "source_diversity"],
                "recovery_action": "replan",
            },
            "created_at": "2026-09-21T00:00:04.100Z",
        },
        {
            "event_id": 5,
            "event_type": "tool.completed",
            "stage": "executing",
            "payload": {"tool_call_id": "tool_2", "tool_name": "web_search"},
            "created_at": "2026-09-21T00:00:06Z",
        },
        {
            "event_id": 6,
            "event_type": "verification.completed",
            "stage": "verifying",
            "payload": {
                "passed": False,
                "failures": ["citation_integrity", "claim_support", "source_diversity"],
                "recovery_action": "replan",
            },
            "created_at": "2026-09-21T00:00:07Z",
        },
        {
            "event_id": 7,
            "event_type": "plan.replanning",
            "stage": "replanning",
            "payload": {
                "failures": ["citation_integrity", "claim_support", "source_diversity"],
                "recovery_action": "replan",
            },
            "created_at": "2026-09-21T00:00:07.100Z",
        },
        {
            "event_id": 8,
            "event_type": "run.cancelled",
            "stage": "cancelled",
            "payload": {},
            "created_at": "2026-09-21T00:00:08Z",
        },
    ]
    payload["tool_calls"].append(
        {
            "tool_call_id": "tool_2",
            "task_id": "task_1",
            "tool_name": "web_search",
            "source_mode": "web",
            "status": "completed",
            "args_hash": "d" * 64,
            "result_present": True,
            "error_code": None,
            "created_at": "2026-09-21T00:00:05Z",
            "completed_at": "2026-09-21T00:00:06Z",
        }
    )

    trace = adapt_deepresearch_snapshot(payload)
    report = evaluate_trace(trace)
    checks = {check.check_id: check for check in report.checks}

    assert trace.metadata["deepresearch.verification_failure_count"] == 2
    assert trace.metadata["deepresearch.repeated_verification_failure_count"] == 1
    assert trace.metadata["deepresearch.misrouted_report_replan_count"] == 2
    assert trace.metadata["deepresearch.tool_calls_after_first_misrouted_replan"] == 1
    assert checks["deepresearch.verification_recovery_routing"].status == CheckStatus.FAIL
    assert "deepresearch.verification_recovery_routing" in report.failure_categories


def test_detects_deepresearch_contract_citation_checkpoint_and_source_failures() -> None:
    payload = deepresearch_snapshot()
    payload["contracts"][0]["passed"] = False
    payload["report"] = "Unsupported citation [ev_missing]"
    payload["checkpoints"][0]["integrity_valid"] = False
    payload["evidence"][0]["source_mode"] = "graphrag"

    report = evaluate_trace(adapt_deepresearch_snapshot(payload))

    assert report.passed is False
    assert {
        "deepresearch.required_contracts_pass",
        "deepresearch.citations_valid",
        "deepresearch.checkpoints_intact",
        "deepresearch.source_isolation",
    }.issubset(report.failure_categories)


def test_deepresearch_http_import_is_first_class() -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))
    response = client.post("/v1/traces/import/deepresearch", json=deepresearch_snapshot())

    assert response.status_code == 200
    assert response.json()["evaluation"]["passed"] is True
    assert response.json()["trace"]["source"]["kind"] == "deepresearch_agent_harness"


def test_exports_deepresearch_sqlite_without_raw_tool_or_checkpoint_content(tmp_path) -> None:
    database = tmp_path / "deepresearch.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (run_id TEXT, session_id TEXT, trigger_message_id TEXT, source_mode TEXT,
          workflow_mode TEXT, status TEXT, current_stage TEXT, usage_json TEXT, error_code TEXT,
          error_message TEXT, created_at TEXT, started_at TEXT, completed_at TEXT, updated_at TEXT);
        CREATE TABLE messages (message_id TEXT, run_id TEXT, role TEXT, content TEXT);
        CREATE TABLE run_events (event_id INTEGER, run_id TEXT, event_type TEXT, stage TEXT,
          payload_json TEXT, created_at TEXT);
        CREATE TABLE tool_calls (tool_call_id TEXT, run_id TEXT, task_id TEXT, tool_name TEXT,
          source_mode TEXT, status TEXT, args_json TEXT, result_json TEXT, error_code TEXT,
          created_at TEXT, completed_at TEXT);
        CREATE TABLE evidence (evidence_id TEXT, run_id TEXT, tool_call_id TEXT, source_mode TEXT,
          provider TEXT, source_id TEXT, content_hash TEXT, metadata_json TEXT, invalidated_at TEXT,
          created_at TEXT);
        CREATE TABLE contract_checks (check_id TEXT, run_id TEXT, kind TEXT, required INTEGER,
          passed INTEGER, verifier TEXT, verifier_version TEXT);
        CREATE TABLE checkpoints (checkpoint_id TEXT, run_id TEXT, version INTEGER, stage TEXT,
          state_json TEXT, state_hash TEXT, created_at TEXT);
        CREATE TABLE plans (plan_id TEXT, run_id TEXT);
        CREATE TABLE tasks (task_id TEXT, run_id TEXT);
        """
    )
    state_json = '{"schema_version":1,"status":"completed"}'
    state_hash = hashlib.sha256(state_json.encode()).hexdigest()
    connection.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "run_db",
            "session_db",
            "message_user",
            "web",
            "deep_research",
            "completed",
            "completed",
            '{"usage":{"tool_calls":1}}',
            None,
            None,
            "2026-09-21T00:00:00Z",
            "2026-09-21T00:00:01Z",
            "2026-09-21T00:00:05Z",
            "2026-09-21T00:00:05Z",
        ),
    )
    connection.executemany(
        "INSERT INTO messages VALUES (?,?,?,?)",
        [
            ("message_user", None, "user", "research task"),
            ("message_assistant", "run_db", "assistant", "answer [ev_1]"),
        ],
    )
    connection.execute(
        "INSERT INTO run_events VALUES (?,?,?,?,?,?)",
        (1, "run_db", "run.completed", "completed", '{"verified":true}', "2026-09-21T00:00:05Z"),
    )
    connection.execute(
        "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "tool_1",
            "run_db",
            "task_1",
            "web_search",
            "web",
            "completed",
            '{"query":"secret-free"}',
            '{"result":1}',
            None,
            "2026-09-21T00:00:02Z",
            "2026-09-21T00:00:03Z",
        ),
    )
    connection.execute(
        "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "ev_1",
            "run_db",
            "tool_1",
            "web",
            "web_search",
            "source-1",
            "d" * 64,
            '{"domain":"example.com","url":"https://example.com/source","extra":{"evidence_scope":"fetched_page","page_fetch_status":"completed"}}',
            None,
            "2026-09-21T00:00:03Z",
        ),
    )
    connection.execute(
        "INSERT INTO contract_checks VALUES (?,?,?,?,?,?,?)",
        ("contract_1", "run_db", "citation_integrity", 1, 1, "deterministic", "1"),
    )
    connection.execute(
        "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
        ("chk_1", "run_db", 1, "completed", state_json, state_hash, "2026-09-21T00:00:05Z"),
    )
    connection.execute("INSERT INTO plans VALUES (?,?)", ("plan_1", "run_db"))
    connection.execute("INSERT INTO tasks VALUES (?,?)", ("task_1", "run_db"))
    connection.commit()
    connection.close()

    snapshot = export_deepresearch_sqlite_snapshot(database, "run_db")
    serialized = json.dumps(snapshot)

    assert snapshot["task"] == "research task"
    assert snapshot["checkpoints"][0]["integrity_valid"] is True
    assert snapshot["tool_calls"][0]["args_hash"]
    assert snapshot["evidence"][0]["evidence_scope"] == "fetched_page"
    assert snapshot["evidence"][0]["page_fetch_status"] == "completed"
    assert "secret-free" not in serialized
    assert "state_json" not in serialized
