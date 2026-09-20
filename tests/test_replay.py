from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from agenttracelab.adapters import adapt_wasmhatch_journal
from agenttracelab.failures import FailureCategory
from agenttracelab.replay import ReplayManifest, run_replay


def _write_manifest(tmp_path: Path, tasks: list[dict], target: dict | None = None) -> Path:
    manifest = {
        "schema_version": "agenttracelab.replay.v1",
        "replay_id": "mock-agent-regression",
        "version": "1.0.0",
        "target": target
        or {
            "endpoint": "http://agent.local/v1/run",
            "timeout_seconds": 2,
            "max_response_bytes": 65536,
        },
        "tasks": tasks,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_replays_all_trace_adapters_and_preserves_expected_bad_case(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
    otlp_payload: dict,
) -> None:
    normalized = adapt_wasmhatch_journal(complete_journal).model_dump(mode="json")
    fixtures = {
        "wasmhatch": complete_journal,
        "normalized": normalized,
        "otlp": otlp_payload,
        "bad-case": incomplete_journal,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        fixture_id = json.loads(request.content)["fixture"]
        return httpx.Response(200, json={"result": fixtures[fixture_id]})

    tasks = [
        {
            "task_id": "wasmhatch-pass",
            "request": {"fixture": "wasmhatch"},
            "adapter": "wasmhatch",
            "response_path": "result",
        },
        {
            "task_id": "normalized-pass",
            "request": {"fixture": "normalized"},
            "adapter": "normalized",
            "response_path": "result",
        },
        {
            "task_id": "otlp-pass",
            "request": {"fixture": "otlp"},
            "adapter": "otlp",
            "response_path": "result",
        },
        {
            "task_id": "expected-bad-case",
            "request": {"fixture": "bad-case"},
            "adapter": "wasmhatch",
            "response_path": "result",
            "expected_pass": False,
            "expected_check_statuses": {"agent.decision_evidence": "fail"},
        },
    ]
    client = httpx.Client(transport=httpx.MockTransport(handler))

    report = run_replay(_write_manifest(tmp_path, tasks), client=client)

    assert report.passed is True
    assert report.matched_tasks == report.task_count == report.completed_tasks == 4
    assert report.evaluation_pass_rate == 75.0
    bad_case = report.results[-1]
    assert bad_case.matched_expectation is True
    assert FailureCategory.DECISION_EVIDENCE in {item.category for item in bad_case.failures}


def test_replay_reports_transport_error_without_leaking_target_body(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret transport detail", request=request)

    tasks = [{"task_id": "offline", "request": {}, "adapter": "normalized"}]
    client = httpx.Client(transport=httpx.MockTransport(handler))

    report = run_replay(_write_manifest(tmp_path, tasks), client=client)

    result = report.results[0]
    assert report.passed is False
    assert result.status == "transport_error"
    assert result.error == "target request failed: ConnectError"
    assert result.failures[0].category == FailureCategory.TRANSPORT


def test_replay_enforces_streaming_response_limit(tmp_path: Path) -> None:
    tasks = [{"task_id": "oversized", "request": {}, "adapter": "normalized"}]
    target = {
        "endpoint": "http://agent.local/v1/run",
        "timeout_seconds": 2,
        "max_response_bytes": 1024,
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * 1025, request=request)
        )
    )

    report = run_replay(_write_manifest(tmp_path, tasks, target), client=client)

    result = report.results[0]
    assert result.status == "contract_error"
    assert result.error == "target response exceeds configured size limit"


def test_replay_reports_missing_response_path(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "missing-path",
            "request": {},
            "adapter": "normalized",
            "response_path": "result.trace",
        }
    ]
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}, request=request))
    )

    report = run_replay(_write_manifest(tmp_path, tasks), client=client)

    assert report.results[0].status == "contract_error"
    assert "response_path does not exist" in (report.results[0].error or "")


def test_replay_requires_configured_bearer_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_REPLAY_TOKEN", raising=False)
    tasks = [{"task_id": "auth", "request": {}, "adapter": "normalized"}]
    target = {
        "endpoint": "https://agent.example/v1/run",
        "bearer_token_env": "AGENT_REPLAY_TOKEN",
    }

    with pytest.raises(ValueError, match="missing replay bearer token"):
        run_replay(_write_manifest(tmp_path, tasks, target))


def test_replay_manifest_rejects_endpoint_credentials() -> None:
    with pytest.raises(ValidationError, match="embedded credentials"):
        ReplayManifest.model_validate(
            {
                "schema_version": "agenttracelab.replay.v1",
                "replay_id": "unsafe",
                "version": "1.0.0",
                "target": {"endpoint": "https://user:password@example.com/run"},
                "tasks": [{"task_id": "one", "request": {}, "adapter": "normalized"}],
            }
        )


def test_checked_in_replay_manifest_is_valid() -> None:
    path = Path(__file__).parents[1] / "evaluation" / "replays" / "example" / "v1" / "manifest.json"

    manifest = ReplayManifest.model_validate_json(path.read_text(encoding="utf-8"))

    assert manifest.replay_id == "office-agent-regression"
    assert manifest.tasks[0].adapter == "wasmhatch"
