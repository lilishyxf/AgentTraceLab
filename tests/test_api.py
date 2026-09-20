from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from agenttracelab.api import create_app


def test_health_reports_current_version() -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.1.0"}


def test_import_retrieve_and_evaluate_wasmhatch_trace(complete_journal: dict) -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    imported = client.post("/v1/traces/import/wasmhatch", json=complete_journal)

    assert imported.status_code == 200
    body = imported.json()
    assert body["trace"]["trace_id"] == complete_journal["runId"]
    assert body["evaluation"]["passed"] is True

    trace = client.get(f"/v1/traces/{complete_journal['runId']}")
    evaluation = client.get(f"/v1/evaluations/{complete_journal['runId']}/latest")

    assert trace.status_code == 200
    assert evaluation.status_code == 200
    assert evaluation.json()["score"] == 100.0


def test_api_rejects_invalid_wasmhatch_sequence(complete_journal: dict) -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))
    payload = copy.deepcopy(complete_journal)
    payload["events"][1]["sequence"] = 7

    response = client.post("/v1/traces/import/wasmhatch", json=payload)

    assert response.status_code == 422
    assert "contiguous sequence" in response.json()["detail"]


def test_api_imports_openinference_otlp_batch(otlp_payload: dict) -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    response = client.post("/v1/traces/import/otlp", json=otlp_payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["trace"]["source"]["kind"] == "openinference"
    assert result["evaluation"]["passed"] is True


def test_api_rejects_malformed_otlp_payload() -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    response = client.post("/v1/traces/import/otlp", json={"resourceSpans": []})

    assert response.status_code == 422
    assert "resourceSpans" in response.json()["detail"]


def test_api_compares_baseline_and_candidate(complete_journal: dict, incomplete_journal: dict) -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))
    assert client.post("/v1/traces/import/wasmhatch", json=incomplete_journal).status_code == 200
    assert client.post("/v1/traces/import/wasmhatch", json=complete_journal).status_code == 200

    response = client.post(
        "/v1/comparisons",
        json={
            "baseline_trace_id": incomplete_journal["runId"],
            "candidate_trace_id": complete_journal["runId"],
        },
    )

    assert response.status_code == 200
    assert response.json()["recommendation"] == "promote"
