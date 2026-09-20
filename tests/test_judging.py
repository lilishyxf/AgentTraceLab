from __future__ import annotations

import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agenttracelab.api import create_app
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.judging import (
    ContentPolicy,
    JudgeBlockedError,
    JudgeConfig,
    JudgeProtocolError,
    JudgeVerdict,
    OpenAICompatibleJudge,
)
from agenttracelab.service import AgentTraceService
from agenttracelab.storage import Database


def _judge_content(*, evidence_span_id: str = "span-1") -> str:
    dimensions = [
        "trajectory_coherence",
        "tool_choice",
        "result_grounding",
        "safety_awareness",
        "efficiency",
    ]
    return json.dumps(
        {
            "verdict": "pass",
            "summary": "The trajectory is coherent and grounded in the recorded tool result.",
            "dimensions": [
                {
                    "name": name,
                    "score": 4 if name != "efficiency" else 3,
                    "rationale": f"Evidence supports {name}.",
                    "evidence_span_ids": [evidence_span_id],
                }
                for name in dimensions
            ],
            "limitations": ["The content policy may omit task text."],
        }
    )


def _response(request: httpx.Request, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "judge-response-1",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
        },
        request=request,
    )


def _config(**overrides) -> JudgeConfig:
    return JudgeConfig(
        base_url="https://judge.example/v1",
        model="judge-model",
        api_key="secret-test-key",
        max_attempts=overrides.pop("max_attempts", 1),
        content_policy=overrides.pop("content_policy", ContentPolicy.METADATA_ONLY),
        **overrides,
    )


def _service() -> AgentTraceService:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    return AgentTraceService(database)


def test_openai_compatible_judge_is_separate_strict_and_persisted(complete_journal: dict) -> None:
    app = create_app("sqlite+pysqlite:///:memory:")
    service = app.state.service
    trace, deterministic = service.import_wasmhatch(complete_journal)
    observed: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers["authorization"]
        observed["payload"] = json.loads(request.content)
        return _response(request, _judge_content(evidence_span_id=trace.spans[0].span_id))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = service.judge_trace(
        trace.trace_id,
        OpenAICompatibleJudge(_config(), client=client),
    )

    assert observed["url"] == "https://judge.example/v1/chat/completions"
    assert observed["authorization"] == "Bearer secret-test-key"
    assert observed["payload"]["response_format"] == {"type": "json_object"}
    user_prompt = observed["payload"]["messages"][1]["content"]
    assert trace.task not in user_prompt
    assert "[content omitted]" in user_prompt
    assert report.deterministic_evaluation_id == deterministic.evaluation_id
    assert report.verdict == JudgeVerdict.PASS
    assert report.qualitative_score == 95.0
    assert report.token_usage["total_tokens"] == 200
    assert service.get_latest_judge_report(trace.trace_id) == report
    response = TestClient(app).get(f"/v1/judgments/{trace.trace_id}/latest")
    assert response.status_code == 200
    assert response.json()["judge_report_id"] == report.judge_report_id


def test_judge_retries_retryable_provider_response(complete_journal: dict) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(complete_journal)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return _response(request, _judge_content(evidence_span_id=trace.spans[0].span_id))

    judge = OpenAICompatibleJudge(
        _config(max_attempts=2),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    report = judge.evaluate(trace, deterministic)

    assert report.request_attempts == 2
    assert attempts == 2


def test_judge_retries_empty_json_mode_content(complete_journal: dict) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(complete_journal)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "" if attempts == 1 else _judge_content(evidence_span_id=trace.spans[0].span_id)
        return _response(request, content)

    judge = OpenAICompatibleJudge(
        _config(max_attempts=2),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    report = judge.evaluate(trace, deterministic)

    assert report.request_attempts == 2
    assert attempts == 2


def test_judge_rejects_malformed_or_unverifiable_output(complete_journal: dict) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(complete_journal)

    malformed_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: _response(request, '{"verdict":"pass"}'))
    )
    unknown_evidence_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: _response(request, _judge_content(evidence_span_id="invented-span"))
        )
    )

    with pytest.raises(JudgeProtocolError, match="strict schema"):
        OpenAICompatibleJudge(_config(), client=malformed_client).evaluate(trace, deterministic)
    with pytest.raises(JudgeProtocolError, match="unknown span IDs"):
        OpenAICompatibleJudge(_config(), client=unknown_evidence_client).evaluate(
            trace,
            deterministic,
        )


def test_judge_blocks_external_request_when_privacy_gate_fails(complete_journal: dict) -> None:
    service = _service()
    trace, _ = service.import_wasmhatch(copy.deepcopy(complete_journal))
    unsafe_trace = trace.model_copy(
        update={"metadata": {**trace.metadata, "credential": "sk-secretvalue123"}}
    )
    deterministic = evaluate_trace(unsafe_trace)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response(request, _judge_content(evidence_span_id=trace.spans[0].span_id))

    judge = OpenAICompatibleJudge(
        _config(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(JudgeBlockedError, match="privacy gate"):
        judge.evaluate(unsafe_trace, deterministic)
    assert called is False


def test_judge_content_requires_explicit_policy(complete_journal: dict) -> None:
    service = _service()
    trace, deterministic = service.import_wasmhatch(complete_journal)
    observed_prompt = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_prompt
        observed_prompt = json.loads(request.content)["messages"][1]["content"]
        return _response(request, _judge_content(evidence_span_id=trace.spans[0].span_id))

    judge = OpenAICompatibleJudge(
        _config(content_policy=ContentPolicy.INCLUDE_CONTENT),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    judge.evaluate(trace, deterministic)

    assert trace.task in observed_prompt
