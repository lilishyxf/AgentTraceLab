from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr
from test_deepresearch_adapter import deepresearch_snapshot

from agenttracelab.claim_evidence import (
    ClaimEvidencePolicy,
    ClaimJudgeConfig,
    ClaimJudgeError,
    ClaimVerdict,
    OpenAICompatibleClaimJudge,
    verify_deepresearch_claims,
)


def test_claim_verifier_separates_citation_integrity_from_semantic_truth() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = (
        "The research source supports the conclusion [ev_gold].\n\n"
        "A second factual assertion has no evidence."
    )
    report = verify_deepresearch_claims(
        payload,
        policy=ClaimEvidencePolicy(min_claim_chars=8, min_lexical_overlap=0.05),
    )

    assert report.metrics.claim_count == 2
    assert report.metrics.cited_claim_count == 1
    assert report.metrics.citation_coverage == 0.5
    assert report.metrics.citation_resolution == 1.0
    assert report.metrics.proxy_supported_claim_count == 1
    assert report.citation_gate_passed is False
    assert report.semantic_gate_passed is None
    assert report.recommendation == "human_review"
    assert report.claims[0].deterministic_verdict == ClaimVerdict.PROXY_SUPPORTED
    assert report.claims[1].deterministic_verdict == ClaimVerdict.MISSING_CITATION
    assert "not proof of factual entailment" in report.limitations[0]


def test_claim_verifier_detects_unresolved_citation() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = "This statement cites evidence that was not exported [ev_missing]."
    report = verify_deepresearch_claims(payload, policy=ClaimEvidencePolicy(min_claim_chars=8))

    assert report.metrics.citation_resolution == 0.0
    assert report.claims[0].missing_evidence_ids == ("ev_missing",)
    assert report.claims[0].final_verdict == ClaimVerdict.UNRESOLVED_CITATION


def test_claim_verifier_applies_trailing_paragraph_citation_to_prior_sentences() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = "The source supports the conclusion. The same source records the result [ev_gold]."
    report = verify_deepresearch_claims(
        payload,
        policy=ClaimEvidencePolicy(min_claim_chars=8, min_lexical_overlap=0.0),
    )

    assert report.metrics.claim_count == 2
    assert report.metrics.cited_claim_count == 2
    assert report.metrics.citation_coverage == 1.0
    assert all(item.citation_ids == ("ev_gold",) for item in report.claims)


def test_claim_verifier_ignores_markdown_table_structure_but_grades_data_rows() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = """# Report

| Event category | Recorded span |
| --- | --- |
| Model generation | The trace records model output [ev_gold] |
"""
    report = verify_deepresearch_claims(
        payload,
        policy=ClaimEvidencePolicy(min_claim_chars=8, min_lexical_overlap=0.0),
    )

    assert report.metrics.claim_count == 1
    assert report.metrics.cited_claim_count == 1
    assert report.metrics.citation_coverage == 1.0
    assert report.claims[0].text == "Model generation   The trace records model output"


def test_independent_claim_judge_upgrades_proxy_to_semantic_verdict() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = "The research source supports the conclusion [ev_gold]."

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        assert request_payload["temperature"] == 0
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdicts": [
                                        {
                                            "claim_id": "claim_0001",
                                            "verdict": "supported",
                                            "rationale": "The exported summary directly supports it.",
                                            "evidence_ids": ["ev_gold"],
                                        }
                                    ],
                                    "limitations": ["Summary-only review."],
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OpenAICompatibleClaimJudge(
        ClaimJudgeConfig(
            base_url="https://judge.test/v1",
            model="judge-model",
            api_key=SecretStr("not-written"),
        ),
        client=client,
    )
    report = verify_deepresearch_claims(
        payload,
        policy=ClaimEvidencePolicy(min_claim_chars=8, min_citation_coverage=1.0),
        judge=judge,
    )

    assert report.semantic_gate_passed is True
    assert report.recommendation == "pass"
    assert report.metrics.semantic_support_rate == 1.0
    assert report.claims[0].final_verdict == ClaimVerdict.SUPPORTED
    assert report.judge_model == "judge-model"


def test_claim_judge_retries_protocol_error_with_bounded_correction() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = "The research source supports the conclusion [ev_gold]."
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        verdict = "not found" if len(requests) == 1 else "unverifiable"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdicts": [
                                        {
                                            "claim_id": "claim_0001",
                                            "verdict": verdict,
                                            "rationale": "The evidence does not establish the whole claim.",
                                            "evidence_ids": ["ev_gold"],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OpenAICompatibleClaimJudge(
        ClaimJudgeConfig(
            base_url="https://judge.test/v1",
            model="judge-model",
            api_key=SecretStr("secret"),
        ),
        client=client,
        sleep=lambda _seconds: None,
    )

    report = verify_deepresearch_claims(payload, judge=judge)

    assert len(requests) == 2
    assert "previous response was rejected" in requests[1]["messages"][-1]["content"].lower()
    assert report.claims[0].final_verdict == ClaimVerdict.UNVERIFIABLE


def test_claim_judge_batches_large_claim_sets_without_losing_verdicts() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = "\n\n".join(
        f"Supported factual statement number {index} [ev_gold]." for index in range(1, 6)
    )
    requested_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        claims = json.loads(request_payload["messages"][1]["content"])["claims"]
        requested_batches.append([item["claim_id"] for item in claims])
        content = {
            "verdicts": [
                {
                    "claim_id": item["claim_id"],
                    "verdict": "supported",
                    "rationale": "The supplied evidence supports this claim.",
                    "evidence_ids": ["ev_gold"],
                }
                for item in claims
            ],
            "limitations": ["Batch-local review."],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OpenAICompatibleClaimJudge(
        ClaimJudgeConfig(
            base_url="https://judge.test/v1",
            model="judge-model",
            api_key=SecretStr("secret"),
            max_claims_per_request=2,
        ),
        client=client,
    )

    report = verify_deepresearch_claims(
        payload,
        policy=ClaimEvidencePolicy(min_claim_chars=8),
        judge=judge,
    )

    assert requested_batches == [
        ["claim_0001", "claim_0002"],
        ["claim_0003", "claim_0004"],
        ["claim_0005"],
    ]
    assert report.metrics.judged_claim_count == 5
    assert report.metrics.supported_claim_count == 5
    assert report.limitations.count("Batch-local review.") == 1


def test_task_fulfillment_judge_rejects_supported_non_answer() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = (
        "The source does not define task, trial, grader, transcript, or outcome [ev_gold]. "
        "Therefore the requested explanation cannot be provided [ev_gold]."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        system = request_payload["messages"][0]["content"]
        if "task-fulfillment verifier" in system:
            content = {
                "verdict": "not_fulfilled",
                "rationale": "The report states that it cannot provide the requested definitions.",
                "missing_requirements": ["Define the requested evaluation terms."],
            }
        else:
            content = {
                "verdicts": [
                    {
                        "claim_id": "claim_0001",
                        "verdict": "supported",
                        "rationale": "The evidence supports the report's statement of insufficiency.",
                        "evidence_ids": ["ev_gold"],
                    },
                    {
                        "claim_id": "claim_0002",
                        "verdict": "supported",
                        "rationale": "The evidence packet is the basis for the stated limitation.",
                        "evidence_ids": ["ev_gold"],
                    },
                ]
            }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OpenAICompatibleClaimJudge(
        ClaimJudgeConfig(
            base_url="https://judge.test/v1",
            model="judge-model",
            api_key=SecretStr("secret"),
        ),
        client=client,
    )

    report = verify_deepresearch_claims(
        payload,
        judge=judge,
        task_prompt="Define task, trial, grader, transcript, and outcome.",
    )

    assert report.semantic_gate_passed is True
    assert report.task_fulfillment_gate_passed is False
    assert report.task_fulfillment_verdict == "not_fulfilled"
    assert report.task_missing_requirements == ("Define the requested evaluation terms.",)
    assert report.recommendation == "hold"


def test_claim_judge_rejects_unknown_evidence_ids() -> None:
    payload = deepresearch_snapshot()
    payload["report"] = "The research source supports the conclusion [ev_gold]."
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "verdicts": [
                                {
                                    "claim_id": "claim_0001",
                                    "verdict": "supported",
                                    "rationale": "Unsupported evidence reference.",
                                    "evidence_ids": ["ev_invented"],
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response)))
    judge = OpenAICompatibleClaimJudge(
        ClaimJudgeConfig(
            base_url="https://judge.test/v1",
            model="judge-model",
            api_key=SecretStr("secret"),
        ),
        client=client,
    )

    with pytest.raises(ClaimJudgeError, match="outside the supplied claim packet"):
        verify_deepresearch_claims(payload, judge=judge)
