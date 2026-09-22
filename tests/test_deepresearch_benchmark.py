from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agenttracelab.claim_evidence import (
    ClaimJudgeOutput,
    ClaimJudgeVerdict,
    ClaimVerdict,
    verify_deepresearch_claims,
)
from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkRunner,
    DeepResearchBenchmarkTask,
    _evaluate_task_acceptance,
    _trial_client_message_id,
    benchmark_tasks_sha256,
    load_deepresearch_benchmark_manifest,
    rescore_deepresearch_benchmark,
)

RELEASE_CANDIDATE_MANIFEST = Path("evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.1.json")
RELEASE_CANDIDATE_REVIEW = Path(
    "evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.1.review-template.json"
)
RELEASE_CANDIDATE_REFERENCES = Path(
    "evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.1.reference-template.json"
)
RELEASE_CANDIDATE_REFERENCE_DRAFT = Path(
    "evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.1.reference-draft.json"
)
RELEASE_CANDIDATE_V02_MANIFEST = Path(
    "evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.2.provisional-judge.json"
)
RELEASE_CANDIDATE_V02_REVIEW = Path(
    "evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.2.review-template.json"
)
RELEASE_CANDIDATE_V02_REFERENCE_DRAFT = Path(
    "evaluation/benchmarks/deepresearch/v1/release-candidate-suite-v0.2.reference-draft.json"
)


def test_trial_client_message_id_is_bounded_for_failure_derived_suite_names() -> None:
    message_id = _trial_client_message_id(
        "benchmark-" + ("release-candidate-challenge-" * 10),
        "task-" + ("long-failure-cluster-" * 10),
        123,
        nonce="a" * 64,
    )

    assert len(message_id) <= 128
    assert message_id.startswith("atl-")
    assert message_id.endswith("-" + ("a" * 32))


def _manifest(tmp_path: Path, *, trials: int = 2) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.deepresearch-benchmark.v1",
                "benchmark_id": "deepresearch-smoke",
                "version": "1.0.0",
                "evidence_class": "integration",
                "target_revision": "test-revision",
                "trials_per_task": trials,
                "poll_interval_seconds": 0.05,
                "timeout_seconds": 5,
                "tasks": [
                    {
                        "task_id": "research-a",
                        "prompt": "Research A",
                        "source_mode": "web",
                        "acceptance_contract": {
                            "contract_id": "research-a-v1",
                            "min_report_chars": 12,
                            "min_independent_sources": 1,
                            "required_concepts": [
                                {
                                    "criterion_id": "supported-answer",
                                    "aliases": ["Supported answer"],
                                }
                            ],
                        },
                    },
                    {
                        "task_id": "research-b",
                        "prompt": "Research B",
                        "source_mode": "web",
                        "acceptance_contract": {
                            "contract_id": "research-b-v1",
                            "min_report_chars": 12,
                            "min_independent_sources": 1,
                            "required_concepts": [
                                {
                                    "criterion_id": "supported-answer",
                                    "aliases": ["Supported answer"],
                                }
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _snapshot(run_id: str) -> dict:
    return {
        "schema_version": "deepresearch-agent.run-snapshot.v1",
        "run": {
            "run_id": run_id,
            "session_id": f"session-{run_id}",
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
        "task": "Research task",
        "report": "Supported answer [ev_1]",
        "events": [
            {
                "event_id": 1,
                "event_type": "plan.replanning",
                "stage": "replanning",
                "payload": {"recovery_action": "replan", "recovery_reason": "insufficient evidence"},
                "created_at": "2026-09-21T00:00:01Z",
            },
            {
                "event_id": 2,
                "event_type": "tool.completed",
                "stage": "executing",
                "payload": {"tool_call_id": "tool_1"},
                "created_at": "2026-09-21T00:00:03Z",
            },
            {
                "event_id": 3,
                "event_type": "verification.completed",
                "stage": "verifying",
                "payload": {"passed": True},
                "created_at": "2026-09-21T00:00:04Z",
            },
            {
                "event_id": 4,
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
                "evidence_id": "ev_1",
                "tool_call_id": "tool_1",
                "source_mode": "web",
                "provider": "web_search",
                "source_id": "https://example.com/source",
                "content_hash": "b" * 64,
                "domain": "example.com",
                "title": "Supported answer",
                "summary": "This source contains a supported answer to the research task.",
                "url": "https://example.com/source",
                "evidence_scope": "fetched_page",
                "page_fetch_status": "completed",
            },
            {
                "evidence_id": "ev_2",
                "tool_call_id": "tool_1",
                "source_mode": "web",
                "provider": "web_search",
                "source_id": "https://docs.example.com/fallback",
                "content_hash": "d" * 64,
                "domain": "docs.example.com",
                "title": "Fallback snippet",
                "summary": "The full page was unavailable, so this bounded snippet was retained.",
                "url": "https://docs.example.com/fallback",
                "evidence_scope": "search_snippet",
                "page_fetch_status": "failed",
                "page_fetch_error_code": "http_status",
            },
        ],
        "contracts": [
            {
                "check_id": "citation",
                "kind": "citation_integrity",
                "required": True,
                "passed": True,
                "verifier": "deterministic",
                "verifier_version": "1",
            }
        ],
        "checkpoints": [
            {
                "checkpoint_id": "checkpoint_1",
                "version": 1,
                "stage": "completed",
                "state_hash": "c" * 64,
                "integrity_valid": True,
                "created_at": "2026-09-21T00:00:05Z",
            }
        ],
        "plan_count": 2,
        "task_count": 1,
        "provenance": {"source": "integration-test"},
    }


def test_runs_repeated_isolated_trials_and_retains_evidence(tmp_path: Path) -> None:
    counters = {"sessions": 0, "runs": 0}
    submitted_messages: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            counters["sessions"] += 1
            return httpx.Response(201, json={"session_id": f"session-{counters['sessions']}"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            counters["runs"] += 1
            submitted_messages.append(json.loads(request.content))
            return httpx.Response(202, json={"run_id": f"run-{counters['runs']}", "status": "queued"})
        if request.method == "GET" and request.url.path.startswith("/api/v1/runs/"):
            return httpx.Response(
                200,
                json={"run_id": request.url.path.rsplit("/", 1)[-1], "status": "completed"},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    output = tmp_path / "results"
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=output,
        client=client,
        snapshot_exporter=lambda _database, run_id: _snapshot(run_id),
        sleep=lambda _seconds: None,
    ) as runner:
        report = runner.run(_manifest(tmp_path))

    assert counters == {"sessions": 4, "runs": 4}
    assert submitted_messages and all(item["evaluation_run"] is True for item in submitted_messages)
    assert report.status == "completed"
    assert report.evidence_class == "integration"
    assert report.metrics.total_trials == 4
    assert report.metrics.pass_rate == 1.0
    assert report.metrics.process_pass_rate == 1.0
    assert report.metrics.citation_gate_pass_rate == 1.0
    assert report.metrics.semantic_evaluation_coverage == 0
    assert report.metrics.semantic_gate_pass_rate is None
    assert report.metrics.task_acceptance_evaluated_trials == 4
    assert report.metrics.task_acceptance_passed_trials == 4
    assert report.metrics.task_acceptance_evaluation_coverage == 1.0
    assert report.metrics.task_acceptance_pass_rate == 1.0
    assert all(trial.task_acceptance_passed is True for trial in report.trials)
    assert report.metrics.evidence_verified_passed_trials == 0
    assert report.metrics.evidence_unverified_potential_passes == 4
    assert report.metrics.evidence_supported_pass_rate_lower_bound == 0.0
    assert report.metrics.evidence_supported_pass_rate_upper_bound == 1.0
    assert report.judge_separation == "not_configured"
    assert report.metrics.pass_at_1_rate == 1.0
    assert report.metrics.pass_power_k_rate == 1.0
    assert report.metrics.mean_token_count == 120
    assert report.metrics.mean_failed_tool_call_count == 0
    assert report.metrics.mean_tool_failure_rate == 0
    assert report.metrics.tool_failure_free_rate == 1
    assert report.metrics.recovered_tool_failure_count == 0
    assert report.metrics.unrecovered_tool_failure_count == 0
    assert report.metrics.mean_citation_validity == 1.0
    assert report.trials[0].full_page_evidence_count == 1
    assert report.trials[0].full_page_evidence_rate == 0.5
    assert report.metrics.full_page_evidence_count == 4
    assert report.metrics.search_snippet_evidence_count == 4
    assert report.metrics.mean_full_page_evidence_rate == 0.5
    assert report.metrics.page_fetch_attempt_count == 8
    assert report.metrics.page_fetch_failed_count == 4
    assert report.metrics.mean_page_fetch_success_rate == 0.5
    assert report.metrics.page_fetch_failure_categories == {"http_status": 4}
    assert len({trial.session_id for trial in report.trials}) == 4
    assert (output / "benchmark-report.json").is_file()
    assert (output / "benchmark-summary.md").is_file()
    assert (output / report.trials[0].snapshot_path).is_file()
    assert (output / report.trials[0].trace_path).is_file()
    assert (output / report.trials[0].evaluation_path).is_file()
    summary = (output / "benchmark-summary.md").read_text(encoding="utf-8")
    assert "Full-page evidence rate: 50.0%" in summary
    assert "Evidence-supported pass-rate bounds: 0.0%–100.0%" in summary
    assert "Page-fetch success rate: 50.0%" in summary
    assert "Page-fetch failure categories: http_status=4" in summary
    assert "not production model-quality claims" in " ".join(report.limitations)


def test_task_acceptance_contract_detects_incomplete_supported_answer() -> None:
    task = DeepResearchBenchmarkTask.model_validate(
        {
            "task_id": "contract-case",
            "prompt": "Explain two required concepts.",
            "acceptance_contract": {
                "contract_id": "contract-case-v1",
                "min_report_chars": 30,
                "min_independent_sources": 2,
                "required_concepts": [
                    {"criterion_id": "alpha", "aliases": ["Alpha"]},
                    {"criterion_id": "beta", "aliases": ["Beta"]},
                ],
                "forbidden_phrases": ["unsupported universal claim"],
            },
        }
    )
    snapshot = _snapshot("run-contract")
    snapshot["report"] = "Alpha is supported [ev_1]."
    snapshot["evidence"] = snapshot["evidence"][:1]

    result = _evaluate_task_acceptance(task, snapshot)

    assert result["passed"] is False
    assert result["coverage"] == 0.5
    assert result["failures"] == (
        "report.min_chars",
        "concept.beta",
        "evidence.independent_sources",
    )


def test_task_acceptance_counts_canonical_urls_not_source_ids() -> None:
    task = DeepResearchBenchmarkTask.model_validate(
        {
            "task_id": "canonical-source-case",
            "prompt": "Use two independent sources.",
            "acceptance_contract": {
                "contract_id": "canonical-source-case-v1",
                "min_independent_sources": 2,
            },
        }
    )
    snapshot = _snapshot("run-canonical-source")
    snapshot["evidence"][0]["source_id"] = "first-alias"
    snapshot["evidence"][0]["url"] = "https://example.com/source/?tracking=one#section"
    snapshot["evidence"][1]["source_id"] = "second-alias"
    snapshot["evidence"][1]["url"] = "https://example.com/source?tracking=two"

    result = _evaluate_task_acceptance(task, snapshot)

    assert result["passed"] is False
    assert result["failures"] == ("evidence.independent_sources",)


def test_task_acceptance_requires_independent_allowed_domain_groups() -> None:
    task = DeepResearchBenchmarkTask.model_validate(
        {
            "task_id": "independent-domain-case",
            "prompt": "Use two independently published sources.",
            "include_domains": ["example.com", "other.test"],
            "acceptance_contract": {
                "contract_id": "independent-domain-case-v1",
                "min_independent_sources": 2,
                "min_independent_domains": 2,
            },
        }
    )
    snapshot = _snapshot("run-independent-domain")
    snapshot["evidence"][0]["url"] = "https://docs.example.com/first"
    snapshot["evidence"][1]["url"] = "https://blog.example.com/second"

    same_publisher = _evaluate_task_acceptance(task, snapshot)
    assert same_publisher["failures"] == ("evidence.independent_domains",)

    snapshot["evidence"][1]["url"] = "https://research.other.test/second"
    independent_publishers = _evaluate_task_acceptance(task, snapshot)
    assert independent_publishers["passed"] is True


def test_acceptance_contract_rejects_impossible_domain_requirement() -> None:
    with pytest.raises(ValueError, match="domains cannot exceed"):
        DeepResearchBenchmarkTask.model_validate(
            {
                "task_id": "impossible-domain-case",
                "prompt": "Impossible contract.",
                "acceptance_contract": {
                    "contract_id": "impossible-domain-case-v1",
                    "min_independent_sources": 1,
                    "min_independent_domains": 2,
                },
            }
        )


def test_independent_domains_require_explicit_non_overlapping_roots() -> None:
    with pytest.raises(ValueError, match="requires explicit include_domains"):
        DeepResearchBenchmarkTask.model_validate(
            {
                "task_id": "missing-domain-roots",
                "prompt": "Missing roots.",
                "acceptance_contract": {
                    "contract_id": "missing-domain-roots-v1",
                    "min_independent_sources": 1,
                    "min_independent_domains": 1,
                },
            }
        )

    with pytest.raises(ValueError, match="cannot exceed allowed domain roots"):
        DeepResearchBenchmarkTask.model_validate(
            {
                "task_id": "too-few-domain-roots",
                "prompt": "Too few roots.",
                "include_domains": ["example.com"],
                "acceptance_contract": {
                    "contract_id": "too-few-domain-roots-v1",
                    "min_independent_sources": 2,
                    "min_independent_domains": 2,
                },
            }
        )

    with pytest.raises(ValueError, match="cannot overlap"):
        DeepResearchBenchmarkTask.model_validate(
            {
                "task_id": "overlapping-domain-roots",
                "prompt": "Overlapping roots.",
                "include_domains": ["example.com", "docs.example.com"],
                "acceptance_contract": {
                    "contract_id": "overlapping-domain-roots-v1",
                    "min_independent_sources": 2,
                    "min_independent_domains": 2,
                },
            }
        )


@pytest.mark.parametrize(
    "domain",
    ["https://example.com", "*.example.com", "example.com/path", "bad_domain"],
)
def test_task_rejects_non_hostname_source_domains(domain: str) -> None:
    with pytest.raises(ValueError, match="source domains"):
        DeepResearchBenchmarkTask.model_validate(
            {
                "task_id": "invalid-domain",
                "prompt": "Invalid domain.",
                "include_domains": [domain],
            }
        )


def test_task_rejects_domains_duplicated_after_normalization() -> None:
    with pytest.raises(ValueError, match="include_domains must be unique"):
        DeepResearchBenchmarkTask.model_validate(
            {
                "task_id": "duplicate-domain",
                "prompt": "Duplicate domain.",
                "include_domains": ["Example.com", "example.com."],
            }
        )


def test_task_acceptance_enforces_allowed_and_excluded_domains() -> None:
    task = DeepResearchBenchmarkTask.model_validate(
        {
            "task_id": "domain-policy-case",
            "prompt": "Use official sources and avoid archived pages.",
            "include_domains": ["example.com"],
            "exclude_domains": ["archive.example.com"],
            "acceptance_contract": {
                "contract_id": "domain-policy-case-v1",
                "min_independent_sources": 1,
            },
        }
    )
    snapshot = _snapshot("run-domain-policy")
    snapshot["evidence"] = [snapshot["evidence"][1]]
    snapshot["evidence"][0]["url"] = "https://docs.example.com/current"

    allowed = _evaluate_task_acceptance(task, snapshot)
    assert allowed["passed"] is True

    snapshot["evidence"][0]["url"] = "https://archive.example.com/old"
    excluded = _evaluate_task_acceptance(task, snapshot)
    assert excluded["failures"] == ("evidence.exclude_domains",)

    snapshot["evidence"][0]["url"] = "https://example.com.evil.test/copied"
    outside = _evaluate_task_acceptance(task, snapshot)
    assert outside["failures"] == ("evidence.include_domains",)


def test_task_acceptance_fails_closed_when_source_url_is_missing() -> None:
    task = DeepResearchBenchmarkTask.model_validate(
        {
            "task_id": "missing-url-case",
            "prompt": "Use one official source.",
            "include_domains": ["example.com"],
            "acceptance_contract": {
                "contract_id": "missing-url-case-v1",
                "min_independent_sources": 1,
            },
        }
    )
    snapshot = _snapshot("run-missing-url")
    snapshot["evidence"] = [snapshot["evidence"][0]]
    snapshot["evidence"][0].pop("url")

    result = _evaluate_task_acceptance(task, snapshot)

    assert result["passed"] is False
    assert result["failures"] == (
        "evidence.valid_urls",
        "evidence.independent_sources",
        "evidence.include_domains",
    )


def test_release_candidate_manifest_has_balanced_executable_outcomes() -> None:
    manifest, _path, _digest = load_deepresearch_benchmark_manifest(RELEASE_CANDIDATE_MANIFEST)

    assert len(manifest.tasks) == 5
    assert manifest.trials_per_task == 3
    assert {task.expected_status for task in manifest.tasks} == {
        "completed",
        "needs_user_input",
    }
    assert all(
        task.acceptance_contract is not None for task in manifest.tasks if task.expected_status == "completed"
    )
    assert manifest.design_evidence.dataset_review_status == "unreviewed"


def test_release_candidate_review_template_is_bound_to_the_exact_task_set() -> None:
    manifest, _path, _digest = load_deepresearch_benchmark_manifest(RELEASE_CANDIDATE_MANIFEST)
    review = json.loads(RELEASE_CANDIDATE_REVIEW.read_text(encoding="utf-8"))

    assert review["reviewed_tasks_sha256"] == benchmark_tasks_sha256(manifest.tasks)
    assert review["review_status"] == "pending"
    assert review["approved"] is False
    assert {item["task_id"] for item in review["tasks"]} == {task.task_id for task in manifest.tasks}


def test_release_candidate_reference_template_covers_the_exact_task_set() -> None:
    manifest, _path, _digest = load_deepresearch_benchmark_manifest(RELEASE_CANDIDATE_MANIFEST)
    references = json.loads(RELEASE_CANDIDATE_REFERENCES.read_text(encoding="utf-8"))

    assert references["schema_version"] == "agenttracelab.deepresearch-reference-solutions.v1"
    assert {item["task_id"] for item in references["tasks"]} == {task.task_id for task in manifest.tasks}
    clarification_entries = {
        item["task_id"]: item for item in references["tasks"] if item["expected_status"] == "needs_user_input"
    }
    assert clarification_entries
    assert all(not item["report"] and not item["evidence"] for item in clarification_entries.values())


def test_release_candidate_reference_draft_passes_every_executable_contract() -> None:
    manifest, _path, _digest = load_deepresearch_benchmark_manifest(RELEASE_CANDIDATE_MANIFEST)
    references = json.loads(RELEASE_CANDIDATE_REFERENCE_DRAFT.read_text(encoding="utf-8"))
    by_id = {item["task_id"]: item for item in references["tasks"]}

    assert references["draft_status"] == "ai-researched-unreviewed"
    assert set(by_id) == {task.task_id for task in manifest.tasks}
    for task in manifest.tasks:
        reference = by_id[task.task_id]
        assert reference["expected_status"] == task.expected_status
        if task.expected_status == "completed":
            outcome = _evaluate_task_acceptance(task, reference)
            assert outcome["passed"] is True, (task.task_id, outcome["failures"])
        else:
            assert not reference["report"]
            assert not reference["evidence"]


def test_release_candidate_v02_review_template_is_bound_to_the_exact_task_set() -> None:
    manifest, _path, _digest = load_deepresearch_benchmark_manifest(RELEASE_CANDIDATE_V02_MANIFEST)
    review = json.loads(RELEASE_CANDIDATE_V02_REVIEW.read_text(encoding="utf-8"))

    assert review["benchmark_id"] == manifest.benchmark_id
    assert review["benchmark_version"] == manifest.version
    assert review["reviewed_tasks_sha256"] == benchmark_tasks_sha256(manifest.tasks)
    assert review["review_status"] == "pending"
    assert review["approved"] is False
    assert {item["task_id"] for item in review["tasks"]} == {task.task_id for task in manifest.tasks}


def test_release_candidate_v02_reference_draft_passes_every_executable_contract() -> None:
    manifest, _path, _digest = load_deepresearch_benchmark_manifest(RELEASE_CANDIDATE_V02_MANIFEST)
    references = json.loads(RELEASE_CANDIDATE_V02_REFERENCE_DRAFT.read_text(encoding="utf-8"))
    by_id = {item["task_id"]: item for item in references["tasks"]}

    assert references["benchmark_id"] == manifest.benchmark_id
    assert references["benchmark_version"] == manifest.version
    assert references["draft_status"] == "ai-researched-unreviewed"
    assert set(by_id) == {task.task_id for task in manifest.tasks}
    for task in manifest.tasks:
        reference = by_id[task.task_id]
        assert reference["expected_status"] == task.expected_status
        if task.expected_status == "completed":
            outcome = _evaluate_task_acceptance(task, reference)
            assert outcome["passed"] is True, (task.task_id, outcome["failures"])
        else:
            assert not reference["report"]
            assert not reference["evidence"]


def test_expected_needs_user_input_is_a_successful_zero_tool_terminal(tmp_path: Path) -> None:
    manifest = tmp_path / "clarification-benchmark.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.deepresearch-benchmark.v1",
                "benchmark_id": "clarification-smoke",
                "version": "1.0.0",
                "evidence_class": "integration",
                "trials_per_task": 1,
                "poll_interval_seconds": 0.05,
                "timeout_seconds": 5,
                "tasks": [
                    {
                        "task_id": "missing-artifact",
                        "prompt": "Evaluate that Agent.",
                        "source_mode": "web",
                        "expected_status": "needs_user_input",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            return httpx.Response(201, json={"session_id": "session-clarify"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(202, json={"run_id": "run-clarify", "status": "queued"})
        if request.method == "GET" and request.url.path == "/api/v1/runs/run-clarify":
            return httpx.Response(200, json={"run_id": "run-clarify", "status": "needs_user_input"})
        return httpx.Response(404)

    snapshot = _snapshot("run-clarify")
    snapshot["run"].update(
        {
            "status": "needs_user_input",
            "current_stage": "needs_user_input",
            "usage": {"usage": {"llm_tokens": 0, "tool_calls": 0}},
            "completed_at": "2026-09-21T00:00:02Z",
        }
    )
    snapshot["report"] = ""
    snapshot["events"] = [
        {
            "event_id": 1,
            "event_type": "run.needs_user_input",
            "stage": "needs_user_input",
            "payload": {"status": "needs_user_input"},
            "created_at": "2026-09-21T00:00:02Z",
        }
    ]
    snapshot["tool_calls"] = []
    snapshot["evidence"] = []
    snapshot["contracts"] = []
    snapshot["checkpoints"] = [
        {
            "checkpoint_id": "checkpoint-clarify",
            "version": 1,
            "stage": "needs_user_input",
            "state_hash": "c" * 64,
            "integrity_valid": True,
            "created_at": "2026-09-21T00:00:02Z",
        }
    ]
    snapshot["plan_count"] = 0
    snapshot["task_count"] = 0

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=tmp_path / "clarification-result",
        client=client,
        snapshot_exporter=lambda _database, _run_id: snapshot,
        sleep=lambda _seconds: None,
    ) as runner:
        report = runner.run(manifest)

    trial = report.trials[0]
    assert report.status == "completed"
    assert report.metrics.completion_rate == 1.0
    assert report.metrics.pass_rate == 1.0
    assert trial.run_status == "needs_user_input"
    assert trial.expected_status == "needs_user_input"
    assert trial.completed is True and trial.passed is True
    assert trial.tool_call_count == 0 and trial.evidence_count == 0
    assert trial.citation_gate_passed is None


def test_terminal_target_failure_is_promoted_to_trial_diagnostics(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            return httpx.Response(201, json={"session_id": "session-failed"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(202, json={"run_id": "run-failed", "status": "queued"})
        if request.method == "GET" and request.url.path == "/api/v1/runs/run-failed":
            return httpx.Response(200, json={"run_id": "run-failed", "status": "failed"})
        return httpx.Response(404)

    snapshot = _snapshot("run-failed")
    snapshot["run"].update(
        {
            "status": "failed",
            "current_stage": "failed",
            "error_code": "APIStatusError",
            "error_message": "Error code: 402 - Insufficient Balance",
            "completed_at": "2026-09-21T00:00:02Z",
        }
    )
    snapshot["report"] = ""

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=tmp_path / "failed-target-result",
        client=client,
        snapshot_exporter=lambda _database, _run_id: snapshot,
        sleep=lambda _seconds: None,
    ) as runner:
        report = runner.run(_manifest(tmp_path, trials=1))

    trial = report.trials[0]
    assert report.status == "failed"
    assert trial.completed is False and trial.passed is False
    assert trial.run_status == "failed"
    assert trial.error_type == "APIStatusError"
    assert "Insufficient Balance" in str(trial.error_message)
    assert "benchmark.target_run_failed" in trial.failure_categories


def test_claim_citation_gate_can_fail_a_process_valid_trial(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            return httpx.Response(201, json={"session_id": "session-1"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(202, json={"run_id": "run-1", "status": "queued"})
        if request.method == "GET" and request.url.path == "/api/v1/runs/run-1":
            return httpx.Response(200, json={"run_id": "run-1", "status": "completed"})
        return httpx.Response(404)

    snapshot = _snapshot("run-1")
    snapshot["report"] = (
        "The supported answer is documented [ev_1].\n\n"
        "A separate factual assertion is presented without any citation.\n\n"
        "Another independent factual assertion is also presented without any citation."
    )
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=tmp_path / "citation-failure",
        client=client,
        snapshot_exporter=lambda _database, _run_id: snapshot,
        sleep=lambda _seconds: None,
    ) as runner:
        report = runner.run(_manifest(tmp_path, trials=1))

    assert report.metrics.process_pass_rate == 1.0
    assert report.metrics.citation_gate_pass_rate == 0.0
    assert report.metrics.pass_rate == 0.0
    assert report.trials[0].process_passed is True
    assert report.trials[0].citation_gate_passed is False
    assert report.trials[0].quality_status == "unverified"
    assert "deepresearch.claim_citation_gate" in report.trials[0].failure_categories


def test_rescores_retained_snapshots_without_rerunning_target(tmp_path: Path) -> None:
    counters = {"sessions": 0, "runs": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            counters["sessions"] += 1
            return httpx.Response(201, json={"session_id": f"session-{counters['sessions']}"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            counters["runs"] += 1
            return httpx.Response(202, json={"run_id": f"run-{counters['runs']}"})
        if request.method == "GET" and request.url.path.startswith("/api/v1/runs/"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404)

    output = tmp_path / "rescore"
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=output,
        client=client,
        snapshot_exporter=lambda _database, run_id: _snapshot(run_id),
        sleep=lambda _seconds: None,
    ) as runner:
        runner.run(_manifest(tmp_path, trials=1))

    rescore_path = output / "benchmark-report-rescored.json"
    rescored = rescore_deepresearch_benchmark(
        output / "benchmark-report.json",
        report_output=rescore_path,
    )

    assert counters == {"sessions": 2, "runs": 2}
    assert rescored.metrics.process_pass_rate == 1.0
    assert rescored.metrics.citation_gate_pass_rate == 1.0
    assert rescored.metrics.semantic_evaluation_coverage == 0.0
    assert rescore_path.is_file()
    assert rescore_path.with_suffix(".md").is_file()


def test_rescore_aggregates_retained_semantic_judge_reports(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, trials=1)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.update({"target_provider": "deepseek", "target_model": "target-model"})
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    counters = {"sessions": 0, "runs": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            counters["sessions"] += 1
            return httpx.Response(201, json={"session_id": f"session-{counters['sessions']}"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            counters["runs"] += 1
            return httpx.Response(202, json={"run_id": f"run-{counters['runs']}"})
        if request.method == "GET" and request.url.path.startswith("/api/v1/runs/"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404)

    def judge_claims(claims, _evidence) -> ClaimJudgeOutput:
        return ClaimJudgeOutput(
            verdicts=tuple(
                ClaimJudgeVerdict(
                    claim_id=claim.claim_id,
                    verdict=ClaimVerdict.SUPPORTED,
                    rationale="The cited exported evidence directly supports the claim.",
                    evidence_ids=claim.resolved_evidence_ids,
                )
                for claim in claims
                if claim.resolved_evidence_ids
            )
        )

    judge = SimpleNamespace(provider="deepseek", model="judge-model", judge=judge_claims)
    output = tmp_path / "semantic-rescore"
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=output,
        client=client,
        snapshot_exporter=lambda _database, run_id: _snapshot(run_id),
        sleep=lambda _seconds: None,
    ) as runner:
        source = runner.run(manifest)

    judge_filename = "claim-evidence-judge.json"
    for trial in source.trials:
        snapshot_path = output / trial.snapshot_path
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        claim_report = verify_deepresearch_claims(snapshot, judge=judge)
        (snapshot_path.parent / judge_filename).write_text(
            json.dumps(claim_report.model_dump(mode="json")),
            encoding="utf-8",
        )

    rescored = rescore_deepresearch_benchmark(
        output / "benchmark-report.json",
        report_output=output / "benchmark-report-semantic.json",
        claim_report_filename=judge_filename,
    )

    assert counters == {"sessions": 2, "runs": 2}
    assert rescored.claim_judge_provider == "deepseek"
    assert rescored.claim_judge_model == "judge-model"
    assert rescored.judge_separation == "different_model"
    assert rescored.metrics.semantic_evaluation_coverage == 1.0
    assert rescored.metrics.semantic_gate_pass_rate == 1.0
    assert rescored.metrics.pass_rate == 1.0
    assert rescored.metrics.evidence_verified_passed_trials == 2
    assert rescored.metrics.evidence_unverified_potential_passes == 0
    assert rescored.metrics.evidence_supported_pass_rate_lower_bound == 1.0
    assert rescored.metrics.evidence_supported_pass_rate_upper_bound == 1.0
    markdown = (output / "benchmark-report-semantic.md").read_text(encoding="utf-8")
    assert "Required-gate pass rate: 100.0%" in markdown
    assert "Semantic grading coverage: 100.0%" in markdown
    assert "Independent semantic grading coverage" not in markdown


def test_rescore_preserves_execution_failure_without_demanding_success_snapshot(
    tmp_path: Path,
) -> None:
    counters = {"sessions": 0, "runs": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            counters["sessions"] += 1
            return httpx.Response(201, json={"session_id": f"session-{counters['sessions']}"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            counters["runs"] += 1
            return httpx.Response(202, json={"run_id": f"run-{counters['runs']}"})
        if request.method == "GET" and request.url.path.startswith("/api/v1/runs/"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404)

    def judge_claims(claims, _evidence) -> ClaimJudgeOutput:
        return ClaimJudgeOutput(
            verdicts=tuple(
                ClaimJudgeVerdict(
                    claim_id=claim.claim_id,
                    verdict=ClaimVerdict.SUPPORTED,
                    rationale="The cited exported evidence directly supports the claim.",
                    evidence_ids=claim.resolved_evidence_ids,
                )
                for claim in claims
                if claim.resolved_evidence_ids
            )
        )

    judge = SimpleNamespace(provider="provider-b", model="judge-model", judge=judge_claims)

    output = tmp_path / "partial-rescore"
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=output,
        client=client,
        snapshot_exporter=lambda _database, run_id: _snapshot(run_id),
        sleep=lambda _seconds: None,
    ) as runner:
        source = runner.run(_manifest(tmp_path))

    for trial in source.trials:
        snapshot_path = output / trial.snapshot_path
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        claim_report = verify_deepresearch_claims(snapshot, judge=judge)
        (snapshot_path.parent / "claim-evidence-judge.json").write_text(
            json.dumps(claim_report.model_dump(mode="json")), encoding="utf-8"
        )

    payload = json.loads((output / "benchmark-report.json").read_text(encoding="utf-8"))
    payload["trials"][0].update(
        {
            "snapshot_path": None,
            "trace_path": None,
            "evaluation_path": None,
            "claim_evidence_path": None,
            "completed": False,
            "passed": False,
            "process_passed": False,
            "citation_gate_passed": None,
            "semantic_gate_passed": None,
            "quality_status": None,
            "error_type": "TimeoutError",
            "error_message": "timed out",
        }
    )
    (output / "benchmark-report.json").write_text(json.dumps(payload), encoding="utf-8")

    rescored = rescore_deepresearch_benchmark(
        output / "benchmark-report.json",
        report_output=output / "benchmark-report-partial-rescored.json",
        claim_report_filename="claim-evidence-judge.json",
    )

    assert rescored.status == "partial"
    assert rescored.metrics.completed_trials == 3
    assert rescored.metrics.evidence_verified_passed_trials == 3
    assert rescored.metrics.evidence_supported_pass_rate_lower_bound == 0.75
    assert rescored.metrics.evidence_supported_pass_rate_upper_bound == 0.75


def test_reports_execution_failures_without_inventing_scores(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/sessions":
            return httpx.Response(503, json={"error": {"message": "target unavailable"}})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "missing.db",
        output_directory=tmp_path / "failed-results",
        client=client,
    ) as runner:
        report = runner.run(_manifest(tmp_path, trials=1))

    assert report.status == "failed"
    assert report.metrics.completed_trials == 0
    assert report.metrics.passed_trials == 0
    assert report.metrics.mean_score is None
    assert report.metrics.failure_categories == {"benchmark.execution_error": 2}
    assert all(item.error_type == "HTTPStatusError" for item in report.trials)


def test_timeout_cancels_waits_for_terminal_and_retains_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, trials=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tasks"] = payload["tasks"][:1]
    payload["timeout_seconds"] = 1
    payload["cancel_grace_seconds"] = 30
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    state = {"cancelled": False, "post_cancel_gets": 0, "cancel_requests": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            return httpx.Response(201, json={"session_id": "session-timeout"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(202, json={"run_id": "run-timeout", "status": "queued"})
        if request.method == "POST" and request.url.path == "/api/v1/runs/run-timeout/cancel":
            state["cancelled"] = True
            state["cancel_requests"] += 1
            return httpx.Response(202, json={"run_id": "run-timeout", "status": "cancelling"})
        if request.method == "GET" and request.url.path == "/api/v1/runs/run-timeout":
            if not state["cancelled"]:
                return httpx.Response(200, json={"run_id": "run-timeout", "status": "executing"})
            state["post_cancel_gets"] += 1
            status = "cancelling" if state["post_cancel_gets"] == 1 else "cancelled"
            return httpx.Response(200, json={"run_id": "run-timeout", "status": status})
        return httpx.Response(404)

    snapshot = _snapshot("run-timeout")
    snapshot["run"].update(
        {
            "status": "cancelled",
            "current_stage": "cancelled",
            "completed_at": "2026-09-21T00:00:08Z",
            "updated_at": "2026-09-21T00:00:08Z",
        }
    )
    snapshot["events"][-1].update({"event_type": "run.cancelled", "stage": "cancelled", "payload": {}})
    ticks = iter(range(100))
    output = tmp_path / "timeout-results"
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=output,
        client=client,
        snapshot_exporter=lambda _database, _run_id: snapshot,
        sleep=lambda _seconds: None,
        monotonic=lambda: float(next(ticks)),
    ) as runner:
        report = runner.run(manifest)

    trial = report.trials[0]
    assert state["cancel_requests"] == 1
    assert state["post_cancel_gets"] == 2
    assert trial.completed is False and trial.passed is False
    assert trial.run_status == "cancelled"
    assert trial.error_type == "TimeoutError"
    assert "benchmark.execution_error" in trial.failure_categories
    assert trial.snapshot_path and (output / trial.snapshot_path).is_file()
    assert trial.trace_path and (output / trial.trace_path).is_file()
    assert trial.evaluation_path and (output / trial.evaluation_path).is_file()
    assert trial.claim_evidence_path is None


def test_timeout_snapshot_capture_failure_preserves_the_original_runner_error(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, trials=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tasks"] = payload["tasks"][:1]
    payload["timeout_seconds"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    cancelled = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            return httpx.Response(201, json={"session_id": "session-timeout"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(202, json={"run_id": "run-timeout"})
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            cancelled = True
            return httpx.Response(202, json={"status": "cancelling"})
        if request.method == "GET" and request.url.path.endswith("/run-timeout"):
            return httpx.Response(200, json={"status": "cancelled" if cancelled else "executing"})
        return httpx.Response(404)

    def broken_exporter(_database, _run_id):
        raise RuntimeError("database snapshot unavailable")

    ticks = iter(range(100))
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=tmp_path / "capture-failure",
        client=client,
        snapshot_exporter=broken_exporter,
        sleep=lambda _seconds: None,
        monotonic=lambda: float(next(ticks)),
    ) as runner:
        report = runner.run(manifest)

    trial = report.trials[0]
    assert trial.error_type == "TimeoutError"
    assert "diagnostic capture failed: RuntimeError" in trial.error_message
    assert trial.snapshot_path is None


def test_manifest_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tasks"][1]["task_id"] = payload["tasks"][0]["task_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="task_id values must be unique"):
        load_deepresearch_benchmark_manifest(path)


def test_manifest_rejects_partial_expected_judge_identity(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expected_claim_judge_provider"] = "provider-b"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="provider, model, and prompt version"):
        load_deepresearch_benchmark_manifest(path)


def test_run_rejects_claim_judge_that_differs_from_pinned_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, trials=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "expected_claim_judge_provider": "provider-b",
            "expected_claim_judge_model": "judge-v1",
            "expected_claim_judge_prompt_version": "claim-evidence.v1",
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    judge = SimpleNamespace(provider="provider-b", model="judge-v2")

    with (
        DeepResearchBenchmarkRunner(
            base_url="http://target.test",
            source_database=tmp_path / "target.db",
            output_directory=tmp_path / "pinned-judge",
            client=client,
            claim_judge=judge,
        ) as runner,
        pytest.raises(ValueError, match="does not match the manifest"),
    ):
        runner.run(manifest)


def test_rejects_same_model_judge_when_different_model_is_required(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, trials=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "target_provider": "deepseek",
            "target_model": "deepseek-flash",
            "minimum_judge_separation": "different_model",
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    judge = SimpleNamespace(provider="deepseek", model="deepseek-flash")

    with (
        DeepResearchBenchmarkRunner(
            base_url="http://target.test",
            source_database=tmp_path / "target.db",
            output_directory=tmp_path / "separation",
            client=client,
            claim_judge=judge,
        ) as runner,
        pytest.raises(ValueError, match="required=different_model, observed=same_model"),
    ):
        runner.run(manifest)


def test_refuses_to_overwrite_an_immutable_benchmark_report(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "benchmark-report.json").write_text("{}", encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    with (
        DeepResearchBenchmarkRunner(
            base_url="http://target.test",
            source_database=tmp_path / "target.db",
            output_directory=output,
            client=client,
        ) as runner,
        pytest.raises(ValueError, match="new immutable output directory"),
    ):
        runner.run(_manifest(tmp_path))
