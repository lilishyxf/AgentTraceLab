from __future__ import annotations

import json

import pytest
from test_deepresearch_benchmark import _snapshot

from agenttracelab.tool_trajectory import (
    ToolTrajectoryAnnotationQueue,
    ToolTrajectoryManifest,
    evaluate_tool_trajectory_manifest,
)


def _manifest(tmp_path, snapshot: dict, *, review_mode: str = "generated-synthetic"):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.tool-trajectory.v1",
                "dataset_id": "tool-trajectory-smoke",
                "version": "1.0.0",
                "review_mode": review_mode,
                "annotator_count": 0,
                "cases": [
                    {
                        "case_id": "research",
                        "snapshot": "snapshot.json",
                        "expectation": {
                            "expected_tool_names": ["web_search"],
                            "allow_additional_tools": False,
                            "min_tool_calls": 1,
                            "max_tool_calls": 2,
                            "required_source_mode": "web",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_evaluates_tool_selection_result_grounding_and_provenance(tmp_path) -> None:
    report = evaluate_tool_trajectory_manifest(_manifest(tmp_path, _snapshot("run-tool")))

    assert report.pass_rate == 1.0
    assert report.mean_tool_selection_score == 1.0
    assert report.mean_argument_observability_score == 1.0
    assert report.mean_result_grounding_score == 1.0
    assert report.promotion_eligible is False


def test_detects_unexpected_unrecovered_tool_failure(tmp_path) -> None:
    snapshot = _snapshot("run-failure")
    snapshot["run"]["usage"]["usage"]["tool_calls"] = 2
    snapshot["tool_calls"].append(
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

    report = evaluate_tool_trajectory_manifest(_manifest(tmp_path, snapshot))

    assert report.pass_rate == 0.0
    assert report.results[0].failed_tool_call_count == 1
    assert report.results[0].unrecovered_tool_failure_count == 1
    assert "recovery_safety" in report.results[0].findings


def test_detects_missing_expected_evidence_domain(tmp_path) -> None:
    manifest_path = _manifest(tmp_path, _snapshot("run-domain"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["expectation"].update(
        {
            "accepted_evidence_domains": ["openai.com"],
            "min_accepted_domain_evidence": 1,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_tool_trajectory_manifest(manifest_path)

    assert report.pass_rate == 0.0
    assert report.results[0].evidence_domain_score == 0.0
    assert "evidence_domain" in report.results[0].findings


def test_recovery_case_requires_an_observed_failure(tmp_path) -> None:
    manifest_path = _manifest(tmp_path, _snapshot("run-no-injected-fault"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["expectation"].update(
        {
            "min_failed_tool_calls": 1,
            "max_failed_tool_calls": 1,
            "min_recovered_tool_failures": 1,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_tool_trajectory_manifest(manifest_path)

    assert report.pass_rate == 0.0
    assert "recovery_safety" in report.results[0].findings


def test_recovery_case_passes_after_a_failed_call_is_recovered(tmp_path) -> None:
    snapshot = _snapshot("run-recovered")
    snapshot["run"]["usage"]["usage"]["tool_calls"] = 2
    snapshot["tool_calls"].insert(
        0,
        {
            "tool_call_id": "tool_failed",
            "task_id": "task_1",
            "tool_name": "web_search",
            "source_mode": "web",
            "status": "failed",
            "args_hash": "f" * 64,
            "result_present": False,
            "error_code": "TIMEOUT",
            "created_at": "2026-09-21T00:00:01.500Z",
            "completed_at": "2026-09-21T00:00:01.900Z",
        },
    )
    manifest_path = _manifest(tmp_path, snapshot)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["expectation"].update(
        {
            "min_failed_tool_calls": 1,
            "max_failed_tool_calls": 1,
            "min_recovered_tool_failures": 1,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_tool_trajectory_manifest(manifest_path)

    assert report.pass_rate == 1.0
    assert report.results[0].recovered_tool_failure_count == 1
    assert report.results[0].unrecovered_tool_failure_count == 0


def test_human_approved_manifest_requires_an_annotator(tmp_path) -> None:
    manifest_path = _manifest(tmp_path, _snapshot("run-human"), review_mode="human-approved")

    with pytest.raises(ValueError, match="at least one annotator"):
        ToolTrajectoryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def test_zero_tool_policy_can_pass_without_fake_grounding(tmp_path) -> None:
    snapshot = _snapshot("run-no-tool")
    snapshot["tool_calls"] = []
    snapshot["evidence"] = []
    snapshot["run"]["usage"]["usage"]["tool_calls"] = 0
    manifest_path = _manifest(tmp_path, snapshot)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["expectation"].update(
        {
            "expected_tool_names": [],
            "min_tool_calls": 0,
            "max_tool_calls": 0,
            "required_source_mode": None,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_tool_trajectory_manifest(manifest_path)

    assert report.pass_rate == 1.0
    assert report.results[0].argument_observability_score == 1.0
    assert report.results[0].result_grounding_score == 1.0


def test_annotation_queue_requires_twenty_cases_and_human_review_for_approval() -> None:
    draft = {
        "case_id": "case-00",
        "prompt": "Find the official documentation.",
        "task_family": "official-source-research",
        "expected_answer_behavior": "Answer with a directly supporting official source.",
        "draft_expectation": {"expected_tool_names": ["ddgs"]},
    }
    payload = {
        "schema_version": "agenttracelab.tool-trajectory-annotation-queue.v1",
        "dataset_id": "queue",
        "version": "1.0.0",
        "status": "human-approved",
        "annotator_count": 1,
        "cases": [draft | {"case_id": f"case-{index:02d}"} for index in range(20)],
    }

    with pytest.raises(ValueError, match="every case must be human-reviewed"):
        ToolTrajectoryAnnotationQueue.model_validate(payload)
