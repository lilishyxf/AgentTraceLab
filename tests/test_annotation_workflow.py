from __future__ import annotations

import json
from pathlib import Path

import httpx
from test_deepresearch_benchmark import _snapshot

from agenttracelab.annotation_workflow import (
    build_annotation_benchmark_manifest,
    materialize_tool_trajectory_manifest,
)
from agenttracelab.deepresearch_benchmark import DeepResearchBenchmarkRunner
from agenttracelab.tool_trajectory import evaluate_tool_trajectory_manifest


def _queue(path: Path) -> Path:
    cases = []
    for index in range(20):
        if index == 18:
            lane = "controlled-fault"
        elif index == 19:
            lane = "manual-only"
        else:
            lane = "live-safe"
        cases.append(
            {
                "case_id": f"case-{index:02d}",
                "prompt": f"Research task {index}",
                "task_family": "official-source-research",
                "execution_lane": lane,
                "expected_answer_behavior": "Answer from the cited official source.",
                "draft_expectation": {
                    "expected_tool_names": ["web_search"],
                    "allow_additional_tools": False,
                    "min_tool_calls": 1,
                    "max_tool_calls": 2,
                    "required_source_mode": "web",
                },
                "expected_domains": ["example.com"],
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.tool-trajectory-annotation-queue.v1",
                "dataset_id": "annotation-workflow",
                "version": "1.0.0",
                "status": "unreviewed",
                "annotator_count": 0,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builds_live_lane_and_materializes_snapshot_trajectory_manifest(tmp_path: Path) -> None:
    queue = _queue(tmp_path / "queue.json")
    benchmark_manifest = tmp_path / "benchmark.json"
    build = build_annotation_benchmark_manifest(
        queue,
        benchmark_manifest,
        benchmark_id="annotation-live",
        target_name="deepresearch",
        target_provider="deepseek",
        target_model="deepseek-flash",
    )

    assert len(build.included_case_ids) == 18
    assert [item.execution_lane for item in build.excluded_cases] == [
        "controlled-fault",
        "manual-only",
    ]

    counter = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal counter
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            counter += 1
            return httpx.Response(201, json={"session_id": f"session-{counter}"})
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(202, json={"run_id": f"run-{counter}", "status": "queued"})
        if request.method == "GET" and request.url.path.startswith("/api/v1/runs/"):
            return httpx.Response(200, json={"run_id": f"run-{counter}", "status": "completed"})
        return httpx.Response(404)

    output = tmp_path / "benchmark-output"
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://target.test")
    with DeepResearchBenchmarkRunner(
        base_url="http://target.test",
        source_database=tmp_path / "target.db",
        output_directory=output,
        client=client,
        snapshot_exporter=lambda _database, run_id: _snapshot(run_id),
        sleep=lambda _seconds: None,
    ) as runner:
        runner.run(benchmark_manifest)

    trajectory_manifest = output / "tool-trajectory-manifest.json"
    materialized = materialize_tool_trajectory_manifest(
        queue,
        output / "benchmark-report.json",
        trajectory_manifest,
    )
    report = evaluate_tool_trajectory_manifest(trajectory_manifest)

    assert len(materialized.included_trial_ids) == 18
    assert materialized.skipped_trial_ids == ()
    assert materialized.manifest.review_mode == "generated-synthetic"
    assert report.pass_rate == 1.0
    assert report.mean_evidence_domain_score == 1.0
