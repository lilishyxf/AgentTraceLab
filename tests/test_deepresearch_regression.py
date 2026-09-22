from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkMetrics,
    DeepResearchBenchmarkReport,
    DeepResearchTaskSummary,
    DeepResearchTrialResult,
)
from agenttracelab.deepresearch_regression import (
    DeepResearchRegressionPolicy,
    build_deepresearch_challenge_set,
    compare_deepresearch_benchmarks,
)


def _trial(
    task_id: str,
    *,
    trace_id: str,
    passed: bool,
    score: float,
    failures: tuple[str, ...] = (),
    latency: int = 1_000,
) -> DeepResearchTrialResult:
    return DeepResearchTrialResult(
        task_id=task_id,
        trial=1,
        session_id=f"session-{trace_id}",
        run_id=f"run-{trace_id}",
        run_status="completed",
        trace_id=trace_id,
        completed=True,
        passed=passed,
        score=score,
        latency_ms=latency,
        token_count=100,
        tool_call_count=1,
        evidence_count=1,
        citation_validity=1.0,
        claim_citation_coverage=1.0,
        claim_citation_resolution=1.0,
        failure_categories=failures,
    )


def _report(
    benchmark_id: str,
    trials: tuple[DeepResearchTrialResult, ...],
    *,
    manifest_sha: str,
    evidence_class: str = "live-system",
) -> DeepResearchBenchmarkReport:
    passed = sum(item.passed for item in trials)
    scores = [item.score for item in trials if item.score is not None]
    latencies = [item.latency_ms for item in trials if item.latency_ms is not None]
    task_ids = tuple(dict.fromkeys(item.task_id for item in trials))
    summaries = tuple(
        DeepResearchTaskSummary(
            task_id=task_id,
            trials=1,
            completed=1,
            passed=int(next(item for item in trials if item.task_id == task_id).passed),
            pass_rate=float(next(item for item in trials if item.task_id == task_id).passed),
            pass_at_1=next(item for item in trials if item.task_id == task_id).passed,
            pass_power_k=next(item for item in trials if item.task_id == task_id).passed,
            mean_score=next(item for item in trials if item.task_id == task_id).score,
            mean_latency_ms=next(item for item in trials if item.task_id == task_id).latency_ms,
        )
        for task_id in task_ids
    )
    metrics = DeepResearchBenchmarkMetrics(
        total_trials=len(trials),
        completed_trials=len(trials),
        passed_trials=passed,
        completion_rate=1.0,
        pass_rate=passed / len(trials),
        pass_rate_wilson_lower_bound_95=0.0,
        pass_at_1_rate=passed / len(trials),
        pass_power_k_rate=passed / len(trials),
        mean_score=sum(scores) / len(scores),
        mean_latency_ms=sum(latencies) / len(latencies),
        p50_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=max(latencies),
        mean_token_count=100,
        mean_tool_call_count=1,
        mean_citation_validity=1.0,
        mean_claim_citation_coverage=1.0,
        mean_claim_citation_resolution=1.0,
    )
    return DeepResearchBenchmarkReport(
        benchmark_id=benchmark_id,
        benchmark_version="1.0.0",
        manifest_sha256=manifest_sha,
        evidence_class=evidence_class,
        target_name="target",
        target_revision="revision",
        target_base_url="http://target.test",
        source_database_file="app.db",
        created_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        status="completed",
        task_count=len(task_ids),
        trials_per_task=1,
        metrics=metrics,
        task_summaries=summaries,
        trials=trials,
        limitations=(),
    )


def test_clusters_baseline_failures_into_a_versioned_challenge_set(tmp_path) -> None:
    manifest_payload = {
        "schema_version": "agenttracelab.deepresearch-benchmark.v1",
        "benchmark_id": "source-suite",
        "version": "1.0.0",
        "evidence_class": "live-system",
        "target_name": "baseline",
        "trials_per_task": 1,
        "tasks": [
            {"task_id": "failing-task", "prompt": "A sufficiently long research prompt."},
            {"task_id": "passing-task", "prompt": "Another sufficiently long research prompt."},
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    baseline = _report(
        "source-suite",
        (
            _trial(
                "failing-task",
                trace_id="base-fail",
                passed=False,
                score=70,
                failures=("deepresearch.source_isolation",),
            ),
            _trial("passing-task", trace_id="base-pass", passed=True, score=100),
        ),
        manifest_sha=manifest_sha,
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")

    challenge, challenge_manifest = build_deepresearch_challenge_set(
        manifest,
        baseline_path,
        output_directory=tmp_path / "challenge",
        candidate_target_name="candidate",
        candidate_target_revision="candidate-revision",
    )

    assert challenge.selected_task_ids == ("failing-task",)
    assert challenge.used_sentinel_selection is False
    assert challenge.clusters[0].failure_categories == ("deepresearch.source_isolation",)
    assert tuple(item.task_id for item in challenge_manifest.tasks) == ("failing-task",)
    assert challenge_manifest.target_revision == "candidate-revision"
    assert (tmp_path / "challenge" / "challenge-manifest.json").is_file()


def test_paired_candidate_regression_can_promote_with_fresh_improved_traces(tmp_path) -> None:
    baseline = _report(
        "baseline",
        (
            _trial(
                "hard-task",
                trace_id="base-trace",
                passed=False,
                score=70,
                failures=("deepresearch.citations_valid",),
            ),
        ),
        manifest_sha="a" * 64,
    )
    candidate = _report(
        "candidate",
        (_trial("hard-task", trace_id="candidate-trace", passed=True, score=100, latency=900),),
        manifest_sha="b" * 64,
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    report = compare_deepresearch_benchmarks(baseline_path, candidate_path)

    assert report.recommendation == "promote"
    assert report.pass_rate_delta == 1.0
    assert report.mean_score_delta == 30.0
    assert report.fresh_trace_passed is True
    assert report.resolved_failure_categories == {"deepresearch.citations_valid": 1}


def test_regression_gate_holds_reused_trace_and_new_safety_failure(tmp_path) -> None:
    baseline = _report(
        "baseline",
        (_trial("hard-task", trace_id="same-trace", passed=True, score=100),),
        manifest_sha="a" * 64,
    )
    candidate = _report(
        "candidate",
        (
            _trial(
                "hard-task",
                trace_id="same-trace",
                passed=False,
                score=80,
                failures=("privacy.no_raw_secret",),
            ),
        ),
        manifest_sha="b" * 64,
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    report = compare_deepresearch_benchmarks(
        baseline_path,
        candidate_path,
        policy=DeepResearchRegressionPolicy(min_candidate_pass_rate=0.5),
    )

    assert report.recommendation == "hold"
    assert report.safety_regressions == 1
    assert report.fresh_trace_passed is False
    assert any("safety regressions" in reason for reason in report.reasons)
    assert any("fresh trace" in reason for reason in report.reasons)
