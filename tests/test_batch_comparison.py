from __future__ import annotations

import json
from pathlib import Path

from agenttracelab.batch import evaluate_wasmhatch_batch
from agenttracelab.batch_comparison import (
    compare_wasmhatch_batches,
    render_batch_comparison_markdown,
)


def _batch_report(
    root: Path,
    *,
    batch_id: str,
    scenario_id: str,
    journal: dict,
    evidence_mode: str = "recorded_local",
):
    root.mkdir()
    (root / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    manifest = {
        "schema_version": "agenttracelab.wasmhatch-batch.v1",
        "batch_id": batch_id,
        "version": "1.0.0",
        "evidence_mode": evidence_mode,
        "entries": [{"scenario_id": scenario_id, "journal": "journal.json"}],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return evaluate_wasmhatch_batch(path)


def test_paired_comparison_blocks_pass_to_fail_regression(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    baseline = _batch_report(
        tmp_path / "baseline",
        batch_id="baseline",
        scenario_id="same-task",
        journal=complete_journal,
    )
    candidate = _batch_report(
        tmp_path / "candidate",
        batch_id="candidate",
        scenario_id="same-task",
        journal=incomplete_journal,
    )

    comparison = compare_wasmhatch_batches(baseline, candidate)

    assert comparison.recommendation == "hold"
    assert comparison.run_regressions == ("same-task",)
    assert comparison.check_regression_count > 0
    assert comparison.mean_paired_score_delta < 0
    assert "statistical significance" in comparison.comparison_notice
    assert "Promotion blockers" in render_batch_comparison_markdown(comparison)


def test_paired_comparison_promotes_fail_to_pass_improvement(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    baseline = _batch_report(
        tmp_path / "baseline",
        batch_id="baseline",
        scenario_id="same-task",
        journal=incomplete_journal,
    )
    candidate = _batch_report(
        tmp_path / "candidate",
        batch_id="candidate",
        scenario_id="same-task",
        journal=complete_journal,
    )

    comparison = compare_wasmhatch_batches(baseline, candidate)

    assert comparison.recommendation == "promote"
    assert comparison.run_improvements == ("same-task",)
    assert comparison.check_improvement_count > 0
    assert comparison.mean_paired_process_reward_delta > 0
    assert comparison.mean_paired_recommended_reward_delta == 1.0
    assert comparison.safety_unblocked_scenarios == ("same-task",)
    scenario = comparison.scenarios[0]
    assert scenario.baseline_safety_blocked is True
    assert scenario.candidate_safety_blocked is False
    assert scenario.recommended_reward_delta == 1.0
    assert comparison.reasons == ()


def test_paired_comparison_holds_different_evidence_modes(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    baseline = _batch_report(
        tmp_path / "baseline",
        batch_id="baseline",
        scenario_id="same-task",
        journal=complete_journal,
        evidence_mode="synthetic_fixture",
    )
    candidate = _batch_report(
        tmp_path / "candidate",
        batch_id="candidate",
        scenario_id="same-task",
        journal=complete_journal,
        evidence_mode="recorded_local",
    )

    comparison = compare_wasmhatch_batches(baseline, candidate)

    assert comparison.recommendation == "hold"
    assert comparison.evidence_modes_match is False
    assert any("different evidence modes" in reason for reason in comparison.reasons)


def test_comparison_holds_when_candidate_drops_baseline_scenario(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    baseline = _batch_report(
        tmp_path / "baseline",
        batch_id="baseline",
        scenario_id="required-task",
        journal=complete_journal,
    )
    candidate = _batch_report(
        tmp_path / "candidate",
        batch_id="candidate",
        scenario_id="new-task",
        journal=complete_journal,
    )

    comparison = compare_wasmhatch_batches(baseline, candidate)

    assert comparison.recommendation == "hold"
    assert comparison.paired_scenario_count == 0
    assert comparison.baseline_only_scenarios == ("required-task",)
    assert comparison.candidate_only_scenarios == ("new-task",)


def test_checked_in_optimization_experiment_has_fixed_reward_deltas() -> None:
    root = Path(__file__).parents[1] / "evaluation" / "experiments" / "wasmhatch-optimization" / "v1"

    baseline = evaluate_wasmhatch_batch(root / "baseline.json")
    candidate = evaluate_wasmhatch_batch(root / "candidate.json")
    comparison = compare_wasmhatch_batches(baseline, candidate)

    assert comparison.paired_scenario_count == 5
    assert comparison.recommendation == "promote"
    assert comparison.mean_paired_score_delta == 12.0
    assert comparison.mean_paired_process_reward_delta == 0.12
    assert comparison.mean_paired_recommended_reward_delta == 0.3
    assert comparison.safety_regression_scenarios == ()
    assert comparison.safety_unblocked_scenarios == ("approval-before-commit",)
    assert comparison.check_improvement_count == 6


def test_recorded_local_instrumentation_closes_observed_trace_gaps() -> None:
    root = Path(__file__).parents[1] / "evaluation" / "recorded" / "wasmhatch-local-demo" / "v1"

    baseline = evaluate_wasmhatch_batch(root / "manifest.json")
    candidate = evaluate_wasmhatch_batch(root / "candidate-manifest.json")
    comparison = compare_wasmhatch_batches(baseline, candidate)
    candidate_journal = json.loads((root / "candidate-journal.json").read_text(encoding="utf-8"))
    script_event = next(event for event in candidate_journal["events"] if event["category"] == "script")

    assert comparison.paired_scenario_count == 1
    assert comparison.recommendation == "promote"
    assert comparison.mean_paired_score_delta == 30.0
    assert comparison.mean_paired_process_reward_delta == 0.3
    assert comparison.mean_paired_recommended_reward_delta == 0.3
    assert comparison.check_regression_count == 0
    assert comparison.check_improvement_count == 3
    assert comparison.run_improvements == ("approved-local-spreadsheet-transform",)
    assert comparison.safety_regression_scenarios == ()
    assert script_event["evidence"]["role"] == "host-workflow"


def test_recorded_local_campaign_passes_across_distinct_browser_runs() -> None:
    root = Path(__file__).parents[1] / "evaluation" / "recorded" / "wasmhatch-local-campaign" / "v1"

    report = evaluate_wasmhatch_batch(root / "manifest.json")
    journals = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*.json")]
    run_journals = [payload for payload in journals if "runId" in payload]

    assert report.evidence_mode == "recorded_local"
    assert report.run_count == 3
    assert report.passed is True
    assert report.passed_runs == 3
    assert report.pass_rate == 100.0
    assert report.state_counts == {"committed": 3}
    assert report.failure_counts == {}
    assert report.derived_metric_totals["scriptRuns"] == 3
    assert report.derived_metric_totals["proposalsPrepared"] == 3
    assert report.derived_metric_totals["approvals"] == 3
    assert report.derived_metric_totals["commits"] == 3
    assert report.derived_metric_totals["rejections"] == 0
    assert report.derived_metric_totals["uncertainOutcomes"] == 0
    assert {result.evaluation.score for result in report.results} == {100.0}
    assert {result.training_reward.process_reward for result in report.results} == {1.0}
    assert {result.training_reward.recommended_reward for result in report.results} == {1.0}
    assert len({payload["runId"] for payload in run_journals}) == 3
    assert all(payload["privacy"]["credentialFieldsIncluded"] is False for payload in run_journals)
    assert all(payload["privacy"]["sourceContentsIncluded"] is False for payload in run_journals)
    assert all(
        next(event for event in payload["events"] if event["category"] == "script")["evidence"]["role"]
        == "host-workflow"
        for payload in run_journals
    )
    assert all(
        any(event["summary"] == "Post-commit readback validated" for event in payload["events"])
        for payload in run_journals
    )
    serialized = json.dumps(run_journals).lower()
    for source_value in ("aya tanaka", "ben", "inv-102", "west", "east"):
        assert source_value not in serialized
