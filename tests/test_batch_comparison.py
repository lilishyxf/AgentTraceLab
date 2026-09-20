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
