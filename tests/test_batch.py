from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenttracelab.batch import evaluate_wasmhatch_batch
from agenttracelab.failures import FailureCategory
from agenttracelab.policy import (
    PromotionPolicy,
    apply_promotion_policy,
    load_promotion_policy,
    render_gate_markdown,
)


def _write_batch(
    tmp_path: Path,
    entries: list[tuple[str, str, dict]],
    *,
    evidence_mode: str = "synthetic_fixture",
) -> Path:
    manifest_entries = []
    for scenario_id, filename, payload in entries:
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
        manifest_entries.append({"scenario_id": scenario_id, "journal": filename})
    manifest = {
        "schema_version": "agenttracelab.wasmhatch-batch.v1",
        "batch_id": "batch-under-test",
        "version": "1.0.0",
        "evidence_mode": evidence_mode,
        "entries": manifest_entries,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_evaluates_wasmhatch_exports_with_exact_denominators(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    path = _write_batch(
        tmp_path,
        [
            ("complete", "complete.json", complete_journal),
            ("bad-case", "bad.json", incomplete_journal),
        ],
    )

    report = evaluate_wasmhatch_batch(path)

    assert report.run_count == 2
    assert report.passed_runs == report.failed_runs == 1
    assert report.pass_rate == 50.0
    assert report.state_counts == {"active": 1, "committed": 1}
    assert report.derived_metric_totals["commits"] == 2
    assert report.failure_counts[FailureCategory.TRACE_CONTRACT] == 1
    assert "do not independently prove" in report.evidence_notice


def test_batch_rejects_journal_path_escape(tmp_path: Path, complete_journal: dict) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (tmp_path / "outside.json").write_text(json.dumps(complete_journal), encoding="utf-8")
    manifest = {
        "schema_version": "agenttracelab.wasmhatch-batch.v1",
        "batch_id": "unsafe",
        "version": "1.0.0",
        "evidence_mode": "recorded_local",
        "entries": [{"scenario_id": "escape", "journal": "../outside.json"}],
    }
    path = batch_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the batch directory"):
        evaluate_wasmhatch_batch(path)


def test_batch_rejects_duplicate_trace_ids(tmp_path: Path, complete_journal: dict) -> None:
    path = _write_batch(
        tmp_path,
        [
            ("first", "first.json", complete_journal),
            ("second", "second.json", complete_journal),
        ],
    )

    with pytest.raises(ValueError, match="duplicate trace ID"):
        evaluate_wasmhatch_batch(path)


def test_promotion_policy_can_promote_recorded_batch(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    report = evaluate_wasmhatch_batch(
        _write_batch(
            tmp_path,
            [("complete", "complete.json", complete_journal)],
            evidence_mode="recorded_local",
        )
    )
    policy = PromotionPolicy(
        schema_version="agenttracelab.promotion-policy.v1",
        policy_id="recorded-release",
    )

    gate = apply_promotion_policy(report, policy)

    assert gate.decision == "promote"
    assert gate.reasons == ()
    assert "AgentTraceLab: PROMOTE" in render_gate_markdown(report, gate)


def test_promotion_policy_blocks_weak_or_synthetic_evidence(
    tmp_path: Path,
    incomplete_journal: dict,
) -> None:
    report = evaluate_wasmhatch_batch(_write_batch(tmp_path, [("bad", "bad.json", incomplete_journal)]))
    policy = PromotionPolicy(
        schema_version="agenttracelab.promotion-policy.v1",
        policy_id="strict-release",
        min_run_count=2,
    )

    gate = apply_promotion_policy(report, policy)

    assert gate.decision == "hold"
    assert any("run_count" in reason for reason in gate.reasons)
    assert any("evidence_mode" in reason for reason in gate.reasons)
    assert any("trace_contract" in reason for reason in gate.reasons)
    assert "Promotion blockers" in render_gate_markdown(report, gate)


def test_checked_in_batch_and_policy_form_a_passing_smoke_gate() -> None:
    root = Path(__file__).parents[1]
    report = evaluate_wasmhatch_batch(root / "evaluation" / "batches" / "smoke" / "v1" / "manifest.json")
    policy = load_promotion_policy(root / "evaluation" / "policies" / "synthetic-smoke.v1.json")

    gate = apply_promotion_policy(report, policy)

    assert report.pass_rate == 100.0
    assert gate.decision == "promote"
