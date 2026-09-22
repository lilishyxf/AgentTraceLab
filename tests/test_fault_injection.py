from __future__ import annotations

import json

import pytest
from test_deepresearch_adapter import deepresearch_snapshot

from agenttracelab.adapters import adapt_deepresearch_snapshot
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.fault_injection import (
    EXPECTED_CHECKS,
    DeepResearchFaultType,
    inject_deepresearch_fault,
    run_deepresearch_fault_suite,
)


@pytest.mark.parametrize("fault_type", list(DeepResearchFaultType))
def test_each_fault_is_detected_by_its_expected_check(fault_type: DeepResearchFaultType) -> None:
    mutant = inject_deepresearch_fault(deepresearch_snapshot(), fault_type)
    report = evaluate_trace(adapt_deepresearch_snapshot(mutant))

    assert set(EXPECTED_CHECKS[fault_type]).issubset(report.failure_categories)


def test_fault_suite_reports_mutation_score_and_retains_mutants(tmp_path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(deepresearch_snapshot()), encoding="utf-8")
    manifest = tmp_path / "faults.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.deepresearch-faults.v1",
                "suite_id": "fault-smoke",
                "version": "1.0.0",
                "cases": [
                    {"fault_id": "citation", "fault_type": "citation_corruption"},
                    {"fault_id": "secret", "fault_type": "secret_leakage"},
                    {"fault_id": "duplicate", "fault_type": "duplicate_tool_call"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fault-results"

    report = run_deepresearch_fault_suite(snapshot, manifest, output_directory=output)

    assert report.original_passed is True
    assert report.case_count == 3
    assert report.detected_count == 3
    assert report.mutation_detection_rate == 1.0
    assert report.missed_fault_ids == ()
    assert (output / "fault-report.json").is_file()
    assert all((output / item.mutant_path).is_file() for item in report.results)


def test_fault_suite_refuses_to_overwrite_report(tmp_path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(deepresearch_snapshot()), encoding="utf-8")
    manifest = tmp_path / "faults.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.deepresearch-faults.v1",
                "suite_id": "fault-smoke",
                "version": "1.0.0",
                "cases": [{"fault_id": "citation", "fault_type": "citation_corruption"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "existing"
    output.mkdir()
    (output / "fault-report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="new output directory"):
        run_deepresearch_fault_suite(snapshot, manifest, output_directory=output)
