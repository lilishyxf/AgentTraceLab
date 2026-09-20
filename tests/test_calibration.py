from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agenttracelab.calibration import (
    CalibrationManifest,
    analyze_judge_calibration,
    load_calibration_manifest,
    render_calibration_markdown,
)
from agenttracelab.cli import main

SMOKE_MANIFEST = Path("evaluation/judge-calibration/smoke/v1/manifest.json")


def test_smoke_calibration_measures_human_and_pairwise_agreement() -> None:
    report = analyze_judge_calibration(load_calibration_manifest(SMOKE_MANIFEST))

    assert report.passed is False
    assert report.total_cases == 4
    by_key = {item.judge_key: item for item in report.judges}
    strong = by_key["judge-a-rubric-v1"]
    weak = by_key["judge-b-rubric-v1"]
    assert strong.passed is True
    assert strong.coverage_percent == 100.0
    assert strong.exact_agreement_percent == 100.0
    assert strong.cohen_kappa == 1.0
    assert strong.dimension_mae == 0.05
    assert weak.passed is False
    assert weak.coverage_percent == 75.0
    assert weak.exact_agreement_percent == 66.67
    assert weak.cohen_kappa == 0.5
    assert weak.review_rate_percent == 66.67
    assert len(weak.violations) == 4
    assert report.pairwise[0].overlapping_cases == 3
    assert report.pairwise[0].cohen_kappa == 0.5


def test_calibration_markdown_exposes_gate_failures() -> None:
    report = analyze_judge_calibration(load_calibration_manifest(SMOKE_MANIFEST))

    markdown = render_calibration_markdown(report)

    assert "Gate: **FAIL**" in markdown
    assert "judge-a-rubric-v1" in markdown
    assert "coverage 75.00% is below 100.00%" in markdown
    assert "Pairwise judge agreement" in markdown


def test_calibration_rejects_inconsistent_judge_provenance() -> None:
    payload = json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))
    payload["cases"][1]["judges"][0]["model"] = "different-model"
    manifest = CalibrationManifest.model_validate(payload)

    with pytest.raises(ValueError, match="inconsistent provenance"):
        analyze_judge_calibration(manifest)


def test_calibration_cli_writes_machine_and_human_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "calibration.json"
    summary = tmp_path / "calibration.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agenttracelab",
            "analyze-judge-calibration",
            str(SMOKE_MANIFEST),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    exit_code = main()

    assert exit_code == 9
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
    assert "Gate: **FAIL**" in summary.read_text(encoding="utf-8")
