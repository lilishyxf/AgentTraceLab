from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agenttracelab.calibration_v2 import (
    CalibrationManifestV2,
    analyze_judge_calibration_v2,
)
from agenttracelab.cli import main
from agenttracelab.judging import JudgeDimensionName


def _dimensions(score: int = 3) -> dict[str, int]:
    return {dimension.value: score for dimension in JudgeDimensionName}


def _annotation(annotation_id: str, annotator_id: str, verdict: str = "pass") -> dict[str, object]:
    return {
        "annotation_id": annotation_id,
        "annotator_id": annotator_id,
        "verdict": verdict,
        "dimensions": _dimensions(),
        "created_at": datetime(2026, 10, 5, tzinfo=UTC).isoformat(),
    }


def _case(case_id: str, *, split: str = "holdout") -> dict[str, object]:
    annotations = [
        _annotation(f"{case_id}-a1", "annotator-1"),
        _annotation(f"{case_id}-a2", "annotator-2"),
    ]
    return {
        "case_id": case_id,
        "trace_id": f"trace-{case_id}",
        "split": split,
        "safety_critical": False,
        "annotations": annotations,
        "adjudicated": {
            "adjudicator_id": "adjudicator-1",
            "source_annotation_ids": [item["annotation_id"] for item in annotations],
            "verdict": "pass",
            "dimensions": _dimensions(),
            "rationale": "reviewed human truth",
            "adjudicated_at": datetime(2026, 10, 5, tzinfo=UTC).isoformat(),
        },
        "judge_outcomes": [
            {
                "judge_key": "judge-primary",
                "identity": {
                    "provider": "offline",
                    "model": "fixture-judge",
                    "prompt_version": "rubric-v2",
                },
                "status": "scored",
                "verdict": "pass",
                "dimensions": _dimensions(),
            }
        ],
    }


def _manifest_payload(*cases: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "agenttracelab.judge-calibration-manifest.v2",
        "dataset_id": "offline-calibration",
        "version": "v2-test",
        "policy": {
            "min_development_cases": 1,
            "min_holdout_cases": 1,
            "min_holdout_annotators": 2,
            "max_holdout_false_block_percent": 10.0,
            "require_zero_safety_false_allows": True,
        },
        "judge_identities": {
            "judge-primary": {
                "provider": "offline",
                "model": "fixture-judge",
                "prompt_version": "rubric-v2",
            }
        },
        "cases": list(cases),
    }


def test_calibration_v2_requires_two_holdout_annotators() -> None:
    case = _case("holdout-one")
    case["annotations"] = case["annotations"][:1]  # type: ignore[index]
    case["adjudicated"]["source_annotation_ids"] = ["holdout-one-a1"]  # type: ignore[index]

    with pytest.raises(
        ValidationError,
        match="holdout case holdout-one requires at least 2 independent annotators",
    ):
        CalibrationManifestV2.model_validate(_manifest_payload(case))


def test_calibration_v2_requires_adjudication_for_disagreement() -> None:
    case = _case("holdout-disagreement")
    case["annotations"][1]["verdict"] = "fail"  # type: ignore[index]
    case["adjudicated"] = None

    with pytest.raises(
        ValidationError,
        match="case holdout-disagreement has unresolved annotation disagreement",
    ):
        CalibrationManifestV2.model_validate(_manifest_payload(case))


def test_calibration_v2_rejects_identity_drift() -> None:
    first = _case("development-one", split="development")
    second = deepcopy(_case("holdout-two"))
    second["judge_outcomes"][0]["identity"]["model"] = "drifted-model"  # type: ignore[index]

    with pytest.raises(
        ValidationError,
        match="judge key judge-primary identity does not match manifest binding",
    ):
        CalibrationManifestV2.model_validate(_manifest_payload(first, second))


def _set_truth(case: dict[str, object], verdict: str) -> None:
    for annotation in case["annotations"]:  # type: ignore[index]
        annotation["verdict"] = verdict
    case["adjudicated"]["verdict"] = verdict  # type: ignore[index]


def _set_outcome(
    case: dict[str, object],
    *,
    status: str,
    verdict: str | None = None,
) -> None:
    outcome = case["judge_outcomes"][0]  # type: ignore[index]
    outcome["status"] = status
    outcome["verdict"] = verdict
    outcome["dimensions"] = _dimensions() if status == "scored" else None
    if status == "scoring_error":
        outcome["error_code"] = "fixture-error"


def test_calibration_v2_keeps_development_and_holdout_separate() -> None:
    development = _case("development-pass", split="development")
    holdout = _case("holdout-fail")
    _set_truth(holdout, "pass")
    _set_outcome(holdout, status="scored", verdict="fail")
    manifest = CalibrationManifestV2.model_validate(_manifest_payload(development, holdout))

    report = analyze_judge_calibration_v2(manifest)

    judge = report.judges[0]
    assert judge.development.exact_agreement_percent == 100.0
    assert judge.holdout.exact_agreement_percent == 0.0
    assert judge.development.total_cases == 1
    assert judge.holdout.total_cases == 1


def test_calibration_v2_counts_false_allow_false_block_abstain_and_error() -> None:
    false_allow = _case("holdout-false-allow")
    _set_truth(false_allow, "fail")
    _set_outcome(false_allow, status="scored", verdict="pass")
    false_block = _case("holdout-false-block")
    _set_truth(false_block, "pass")
    _set_outcome(false_block, status="scored", verdict="fail")
    abstained = _case("holdout-abstained")
    _set_outcome(abstained, status="abstained")
    errored = _case("holdout-error")
    _set_outcome(errored, status="scoring_error")
    payload = _manifest_payload(false_allow, false_block, abstained, errored)
    payload["policy"]["min_holdout_cases"] = 4  # type: ignore[index]
    manifest = CalibrationManifestV2.model_validate(payload)

    metrics = analyze_judge_calibration_v2(manifest).judges[0].holdout

    assert metrics.total_cases == 4
    assert metrics.scored_cases == 2
    assert metrics.false_allow_count == 1
    assert metrics.false_block_count == 1
    assert metrics.abstention_count == 1
    assert metrics.scoring_error_count == 1
    assert metrics.coverage_percent == 50.0
    assert metrics.abstention_percent == 25.0
    assert metrics.scoring_error_percent == 25.0
    assert metrics.exact_agreement_percent == 0.0


def test_development_success_cannot_mask_holdout_failure() -> None:
    development = _case("development-success", split="development")
    holdout = _case("holdout-false-block")
    _set_truth(holdout, "pass")
    _set_outcome(holdout, status="scored", verdict="fail")
    manifest = CalibrationManifestV2.model_validate(_manifest_payload(development, holdout))

    report = analyze_judge_calibration_v2(manifest)

    assert report.judges[0].development.passed is True
    assert report.judges[0].holdout.passed is False
    assert report.passed is False
    assert any("false-block rate" in item for item in report.judges[0].holdout.violations)


def test_safety_critical_false_allow_blocks_holdout() -> None:
    holdout = _case("holdout-safety")
    holdout["safety_critical"] = True
    _set_truth(holdout, "fail")
    _set_outcome(holdout, status="scored", verdict="pass")
    manifest = CalibrationManifestV2.model_validate(_manifest_payload(holdout))

    report = analyze_judge_calibration_v2(manifest)

    assert report.judges[0].holdout.safety_false_allow_count == 1
    assert report.judges[0].holdout.passed is False
    assert any("safety-critical false allows" in item for item in report.judges[0].holdout.violations)


def test_calibration_v2_cli_writes_immutable_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    summary = tmp_path / "report.md"
    manifest.write_text(
        json.dumps(
            _manifest_payload(
                _case("development-cli", split="development"),
                _case("holdout-cli"),
            )
        ),
        encoding="utf-8",
    )
    argv = [
        "agenttracelab",
        "analyze-judge-calibration-v2",
        str(manifest),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        "agenttracelab.judge-calibration.v2"
    )
    assert "Development and holdout metrics" in summary.read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 18
