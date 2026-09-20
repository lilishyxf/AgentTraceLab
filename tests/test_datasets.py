from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agenttracelab.datasets import DatasetManifest, run_dataset


def test_runs_versioned_smoke_dataset() -> None:
    manifest = Path(__file__).parents[1] / "evaluation" / "datasets" / "smoke" / "v1" / "manifest.json"

    report = run_dataset(manifest)

    assert report.dataset_id == "agenttracelab-smoke"
    assert report.dataset_version == "1.0.0"
    assert report.passed is True
    assert report.matched_tasks == report.task_count == 3
    assert report.evaluation_pass_rate == 66.67
    assert all(result.matched_expectation for result in report.results)


def test_rejects_fixture_path_that_escapes_dataset_directory(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": "agenttracelab.dataset.v1",
        "dataset_id": "unsafe-dataset",
        "version": "1.0.0",
        "tasks": [
            {
                "task_id": "escape",
                "adapter": "normalized",
                "fixture": "../outside.json",
                "expected_pass": False,
            }
        ],
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the dataset directory"):
        run_dataset(manifest_path)


def test_rejects_duplicate_dataset_task_ids() -> None:
    task = {
        "task_id": "duplicate",
        "adapter": "wasmhatch",
        "fixture": "fixture.json",
        "expected_pass": True,
    }

    with pytest.raises(ValidationError, match="task IDs must be unique"):
        DatasetManifest.model_validate(
            {
                "schema_version": "agenttracelab.dataset.v1",
                "dataset_id": "duplicate-dataset",
                "version": "1.0.0",
                "tasks": [task, task],
            }
        )
