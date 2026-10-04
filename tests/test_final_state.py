from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from agenttracelab.cli import main
from agenttracelab.final_state import (
    EnvironmentSnapshot,
    FilesystemEnvironmentAdapter,
    FinalStateContract,
    SQLiteEnvironmentAdapter,
    load_final_state_contract,
    verify_final_state,
)


def test_filesystem_reset_reproduces_contents_in_distinct_workspaces(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "input.csv").write_bytes(b"id,status\n1,pending\n")
    (fixture / "notes.txt").write_text("protected", encoding="utf-8")
    adapter = FilesystemEnvironmentAdapter(fixture, tmp_path / "workspaces")

    first = adapter.reset("seed-1")
    second = adapter.reset("seed-1")

    assert first.workspace_path != second.workspace_path
    assert first.fixture_sha256 == second.fixture_sha256
    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.files["input.csv"].text == "id,status\n1,pending\n"


def test_filesystem_adapter_rejects_escape(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        os.symlink(outside, fixture / "escape.txt")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"platform cannot create symlinks: {exc}")

    adapter = FilesystemEnvironmentAdapter(fixture, tmp_path / "workspaces")

    with pytest.raises(ValueError, match="fixture entry escapes fixture root"):
        adapter.reset("seed-escape")


def test_sqlite_snapshot_is_read_only_and_stable(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.sqlite3"
    with sqlite3.connect(fixture) as connection:
        connection.executescript(
            """
            CREATE TABLE items (id INTEGER PRIMARY KEY, status TEXT NOT NULL);
            INSERT INTO items (id, status) VALUES (2, 'done'), (1, 'pending');
            CREATE VIEW pending_items AS SELECT id FROM items WHERE status = 'pending';
            CREATE TRIGGER reject_delete BEFORE DELETE ON items
            BEGIN
              SELECT RAISE(ABORT, 'no delete');
            END;
            """
        )
    adapter = SQLiteEnvironmentAdapter(fixture, tmp_path / "workspaces")
    reset = adapter.reset("seed-sqlite")

    first = adapter.snapshot()
    second = adapter.snapshot()

    assert first.snapshot_sha256 == second.snapshot_sha256 == reset.snapshot_sha256
    assert list(first.sqlite_tables["items"].rows) == [
        {"id": 1, "status": "pending"},
        {"id": 2, "status": "done"},
    ]
    assert {item.object_type for item in first.sqlite_schema} == {"table", "trigger", "view"}
    with sqlite3.connect(Path(first.workspace_path) / "fixture.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def _filesystem_adapter(
    tmp_path: Path,
) -> tuple[FilesystemEnvironmentAdapter, EnvironmentSnapshot]:
    fixture = tmp_path / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "records.csv").write_bytes(b"id,status\n1,pending\n")
    (fixture / "protected.txt").write_text("keep", encoding="utf-8")
    adapter = FilesystemEnvironmentAdapter(fixture, tmp_path / "workspaces")
    return adapter, adapter.reset("contract-seed")


def _contract_payload(*, alternatives: list[dict[str, object]], max_changes: int = 1) -> dict[str, object]:
    return {
        "schema_version": "agenttracelab.final-state-contract.v1",
        "task_id": "fixture-task",
        "version": "v1",
        "adapter_kind": "filesystem",
        "fixture_reference": "fixture/files",
        "valid_alternatives": alternatives,
        "protected_predicates": [],
        "max_changed_resources": max_changes,
    }


def test_two_legal_trajectories_satisfy_one_contract(tmp_path: Path) -> None:
    contract = FinalStateContract.model_validate(
        _contract_payload(
            alternatives=[
                {
                    "alternative_id": "approved",
                    "required_predicates": [
                        {
                            "operation": "csv_row_exists",
                            "path": "records.csv",
                            "row": {"id": "1", "status": "approved"},
                        }
                    ],
                },
                {
                    "alternative_id": "accepted",
                    "required_predicates": [
                        {
                            "operation": "csv_row_exists",
                            "path": "records.csv",
                            "row": {"id": "1", "status": "accepted"},
                        }
                    ],
                },
            ]
        )
    )

    first_adapter, first_initial = _filesystem_adapter(tmp_path / "first")
    (Path(first_initial.workspace_path) / "records.csv").write_bytes(
        b"id,status\n1,approved\n"
    )
    first_report = verify_final_state(contract, first_initial, first_adapter.snapshot())

    second_adapter, second_initial = _filesystem_adapter(tmp_path / "second")
    (Path(second_initial.workspace_path) / "records.csv").write_bytes(
        b"id,status\n1,accepted\n"
    )
    second_report = verify_final_state(contract, second_initial, second_adapter.snapshot())

    assert first_report.passed is True
    assert first_report.selected_alternative_id == "approved"
    assert second_report.passed is True
    assert second_report.selected_alternative_id == "accepted"


def test_wrong_record_fails_required_state(tmp_path: Path) -> None:
    adapter, initial = _filesystem_adapter(tmp_path)
    contract = FinalStateContract.model_validate(
        _contract_payload(
            alternatives=[
                {
                    "alternative_id": "done",
                    "required_predicates": [
                        {
                            "operation": "csv_row_exists",
                            "path": "records.csv",
                            "row": {"id": "1", "status": "done"},
                        }
                    ],
                }
            ]
        )
    )

    report = verify_final_state(contract, initial, adapter.snapshot())

    assert report.passed is False
    assert report.selected_alternative_id is None
    assert "done" in report.alternative_results[0].failures[0]


def test_extra_write_fails_side_effect_bound(tmp_path: Path) -> None:
    adapter, initial = _filesystem_adapter(tmp_path)
    workspace = Path(initial.workspace_path)
    (workspace / "records.csv").write_bytes(b"id,status\n1,done\n")
    (workspace / "unexpected.txt").write_text("extra", encoding="utf-8")
    contract = FinalStateContract.model_validate(
        _contract_payload(
            alternatives=[
                {
                    "alternative_id": "done",
                    "required_predicates": [
                        {
                            "operation": "csv_row_exists",
                            "path": "records.csv",
                            "row": {"id": "1", "status": "done"},
                        }
                    ],
                }
            ],
            max_changes=1,
        )
    )

    report = verify_final_state(contract, initial, adapter.snapshot())

    assert report.selected_alternative_id == "done"
    assert report.changed_resource_count == 2
    assert report.side_effects_within_bound is False
    assert report.passed is False


def test_protected_state_change_always_fails(tmp_path: Path) -> None:
    adapter, initial = _filesystem_adapter(tmp_path)
    workspace = Path(initial.workspace_path)
    (workspace / "records.csv").write_bytes(b"id,status\n1,done\n")
    (workspace / "protected.txt").write_text("changed", encoding="utf-8")
    payload = _contract_payload(
        alternatives=[
            {
                "alternative_id": "done",
                "required_predicates": [
                    {
                        "operation": "csv_row_exists",
                        "path": "records.csv",
                        "row": {"id": "1", "status": "done"},
                    }
                ],
            }
        ],
        max_changes=2,
    )
    payload["protected_predicates"] = [
        {"operation": "protected_equals", "path": "protected.txt"}
    ]
    contract = FinalStateContract.model_validate(payload)

    report = verify_final_state(contract, initial, adapter.snapshot())

    assert report.selected_alternative_id == "done"
    assert report.protected_state_passed is False
    assert report.passed is False


def test_no_write_contract_rejects_false_completion(tmp_path: Path) -> None:
    adapter, initial = _filesystem_adapter(tmp_path)
    (Path(initial.workspace_path) / "claimed-done.txt").write_text("done", encoding="utf-8")
    contract = FinalStateContract.model_validate(
        _contract_payload(
            alternatives=[{"alternative_id": "clarify", "required_predicates": []}],
            max_changes=0,
        )
    )

    report = verify_final_state(contract, initial, adapter.snapshot())

    assert report.selected_alternative_id == "clarify"
    assert report.side_effects_within_bound is False
    assert report.passed is False


def test_contract_rejects_unsafe_predicate() -> None:
    payload = _contract_payload(
        alternatives=[
            {
                "alternative_id": "unsafe",
                "required_predicates": [
                    {"operation": "python_eval", "expression": "__import__('os').system('whoami')"}
                ],
            }
        ]
    )

    with pytest.raises(ValidationError, match="Input should be"):
        FinalStateContract.model_validate(payload)


def test_filesystem_contract_rejects_sqlite_protected_predicate() -> None:
    payload = _contract_payload(
        alternatives=[{"alternative_id": "unchanged", "required_predicates": []}],
        max_changes=0,
    )
    payload["protected_predicates"] = [
        {"operation": "protected_equals", "table": "items"},
        {"operation": "schema_equals"},
    ]

    with pytest.raises(
        ValidationError,
        match="filesystem contract cannot protect SQLite tables or schema",
    ):
        FinalStateContract.model_validate(payload)


def test_sqlite_absence_does_not_pass_when_table_missing() -> None:
    snapshot = EnvironmentSnapshot(
        adapter_kind="sqlite",
        environment_id="sqlite-empty",
        seed="seed",
        fixture_sha256="a" * 64,
        snapshot_sha256="b" * 64,
        workspace_path="unused",
    )
    contract = FinalStateContract(
        task_id="missing-table",
        version="v1",
        adapter_kind="sqlite",
        fixture_reference="fixture.sqlite3",
        valid_alternatives=(
            {
                "alternative_id": "row-absent",
                "required_predicates": [
                    {
                        "operation": "sqlite_row_absent",
                        "table": "items",
                        "row": {"id": 1},
                    }
                ],
            },
        ),
        max_changed_resources=0,
    )

    report = verify_final_state(contract, snapshot, snapshot)

    assert report.passed is False
    assert "is missing" in report.alternative_results[0].failures[0]


def test_report_hashes_are_stable(tmp_path: Path) -> None:
    adapter, initial = _filesystem_adapter(tmp_path)
    contract = FinalStateContract.model_validate(
        _contract_payload(
            alternatives=[{"alternative_id": "unchanged", "required_predicates": []}],
            max_changes=0,
        )
    )
    final = adapter.snapshot()

    first = verify_final_state(contract, initial, final)
    second = verify_final_state(contract, initial, final)

    assert first.contract_sha256 == second.contract_sha256
    assert first.report_sha256 == second.report_sha256


def test_final_state_cli_verifies_snapshot_artifacts_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, initial = _filesystem_adapter(tmp_path / "environment")
    (Path(initial.workspace_path) / "records.csv").write_bytes(b"id,status\n1,done\n")
    final = adapter.snapshot()
    contract_payload = _contract_payload(
        alternatives=[
            {
                "alternative_id": "done",
                "required_predicates": [
                    {
                        "operation": "csv_row_exists",
                        "path": "records.csv",
                        "row": {"id": "1", "status": "done"},
                    }
                ],
            }
        ]
    )
    contract_path = tmp_path / "contract.json"
    initial_path = tmp_path / "initial.json"
    final_path = tmp_path / "final.json"
    output = tmp_path / "report.json"
    summary = tmp_path / "report.md"
    contract_path.write_text(json.dumps(contract_payload), encoding="utf-8")
    initial_path.write_text(initial.model_dump_json(), encoding="utf-8")
    final_path.write_text(final.model_dump_json(), encoding="utf-8")
    argv = [
        "agenttracelab",
        "verify-final-state",
        str(contract_path),
        "--initial",
        str(initial_path),
        "--final",
        str(final_path),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert "Final-state verification" in summary.read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 18


def test_offline_final_state_contracts_are_valid() -> None:
    root = Path("evaluation/final-state/offline-v1")
    index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    contracts = [load_final_state_contract(root / path)[0] for path in index["contracts"]]

    assert len(contracts) == 6
    assert {contract.adapter_kind for contract in contracts} == {"filesystem", "sqlite"}
    assert sum(contract.max_changed_resources == 0 for contract in contracts) == 2
