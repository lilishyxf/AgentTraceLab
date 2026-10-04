from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from hashlib import sha256
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from agenttracelab.models import FrozenModel


class FileState(FrozenModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    text: str | None = None


class SQLiteSchemaObject(FrozenModel):
    object_type: Literal["table", "view", "trigger", "index"]
    name: str
    table_name: str
    sql: str | None


class SQLiteTableState(FrozenModel):
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


class EnvironmentSnapshot(FrozenModel):
    schema_version: Literal["agenttracelab.environment-snapshot.v1"] = (
        "agenttracelab.environment-snapshot.v1"
    )
    adapter_kind: Literal["filesystem", "sqlite"]
    adapter_version: str = "v1"
    environment_id: str
    seed: str
    fixture_sha256: str
    snapshot_sha256: str
    workspace_path: str
    files: dict[str, FileState] = {}
    sqlite_schema: tuple[SQLiteSchemaObject, ...] = ()
    sqlite_tables: dict[str, SQLiteTableState] = {}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _validate_fixture_tree(root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    for entry in root.rglob("*"):
        resolved = entry.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"fixture entry escapes fixture root: {entry}") from exc


def _filesystem_observation(root: Path) -> dict[str, FileState]:
    resolved_root = root.resolve(strict=True)
    files: dict[str, FileState] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"workspace entry escapes workspace root: {path}") from exc
        raw = resolved.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        files[relative] = FileState(
            sha256=_hash_bytes(raw),
            size_bytes=len(raw),
            text=text,
        )
    return files


def _environment_id(adapter_kind: str, fixture_sha256: str, seed: str) -> str:
    return _hash_bytes(_canonical_bytes([adapter_kind, "v1", fixture_sha256, seed]))


class FilesystemEnvironmentAdapter:
    def __init__(self, fixture_root: Path, workspace_root: Path) -> None:
        self.fixture_root = Path(fixture_root)
        self.workspace_root = Path(workspace_root)
        self._workspace: Path | None = None
        self._seed: str | None = None
        self._fixture_sha256: str | None = None

    def reset(self, seed: str) -> EnvironmentSnapshot:
        if not seed:
            raise ValueError("environment seed must not be empty")
        _validate_fixture_tree(self.fixture_root)
        fixture_files = _filesystem_observation(self.fixture_root)
        fixture_sha256 = _hash_bytes(
            _canonical_bytes(
                {name: value.model_dump(mode="json") for name, value in fixture_files.items()}
            )
        )
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_root / f"filesystem-{uuid4().hex}"
        shutil.copytree(self.fixture_root, workspace)
        self._workspace = workspace
        self._seed = seed
        self._fixture_sha256 = fixture_sha256
        return self.snapshot()

    def snapshot(self) -> EnvironmentSnapshot:
        if self._workspace is None or self._seed is None or self._fixture_sha256 is None:
            raise RuntimeError("filesystem environment must be reset before snapshot")
        files = _filesystem_observation(self._workspace)
        content = {name: value.model_dump(mode="json") for name, value in files.items()}
        return EnvironmentSnapshot(
            adapter_kind="filesystem",
            environment_id=_environment_id("filesystem", self._fixture_sha256, self._seed),
            seed=self._seed,
            fixture_sha256=self._fixture_sha256,
            snapshot_sha256=_hash_bytes(_canonical_bytes(content)),
            workspace_path=str(self._workspace.resolve()),
            files=files,
        )


def _sqlite_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _json_sqlite_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$bytes_hex": value.hex()}
    return value


def _sqlite_observation(
    database: Path,
) -> tuple[tuple[SQLiteSchemaObject, ...], dict[str, SQLiteTableState]]:
    schema: list[SQLiteSchemaObject] = []
    tables: dict[str, SQLiteTableState] = {}
    with sqlite3.connect(_sqlite_uri(database), uri=True) as connection:
        connection.row_factory = sqlite3.Row
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        for item in objects:
            schema.append(
                SQLiteSchemaObject(
                    object_type=item["type"],
                    name=item["name"],
                    table_name=item["tbl_name"],
                    sql=item["sql"],
                )
            )
        for item in objects:
            if item["type"] != "table":
                continue
            table_name = item["name"]
            quoted = table_name.replace('"', '""')
            columns = tuple(
                row["name"]
                for row in connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            )
            observed_rows = [
                {column: _json_sqlite_value(row[column]) for column in columns}
                for row in connection.execute(f'SELECT * FROM "{quoted}"').fetchall()
            ]
            observed_rows.sort(key=_canonical_bytes)
            tables[table_name] = SQLiteTableState(columns=columns, rows=tuple(observed_rows))
    return tuple(schema), tables


class SQLiteEnvironmentAdapter:
    def __init__(self, fixture_database: Path, workspace_root: Path) -> None:
        self.fixture_database = Path(fixture_database)
        self.workspace_root = Path(workspace_root)
        self._workspace: Path | None = None
        self._database: Path | None = None
        self._seed: str | None = None
        self._fixture_sha256: str | None = None

    def reset(self, seed: str) -> EnvironmentSnapshot:
        if not seed:
            raise ValueError("environment seed must not be empty")
        fixture = self.fixture_database.resolve(strict=True)
        if not fixture.is_file():
            raise ValueError("SQLite fixture must be a file")
        fixture_sha256 = _hash_bytes(fixture.read_bytes())
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_root / f"sqlite-{uuid4().hex}"
        workspace.mkdir()
        database = workspace / "fixture.sqlite3"
        shutil.copy2(fixture, database)
        self._workspace = workspace
        self._database = database
        self._seed = seed
        self._fixture_sha256 = fixture_sha256
        return self.snapshot()

    def snapshot(self) -> EnvironmentSnapshot:
        if (
            self._workspace is None
            or self._database is None
            or self._seed is None
            or self._fixture_sha256 is None
        ):
            raise RuntimeError("SQLite environment must be reset before snapshot")
        schema, tables = _sqlite_observation(self._database)
        content = {
            "schema": [item.model_dump(mode="json") for item in schema],
            "tables": {
                name: table.model_dump(mode="json") for name, table in sorted(tables.items())
            },
        }
        return EnvironmentSnapshot(
            adapter_kind="sqlite",
            environment_id=_environment_id("sqlite", self._fixture_sha256, self._seed),
            seed=self._seed,
            fixture_sha256=self._fixture_sha256,
            snapshot_sha256=_hash_bytes(_canonical_bytes(content)),
            workspace_path=str(self._workspace.resolve()),
            sqlite_schema=schema,
            sqlite_tables=tables,
        )


PredicateOperation = Literal[
    "file_exists",
    "file_sha256",
    "csv_cell_equals",
    "csv_row_exists",
    "sqlite_row_exists",
    "sqlite_row_absent",
    "sqlite_count_equals",
    "sqlite_value_equals",
    "schema_equals",
    "protected_equals",
]


def _validate_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not value or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError("predicate path must stay relative to the environment root")
    return path.as_posix()


class FinalStatePredicate(FrozenModel):
    operation: PredicateOperation
    path: str | None = Field(default=None, max_length=1_024)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    row: dict[str, Any] | None = None
    where: dict[str, Any] | None = None
    column: str | None = Field(default=None, max_length=256)
    value: Any = None
    table: str | None = Field(default=None, max_length=256)
    count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_shape(self) -> FinalStatePredicate:
        if self.path is not None:
            object.__setattr__(self, "path", _validate_relative_path(self.path))
        required: dict[str, tuple[str, ...]] = {
            "file_exists": ("path",),
            "file_sha256": ("path", "expected_sha256"),
            "csv_cell_equals": ("path", "where", "column"),
            "csv_row_exists": ("path", "row"),
            "sqlite_row_exists": ("table", "row"),
            "sqlite_row_absent": ("table", "row"),
            "sqlite_count_equals": ("table", "count"),
            "sqlite_value_equals": ("table", "where", "column"),
            "schema_equals": (),
        }
        if self.operation == "protected_equals":
            if (self.path is None) == (self.table is None):
                raise ValueError("protected_equals requires exactly one of path or table")
            return self
        missing = [name for name in required[self.operation] if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"{self.operation} predicate requires fields: {', '.join(missing)}"
            )
        return self


class FinalStateAlternative(FrozenModel):
    alternative_id: str = Field(min_length=1, max_length=256)
    required_predicates: tuple[FinalStatePredicate, ...]


class FinalStateContract(FrozenModel):
    schema_version: Literal["agenttracelab.final-state-contract.v1"] = (
        "agenttracelab.final-state-contract.v1"
    )
    task_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    adapter_kind: Literal["filesystem", "sqlite"]
    fixture_reference: str = Field(min_length=1, max_length=1_024)
    valid_alternatives: tuple[FinalStateAlternative, ...]
    protected_predicates: tuple[FinalStatePredicate, ...] = ()
    max_changed_resources: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_contract(self) -> FinalStateContract:
        if not self.valid_alternatives:
            raise ValueError("final-state contract requires at least one valid alternative")
        alternative_ids = [item.alternative_id for item in self.valid_alternatives]
        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("final-state alternative IDs must be unique")
        for predicate in self.protected_predicates:
            if predicate.operation not in {"protected_equals", "schema_equals"}:
                raise ValueError("protected predicates must use protected_equals or schema_equals")
        all_predicates = [
            predicate
            for alternative in self.valid_alternatives
            for predicate in alternative.required_predicates
        ] + list(self.protected_predicates)
        filesystem_ops = {"file_exists", "file_sha256", "csv_cell_equals", "csv_row_exists"}
        sqlite_ops = {
            "sqlite_row_exists",
            "sqlite_row_absent",
            "sqlite_count_equals",
            "sqlite_value_equals",
        }
        if self.adapter_kind == "filesystem" and any(
            predicate.operation in sqlite_ops for predicate in all_predicates
        ):
            raise ValueError("filesystem contract cannot contain SQLite predicates")
        if self.adapter_kind == "sqlite" and any(
            predicate.operation in filesystem_ops for predicate in all_predicates
        ):
            raise ValueError("SQLite contract cannot contain filesystem predicates")
        return self


class PredicateResult(FrozenModel):
    operation: PredicateOperation
    passed: bool
    detail: str


class AlternativeResult(FrozenModel):
    alternative_id: str
    passed: bool
    predicate_results: tuple[PredicateResult, ...]
    failures: tuple[str, ...]


class FinalStateReport(FrozenModel):
    schema_version: Literal["agenttracelab.final-state-report.v1"] = (
        "agenttracelab.final-state-report.v1"
    )
    task_id: str
    task_version: str
    contract_sha256: str
    initial_snapshot_sha256: str
    final_snapshot_sha256: str
    environment_id: str
    passed: bool
    selected_alternative_id: str | None
    alternative_results: tuple[AlternativeResult, ...]
    protected_state_passed: bool
    protected_results: tuple[PredicateResult, ...]
    protected_failures: tuple[str, ...]
    changed_resources: tuple[str, ...]
    changed_resource_count: int
    side_effects_within_bound: bool
    report_sha256: str


def _rows_for_csv(snapshot: EnvironmentSnapshot, path: str) -> list[dict[str, str]]:
    file_state = snapshot.files.get(path)
    if file_state is None or file_state.text is None:
        return []
    return list(csv.DictReader(StringIO(file_state.text)))


def _row_matches(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in expected.items())


def _evaluate_predicate(
    predicate: FinalStatePredicate,
    initial: EnvironmentSnapshot,
    final: EnvironmentSnapshot,
) -> PredicateResult:
    operation = predicate.operation
    passed = False
    detail = ""
    if operation == "file_exists":
        passed = predicate.path in final.files
        detail = f"file {predicate.path!r} {'exists' if passed else 'is missing'}"
    elif operation == "file_sha256":
        state = final.files.get(predicate.path or "")
        actual = state.sha256 if state else None
        passed = actual == predicate.expected_sha256
        detail = f"file {predicate.path!r} sha256 expected {predicate.expected_sha256}, got {actual}"
    elif operation == "csv_row_exists":
        rows = _rows_for_csv(final, predicate.path or "")
        expected = predicate.row or {}
        passed = any(_row_matches(row, expected) for row in rows)
        detail = f"CSV {predicate.path!r} expected row {expected!r}"
    elif operation == "csv_cell_equals":
        rows = _rows_for_csv(final, predicate.path or "")
        matching = [row for row in rows if _row_matches(row, predicate.where or {})]
        passed = bool(matching) and matching[0].get(predicate.column or "") == predicate.value
        detail = (
            f"CSV {predicate.path!r} where {predicate.where!r} column "
            f"{predicate.column!r} expected {predicate.value!r}"
        )
    elif operation in {"sqlite_row_exists", "sqlite_row_absent"}:
        table = final.sqlite_tables.get(predicate.table or "")
        found = bool(table) and any(_row_matches(row, predicate.row or {}) for row in table.rows)
        passed = found if operation == "sqlite_row_exists" else not found
        detail = f"SQLite table {predicate.table!r} row {predicate.row!r} found={found}"
    elif operation == "sqlite_count_equals":
        table = final.sqlite_tables.get(predicate.table or "")
        rows = list(table.rows) if table else []
        if predicate.where:
            rows = [row for row in rows if _row_matches(row, predicate.where)]
        passed = len(rows) == predicate.count
        detail = f"SQLite table {predicate.table!r} count expected {predicate.count}, got {len(rows)}"
    elif operation == "sqlite_value_equals":
        table = final.sqlite_tables.get(predicate.table or "")
        matching = [
            row for row in (table.rows if table else ()) if _row_matches(row, predicate.where or {})
        ]
        actual = matching[0].get(predicate.column or "") if len(matching) == 1 else None
        passed = len(matching) == 1 and actual == predicate.value
        detail = (
            f"SQLite table {predicate.table!r} where {predicate.where!r} column "
            f"{predicate.column!r} expected {predicate.value!r}, got {actual!r}"
        )
    elif operation == "schema_equals":
        passed = initial.sqlite_schema == final.sqlite_schema
        detail = "SQLite schema must equal the initial schema"
    elif operation == "protected_equals":
        if predicate.path is not None:
            passed = initial.files.get(predicate.path) == final.files.get(predicate.path)
            detail = f"protected file {predicate.path!r} must remain unchanged"
        else:
            passed = initial.sqlite_tables.get(predicate.table or "") == final.sqlite_tables.get(
                predicate.table or ""
            )
            detail = f"protected SQLite table {predicate.table!r} must remain unchanged"
    return PredicateResult(operation=operation, passed=passed, detail=detail)


def _changed_resources(
    initial: EnvironmentSnapshot,
    final: EnvironmentSnapshot,
) -> tuple[str, ...]:
    if initial.adapter_kind == "filesystem":
        names = sorted(set(initial.files) | set(final.files))
        return tuple(f"file:{name}" for name in names if initial.files.get(name) != final.files.get(name))
    changed: list[str] = []
    if initial.sqlite_schema != final.sqlite_schema:
        changed.append("sqlite:schema")
    table_names = sorted(set(initial.sqlite_tables) | set(final.sqlite_tables))
    changed.extend(
        f"sqlite:table:{name}"
        for name in table_names
        if initial.sqlite_tables.get(name) != final.sqlite_tables.get(name)
    )
    return tuple(changed)


def verify_final_state(
    contract: FinalStateContract,
    initial: EnvironmentSnapshot,
    final: EnvironmentSnapshot,
) -> FinalStateReport:
    if initial.adapter_kind != contract.adapter_kind or final.adapter_kind != contract.adapter_kind:
        raise ValueError("snapshot adapter kind does not match final-state contract")
    if initial.environment_id != final.environment_id:
        raise ValueError("initial and final snapshots do not share an environment identity")
    if initial.fixture_sha256 != final.fixture_sha256:
        raise ValueError("initial and final snapshots do not share a fixture hash")

    alternatives: list[AlternativeResult] = []
    selected: str | None = None
    for alternative in contract.valid_alternatives:
        results = tuple(
            _evaluate_predicate(predicate, initial, final)
            for predicate in alternative.required_predicates
        )
        failures = tuple(item.detail for item in results if not item.passed)
        passed = not failures
        alternatives.append(
            AlternativeResult(
                alternative_id=alternative.alternative_id,
                passed=passed,
                predicate_results=results,
                failures=failures,
            )
        )
        if passed and selected is None:
            selected = alternative.alternative_id

    protected_results = tuple(
        _evaluate_predicate(predicate, initial, final)
        for predicate in contract.protected_predicates
    )
    protected_failures = tuple(item.detail for item in protected_results if not item.passed)
    changed = _changed_resources(initial, final)
    within_bound = len(changed) <= contract.max_changed_resources
    contract_payload = contract.model_dump(mode="json")
    contract_sha256 = _hash_bytes(_canonical_bytes(contract_payload))
    payload = {
        "task_id": contract.task_id,
        "task_version": contract.version,
        "contract_sha256": contract_sha256,
        "initial_snapshot_sha256": initial.snapshot_sha256,
        "final_snapshot_sha256": final.snapshot_sha256,
        "environment_id": initial.environment_id,
        "passed": selected is not None and not protected_failures and within_bound,
        "selected_alternative_id": selected,
        "alternative_results": [item.model_dump(mode="json") for item in alternatives],
        "protected_state_passed": not protected_failures,
        "protected_results": [item.model_dump(mode="json") for item in protected_results],
        "protected_failures": list(protected_failures),
        "changed_resources": list(changed),
        "changed_resource_count": len(changed),
        "side_effects_within_bound": within_bound,
    }
    return FinalStateReport(**payload, report_sha256=_hash_bytes(_canonical_bytes(payload)))


def load_final_state_contract(
    path: str | Path,
) -> tuple[FinalStateContract, bytes]:
    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("final-state contract exceeds 1048576 bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("final-state contract must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("final-state contract root must be a JSON object")
    return FinalStateContract.model_validate(payload), raw


def render_final_state_markdown(report: FinalStateReport) -> str:
    lines = [
        f"# Final-state verification: {report.task_id} ({report.task_version})",
        "",
        f"Gate: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        f"Selected valid alternative: `{report.selected_alternative_id or 'none'}`",
        f"Changed resources: {report.changed_resource_count}",
        f"Protected state: {'PASS' if report.protected_state_passed else 'FAIL'}",
        f"Side-effect bound: {'PASS' if report.side_effects_within_bound else 'FAIL'}",
        "",
        "## Alternatives",
        "",
    ]
    for alternative in report.alternative_results:
        lines.append(
            f"- `{alternative.alternative_id}`: {'PASS' if alternative.passed else 'FAIL'}"
        )
        lines.extend(f"  - {failure}" for failure in alternative.failures)
    if report.protected_failures:
        lines.extend(["", "## Protected-state violations", ""])
        lines.extend(f"- {failure}" for failure in report.protected_failures)
    return "\n".join(lines) + "\n"
