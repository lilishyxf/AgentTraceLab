from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[2] / "datasets" / "smoke" / "v1" / "fixtures" / "wasmhatch_complete.json"


def _retarget(payload: dict[str, Any], *, run_id: str, task: str) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    previous_run_id = result["runId"]
    result["runId"] = run_id
    result["context"]["task"] = task
    for event in result["events"]:
        tool_call_id = event["evidence"].get("tool_call_id")
        if isinstance(tool_call_id, str):
            event["evidence"]["tool_call_id"] = tool_call_id.replace(previous_run_id, run_id)
    return result


def _missing_decision(payload: dict[str, Any]) -> None:
    payload["events"][0]["evidence"] = {}


def _unterminated_tool(payload: dict[str, Any]) -> None:
    payload["events"][-1]["outcome"] = "info"


def _unapproved_commit(payload: dict[str, Any]) -> None:
    payload["events"][3]["outcome"] = "rejected"
    payload["metrics"]["approvals"] = 0
    payload["metrics"]["rejections"] = 1


def _missing_readback(payload: dict[str, Any]) -> None:
    payload["events"][-1]["summary"] = "Final tool result recorded"
    payload["events"][-1]["detail"] = "The run stopped without a post-commit readback."


def _corrupt_metrics(payload: dict[str, Any]) -> None:
    payload["metrics"]["commits"] = 0


SCENARIOS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("decision-evidence", _missing_decision),
    ("tool-result", _unterminated_tool),
    ("approval-before-commit", _unapproved_commit),
    ("post-commit-validation", _missing_readback),
    ("metric-integrity", _corrupt_metrics),
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(value, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    baseline_entries: list[dict[str, Any]] = []
    candidate_entries: list[dict[str, Any]] = []
    for index, (scenario_id, mutate) in enumerate(SCENARIOS, start=1):
        task = f"Synthetic WasmHatch regression scenario: {scenario_id}."
        baseline = _retarget(source, run_id=f"run_journal_{256 + index:032x}", task=task)
        candidate = _retarget(source, run_id=f"run_journal_{512 + index:032x}", task=task)
        mutate(baseline)
        baseline_path = Path("fixtures") / f"baseline-{scenario_id}.json"
        candidate_path = Path("fixtures") / f"candidate-{scenario_id}.json"
        _write_json(ROOT / baseline_path, baseline)
        _write_json(ROOT / candidate_path, candidate)
        baseline_entries.append(
            {
                "scenario_id": scenario_id,
                "journal": baseline_path.as_posix(),
                "tags": ["synthetic", "baseline", scenario_id],
            }
        )
        candidate_entries.append(
            {
                "scenario_id": scenario_id,
                "journal": candidate_path.as_posix(),
                "tags": ["synthetic", "candidate", scenario_id],
            }
        )

    shared = {
        "schema_version": "agenttracelab.wasmhatch-batch.v1",
        "version": "1.0.0",
        "evidence_mode": "synthetic_fixture",
    }
    _write_json(
        ROOT / "baseline.json",
        {**shared, "batch_id": "wasmhatch-optimization-baseline", "entries": baseline_entries},
    )
    _write_json(
        ROOT / "candidate.json",
        {**shared, "batch_id": "wasmhatch-optimization-candidate", "entries": candidate_entries},
    )


if __name__ == "__main__":
    main()
