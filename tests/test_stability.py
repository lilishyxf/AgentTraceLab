from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from agenttracelab.gepa_runtime import GepaExperimentReport
from agenttracelab.stability import (
    GepaStabilityPolicy,
    run_gepa_stability_gate,
    run_gepa_stability_report,
    wilson_lower_bound,
)


def _replace_run_id(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    original = payload["runId"]
    return json.loads(json.dumps(payload).replace(original, run_id))


def _write_contracts(tmp_path: Path) -> tuple[Path, Path, GepaExperimentReport]:
    replay = {
        "schema_version": "agenttracelab.replay.v1",
        "replay_id": "stability-held-out",
        "version": "1.0.0",
        "target": {"endpoint": "http://agent.test/v1/run"},
        "tasks": [
            {
                "task_id": "held-out-one",
                "request": {
                    "task": "Execute unseen workflow one.",
                    "agent_config": {"system_prompt": "placeholder"},
                },
                "adapter": "wasmhatch",
                "response_path": "trace",
                "expected_pass": True,
            },
            {
                "task_id": "held-out-two",
                "request": {
                    "task": "Execute unseen workflow two.",
                    "agent_config": {"system_prompt": "placeholder"},
                },
                "adapter": "wasmhatch",
                "response_path": "trace",
                "expected_pass": True,
            },
        ],
    }
    replay_path = tmp_path / "held-out.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    experiment = {
        "schema_version": "agenttracelab.gepa-experiment.v1",
        "experiment_id": "stability-test",
        "version": "1.0.0",
        "seed_candidate": {"system_prompt": "Commit immediately."},
        "objective": "Preserve safety and validation.",
        "replay_manifest": "held-out.json",
        "bindings": [
            {
                "candidate_key": "system_prompt",
                "request_path": "agent_config.system_prompt",
            }
        ],
    }
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    report = GepaExperimentReport(
        experiment_id="stability-test",
        experiment_version="1.0.0",
        created_at=datetime.now(UTC),
        engine_version="0.1.4",
        completed=True,
        improved=True,
        seed_score=0.0,
        best_score=1.0,
        best_candidate={"system_prompt": "Require approval-before-commit and validate."},
        candidate_count=2,
        total_metric_calls=4,
        candidate_fingerprints=("a" * 64, "b" * 64),
        observed_trace_ids=("run_journal_ffffffffffffffffffffffffffffffff",),
        evaluations=(),
        limitations=("test report",),
    )
    return experiment_path, replay_path, report


def test_wilson_lower_bound_is_conservative() -> None:
    assert wilson_lower_bound(10, 10) == 0.7225
    assert wilson_lower_bound(0, 10) == 0.0
    with pytest.raises(ValueError, match="between zero and total"):
        wilson_lower_bound(11, 10)


def test_stability_gate_promotes_consistently_safe_candidate(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    experiment_path, replay_path, experiment_report = _write_contracts(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        source = complete_journal if "approval-before-commit" in prompt else incomplete_journal
        trace = _replace_run_id(copy.deepcopy(source), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = run_gepa_stability_report(
            stability_id="stable-candidate",
            stability_version="1.0.0",
            experiment_manifest_path=experiment_path,
            experiment_report=experiment_report,
            held_out_replay_path=replay_path,
            policy=GepaStabilityPolicy(
                trials_per_task=3,
                min_candidate_success_lower_bound=0.6,
            ),
            replay_client=client,
        )
    finally:
        client.close()

    assert report.paired_sample_count == 6
    assert report.baseline_mean_score == 0.0
    assert report.candidate_mean_score == 1.0
    assert report.paired_mean_delta == 1.0
    assert report.candidate_successes == 6
    assert report.candidate_success_lower_bound == 0.6097
    assert report.safety_regressions == 0
    assert report.fresh_trace_passed
    assert report.recommendation == "promote"


def test_stability_gate_holds_intermittent_candidate(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    experiment_path, replay_path, experiment_report = _write_contracts(tmp_path)
    calls = 0
    candidate_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls, candidate_calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        source = incomplete_journal
        if "approval-before-commit" in prompt:
            candidate_calls += 1
            source = complete_journal if candidate_calls % 2 else incomplete_journal
        trace = _replace_run_id(copy.deepcopy(source), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = run_gepa_stability_report(
            stability_id="unstable-candidate",
            stability_version="1.0.0",
            experiment_manifest_path=experiment_path,
            experiment_report=experiment_report,
            held_out_replay_path=replay_path,
            policy=GepaStabilityPolicy(trials_per_task=3),
            replay_client=client,
        )
    finally:
        client.close()

    assert report.candidate_successes == 3
    assert report.candidate_success_rate == 0.5
    assert report.candidate_success_lower_bound == 0.1876
    assert report.recommendation == "hold"
    assert any("candidate mean score" in reason for reason in report.reasons)
    assert any("success lower bound" in reason for reason in report.reasons)


def test_stability_gate_holds_training_trace_reuse(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    experiment_path, replay_path, experiment_report = _write_contracts(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        source = complete_journal if "approval-before-commit" in prompt else incomplete_journal
        run_id = experiment_report.observed_trace_ids[0] if calls == 1 else f"run_journal_{calls:032x}"
        return httpx.Response(
            200,
            json={"trace": _replace_run_id(copy.deepcopy(source), run_id)},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = run_gepa_stability_report(
            stability_id="reused-trace",
            stability_version="1.0.0",
            experiment_manifest_path=experiment_path,
            experiment_report=experiment_report,
            held_out_replay_path=replay_path,
            policy=GepaStabilityPolicy(trials_per_task=2),
            replay_client=client,
        )
    finally:
        client.close()

    assert experiment_report.observed_trace_ids[0] in report.reused_trace_ids
    assert not report.fresh_trace_passed
    assert report.recommendation == "hold"


def test_path_based_stability_gate_resolves_relative_artifacts(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    experiment_path, replay_path, experiment_report = _write_contracts(tmp_path)
    report_path = tmp_path / "experiment-report.json"
    report_path.write_text(experiment_report.model_dump_json(indent=2), encoding="utf-8")
    stability_path = tmp_path / "stability.json"
    stability_path.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.gepa-stability.v1",
                "stability_id": "path-based",
                "version": "1.0.0",
                "experiment_manifest": experiment_path.name,
                "experiment_report": report_path.name,
                "held_out_replay_manifest": replay_path.name,
                "policy": {
                    "trials_per_task": 3,
                    "min_candidate_success_lower_bound": 0.6,
                },
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        source = complete_journal if "approval-before-commit" in prompt else incomplete_journal
        trace = _replace_run_id(copy.deepcopy(source), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = run_gepa_stability_gate(stability_path, replay_client=client)
    finally:
        client.close()

    assert report.stability_id == "path-based"
    assert report.recommendation == "promote"
