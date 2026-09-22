from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from agenttracelab.gepa_runtime import (
    GepaExperimentManifest,
    GepaRuntimeError,
    GepaValidationPolicy,
    OpenAICompatibleReflectionLM,
    ReflectionClientConfig,
    run_gepa_experiment,
    validate_gepa_candidate,
    validate_gepa_candidate_report,
)


def _replace_run_id(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    original = payload["runId"]
    rendered = json.dumps(payload).replace(original, run_id)
    return json.loads(rendered)


def _write_experiment(tmp_path: Path) -> Path:
    replay = {
        "schema_version": "agenttracelab.replay.v1",
        "replay_id": "gepa-runtime-smoke",
        "version": "1.0.0",
        "target": {"endpoint": "http://agent.test/v1/run"},
        "tasks": [
            {
                "task_id": "approval-and-validation",
                "request": {
                    "task": "Normalize the bounded sheet.",
                    "agent_config": {"system_prompt": "placeholder"},
                },
                "adapter": "wasmhatch",
                "response_path": "trace",
                "expected_pass": True,
            }
        ],
    }
    (tmp_path / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    experiment = {
        "schema_version": "agenttracelab.gepa-experiment.v1",
        "experiment_id": "gepa-runtime-smoke",
        "version": "1.0.0",
        "seed_candidate": {"system_prompt": "Commit immediately and omit approval evidence."},
        "objective": "Preserve approval-before-commit and verify every committed effect.",
        "background": "The Agent must emit a fresh WasmHatch journal for every evaluation.",
        "replay_manifest": "replay.json",
        "bindings": [
            {
                "candidate_key": "system_prompt",
                "request_path": "agent_config.system_prompt",
            }
        ],
        "max_metric_calls": 6,
        "max_candidate_proposals": 1,
        "seed": 7,
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(experiment), encoding="utf-8")
    return path


def _safe_proposer(
    candidate: dict[str, str],
    reflective_dataset: object,
    components_to_update: list[str],
) -> dict[str, str]:
    del candidate, reflective_dataset, components_to_update
    return {
        "system_prompt": (
            "Inspect first, require approval-before-commit, execute once, and validate by readback."
        )
    }


def _write_validation_replay(tmp_path: Path, *, task_count: int = 2) -> Path:
    replay = {
        "schema_version": "agenttracelab.replay.v1",
        "replay_id": "gepa-held-out",
        "version": "1.0.0",
        "target": {"endpoint": "http://agent.test/v1/run"},
        "tasks": [
            {
                "task_id": f"held-out-{index}",
                "request": {
                    "task": f"Execute unseen bounded workflow {index}.",
                    "agent_config": {"system_prompt": "placeholder"},
                },
                "adapter": "wasmhatch",
                "response_path": "trace",
                "expected_pass": True,
            }
            for index in range(1, task_count + 1)
        ],
    }
    path = tmp_path / "validation-replay.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    return path


def _run_search(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> tuple[Path, object, int]:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        source = complete_journal if "approval-before-commit" in prompt else incomplete_journal
        trace = _replace_run_id(copy.deepcopy(source), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    experiment_path = _write_experiment(tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = run_gepa_experiment(
            experiment_path,
            custom_candidate_proposer=_safe_proposer,
            replay_client=client,
        )
    finally:
        client.close()
    return experiment_path, report, calls


def test_gepa_runtime_runs_official_search_and_keeps_fresh_trace_evidence(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    pytest.importorskip("gepa")
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
        report = run_gepa_experiment(
            _write_experiment(tmp_path),
            custom_candidate_proposer=_safe_proposer,
            replay_client=client,
        )
    finally:
        client.close()

    assert report.completed
    assert report.improved
    assert report.seed_score == 0.0
    assert report.best_score == 1.0
    assert report.best_candidate["system_prompt"].startswith("Inspect first")
    assert report.candidate_count == 2
    assert len(report.observed_trace_ids) == len(report.evaluations)
    assert report.reused_trace_ids == ()
    assert all(item.fresh_trace_passed for item in report.evaluations)
    assert any(item.scores["safety_compliance"] == 0.0 for item in report.evaluations)
    assert any(item.scores["safety_compliance"] == 1.0 for item in report.evaluations)


def test_gepa_runtime_zeros_reused_trace_candidate(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    pytest.importorskip("gepa")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"trace": complete_journal})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = run_gepa_experiment(
            _write_experiment(tmp_path),
            custom_candidate_proposer=_safe_proposer,
            replay_client=client,
        )
    finally:
        client.close()

    assert report.reused_trace_ids == (complete_journal["runId"],)
    assert any(not item.fresh_trace_passed and item.score == 0.0 for item in report.evaluations)


def test_held_out_validation_promotes_fresh_safe_candidate(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    pytest.importorskip("gepa")
    experiment_path, search_report, calls = _run_search(tmp_path, complete_journal, incomplete_journal)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        assert "approval-before-commit" in prompt
        trace = _replace_run_id(copy.deepcopy(complete_journal), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = validate_gepa_candidate_report(
            validation_id="held-out-gate",
            validation_version="1.0.0",
            experiment_manifest_path=experiment_path,
            experiment_report=search_report,
            validation_replay_path=_write_validation_replay(tmp_path),
            policy=GepaValidationPolicy(
                min_validation_score=1.0,
                min_task_count=2,
                max_train_validation_gap=0.0,
            ),
            replay_client=client,
        )
    finally:
        client.close()

    assert report.validation_score == 1.0
    assert report.train_validation_gap == 0.0
    assert report.validation_task_count == 2
    assert report.fresh_trace_passed
    assert report.reused_trace_ids == ()
    assert set(report.trace_ids).isdisjoint(search_report.observed_trace_ids)
    assert report.recommendation == "promote"
    assert report.reasons == ()


def test_held_out_validation_holds_reused_training_trace(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    pytest.importorskip("gepa")
    experiment_path, search_report, _ = _run_search(tmp_path, complete_journal, incomplete_journal)
    reused_id = search_report.observed_trace_ids[0]

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        trace = _replace_run_id(copy.deepcopy(complete_journal), reused_id)
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = validate_gepa_candidate_report(
            validation_id="held-out-gate",
            validation_version="1.0.0",
            experiment_manifest_path=experiment_path,
            experiment_report=search_report,
            validation_replay_path=_write_validation_replay(tmp_path, task_count=1),
            replay_client=client,
        )
    finally:
        client.close()

    assert report.validation_score == 0.0
    assert report.reused_trace_ids == (reused_id,)
    assert not report.fresh_trace_passed
    assert report.recommendation == "hold"
    assert "validation did not produce fresh trace evidence" in report.reasons


def test_held_out_validation_holds_fresh_unsafe_candidate(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    pytest.importorskip("gepa")
    experiment_path, search_report, calls = _run_search(tmp_path, complete_journal, incomplete_journal)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        trace = _replace_run_id(copy.deepcopy(incomplete_journal), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = validate_gepa_candidate_report(
            validation_id="held-out-gate",
            validation_version="1.0.0",
            experiment_manifest_path=experiment_path,
            experiment_report=search_report,
            validation_replay_path=_write_validation_replay(tmp_path, task_count=1),
            replay_client=client,
        )
    finally:
        client.close()

    assert report.fresh_trace_passed
    assert report.validation_scores["safety_compliance"] == 0.0
    assert report.recommendation == "hold"
    assert "validation safety compliance is below 1.0" in report.reasons


def test_path_based_validation_resolves_relative_artifacts(
    tmp_path: Path,
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    pytest.importorskip("gepa")
    experiment_path, search_report, calls = _run_search(tmp_path, complete_journal, incomplete_journal)
    report_path = tmp_path / "search-report.json"
    report_path.write_text(search_report.model_dump_json(indent=2), encoding="utf-8")
    _write_validation_replay(tmp_path, task_count=1)
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "agenttracelab.gepa-validation.v1",
                "validation_id": "relative-path-gate",
                "version": "1.0.0",
                "experiment_manifest": experiment_path.name,
                "experiment_report": report_path.name,
                "validation_replay_manifest": "validation-replay.json",
                "policy": {"min_validation_score": 1.0, "max_train_validation_gap": 0.0},
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        trace = _replace_run_id(copy.deepcopy(complete_journal), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        report = validate_gepa_candidate(manifest_path, replay_client=client)
    finally:
        client.close()

    assert report.validation_id == "relative-path-gate"
    assert report.recommendation == "promote"


def test_gepa_manifest_rejects_binding_without_seed_candidate() -> None:
    with pytest.raises(ValueError, match="missing seed candidate"):
        GepaExperimentManifest.model_validate(
            {
                "schema_version": "agenttracelab.gepa-experiment.v1",
                "experiment_id": "bad",
                "version": "1",
                "seed_candidate": {"prompt": "seed"},
                "objective": "Improve safely.",
                "replay_manifest": "replay.json",
                "bindings": [{"candidate_key": "missing", "request_path": "config.prompt"}],
            }
        )


def test_gepa_runtime_rejects_missing_request_binding_path(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    pytest.importorskip("gepa")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"trace": complete_journal})

    experiment_path = _write_experiment(tmp_path)
    payload = json.loads(experiment_path.read_text(encoding="utf-8"))
    payload["bindings"][0]["request_path"] = "agent_config.missing"
    experiment_path.write_text(json.dumps(payload), encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GepaRuntimeError, match="binding path does not exist"):
            run_gepa_experiment(
                experiment_path,
                custom_candidate_proposer=_safe_proposer,
                replay_client=client,
            )
    finally:
        client.close()


def test_openai_compatible_reflection_lm_uses_configured_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "improved"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleReflectionLM(
        ReflectionClientConfig(
            base_url="https://models.example/v1",
            model="reflection-model",
            api_key="secret",
        ),
        client=client,
    )
    try:
        assert model("reflect on this") == "improved"
    finally:
        client.close()

    assert captured["url"] == "https://models.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "reflection-model"
