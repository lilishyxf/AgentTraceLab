from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from agenttracelab.cli import (
    _agent_lightning_reward_event,
    _curate_gepa_hard_cases,
    _gepa_evaluation,
    _mine_gepa_hard_cases,
    _optimization_feedback,
    _run_gepa_experiment,
    _run_gepa_stability_gate,
    _training_reward,
    _validate_gepa_candidate,
    build_parser,
)
from agenttracelab.service import AgentTraceService
from agenttracelab.storage import Database


def _database_url(tmp_path: Path) -> str:
    return f"sqlite+pysqlite:///{(tmp_path / 'optimization.db').as_posix()}"


def test_cli_writes_feedback_and_reward_json(
    tmp_path: Path,
    incomplete_journal: dict,
) -> None:
    database_url = _database_url(tmp_path)
    database = Database(database_url)
    database.create_schema()
    service = AgentTraceService(database)
    trace, _ = service.import_wasmhatch(incomplete_journal)
    feedback_output = tmp_path / "feedback.json"
    reward_output = tmp_path / "reward.json"

    assert (
        _optimization_feedback(
            Namespace(
                database_url=database_url,
                trace_id=trace.trace_id,
                output=str(feedback_output),
            )
        )
        == 0
    )
    assert (
        _training_reward(
            Namespace(
                database_url=database_url,
                trace_id=trace.trace_id,
                output=str(reward_output),
            )
        )
        == 0
    )
    assert json.loads(feedback_output.read_text(encoding="utf-8"))["gate_passed"] is False
    assert json.loads(reward_output.read_text(encoding="utf-8"))["safety_blocked"] is True


def test_cli_writes_agent_lightning_v1_reward_event(
    tmp_path: Path,
    incomplete_journal: dict,
) -> None:
    database_url = _database_url(tmp_path)
    database = Database(database_url)
    database.create_schema()
    service = AgentTraceService(database)
    trace, _ = service.import_wasmhatch(incomplete_journal)
    output = tmp_path / "agent-lightning-reward-event.json"

    assert (
        _agent_lightning_reward_event(
            Namespace(
                database_url=database_url,
                trace_id=trace.trace_id,
                rollout_id="rollout-cli",
                attempt_id="0",
                output=str(output),
            )
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["endpoint_path"] == "/rollouts/rollout-cli/attempt/0/events"
    assert payload["body"]["event_type"] == "reward"
    assert payload["body"]["data"]["value"] == 0.0


def test_cli_rejects_unsafe_agent_lightning_rollout_id(
    tmp_path: Path,
    complete_journal: dict,
) -> None:
    database_url = _database_url(tmp_path)
    database = Database(database_url)
    database.create_schema()
    service = AgentTraceService(database)
    trace, _ = service.import_wasmhatch(complete_journal)

    assert (
        _agent_lightning_reward_event(
            Namespace(
                database_url=database_url,
                trace_id=trace.trace_id,
                rollout_id="../other-rollout",
                attempt_id="0",
                output=None,
            )
        )
        == 10
    )


def test_cli_writes_gepa_evaluation_record(
    tmp_path: Path,
    incomplete_journal: dict,
) -> None:
    database_url = _database_url(tmp_path)
    database = Database(database_url)
    database.create_schema()
    service = AgentTraceService(database)
    trace, _ = service.import_wasmhatch(incomplete_journal)
    output = tmp_path / "gepa-evaluation.json"

    assert (
        _gepa_evaluation(
            Namespace(
                database_url=database_url,
                trace_id=trace.trace_id,
                output=str(output),
            )
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "agenttracelab.gepa-evaluation.v1"
    assert payload["score"] == 0.0
    assert payload["info"]["scores"]["safety_compliance"] == 0.0


def test_cli_missing_trace_is_non_success(tmp_path: Path) -> None:
    args = Namespace(database_url=_database_url(tmp_path), trace_id="missing", output=None)

    assert _optimization_feedback(args) == 11
    assert _training_reward(args) == 11
    assert _gepa_evaluation(args) == 11
    assert (
        _agent_lightning_reward_event(
            Namespace(
                database_url=args.database_url,
                trace_id="missing",
                rollout_id="rollout-missing",
                attempt_id="0",
                output=None,
            )
        )
        == 11
    )


def test_parser_registers_optimization_commands() -> None:
    parser = build_parser()

    feedback = parser.parse_args(["optimization-feedback", "trace-one"])
    reward = parser.parse_args(["training-reward", "trace-one"])
    agent_lightning = parser.parse_args(
        [
            "agent-lightning-reward-event",
            "trace-one",
            "--rollout-id",
            "rollout-one",
        ]
    )
    gepa = parser.parse_args(["gepa-evaluation", "trace-one"])
    gepa_run = parser.parse_args(
        [
            "run-gepa-experiment",
            "experiment.json",
            "--reflection-base-url",
            "https://models.example/v1",
            "--reflection-model",
            "reflection-model",
        ]
    )
    gepa_validation = parser.parse_args(
        ["validate-gepa-candidate", "validation.json", "--output", "report.json"]
    )
    gepa_stability = parser.parse_args(
        ["run-gepa-stability-gate", "stability.json", "--output", "stability-report.json"]
    )
    hard_cases = parser.parse_args(
        [
            "mine-gepa-hard-cases",
            "stability-report.json",
            "--dataset-id",
            "challenge-set",
            "--dataset-version",
            "1.0.0",
        ]
    )
    curation = parser.parse_args(
        [
            "curate-gepa-hard-cases",
            "hard-cases.json",
            "resolutions.json",
            "--output",
            "curated.json",
        ]
    )

    assert feedback.trace_id == "trace-one"
    assert reward.trace_id == "trace-one"
    assert agent_lightning.trace_id == "trace-one"
    assert agent_lightning.rollout_id == "rollout-one"
    assert agent_lightning.attempt_id == "0"
    assert gepa.trace_id == "trace-one"
    assert gepa_run.manifest == "experiment.json"
    assert gepa_run.reflection_base_url == "https://models.example/v1"
    assert gepa_run.reflection_model == "reflection-model"
    assert gepa_validation.manifest == "validation.json"
    assert gepa_validation.output == "report.json"
    assert gepa_stability.manifest == "stability.json"
    assert gepa_stability.output == "stability-report.json"
    assert hard_cases.report == "stability-report.json"
    assert hard_cases.dataset_id == "challenge-set"
    assert hard_cases.dataset_version == "1.0.0"
    assert curation.queue == "hard-cases.json"
    assert curation.resolutions == "resolutions.json"
    assert curation.output == "curated.json"


def test_parser_accepts_slow_claim_judge_timeout_overrides() -> None:
    parser = build_parser()

    verify = parser.parse_args(
        [
            "verify-deepresearch-claims",
            "snapshot.json",
            "--judge-timeout-seconds",
            "180",
            "--judge-max-tokens",
            "12000",
        ]
    )
    benchmark = parser.parse_args(
        [
            "run-deepresearch-benchmark",
            "manifest.json",
            "--output-directory",
            "results",
            "--claim-judge-timeout-seconds",
            "240",
            "--claim-judge-max-tokens",
            "16000",
        ]
    )

    assert verify.judge_timeout_seconds == 180
    assert verify.judge_max_tokens == 12000
    assert benchmark.claim_judge_timeout_seconds == 240
    assert benchmark.claim_judge_max_tokens == 16000


def test_gepa_run_cli_requires_explicit_reflection_configuration() -> None:
    assert (
        _run_gepa_experiment(
            Namespace(
                manifest="experiment.json",
                reflection_base_url=None,
                reflection_model=None,
                api_key_env="AGENTTRACELAB_GEPA_API_KEY",
                output=None,
            )
        )
        == 12
    )


def test_gepa_validation_cli_reports_missing_manifest_as_non_success() -> None:
    assert (
        _validate_gepa_candidate(
            Namespace(
                manifest="missing-validation.json",
                output=None,
            )
        )
        == 13
    )


def test_gepa_stability_cli_reports_missing_manifest_as_non_success() -> None:
    assert (
        _run_gepa_stability_gate(
            Namespace(
                manifest="missing-stability.json",
                output=None,
            )
        )
        == 14
    )


def test_hard_case_cli_reports_missing_stability_report_as_non_success() -> None:
    assert (
        _mine_gepa_hard_cases(
            Namespace(
                report="missing-stability.json",
                dataset_id="challenge-set",
                dataset_version="1.0.0",
                max_cases=50,
                max_per_signature=5,
                exclude_candidate_failures=False,
                exclude_score_regressions=False,
                exclude_resolved_baseline_failures=False,
                output=None,
            )
        )
        == 15
    )


def test_curation_cli_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "curated.json"
    output.write_text("do-not-overwrite", encoding="utf-8")

    assert (
        _curate_gepa_hard_cases(
            Namespace(
                queue="hard-cases.json",
                resolutions="resolutions.json",
                output=str(output),
            )
        )
        == 16
    )
    assert output.read_text(encoding="utf-8") == "do-not-overwrite"
