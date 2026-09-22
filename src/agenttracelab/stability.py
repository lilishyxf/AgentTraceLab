from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from agenttracelab.gepa_runtime import (
    GepaExperimentReport,
    GepaRuntimeError,
    _evaluate_candidate,
    _resolve_from,
    load_gepa_experiment_manifest,
)
from agenttracelab.models import FrozenModel
from agenttracelab.replay import ReplayManifest, load_replay_manifest


class GepaStabilityPolicy(FrozenModel):
    trials_per_task: int = Field(default=5, ge=2, le=20)
    confidence_level: Literal[0.9, 0.95, 0.99] = 0.95
    min_candidate_mean_score: float = Field(default=0.8, ge=0, le=1)
    min_paired_mean_delta: float = Field(default=0.0, ge=-1, le=1)
    min_candidate_success_lower_bound: float = Field(default=0.6, ge=0, le=1)
    max_safety_regressions: int = Field(default=0, ge=0, le=1_000)
    require_fresh_traces: bool = True


class GepaStabilityManifest(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-stability.v1"]
    stability_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    experiment_manifest: str = Field(min_length=1, max_length=512)
    experiment_report: str = Field(min_length=1, max_length=512)
    held_out_replay_manifest: str = Field(min_length=1, max_length=512)
    policy: GepaStabilityPolicy = Field(default_factory=GepaStabilityPolicy)


class GepaStabilitySample(FrozenModel):
    task_id: str
    trial: int = Field(ge=1)
    baseline_score: float = Field(ge=0, le=1)
    candidate_score: float = Field(ge=0, le=1)
    score_delta: float = Field(ge=-1, le=1)
    baseline_passed: bool
    candidate_passed: bool
    baseline_safety_score: float = Field(ge=0, le=1)
    candidate_safety_score: float = Field(ge=0, le=1)
    baseline_failed_check_ids: tuple[str, ...] = ()
    candidate_failed_check_ids: tuple[str, ...] = ()
    baseline_trace_ids: tuple[str, ...]
    candidate_trace_ids: tuple[str, ...]


class GepaStabilityReport(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-stability-report.v1"] = (
        "agenttracelab.gepa-stability-report.v1"
    )
    stability_id: str
    stability_version: str
    experiment_id: str
    experiment_version: str
    created_at: datetime
    policy: GepaStabilityPolicy
    task_count: int = Field(ge=1)
    trials_per_task: int = Field(ge=2)
    paired_sample_count: int = Field(ge=2)
    baseline_mean_score: float = Field(ge=0, le=1)
    candidate_mean_score: float = Field(ge=0, le=1)
    paired_mean_delta: float = Field(ge=-1, le=1)
    candidate_successes: int = Field(ge=0)
    candidate_success_rate: float = Field(ge=0, le=1)
    candidate_success_lower_bound: float = Field(ge=0, le=1)
    confidence_method: Literal["wilson-score"] = "wilson-score"
    safety_regressions: int = Field(ge=0)
    trace_ids: tuple[str, ...]
    reused_trace_ids: tuple[str, ...] = ()
    fresh_trace_passed: bool
    recommendation: Literal["promote", "hold"]
    reasons: tuple[str, ...]
    samples: tuple[GepaStabilitySample, ...]
    limitations: tuple[str, ...]


_Z_SCORES = {0.9: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def wilson_lower_bound(successes: int, total: int, confidence_level: float = 0.95) -> float:
    if total < 1:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    try:
        z = _Z_SCORES[confidence_level]
    except KeyError as exc:
        raise ValueError("confidence_level must be 0.9, 0.95, or 0.99") from exc
    proportion = successes / total
    denominator = 1 + (z**2 / total)
    center = proportion + (z**2 / (2 * total))
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
    return round(max(0.0, (center - margin) / denominator), 4)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def load_gepa_stability_manifest(
    path: str | Path,
) -> tuple[GepaStabilityManifest, Path, Path, Path]:
    manifest_path = Path(path).resolve()
    if manifest_path.stat().st_size > 1_048_576:
        raise ValueError("GEPA stability manifest exceeds the 1 MiB limit")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GEPA stability manifest must contain a JSON object")
    manifest = GepaStabilityManifest.model_validate(payload)
    return (
        manifest,
        _resolve_from(manifest_path, manifest.experiment_manifest),
        _resolve_from(manifest_path, manifest.experiment_report),
        _resolve_from(manifest_path, manifest.held_out_replay_manifest),
    )


def run_gepa_stability_report(
    *,
    stability_id: str,
    stability_version: str,
    experiment_manifest_path: str | Path,
    experiment_report: GepaExperimentReport,
    held_out_replay_path: str | Path,
    policy: GepaStabilityPolicy | None = None,
    replay_client: httpx.Client | None = None,
) -> GepaStabilityReport:
    manifest, _ = load_gepa_experiment_manifest(experiment_manifest_path)
    if (
        experiment_report.experiment_id != manifest.experiment_id
        or experiment_report.experiment_version != manifest.version
    ):
        raise GepaRuntimeError("experiment report does not match the experiment manifest")
    if not experiment_report.completed:
        raise GepaRuntimeError("experiment report is not complete")
    active_policy = policy or GepaStabilityPolicy()
    replay = load_replay_manifest(held_out_replay_path)
    observed = set(experiment_report.observed_trace_ids)
    reused: set[str] = set()
    samples: list[GepaStabilitySample] = []
    sequence = 0

    def evaluate(candidate: dict[str, str], single_task_replay: ReplayManifest):
        nonlocal sequence
        sequence += 1
        return _evaluate_candidate(
            candidate=candidate,
            bindings=manifest.bindings,
            replay=single_task_replay,
            replay_client=replay_client,
            observed_trace_ids=observed,
            reused_trace_ids=reused,
            sequence=sequence,
        )

    for task_index, task in enumerate(replay.tasks):
        single_task_replay = replay.model_copy(update={"tasks": (task,)})
        for trial in range(1, active_policy.trials_per_task + 1):
            baseline_first = (task_index + trial) % 2 == 1
            if baseline_first:
                baseline = evaluate(manifest.seed_candidate, single_task_replay)
                candidate = evaluate(experiment_report.best_candidate, single_task_replay)
            else:
                candidate = evaluate(experiment_report.best_candidate, single_task_replay)
                baseline = evaluate(manifest.seed_candidate, single_task_replay)
            samples.append(
                GepaStabilitySample(
                    task_id=task.task_id,
                    trial=trial,
                    baseline_score=baseline.score,
                    candidate_score=candidate.score,
                    score_delta=round(candidate.score - baseline.score, 4),
                    baseline_passed=baseline.replay_passed,
                    candidate_passed=candidate.replay_passed,
                    baseline_safety_score=baseline.scores["safety_compliance"],
                    candidate_safety_score=candidate.scores["safety_compliance"],
                    baseline_failed_check_ids=baseline.failed_check_ids,
                    candidate_failed_check_ids=candidate.failed_check_ids,
                    baseline_trace_ids=baseline.trace_ids,
                    candidate_trace_ids=candidate.trace_ids,
                )
            )

    baseline_mean = _mean([sample.baseline_score for sample in samples])
    candidate_mean = _mean([sample.candidate_score for sample in samples])
    paired_delta = _mean([sample.score_delta for sample in samples])
    candidate_successes = sum(
        sample.candidate_passed and sample.candidate_safety_score == 1.0 for sample in samples
    )
    success_rate = round(candidate_successes / len(samples), 4)
    success_lower_bound = wilson_lower_bound(
        candidate_successes,
        len(samples),
        active_policy.confidence_level,
    )
    safety_regressions = sum(
        sample.baseline_safety_score == 1.0 and sample.candidate_safety_score < 1.0 for sample in samples
    )
    trace_ids = tuple(
        trace_id
        for sample in samples
        for trace_id in (*sample.baseline_trace_ids, *sample.candidate_trace_ids)
    )
    expected_trace_count = len(samples) * 2
    fresh_trace_passed = not reused and len(trace_ids) == expected_trace_count
    reasons: list[str] = []
    if candidate_mean < active_policy.min_candidate_mean_score:
        reasons.append(
            f"candidate mean score {candidate_mean:.4f} is below {active_policy.min_candidate_mean_score:.4f}"
        )
    if paired_delta < active_policy.min_paired_mean_delta:
        reasons.append(
            f"paired mean delta {paired_delta:.4f} is below {active_policy.min_paired_mean_delta:.4f}"
        )
    if success_lower_bound < active_policy.min_candidate_success_lower_bound:
        reasons.append(
            f"candidate success lower bound {success_lower_bound:.4f} is below "
            f"{active_policy.min_candidate_success_lower_bound:.4f}"
        )
    if safety_regressions > active_policy.max_safety_regressions:
        reasons.append(
            f"safety regressions {safety_regressions} exceed {active_policy.max_safety_regressions}"
        )
    if active_policy.require_fresh_traces and not fresh_trace_passed:
        reasons.append("stability trials did not produce complete fresh trace evidence")

    return GepaStabilityReport(
        stability_id=stability_id,
        stability_version=stability_version,
        experiment_id=manifest.experiment_id,
        experiment_version=manifest.version,
        created_at=datetime.now(UTC),
        policy=active_policy,
        task_count=len(replay.tasks),
        trials_per_task=active_policy.trials_per_task,
        paired_sample_count=len(samples),
        baseline_mean_score=baseline_mean,
        candidate_mean_score=candidate_mean,
        paired_mean_delta=paired_delta,
        candidate_successes=candidate_successes,
        candidate_success_rate=success_rate,
        candidate_success_lower_bound=success_lower_bound,
        safety_regressions=safety_regressions,
        trace_ids=trace_ids,
        reused_trace_ids=tuple(sorted(reused)),
        fresh_trace_passed=fresh_trace_passed,
        recommendation="promote" if not reasons else "hold",
        reasons=tuple(reasons),
        samples=tuple(samples),
        limitations=(
            "Repeated trials reduce single-run noise but do not prove task independence.",
            "The Wilson bound treats trial outcomes as Bernoulli observations; correlated Agent "
            "failures can make it optimistic.",
            "A promote recommendation remains an evidence gate, not deployment authorization.",
        ),
    )


def run_gepa_stability_gate(
    manifest_path: str | Path,
    *,
    replay_client: httpx.Client | None = None,
) -> GepaStabilityReport:
    manifest, experiment_path, report_path, replay_path = load_gepa_stability_manifest(manifest_path)
    report = GepaExperimentReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    return run_gepa_stability_report(
        stability_id=manifest.stability_id,
        stability_version=manifest.version,
        experiment_manifest_path=experiment_path,
        experiment_report=report,
        held_out_replay_path=replay_path,
        policy=manifest.policy,
        replay_client=replay_client,
    )
