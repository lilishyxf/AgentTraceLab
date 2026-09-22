from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field, SecretStr, model_validator

from agenttracelab.models import EvaluationReport, FrozenModel, TraceEnvelope
from agenttracelab.optimization import build_gepa_evaluation
from agenttracelab.replay import ReplayManifest, load_replay_manifest, run_replay_manifest


class GepaRuntimeError(RuntimeError):
    pass


class GepaDependencyError(GepaRuntimeError):
    pass


class GepaReflectionError(GepaRuntimeError):
    pass


class GepaCandidateBinding(FrozenModel):
    candidate_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    request_path: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_request_path(self) -> GepaCandidateBinding:
        segments = self.request_path.split(".")
        if any(not segment or segment in {"__proto__", "constructor", "prototype"} for segment in segments):
            raise ValueError("request_path must contain safe non-empty object keys")
        return self


class GepaExperimentManifest(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-experiment.v1"]
    experiment_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    seed_candidate: dict[str, str] = Field(min_length=1, max_length=16)
    objective: str = Field(min_length=1, max_length=4_096)
    background: str = Field(default="", max_length=8_192)
    replay_manifest: str = Field(min_length=1, max_length=512)
    bindings: tuple[GepaCandidateBinding, ...] = Field(min_length=1, max_length=16)
    max_metric_calls: int = Field(default=20, ge=2, le=1_000)
    max_candidate_proposals: int = Field(default=5, ge=1, le=100)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def validate_candidates_and_bindings(self) -> GepaExperimentManifest:
        if any(not key or len(key) > 128 for key in self.seed_candidate):
            raise ValueError("candidate keys must contain 1-128 characters")
        if any(not value or len(value) > 32_000 for value in self.seed_candidate.values()):
            raise ValueError("candidate values must contain 1-32000 characters")
        if sum(len(value) for value in self.seed_candidate.values()) > 128_000:
            raise ValueError("candidate values exceed the 128000 character total limit")
        binding_keys = [binding.candidate_key for binding in self.bindings]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("candidate bindings must be unique")
        missing = set(binding_keys) - set(self.seed_candidate)
        if missing:
            raise ValueError("bindings reference missing seed candidate keys: " + ", ".join(sorted(missing)))
        return self


class GepaCandidateEvaluation(FrozenModel):
    sequence: int = Field(ge=1)
    candidate_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    score: float = Field(ge=0, le=1)
    scores: dict[str, float]
    trace_ids: tuple[str, ...]
    failed_check_ids: tuple[str, ...] = ()
    fresh_trace_passed: bool
    replay_passed: bool
    diagnostics: str = Field(min_length=1, max_length=16_000)


class GepaExperimentReport(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-experiment-report.v1"] = (
        "agenttracelab.gepa-experiment-report.v1"
    )
    experiment_id: str
    experiment_version: str
    created_at: datetime
    engine: Literal["gepa.optimize_anything"] = "gepa.optimize_anything"
    engine_version: str
    completed: bool
    improved: bool
    seed_score: float = Field(ge=0, le=1)
    best_score: float = Field(ge=0, le=1)
    best_candidate: dict[str, str]
    candidate_count: int = Field(ge=1)
    total_metric_calls: int = Field(ge=1)
    candidate_fingerprints: tuple[str, ...]
    observed_trace_ids: tuple[str, ...]
    reused_trace_ids: tuple[str, ...] = ()
    evaluations: tuple[GepaCandidateEvaluation, ...]
    limitations: tuple[str, ...]


class GepaValidationPolicy(FrozenModel):
    min_validation_score: float = Field(default=0.8, ge=0, le=1)
    min_task_count: int = Field(default=1, ge=1, le=100)
    max_train_validation_gap: float = Field(default=0.15, ge=0, le=1)
    require_safety_pass: bool = True
    require_fresh_traces: bool = True
    require_expectations_match: bool = True


class GepaValidationManifest(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-validation.v1"]
    validation_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    experiment_manifest: str = Field(min_length=1, max_length=512)
    experiment_report: str = Field(min_length=1, max_length=512)
    validation_replay_manifest: str = Field(min_length=1, max_length=512)
    policy: GepaValidationPolicy = Field(default_factory=GepaValidationPolicy)


class GepaCandidateValidationReport(FrozenModel):
    schema_version: Literal["agenttracelab.gepa-candidate-validation.v1"] = (
        "agenttracelab.gepa-candidate-validation.v1"
    )
    validation_id: str
    validation_version: str
    experiment_id: str
    experiment_version: str
    created_at: datetime
    training_score: float = Field(ge=0, le=1)
    validation_score: float = Field(ge=0, le=1)
    train_validation_gap: float = Field(ge=-1, le=1)
    validation_scores: dict[str, float]
    validation_task_count: int = Field(ge=1)
    trace_ids: tuple[str, ...]
    reused_trace_ids: tuple[str, ...] = ()
    failed_check_ids: tuple[str, ...] = ()
    fresh_trace_passed: bool
    expectations_matched: bool
    recommendation: Literal["promote", "hold"]
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]


class ReflectionClientConfig(FrozenModel):
    base_url: str = Field(min_length=1, max_length=2_048)
    model: str = Field(min_length=1, max_length=256)
    api_key: SecretStr
    timeout_seconds: float = Field(default=60.0, ge=1, le=300)
    max_attempts: int = Field(default=3, ge=1, le=5)
    max_tokens: int = Field(default=4_096, ge=256, le=16_384)

    @model_validator(mode="after")
    def validate_base_url(self) -> ReflectionClientConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return self


class OpenAICompatibleReflectionLM:
    def __init__(
        self,
        config: ReflectionClientConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._client = client
        self._sleep = sleep

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        request_payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "content-type": "application/json",
        }
        owned_client = self._client is None
        client = self._client or httpx.Client(timeout=self.config.timeout_seconds)
        try:
            for attempt in range(1, self.config.max_attempts + 1):
                try:
                    response = client.post(url, headers=headers, json=request_payload)
                except httpx.HTTPError as exc:
                    if attempt == self.config.max_attempts:
                        raise GepaReflectionError(
                            f"reflection request failed after {attempt} attempts"
                        ) from exc
                    self._sleep(0.25 * (2 ** (attempt - 1)))
                    continue
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    if attempt == self.config.max_attempts:
                        raise GepaReflectionError(
                            "reflection provider returned "
                            f"HTTP {response.status_code} after {attempt} attempts"
                        )
                    self._sleep(0.25 * (2 ** (attempt - 1)))
                    continue
                if not 200 <= response.status_code < 300:
                    raise GepaReflectionError(
                        f"reflection provider rejected the request with HTTP {response.status_code}"
                    )
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise GepaReflectionError(
                        "reflection response is not a valid chat-completions envelope"
                    ) from exc
                if not isinstance(content, str) or not content.strip():
                    raise GepaReflectionError("reflection response content is empty")
                if len(content) > 128_000:
                    raise GepaReflectionError("reflection response exceeds the 128000 character limit")
                return content
        finally:
            if owned_client:
                client.close()
        raise GepaReflectionError("reflection request exhausted without a response")


def load_gepa_experiment_manifest(path: str | Path) -> tuple[GepaExperimentManifest, Path]:
    manifest_path = Path(path).resolve()
    if manifest_path.stat().st_size > 1_048_576:
        raise ValueError("GEPA experiment manifest exceeds the 1 MiB limit")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GEPA experiment manifest must contain a JSON object")
    manifest = GepaExperimentManifest.model_validate(payload)
    replay_path = Path(manifest.replay_manifest)
    if not replay_path.is_absolute():
        replay_path = manifest_path.parent / replay_path
    return manifest, replay_path.resolve()


def _resolve_from(base_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_path.parent / path
    return path.resolve()


def load_gepa_validation_manifest(
    path: str | Path,
) -> tuple[GepaValidationManifest, Path, Path, Path]:
    manifest_path = Path(path).resolve()
    if manifest_path.stat().st_size > 1_048_576:
        raise ValueError("GEPA validation manifest exceeds the 1 MiB limit")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GEPA validation manifest must contain a JSON object")
    manifest = GepaValidationManifest.model_validate(payload)
    return (
        manifest,
        _resolve_from(manifest_path, manifest.experiment_manifest),
        _resolve_from(manifest_path, manifest.experiment_report),
        _resolve_from(manifest_path, manifest.validation_replay_manifest),
    )


def _candidate_fingerprint(candidate: Mapping[str, str]) -> str:
    encoded = json.dumps(candidate, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _set_existing_path(request: dict[str, Any], path: str, value: str) -> None:
    segments = path.split(".")
    cursor: Any = request
    for segment in segments[:-1]:
        if not isinstance(cursor, dict) or segment not in cursor:
            raise GepaRuntimeError(f"candidate binding path does not exist: {path}")
        cursor = cursor[segment]
    leaf = segments[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise GepaRuntimeError(f"candidate binding path does not exist: {path}")
    cursor[leaf] = value


def _bind_candidate(
    replay: ReplayManifest,
    candidate: Mapping[str, str],
    bindings: Sequence[GepaCandidateBinding],
) -> ReplayManifest:
    tasks = []
    for task in replay.tasks:
        request = copy.deepcopy(task.request)
        for binding in bindings:
            if binding.candidate_key not in candidate:
                raise GepaRuntimeError(f"candidate is missing bound key: {binding.candidate_key}")
            _set_existing_path(request, binding.request_path, candidate[binding.candidate_key])
        tasks.append(task.model_copy(update={"request": request}))
    return replay.model_copy(update={"tasks": tuple(tasks)})


def _mean(values: Sequence[float], denominator: int) -> float:
    return round(sum(values) / denominator, 4) if denominator else 0.0


def _evaluate_candidate(
    *,
    candidate: Mapping[str, str],
    bindings: Sequence[GepaCandidateBinding],
    replay: ReplayManifest,
    replay_client: httpx.Client | None,
    observed_trace_ids: set[str],
    reused_trace_ids: set[str],
    sequence: int,
) -> GepaCandidateEvaluation:
    fingerprint = _candidate_fingerprint(candidate)
    captured: list[tuple[TraceEnvelope, EvaluationReport]] = []
    bound_replay = _bind_candidate(replay, candidate, bindings)
    replay_report = run_replay_manifest(
        bound_replay,
        client=replay_client,
        on_evaluated=lambda trace, evaluation: captured.append((trace, evaluation)),
    )
    records = [build_gepa_evaluation(trace, evaluation) for trace, evaluation in captured]
    trace_ids = tuple(trace.trace_id for trace, _ in captured)
    duplicates = {trace_id for trace_id in trace_ids if trace_ids.count(trace_id) > 1}
    duplicates.update(set(trace_ids) & observed_trace_ids)
    fresh_trace_passed = not duplicates and len(trace_ids) == replay_report.task_count
    observed_trace_ids.update(trace_ids)
    reused_trace_ids.update(duplicates)
    denominator = replay_report.task_count
    scores = {
        "deterministic_gate": _mean(
            [record.info.scores["deterministic_gate"] for record in records], denominator
        ),
        "process_quality": _mean([record.info.scores["process_quality"] for record in records], denominator),
        "safety_compliance": _mean(
            [record.info.scores["safety_compliance"] for record in records], denominator
        ),
        "fresh_trace_integrity": 1.0 if fresh_trace_passed else 0.0,
    }
    score = _mean([record.score for record in records], denominator)
    if not fresh_trace_passed:
        score = 0.0
    failed_check_ids = tuple(
        dict.fromkeys(check_id for record in records for check_id in record.info.failed_check_ids)
    )
    diagnostics = [
        f"{result.task_id}: {result.status}; "
        f"{'matched' if result.matched_expectation else 'did not match'} expectation."
        for result in replay_report.results
    ]
    diagnostics.extend(record.info.diagnostics for record in records)
    if duplicates:
        diagnostics.append("Fresh-trace gate failed for reused trace IDs: " + ", ".join(sorted(duplicates)))
    rendered_diagnostics = "\n".join(diagnostics)
    if len(rendered_diagnostics) > 16_000:
        rendered_diagnostics = f"{rendered_diagnostics[:15_985]}...[truncated]"
    return GepaCandidateEvaluation(
        sequence=sequence,
        candidate_fingerprint=fingerprint,
        score=score,
        scores=scores,
        trace_ids=trace_ids,
        failed_check_ids=failed_check_ids,
        fresh_trace_passed=fresh_trace_passed,
        replay_passed=replay_report.passed,
        diagnostics=rendered_diagnostics,
    )


def run_gepa_experiment(
    manifest_path: str | Path,
    *,
    reflection_lm: Callable[[str | list[dict[str, Any]]], str] | None = None,
    custom_candidate_proposer: Callable[
        [dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]],
        dict[str, str],
    ]
    | None = None,
    replay_client: httpx.Client | None = None,
) -> GepaExperimentReport:
    manifest, replay_path = load_gepa_experiment_manifest(manifest_path)
    replay = load_replay_manifest(replay_path)
    if reflection_lm is None and custom_candidate_proposer is None:
        raise GepaRuntimeError("a reflection LM or custom candidate proposer is required")
    try:
        from gepa.optimize_anything import (
            EngineConfig,
            GEPAConfig,
            ReflectionConfig,
            optimize_anything,
        )
    except ImportError as exc:
        raise GepaDependencyError(
            "GEPA runtime is not installed; run `uv sync --extra optimization`"
        ) from exc

    try:
        engine_version = package_version("gepa")
    except PackageNotFoundError:
        engine_version = "unknown"

    observed_trace_ids: set[str] = set()
    reused_trace_ids: set[str] = set()
    evaluations: list[GepaCandidateEvaluation] = []

    def evaluator(candidate: dict[str, str]) -> tuple[float, dict[str, Any]]:
        evaluation = _evaluate_candidate(
            candidate=candidate,
            bindings=manifest.bindings,
            replay=replay,
            replay_client=replay_client,
            observed_trace_ids=observed_trace_ids,
            reused_trace_ids=reused_trace_ids,
            sequence=len(evaluations) + 1,
        )
        evaluations.append(evaluation)
        side_info = evaluation.model_dump(mode="json")
        side_info["preservation_constraints"] = (
            "Do not weaken approval-before-commit or privacy gates.",
            "Every candidate evaluation must produce fresh trace IDs.",
            "Do not change the replay task contract or evaluator to improve the score.",
        )
        return evaluation.score, side_info

    config = GEPAConfig(
        engine=EngineConfig(
            seed=manifest.seed,
            max_metric_calls=manifest.max_metric_calls,
            max_candidate_proposals=manifest.max_candidate_proposals,
            parallel=False,
            cache_evaluation=False,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            custom_candidate_proposer=custom_candidate_proposer,
            skip_perfect_score=False,
        ),
    )
    result = optimize_anything(
        seed_candidate=manifest.seed_candidate,
        evaluator=evaluator,
        objective=manifest.objective,
        background=manifest.background,
        config=config,
    )
    if not evaluations:
        raise GepaRuntimeError("GEPA completed without evaluating any candidate")
    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in best_candidate.items()
    ):
        raise GepaRuntimeError("GEPA returned an invalid best candidate")
    best_score = round(float(result.val_aggregate_scores[result.best_idx]), 4)
    seed_fingerprint = _candidate_fingerprint(manifest.seed_candidate)
    seed_scores = [item.score for item in evaluations if item.candidate_fingerprint == seed_fingerprint]
    seed_score = seed_scores[0] if seed_scores else evaluations[0].score
    candidates = tuple(_candidate_fingerprint(candidate) for candidate in result.candidates)
    total_metric_calls = int(result.total_metric_calls or len(evaluations))
    return GepaExperimentReport(
        experiment_id=manifest.experiment_id,
        experiment_version=manifest.version,
        created_at=datetime.now(UTC),
        engine_version=engine_version,
        completed=True,
        improved=best_score > seed_score,
        seed_score=seed_score,
        best_score=best_score,
        best_candidate=best_candidate,
        candidate_count=len(result.candidates),
        total_metric_calls=max(total_metric_calls, 1),
        candidate_fingerprints=candidates,
        observed_trace_ids=tuple(sorted(observed_trace_ids)),
        reused_trace_ids=tuple(sorted(reused_trace_ids)),
        evaluations=tuple(evaluations),
        limitations=(
            "The report measures the configured replay tasks, not general model quality.",
            "A deterministic custom proposer is an integration smoke, not LLM reflection.",
            "Promotion still requires held-out tasks and review of the selected candidate.",
        ),
    )


def validate_gepa_candidate_report(
    *,
    validation_id: str,
    validation_version: str,
    experiment_manifest_path: str | Path,
    experiment_report: GepaExperimentReport,
    validation_replay_path: str | Path,
    policy: GepaValidationPolicy | None = None,
    replay_client: httpx.Client | None = None,
) -> GepaCandidateValidationReport:
    manifest, _ = load_gepa_experiment_manifest(experiment_manifest_path)
    if (
        experiment_report.experiment_id != manifest.experiment_id
        or experiment_report.experiment_version != manifest.version
    ):
        raise GepaRuntimeError("experiment report does not match the experiment manifest")
    if not experiment_report.completed:
        raise GepaRuntimeError("experiment report is not complete")
    active_policy = policy or GepaValidationPolicy()
    replay = load_replay_manifest(validation_replay_path)
    observed = set(experiment_report.observed_trace_ids)
    reused: set[str] = set()
    evaluation = _evaluate_candidate(
        candidate=experiment_report.best_candidate,
        bindings=manifest.bindings,
        replay=replay,
        replay_client=replay_client,
        observed_trace_ids=observed,
        reused_trace_ids=reused,
        sequence=1,
    )
    gap = round(experiment_report.best_score - evaluation.score, 4)
    reasons: list[str] = []
    if len(replay.tasks) < active_policy.min_task_count:
        reasons.append(f"validation task count {len(replay.tasks)} is below {active_policy.min_task_count}")
    if evaluation.score < active_policy.min_validation_score:
        reasons.append(
            f"validation score {evaluation.score:.4f} is below {active_policy.min_validation_score:.4f}"
        )
    if gap > active_policy.max_train_validation_gap:
        reasons.append(f"train-validation gap {gap:.4f} exceeds {active_policy.max_train_validation_gap:.4f}")
    if active_policy.require_safety_pass and evaluation.scores["safety_compliance"] < 1.0:
        reasons.append("validation safety compliance is below 1.0")
    if active_policy.require_fresh_traces and not evaluation.fresh_trace_passed:
        reasons.append("validation did not produce fresh trace evidence")
    if active_policy.require_expectations_match and not evaluation.replay_passed:
        reasons.append("one or more validation tasks did not match expectations")
    return GepaCandidateValidationReport(
        validation_id=validation_id,
        validation_version=validation_version,
        experiment_id=manifest.experiment_id,
        experiment_version=manifest.version,
        created_at=datetime.now(UTC),
        training_score=experiment_report.best_score,
        validation_score=evaluation.score,
        train_validation_gap=gap,
        validation_scores=evaluation.scores,
        validation_task_count=len(replay.tasks),
        trace_ids=evaluation.trace_ids,
        reused_trace_ids=tuple(sorted(reused)),
        failed_check_ids=evaluation.failed_check_ids,
        fresh_trace_passed=evaluation.fresh_trace_passed,
        expectations_matched=evaluation.replay_passed,
        recommendation="promote" if not reasons else "hold",
        reasons=tuple(reasons),
        limitations=(
            "Validation quality depends on how representative and independent the held-out tasks are.",
            "A promote recommendation is an evidence gate, not automatic production deployment.",
        ),
    )


def validate_gepa_candidate(
    manifest_path: str | Path,
    *,
    replay_client: httpx.Client | None = None,
) -> GepaCandidateValidationReport:
    manifest, experiment_path, report_path, replay_path = load_gepa_validation_manifest(manifest_path)
    report = GepaExperimentReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    return validate_gepa_candidate_report(
        validation_id=manifest.validation_id,
        validation_version=manifest.version,
        experiment_manifest_path=experiment_path,
        experiment_report=report,
        validation_replay_path=replay_path,
        policy=manifest.policy,
        replay_client=replay_client,
    )


def reflection_lm_from_environment(
    *,
    base_url: str,
    model: str,
    api_key_env: str,
) -> OpenAICompatibleReflectionLM:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise GepaRuntimeError(f"missing reflection API key environment variable: {api_key_env}")
    return OpenAICompatibleReflectionLM(
        ReflectionClientConfig(base_url=base_url, model=model, api_key=api_key)
    )
