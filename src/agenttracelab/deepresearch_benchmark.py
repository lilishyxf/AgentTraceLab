from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from pydantic import Field, model_validator

from agenttracelab.adapters import adapt_deepresearch_snapshot, export_deepresearch_sqlite_snapshot
from agenttracelab.claim_evidence import (
    CLAIM_JUDGE_PROMPT_VERSION,
    ClaimEvidenceReport,
    ClaimJudgeError,
    ClaimSupportJudge,
    verify_deepresearch_claims,
)
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.models import EvaluationReport, FrozenModel, TraceEnvelope
from agenttracelab.stability import wilson_lower_bound

TERMINAL_RUN_STATUSES = frozenset(
    {
        "completed",
        "failed",
        "budget_exhausted",
        "cancelled",
        "paused",
        "needs_user_input",
        "interrupted",
    }
)
JudgeSeparation = Literal["not_configured", "unknown", "same_model", "different_model", "different_provider"]
JUDGE_SEPARATION_RANK = {
    "not_configured": 0,
    "unknown": 0,
    "same_model": 0,
    "different_model": 1,
    "different_provider": 2,
}


def _trial_client_message_id(
    benchmark_id: str,
    task_id: str,
    trial: int,
    *,
    nonce: str | None = None,
) -> str:
    """Build an opaque idempotency key within the target API's 128-character limit."""
    identity = f"{benchmark_id}\0{task_id}\0{trial}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:24]
    unique = (nonce or uuid4().hex)[:32]
    return f"atl-{digest}-{trial}-{unique}"


class RequiredConcept(FrozenModel):
    criterion_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    aliases: tuple[str, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_aliases(self) -> RequiredConcept:
        normalized = [alias.strip().casefold() for alias in self.aliases]
        if any(not alias for alias in normalized):
            raise ValueError("required concept aliases cannot be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("required concept aliases must be unique")
        return self


class TaskAcceptanceContract(FrozenModel):
    contract_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    min_report_chars: int = Field(default=0, ge=0, le=100_000)
    min_independent_sources: int = Field(default=0, ge=0, le=100)
    min_independent_domains: int = Field(default=0, ge=0, le=100)
    required_concepts: tuple[RequiredConcept, ...] = Field(default=(), max_length=50)
    forbidden_phrases: tuple[str, ...] = Field(default=(), max_length=50)

    @model_validator(mode="after")
    def validate_contract(self) -> TaskAcceptanceContract:
        criterion_ids = [item.criterion_id for item in self.required_concepts]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("acceptance-contract criterion IDs must be unique")
        forbidden = [item.strip().casefold() for item in self.forbidden_phrases]
        if any(not item for item in forbidden):
            raise ValueError("forbidden phrases cannot be blank")
        if len(forbidden) != len(set(forbidden)):
            raise ValueError("forbidden phrases must be unique")
        if self.min_independent_sources and self.min_independent_domains > self.min_independent_sources:
            raise ValueError("minimum independent domains cannot exceed minimum independent sources")
        if not (
            self.min_report_chars
            or self.min_independent_sources
            or self.min_independent_domains
            or self.required_concepts
            or self.forbidden_phrases
        ):
            raise ValueError("acceptance contract must contain at least one executable check")
        return self


def task_acceptance_contract_sha256(contract: TaskAcceptanceContract) -> str:
    serialized = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def benchmark_tasks_sha256(tasks: tuple[DeepResearchBenchmarkTask, ...]) -> str:
    serialized = json.dumps(
        [task.model_dump(mode="json") for task in tasks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class DeepResearchBenchmarkTask(FrozenModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    prompt: str = Field(min_length=1, max_length=20_000)
    source_mode: Literal["web", "graphrag"] = "web"
    workflow_mode: Literal["deep_research", "plan_execute_report"] = "deep_research"
    report_type: Literal["brief", "long_document"] = "brief"
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    expected_status: Literal["completed", "needs_user_input"] = "completed"
    acceptance_contract: TaskAcceptanceContract | None = None

    @model_validator(mode="after")
    def validate_acceptance_contract(self) -> DeepResearchBenchmarkTask:
        if self.expected_status != "completed" and self.acceptance_contract is not None:
            raise ValueError("acceptance_contract is only valid for completed tasks")
        include_domains = tuple(_normalize_configured_domain(item) for item in self.include_domains)
        exclude_domains = tuple(_normalize_configured_domain(item) for item in self.exclude_domains)
        if len(include_domains) != len(set(include_domains)):
            raise ValueError("include_domains must be unique after normalization")
        if len(exclude_domains) != len(set(exclude_domains)):
            raise ValueError("exclude_domains must be unique after normalization")
        if set(include_domains) & set(exclude_domains):
            raise ValueError("the same domain cannot be both included and excluded")
        if self.acceptance_contract is not None and self.acceptance_contract.min_independent_domains:
            minimum = self.acceptance_contract.min_independent_domains
            if not include_domains:
                raise ValueError("minimum independent domains requires explicit include_domains roots")
            if minimum > len(include_domains):
                raise ValueError("minimum independent domains cannot exceed allowed domain roots")
            for index, domain in enumerate(include_domains):
                for other in include_domains[index + 1 :]:
                    if _domain_matches(domain, other) or _domain_matches(other, domain):
                        raise ValueError(
                            "independent-domain roots cannot overlap by parent/subdomain relationship"
                        )
        return self


class BenchmarkDesignEvidence(FrozenModel):
    """Version-bound evidence about whether the benchmark itself is trustworthy."""

    suite_kind: Literal["unspecified", "capability", "regression", "mixed"] = "unspecified"
    dataset_review_status: Literal["unspecified", "unreviewed", "generated-synthetic", "human-reviewed"] = (
        "unspecified"
    )
    task_source: Literal["unspecified", "synthetic", "recorded-production", "expert-authored", "mixed"] = (
        "unspecified"
    )
    reference_solution_coverage: float | None = Field(default=None, ge=0, le=1)
    outcome_grader_coverage: float | None = Field(default=None, ge=0, le=1)
    transcript_review_count: int | None = Field(default=None, ge=0)
    production_case_count: int | None = Field(default=None, ge=0)
    reference_solution_artifact_path: str | None = Field(default=None, min_length=1, max_length=512)
    reference_solution_artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dataset_review_artifact_path: str | None = Field(default=None, min_length=1, max_length=512)
    dataset_review_artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    judge_calibration_id: str | None = Field(default=None, min_length=1, max_length=256)
    judge_calibration_artifact_path: str | None = Field(default=None, min_length=1, max_length=512)
    judge_calibration_artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    environment_isolation: Literal["unspecified", "none", "session", "process", "container", "vm"] = (
        "unspecified"
    )
    environment_fingerprint: str | None = Field(default=None, min_length=1, max_length=512)
    resource_profile: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_artifact_bindings(self) -> BenchmarkDesignEvidence:
        pairs = (
            (
                self.reference_solution_artifact_path,
                self.reference_solution_artifact_sha256,
                "reference solution",
            ),
            (
                self.dataset_review_artifact_path,
                self.dataset_review_artifact_sha256,
                "dataset review",
            ),
            (
                self.judge_calibration_artifact_path,
                self.judge_calibration_artifact_sha256,
                "judge calibration",
            ),
        )
        for path, digest, label in pairs:
            if bool(path) != bool(digest):
                raise ValueError(f"{label} artifact path and SHA-256 must be declared together")
        return self


class DeepResearchBenchmarkManifest(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-benchmark.v1"]
    benchmark_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2_000)
    evidence_class: Literal["live-system", "integration", "synthetic"]
    target_name: str = Field(default="deepresearch-agent-harness", min_length=1, max_length=128)
    target_revision: str | None = Field(default=None, max_length=128)
    target_provider: str | None = Field(default=None, min_length=1, max_length=128)
    target_model: str | None = Field(default=None, min_length=1, max_length=256)
    expected_claim_judge_provider: str | None = Field(default=None, min_length=1, max_length=128)
    expected_claim_judge_model: str | None = Field(default=None, min_length=1, max_length=256)
    expected_claim_judge_prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    minimum_judge_separation: Literal["none", "different_model", "different_provider"] = "none"
    trials_per_task: int = Field(default=3, ge=1, le=20)
    poll_interval_seconds: float = Field(default=1.0, ge=0.05, le=60)
    timeout_seconds: float = Field(default=900, ge=1, le=14_400)
    cancel_on_timeout: bool = True
    cancel_grace_seconds: float = Field(default=30, ge=0.05, le=300)
    design_evidence: BenchmarkDesignEvidence = Field(default_factory=BenchmarkDesignEvidence)
    tasks: tuple[DeepResearchBenchmarkTask, ...] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_unique_tasks(self) -> DeepResearchBenchmarkManifest:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("DeepResearch benchmark task_id values must be unique")
        judge_identity = (
            self.expected_claim_judge_provider,
            self.expected_claim_judge_model,
            self.expected_claim_judge_prompt_version,
        )
        if any(judge_identity) and not all(judge_identity):
            raise ValueError(
                "expected Claim Judge provider, model, and prompt version must be declared together"
            )
        return self


class DeepResearchTrialResult(FrozenModel):
    task_id: str
    trial: int = Field(ge=1)
    session_id: str | None = None
    run_id: str | None = None
    run_status: str
    expected_status: Literal["completed", "needs_user_input"] = "completed"
    trace_id: str | None = None
    completed: bool
    passed: bool
    process_passed: bool | None = None
    citation_gate_passed: bool | None = None
    semantic_gate_passed: bool | None = None
    task_fulfillment_gate_passed: bool | None = None
    task_fulfillment_verdict: str | None = None
    task_missing_requirements: tuple[str, ...] = ()
    task_acceptance_contract_id: str | None = None
    task_acceptance_contract_sha256: str | None = None
    task_acceptance_passed: bool | None = None
    task_acceptance_coverage: float | None = Field(default=None, ge=0, le=1)
    task_acceptance_failures: tuple[str, ...] = ()
    quality_status: Literal["verified_pass", "verified_fail", "unverified"] | None = None
    score: float | None = Field(default=None, ge=0, le=100)
    latency_ms: int | None = Field(default=None, ge=0)
    token_count: int | None = Field(default=None, ge=0)
    tool_call_count: int | None = Field(default=None, ge=0)
    failed_tool_call_count: int | None = Field(default=None, ge=0)
    tool_failure_rate: float | None = Field(default=None, ge=0, le=1)
    recovered_tool_failure_count: int | None = Field(default=None, ge=0)
    unrecovered_tool_failure_count: int | None = Field(default=None, ge=0)
    evidence_count: int | None = Field(default=None, ge=0)
    full_page_evidence_count: int | None = Field(default=None, ge=0)
    search_snippet_evidence_count: int | None = Field(default=None, ge=0)
    full_page_evidence_rate: float | None = Field(default=None, ge=0, le=1)
    page_fetch_attempt_count: int | None = Field(default=None, ge=0)
    page_fetch_failed_count: int | None = Field(default=None, ge=0)
    page_fetch_success_rate: float | None = Field(default=None, ge=0, le=1)
    page_fetch_failure_categories: dict[str, int] = Field(default_factory=dict)
    citation_validity: float | None = Field(default=None, ge=0, le=1)
    replan_count: int | None = Field(default=None, ge=0)
    claim_citation_coverage: float | None = Field(default=None, ge=0, le=1)
    claim_citation_resolution: float | None = Field(default=None, ge=0, le=1)
    claim_proxy_support_rate: float | None = Field(default=None, ge=0, le=1)
    claim_semantic_support_rate: float | None = Field(default=None, ge=0, le=1)
    claim_recommendation: str | None = None
    failure_categories: tuple[str, ...] = ()
    snapshot_path: str | None = None
    trace_path: str | None = None
    evaluation_path: str | None = None
    claim_evidence_path: str | None = None
    claim_judge_error: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class DeepResearchTaskSummary(FrozenModel):
    task_id: str
    trials: int = Field(ge=1)
    completed: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    pass_at_1: bool
    pass_power_k: bool
    mean_score: float | None = Field(default=None, ge=0, le=100)
    mean_latency_ms: float | None = Field(default=None, ge=0)
    failure_categories: dict[str, int] = Field(default_factory=dict)


class DeepResearchBenchmarkMetrics(FrozenModel):
    total_trials: int = Field(ge=1)
    completed_trials: int = Field(ge=0)
    passed_trials: int = Field(ge=0)
    process_passed_trials: int = Field(default=0, ge=0)
    citation_gate_passed_trials: int = Field(default=0, ge=0)
    semantic_evaluated_trials: int = Field(default=0, ge=0)
    semantic_gate_passed_trials: int = Field(default=0, ge=0)
    task_fulfillment_evaluated_trials: int = Field(default=0, ge=0)
    task_fulfillment_passed_trials: int = Field(default=0, ge=0)
    completion_rate: float = Field(ge=0, le=1)
    pass_rate: float = Field(ge=0, le=1)
    process_pass_rate: float = Field(default=0, ge=0, le=1)
    citation_gate_pass_rate: float = Field(default=0, ge=0, le=1)
    semantic_evaluation_coverage: float = Field(default=0, ge=0, le=1)
    semantic_gate_pass_rate: float | None = Field(default=None, ge=0, le=1)
    task_fulfillment_evaluation_coverage: float = Field(default=0, ge=0, le=1)
    task_fulfillment_pass_rate: float | None = Field(default=None, ge=0, le=1)
    task_acceptance_evaluated_trials: int | None = Field(default=None, ge=0)
    task_acceptance_passed_trials: int | None = Field(default=None, ge=0)
    task_acceptance_evaluation_coverage: float | None = Field(default=None, ge=0, le=1)
    task_acceptance_pass_rate: float | None = Field(default=None, ge=0, le=1)
    evidence_verified_passed_trials: int | None = Field(default=None, ge=0)
    evidence_unverified_potential_passes: int | None = Field(default=None, ge=0)
    evidence_supported_pass_rate_lower_bound: float | None = Field(default=None, ge=0, le=1)
    evidence_supported_pass_rate_upper_bound: float | None = Field(default=None, ge=0, le=1)
    pass_rate_wilson_lower_bound_95: float = Field(ge=0, le=1)
    pass_at_1_rate: float = Field(ge=0, le=1)
    pass_power_k_rate: float = Field(ge=0, le=1)
    mean_score: float | None = Field(default=None, ge=0, le=100)
    mean_latency_ms: float | None = Field(default=None, ge=0)
    p50_latency_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    mean_token_count: float | None = Field(default=None, ge=0)
    mean_tool_call_count: float | None = Field(default=None, ge=0)
    mean_failed_tool_call_count: float | None = Field(default=None, ge=0)
    mean_tool_failure_rate: float | None = Field(default=None, ge=0, le=1)
    tool_failure_free_rate: float | None = Field(default=None, ge=0, le=1)
    recovered_tool_failure_count: int = Field(default=0, ge=0)
    unrecovered_tool_failure_count: int = Field(default=0, ge=0)
    full_page_evidence_count: int = Field(default=0, ge=0)
    search_snippet_evidence_count: int = Field(default=0, ge=0)
    mean_full_page_evidence_rate: float | None = Field(default=None, ge=0, le=1)
    page_fetch_attempt_count: int = Field(default=0, ge=0)
    page_fetch_failed_count: int = Field(default=0, ge=0)
    mean_page_fetch_success_rate: float | None = Field(default=None, ge=0, le=1)
    page_fetch_failure_categories: dict[str, int] = Field(default_factory=dict)
    mean_citation_validity: float | None = Field(default=None, ge=0, le=1)
    mean_claim_citation_coverage: float | None = Field(default=None, ge=0, le=1)
    mean_claim_citation_resolution: float | None = Field(default=None, ge=0, le=1)
    mean_claim_proxy_support_rate: float | None = Field(default=None, ge=0, le=1)
    mean_claim_semantic_support_rate: float | None = Field(default=None, ge=0, le=1)
    failure_categories: dict[str, int] = Field(default_factory=dict)


class DeepResearchBenchmarkReport(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-benchmark-report.v1"] = (
        "agenttracelab.deepresearch-benchmark-report.v1"
    )
    benchmark_id: str
    benchmark_version: str
    manifest_sha256: str
    evidence_class: Literal["live-system", "integration", "synthetic"]
    target_name: str
    target_revision: str | None = None
    target_provider: str | None = None
    target_model: str | None = None
    claim_judge_provider: str | None = None
    claim_judge_model: str | None = None
    claim_judge_prompt_version: str | None = None
    judge_separation: JudgeSeparation = "not_configured"
    target_base_url: str
    source_database_file: str
    created_at: datetime
    finished_at: datetime
    status: Literal["completed", "partial", "failed"]
    task_count: int = Field(ge=1)
    trials_per_task: int = Field(ge=1)
    metrics: DeepResearchBenchmarkMetrics
    task_summaries: tuple[DeepResearchTaskSummary, ...]
    trials: tuple[DeepResearchTrialResult, ...]
    limitations: tuple[str, ...]


SnapshotExporter = Callable[[str | Path, str], dict[str, Any]]
TraceEvaluator = Callable[[dict[str, Any]], tuple[TraceEnvelope, EvaluationReport]]


def load_deepresearch_benchmark_manifest(
    path: str | Path,
) -> tuple[DeepResearchBenchmarkManifest, Path, str]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"DeepResearch benchmark manifest does not exist: {manifest_path}")
    if manifest_path.stat().st_size > 1_048_576:
        raise ValueError("DeepResearch benchmark manifest exceeds the 1 MiB limit")
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("DeepResearch benchmark manifest must contain a JSON object")
    return (
        DeepResearchBenchmarkManifest.model_validate(payload),
        manifest_path,
        hashlib.sha256(raw).hexdigest(),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(f"{rendered}\n", encoding="utf-8")
    temporary.replace(path)


def _mean(values: list[float | int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def _token_count(usage: dict[str, Any]) -> int | None:
    nested = usage.get("usage") if isinstance(usage.get("usage"), dict) else usage
    for key in ("total_tokens", "llm_tokens", "token_count", "tokens"):
        value = nested.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    prompt = nested.get("prompt_tokens")
    completion = nested.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return max(0, prompt) + max(0, completion)
    return None


def _evaluate_snapshot(snapshot: dict[str, Any]) -> tuple[TraceEnvelope, EvaluationReport]:
    trace = adapt_deepresearch_snapshot(snapshot)
    return trace, evaluate_trace(trace)


def _judge_separation(
    manifest: DeepResearchBenchmarkManifest,
    judge: ClaimSupportJudge | None,
) -> JudgeSeparation:
    if judge is None:
        return "not_configured"
    return _judge_separation_for_identity(
        target_provider=manifest.target_provider,
        target_model=manifest.target_model,
        judge_provider=judge.provider,
        judge_model=judge.model,
    )


def _judge_separation_for_identity(
    *,
    target_provider: str | None,
    target_model: str | None,
    judge_provider: str | None,
    judge_model: str | None,
) -> JudgeSeparation:
    if not judge_provider or not judge_model:
        return "not_configured"
    if not target_provider or not target_model:
        return "unknown"
    normalized_target_provider = target_provider.casefold()
    normalized_target_model = target_model.casefold()
    normalized_judge_provider = judge_provider.casefold()
    normalized_judge_model = judge_model.casefold()
    if normalized_target_provider != normalized_judge_provider:
        return "different_provider"
    if normalized_target_model != normalized_judge_model:
        return "different_model"
    return "same_model"


def _evaluate_task_acceptance(
    task: DeepResearchBenchmarkTask,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    contract = task.acceptance_contract
    if contract is None:
        return {
            "contract_id": None,
            "contract_sha256": None,
            "passed": None,
            "coverage": None,
            "failures": (),
        }

    report = str(snapshot.get("report") or "").strip()
    normalized_report = " ".join(report.casefold().split())
    checks: list[tuple[str, bool]] = []
    if contract.min_report_chars:
        checks.append(("report.min_chars", len(report) >= contract.min_report_chars))
    for concept in contract.required_concepts:
        matched = any(
            " ".join(alias.strip().casefold().split()) in normalized_report for alias in concept.aliases
        )
        checks.append((f"concept.{concept.criterion_id}", matched))

    evidence = tuple(item for item in snapshot.get("evidence") or () if isinstance(item, dict))
    source_constraints_enabled = bool(
        contract.min_independent_sources
        or contract.min_independent_domains
        or task.include_domains
        or task.exclude_domains
    )
    parsed_sources = tuple(_parse_evidence_source(item) for item in evidence)
    valid_sources = tuple(item for item in parsed_sources if item is not None)
    if source_constraints_enabled:
        checks.append(("evidence.valid_urls", len(valid_sources) == len(evidence)))
    if contract.min_independent_sources:
        checks.append(
            (
                "evidence.independent_sources",
                len({canonical_url for canonical_url, _hostname in valid_sources})
                >= contract.min_independent_sources,
            )
        )
    if contract.min_independent_domains:
        checks.append(
            (
                "evidence.independent_domains",
                len(
                    {
                        _source_domain_group(hostname, task.include_domains)
                        for _canonical_url, hostname in valid_sources
                    }
                )
                >= contract.min_independent_domains,
            )
        )
    if task.include_domains:
        checks.append(
            (
                "evidence.include_domains",
                all(
                    any(_domain_matches(hostname, domain) for domain in task.include_domains)
                    for _canonical_url, hostname in valid_sources
                )
                and len(valid_sources) == len(evidence),
            )
        )
    if task.exclude_domains:
        checks.append(
            (
                "evidence.exclude_domains",
                all(
                    not any(_domain_matches(hostname, domain) for domain in task.exclude_domains)
                    for _canonical_url, hostname in valid_sources
                )
                and len(valid_sources) == len(evidence),
            )
        )
    for index, phrase in enumerate(contract.forbidden_phrases, start=1):
        normalized_phrase = " ".join(phrase.strip().casefold().split())
        checks.append((f"forbidden.{index:02d}", normalized_phrase not in normalized_report))

    failures = tuple(check_id for check_id, passed in checks if not passed)
    return {
        "contract_id": contract.contract_id,
        "contract_sha256": task_acceptance_contract_sha256(contract),
        "passed": not failures,
        "coverage": round((len(checks) - len(failures)) / len(checks), 4),
        "failures": failures,
    }


def _parse_evidence_source(item: dict[str, Any]) -> tuple[str, str] | None:
    raw_url = str(item.get("url") or "").strip()
    if not raw_url:
        return None
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in hostname)
    ):
        return None

    normalized_port = ""
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        normalized_port = f":{port}"
    normalized_path = parsed.path or "/"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/") or "/"
    canonical_url = urlunsplit(
        (scheme, f"{hostname}{normalized_port}", normalized_path, "", "")
    )
    return canonical_url, hostname


def _domain_matches(hostname: str, configured_domain: str) -> bool:
    domain = configured_domain.strip().casefold().rstrip(".")
    return bool(domain) and (hostname == domain or hostname.endswith(f".{domain}"))


def _normalize_configured_domain(value: str) -> str:
    domain = value.strip().casefold().rstrip(".")
    if not domain or len(domain) > 253 or "://" in domain or any(
        character in domain for character in "/?#@:*"
    ):
        raise ValueError("source domains must be plain DNS hostnames without URL syntax or wildcards")
    labels = domain.split(".")
    if any(
        not label
        or len(label) > 63
        or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) is None
        for label in labels
    ):
        raise ValueError("source domains must contain valid DNS labels")
    return domain


def _source_domain_group(hostname: str, include_domains: tuple[str, ...]) -> str:
    matching_domains = {
        domain.strip().casefold().rstrip(".")
        for domain in include_domains
        if _domain_matches(hostname, domain)
    }
    if not matching_domains:
        return hostname
    return max(matching_domains, key=len)


def _quality_gate_outcome(
    status: str,
    evaluation: EvaluationReport,
    claim_report: ClaimEvidenceReport,
    task_acceptance_passed: bool | None,
) -> tuple[bool, bool, bool | None, str, tuple[str, ...], bool]:
    process_passed = status == "completed" and evaluation.passed
    citation_gate_passed = claim_report.citation_gate_passed
    semantic_gate_passed = claim_report.semantic_gate_passed
    task_fulfillment_gate_passed = claim_report.task_fulfillment_gate_passed
    task_fulfillment_satisfied = (
        not claim_report.task_fulfillment_required or task_fulfillment_gate_passed is True
    )
    quality_status = "unverified"
    if (
        semantic_gate_passed is True
        and task_acceptance_passed is True
        and task_fulfillment_satisfied
    ):
        quality_status = "verified_pass"
    elif (
        semantic_gate_passed is False
        or task_acceptance_passed is False
        or (
            claim_report.task_fulfillment_required
            and task_fulfillment_gate_passed is False
        )
    ):
        quality_status = "verified_fail"
    claim_failures: tuple[str, ...] = ()
    if not citation_gate_passed:
        claim_failures += ("deepresearch.claim_citation_gate",)
    if semantic_gate_passed is False:
        claim_failures += ("deepresearch.claim_semantic_gate",)
    if task_acceptance_passed is False:
        claim_failures += ("deepresearch.task_acceptance_contract",)
    if claim_report.task_fulfillment_required and task_fulfillment_gate_passed is False:
        claim_failures += ("deepresearch.task_fulfillment_gate",)
    failure_categories = tuple(dict.fromkeys((*evaluation.failure_categories, *claim_failures)))
    passed = (
        process_passed
        and citation_gate_passed
        and semantic_gate_passed is not False
        and task_acceptance_passed is not False
        and (
            not claim_report.task_fulfillment_required
            or task_fulfillment_gate_passed is not False
        )
    )
    return (
        process_passed,
        citation_gate_passed,
        semantic_gate_passed,
        quality_status,
        failure_categories,
        passed,
    )


def _score_expected_outcome(
    *,
    expected_status: Literal["completed", "needs_user_input"],
    status: str,
    evaluation: EvaluationReport,
    claim_report: ClaimEvidenceReport,
    task_acceptance_passed: bool | None,
    tool_call_count: int,
    evidence_count: int,
) -> dict[str, Any]:
    if expected_status == "completed":
        (
            process_passed,
            citation_gate_passed,
            semantic_gate_passed,
            quality_status,
            failure_categories,
            passed,
        ) = _quality_gate_outcome(
            status,
            evaluation,
            claim_report,
            task_acceptance_passed,
        )
        return {
            "completed": status == "completed",
            "passed": passed,
            "process_passed": process_passed,
            "citation_gate_passed": citation_gate_passed,
            "semantic_gate_passed": semantic_gate_passed,
            "task_fulfillment_gate_passed": claim_report.task_fulfillment_gate_passed,
            "task_fulfillment_verdict": (
                claim_report.task_fulfillment_verdict.value
                if claim_report.task_fulfillment_verdict is not None
                else None
            ),
            "task_missing_requirements": claim_report.task_missing_requirements,
            "quality_status": quality_status,
            "score": evaluation.score,
            "failure_categories": failure_categories,
            "citation_validity": None,
            "claim_citation_coverage": claim_report.metrics.citation_coverage,
            "claim_citation_resolution": claim_report.metrics.citation_resolution,
            "claim_proxy_support_rate": claim_report.metrics.proxy_support_rate,
            "claim_semantic_support_rate": claim_report.metrics.semantic_support_rate,
            "claim_recommendation": claim_report.recommendation,
        }

    failures: list[str] = []
    if status != "needs_user_input":
        failures.append("deepresearch.expected_needs_user_input")
    if tool_call_count:
        failures.append("deepresearch.unexpected_tool_call")
    if evidence_count:
        failures.append("deepresearch.unexpected_evidence")
    passed = not failures
    return {
        "completed": status == "needs_user_input",
        "passed": passed,
        "process_passed": passed,
        "citation_gate_passed": None,
        "semantic_gate_passed": None,
        "task_fulfillment_gate_passed": None,
        "task_fulfillment_verdict": None,
        "task_missing_requirements": (),
        "quality_status": None,
        "score": None,
        "failure_categories": tuple(failures),
        "citation_validity": None,
        "claim_citation_coverage": None,
        "claim_citation_resolution": None,
        "claim_proxy_support_rate": None,
        "claim_semantic_support_rate": None,
        "claim_recommendation": "correct clarification without retrieval" if passed else None,
    }


class DeepResearchBenchmarkRunner:
    def __init__(
        self,
        *,
        base_url: str,
        source_database: str | Path,
        output_directory: str | Path,
        client: httpx.Client | None = None,
        bearer_token: str | None = None,
        snapshot_exporter: SnapshotExporter = export_deepresearch_sqlite_snapshot,
        trace_evaluator: TraceEvaluator = _evaluate_snapshot,
        claim_judge: ClaimSupportJudge | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.source_database = Path(source_database).resolve()
        self.output_directory = Path(output_directory).resolve()
        self.snapshot_exporter = snapshot_exporter
        self.trace_evaluator = trace_evaluator
        self.claim_judge = claim_judge
        self.sleep = sleep
        self.monotonic = monotonic
        headers = {"Accept": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._owns_client = client is None
        self.client = client or httpx.Client(base_url=self.base_url, headers=headers, timeout=30)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> DeepResearchBenchmarkRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"DeepResearch endpoint returned a non-object payload: {path}")
        return payload

    def _poll_run(
        self,
        run_id: str,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float,
        cancel_on_timeout: bool,
        cancel_grace_seconds: float,
    ) -> dict[str, Any]:
        deadline = self.monotonic() + timeout_seconds
        while True:
            payload = self._request_json("GET", f"/api/v1/runs/{run_id}")
            status = str(payload.get("status") or "unknown")
            if status in TERMINAL_RUN_STATUSES:
                return payload
            if self.monotonic() >= deadline:
                if cancel_on_timeout:
                    with suppress(httpx.HTTPError, ValueError):
                        self._request_json("POST", f"/api/v1/runs/{run_id}/cancel")
                        cancel_deadline = self.monotonic() + cancel_grace_seconds
                        while True:
                            cancelled = self._request_json("GET", f"/api/v1/runs/{run_id}")
                            if str(cancelled.get("status") or "unknown") in TERMINAL_RUN_STATUSES:
                                break
                            if self.monotonic() >= cancel_deadline:
                                break
                            self.sleep(poll_interval_seconds)
                raise TimeoutError(f"DeepResearch run {run_id} exceeded {timeout_seconds:g}s")
            self.sleep(poll_interval_seconds)

    def _capture_failed_run(
        self,
        *,
        task: DeepResearchBenchmarkTask,
        trial: int,
        trial_directory: Path,
        session_id: str | None,
        run_id: str,
        exc: Exception,
    ) -> DeepResearchTrialResult:
        snapshot = self.snapshot_exporter(self.source_database, run_id)
        trace, evaluation = self.trace_evaluator(snapshot)
        snapshot_path = trial_directory / "run-snapshot.json"
        trace_path = trial_directory / "normalized-trace.json"
        evaluation_path = trial_directory / "evaluation-report.json"
        _write_json(snapshot_path, snapshot)
        _write_json(trace_path, trace.model_dump(mode="json"))
        _write_json(evaluation_path, evaluation.model_dump(mode="json"))

        run_data = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
        citations = trace.metadata.get("deepresearch.citations") or {}
        latency_ms = None
        if run_data.get("started_at") and run_data.get("completed_at"):
            start_dt = datetime.fromisoformat(str(run_data["started_at"]).replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(str(run_data["completed_at"]).replace("Z", "+00:00"))
            latency_ms = max(0, round((end_dt - start_dt).total_seconds() * 1_000))
        tool_call_count = int(trace.metadata.get("deepresearch.tool_call_count", 0) or 0)
        failed_tool_call_count = int(trace.metadata.get("deepresearch.failed_tool_call_count", 0) or 0)
        failure_categories = tuple(
            dict.fromkeys(("benchmark.execution_error", *evaluation.failure_categories))
        )
        return DeepResearchTrialResult(
            task_id=task.task_id,
            trial=trial,
            session_id=session_id,
            run_id=run_id,
            run_status=str(run_data.get("status") or "runner_error"),
            expected_status=task.expected_status,
            trace_id=trace.trace_id,
            completed=False,
            passed=False,
            process_passed=False,
            quality_status="unverified",
            latency_ms=latency_ms,
            token_count=_token_count(run_data.get("usage") or {}),
            tool_call_count=tool_call_count,
            failed_tool_call_count=failed_tool_call_count,
            tool_failure_rate=(
                round(failed_tool_call_count / tool_call_count, 4) if tool_call_count else 0.0
            ),
            recovered_tool_failure_count=int(
                trace.metadata.get("deepresearch.recovered_tool_failure_count", 0) or 0
            ),
            unrecovered_tool_failure_count=int(
                trace.metadata.get("deepresearch.unrecovered_tool_failure_count", 0) or 0
            ),
            evidence_count=len(snapshot.get("evidence") or ()),
            full_page_evidence_count=int(trace.metadata.get("deepresearch.full_page_evidence_count", 0) or 0),
            search_snippet_evidence_count=int(
                trace.metadata.get("deepresearch.search_snippet_evidence_count", 0) or 0
            ),
            full_page_evidence_rate=trace.metadata.get("deepresearch.full_page_evidence_rate"),
            page_fetch_attempt_count=int(trace.metadata.get("deepresearch.page_fetch_attempt_count", 0) or 0),
            page_fetch_failed_count=int(trace.metadata.get("deepresearch.page_fetch_failed_count", 0) or 0),
            page_fetch_success_rate=trace.metadata.get("deepresearch.page_fetch_success_rate"),
            page_fetch_failure_categories=dict(
                trace.metadata.get("deepresearch.page_fetch_failure_categories") or {}
            ),
            citation_validity=citations.get("validity"),
            replan_count=sum(
                event.get("event_type") == "plan.replanning"
                for event in snapshot.get("events") or ()
                if isinstance(event, dict)
            ),
            failure_categories=failure_categories,
            snapshot_path=snapshot_path.relative_to(self.output_directory).as_posix(),
            trace_path=trace_path.relative_to(self.output_directory).as_posix(),
            evaluation_path=evaluation_path.relative_to(self.output_directory).as_posix(),
            error_type=type(exc).__name__,
            error_message=str(exc)[:1_000],
        )

    def _run_trial(
        self,
        task: DeepResearchBenchmarkTask,
        trial: int,
        manifest: DeepResearchBenchmarkManifest,
    ) -> DeepResearchTrialResult:
        trial_directory = self.output_directory / "trials" / task.task_id / f"trial-{trial:02d}"
        session_id: str | None = None
        run_id: str | None = None
        started = self.monotonic()
        try:
            session = self._request_json(
                "POST",
                "/api/v1/sessions",
                json={"title": f"benchmark:{manifest.benchmark_id}:{task.task_id}:{trial}"},
            )
            session_id = str(session["session_id"])
            accepted = self._request_json(
                "POST",
                f"/api/v1/sessions/{session_id}/messages",
                json={
                    "client_message_id": _trial_client_message_id(
                        manifest.benchmark_id,
                        task.task_id,
                        trial,
                    ),
                    "content": task.prompt,
                    "source_mode": task.source_mode,
                    "workflow_mode": task.workflow_mode,
                    "report_type": task.report_type,
                    "include_domains": list(task.include_domains),
                    "exclude_domains": list(task.exclude_domains),
                    "evaluation_run": True,
                },
            )
            run_id = str(accepted["run_id"])
            run = self._poll_run(
                run_id,
                poll_interval_seconds=manifest.poll_interval_seconds,
                timeout_seconds=manifest.timeout_seconds,
                cancel_on_timeout=manifest.cancel_on_timeout,
                cancel_grace_seconds=manifest.cancel_grace_seconds,
            )
            snapshot = self.snapshot_exporter(self.source_database, run_id)
            trace, evaluation = self.trace_evaluator(snapshot)
            claim_judge_error = None
            try:
                claim_report = verify_deepresearch_claims(
                    snapshot,
                    judge=self.claim_judge,
                    task_prompt=(task.prompt if task.expected_status == "completed" else None),
                )
            except ClaimJudgeError as exc:
                claim_judge_error = str(exc)[:1_000]
                claim_report = verify_deepresearch_claims(
                    snapshot,
                    task_prompt=(task.prompt if task.expected_status == "completed" else None),
                )
            snapshot_path = trial_directory / "run-snapshot.json"
            trace_path = trial_directory / "normalized-trace.json"
            evaluation_path = trial_directory / "evaluation-report.json"
            claim_path = trial_directory / "claim-evidence-report.json"
            _write_json(snapshot_path, snapshot)
            _write_json(trace_path, trace.model_dump(mode="json"))
            _write_json(evaluation_path, evaluation.model_dump(mode="json"))
            _write_json(claim_path, claim_report.model_dump(mode="json"))

            run_data = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
            status = str(run_data.get("status") or run.get("status") or "unknown")
            started_at = run_data.get("started_at")
            completed_at = run_data.get("completed_at")
            latency_ms = round((self.monotonic() - started) * 1_000)
            if started_at and completed_at:
                start_dt = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
                latency_ms = max(0, round((end_dt - start_dt).total_seconds() * 1_000))
            citations = trace.metadata.get("deepresearch.citations") or {}
            tool_call_count = int(trace.metadata.get("deepresearch.tool_call_count", 0) or 0)
            failed_tool_call_count = int(trace.metadata.get("deepresearch.failed_tool_call_count", 0) or 0)
            evidence_count = len(snapshot.get("evidence") or ())
            acceptance = _evaluate_task_acceptance(task, snapshot)
            outcome = _score_expected_outcome(
                expected_status=task.expected_status,
                status=status,
                evaluation=evaluation,
                claim_report=claim_report,
                task_acceptance_passed=acceptance["passed"],
                tool_call_count=tool_call_count,
                evidence_count=evidence_count,
            )
            target_run_failed = status in {"failed", "cancelled"}
            failure_categories = tuple(
                dict.fromkeys(
                    (
                        *outcome["failure_categories"],
                        *(("benchmark.target_run_failed",) if target_run_failed else ()),
                    )
                )
            )
            return DeepResearchTrialResult(
                task_id=task.task_id,
                trial=trial,
                session_id=session_id,
                run_id=run_id,
                run_status=status,
                expected_status=task.expected_status,
                trace_id=trace.trace_id,
                completed=outcome["completed"],
                passed=outcome["passed"],
                process_passed=outcome["process_passed"],
                citation_gate_passed=outcome["citation_gate_passed"],
                semantic_gate_passed=outcome["semantic_gate_passed"],
                task_fulfillment_gate_passed=outcome["task_fulfillment_gate_passed"],
                task_fulfillment_verdict=outcome["task_fulfillment_verdict"],
                task_missing_requirements=outcome["task_missing_requirements"],
                task_acceptance_contract_id=acceptance["contract_id"],
                task_acceptance_contract_sha256=acceptance["contract_sha256"],
                task_acceptance_passed=acceptance["passed"],
                task_acceptance_coverage=acceptance["coverage"],
                task_acceptance_failures=acceptance["failures"],
                quality_status=outcome["quality_status"],
                score=outcome["score"],
                latency_ms=latency_ms,
                token_count=_token_count(run_data.get("usage") or {}),
                tool_call_count=tool_call_count,
                failed_tool_call_count=failed_tool_call_count,
                tool_failure_rate=(
                    round(failed_tool_call_count / tool_call_count, 4) if tool_call_count else 0.0
                ),
                recovered_tool_failure_count=int(
                    trace.metadata.get("deepresearch.recovered_tool_failure_count", 0) or 0
                ),
                unrecovered_tool_failure_count=int(
                    trace.metadata.get("deepresearch.unrecovered_tool_failure_count", 0) or 0
                ),
                evidence_count=evidence_count,
                full_page_evidence_count=int(
                    trace.metadata.get("deepresearch.full_page_evidence_count", 0) or 0
                ),
                search_snippet_evidence_count=int(
                    trace.metadata.get("deepresearch.search_snippet_evidence_count", 0) or 0
                ),
                full_page_evidence_rate=trace.metadata.get("deepresearch.full_page_evidence_rate"),
                page_fetch_attempt_count=int(
                    trace.metadata.get("deepresearch.page_fetch_attempt_count", 0) or 0
                ),
                page_fetch_failed_count=int(
                    trace.metadata.get("deepresearch.page_fetch_failed_count", 0) or 0
                ),
                page_fetch_success_rate=trace.metadata.get("deepresearch.page_fetch_success_rate"),
                page_fetch_failure_categories=dict(
                    trace.metadata.get("deepresearch.page_fetch_failure_categories") or {}
                ),
                citation_validity=(
                    citations.get("validity")
                    if task.expected_status == "completed"
                    else outcome["citation_validity"]
                ),
                replan_count=sum(
                    event.get("event_type") == "plan.replanning"
                    for event in snapshot.get("events") or ()
                    if isinstance(event, dict)
                ),
                claim_citation_coverage=outcome["claim_citation_coverage"],
                claim_citation_resolution=outcome["claim_citation_resolution"],
                claim_proxy_support_rate=outcome["claim_proxy_support_rate"],
                claim_semantic_support_rate=outcome["claim_semantic_support_rate"],
                claim_recommendation=outcome["claim_recommendation"],
                failure_categories=failure_categories,
                snapshot_path=snapshot_path.relative_to(self.output_directory).as_posix(),
                trace_path=trace_path.relative_to(self.output_directory).as_posix(),
                evaluation_path=evaluation_path.relative_to(self.output_directory).as_posix(),
                claim_evidence_path=claim_path.relative_to(self.output_directory).as_posix(),
                claim_judge_error=claim_judge_error,
                error_type=(
                    str(run_data.get("error_code") or "TargetRunFailed")
                    if target_run_failed
                    else None
                ),
                error_message=(
                    str(run_data.get("error_message") or f"target run ended with status {status}")[:1_000]
                    if target_run_failed
                    else None
                ),
            )
        except (httpx.HTTPError, KeyError, OSError, TimeoutError, ValueError) as exc:
            if run_id is not None:
                try:
                    return self._capture_failed_run(
                        task=task,
                        trial=trial,
                        trial_directory=trial_directory,
                        session_id=session_id,
                        run_id=run_id,
                        exc=exc,
                    )
                # Diagnostic capture is best-effort and must never replace the
                # original benchmark failure with a secondary exporter error.
                except Exception as capture_exc:
                    error_message = (
                        f"{str(exc)[:700]} | diagnostic capture failed: "
                        f"{type(capture_exc).__name__}: {str(capture_exc)[:200]}"
                    )
                else:  # pragma: no cover - the successful path returns above
                    error_message = str(exc)[:1_000]
            else:
                error_message = str(exc)[:1_000]
            return DeepResearchTrialResult(
                task_id=task.task_id,
                trial=trial,
                session_id=session_id,
                run_id=run_id,
                run_status="runner_error",
                expected_status=task.expected_status,
                completed=False,
                passed=False,
                failure_categories=("benchmark.execution_error",),
                error_type=type(exc).__name__,
                error_message=error_message,
            )

    def run(self, manifest_path: str | Path) -> DeepResearchBenchmarkReport:
        manifest, _, manifest_sha256 = load_deepresearch_benchmark_manifest(manifest_path)
        expected_judge_identity = (
            manifest.expected_claim_judge_provider,
            manifest.expected_claim_judge_model,
            manifest.expected_claim_judge_prompt_version,
        )
        actual_judge_identity = (
            self.claim_judge.provider if self.claim_judge else None,
            self.claim_judge.model if self.claim_judge else None,
            CLAIM_JUDGE_PROMPT_VERSION if self.claim_judge else None,
        )
        if any(expected_judge_identity) and expected_judge_identity != actual_judge_identity:
            raise ValueError(
                "configured Claim Judge identity does not match the manifest: "
                f"expected={expected_judge_identity}, observed={actual_judge_identity}"
            )
        judge_separation = _judge_separation(manifest, self.claim_judge)
        required_rank = {
            "none": 0,
            "different_model": 1,
            "different_provider": 2,
        }[manifest.minimum_judge_separation]
        if JUDGE_SEPARATION_RANK[judge_separation] < required_rank:
            raise ValueError(
                "claim Judge separation policy is not satisfied: "
                f"required={manifest.minimum_judge_separation}, observed={judge_separation}"
            )
        report_path = self.output_directory / "benchmark-report.json"
        if report_path.exists():
            raise ValueError(
                "benchmark output already contains benchmark-report.json; "
                "use a new immutable output directory"
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC)
        trials = tuple(
            self._run_trial(task, trial, manifest)
            for task in manifest.tasks
            for trial in range(1, manifest.trials_per_task + 1)
        )
        task_summaries = tuple(_summarize_task(task.task_id, trials) for task in manifest.tasks)
        metrics = _summarize_benchmark(trials, task_summaries)
        if metrics.completed_trials == metrics.total_trials:
            status = "completed"
        elif metrics.completed_trials:
            status = "partial"
        else:
            status = "failed"
        limitations = [
            "Claim lexical overlap is only a retrieval-support proxy; semantic support requires a Judge "
            "or human review.",
            "Pass^k is reported as all repeated trials passing; "
            "it measures reliability, not best-of-k success.",
        ]
        if manifest.evidence_class != "live-system":
            limitations.append(
                f"Evidence class is {manifest.evidence_class}; "
                "results are not production model-quality claims."
            )
        if judge_separation in {"not_configured", "unknown", "same_model"}:
            limitations.append(
                f"Claim Judge separation is {judge_separation}; semantic ratings must not be "
                "described as independent."
            )
        report = DeepResearchBenchmarkReport(
            benchmark_id=manifest.benchmark_id,
            benchmark_version=manifest.version,
            manifest_sha256=manifest_sha256,
            evidence_class=manifest.evidence_class,
            target_name=manifest.target_name,
            target_revision=manifest.target_revision,
            target_provider=manifest.target_provider,
            target_model=manifest.target_model,
            claim_judge_provider=self.claim_judge.provider if self.claim_judge else None,
            claim_judge_model=self.claim_judge.model if self.claim_judge else None,
            claim_judge_prompt_version=(CLAIM_JUDGE_PROMPT_VERSION if self.claim_judge else None),
            judge_separation=judge_separation,
            target_base_url=self.base_url,
            source_database_file=self.source_database.name,
            created_at=created_at,
            finished_at=datetime.now(UTC),
            status=status,
            task_count=len(manifest.tasks),
            trials_per_task=manifest.trials_per_task,
            metrics=metrics,
            task_summaries=task_summaries,
            trials=trials,
            limitations=tuple(limitations),
        )
        _write_json(report_path, report.model_dump(mode="json"))
        (self.output_directory / "benchmark-summary.md").write_text(
            render_deepresearch_benchmark_markdown(report), encoding="utf-8"
        )
        return report


def _summarize_task(task_id: str, trials: tuple[DeepResearchTrialResult, ...]) -> DeepResearchTaskSummary:
    selected = sorted((item for item in trials if item.task_id == task_id), key=lambda item: item.trial)
    passed = sum(item.passed for item in selected)
    scores = [item.score for item in selected if item.score is not None]
    latencies = [item.latency_ms for item in selected if item.latency_ms is not None]
    failures = Counter(category for item in selected for category in item.failure_categories)
    return DeepResearchTaskSummary(
        task_id=task_id,
        trials=len(selected),
        completed=sum(item.completed for item in selected),
        passed=passed,
        pass_rate=round(passed / len(selected), 4),
        pass_at_1=selected[0].passed,
        pass_power_k=passed == len(selected),
        mean_score=_mean(scores),
        mean_latency_ms=_mean(latencies),
        failure_categories=dict(sorted(failures.items())),
    )


def _summarize_benchmark(
    trials: tuple[DeepResearchTrialResult, ...],
    tasks: tuple[DeepResearchTaskSummary, ...],
) -> DeepResearchBenchmarkMetrics:
    total = len(trials)
    passed = sum(item.passed for item in trials)
    process_passed = sum(item.process_passed is True for item in trials)
    citation_evaluated = sum(item.citation_gate_passed is not None for item in trials)
    citation_passed = sum(item.citation_gate_passed is True for item in trials)
    semantic_evaluated = sum(item.semantic_gate_passed is not None for item in trials)
    semantic_passed = sum(item.semantic_gate_passed is True for item in trials)
    answerable_trials = sum(item.expected_status == "completed" for item in trials)
    task_fulfillment_evaluated = sum(
        item.task_fulfillment_gate_passed is not None for item in trials
    )
    task_fulfillment_passed = sum(item.task_fulfillment_gate_passed is True for item in trials)
    acceptance_evaluated = sum(item.task_acceptance_passed is not None for item in trials)
    acceptance_passed = sum(item.task_acceptance_passed is True for item in trials)
    evidence_verified_passed = sum(
        item.passed and (item.expected_status == "needs_user_input" or item.quality_status == "verified_pass")
        for item in trials
    )
    evidence_unverified_potential_passes = sum(
        item.passed and item.expected_status == "completed" and item.quality_status == "unverified"
        for item in trials
    )
    scores = [item.score for item in trials if item.score is not None]
    latencies = [item.latency_ms for item in trials if item.latency_ms is not None]
    tokens = [item.token_count for item in trials if item.token_count is not None]
    tools = [item.tool_call_count for item in trials if item.tool_call_count is not None]
    failed_tools = [item.failed_tool_call_count for item in trials if item.failed_tool_call_count is not None]
    tool_failure_rates = [item.tool_failure_rate for item in trials if item.tool_failure_rate is not None]
    citations = [item.citation_validity for item in trials if item.citation_validity is not None]
    claim_coverage = [
        item.claim_citation_coverage for item in trials if item.claim_citation_coverage is not None
    ]
    claim_resolution = [
        item.claim_citation_resolution for item in trials if item.claim_citation_resolution is not None
    ]
    claim_proxy = [
        item.claim_proxy_support_rate for item in trials if item.claim_proxy_support_rate is not None
    ]
    claim_semantic = [
        item.claim_semantic_support_rate for item in trials if item.claim_semantic_support_rate is not None
    ]
    full_page_rates = [
        item.full_page_evidence_rate for item in trials if item.full_page_evidence_rate is not None
    ]
    page_fetch_success_rates = [
        item.page_fetch_success_rate for item in trials if item.page_fetch_success_rate is not None
    ]
    page_fetch_failures = Counter(
        {
            category: sum(item.page_fetch_failure_categories.get(category, 0) for item in trials)
            for category in {category for item in trials for category in item.page_fetch_failure_categories}
        }
    )
    failures = Counter(category for item in trials for category in item.failure_categories)
    return DeepResearchBenchmarkMetrics(
        total_trials=total,
        completed_trials=sum(item.completed for item in trials),
        passed_trials=passed,
        process_passed_trials=process_passed,
        citation_gate_passed_trials=citation_passed,
        semantic_evaluated_trials=semantic_evaluated,
        semantic_gate_passed_trials=semantic_passed,
        task_fulfillment_evaluated_trials=task_fulfillment_evaluated,
        task_fulfillment_passed_trials=task_fulfillment_passed,
        completion_rate=round(sum(item.completed for item in trials) / total, 4),
        pass_rate=round(passed / total, 4),
        process_pass_rate=round(process_passed / total, 4),
        citation_gate_pass_rate=(
            round(citation_passed / citation_evaluated, 4) if citation_evaluated else 0.0
        ),
        semantic_evaluation_coverage=(
            round(semantic_evaluated / answerable_trials, 4) if answerable_trials else 1.0
        ),
        semantic_gate_pass_rate=(
            round(semantic_passed / semantic_evaluated, 4) if semantic_evaluated else None
        ),
        task_fulfillment_evaluation_coverage=(
            round(task_fulfillment_evaluated / answerable_trials, 4) if answerable_trials else 1.0
        ),
        task_fulfillment_pass_rate=(
            round(task_fulfillment_passed / task_fulfillment_evaluated, 4)
            if task_fulfillment_evaluated
            else None
        ),
        task_acceptance_evaluated_trials=acceptance_evaluated,
        task_acceptance_passed_trials=acceptance_passed,
        task_acceptance_evaluation_coverage=(
            round(acceptance_evaluated / answerable_trials, 4) if answerable_trials else 1.0
        ),
        task_acceptance_pass_rate=(
            round(acceptance_passed / acceptance_evaluated, 4) if acceptance_evaluated else None
        ),
        evidence_verified_passed_trials=evidence_verified_passed,
        evidence_unverified_potential_passes=evidence_unverified_potential_passes,
        evidence_supported_pass_rate_lower_bound=round(evidence_verified_passed / total, 4),
        evidence_supported_pass_rate_upper_bound=round(
            (evidence_verified_passed + evidence_unverified_potential_passes) / total,
            4,
        ),
        pass_rate_wilson_lower_bound_95=wilson_lower_bound(passed, total),
        pass_at_1_rate=round(sum(item.pass_at_1 for item in tasks) / len(tasks), 4),
        pass_power_k_rate=round(sum(item.pass_power_k for item in tasks) / len(tasks), 4),
        mean_score=_mean(scores),
        mean_latency_ms=_mean(latencies),
        p50_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
        mean_token_count=_mean(tokens),
        mean_tool_call_count=_mean(tools),
        mean_failed_tool_call_count=_mean(failed_tools),
        mean_tool_failure_rate=_mean(tool_failure_rates),
        tool_failure_free_rate=(
            round(sum(count == 0 for count in failed_tools) / len(failed_tools), 4) if failed_tools else None
        ),
        recovered_tool_failure_count=sum(item.recovered_tool_failure_count or 0 for item in trials),
        unrecovered_tool_failure_count=sum(item.unrecovered_tool_failure_count or 0 for item in trials),
        full_page_evidence_count=sum(item.full_page_evidence_count or 0 for item in trials),
        search_snippet_evidence_count=sum(item.search_snippet_evidence_count or 0 for item in trials),
        mean_full_page_evidence_rate=_mean(full_page_rates),
        page_fetch_attempt_count=sum(item.page_fetch_attempt_count or 0 for item in trials),
        page_fetch_failed_count=sum(item.page_fetch_failed_count or 0 for item in trials),
        mean_page_fetch_success_rate=_mean(page_fetch_success_rates),
        page_fetch_failure_categories=dict(sorted(page_fetch_failures.items())),
        mean_citation_validity=_mean(citations),
        mean_claim_citation_coverage=_mean(claim_coverage),
        mean_claim_citation_resolution=_mean(claim_resolution),
        mean_claim_proxy_support_rate=_mean(claim_proxy),
        mean_claim_semantic_support_rate=_mean(claim_semantic),
        failure_categories=dict(sorted(failures.items())),
    )


def render_deepresearch_benchmark_markdown(report: DeepResearchBenchmarkReport) -> str:
    metrics = report.metrics
    lines = [
        f"# DeepResearch benchmark: {report.benchmark_id} {report.benchmark_version}",
        "",
        f"- Status: `{report.status}`",
        f"- Evidence class: `{report.evidence_class}`",
        f"- Target revision: `{report.target_revision or 'unspecified'}`",
        f"- Target provider/model: `{report.target_provider or 'unspecified'}` / "
        f"`{report.target_model or 'unspecified'}`",
        f"- Claim Judge provider/model: `{report.claim_judge_provider or 'not configured'}` / "
        f"`{report.claim_judge_model or 'not configured'}`",
        f"- Claim Judge prompt version: `{report.claim_judge_prompt_version or 'not configured'}`",
        f"- Judge separation: `{report.judge_separation}`",
        f"- Trials: {metrics.total_trials} ({report.task_count} tasks × {report.trials_per_task})",
        f"- Completion rate: {metrics.completion_rate:.1%}",
        f"- Required-gate pass rate: {metrics.pass_rate:.1%}",
        f"- Deterministic process pass rate: {metrics.process_pass_rate:.1%}",
        f"- Claim citation-gate pass rate: {metrics.citation_gate_pass_rate:.1%}",
        f"- Semantic grading coverage: {metrics.semantic_evaluation_coverage:.1%}",
        (
            f"- Semantic gate pass rate: {metrics.semantic_gate_pass_rate:.1%}"
            if metrics.semantic_gate_pass_rate is not None
            else "- Semantic gate pass rate: unverified (no independent Judge)"
        ),
        f"- Task-fulfillment Judge coverage: "
        f"{metrics.task_fulfillment_evaluation_coverage:.1%}",
        (
            f"- Task-fulfillment Judge pass rate: {metrics.task_fulfillment_pass_rate:.1%}"
            if metrics.task_fulfillment_pass_rate is not None
            else "- Task-fulfillment Judge pass rate: unverified"
        ),
        (
            f"- Task-acceptance contract coverage: {metrics.task_acceptance_evaluation_coverage:.1%}"
            if metrics.task_acceptance_evaluation_coverage is not None
            else "- Task-acceptance contract coverage: unavailable (legacy report)"
        ),
        (
            f"- Task-acceptance contract pass rate: {metrics.task_acceptance_pass_rate:.1%}"
            if metrics.task_acceptance_pass_rate is not None
            else "- Task-acceptance contract pass rate: unverified"
        ),
        (
            "- Evidence-supported pass-rate bounds: "
            f"{metrics.evidence_supported_pass_rate_lower_bound:.1%}–"
            f"{metrics.evidence_supported_pass_rate_upper_bound:.1%} "
            f"({metrics.evidence_verified_passed_trials} verified; "
            f"{metrics.evidence_unverified_potential_passes} unverified potential passes)"
            if metrics.evidence_supported_pass_rate_lower_bound is not None
            and metrics.evidence_supported_pass_rate_upper_bound is not None
            else "- Evidence-supported pass-rate bounds: unavailable (legacy report)"
        ),
        f"- 95% Wilson lower bound: {metrics.pass_rate_wilson_lower_bound_95:.1%}",
        f"- Pass@1: {metrics.pass_at_1_rate:.1%}",
        f"- Pass^{report.trials_per_task}: {metrics.pass_power_k_rate:.1%}",
        f"- Mean tool failure rate: {metrics.mean_tool_failure_rate:.1%}"
        if metrics.mean_tool_failure_rate is not None
        else "- Mean tool failure rate: n/a",
        f"- Tool-failure-free trials: {metrics.tool_failure_free_rate:.1%}"
        if metrics.tool_failure_free_rate is not None
        else "- Tool-failure-free trials: n/a",
        f"- Recovered / unrecovered tool failures: "
        f"{metrics.recovered_tool_failure_count} / {metrics.unrecovered_tool_failure_count}",
        (
            f"- Full-page evidence rate: {metrics.mean_full_page_evidence_rate:.1%} "
            f"({metrics.full_page_evidence_count} fetched pages; "
            f"{metrics.search_snippet_evidence_count} search snippets)"
            if metrics.mean_full_page_evidence_rate is not None
            else "- Full-page evidence rate: n/a (no scoped evidence)"
        ),
        (
            f"- Page-fetch success rate: {metrics.mean_page_fetch_success_rate:.1%} "
            f"({metrics.page_fetch_attempt_count - metrics.page_fetch_failed_count}/"
            f"{metrics.page_fetch_attempt_count} attempts)"
            if metrics.mean_page_fetch_success_rate is not None
            else "- Page-fetch success rate: n/a (no fetch attempts)"
        ),
        (
            "- Page-fetch failure categories: "
            + ", ".join(
                f"{category}={count}" for category, count in metrics.page_fetch_failure_categories.items()
            )
            if metrics.page_fetch_failure_categories
            else "- Page-fetch failure categories: none"
        ),
        "",
        "## Per-task results",
        "",
        "| Task | Passed | Completed | Pass@1 | Pass^k | Mean score |",
        "|---|---:|---:|:---:|:---:|---:|",
    ]
    for task in report.task_summaries:
        score = "n/a" if task.mean_score is None else f"{task.mean_score:.2f}"
        lines.append(
            f"| {task.task_id} | {task.passed}/{task.trials} | {task.completed}/{task.trials} | "
            f"{'yes' if task.pass_at_1 else 'no'} | {'yes' if task.pass_power_k else 'no'} | {score} |"
        )
    lines.extend(["", "## Evidence boundary", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"


def rescore_deepresearch_benchmark(
    source_report: str | Path,
    *,
    report_output: str | Path,
    claim_report_filename: str | None = None,
) -> DeepResearchBenchmarkReport:
    """Reapply the current deterministic and claim gates to retained trial snapshots."""
    source_path = Path(source_report).resolve()
    output_path = Path(report_output).resolve()
    if output_path.exists():
        raise ValueError("rescore output already exists; choose a new immutable report path")
    if output_path.parent != source_path.parent:
        raise ValueError("rescore output must stay beside the source report so artifact links remain valid")
    source = DeepResearchBenchmarkReport.model_validate_json(source_path.read_text(encoding="utf-8"))
    if claim_report_filename and Path(claim_report_filename).name != claim_report_filename:
        raise ValueError("claim_report_filename must be a filename, not a path")
    rescored_trials: list[DeepResearchTrialResult] = []
    judge_identities: set[tuple[str | None, str | None, str | None]] = set()
    for trial in source.trials:
        if not trial.snapshot_path:
            if claim_report_filename and trial.completed:
                raise ValueError(
                    "every completed trial must retain a snapshot when claim reports are supplied"
                )
            rescored_trials.append(trial)
            continue
        snapshot_path = source_path.parent / trial.snapshot_path
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        trace, evaluation = _evaluate_snapshot(snapshot)
        if claim_report_filename:
            claim_path = snapshot_path.parent / claim_report_filename
            claim_report = ClaimEvidenceReport.model_validate_json(claim_path.read_text(encoding="utf-8"))
            run_data = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
            if claim_report.run_id != str(run_data.get("run_id") or ""):
                raise ValueError(f"retained claim report run_id mismatch: {claim_path}")
            if claim_report.trace_id != trace.trace_id:
                raise ValueError(f"retained claim report trace_id mismatch: {claim_path}")
            if trial.expected_status == "completed":
                judge_identities.add(
                    (
                        claim_report.judge_provider,
                        claim_report.judge_model,
                        claim_report.prompt_version,
                    )
                )
        else:
            claim_report = verify_deepresearch_claims(snapshot)
        run_data = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
        status = str(run_data.get("status") or trial.run_status)
        tool_call_count = int(trace.metadata.get("deepresearch.tool_call_count", 0) or 0)
        failed_tool_call_count = int(trace.metadata.get("deepresearch.failed_tool_call_count", 0) or 0)
        evidence_count = len(snapshot.get("evidence") or ())
        outcome = _score_expected_outcome(
            expected_status=trial.expected_status,
            status=status,
            evaluation=evaluation,
            claim_report=claim_report,
            task_acceptance_passed=trial.task_acceptance_passed,
            tool_call_count=tool_call_count,
            evidence_count=evidence_count,
        )
        rescored_trials.append(
            trial.model_copy(
                update={
                    "trace_id": trace.trace_id,
                    "completed": outcome["completed"],
                    "passed": outcome["passed"],
                    "process_passed": outcome["process_passed"],
                    "citation_gate_passed": outcome["citation_gate_passed"],
                    "semantic_gate_passed": outcome["semantic_gate_passed"],
                    "task_fulfillment_gate_passed": outcome[
                        "task_fulfillment_gate_passed"
                    ],
                    "task_fulfillment_verdict": outcome["task_fulfillment_verdict"],
                    "task_missing_requirements": outcome["task_missing_requirements"],
                    "quality_status": outcome["quality_status"],
                    "score": outcome["score"],
                    "tool_call_count": tool_call_count,
                    "failed_tool_call_count": failed_tool_call_count,
                    "tool_failure_rate": (
                        round(failed_tool_call_count / tool_call_count, 4) if tool_call_count else 0.0
                    ),
                    "recovered_tool_failure_count": int(
                        trace.metadata.get("deepresearch.recovered_tool_failure_count", 0) or 0
                    ),
                    "unrecovered_tool_failure_count": int(
                        trace.metadata.get("deepresearch.unrecovered_tool_failure_count", 0) or 0
                    ),
                    "evidence_count": evidence_count,
                    "full_page_evidence_count": int(
                        trace.metadata.get("deepresearch.full_page_evidence_count", 0) or 0
                    ),
                    "search_snippet_evidence_count": int(
                        trace.metadata.get("deepresearch.search_snippet_evidence_count", 0) or 0
                    ),
                    "full_page_evidence_rate": trace.metadata.get("deepresearch.full_page_evidence_rate"),
                    "page_fetch_attempt_count": int(
                        trace.metadata.get("deepresearch.page_fetch_attempt_count", 0) or 0
                    ),
                    "page_fetch_failed_count": int(
                        trace.metadata.get("deepresearch.page_fetch_failed_count", 0) or 0
                    ),
                    "page_fetch_success_rate": trace.metadata.get("deepresearch.page_fetch_success_rate"),
                    "page_fetch_failure_categories": dict(
                        trace.metadata.get("deepresearch.page_fetch_failure_categories") or {}
                    ),
                    "citation_validity": (
                        (trace.metadata.get("deepresearch.citations") or {}).get("validity")
                        if trial.expected_status == "completed"
                        else outcome["citation_validity"]
                    ),
                    "claim_citation_coverage": outcome["claim_citation_coverage"],
                    "claim_citation_resolution": outcome["claim_citation_resolution"],
                    "claim_proxy_support_rate": outcome["claim_proxy_support_rate"],
                    "claim_semantic_support_rate": outcome["claim_semantic_support_rate"],
                    "claim_recommendation": outcome["claim_recommendation"],
                    "failure_categories": outcome["failure_categories"],
                }
            )
        )
    trials = tuple(rescored_trials)
    task_ids = tuple(item.task_id for item in source.task_summaries)
    task_summaries = tuple(_summarize_task(task_id, trials) for task_id in task_ids)
    metrics = _summarize_benchmark(trials, task_summaries)
    if metrics.completed_trials == metrics.total_trials:
        status = "completed"
    elif metrics.completed_trials:
        status = "partial"
    else:
        status = "failed"
    now = datetime.now(UTC)
    claim_judge_provider = source.claim_judge_provider
    claim_judge_model = source.claim_judge_model
    claim_judge_prompt_version = source.claim_judge_prompt_version
    judge_separation = source.judge_separation
    if claim_report_filename:
        if len(judge_identities) != 1:
            raise ValueError(
                "retained claim reports must use one consistent Judge provider/model/prompt version"
            )
        claim_judge_provider, claim_judge_model, claim_judge_prompt_version = next(iter(judge_identities))
        judge_separation = _judge_separation_for_identity(
            target_provider=source.target_provider,
            target_model=source.target_model,
            judge_provider=claim_judge_provider,
            judge_model=claim_judge_model,
        )
    preserved_limitations = tuple(
        item
        for item in source.limitations
        if not item.startswith("Claim Judge separation is ")
        and not item.startswith("Independent semantic grading remains ")
    )
    additional_limitations = [
        "This is an offline rescore of retained snapshots; it does not rerun the target Agent."
    ]
    if claim_report_filename:
        additional_limitations.append(
            "Semantic ratings were loaded from retained per-trial claim reports and were not rerun."
        )
        if judge_separation == "different_model":
            additional_limitations.append(
                "The Judge uses a different model from the same provider; this is model-separated, "
                "not provider-independent."
            )
        elif judge_separation in {"not_configured", "unknown", "same_model"}:
            additional_limitations.append(
                f"Claim Judge separation is {judge_separation}; semantic ratings must not be "
                "described as independent."
            )
    else:
        additional_limitations.append(
            "Independent semantic grading remains unverified unless the retained report already includes it."
        )
    limitations = tuple(
        dict.fromkeys(
            (
                *preserved_limitations,
                *additional_limitations,
            )
        )
    )
    report = source.model_copy(
        update={
            "created_at": now,
            "finished_at": now,
            "claim_judge_provider": claim_judge_provider,
            "claim_judge_model": claim_judge_model,
            "claim_judge_prompt_version": claim_judge_prompt_version,
            "judge_separation": judge_separation,
            "status": status,
            "metrics": metrics,
            "task_summaries": task_summaries,
            "trials": trials,
            "limitations": limitations,
        }
    )
    _write_json(output_path, report.model_dump(mode="json"))
    output_path.with_suffix(".md").write_text(
        render_deepresearch_benchmark_markdown(report), encoding="utf-8"
    )
    return report


def run_deepresearch_benchmark(
    manifest_path: str | Path,
    *,
    base_url: str,
    source_database: str | Path,
    output_directory: str | Path,
    bearer_token: str | None = None,
    client: httpx.Client | None = None,
    trace_evaluator: TraceEvaluator = _evaluate_snapshot,
    claim_judge: ClaimSupportJudge | None = None,
) -> DeepResearchBenchmarkReport:
    with DeepResearchBenchmarkRunner(
        base_url=base_url,
        source_database=source_database,
        output_directory=output_directory,
        bearer_token=bearer_token,
        client=client,
        trace_evaluator=trace_evaluator,
        claim_judge=claim_judge,
    ) as runner:
        return runner.run(manifest_path)
