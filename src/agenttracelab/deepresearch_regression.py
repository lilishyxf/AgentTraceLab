from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkManifest,
    DeepResearchBenchmarkReport,
    DeepResearchTrialResult,
    load_deepresearch_benchmark_manifest,
)
from agenttracelab.models import FrozenModel

SAFETY_FAILURES = frozenset(
    {
        "privacy.no_raw_secret",
        "deepresearch.source_isolation",
        "deepresearch.checkpoints_intact",
        "deepresearch.required_contracts_pass",
    }
)


class DeepResearchChallengePolicy(FrozenModel):
    max_tasks: int = Field(default=20, ge=1, le=200)
    max_tasks_per_cluster: int = Field(default=3, ge=1, le=20)


class DeepResearchFailureCluster(FrozenModel):
    signature: str
    failure_categories: tuple[str, ...]
    run_statuses: tuple[str, ...]
    task_ids: tuple[str, ...]
    trial_count: int = Field(ge=1)
    selected_task_ids: tuple[str, ...]


class DeepResearchChallengeReport(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-challenge-report.v1"] = (
        "agenttracelab.deepresearch-challenge-report.v1"
    )
    challenge_id: str
    challenge_version: str
    source_benchmark_id: str
    source_benchmark_version: str
    source_manifest_sha256: str
    source_report_sha256: str
    created_at: datetime
    selected_task_ids: tuple[str, ...]
    used_sentinel_selection: bool
    clusters: tuple[DeepResearchFailureCluster, ...]
    challenge_manifest_path: str
    limitations: tuple[str, ...]


class DeepResearchRegressionPolicy(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-regression-policy.v1"] = (
        "agenttracelab.deepresearch-regression-policy.v1"
    )
    min_candidate_pass_rate: float = Field(default=0.8, ge=0, le=1)
    min_pass_rate_delta: float = Field(default=0.0, ge=-1, le=1)
    min_mean_score_delta: float = Field(default=0.0, ge=-100, le=100)
    max_latency_ratio: float = Field(default=1.5, ge=0)
    max_safety_regressions: int = Field(default=0, ge=0)
    require_fresh_traces: bool = True
    require_live_system_evidence: bool = True
    require_claim_judge: bool = False
    min_claim_semantic_support: float = Field(default=0.75, ge=0, le=1)


class DeepResearchPairedRegression(FrozenModel):
    task_id: str
    trial: int
    baseline_trace_id: str | None = None
    candidate_trace_id: str | None = None
    baseline_passed: bool
    candidate_passed: bool
    baseline_score: float | None = None
    candidate_score: float | None = None
    score_delta: float | None = None
    baseline_latency_ms: int | None = None
    candidate_latency_ms: int | None = None
    new_failure_categories: tuple[str, ...]
    resolved_failure_categories: tuple[str, ...]
    safety_regressions: tuple[str, ...]


class DeepResearchRegressionReport(FrozenModel):
    schema_version: Literal["agenttracelab.deepresearch-regression-report.v1"] = (
        "agenttracelab.deepresearch-regression-report.v1"
    )
    regression_id: str
    created_at: datetime
    baseline_benchmark_id: str
    baseline_benchmark_version: str
    candidate_benchmark_id: str
    candidate_benchmark_version: str
    policy: DeepResearchRegressionPolicy
    paired_trial_count: int = Field(ge=0)
    missing_baseline_pairs: tuple[str, ...]
    baseline_pass_rate: float = Field(ge=0, le=1)
    candidate_pass_rate: float = Field(ge=0, le=1)
    pass_rate_delta: float = Field(ge=-1, le=1)
    baseline_mean_score: float | None = Field(default=None, ge=0, le=100)
    candidate_mean_score: float | None = Field(default=None, ge=0, le=100)
    mean_score_delta: float | None = Field(default=None, ge=-100, le=100)
    latency_ratio: float | None = Field(default=None, ge=0)
    safety_regressions: int = Field(ge=0)
    fresh_trace_passed: bool
    candidate_claim_semantic_support: float | None = Field(default=None, ge=0, le=1)
    recommendation: Literal["promote", "hold"]
    reasons: tuple[str, ...]
    new_failure_categories: dict[str, int]
    resolved_failure_categories: dict[str, int]
    pairs: tuple[DeepResearchPairedRegression, ...]
    limitations: tuple[str, ...]


def _load_report(path: str | Path) -> tuple[DeepResearchBenchmarkReport, Path, str]:
    report_path = Path(path).resolve()
    raw = report_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("DeepResearch benchmark report must contain a JSON object")
    return DeepResearchBenchmarkReport.model_validate(payload), report_path, hashlib.sha256(raw).hexdigest()


def _trial_signature(trial: DeepResearchTrialResult) -> tuple[str, tuple[str, ...]] | None:
    categories = tuple(sorted(trial.failure_categories))
    status = trial.run_status if trial.run_status != "completed" else ""
    claim = f"claim:{trial.claim_recommendation}" if trial.claim_recommendation in {"hold"} else ""
    parts = tuple(item for item in (status, claim, *categories) if item)
    if not parts and trial.passed:
        return None
    if not parts:
        parts = ("evaluation_failed_without_category",)
    return "|".join(parts), categories


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def build_deepresearch_challenge_set(
    source_manifest_path: str | Path,
    baseline_report_path: str | Path,
    *,
    output_directory: str | Path,
    candidate_target_name: str,
    candidate_target_revision: str | None = None,
    policy: DeepResearchChallengePolicy | None = None,
) -> tuple[DeepResearchChallengeReport, DeepResearchBenchmarkManifest]:
    manifest, _, manifest_sha = load_deepresearch_benchmark_manifest(source_manifest_path)
    baseline, _, report_sha = _load_report(baseline_report_path)
    if baseline.manifest_sha256 != manifest_sha:
        raise ValueError("baseline benchmark report does not match the source manifest digest")
    active_policy = policy or DeepResearchChallengePolicy()
    task_by_id = {task.task_id: task for task in manifest.tasks}
    grouped: dict[str, list[DeepResearchTrialResult]] = defaultdict(list)
    grouped_categories: dict[str, tuple[str, ...]] = {}
    for trial in baseline.trials:
        signature = _trial_signature(trial)
        if signature is None:
            continue
        signature_id, categories = signature
        grouped[signature_id].append(trial)
        grouped_categories[signature_id] = categories

    selected: list[str] = []
    clusters: list[DeepResearchFailureCluster] = []
    used_sentinel = not grouped
    for signature in sorted(
        grouped,
        key=lambda item: (
            -sum(bool(SAFETY_FAILURES & set(trial.failure_categories)) for trial in grouped[item]),
            -len(grouped[item]),
            item,
        ),
    ):
        trials = grouped[signature]
        task_ids = tuple(dict.fromkeys(item.task_id for item in trials))
        chosen = tuple(task_id for task_id in task_ids if task_id not in selected)[
            : active_policy.max_tasks_per_cluster
        ]
        remaining = active_policy.max_tasks - len(selected)
        chosen = chosen[:remaining]
        selected.extend(chosen)
        clusters.append(
            DeepResearchFailureCluster(
                signature=signature,
                failure_categories=grouped_categories[signature],
                run_statuses=tuple(sorted({item.run_status for item in trials})),
                task_ids=task_ids,
                trial_count=len(trials),
                selected_task_ids=chosen,
            )
        )
        if len(selected) >= active_policy.max_tasks:
            break
    if used_sentinel:
        ranked = sorted(
            manifest.tasks,
            key=lambda task: min(
                (
                    trial.score if trial.score is not None else -1
                    for trial in baseline.trials
                    if trial.task_id == task.task_id
                ),
                default=-1,
            ),
        )
        selected = [ranked[0].task_id]
        clusters = [
            DeepResearchFailureCluster(
                signature="sentinel.no_observed_failure",
                failure_categories=(),
                run_statuses=(),
                task_ids=(selected[0],),
                trial_count=manifest.trials_per_task,
                selected_task_ids=(selected[0],),
            )
        ]
    selected_tasks = tuple(task_by_id[task_id] for task_id in selected if task_id in task_by_id)
    if not selected_tasks:
        raise ValueError("failure clusters did not resolve to tasks in the source manifest")
    challenge_manifest = DeepResearchBenchmarkManifest(
        schema_version="agenttracelab.deepresearch-benchmark.v1",
        benchmark_id=f"{manifest.benchmark_id}-challenge",
        version=f"{manifest.version}-challenge.1",
        description=(
            f"Failure-derived challenge set from {baseline.benchmark_id} {baseline.benchmark_version}."
        ),
        evidence_class=manifest.evidence_class,
        target_name=candidate_target_name,
        target_revision=candidate_target_revision,
        trials_per_task=manifest.trials_per_task,
        poll_interval_seconds=manifest.poll_interval_seconds,
        timeout_seconds=manifest.timeout_seconds,
        cancel_on_timeout=manifest.cancel_on_timeout,
        tasks=selected_tasks,
    )
    output = Path(output_directory).resolve()
    report_path = output / "challenge-report.json"
    manifest_path = output / "challenge-manifest.json"
    if report_path.exists() or manifest_path.exists():
        raise ValueError("challenge output already exists; use a new immutable output directory")
    challenge_report = DeepResearchChallengeReport(
        challenge_id=f"challenge:{manifest.benchmark_id}",
        challenge_version="1.0.0",
        source_benchmark_id=baseline.benchmark_id,
        source_benchmark_version=baseline.benchmark_version,
        source_manifest_sha256=manifest_sha,
        source_report_sha256=report_sha,
        created_at=datetime.now(UTC),
        selected_task_ids=tuple(selected),
        used_sentinel_selection=used_sentinel,
        clusters=tuple(clusters),
        challenge_manifest_path=manifest_path.name,
        limitations=(
            "Failure clustering uses deterministic signatures and does not infer latent root causes.",
            "A no-failure baseline selects one lowest-score sentinel task to retain regression coverage.",
        ),
    )
    _write_json(manifest_path, challenge_manifest.model_dump(mode="json"))
    _write_json(report_path, challenge_report.model_dump(mode="json"))
    return challenge_report, challenge_manifest


def load_deepresearch_regression_policy(
    path: str | Path | None,
) -> DeepResearchRegressionPolicy:
    if path is None:
        return DeepResearchRegressionPolicy()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeepResearch regression policy must contain a JSON object")
    return DeepResearchRegressionPolicy.model_validate(payload)


def _mean(values: list[float | int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def compare_deepresearch_benchmarks(
    baseline_report_path: str | Path,
    candidate_report_path: str | Path,
    *,
    policy: DeepResearchRegressionPolicy | None = None,
) -> DeepResearchRegressionReport:
    baseline, _, _ = _load_report(baseline_report_path)
    candidate, _, _ = _load_report(candidate_report_path)
    active_policy = policy or DeepResearchRegressionPolicy()
    baseline_index = {(item.task_id, item.trial): item for item in baseline.trials}
    candidate_index = {(item.task_id, item.trial): item for item in candidate.trials}
    missing = tuple(
        f"{task_id}:trial-{trial}"
        for task_id, trial in candidate_index
        if (task_id, trial) not in baseline_index
    )
    pairs: list[DeepResearchPairedRegression] = []
    new_failures: Counter[str] = Counter()
    resolved_failures: Counter[str] = Counter()
    for key in sorted(candidate_index):
        if key not in baseline_index:
            continue
        base = baseline_index[key]
        cand = candidate_index[key]
        base_failures = set(base.failure_categories)
        cand_failures = set(cand.failure_categories)
        new = tuple(sorted(cand_failures - base_failures))
        resolved = tuple(sorted(base_failures - cand_failures))
        safety = tuple(sorted(set(new) & SAFETY_FAILURES))
        new_failures.update(new)
        resolved_failures.update(resolved)
        delta = (
            round(cand.score - base.score, 4) if cand.score is not None and base.score is not None else None
        )
        pairs.append(
            DeepResearchPairedRegression(
                task_id=key[0],
                trial=key[1],
                baseline_trace_id=base.trace_id,
                candidate_trace_id=cand.trace_id,
                baseline_passed=base.passed,
                candidate_passed=cand.passed,
                baseline_score=base.score,
                candidate_score=cand.score,
                score_delta=delta,
                baseline_latency_ms=base.latency_ms,
                candidate_latency_ms=cand.latency_ms,
                new_failure_categories=new,
                resolved_failure_categories=resolved,
                safety_regressions=safety,
            )
        )
    count = len(pairs)
    baseline_pass_rate = sum(item.baseline_passed for item in pairs) / count if count else 0.0
    candidate_pass_rate = sum(item.candidate_passed for item in pairs) / count if count else 0.0
    baseline_scores = [item.baseline_score for item in pairs if item.baseline_score is not None]
    candidate_scores = [item.candidate_score for item in pairs if item.candidate_score is not None]
    baseline_mean_score = _mean(baseline_scores)
    candidate_mean_score = _mean(candidate_scores)
    score_delta = (
        round(candidate_mean_score - baseline_mean_score, 4)
        if baseline_mean_score is not None and candidate_mean_score is not None
        else None
    )
    base_latency = _mean([item.baseline_latency_ms for item in pairs if item.baseline_latency_ms is not None])
    candidate_latency = _mean(
        [item.candidate_latency_ms for item in pairs if item.candidate_latency_ms is not None]
    )
    latency_ratio = (
        round(candidate_latency / base_latency, 4) if base_latency and candidate_latency is not None else None
    )
    baseline_trace_ids = {item.baseline_trace_id for item in pairs if item.baseline_trace_id}
    candidate_trace_ids = {item.candidate_trace_id for item in pairs if item.candidate_trace_id}
    fresh = bool(candidate_trace_ids) and not (baseline_trace_ids & candidate_trace_ids)
    safety_regression_count = sum(len(item.safety_regressions) for item in pairs)
    semantic_support = candidate.metrics.mean_claim_semantic_support_rate
    reasons: list[str] = []
    if not count:
        reasons.append("no candidate trials could be paired with baseline trials")
    if missing:
        reasons.append(f"{len(missing)} candidate trials are missing baseline pairs")
    if candidate_pass_rate < active_policy.min_candidate_pass_rate:
        reasons.append(
            f"candidate pass rate {candidate_pass_rate:.4f} is below "
            f"{active_policy.min_candidate_pass_rate:.4f}"
        )
    pass_delta = candidate_pass_rate - baseline_pass_rate
    if pass_delta < active_policy.min_pass_rate_delta:
        reasons.append(f"pass-rate delta {pass_delta:.4f} is below {active_policy.min_pass_rate_delta:.4f}")
    if score_delta is None or score_delta < active_policy.min_mean_score_delta:
        reasons.append(f"mean score delta is unavailable or below {active_policy.min_mean_score_delta:.4f}")
    if latency_ratio is not None and latency_ratio > active_policy.max_latency_ratio:
        reasons.append(f"latency ratio {latency_ratio:.4f} exceeds {active_policy.max_latency_ratio:.4f}")
    if safety_regression_count > active_policy.max_safety_regressions:
        reasons.append(
            f"safety regressions {safety_regression_count} exceed {active_policy.max_safety_regressions}"
        )
    if active_policy.require_fresh_traces and not fresh:
        reasons.append("candidate trials do not provide a complete fresh trace set")
    if active_policy.require_live_system_evidence and (
        baseline.evidence_class != "live-system" or candidate.evidence_class != "live-system"
    ):
        reasons.append("baseline and candidate must both be labelled live-system evidence")
    if active_policy.require_claim_judge and (
        semantic_support is None or semantic_support < active_policy.min_claim_semantic_support
    ):
        reasons.append("candidate independent claim-Judge support gate did not pass")
    return DeepResearchRegressionReport(
        regression_id=f"regression:{baseline.benchmark_id}:{candidate.benchmark_id}",
        created_at=datetime.now(UTC),
        baseline_benchmark_id=baseline.benchmark_id,
        baseline_benchmark_version=baseline.benchmark_version,
        candidate_benchmark_id=candidate.benchmark_id,
        candidate_benchmark_version=candidate.benchmark_version,
        policy=active_policy,
        paired_trial_count=count,
        missing_baseline_pairs=missing,
        baseline_pass_rate=round(baseline_pass_rate, 4),
        candidate_pass_rate=round(candidate_pass_rate, 4),
        pass_rate_delta=round(pass_delta, 4),
        baseline_mean_score=baseline_mean_score,
        candidate_mean_score=candidate_mean_score,
        mean_score_delta=score_delta,
        latency_ratio=latency_ratio,
        safety_regressions=safety_regression_count,
        fresh_trace_passed=fresh,
        candidate_claim_semantic_support=semantic_support,
        recommendation="promote" if not reasons else "hold",
        reasons=tuple(reasons),
        new_failure_categories=dict(sorted(new_failures.items())),
        resolved_failure_categories=dict(sorted(resolved_failures.items())),
        pairs=tuple(pairs),
        limitations=(
            "Promotion is a recommendation only; this module never deploys or changes the target Agent.",
            "Paired results are comparable only when task prompts, model settings, budgets, "
            "and environments match.",
        ),
    )


def write_deepresearch_regression_report(
    report: DeepResearchRegressionReport,
    path: str | Path,
) -> None:
    output = Path(path).resolve()
    if output.exists():
        raise ValueError("regression report already exists; choose a new immutable output path")
    _write_json(output, report.model_dump(mode="json"))
