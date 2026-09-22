from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from agenttracelab.models import FrozenModel
from agenttracelab.stability import GepaStabilityReport, GepaStabilitySample


class HardCaseMiningPolicy(FrozenModel):
    max_cases: int = Field(default=50, ge=1, le=1_000)
    max_per_signature: int = Field(default=5, ge=1, le=100)
    include_candidate_failures: bool = True
    include_score_regressions: bool = True
    include_resolved_baseline_failures: bool = True


class HardCaseEntry(FrozenModel):
    case_id: str = Field(pattern=r"^hardcase_[a-f0-9]{24}$")
    case_kind: Literal[
        "candidate_failure",
        "safety_regression",
        "score_regression",
        "resolved_baseline_failure",
    ]
    severity: Literal["critical", "high", "medium"]
    priority_score: float = Field(ge=0, le=200)
    task_id: str
    trial: int = Field(ge=1)
    failure_signature: tuple[str, ...] = Field(min_length=1)
    baseline_score: float = Field(ge=0, le=1)
    candidate_score: float = Field(ge=0, le=1)
    score_delta: float = Field(ge=-1, le=1)
    baseline_trace_ids: tuple[str, ...]
    candidate_trace_ids: tuple[str, ...]
    reasons: tuple[str, ...] = Field(min_length=1)


class HardCaseMiningReport(FrozenModel):
    schema_version: Literal["agenttracelab.hard-case-mining.v1"] = "agenttracelab.hard-case-mining.v1"
    dataset_id: str
    dataset_version: str
    created_at: datetime
    source_stability_id: str
    source_stability_version: str
    policy: HardCaseMiningPolicy
    source_sample_count: int = Field(ge=1)
    eligible_case_count: int = Field(ge=0)
    selected_case_count: int = Field(ge=0)
    distinct_signature_count: int = Field(ge=0)
    selected_signature_count: int = Field(ge=0)
    signature_coverage: float = Field(ge=0, le=1)
    cases: tuple[HardCaseEntry, ...]
    limitations: tuple[str, ...]


def _stable_case_id(report: GepaStabilityReport, sample: GepaStabilitySample) -> str:
    payload = {
        "stability_id": report.stability_id,
        "stability_version": report.stability_version,
        "task_id": sample.task_id,
        "trial": sample.trial,
        "baseline_trace_ids": sample.baseline_trace_ids,
        "candidate_trace_ids": sample.candidate_trace_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"hardcase_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _entry_for_sample(
    report: GepaStabilityReport,
    sample: GepaStabilitySample,
    policy: HardCaseMiningPolicy,
) -> HardCaseEntry | None:
    kind: str | None = None
    severity: str | None = None
    signature: tuple[str, ...] = ()
    reasons: list[str] = []

    safety_regression = sample.baseline_safety_score == 1.0 and sample.candidate_safety_score < 1.0
    candidate_failure = not sample.candidate_passed or sample.candidate_score < 1.0
    score_regression = sample.score_delta < 0
    resolved_baseline = not sample.baseline_passed and sample.candidate_passed

    if safety_regression and policy.include_candidate_failures:
        kind = "safety_regression"
        severity = "critical"
        signature = tuple(sorted({*sample.candidate_failed_check_ids, "safety.regression"}))
        reasons.append("candidate safety score regressed relative to a safe baseline")
    elif candidate_failure and policy.include_candidate_failures:
        kind = "candidate_failure"
        severity = "critical" if sample.candidate_safety_score < 1.0 else "high"
        signature = tuple(sorted(sample.candidate_failed_check_ids or ("replay.expectation_mismatch",)))
        reasons.append("candidate failed a held-out expectation or deterministic gate")
    elif score_regression and policy.include_score_regressions:
        kind = "score_regression"
        severity = "high"
        signature = tuple(sorted(sample.candidate_failed_check_ids or ("score.paired_regression",)))
        reasons.append("candidate score was lower than its paired baseline")
    elif resolved_baseline and policy.include_resolved_baseline_failures:
        kind = "resolved_baseline_failure"
        severity = "medium"
        signature = tuple(sorted(sample.baseline_failed_check_ids or ("baseline.expectation_mismatch",)))
        reasons.append("candidate resolved a baseline failure and should retain a regression guard")
    else:
        return None

    severity_weight = {"critical": 100.0, "high": 70.0, "medium": 40.0}[severity]
    priority = severity_weight
    priority += max(0.0, -sample.score_delta) * 20
    priority += (1 - sample.candidate_score) * 10
    if kind == "resolved_baseline_failure":
        priority += max(0.0, sample.score_delta) * 10
    return HardCaseEntry(
        case_id=_stable_case_id(report, sample),
        case_kind=kind,
        severity=severity,
        priority_score=round(priority, 4),
        task_id=sample.task_id,
        trial=sample.trial,
        failure_signature=signature,
        baseline_score=sample.baseline_score,
        candidate_score=sample.candidate_score,
        score_delta=sample.score_delta,
        baseline_trace_ids=sample.baseline_trace_ids,
        candidate_trace_ids=sample.candidate_trace_ids,
        reasons=tuple(reasons),
    )


def mine_gepa_hard_cases(
    report: GepaStabilityReport,
    *,
    dataset_id: str,
    dataset_version: str,
    policy: HardCaseMiningPolicy | None = None,
) -> HardCaseMiningReport:
    if not dataset_id or len(dataset_id) > 128:
        raise ValueError("dataset_id must contain 1-128 characters")
    if not dataset_version or len(dataset_version) > 64:
        raise ValueError("dataset_version must contain 1-64 characters")
    active_policy = policy or HardCaseMiningPolicy()
    eligible = [
        entry
        for sample in report.samples
        if (entry := _entry_for_sample(report, sample, active_policy)) is not None
    ]
    eligible.sort(key=lambda item: (-item.priority_score, item.case_id))
    buckets: dict[str, list[HardCaseEntry]] = defaultdict(list)
    for entry in eligible:
        signature_key = f"{entry.case_kind}|{'|'.join(entry.failure_signature)}"
        buckets[signature_key].append(entry)
    bucket_order = sorted(
        buckets,
        key=lambda key: (-buckets[key][0].priority_score, key),
    )
    selected: list[HardCaseEntry] = []
    selected_per_signature: dict[str, int] = defaultdict(int)
    while len(selected) < active_policy.max_cases:
        progressed = False
        for key in bucket_order:
            if len(selected) >= active_policy.max_cases:
                break
            if selected_per_signature[key] >= active_policy.max_per_signature or not buckets[key]:
                continue
            selected.append(buckets[key].pop(0))
            selected_per_signature[key] += 1
            progressed = True
        if not progressed:
            break
    selected_signature_count = sum(count > 0 for count in selected_per_signature.values())
    signature_count = len(bucket_order)
    coverage = round(selected_signature_count / signature_count, 4) if signature_count else 0.0
    return HardCaseMiningReport(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        created_at=datetime.now(UTC),
        source_stability_id=report.stability_id,
        source_stability_version=report.stability_version,
        policy=active_policy,
        source_sample_count=len(report.samples),
        eligible_case_count=len(eligible),
        selected_case_count=len(selected),
        distinct_signature_count=signature_count,
        selected_signature_count=selected_signature_count,
        signature_coverage=coverage,
        cases=tuple(selected),
        limitations=(
            "The queue contains trace references and check IDs, not raw prompts or tool payloads.",
            "Failure signatures support triage and diversity; they do not establish root cause.",
            "Human review is required before adding a case to training or release-gate datasets.",
        ),
    )


def mine_gepa_hard_cases_file(
    report_path: str | Path,
    *,
    dataset_id: str,
    dataset_version: str,
    policy: HardCaseMiningPolicy | None = None,
) -> HardCaseMiningReport:
    path = Path(report_path).resolve()
    if path.stat().st_size > 16_777_216:
        raise ValueError("GEPA stability report exceeds the 16 MiB limit")
    report = GepaStabilityReport.model_validate_json(path.read_text(encoding="utf-8"))
    return mine_gepa_hard_cases(
        report,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        policy=policy,
    )
