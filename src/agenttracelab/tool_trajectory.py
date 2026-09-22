from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenttracelab.adapters.deepresearch import DeepResearchSnapshot
from agenttracelab.models import FrozenModel


class ToolTrajectoryExpectation(FrozenModel):
    expected_tool_names: tuple[str, ...] = ()
    ordered: bool = False
    allow_additional_tools: bool = True
    min_tool_calls: int = Field(default=1, ge=0, le=100)
    max_tool_calls: int = Field(default=10, ge=0, le=100)
    required_source_mode: str | None = None
    accepted_evidence_domains: tuple[str, ...] = ()
    min_accepted_domain_evidence: int = Field(default=0, ge=0, le=100)
    require_result_present: bool = True
    min_evidence_per_completed_call: int = Field(default=1, ge=0, le=100)
    min_failed_tool_calls: int = Field(default=0, ge=0, le=100)
    max_failed_tool_calls: int = Field(default=0, ge=0, le=100)
    min_recovered_tool_failures: int = Field(default=0, ge=0, le=100)
    max_unrecovered_tool_failures: int = Field(default=0, ge=0, le=100)
    max_duplicate_tool_calls: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_call_range(self) -> ToolTrajectoryExpectation:
        if self.min_tool_calls > self.max_tool_calls:
            raise ValueError("min_tool_calls must not exceed max_tool_calls")
        if self.min_failed_tool_calls > self.max_failed_tool_calls:
            raise ValueError("min_failed_tool_calls must not exceed max_failed_tool_calls")
        if self.min_recovered_tool_failures > self.max_failed_tool_calls:
            raise ValueError("min_recovered_tool_failures cannot exceed max_failed_tool_calls")
        if self.min_accepted_domain_evidence and not self.accepted_evidence_domains:
            raise ValueError("accepted_evidence_domains are required when a minimum domain count is set")
        return self


class ToolTrajectoryCase(FrozenModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    snapshot: str = Field(min_length=1, max_length=512)
    expectation: ToolTrajectoryExpectation
    tags: tuple[str, ...] = ()


class ToolTrajectoryManifest(FrozenModel):
    schema_version: Literal["agenttracelab.tool-trajectory.v1"]
    dataset_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    review_mode: Literal["human-approved", "generated-synthetic"]
    annotator_count: int = Field(default=0, ge=0, le=100)
    cases: tuple[ToolTrajectoryCase, ...] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_review_provenance(self) -> ToolTrajectoryManifest:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("tool-trajectory case_id values must be unique")
        if self.review_mode == "human-approved" and self.annotator_count < 1:
            raise ValueError("human-approved tool trajectories require at least one annotator")
        if self.review_mode == "generated-synthetic" and self.annotator_count:
            raise ValueError("generated-synthetic tool trajectories cannot claim human annotators")
        return self


class ToolTrajectoryAnnotationCase(FrozenModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    prompt: str = Field(min_length=1, max_length=4000)
    task_family: Literal[
        "official-source-research",
        "conflicting-evidence",
        "insufficient-evidence",
        "tool-recovery",
        "tool-efficiency",
        "clarification",
        "safety",
    ]
    execution_lane: Literal["live-safe", "controlled-fault", "manual-only"] = "live-safe"
    expected_answer_behavior: str = Field(min_length=1, max_length=2000)
    draft_expectation: ToolTrajectoryExpectation
    expected_domains: tuple[str, ...] = ()
    review_status: Literal["unreviewed", "human-reviewed"] = "unreviewed"
    review_notes: str | None = Field(default=None, max_length=2000)
    tags: tuple[str, ...] = ()


class ToolTrajectoryAnnotationQueue(FrozenModel):
    schema_version: Literal["agenttracelab.tool-trajectory-annotation-queue.v1"]
    dataset_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    status: Literal["unreviewed", "in-review", "human-approved"]
    annotator_count: int = Field(default=0, ge=0, le=100)
    cases: tuple[ToolTrajectoryAnnotationCase, ...] = Field(min_length=20, max_length=500)

    @model_validator(mode="after")
    def validate_review_status(self) -> ToolTrajectoryAnnotationQueue:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("annotation queue case_id values must be unique")
        reviewed = sum(case.review_status == "human-reviewed" for case in self.cases)
        if self.status == "unreviewed" and (self.annotator_count or reviewed):
            raise ValueError("an unreviewed queue cannot claim annotators or reviewed cases")
        if self.status == "human-approved":
            if self.annotator_count < 1:
                raise ValueError("a human-approved queue requires at least one annotator")
            if reviewed != len(self.cases):
                raise ValueError("every case must be human-reviewed before queue approval")
        return self


class ToolTrajectoryCaseResult(FrozenModel):
    case_id: str
    trace_run_id: str
    passed: bool
    tool_selection_score: float = Field(ge=0, le=1)
    argument_observability_score: float = Field(ge=0, le=1)
    call_budget_score: float = Field(ge=0, le=1)
    source_policy_score: float = Field(ge=0, le=1)
    evidence_domain_score: float = Field(ge=0, le=1)
    result_grounding_score: float = Field(ge=0, le=1)
    recovery_safety_score: float = Field(ge=0, le=1)
    actual_tool_names: tuple[str, ...]
    tool_call_count: int = Field(ge=0)
    failed_tool_call_count: int = Field(ge=0)
    recovered_tool_failure_count: int = Field(ge=0)
    unrecovered_tool_failure_count: int = Field(ge=0)
    duplicate_tool_call_count: int = Field(ge=0)
    findings: tuple[str, ...] = ()


class ToolTrajectoryReport(FrozenModel):
    schema_version: Literal["agenttracelab.tool-trajectory-report.v1"] = (
        "agenttracelab.tool-trajectory-report.v1"
    )
    dataset_id: str
    dataset_version: str
    review_mode: Literal["human-approved", "generated-synthetic"]
    promotion_eligible: bool
    created_at: datetime
    case_count: int = Field(ge=1)
    passed_cases: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    mean_tool_selection_score: float = Field(ge=0, le=1)
    mean_argument_observability_score: float = Field(ge=0, le=1)
    mean_call_budget_score: float = Field(ge=0, le=1)
    mean_source_policy_score: float = Field(ge=0, le=1)
    mean_evidence_domain_score: float = Field(ge=0, le=1)
    mean_result_grounding_score: float = Field(ge=0, le=1)
    mean_recovery_safety_score: float = Field(ge=0, le=1)
    results: tuple[ToolTrajectoryCaseResult, ...]
    limitations: tuple[str, ...]


def _safe_snapshot_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"snapshot path escapes the dataset directory: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"snapshot does not exist: {relative}")
    if candidate.stat().st_size > 10_485_760:
        raise ValueError(f"snapshot exceeds the 10 MiB limit: {relative}")
    return candidate


def _ordered_subset(expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    position = 0
    for name in actual:
        if position < len(expected) and name == expected[position]:
            position += 1
    return position == len(expected)


def _duplicate_count(snapshot: DeepResearchSnapshot) -> int:
    fingerprints: dict[str, int] = {}
    for call in snapshot.tool_calls:
        if not call.args_hash:
            continue
        key = f"{call.task_id}|{call.tool_name}|{call.source_mode}|{call.args_hash}"
        fingerprints[key] = fingerprints.get(key, 0) + 1
    return sum(count - 1 for count in fingerprints.values() if count > 1)


def _unrecovered_failure_count(snapshot: DeepResearchSnapshot) -> int:
    ordered = sorted(snapshot.tool_calls, key=lambda item: (item.created_at, item.tool_call_id))
    failed_statuses = {"failed", "cancelled", "denied"}
    unrecovered = 0
    for failed in (item for item in ordered if item.status in failed_statuses):
        recovered = any(
            candidate.status == "completed"
            and candidate.task_id == failed.task_id
            and candidate.created_at > failed.created_at
            for candidate in ordered
        )
        unrecovered += not recovered
    return unrecovered


def _evaluate_case(case: ToolTrajectoryCase, snapshot: DeepResearchSnapshot) -> ToolTrajectoryCaseResult:
    expectation = case.expectation
    calls = tuple(sorted(snapshot.tool_calls, key=lambda item: (item.created_at, item.tool_call_id)))
    names = tuple(call.tool_name for call in calls)
    expected = expectation.expected_tool_names
    expected_present = (
        _ordered_subset(expected, names) if expectation.ordered else set(expected).issubset(set(names))
    )
    no_extras = expectation.allow_additional_tools or set(names).issubset(set(expected))
    tool_selection_score = float(expected_present and no_extras)
    argument_observability_score = (
        sum(bool(call.args_hash) for call in calls) / len(calls)
        if calls
        else float(expectation.min_tool_calls == 0)
    )
    call_budget_score = float(expectation.min_tool_calls <= len(calls) <= expectation.max_tool_calls)
    source_policy_score = float(
        expectation.required_source_mode is None
        or all(call.source_mode == expectation.required_source_mode for call in calls)
    )
    accepted_domains = tuple(domain.casefold().strip(".") for domain in expectation.accepted_evidence_domains)
    accepted_domain_evidence = sum(
        any(
            evidence.domain.casefold() == domain or evidence.domain.casefold().endswith(f".{domain}")
            for domain in accepted_domains
        )
        for evidence in snapshot.evidence
        if evidence.domain
    )
    evidence_domain_score = float(accepted_domain_evidence >= expectation.min_accepted_domain_evidence)
    evidence_counts: dict[str, int] = {}
    for evidence in snapshot.evidence:
        if evidence.tool_call_id:
            evidence_counts[evidence.tool_call_id] = evidence_counts.get(evidence.tool_call_id, 0) + 1
    completed = [call for call in calls if call.status == "completed"]
    grounded = all(
        (not expectation.require_result_present or call.result_present)
        and evidence_counts.get(call.tool_call_id, 0) >= expectation.min_evidence_per_completed_call
        for call in completed
    )
    result_grounding_score = float(
        grounded and (bool(completed) or (not calls and expectation.min_tool_calls == 0))
    )
    failed_count = sum(call.status in {"failed", "cancelled", "denied"} for call in calls)
    unrecovered = _unrecovered_failure_count(snapshot)
    recovered = failed_count - unrecovered
    duplicates = _duplicate_count(snapshot)
    recovery_safety_score = float(
        expectation.min_failed_tool_calls <= failed_count <= expectation.max_failed_tool_calls
        and recovered >= expectation.min_recovered_tool_failures
        and unrecovered <= expectation.max_unrecovered_tool_failures
        and duplicates <= expectation.max_duplicate_tool_calls
    )
    scores = (
        tool_selection_score,
        argument_observability_score,
        call_budget_score,
        source_policy_score,
        evidence_domain_score,
        result_grounding_score,
        recovery_safety_score,
    )
    findings: list[str] = []
    labels = (
        ("tool_selection", tool_selection_score),
        ("argument_observability", argument_observability_score),
        ("call_budget", call_budget_score),
        ("source_policy", source_policy_score),
        ("evidence_domain", evidence_domain_score),
        ("result_grounding", result_grounding_score),
        ("recovery_safety", recovery_safety_score),
    )
    findings.extend(name for name, score in labels if score < 1)
    return ToolTrajectoryCaseResult(
        case_id=case.case_id,
        trace_run_id=snapshot.run.run_id,
        passed=all(score == 1 for score in scores),
        tool_selection_score=tool_selection_score,
        argument_observability_score=round(argument_observability_score, 4),
        call_budget_score=call_budget_score,
        source_policy_score=source_policy_score,
        evidence_domain_score=evidence_domain_score,
        result_grounding_score=result_grounding_score,
        recovery_safety_score=recovery_safety_score,
        actual_tool_names=names,
        tool_call_count=len(calls),
        failed_tool_call_count=failed_count,
        recovered_tool_failure_count=recovered,
        unrecovered_tool_failure_count=unrecovered,
        duplicate_tool_call_count=duplicates,
        findings=tuple(findings),
    )


def evaluate_tool_trajectory_manifest(path: str | Path) -> ToolTrajectoryReport:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ToolTrajectoryManifest.model_validate(payload)
    results = tuple(
        _evaluate_case(
            case,
            DeepResearchSnapshot.model_validate(
                json.loads(
                    _safe_snapshot_path(manifest_path.parent, case.snapshot).read_text(encoding="utf-8")
                )
            ),
        )
        for case in manifest.cases
    )
    case_count = len(results)
    passed = sum(result.passed for result in results)
    return ToolTrajectoryReport(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        review_mode=manifest.review_mode,
        promotion_eligible=manifest.review_mode == "human-approved",
        created_at=datetime.now(UTC),
        case_count=case_count,
        passed_cases=passed,
        pass_rate=round(passed / case_count, 4),
        mean_tool_selection_score=round(
            sum(result.tool_selection_score for result in results) / case_count, 4
        ),
        mean_argument_observability_score=round(
            sum(result.argument_observability_score for result in results) / case_count, 4
        ),
        mean_call_budget_score=round(sum(result.call_budget_score for result in results) / case_count, 4),
        mean_source_policy_score=round(sum(result.source_policy_score for result in results) / case_count, 4),
        mean_evidence_domain_score=round(
            sum(result.evidence_domain_score for result in results) / case_count, 4
        ),
        mean_result_grounding_score=round(
            sum(result.result_grounding_score for result in results) / case_count, 4
        ),
        mean_recovery_safety_score=round(
            sum(result.recovery_safety_score for result in results) / case_count, 4
        ),
        results=results,
        limitations=(
            "Argument hashes prove stable recording, not semantic argument correctness.",
            "Generated-synthetic references are test scaffolding and are not human gold labels.",
            "Tool-trajectory scores do not establish final-answer factual correctness.",
        ),
    )


def load_tool_trajectory_annotation_queue(
    path: str | Path,
) -> ToolTrajectoryAnnotationQueue:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ToolTrajectoryAnnotationQueue.model_validate(payload)


def render_tool_trajectory_markdown(report: ToolTrajectoryReport) -> str:
    lines = [
        f"# Tool trajectory evaluation: {report.dataset_id} {report.dataset_version}",
        "",
        f"- Review mode: `{report.review_mode}`",
        f"- Promotion eligible: `{str(report.promotion_eligible).lower()}`",
        f"- Passed cases: {report.passed_cases}/{report.case_count}",
        f"- Pass rate: {report.pass_rate:.1%}",
        f"- Mean tool-selection score: {report.mean_tool_selection_score:.1%}",
        f"- Mean argument-observability score: {report.mean_argument_observability_score:.1%}",
        f"- Mean call-budget score: {report.mean_call_budget_score:.1%}",
        f"- Mean source-policy score: {report.mean_source_policy_score:.1%}",
        f"- Mean evidence-domain score: {report.mean_evidence_domain_score:.1%}",
        f"- Mean result-grounding score: {report.mean_result_grounding_score:.1%}",
        f"- Mean recovery-safety score: {report.mean_recovery_safety_score:.1%}",
        "",
        "## Cases",
        "",
        "| Case | Passed | Tools | Failed | Recovered | Unrecovered | Findings |",
        "|---|:---:|---|---:|---:|---:|---|",
    ]
    for result in report.results:
        lines.append(
            f"| {result.case_id} | {'yes' if result.passed else 'no'} | "
            f"{', '.join(result.actual_tool_names) or 'none'} | {result.failed_tool_call_count} | "
            f"{result.recovered_tool_failure_count} | {result.unrecovered_tool_failure_count} | "
            f"{', '.join(result.findings) or 'none'} |"
        )
    lines.extend(["", "## Evidence boundary", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"
