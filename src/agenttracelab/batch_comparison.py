from __future__ import annotations

from datetime import UTC, datetime
from statistics import fmean
from typing import Literal

from pydantic import Field

from agenttracelab.batch import WasmHatchBatchReport
from agenttracelab.comparison import compare_evaluations
from agenttracelab.models import FrozenModel


class ScenarioComparison(FrozenModel):
    scenario_id: str
    baseline_trace_id: str
    candidate_trace_id: str
    baseline_passed: bool
    candidate_passed: bool
    baseline_score: float = Field(ge=0, le=100)
    candidate_score: float = Field(ge=0, le=100)
    score_delta: float
    check_regressions: tuple[str, ...]
    check_improvements: tuple[str, ...]
    baseline_process_reward: float = Field(default=0.0, ge=0, le=1)
    candidate_process_reward: float = Field(default=0.0, ge=0, le=1)
    process_reward_delta: float = Field(default=0.0, ge=-1, le=1)
    baseline_recommended_reward: float = Field(default=0.0, ge=0, le=1)
    candidate_recommended_reward: float = Field(default=0.0, ge=0, le=1)
    recommended_reward_delta: float = Field(default=0.0, ge=-1, le=1)
    baseline_safety_blocked: bool = False
    candidate_safety_blocked: bool = False


class BatchComparisonReport(FrozenModel):
    schema_version: Literal["agenttracelab.batch-comparison.v2"] = "agenttracelab.batch-comparison.v2"
    baseline_batch_id: str
    baseline_version: str
    candidate_batch_id: str
    candidate_version: str
    created_at: datetime
    paired_scenario_count: int = Field(ge=0)
    baseline_only_scenarios: tuple[str, ...]
    candidate_only_scenarios: tuple[str, ...]
    run_regressions: tuple[str, ...]
    run_improvements: tuple[str, ...]
    check_regression_count: int = Field(ge=0)
    check_improvement_count: int = Field(ge=0)
    mean_paired_score_delta: float
    mean_paired_process_reward_delta: float = Field(default=0.0, ge=-1, le=1)
    mean_paired_recommended_reward_delta: float = Field(default=0.0, ge=-1, le=1)
    safety_regression_scenarios: tuple[str, ...] = ()
    safety_unblocked_scenarios: tuple[str, ...] = ()
    evidence_modes_match: bool
    recommendation: Literal["promote", "hold"]
    reasons: tuple[str, ...]
    comparison_notice: str
    scenarios: tuple[ScenarioComparison, ...]


def compare_wasmhatch_batches(
    baseline: WasmHatchBatchReport,
    candidate: WasmHatchBatchReport,
) -> BatchComparisonReport:
    baseline_results = {result.scenario_id: result for result in baseline.results}
    candidate_results = {result.scenario_id: result for result in candidate.results}
    baseline_ids = set(baseline_results)
    candidate_ids = set(candidate_results)
    paired_ids = sorted(baseline_ids & candidate_ids)
    baseline_only = tuple(sorted(baseline_ids - candidate_ids))
    candidate_only = tuple(sorted(candidate_ids - baseline_ids))

    scenarios: list[ScenarioComparison] = []
    for scenario_id in paired_ids:
        baseline_result = baseline_results[scenario_id]
        candidate_result = candidate_results[scenario_id]
        comparison = compare_evaluations(
            baseline_result.evaluation,
            candidate_result.evaluation,
        )
        scenarios.append(
            ScenarioComparison(
                scenario_id=scenario_id,
                baseline_trace_id=baseline_result.trace_id,
                candidate_trace_id=candidate_result.trace_id,
                baseline_passed=baseline_result.evaluation.passed,
                candidate_passed=candidate_result.evaluation.passed,
                baseline_score=baseline_result.evaluation.score,
                candidate_score=candidate_result.evaluation.score,
                score_delta=comparison.score_delta,
                check_regressions=comparison.regressions,
                check_improvements=comparison.improvements,
                baseline_process_reward=baseline_result.training_reward.process_reward,
                candidate_process_reward=candidate_result.training_reward.process_reward,
                process_reward_delta=round(
                    candidate_result.training_reward.process_reward
                    - baseline_result.training_reward.process_reward,
                    4,
                ),
                baseline_recommended_reward=baseline_result.training_reward.recommended_reward,
                candidate_recommended_reward=candidate_result.training_reward.recommended_reward,
                recommended_reward_delta=round(
                    candidate_result.training_reward.recommended_reward
                    - baseline_result.training_reward.recommended_reward,
                    4,
                ),
                baseline_safety_blocked=baseline_result.training_reward.safety_blocked,
                candidate_safety_blocked=candidate_result.training_reward.safety_blocked,
            )
        )

    run_regressions = tuple(
        result.scenario_id for result in scenarios if result.baseline_passed and not result.candidate_passed
    )
    run_improvements = tuple(
        result.scenario_id for result in scenarios if not result.baseline_passed and result.candidate_passed
    )
    check_regression_count = sum(len(result.check_regressions) for result in scenarios)
    check_improvement_count = sum(len(result.check_improvements) for result in scenarios)
    mean_score_delta = round(fmean(result.score_delta for result in scenarios), 2) if scenarios else 0.0
    mean_process_reward_delta = (
        round(fmean(result.process_reward_delta for result in scenarios), 4) if scenarios else 0.0
    )
    mean_recommended_reward_delta = (
        round(fmean(result.recommended_reward_delta for result in scenarios), 4) if scenarios else 0.0
    )
    safety_regressions = tuple(
        result.scenario_id
        for result in scenarios
        if not result.baseline_safety_blocked and result.candidate_safety_blocked
    )
    safety_unblocked = tuple(
        result.scenario_id
        for result in scenarios
        if result.baseline_safety_blocked and not result.candidate_safety_blocked
    )
    evidence_modes_match = baseline.evidence_mode == candidate.evidence_mode

    reasons: list[str] = []
    if not paired_ids:
        reasons.append("baseline and candidate contain no paired scenarios")
    if baseline_only:
        reasons.append(f"candidate is missing {len(baseline_only)} baseline scenario(s)")
    if run_regressions:
        reasons.append(f"{len(run_regressions)} paired scenario(s) regressed from pass to fail")
    if check_regression_count:
        reasons.append(f"{check_regression_count} deterministic check regression(s) detected")
    if safety_regressions:
        reasons.append(f"{len(safety_regressions)} paired scenario(s) introduced a safety block")
    if not candidate.passed:
        reasons.append("candidate batch contains one or more failed runs")
    if not evidence_modes_match:
        reasons.append("baseline and candidate use different evidence modes")

    return BatchComparisonReport(
        baseline_batch_id=baseline.batch_id,
        baseline_version=baseline.batch_version,
        candidate_batch_id=candidate.batch_id,
        candidate_version=candidate.batch_version,
        created_at=datetime.now(UTC),
        paired_scenario_count=len(paired_ids),
        baseline_only_scenarios=baseline_only,
        candidate_only_scenarios=candidate_only,
        run_regressions=run_regressions,
        run_improvements=run_improvements,
        check_regression_count=check_regression_count,
        check_improvement_count=check_improvement_count,
        mean_paired_score_delta=mean_score_delta,
        mean_paired_process_reward_delta=mean_process_reward_delta,
        mean_paired_recommended_reward_delta=mean_recommended_reward_delta,
        safety_regression_scenarios=safety_regressions,
        safety_unblocked_scenarios=safety_unblocked,
        evidence_modes_match=evidence_modes_match,
        recommendation="hold" if reasons else "promote",
        reasons=tuple(reasons),
        comparison_notice=(
            "This is an exact paired comparison of deterministic checks. The mean delta is descriptive; "
            "no statistical significance is claimed. Candidate-only scenarios have no paired baseline."
        ),
        scenarios=tuple(scenarios),
    )


def render_batch_comparison_markdown(report: BatchComparisonReport) -> str:
    lines = [
        f"# AgentTraceLab comparison: {report.recommendation.upper()}",
        "",
        f"- Baseline: `{report.baseline_batch_id}` version `{report.baseline_version}`",
        f"- Candidate: `{report.candidate_batch_id}` version `{report.candidate_version}`",
        f"- Paired scenarios: {report.paired_scenario_count}",
        f"- Mean paired score delta: {report.mean_paired_score_delta:+.2f}",
        f"- Mean process reward delta: {report.mean_paired_process_reward_delta:+.4f}",
        f"- Mean recommended reward delta: {report.mean_paired_recommended_reward_delta:+.4f}",
        f"- Safety regressions / unblocked: {len(report.safety_regression_scenarios)} / "
        f"{len(report.safety_unblocked_scenarios)}",
        f"- Run regressions / improvements: {len(report.run_regressions)} / {len(report.run_improvements)}",
        (
            "- Check regressions / improvements: "
            f"{report.check_regression_count} / {report.check_improvement_count}"
        ),
    ]
    if report.baseline_only_scenarios:
        lines.extend(
            [
                "",
                "## Missing baseline coverage",
                "",
                *(f"- `{scenario}`" for scenario in report.baseline_only_scenarios),
            ]
        )
    if report.candidate_only_scenarios:
        lines.extend(
            [
                "",
                "## Candidate-only scenarios",
                "",
                *(f"- `{scenario}`" for scenario in report.candidate_only_scenarios),
            ]
        )
    if report.reasons:
        lines.extend(["", "## Promotion blockers", "", *(f"- {reason}" for reason in report.reasons)])
    lines.extend(["", "## Paired results", ""])
    lines.extend(
        (
            f"- `{item.scenario_id}`: {item.baseline_score:.2f} → {item.candidate_score:.2f} "
            f"({item.score_delta:+.2f}); process_reward={item.process_reward_delta:+.4f}, "
            f"recommended_reward={item.recommended_reward_delta:+.4f}; "
            f"regressions={len(item.check_regressions)}, "
            f"improvements={len(item.check_improvements)}"
        )
        for item in report.scenarios
    )
    lines.extend(["", f"> {report.comparison_notice}", ""])
    return "\n".join(lines)
