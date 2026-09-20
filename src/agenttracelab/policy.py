from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from agenttracelab.batch import EvidenceMode, WasmHatchBatchReport
from agenttracelab.failures import FailureCategory
from agenttracelab.models import FrozenModel

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class PromotionPolicy(FrozenModel):
    schema_version: Literal["agenttracelab.promotion-policy.v1"]
    policy_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    min_run_count: int = Field(default=1, ge=1, le=100)
    min_pass_rate: float = Field(default=100.0, ge=0, le=100)
    max_failed_runs: int = Field(default=0, ge=0, le=100)
    allowed_evidence_modes: tuple[EvidenceMode, ...] = (
        "recorded_local",
        "live_provider",
    )
    forbidden_failure_categories: tuple[FailureCategory, ...] = (
        FailureCategory.PRIVACY,
        FailureCategory.SAFETY_AUTHORIZATION,
        FailureCategory.TRACE_CONTRACT,
    )
    max_conflicts: int = Field(default=0, ge=0)
    max_uncertain_outcomes: int = Field(default=0, ge=0)


class PromotionGateReport(FrozenModel):
    schema_version: Literal["agenttracelab.promotion-gate.v1"] = "agenttracelab.promotion-gate.v1"
    batch_id: str
    batch_version: str
    policy_id: str
    created_at: datetime
    decision: Literal["promote", "hold"]
    reasons: tuple[str, ...]


def load_promotion_policy(path: str | Path) -> PromotionPolicy:
    policy_path = Path(path).resolve()
    if not policy_path.is_file():
        raise ValueError(f"promotion policy does not exist: {policy_path.name}")
    if policy_path.stat().st_size > 65_536:
        raise ValueError("promotion policy exceeds the 64 KiB limit")
    value = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("promotion policy must contain a JSON object")
    return PromotionPolicy.model_validate(value)


def apply_promotion_policy(
    report: WasmHatchBatchReport,
    policy: PromotionPolicy,
) -> PromotionGateReport:
    reasons: list[str] = []
    if report.run_count < policy.min_run_count:
        reasons.append(f"run_count {report.run_count} is below required {policy.min_run_count}")
    if report.pass_rate < policy.min_pass_rate:
        reasons.append(f"pass_rate {report.pass_rate:.2f}% is below required {policy.min_pass_rate:.2f}%")
    if report.failed_runs > policy.max_failed_runs:
        reasons.append(f"failed_runs {report.failed_runs} exceeds allowed {policy.max_failed_runs}")
    if report.evidence_mode not in policy.allowed_evidence_modes:
        reasons.append(f"evidence_mode {report.evidence_mode} is not allowed by this policy")
    for category in policy.forbidden_failure_categories:
        count = report.failure_counts.get(category, 0)
        if count:
            reasons.append(f"forbidden failure category {category.value} occurred {count} time(s)")
    conflicts = report.derived_metric_totals.get("conflicts", 0)
    if conflicts > policy.max_conflicts:
        reasons.append(f"conflicts {conflicts} exceeds allowed {policy.max_conflicts}")
    uncertain = report.derived_metric_totals.get("uncertainOutcomes", 0)
    if uncertain > policy.max_uncertain_outcomes:
        reasons.append(f"uncertain outcomes {uncertain} exceeds allowed {policy.max_uncertain_outcomes}")
    return PromotionGateReport(
        batch_id=report.batch_id,
        batch_version=report.batch_version,
        policy_id=policy.policy_id,
        created_at=datetime.now(UTC),
        decision="hold" if reasons else "promote",
        reasons=tuple(reasons),
    )


def render_gate_markdown(
    report: WasmHatchBatchReport,
    gate: PromotionGateReport | None,
) -> str:
    decision = gate.decision.upper() if gate else ("PASS" if report.passed else "FAIL")
    lines = [
        f"# AgentTraceLab: {decision}",
        "",
        f"- Batch: `{report.batch_id}` version `{report.batch_version}`",
        f"- Evidence mode: `{report.evidence_mode}`",
        f"- Runs: {report.run_count} ({report.passed_runs} passed, {report.failed_runs} failed)",
        f"- Pass rate: {report.pass_rate:.2f}%",
    ]
    if report.failure_counts:
        lines.extend(["", "## Failure categories", ""])
        lines.extend(f"- `{category.value}`: {count}" for category, count in report.failure_counts.items())
    if gate and gate.reasons:
        lines.extend(["", "## Promotion blockers", ""])
        lines.extend(f"- {reason}" for reason in gate.reasons)
    lines.extend(["", f"> {report.evidence_notice}", ""])
    return "\n".join(lines)
