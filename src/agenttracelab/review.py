from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from agenttracelab.judging import (
    JudgeDimensionName,
    JudgeVerdict,
    LlmJudgeReport,
)
from agenttracelab.models import EvaluationReport, FrozenModel


class ReviewState(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"


class ReviewPriority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class ReviewReason(StrEnum):
    DETERMINISTIC_JUDGE_CONFLICT = "deterministic_judge_conflict"
    JUDGE_DEFERRED = "judge_deferred"
    LOW_DIMENSION_SCORE = "low_dimension_score"
    FORCED_SAMPLE = "forced_sample"


class ReviewResolutionInput(FrozenModel):
    reviewer_id: str = Field(min_length=1, max_length=256)
    verdict: JudgeVerdict
    dimensions: dict[JudgeDimensionName, int]
    notes: str = Field(default="", max_length=4_000)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ReviewResolutionInput:
        if set(self.dimensions) != set(JudgeDimensionName):
            raise ValueError("review resolution must score every judge dimension exactly once")
        if any(score < 0 or score > 4 for score in self.dimensions.values()):
            raise ValueError("review dimension scores must be between 0 and 4")
        return self


class ReviewResolution(ReviewResolutionInput):
    resolved_at: datetime


class ReviewItem(FrozenModel):
    schema_version: Literal["agenttracelab.review-item.v1"] = "agenttracelab.review-item.v1"
    review_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=256)
    deterministic_evaluation_id: str = Field(min_length=1, max_length=256)
    judge_report_id: str = Field(min_length=1, max_length=256)
    priority: ReviewPriority
    reasons: tuple[ReviewReason, ...]
    state: ReviewState
    created_at: datetime
    resolution: ReviewResolution | None = None

    @model_validator(mode="after")
    def validate_state(self) -> ReviewItem:
        if not self.reasons:
            raise ValueError("review item must contain at least one reason")
        if self.state == ReviewState.PENDING and self.resolution is not None:
            raise ValueError("pending review item cannot contain a resolution")
        if self.state == ReviewState.RESOLVED and self.resolution is None:
            raise ValueError("resolved review item must contain a resolution")
        return self


def select_for_review(
    deterministic: EvaluationReport,
    judge: LlmJudgeReport,
    *,
    force: bool = False,
) -> ReviewItem | None:
    if deterministic.trace_id != judge.trace_id:
        raise ValueError("deterministic and judge reports must refer to the same trace")
    if deterministic.evaluation_id != judge.deterministic_evaluation_id:
        raise ValueError("judge report was not produced from the supplied deterministic evaluation")

    reasons: list[ReviewReason] = []
    judge_passed = judge.verdict == JudgeVerdict.PASS
    if deterministic.passed != judge_passed and judge.verdict != JudgeVerdict.REVIEW:
        reasons.append(ReviewReason.DETERMINISTIC_JUDGE_CONFLICT)
    if judge.verdict == JudgeVerdict.REVIEW:
        reasons.append(ReviewReason.JUDGE_DEFERRED)
    if any(item.score <= 1 for item in judge.dimensions):
        reasons.append(ReviewReason.LOW_DIMENSION_SCORE)
    if force and not reasons:
        reasons.append(ReviewReason.FORCED_SAMPLE)
    if not reasons:
        return None

    if ReviewReason.DETERMINISTIC_JUDGE_CONFLICT in reasons and not deterministic.passed:
        priority = ReviewPriority.CRITICAL
    elif ReviewReason.DETERMINISTIC_JUDGE_CONFLICT in reasons:
        priority = ReviewPriority.HIGH
    else:
        priority = ReviewPriority.MEDIUM
    return ReviewItem(
        review_id=f"review_{uuid4().hex}",
        trace_id=deterministic.trace_id,
        deterministic_evaluation_id=deterministic.evaluation_id,
        judge_report_id=judge.judge_report_id,
        priority=priority,
        reasons=tuple(dict.fromkeys(reasons)),
        state=ReviewState.PENDING,
        created_at=datetime.now(UTC),
    )


def resolve_review(item: ReviewItem, resolution: ReviewResolutionInput) -> ReviewItem:
    if item.state != ReviewState.PENDING:
        raise ValueError("review item is already resolved")
    payload = item.model_dump()
    payload.update(
        state=ReviewState.RESOLVED,
        resolution=ReviewResolution(
            **resolution.model_dump(),
            resolved_at=datetime.now(UTC),
        ),
    )
    return ReviewItem.model_validate(payload)
