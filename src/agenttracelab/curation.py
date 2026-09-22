from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenttracelab.mining import HardCaseEntry, HardCaseMiningReport
from agenttracelab.models import FrozenModel


class HardCaseResolution(FrozenModel):
    case_id: str = Field(pattern=r"^hardcase_[a-f0-9]{24}$")
    decision: Literal["regression_guard", "training_candidate", "reject"]
    rationale_code: Literal[
        "confirmed_fix",
        "confirmed_failure",
        "duplicate",
        "insufficient_evidence",
        "out_of_scope",
        "privacy_risk",
    ]


class HardCaseResolutionSet(FrozenModel):
    schema_version: Literal["agenttracelab.hard-case-resolutions.v1"]
    adjudication_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    source_dataset_id: str = Field(min_length=1, max_length=128)
    source_dataset_version: str = Field(min_length=1, max_length=64)
    review_mode: Literal["human", "generated-synthetic"]
    reviewer_id: str = Field(min_length=1, max_length=128)
    reviewed_at: datetime
    require_complete: bool = True
    split_seed: str = Field(default="agenttracelab", min_length=1, max_length=128)
    train_ratio: float = Field(default=0.7, ge=0, le=1)
    validation_ratio: float = Field(default=0.15, ge=0, le=1)
    test_ratio: float = Field(default=0.15, ge=0, le=1)
    resolutions: tuple[HardCaseResolution, ...] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_ratios(self) -> HardCaseResolutionSet:
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("train, validation, and test ratios must sum to 1.0")
        return self


class CuratedCase(FrozenModel):
    source: HardCaseEntry
    decision: Literal["regression_guard", "training_candidate"]
    split: Literal["train", "validation", "test", "regression"]
    rationale_code: str


class CuratedChallengeSet(FrozenModel):
    schema_version: Literal["agenttracelab.curated-challenge-set.v1"] = (
        "agenttracelab.curated-challenge-set.v1"
    )
    adjudication_id: str
    version: str
    created_at: datetime
    source_dataset_id: str
    source_dataset_version: str
    review_mode: Literal["human", "generated-synthetic"]
    reviewer_id: str
    reviewed_at: datetime
    decision_counts: dict[str, int]
    split_counts: dict[str, int]
    included_case_count: int = Field(ge=0)
    rejected_case_ids: tuple[str, ...]
    cases: tuple[CuratedCase, ...]
    limitations: tuple[str, ...]


def load_hard_case_resolution_set(path: str | Path) -> HardCaseResolutionSet:
    resolution_path = Path(path).resolve()
    if resolution_path.stat().st_size > 4_194_304:
        raise ValueError("hard-case resolution set exceeds the 4 MiB limit")
    payload = json.loads(resolution_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hard-case resolution set must contain a JSON object")
    return HardCaseResolutionSet.model_validate(payload)


def load_hard_case_mining_report(path: str | Path) -> HardCaseMiningReport:
    report_path = Path(path).resolve()
    if report_path.stat().st_size > 16_777_216:
        raise ValueError("hard-case mining report exceeds the 16 MiB limit")
    return HardCaseMiningReport.model_validate_json(report_path.read_text(encoding="utf-8"))


def _training_split(task_id: str, resolutions: HardCaseResolutionSet) -> str:
    digest = hashlib.sha256(f"{resolutions.split_seed}:{task_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < resolutions.train_ratio:
        return "train"
    if value < resolutions.train_ratio + resolutions.validation_ratio:
        return "validation"
    return "test"


def curate_hard_cases(
    queue: HardCaseMiningReport,
    resolutions: HardCaseResolutionSet,
) -> CuratedChallengeSet:
    if (
        resolutions.source_dataset_id != queue.dataset_id
        or resolutions.source_dataset_version != queue.dataset_version
    ):
        raise ValueError("resolution source does not match the hard-case queue")
    case_by_id = {case.case_id: case for case in queue.cases}
    if len(case_by_id) != len(queue.cases):
        raise ValueError("hard-case queue contains duplicate case IDs")
    resolution_ids = [resolution.case_id for resolution in resolutions.resolutions]
    duplicate_ids = sorted(case_id for case_id, count in Counter(resolution_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError("duplicate hard-case resolutions: " + ", ".join(duplicate_ids))
    unknown_ids = sorted(set(resolution_ids) - set(case_by_id))
    if unknown_ids:
        raise ValueError("resolutions reference unknown cases: " + ", ".join(unknown_ids))
    missing_ids = sorted(set(case_by_id) - set(resolution_ids))
    if resolutions.require_complete and missing_ids:
        raise ValueError("complete adjudication is missing cases: " + ", ".join(missing_ids))

    curated: list[CuratedCase] = []
    rejected: list[str] = []
    for resolution in resolutions.resolutions:
        source = case_by_id[resolution.case_id]
        if resolution.decision == "reject":
            rejected.append(resolution.case_id)
            continue
        if resolution.decision == "regression_guard":
            if source.case_kind != "resolved_baseline_failure":
                raise ValueError(f"regression_guard requires a resolved baseline failure: {source.case_id}")
            if resolution.rationale_code != "confirmed_fix":
                raise ValueError(f"regression_guard requires confirmed_fix rationale: {source.case_id}")
            split = "regression"
        else:
            if resolution.rationale_code != "confirmed_failure":
                raise ValueError(f"training_candidate requires confirmed_failure rationale: {source.case_id}")
            split = _training_split(source.task_id, resolutions)
        curated.append(
            CuratedCase(
                source=source,
                decision=resolution.decision,
                split=split,
                rationale_code=resolution.rationale_code,
            )
        )

    decision_counts = Counter(resolution.decision for resolution in resolutions.resolutions)
    split_counts = Counter(case.split for case in curated)
    return CuratedChallengeSet(
        adjudication_id=resolutions.adjudication_id,
        version=resolutions.version,
        created_at=datetime.now(UTC),
        source_dataset_id=queue.dataset_id,
        source_dataset_version=queue.dataset_version,
        review_mode=resolutions.review_mode,
        reviewer_id=resolutions.reviewer_id,
        reviewed_at=resolutions.reviewed_at,
        decision_counts=dict(sorted(decision_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        included_case_count=len(curated),
        rejected_case_ids=tuple(sorted(rejected)),
        cases=tuple(curated),
        limitations=(
            "Curated entries remain content-free trace pointers, not materialized training examples.",
            "A grouped deterministic split reduces task leakage but does not detect semantic duplicates.",
            "Generated-synthetic review mode is test evidence and must not be presented as human labeling.",
        ),
    )


def curate_hard_cases_files(
    queue_path: str | Path,
    resolutions_path: str | Path,
) -> CuratedChallengeSet:
    return curate_hard_cases(
        load_hard_case_mining_report(queue_path),
        load_hard_case_resolution_set(resolutions_path),
    )
