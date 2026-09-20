from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenttracelab.judging import JudgeDimensionName, JudgeVerdict
from agenttracelab.models import FrozenModel


class CalibrationLabel(FrozenModel):
    verdict: JudgeVerdict
    dimensions: dict[JudgeDimensionName, int]

    @model_validator(mode="after")
    def validate_dimensions(self) -> CalibrationLabel:
        if set(self.dimensions) != set(JudgeDimensionName):
            raise ValueError("calibration label must score every judge dimension exactly once")
        if any(score < 0 or score > 4 for score in self.dimensions.values()):
            raise ValueError("calibration dimension scores must be between 0 and 4")
        return self


class HumanCalibrationLabel(CalibrationLabel):
    annotator_count: int = Field(ge=1)
    notes: str = Field(default="", max_length=2_000)


class RecordedJudgeRating(CalibrationLabel):
    judge_key: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=128)
    judge_report_id: str | None = Field(default=None, max_length=256)


class CalibrationCaseProvenance(FrozenModel):
    source_kind: Literal["human_review"] = "human_review"
    review_id: str = Field(min_length=1, max_length=256)
    deterministic_evaluation_id: str = Field(min_length=1, max_length=256)
    judge_report_id: str = Field(min_length=1, max_length=256)
    review_reasons: tuple[str, ...]
    resolved_at: datetime


class CalibrationCase(FrozenModel):
    case_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=256)
    human: HumanCalibrationLabel
    judges: tuple[RecordedJudgeRating, ...]
    provenance: CalibrationCaseProvenance | None = None

    @model_validator(mode="after")
    def validate_judge_keys(self) -> CalibrationCase:
        keys = [item.judge_key for item in self.judges]
        if len(keys) != len(set(keys)):
            raise ValueError("a calibration case cannot contain duplicate judge keys")
        return self


class CalibrationPolicy(FrozenModel):
    min_cases: int = Field(default=3, ge=1)
    min_coverage_percent: float = Field(default=100.0, ge=0, le=100)
    min_exact_agreement_percent: float = Field(default=75.0, ge=0, le=100)
    min_cohen_kappa: float = Field(default=0.6, ge=-1, le=1)
    max_review_rate_percent: float = Field(default=50.0, ge=0, le=100)
    max_dimension_mae: float = Field(default=0.75, ge=0, le=4)


class CalibrationManifest(FrozenModel):
    schema_version: Literal["agenttracelab.judge-calibration-manifest.v1"] = (
        "agenttracelab.judge-calibration-manifest.v1"
    )
    dataset_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=2_000)
    policy: CalibrationPolicy = CalibrationPolicy()
    cases: tuple[CalibrationCase, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> CalibrationManifest:
        case_ids = [case.case_id for case in self.cases]
        if not case_ids:
            raise ValueError("calibration manifest must contain at least one case")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("calibration case IDs must be unique")
        return self


class JudgeCalibrationMetrics(FrozenModel):
    judge_key: str
    provider: str
    model: str
    prompt_version: str
    total_cases: int
    covered_cases: int
    coverage_percent: float
    exact_agreement_percent: float
    cohen_kappa: float
    review_rate_percent: float
    dimension_mae: float
    confusion_matrix: dict[str, dict[str, int]]
    violations: tuple[str, ...]
    passed: bool


class PairwiseJudgeAgreement(FrozenModel):
    left_judge_key: str
    right_judge_key: str
    overlapping_cases: int
    exact_agreement_percent: float
    cohen_kappa: float


class JudgeCalibrationReport(FrozenModel):
    schema_version: Literal["agenttracelab.judge-calibration.v1"] = "agenttracelab.judge-calibration.v1"
    dataset_id: str
    dataset_version: str
    created_at: datetime
    total_cases: int
    passed: bool
    judges: tuple[JudgeCalibrationMetrics, ...]
    pairwise: tuple[PairwiseJudgeAgreement, ...]


def _percentage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _cohen_kappa(left: list[JudgeVerdict], right: list[JudgeVerdict]) -> float:
    if len(left) != len(right):
        raise ValueError("Cohen's kappa inputs must have the same length")
    if not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right) for label in JudgeVerdict
    )
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def _confusion_matrix(
    human: list[JudgeVerdict],
    predicted: list[JudgeVerdict],
) -> dict[str, dict[str, int]]:
    matrix = {expected.value: {actual.value: 0 for actual in JudgeVerdict} for expected in JudgeVerdict}
    for expected, actual in zip(human, predicted, strict=True):
        matrix[expected.value][actual.value] += 1
    return matrix


def _violations(
    policy: CalibrationPolicy,
    *,
    total_cases: int,
    covered_cases: int,
    coverage_percent: float,
    exact_agreement_percent: float,
    cohen_kappa: float,
    review_rate_percent: float,
    dimension_mae: float,
) -> tuple[str, ...]:
    violations: list[str] = []
    if total_cases < policy.min_cases:
        violations.append(f"dataset has {total_cases} cases; requires at least {policy.min_cases}")
    if covered_cases == 0:
        violations.append("judge has no recorded calibration cases")
        return tuple(violations)
    if coverage_percent < policy.min_coverage_percent:
        violations.append(f"coverage {coverage_percent:.2f}% is below {policy.min_coverage_percent:.2f}%")
    if exact_agreement_percent < policy.min_exact_agreement_percent:
        violations.append(
            "exact agreement "
            f"{exact_agreement_percent:.2f}% is below {policy.min_exact_agreement_percent:.2f}%"
        )
    if cohen_kappa < policy.min_cohen_kappa:
        violations.append(f"Cohen's kappa {cohen_kappa:.4f} is below {policy.min_cohen_kappa:.4f}")
    if review_rate_percent > policy.max_review_rate_percent:
        violations.append(
            f"review rate {review_rate_percent:.2f}% exceeds {policy.max_review_rate_percent:.2f}%"
        )
    if dimension_mae > policy.max_dimension_mae:
        violations.append(f"dimension MAE {dimension_mae:.4f} exceeds {policy.max_dimension_mae:.4f}")
    return tuple(violations)


def analyze_judge_calibration(manifest: CalibrationManifest) -> JudgeCalibrationReport:
    ratings_by_judge: dict[str, dict[str, RecordedJudgeRating]] = defaultdict(dict)
    metadata_by_judge: dict[str, tuple[str, str, str]] = {}
    cases_by_id = {case.case_id: case for case in manifest.cases}
    for case in manifest.cases:
        for rating in case.judges:
            metadata = (rating.provider, rating.model, rating.prompt_version)
            previous = metadata_by_judge.setdefault(rating.judge_key, metadata)
            if previous != metadata:
                raise ValueError(f"judge key {rating.judge_key} has inconsistent provenance")
            ratings_by_judge[rating.judge_key][case.case_id] = rating

    judge_metrics: list[JudgeCalibrationMetrics] = []
    for judge_key in sorted(ratings_by_judge):
        ratings = ratings_by_judge[judge_key]
        covered_case_ids = sorted(ratings)
        human_verdicts = [cases_by_id[case_id].human.verdict for case_id in covered_case_ids]
        judge_verdicts = [ratings[case_id].verdict for case_id in covered_case_ids]
        exact = sum(
            human == predicted for human, predicted in zip(human_verdicts, judge_verdicts, strict=True)
        )
        absolute_errors = [
            abs(cases_by_id[case_id].human.dimensions[dimension] - ratings[case_id].dimensions[dimension])
            for case_id in covered_case_ids
            for dimension in JudgeDimensionName
        ]
        coverage = _percentage(len(covered_case_ids), len(manifest.cases))
        agreement = _percentage(exact, len(covered_case_ids))
        kappa = _cohen_kappa(human_verdicts, judge_verdicts)
        review_rate = _percentage(
            sum(verdict == JudgeVerdict.REVIEW for verdict in judge_verdicts),
            len(judge_verdicts),
        )
        dimension_mae = round(sum(absolute_errors) / len(absolute_errors), 4)
        violations = _violations(
            manifest.policy,
            total_cases=len(manifest.cases),
            covered_cases=len(covered_case_ids),
            coverage_percent=coverage,
            exact_agreement_percent=agreement,
            cohen_kappa=kappa,
            review_rate_percent=review_rate,
            dimension_mae=dimension_mae,
        )
        provider, model, prompt_version = metadata_by_judge[judge_key]
        judge_metrics.append(
            JudgeCalibrationMetrics(
                judge_key=judge_key,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                total_cases=len(manifest.cases),
                covered_cases=len(covered_case_ids),
                coverage_percent=coverage,
                exact_agreement_percent=agreement,
                cohen_kappa=kappa,
                review_rate_percent=review_rate,
                dimension_mae=dimension_mae,
                confusion_matrix=_confusion_matrix(human_verdicts, judge_verdicts),
                violations=violations,
                passed=not violations,
            )
        )

    pairwise: list[PairwiseJudgeAgreement] = []
    for left_key, right_key in combinations(sorted(ratings_by_judge), 2):
        overlap = sorted(set(ratings_by_judge[left_key]) & set(ratings_by_judge[right_key]))
        left_verdicts = [ratings_by_judge[left_key][case_id].verdict for case_id in overlap]
        right_verdicts = [ratings_by_judge[right_key][case_id].verdict for case_id in overlap]
        exact = sum(left == right for left, right in zip(left_verdicts, right_verdicts, strict=True))
        pairwise.append(
            PairwiseJudgeAgreement(
                left_judge_key=left_key,
                right_judge_key=right_key,
                overlapping_cases=len(overlap),
                exact_agreement_percent=_percentage(exact, len(overlap)),
                cohen_kappa=_cohen_kappa(left_verdicts, right_verdicts),
            )
        )

    return JudgeCalibrationReport(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        created_at=datetime.now(UTC),
        total_cases=len(manifest.cases),
        passed=bool(judge_metrics) and all(item.passed for item in judge_metrics),
        judges=tuple(judge_metrics),
        pairwise=tuple(pairwise),
    )


def load_calibration_manifest(path: str | Path) -> CalibrationManifest:
    return CalibrationManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def render_calibration_markdown(report: JudgeCalibrationReport) -> str:
    lines = [
        f"# Judge calibration: {report.dataset_id} ({report.dataset_version})",
        "",
        f"Gate: **{'PASS' if report.passed else 'FAIL'}** | Cases: {report.total_cases}",
        "",
        "| Judge | Coverage | Exact agreement | Cohen's kappa | Review rate | Dimension MAE | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for judge in report.judges:
        lines.append(
            f"| {judge.judge_key} | {judge.coverage_percent:.2f}% | "
            f"{judge.exact_agreement_percent:.2f}% | {judge.cohen_kappa:.4f} | "
            f"{judge.review_rate_percent:.2f}% | {judge.dimension_mae:.4f} | "
            f"{'PASS' if judge.passed else 'FAIL'} |"
        )
        if judge.violations:
            lines.extend(f"  - {judge.judge_key}: {violation}" for violation in judge.violations)
    if report.pairwise:
        lines.extend(
            [
                "",
                "## Pairwise judge agreement",
                "",
                "| Left | Right | Overlap | Exact agreement | Cohen's kappa |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {item.left_judge_key} | {item.right_judge_key} | {item.overlapping_cases} | "
            f"{item.exact_agreement_percent:.2f}% | {item.cohen_kappa:.4f} |"
            for item in report.pairwise
        )
    return "\n".join(lines) + "\n"
