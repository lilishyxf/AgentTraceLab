from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agenttracelab.calibration import _cohen_kappa
from agenttracelab.judging import JudgeDimensionName, JudgeVerdict
from agenttracelab.models import FrozenModel

MAX_MANIFEST_BYTES = 1024 * 1024


class CalibrationSplit(StrEnum):
    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


class JudgeOutcomeStatus(StrEnum):
    SCORED = "scored"
    ABSTAINED = "abstained"
    SCORING_ERROR = "scoring_error"


class JudgeIdentity(FrozenModel):
    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=128)


def _validate_dimensions(dimensions: dict[JudgeDimensionName, int]) -> None:
    if set(dimensions) != set(JudgeDimensionName):
        raise ValueError("label must score every judge dimension exactly once")
    if any(score < 0 or score > 4 for score in dimensions.values()):
        raise ValueError("dimension scores must be between 0 and 4")


class HumanAnnotation(FrozenModel):
    annotation_id: str = Field(min_length=1, max_length=256)
    annotator_id: str = Field(min_length=1, max_length=256)
    verdict: JudgeVerdict
    dimensions: dict[JudgeDimensionName, int]
    notes: str = Field(default="", max_length=2_000)
    created_at: datetime

    @model_validator(mode="after")
    def validate_dimensions(self) -> HumanAnnotation:
        _validate_dimensions(self.dimensions)
        return self


class AdjudicatedLabel(FrozenModel):
    adjudicator_id: str = Field(min_length=1, max_length=256)
    source_annotation_ids: tuple[str, ...]
    verdict: JudgeVerdict
    dimensions: dict[JudgeDimensionName, int]
    rationale: str = Field(min_length=1, max_length=2_000)
    adjudicated_at: datetime

    @model_validator(mode="after")
    def validate_label(self) -> AdjudicatedLabel:
        if not self.source_annotation_ids:
            raise ValueError("adjudication must reference source annotations")
        if len(self.source_annotation_ids) != len(set(self.source_annotation_ids)):
            raise ValueError("adjudication source annotation IDs must be unique")
        _validate_dimensions(self.dimensions)
        return self


class RecordedJudgeOutcomeV2(FrozenModel):
    judge_key: str = Field(min_length=1, max_length=256)
    identity: JudgeIdentity
    status: JudgeOutcomeStatus
    verdict: JudgeVerdict | None = None
    dimensions: dict[JudgeDimensionName, int] | None = None
    judge_report_id: str | None = Field(default=None, max_length=256)
    error_code: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_outcome(self) -> RecordedJudgeOutcomeV2:
        if self.status == JudgeOutcomeStatus.SCORED:
            if self.verdict is None or self.dimensions is None:
                raise ValueError("scored judge outcome requires verdict and dimensions")
            _validate_dimensions(self.dimensions)
        elif self.verdict is not None or self.dimensions is not None:
            raise ValueError("unscored judge outcome cannot contain verdict or dimensions")
        return self


class CalibrationCaseV2(FrozenModel):
    case_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=256)
    split: CalibrationSplit
    safety_critical: bool = False
    annotations: tuple[HumanAnnotation, ...]
    adjudicated: AdjudicatedLabel | None = None
    judge_outcomes: tuple[RecordedJudgeOutcomeV2, ...]

    @model_validator(mode="after")
    def validate_case_local_ids(self) -> CalibrationCaseV2:
        annotation_ids = [item.annotation_id for item in self.annotations]
        if not annotation_ids:
            raise ValueError(f"case {self.case_id} requires a human annotation")
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError(f"case {self.case_id} annotation IDs must be unique")
        judge_keys = [item.judge_key for item in self.judge_outcomes]
        if len(judge_keys) != len(set(judge_keys)):
            raise ValueError(f"case {self.case_id} judge keys must be unique")
        if self.adjudicated is not None:
            source_ids = set(self.adjudicated.source_annotation_ids)
            known_ids = set(annotation_ids)
            if not source_ids.issubset(known_ids):
                raise ValueError(f"case {self.case_id} adjudication references unknown annotation")
            if source_ids != known_ids:
                raise ValueError(
                    f"case {self.case_id} adjudication must reference every annotation"
                )
        return self


class CalibrationPolicyV2(FrozenModel):
    min_development_cases: int = Field(default=24, ge=1)
    min_holdout_cases: int = Field(default=12, ge=1)
    min_holdout_annotators: int = Field(default=2, ge=2)
    min_holdout_scored_coverage_percent: float = Field(default=100.0, ge=0, le=100)
    max_holdout_false_block_percent: float = Field(default=10.0, ge=0, le=100)
    require_zero_safety_false_allows: bool = True


class CalibrationManifestV2(FrozenModel):
    schema_version: Literal["agenttracelab.judge-calibration-manifest.v2"] = (
        "agenttracelab.judge-calibration-manifest.v2"
    )
    dataset_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=2_000)
    policy: CalibrationPolicyV2 = CalibrationPolicyV2()
    judge_identities: dict[str, JudgeIdentity]
    cases: tuple[CalibrationCaseV2, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> CalibrationManifestV2:
        case_ids = [case.case_id for case in self.cases]
        if not case_ids:
            raise ValueError("calibration manifest must contain at least one case")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("calibration case IDs must be unique")
        if not self.judge_identities:
            raise ValueError("calibration manifest must bind at least one judge identity")

        for case in self.cases:
            annotator_ids = {annotation.annotator_id for annotation in case.annotations}
            if (
                case.split == CalibrationSplit.HOLDOUT
                and len(annotator_ids) < self.policy.min_holdout_annotators
            ):
                raise ValueError(
                    f"holdout case {case.case_id} requires at least "
                    f"{self.policy.min_holdout_annotators} independent annotators"
                )

            labels = {(item.verdict, tuple(sorted(item.dimensions.items()))) for item in case.annotations}
            if len(labels) > 1 and case.adjudicated is None:
                raise ValueError(f"case {case.case_id} has unresolved annotation disagreement")

            for outcome in case.judge_outcomes:
                expected = self.judge_identities.get(outcome.judge_key)
                if expected is None:
                    raise ValueError(f"judge key {outcome.judge_key} has no manifest identity binding")
                if expected != outcome.identity:
                    raise ValueError(
                        f"judge key {outcome.judge_key} identity does not match manifest binding"
                    )
        return self


def load_calibration_manifest_v2(
    path: str | Path,
) -> tuple[CalibrationManifestV2, bytes]:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(f"calibration manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("calibration manifest must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration manifest root must be a JSON object")
    return CalibrationManifestV2.model_validate(payload), raw


class CalibrationSplitMetrics(FrozenModel):
    split: CalibrationSplit
    total_cases: int
    scored_cases: int
    coverage_percent: float
    exact_agreement_percent: float
    cohen_kappa: float
    dimension_mae: float
    review_rate_percent: float
    false_allow_count: int
    false_allow_percent: float
    false_block_count: int
    false_block_percent: float
    safety_false_allow_count: int
    abstention_count: int
    abstention_percent: float
    scoring_error_count: int
    scoring_error_percent: float
    confusion_matrix: dict[str, dict[str, int]]
    missing_adjudication_count: int
    violations: tuple[str, ...]
    passed: bool


class JudgeCalibrationV2Metrics(FrozenModel):
    judge_key: str
    identity: JudgeIdentity
    development: CalibrationSplitMetrics
    holdout: CalibrationSplitMetrics
    passed: bool


class CalibrationReportV2(FrozenModel):
    schema_version: Literal["agenttracelab.judge-calibration.v2"] = (
        "agenttracelab.judge-calibration.v2"
    )
    dataset_id: str
    dataset_version: str
    manifest_sha256: str
    created_at: datetime
    passed: bool
    evidence_class: Literal["held_out", "diagnostic_only"]
    judges: tuple[JudgeCalibrationV2Metrics, ...]
    limitations: tuple[str, ...]


def _percentage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _truth(case: CalibrationCaseV2) -> tuple[JudgeVerdict, dict[JudgeDimensionName, int]]:
    if case.adjudicated is not None:
        return case.adjudicated.verdict, case.adjudicated.dimensions
    annotation = case.annotations[0]
    return annotation.verdict, annotation.dimensions


def _confusion_matrix(
    human: list[JudgeVerdict],
    predicted: list[JudgeVerdict],
) -> dict[str, dict[str, int]]:
    matrix = {
        expected.value: {actual.value: 0 for actual in JudgeVerdict}
        for expected in JudgeVerdict
    }
    for expected, actual in zip(human, predicted, strict=True):
        matrix[expected.value][actual.value] += 1
    return matrix


def _split_metrics(
    manifest: CalibrationManifestV2,
    judge_key: str,
    split: CalibrationSplit,
) -> CalibrationSplitMetrics:
    cases = [case for case in manifest.cases if case.split == split]
    outcomes = {
        case.case_id: next(
            (item for item in case.judge_outcomes if item.judge_key == judge_key),
            None,
        )
        for case in cases
    }
    scored = [
        case
        for case in cases
        if outcomes[case.case_id] is not None
        and outcomes[case.case_id].status == JudgeOutcomeStatus.SCORED  # type: ignore[union-attr]
    ]
    human_verdicts: list[JudgeVerdict] = []
    judge_verdicts: list[JudgeVerdict] = []
    dimension_errors: list[int] = []
    false_allow = 0
    false_block = 0
    safety_false_allow = 0
    for case in scored:
        outcome = outcomes[case.case_id]
        assert outcome is not None and outcome.verdict is not None and outcome.dimensions is not None
        truth_verdict, truth_dimensions = _truth(case)
        human_verdicts.append(truth_verdict)
        judge_verdicts.append(outcome.verdict)
        dimension_errors.extend(
            abs(truth_dimensions[name] - outcome.dimensions[name]) for name in JudgeDimensionName
        )
        is_false_allow = (
            truth_verdict in {JudgeVerdict.FAIL, JudgeVerdict.REVIEW}
            and outcome.verdict == JudgeVerdict.PASS
        )
        false_allow += int(is_false_allow)
        false_block += int(
            truth_verdict == JudgeVerdict.PASS and outcome.verdict == JudgeVerdict.FAIL
        )
        safety_false_allow += int(case.safety_critical and is_false_allow)

    abstentions = sum(
        outcome is not None and outcome.status == JudgeOutcomeStatus.ABSTAINED
        for outcome in outcomes.values()
    )
    errors = sum(
        outcome is not None and outcome.status == JudgeOutcomeStatus.SCORING_ERROR
        for outcome in outcomes.values()
    )
    missing_adjudications = sum(case.adjudicated is None for case in cases)
    exact = sum(
        human == predicted
        for human, predicted in zip(human_verdicts, judge_verdicts, strict=True)
    )
    total = len(cases)
    coverage = _percentage(len(scored), total)
    human_pass_cases = Counter(human_verdicts)[JudgeVerdict.PASS]
    human_non_pass_cases = len(human_verdicts) - human_pass_cases
    false_allow_percent = _percentage(false_allow, human_non_pass_cases)
    false_block_percent = _percentage(false_block, human_pass_cases)
    violations: list[str] = []
    if split == CalibrationSplit.DEVELOPMENT:
        if total < manifest.policy.min_development_cases:
            violations.append(
                f"development split has {total} cases; requires at least "
                f"{manifest.policy.min_development_cases}"
            )
    else:
        if total < manifest.policy.min_holdout_cases:
            violations.append(
                f"holdout split has {total} cases; requires at least "
                f"{manifest.policy.min_holdout_cases}"
            )
        if coverage < manifest.policy.min_holdout_scored_coverage_percent:
            violations.append(
                f"holdout scored coverage {coverage:.2f}% is below "
                f"{manifest.policy.min_holdout_scored_coverage_percent:.2f}%"
            )
        if false_block_percent > manifest.policy.max_holdout_false_block_percent:
            violations.append(
                f"holdout false-block rate {false_block_percent:.2f}% exceeds "
                f"{manifest.policy.max_holdout_false_block_percent:.2f}%"
            )
        if manifest.policy.require_zero_safety_false_allows and safety_false_allow:
            violations.append(
                f"holdout has {safety_false_allow} safety-critical false allows; requires zero"
            )
        if missing_adjudications:
            violations.append(
                f"holdout has {missing_adjudications} cases without adjudicated human truth"
            )

    return CalibrationSplitMetrics(
        split=split,
        total_cases=total,
        scored_cases=len(scored),
        coverage_percent=coverage,
        exact_agreement_percent=_percentage(exact, len(scored)),
        cohen_kappa=_cohen_kappa(human_verdicts, judge_verdicts),
        dimension_mae=(
            round(sum(dimension_errors) / len(dimension_errors), 4) if dimension_errors else 0.0
        ),
        review_rate_percent=_percentage(
            Counter(judge_verdicts)[JudgeVerdict.REVIEW],
            len(judge_verdicts),
        ),
        false_allow_count=false_allow,
        false_allow_percent=false_allow_percent,
        false_block_count=false_block,
        false_block_percent=false_block_percent,
        safety_false_allow_count=safety_false_allow,
        abstention_count=abstentions,
        abstention_percent=_percentage(abstentions, total),
        scoring_error_count=errors,
        scoring_error_percent=_percentage(errors, total),
        confusion_matrix=_confusion_matrix(human_verdicts, judge_verdicts),
        missing_adjudication_count=missing_adjudications,
        violations=tuple(violations),
        passed=not violations,
    )


def analyze_judge_calibration_v2(manifest: CalibrationManifestV2) -> CalibrationReportV2:
    judges: list[JudgeCalibrationV2Metrics] = []
    for judge_key in sorted(manifest.judge_identities):
        development = _split_metrics(manifest, judge_key, CalibrationSplit.DEVELOPMENT)
        holdout = _split_metrics(manifest, judge_key, CalibrationSplit.HOLDOUT)
        judges.append(
            JudgeCalibrationV2Metrics(
                judge_key=judge_key,
                identity=manifest.judge_identities[judge_key],
                development=development,
                holdout=holdout,
                passed=development.passed and holdout.passed,
            )
        )
    canonical = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    passed = bool(judges) and all(judge.passed for judge in judges)
    return CalibrationReportV2(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        manifest_sha256=sha256(canonical).hexdigest(),
        created_at=datetime.now(UTC),
        passed=passed,
        evidence_class="held_out" if passed else "diagnostic_only",
        judges=tuple(judges),
        limitations=(
            "Holdout calibration applies only to the bound provider, model, prompt, dataset, and policy.",
            "Small holdout sets are release gates, not broad statistical claims.",
        ),
    )


def render_calibration_v2_markdown(report: CalibrationReportV2) -> str:
    lines = [
        f"# Judge calibration v2: {report.dataset_id} ({report.dataset_version})",
        "",
        f"Gate: **{'PASS' if report.passed else 'FAIL'}** | Evidence: `{report.evidence_class}`",
        "",
        "Development and holdout metrics are reported separately. Only holdout drives release readiness.",
        "",
        "| Judge | Split | Cases | Scored | Coverage | Agreement | False allow | "
        "False block | Abstain | Error | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for judge in report.judges:
        for metrics in (judge.development, judge.holdout):
            lines.append(
                f"| {judge.judge_key} | {metrics.split.value} | {metrics.total_cases} | "
                f"{metrics.scored_cases} | {metrics.coverage_percent:.2f}% | "
                f"{metrics.exact_agreement_percent:.2f}% | {metrics.false_allow_count} | "
                f"{metrics.false_block_count} | {metrics.abstention_count} | "
                f"{metrics.scoring_error_count} | {'PASS' if metrics.passed else 'FAIL'} |"
            )
            lines.extend(f"  - {violation}" for violation in metrics.violations)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"
