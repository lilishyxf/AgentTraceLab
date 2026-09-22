from __future__ import annotations

import json

import pytest

import agenttracelab
from agenttracelab.adapters import adapt_wasmhatch_journal
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.optimization import (
    build_agent_lightning_reward_event,
    build_gepa_evaluation,
    build_optimization_feedback,
    build_training_reward,
    localize_failures,
)


def test_public_package_exports_optimization_contracts() -> None:
    assert agenttracelab.__version__ == "2.3.0"
    assert agenttracelab.build_agent_lightning_reward_event is build_agent_lightning_reward_event
    assert agenttracelab.build_gepa_evaluation is build_gepa_evaluation
    assert agenttracelab.build_optimization_feedback is build_optimization_feedback
    assert agenttracelab.build_training_reward is build_training_reward


def test_localizes_failed_checks_without_copying_trace_content(incomplete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(incomplete_journal)
    report = evaluate_trace(trace)

    localization = localize_failures(trace, report)

    assert localization.trace_id == trace.trace_id
    assert localization.evaluation_id == report.evaluation_id
    assert localization.primary_failure_span_id == trace.spans[0].span_id
    assert {finding.check_id for finding in localization.findings} == set(report.failure_categories)
    known_span_ids = {span.span_id for span in trace.spans}
    assert all(span_id in known_span_ids for finding in localization.findings for span_id in finding.span_ids)

    serialized = json.dumps(localization.model_dump(mode="json"), ensure_ascii=False)
    assert trace.task not in serialized
    assert "The source schema has not been inspected" not in serialized
    assert "Tool returned without validation" not in serialized


def test_localization_rejects_mismatched_trace_and_evaluation(
    complete_journal: dict,
    incomplete_journal: dict,
) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)
    other_report = evaluate_trace(adapt_wasmhatch_journal(incomplete_journal))

    with pytest.raises(ValueError, match="same trace"):
        localize_failures(trace, other_report)


def test_passing_trace_has_no_primary_failure(complete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)
    report = evaluate_trace(trace)

    localization = localize_failures(trace, report)

    assert localization.findings == ()
    assert localization.primary_failure_span_id is None


def test_builds_gepa_compatible_score_and_actionable_feedback(
    incomplete_journal: dict,
) -> None:
    trace = adapt_wasmhatch_journal(incomplete_journal)
    report = evaluate_trace(trace)

    feedback = build_optimization_feedback(trace, report)

    assert feedback.score == report.score / 100
    assert feedback.gate_passed is False
    assert feedback.primary_failure_span_id == trace.spans[0].span_id
    assert "agent.decision_evidence" in feedback.feedback
    assert "Record observation" in feedback.feedback
    assert "The source schema has not been inspected" not in feedback.feedback
    assert feedback.findings


def test_passing_feedback_says_to_preserve_behavior(complete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)
    report = evaluate_trace(trace)

    feedback = build_optimization_feedback(trace, report)

    assert feedback.score == 1.0
    assert feedback.gate_passed is True
    assert feedback.findings == ()
    assert "preserve" in feedback.feedback.lower()


def test_safety_failure_zeros_recommended_training_reward(incomplete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(incomplete_journal)
    report = evaluate_trace(trace)

    reward = build_training_reward(trace, report)

    assert reward.outcome_reward == 0.0
    assert reward.process_reward == report.score / 100
    assert reward.safety_blocked is True
    assert reward.recommended_reward == 0.0
    approval = next(
        component for component in reward.components if component.check_id == "safety.approval_before_commit"
    )
    assert approval.value == 0.0
    assert approval.hard_gate is True
    assert approval.span_ids == (trace.spans[4].span_id,)


def test_non_safety_failure_preserves_dense_process_reward(complete_journal: dict) -> None:
    complete_journal["events"][0]["evidence"] = {}
    trace = adapt_wasmhatch_journal(complete_journal)
    report = evaluate_trace(trace)

    reward = build_training_reward(trace, report)

    assert reward.outcome_reward == 0.0
    assert reward.process_reward < 1.0
    assert reward.safety_blocked is False
    assert reward.recommended_reward == reward.process_reward


def test_builds_agent_lightning_v1_reward_event_for_passing_trace(complete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)
    report = evaluate_trace(trace)

    exported = build_agent_lightning_reward_event(
        trace,
        report,
        rollout_id="rollout-123",
        attempt_id="attempt-2",
    )

    assert exported.schema_version == "agenttracelab.agent-lightning-reward-event.v1"
    assert exported.source_trace_id == trace.trace_id
    assert exported.source_evaluation_id == report.evaluation_id
    assert exported.rollout_id == "rollout-123"
    assert exported.attempt_id == "attempt-2"
    assert exported.endpoint_path == "/rollouts/rollout-123/attempt/attempt-2/events"
    assert exported.body.event_type == "reward"
    assert exported.body.data.value == 1.0
    assert exported.body.data.source == "agenttracelab"
    assert exported.body.data.reason == "deterministic_gate_passed"
    assert "10/10" in exported.body.data.message


def test_agent_lightning_reward_event_preserves_safety_gate_and_excludes_content(
    incomplete_journal: dict,
) -> None:
    trace = adapt_wasmhatch_journal(incomplete_journal)
    report = evaluate_trace(trace)

    exported = build_agent_lightning_reward_event(trace, report, rollout_id="rollout-safe")

    assert exported.body.data.value == 0.0
    assert exported.body.data.reason == "safety_gate_blocked"
    serialized = json.dumps(exported.model_dump(mode="json"), ensure_ascii=False)
    assert trace.task not in serialized
    assert "The source schema has not been inspected" not in serialized
    assert "Tool returned without validation" not in serialized


def test_agent_lightning_reward_event_rejects_unsafe_endpoint_ids(complete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)
    report = evaluate_trace(trace)

    with pytest.raises(ValueError, match="rollout_id"):
        build_agent_lightning_reward_event(trace, report, rollout_id="../other-rollout")


def test_builds_gepa_evaluator_result_with_actionable_side_info(complete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(complete_journal)
    report = evaluate_trace(trace)

    exported = build_gepa_evaluation(trace, report)

    assert exported.schema_version == "agenttracelab.gepa-evaluation.v1"
    assert exported.source_trace_id == trace.trace_id
    assert exported.source_evaluation_id == report.evaluation_id
    assert exported.score == 1.0
    assert exported.info.scores == {
        "deterministic_gate": 1.0,
        "process_quality": 1.0,
        "safety_compliance": 1.0,
    }
    assert exported.info.failed_check_ids == ()
    assert "preserve" in exported.info.diagnostics.lower()
    assert any("approval-before-commit" in item for item in exported.info.preservation_constraints)


def test_gepa_primary_score_is_safety_gated_and_content_free(incomplete_journal: dict) -> None:
    trace = adapt_wasmhatch_journal(incomplete_journal)
    report = evaluate_trace(trace)

    exported = build_gepa_evaluation(trace, report)

    assert exported.score == 0.0
    assert exported.info.scores["process_quality"] == report.score / 100
    assert exported.info.scores["safety_compliance"] == 0.0
    assert "safety.approval_before_commit" in exported.info.failed_check_ids
    serialized = json.dumps(exported.model_dump(mode="json"), ensure_ascii=False)
    assert trace.task not in serialized
    assert "The source schema has not been inspected" not in serialized
    assert "Tool returned without validation" not in serialized
