from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import SecretStr

from agenttracelab.adapters import export_deepresearch_sqlite_snapshot
from agenttracelab.annotation_workflow import (
    build_annotation_benchmark_manifest,
    materialize_tool_trajectory_manifest,
)
from agenttracelab.batch import evaluate_wasmhatch_batch
from agenttracelab.batch_comparison import (
    compare_wasmhatch_batches,
    render_batch_comparison_markdown,
)
from agenttracelab.calibration import (
    analyze_judge_calibration,
    load_calibration_manifest,
    render_calibration_markdown,
)
from agenttracelab.claim_evidence import (
    ClaimJudgeConfig,
    ClaimJudgeError,
    OpenAICompatibleClaimJudge,
    verify_deepresearch_claims,
)
from agenttracelab.curation import curate_hard_cases_files
from agenttracelab.datasets import run_dataset
from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkRunner,
    render_deepresearch_benchmark_markdown,
    rescore_deepresearch_benchmark,
)
from agenttracelab.deepresearch_regression import (
    build_deepresearch_challenge_set,
    compare_deepresearch_benchmarks,
    load_deepresearch_regression_policy,
    write_deepresearch_regression_report,
)
from agenttracelab.fault_injection import run_deepresearch_fault_suite
from agenttracelab.gepa_runtime import (
    GepaRuntimeError,
    reflection_lm_from_environment,
    run_gepa_experiment,
    validate_gepa_candidate,
)
from agenttracelab.judging import (
    ContentPolicy,
    JudgeConfig,
    JudgeError,
    JudgeVerdict,
    OpenAICompatibleJudge,
)
from agenttracelab.mining import HardCaseMiningPolicy, mine_gepa_hard_cases_file
from agenttracelab.policy import apply_promotion_policy, load_promotion_policy, render_gate_markdown
from agenttracelab.readiness import (
    audit_deepresearch_evidence,
    render_evaluation_readiness_markdown,
)
from agenttracelab.replay import run_replay
from agenttracelab.reporting import render_comparison_html, render_comparison_junit
from agenttracelab.review import ReviewResolutionInput, ReviewState
from agenttracelab.service import AgentTraceService, TraceNotFoundError
from agenttracelab.stability import run_gepa_stability_gate
from agenttracelab.storage import Database
from agenttracelab.tool_trajectory import (
    evaluate_tool_trajectory_manifest,
    load_tool_trajectory_annotation_queue,
    render_tool_trajectory_markdown,
)


def _service(database_url: str) -> AgentTraceService:
    database = Database(database_url)
    database.create_schema()
    return AgentTraceService(database)


def _import_wasmhatch(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    trace, report = _service(args.database_url).import_wasmhatch(payload)
    result = {
        "trace_id": trace.trace_id,
        "passed": report.passed,
        "score": report.score,
        "failure_categories": report.failure_categories,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


def _import_deepresearch(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    trace, report = _service(args.database_url).import_deepresearch(payload)
    if args.report_output:
        report_output = Path(args.report_output)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            f"{json.dumps(report.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    result = {
        "trace_id": trace.trace_id,
        "passed": report.passed,
        "score": report.score,
        "failure_categories": report.failure_categories,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


def _evaluate_deepresearch_db(args: argparse.Namespace) -> int:
    payload = export_deepresearch_sqlite_snapshot(args.source_database, args.run_id)
    if args.snapshot_output:
        output = Path(args.snapshot_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    trace, report = _service(args.database_url).import_deepresearch(payload)
    if args.report_output:
        report_output = Path(args.report_output)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            f"{json.dumps(report.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    result = {
        "source_run_id": args.run_id,
        "trace_id": trace.trace_id,
        "passed": report.passed,
        "score": report.score,
        "failure_categories": report.failure_categories,
        "snapshot_output": args.snapshot_output,
        "report_output": args.report_output,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


def _run_deepresearch_benchmark(args: argparse.Namespace) -> int:
    missing = [
        name
        for name, value in (
            ("--base-url", args.base_url),
            ("--source-database", args.source_database),
        )
        if not value
    ]
    if missing:
        print(
            json.dumps(
                {"error": "missing DeepResearch benchmark target configuration", "required": missing},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 18
    bearer_token = os.getenv(args.api_key_env) if args.api_key_env else None
    claim_judge = None
    if args.claim_judge_base_url or args.claim_judge_model:
        claim_api_key = os.getenv(args.claim_judge_api_key_env)
        missing_judge = [
            name
            for name, value in (
                ("--claim-judge-base-url", args.claim_judge_base_url),
                ("--claim-judge-model", args.claim_judge_model),
                ("--claim-judge-provider", args.claim_judge_provider),
                (f"environment:{args.claim_judge_api_key_env}", claim_api_key),
            )
            if not value
        ]
        if missing_judge:
            print(
                json.dumps(
                    {"error": "incomplete claim Judge configuration", "required": missing_judge},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 18
        claim_judge = OpenAICompatibleClaimJudge(
            ClaimJudgeConfig(
                base_url=args.claim_judge_base_url,
                model=args.claim_judge_model,
                api_key=SecretStr(claim_api_key),
                provider=args.claim_judge_provider,
                timeout_seconds=args.claim_judge_timeout_seconds,
                max_tokens=args.claim_judge_max_tokens,
                max_claims_per_request=args.claim_judge_max_claims_per_request,
            )
        )
    service = _service(args.database_url)
    try:
        with DeepResearchBenchmarkRunner(
            base_url=args.base_url,
            source_database=args.source_database,
            output_directory=args.output_directory,
            bearer_token=bearer_token,
            trace_evaluator=service.import_deepresearch,
            claim_judge=claim_judge,
        ) as runner:
            report = runner.run(args.manifest)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 18
    print(
        json.dumps(
            {
                "benchmark_id": report.benchmark_id,
                "benchmark_version": report.benchmark_version,
                "status": report.status,
                "evidence_class": report.evidence_class,
                "total_trials": report.metrics.total_trials,
                "completed_trials": report.metrics.completed_trials,
                "passed_trials": report.metrics.passed_trials,
                "pass_rate": report.metrics.pass_rate,
                "pass_power_k_rate": report.metrics.pass_power_k_rate,
                "report": str(Path(args.output_directory).resolve() / "benchmark-report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.status != "failed" else 18


def _rescore_deepresearch_benchmark(args: argparse.Namespace) -> int:
    try:
        report = rescore_deepresearch_benchmark(
            args.source_report,
            report_output=args.report_output,
            claim_report_filename=args.claim_report_filename,
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 18
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            render_deepresearch_benchmark_markdown(report),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "benchmark_id": report.benchmark_id,
                "status": report.status,
                "total_trials": report.metrics.total_trials,
                "passed_trials": report.metrics.passed_trials,
                "process_pass_rate": report.metrics.process_pass_rate,
                "citation_gate_pass_rate": report.metrics.citation_gate_pass_rate,
                "semantic_evaluation_coverage": report.metrics.semantic_evaluation_coverage,
                "report": str(Path(args.report_output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _evaluate_tool_trajectory(args: argparse.Namespace) -> int:
    try:
        report = evaluate_tool_trajectory_manifest(args.manifest)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 23

    _write_artifact(report.model_dump(mode="json"), args.output)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_tool_trajectory_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_id": report.dataset_id,
                "review_mode": report.review_mode,
                "promotion_eligible": report.promotion_eligible,
                "passed_cases": report.passed_cases,
                "case_count": report.case_count,
                "pass_rate": report.pass_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.passed_cases == report.case_count else 23


def _validate_tool_trajectory_annotations(args: argparse.Namespace) -> int:
    try:
        queue = load_tool_trajectory_annotation_queue(args.queue)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 24
    reviewed_cases = sum(case.review_status == "human-reviewed" for case in queue.cases)
    print(
        json.dumps(
            {
                "dataset_id": queue.dataset_id,
                "version": queue.version,
                "status": queue.status,
                "case_count": len(queue.cases),
                "reviewed_cases": reviewed_cases,
                "annotator_count": queue.annotator_count,
                "gold_ready": queue.status == "human-approved",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _build_tool_trajectory_benchmark(args: argparse.Namespace) -> int:
    try:
        result = build_annotation_benchmark_manifest(
            args.queue,
            args.output,
            benchmark_id=args.benchmark_id,
            target_name=args.target_name,
            target_provider=args.target_provider,
            target_model=args.target_model,
            target_revision=args.target_revision,
            trials_per_task=args.trials_per_task,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 25
    print(
        json.dumps(
            {
                "benchmark_id": result.manifest.benchmark_id,
                "live_case_count": len(result.included_case_ids),
                "excluded_case_count": len(result.excluded_cases),
                "excluded_cases": [item.model_dump(mode="json") for item in result.excluded_cases],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _materialize_tool_trajectory_manifest(args: argparse.Namespace) -> int:
    try:
        result = materialize_tool_trajectory_manifest(
            args.queue,
            args.benchmark_report,
            args.output,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 26
    print(
        json.dumps(
            {
                "dataset_id": result.manifest.dataset_id,
                "review_mode": result.manifest.review_mode,
                "included_trials": len(result.included_trial_ids),
                "skipped_trials": len(result.skipped_trial_ids),
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _verify_deepresearch_claims(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
        judge = None
        if args.judge_base_url or args.judge_model:
            api_key = os.getenv(args.api_key_env)
            missing = [
                name
                for name, value in (
                    ("--judge-base-url", args.judge_base_url),
                    ("--judge-model", args.judge_model),
                    (f"environment:{args.api_key_env}", api_key),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"incomplete claim Judge configuration: {', '.join(missing)}")
            judge = OpenAICompatibleClaimJudge(
                ClaimJudgeConfig(
                    base_url=args.judge_base_url,
                    model=args.judge_model,
                    api_key=SecretStr(api_key),
                    provider=args.judge_provider,
                    timeout_seconds=args.judge_timeout_seconds,
                    max_tokens=args.judge_max_tokens,
                    max_claims_per_request=args.judge_max_claims_per_request,
                )
            )
        task_prompt = payload.get("task") if isinstance(payload.get("task"), str) else None
        report = verify_deepresearch_claims(
            payload,
            judge=judge,
            task_prompt=task_prompt,
        )
    except (ClaimJudgeError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 19
    _write_artifact(report.model_dump(mode="json"), args.output)
    return 19 if report.recommendation == "hold" else 0


def _inject_deepresearch_faults(args: argparse.Namespace) -> int:
    try:
        report = run_deepresearch_fault_suite(
            args.snapshot,
            args.manifest,
            output_directory=args.output_directory,
            retain_mutants=not args.no_retain_mutants,
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 20
    print(
        json.dumps(
            {
                "suite_id": report.suite_id,
                "source_run_id": report.source_run_id,
                "case_count": report.case_count,
                "detected_count": report.detected_count,
                "mutation_detection_rate": report.mutation_detection_rate,
                "missed_fault_ids": report.missed_fault_ids,
                "report": str(Path(args.output_directory).resolve() / "fault-report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not report.missed_fault_ids else 20


def _build_deepresearch_challenge(args: argparse.Namespace) -> int:
    try:
        report, manifest = build_deepresearch_challenge_set(
            args.source_manifest,
            args.baseline_report,
            output_directory=args.output_directory,
            candidate_target_name=args.candidate_target_name,
            candidate_target_revision=args.candidate_target_revision,
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 21
    print(
        json.dumps(
            {
                "challenge_id": report.challenge_id,
                "selected_task_ids": report.selected_task_ids,
                "cluster_count": len(report.clusters),
                "challenge_manifest": str(
                    Path(args.output_directory).resolve() / report.challenge_manifest_path
                ),
                "trials_per_task": manifest.trials_per_task,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _compare_deepresearch_benchmarks(args: argparse.Namespace) -> int:
    try:
        policy = load_deepresearch_regression_policy(args.policy)
        report = compare_deepresearch_benchmarks(
            args.baseline_report,
            args.candidate_report,
            policy=policy,
        )
        if args.output:
            write_deepresearch_regression_report(report, args.output)
        else:
            _write_artifact(report.model_dump(mode="json"), None)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 21
    return 0 if report.recommendation == "promote" else 21


def _run_deepresearch_regression_loop(args: argparse.Namespace) -> int:
    output = Path(args.output_directory).resolve()
    regression_path = output / "regression-report.json"
    if regression_path.exists():
        print(
            json.dumps(
                {"error": "regression-report.json already exists; use a new output directory"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 21
    try:
        challenge_report, _ = build_deepresearch_challenge_set(
            args.source_manifest,
            args.baseline_report,
            output_directory=output / "challenge",
            candidate_target_name=args.candidate_target_name,
            candidate_target_revision=args.candidate_target_revision,
        )
        challenge_manifest = output / "challenge" / challenge_report.challenge_manifest_path
        service = _service(args.database_url)
        token = os.getenv(args.api_key_env) if args.api_key_env else None
        with DeepResearchBenchmarkRunner(
            base_url=args.candidate_base_url,
            source_database=args.candidate_source_database,
            output_directory=output / "candidate-benchmark",
            bearer_token=token,
            trace_evaluator=service.import_deepresearch,
        ) as runner:
            candidate = runner.run(challenge_manifest)
        policy = load_deepresearch_regression_policy(args.policy)
        regression = compare_deepresearch_benchmarks(
            args.baseline_report,
            output / "candidate-benchmark" / "benchmark-report.json",
            policy=policy,
        )
        write_deepresearch_regression_report(regression, regression_path)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 21
    print(
        json.dumps(
            {
                "challenge_tasks": challenge_report.selected_task_ids,
                "candidate_status": candidate.status,
                "candidate_pass_rate": regression.candidate_pass_rate,
                "pass_rate_delta": regression.pass_rate_delta,
                "safety_regressions": regression.safety_regressions,
                "recommendation": regression.recommendation,
                "reasons": regression.reasons,
                "report": str(regression_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if regression.recommendation == "promote" else 21


def _compare(args: argparse.Namespace) -> int:
    report = _service(args.database_url).compare(args.baseline, args.candidate)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if report.recommendation == "promote" else 3


def _write_artifact(payload: object, output_path: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)


def _optimization_feedback(args: argparse.Namespace) -> int:
    try:
        artifact = _service(args.database_url).get_optimization_feedback(args.trace_id)
    except TraceNotFoundError as exc:
        print(json.dumps({"error": f"trace or evaluation not found: {exc}"}), file=sys.stderr)
        return 11
    _write_artifact(artifact.model_dump(mode="json"), args.output)
    return 0


def _training_reward(args: argparse.Namespace) -> int:
    try:
        artifact = _service(args.database_url).get_training_reward(args.trace_id)
    except TraceNotFoundError as exc:
        print(json.dumps({"error": f"trace or evaluation not found: {exc}"}), file=sys.stderr)
        return 11
    _write_artifact(artifact.model_dump(mode="json"), args.output)
    return 0


def _agent_lightning_reward_event(args: argparse.Namespace) -> int:
    try:
        artifact = _service(args.database_url).get_agent_lightning_reward_event(
            args.trace_id,
            rollout_id=args.rollout_id,
            attempt_id=args.attempt_id,
        )
    except TraceNotFoundError as exc:
        print(json.dumps({"error": f"trace or evaluation not found: {exc}"}), file=sys.stderr)
        return 11
    except ValueError as exc:
        print(json.dumps({"error": f"invalid Agent Lightning event mapping: {exc}"}), file=sys.stderr)
        return 10
    _write_artifact(artifact.model_dump(mode="json"), args.output)
    return 0


def _gepa_evaluation(args: argparse.Namespace) -> int:
    try:
        artifact = _service(args.database_url).get_gepa_evaluation(args.trace_id)
    except TraceNotFoundError as exc:
        print(json.dumps({"error": f"trace or evaluation not found: {exc}"}), file=sys.stderr)
        return 11
    _write_artifact(artifact.model_dump(mode="json"), args.output)
    return 0


def _run_gepa_experiment(args: argparse.Namespace) -> int:
    missing = [
        name
        for name, value in (
            ("--reflection-base-url", args.reflection_base_url),
            ("--reflection-model", args.reflection_model),
        )
        if not value
    ]
    if missing:
        print(
            json.dumps(
                {"error": "missing GEPA reflection configuration", "required": missing},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 12
    try:
        reflection_lm = reflection_lm_from_environment(
            base_url=args.reflection_base_url,
            model=args.reflection_model,
            api_key_env=args.api_key_env,
        )
        report = run_gepa_experiment(args.manifest, reflection_lm=reflection_lm)
    except (GepaRuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 12
    _write_artifact(report.model_dump(mode="json"), args.output)
    return 0


def _validate_gepa_candidate(args: argparse.Namespace) -> int:
    try:
        report = validate_gepa_candidate(args.manifest)
    except (GepaRuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 13
    _write_artifact(report.model_dump(mode="json"), args.output)
    return 0 if report.recommendation == "promote" else 13


def _run_gepa_stability_gate(args: argparse.Namespace) -> int:
    try:
        report = run_gepa_stability_gate(args.manifest)
    except (GepaRuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 14
    _write_artifact(report.model_dump(mode="json"), args.output)
    return 0 if report.recommendation == "promote" else 14


def _mine_gepa_hard_cases(args: argparse.Namespace) -> int:
    try:
        report = mine_gepa_hard_cases_file(
            args.report,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            policy=HardCaseMiningPolicy(
                max_cases=args.max_cases,
                max_per_signature=args.max_per_signature,
                include_candidate_failures=not args.exclude_candidate_failures,
                include_score_regressions=not args.exclude_score_regressions,
                include_resolved_baseline_failures=not args.exclude_resolved_baseline_failures,
            ),
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 15
    _write_artifact(report.model_dump(mode="json"), args.output)
    return 0


def _curate_gepa_hard_cases(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        print(
            json.dumps(
                {"error": "output already exists; choose a new immutable dataset version"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 16
    try:
        report = curate_hard_cases_files(args.queue, args.resolutions)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 16
    _write_artifact(report.model_dump(mode="json"), args.output)
    return 0


def _run_dataset(args: argparse.Namespace) -> int:
    report = run_dataset(args.manifest)
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.passed else 4


def _run_replay(args: argparse.Namespace) -> int:
    report = run_replay(args.manifest)
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.passed else 5


def _evaluate_wasmhatch_batch(args: argparse.Namespace) -> int:
    report = evaluate_wasmhatch_batch(args.manifest)
    policy = load_promotion_policy(args.policy) if args.policy else None
    gate = apply_promotion_policy(report, policy) if policy else None
    payload = {
        "batch": report.model_dump(mode="json"),
        "gate": gate.model_dump(mode="json") if gate else None,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(render_gate_markdown(report, gate), encoding="utf-8")
    accepted = gate.decision == "promote" if gate else report.passed
    return 0 if accepted else 6


def _compare_wasmhatch_batches(args: argparse.Namespace) -> int:
    baseline = evaluate_wasmhatch_batch(args.baseline_manifest)
    candidate = evaluate_wasmhatch_batch(args.candidate_manifest)
    report = compare_wasmhatch_batches(baseline, candidate)
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(render_batch_comparison_markdown(report), encoding="utf-8")
    if args.html:
        html = Path(args.html)
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(render_comparison_html(report), encoding="utf-8")
    if args.junit:
        junit = Path(args.junit)
        junit.parent.mkdir(parents=True, exist_ok=True)
        junit.write_text(render_comparison_junit(report), encoding="utf-8")
    return 0 if report.recommendation == "promote" else 7


def _judge_trace(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env)
    missing = [
        name
        for name, value in (
            ("--base-url", args.base_url),
            ("--model", args.model),
            (args.api_key_env, api_key),
        )
        if not value
    ]
    if missing:
        print(
            json.dumps(
                {"error": "missing judge configuration", "required": missing},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 8
    try:
        config = JudgeConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            provider=args.provider,
            content_policy=(
                ContentPolicy.INCLUDE_CONTENT if args.include_content else ContentPolicy.METADATA_ONLY
            ),
        )
        report = _service(args.database_url).judge_trace(
            args.trace_id,
            OpenAICompatibleJudge(config),
        )
    except (JudgeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 8
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.verdict == JudgeVerdict.PASS else 8


def _analyze_judge_calibration(args: argparse.Namespace) -> int:
    report = analyze_judge_calibration(load_calibration_manifest(args.manifest))
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(render_calibration_markdown(report), encoding="utf-8")
    return 0 if report.passed else 9


def _audit_deepresearch_evidence(args: argparse.Namespace) -> int:
    try:
        report = audit_deepresearch_evidence(args.manifest, args.report, args.policy)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 19
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        if output.exists():
            print(
                json.dumps(
                    {"error": "readiness output already exists; use a new immutable path"},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 19
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    if args.summary:
        summary = Path(args.summary)
        if summary.exists():
            print(
                json.dumps(
                    {"error": "readiness summary already exists; use a new immutable path"},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 19
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(render_evaluation_readiness_markdown(report), encoding="utf-8")
    return 0 if report.ready else 19


def _queue_review(args: argparse.Namespace) -> int:
    try:
        item = _service(args.database_url).queue_review(args.trace_id, force=args.force)
    except TraceNotFoundError as exc:
        print(json.dumps({"error": f"trace, evaluation, or judge report not found: {exc}"}), file=sys.stderr)
        return 10
    if item is None:
        print(
            json.dumps({"error": "latest reports do not meet review selection rules"}),
            file=sys.stderr,
        )
        return 10
    print(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _list_reviews(args: argparse.Namespace) -> int:
    state = ReviewState(args.state) if args.state else None
    items = _service(args.database_url).list_review_items(state)
    rendered = json.dumps(
        {"items": [item.model_dump(mode="json") for item in items]},
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


def _resolve_review(args: argparse.Namespace) -> int:
    try:
        resolution = ReviewResolutionInput.model_validate_json(
            Path(args.resolution).read_text(encoding="utf-8")
        )
        item = _service(args.database_url).resolve_review_item(args.review_id, resolution)
    except (OSError, ValueError, TraceNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 10
    print(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _export_review_calibration(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        error = " ".join(
            (
                "output already exists;",
                "choose a new dataset version instead of overwriting labels",
            )
        )
        print(
            json.dumps({"error": error}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 10
    try:
        manifest = _service(args.database_url).export_resolved_reviews(
            dataset_id=args.dataset_id,
            version=args.version,
            description=args.description,
        )
    except (ValueError, TraceNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 10
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(manifest.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "cases": len(manifest.cases)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenttracelab")
    parser.add_argument(
        "--database-url",
        default="sqlite+pysqlite:///./agenttracelab.db",
        help="SQLAlchemy database URL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-wasmhatch",
        help="Import and evaluate a WasmHatch run journal.",
    )
    import_parser.add_argument("path")
    import_parser.set_defaults(handler=_import_wasmhatch)

    deepresearch_import_parser = subparsers.add_parser(
        "import-deepresearch",
        help="Import and evaluate a DeepResearch Agent Harness run snapshot.",
    )
    deepresearch_import_parser.add_argument("path")
    deepresearch_import_parser.add_argument("--report-output")
    deepresearch_import_parser.set_defaults(handler=_import_deepresearch)

    deepresearch_db_parser = subparsers.add_parser(
        "evaluate-deepresearch-db",
        help="Export one run from a DeepResearch SQLite database and evaluate it read-only.",
    )
    deepresearch_db_parser.add_argument("source_database")
    deepresearch_db_parser.add_argument("run_id")
    deepresearch_db_parser.add_argument("--snapshot-output")
    deepresearch_db_parser.add_argument("--report-output")
    deepresearch_db_parser.set_defaults(handler=_evaluate_deepresearch_db)

    deepresearch_benchmark_parser = subparsers.add_parser(
        "run-deepresearch-benchmark",
        help="Run repeated versioned tasks against a live DeepResearch Agent and retain each trace.",
    )
    deepresearch_benchmark_parser.add_argument("manifest")
    deepresearch_benchmark_parser.add_argument(
        "--base-url",
        default=os.getenv("AGENTTRACELAB_DEEPRESEARCH_BASE_URL"),
        help="DeepResearch API base URL, for example http://127.0.0.1:8000.",
    )
    deepresearch_benchmark_parser.add_argument(
        "--source-database",
        default=os.getenv("AGENTTRACELAB_DEEPRESEARCH_DATABASE"),
        help="Local DeepResearch SQLite database used for read-only full-trace export.",
    )
    deepresearch_benchmark_parser.add_argument("--output-directory", required=True)
    deepresearch_benchmark_parser.add_argument(
        "--api-key-env",
        default="AGENTTRACELAB_DEEPRESEARCH_API_KEY",
        help="Optional environment variable containing a bearer token; the token is never written.",
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-base-url",
        default=os.getenv("AGENTTRACELAB_CLAIM_JUDGE_BASE_URL"),
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-model",
        default=os.getenv("AGENTTRACELAB_CLAIM_JUDGE_MODEL"),
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-provider",
        default=os.getenv("AGENTTRACELAB_CLAIM_JUDGE_PROVIDER"),
        help="Actual inference provider identity, not merely the OpenAI-compatible protocol.",
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-api-key-env",
        default="AGENTTRACELAB_CLAIM_JUDGE_API_KEY",
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-timeout-seconds",
        type=float,
        default=45.0,
        help="Per-attempt Claim Judge timeout; useful for slower reasoning models (max 300).",
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-max-tokens",
        type=int,
        default=4_000,
        help="Claim Judge output budget, including provider-side reasoning tokens (max 16384).",
    )
    deepresearch_benchmark_parser.add_argument(
        "--claim-judge-max-claims-per-request",
        type=int,
        default=20,
        help="Maximum claim verdicts requested in one Judge call (max 100).",
    )
    deepresearch_benchmark_parser.set_defaults(handler=_run_deepresearch_benchmark)

    deepresearch_rescore_parser = subparsers.add_parser(
        "rescore-deepresearch-benchmark",
        help="Reapply current process and claim gates to retained benchmark snapshots.",
    )
    deepresearch_rescore_parser.add_argument("source_report")
    deepresearch_rescore_parser.add_argument("--report-output", required=True)
    deepresearch_rescore_parser.add_argument("--summary")
    deepresearch_rescore_parser.add_argument(
        "--claim-report-filename",
        help=(
            "Optional per-trial retained Claim Judge report filename to aggregate without "
            "rerunning the target or Judge."
        ),
    )
    deepresearch_rescore_parser.set_defaults(handler=_rescore_deepresearch_benchmark)

    tool_trajectory_parser = subparsers.add_parser(
        "evaluate-tool-trajectory",
        help="Score retained Agent tool trajectories against versioned reference policies.",
    )
    tool_trajectory_parser.add_argument("manifest")
    tool_trajectory_parser.add_argument("--output")
    tool_trajectory_parser.add_argument("--summary")
    tool_trajectory_parser.set_defaults(handler=_evaluate_tool_trajectory)

    tool_annotation_parser = subparsers.add_parser(
        "validate-tool-trajectory-annotations",
        help="Validate provenance and review completeness for a tool-trajectory annotation queue.",
    )
    tool_annotation_parser.add_argument("queue")
    tool_annotation_parser.set_defaults(handler=_validate_tool_trajectory_annotations)

    tool_benchmark_parser = subparsers.add_parser(
        "build-tool-trajectory-benchmark",
        help="Create a live-safe benchmark manifest from an annotation queue.",
    )
    tool_benchmark_parser.add_argument("queue")
    tool_benchmark_parser.add_argument("--output", required=True)
    tool_benchmark_parser.add_argument("--benchmark-id", required=True)
    tool_benchmark_parser.add_argument("--target-name", required=True)
    tool_benchmark_parser.add_argument("--target-provider", required=True)
    tool_benchmark_parser.add_argument("--target-model", required=True)
    tool_benchmark_parser.add_argument("--target-revision")
    tool_benchmark_parser.add_argument("--trials-per-task", type=int, default=1)
    tool_benchmark_parser.set_defaults(handler=_build_tool_trajectory_benchmark)

    materialize_parser = subparsers.add_parser(
        "materialize-tool-trajectory-manifest",
        help="Bind benchmark snapshots back to their annotation expectations.",
    )
    materialize_parser.add_argument("queue")
    materialize_parser.add_argument("benchmark_report")
    materialize_parser.add_argument("--output", required=True)
    materialize_parser.set_defaults(handler=_materialize_tool_trajectory_manifest)

    claim_parser = subparsers.add_parser(
        "verify-deepresearch-claims",
        help="Audit report claims against exported evidence summaries, optionally with an independent Judge.",
    )
    claim_parser.add_argument("snapshot")
    claim_parser.add_argument(
        "--judge-base-url",
        default=os.getenv("AGENTTRACELAB_CLAIM_JUDGE_BASE_URL"),
    )
    claim_parser.add_argument(
        "--judge-model",
        default=os.getenv("AGENTTRACELAB_CLAIM_JUDGE_MODEL"),
    )
    claim_parser.add_argument(
        "--judge-provider",
        default=os.getenv("AGENTTRACELAB_CLAIM_JUDGE_PROVIDER", "openai-compatible"),
    )
    claim_parser.add_argument("--api-key-env", default="AGENTTRACELAB_CLAIM_JUDGE_API_KEY")
    claim_parser.add_argument(
        "--judge-timeout-seconds",
        type=float,
        default=45.0,
        help="Per-attempt Claim Judge timeout; useful for slower reasoning models (max 300).",
    )
    claim_parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=4_000,
        help="Claim Judge output budget, including provider-side reasoning tokens (max 16384).",
    )
    claim_parser.add_argument(
        "--judge-max-claims-per-request",
        type=int,
        default=20,
        help="Maximum claim verdicts requested in one Judge call (max 100).",
    )
    claim_parser.add_argument("--output")
    claim_parser.set_defaults(handler=_verify_deepresearch_claims)

    fault_parser = subparsers.add_parser(
        "inject-deepresearch-faults",
        help="Mutate a copied DeepResearch snapshot and measure evaluator fault-detection sensitivity.",
    )
    fault_parser.add_argument("snapshot")
    fault_parser.add_argument("manifest")
    fault_parser.add_argument("--output-directory", required=True)
    fault_parser.add_argument("--no-retain-mutants", action="store_true")
    fault_parser.set_defaults(handler=_inject_deepresearch_faults)

    challenge_parser = subparsers.add_parser(
        "build-deepresearch-challenge",
        help="Cluster baseline failures and create an immutable candidate challenge manifest.",
    )
    challenge_parser.add_argument("source_manifest")
    challenge_parser.add_argument("baseline_report")
    challenge_parser.add_argument("--output-directory", required=True)
    challenge_parser.add_argument("--candidate-target-name", required=True)
    challenge_parser.add_argument("--candidate-target-revision")
    challenge_parser.set_defaults(handler=_build_deepresearch_challenge)

    deepresearch_compare_parser = subparsers.add_parser(
        "compare-deepresearch-benchmarks",
        help="Apply paired reliability, score, latency, safety, and freshness promotion gates.",
    )
    deepresearch_compare_parser.add_argument("baseline_report")
    deepresearch_compare_parser.add_argument("candidate_report")
    deepresearch_compare_parser.add_argument("--policy")
    deepresearch_compare_parser.add_argument("--output")
    deepresearch_compare_parser.set_defaults(handler=_compare_deepresearch_benchmarks)

    deepresearch_loop_parser = subparsers.add_parser(
        "run-deepresearch-regression-loop",
        help="Cluster failures, run a candidate challenge benchmark, and emit a promote/hold decision.",
    )
    deepresearch_loop_parser.add_argument("source_manifest")
    deepresearch_loop_parser.add_argument("baseline_report")
    deepresearch_loop_parser.add_argument("--candidate-base-url", required=True)
    deepresearch_loop_parser.add_argument("--candidate-source-database", required=True)
    deepresearch_loop_parser.add_argument("--candidate-target-name", required=True)
    deepresearch_loop_parser.add_argument("--candidate-target-revision")
    deepresearch_loop_parser.add_argument("--policy")
    deepresearch_loop_parser.add_argument("--output-directory", required=True)
    deepresearch_loop_parser.add_argument(
        "--api-key-env",
        default="AGENTTRACELAB_DEEPRESEARCH_API_KEY",
    )
    deepresearch_loop_parser.set_defaults(handler=_run_deepresearch_regression_loop)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare latest evaluations for two stored traces.",
    )
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.set_defaults(handler=_compare)

    optimization_parser = subparsers.add_parser(
        "optimization-feedback",
        help="Build privacy-safe score and actionable feedback for reflective optimizers.",
    )
    optimization_parser.add_argument("trace_id")
    optimization_parser.add_argument("--output")
    optimization_parser.set_defaults(handler=_optimization_feedback)

    reward_parser = subparsers.add_parser(
        "training-reward",
        help="Build an evidence-linked reward record for an evaluated trace.",
    )
    reward_parser.add_argument("trace_id")
    reward_parser.add_argument("--output")
    reward_parser.set_defaults(handler=_training_reward)

    agent_lightning_parser = subparsers.add_parser(
        "agent-lightning-reward-event",
        help="Build an Agent Lightning v1.0 reward EventCreate payload without sending it.",
    )
    agent_lightning_parser.add_argument("trace_id")
    agent_lightning_parser.add_argument("--rollout-id", required=True)
    agent_lightning_parser.add_argument("--attempt-id", default="0")
    agent_lightning_parser.add_argument("--output")
    agent_lightning_parser.set_defaults(handler=_agent_lightning_reward_event)

    gepa_parser = subparsers.add_parser(
        "gepa-evaluation",
        help="Build GEPA optimize_anything score and side-info data for a stored trace.",
    )
    gepa_parser.add_argument("trace_id")
    gepa_parser.add_argument("--output")
    gepa_parser.set_defaults(handler=_gepa_evaluation)

    gepa_run_parser = subparsers.add_parser(
        "run-gepa-experiment",
        help="Run bounded GEPA candidate search against a trace-returning Agent endpoint.",
    )
    gepa_run_parser.add_argument("manifest")
    gepa_run_parser.add_argument(
        "--reflection-base-url",
        default=os.getenv("AGENTTRACELAB_GEPA_BASE_URL"),
        help="OpenAI-compatible base URL ending before /chat/completions.",
    )
    gepa_run_parser.add_argument(
        "--reflection-model",
        default=os.getenv("AGENTTRACELAB_GEPA_MODEL"),
    )
    gepa_run_parser.add_argument(
        "--api-key-env",
        default="AGENTTRACELAB_GEPA_API_KEY",
        help="Name of the environment variable containing the reflection API key.",
    )
    gepa_run_parser.add_argument("--output")
    gepa_run_parser.set_defaults(handler=_run_gepa_experiment)

    gepa_validation_parser = subparsers.add_parser(
        "validate-gepa-candidate",
        help="Run the selected GEPA candidate against held-out replay tasks and apply promotion gates.",
    )
    gepa_validation_parser.add_argument("manifest")
    gepa_validation_parser.add_argument("--output")
    gepa_validation_parser.set_defaults(handler=_validate_gepa_candidate)

    gepa_stability_parser = subparsers.add_parser(
        "run-gepa-stability-gate",
        help="Repeat held-out baseline/candidate trials and apply a confidence-bound gate.",
    )
    gepa_stability_parser.add_argument("manifest")
    gepa_stability_parser.add_argument("--output")
    gepa_stability_parser.set_defaults(handler=_run_gepa_stability_gate)

    hard_case_parser = subparsers.add_parser(
        "mine-gepa-hard-cases",
        help="Mine a diverse, content-free annotation queue from a GEPA stability report.",
    )
    hard_case_parser.add_argument("report")
    hard_case_parser.add_argument("--dataset-id", required=True)
    hard_case_parser.add_argument("--dataset-version", required=True)
    hard_case_parser.add_argument("--max-cases", type=int, default=50)
    hard_case_parser.add_argument("--max-per-signature", type=int, default=5)
    hard_case_parser.add_argument("--exclude-candidate-failures", action="store_true")
    hard_case_parser.add_argument("--exclude-score-regressions", action="store_true")
    hard_case_parser.add_argument("--exclude-resolved-baseline-failures", action="store_true")
    hard_case_parser.add_argument("--output")
    hard_case_parser.set_defaults(handler=_mine_gepa_hard_cases)

    curate_parser = subparsers.add_parser(
        "curate-gepa-hard-cases",
        help="Validate hard-case adjudication and write an immutable content-free challenge set.",
    )
    curate_parser.add_argument("queue")
    curate_parser.add_argument("resolutions")
    curate_parser.add_argument("--output", required=True)
    curate_parser.set_defaults(handler=_curate_gepa_hard_cases)

    dataset_parser = subparsers.add_parser(
        "run-dataset",
        help="Run a versioned offline trace dataset and verify its expected checks.",
    )
    dataset_parser.add_argument("manifest")
    dataset_parser.add_argument("--output")
    dataset_parser.set_defaults(handler=_run_dataset)

    replay_parser = subparsers.add_parser(
        "run-replay",
        help="Run bounded HTTP replay tasks against an Agent trace endpoint.",
    )
    replay_parser.add_argument("manifest")
    replay_parser.add_argument("--output")
    replay_parser.set_defaults(handler=_run_replay)

    batch_parser = subparsers.add_parser(
        "evaluate-wasmhatch-batch",
        help="Evaluate exported WasmHatch journals and apply an optional promotion policy.",
    )
    batch_parser.add_argument("manifest")
    batch_parser.add_argument("--policy")
    batch_parser.add_argument("--output")
    batch_parser.add_argument("--summary")
    batch_parser.set_defaults(handler=_evaluate_wasmhatch_batch)

    batch_comparison_parser = subparsers.add_parser(
        "compare-wasmhatch-batches",
        help="Compare paired scenarios in baseline and candidate WasmHatch export batches.",
    )
    batch_comparison_parser.add_argument("baseline_manifest")
    batch_comparison_parser.add_argument("candidate_manifest")
    batch_comparison_parser.add_argument("--output")
    batch_comparison_parser.add_argument("--summary")
    batch_comparison_parser.add_argument("--html")
    batch_comparison_parser.add_argument("--junit")
    batch_comparison_parser.set_defaults(handler=_compare_wasmhatch_batches)

    judge_parser = subparsers.add_parser(
        "judge-trace",
        help="Run a separate OpenAI-compatible qualitative judge for a stored trace.",
    )
    judge_parser.add_argument("trace_id")
    judge_parser.add_argument(
        "--base-url",
        default=os.getenv("AGENTTRACELAB_JUDGE_BASE_URL"),
        help="OpenAI-compatible base URL ending at the API version, without /chat/completions.",
    )
    judge_parser.add_argument("--model", default=os.getenv("AGENTTRACELAB_JUDGE_MODEL"))
    judge_parser.add_argument(
        "--provider",
        default=os.getenv("AGENTTRACELAB_JUDGE_PROVIDER", "openai-compatible"),
    )
    judge_parser.add_argument(
        "--api-key-env",
        default="AGENTTRACELAB_JUDGE_API_KEY",
        help="Name of the environment variable containing the API key.",
    )
    judge_parser.add_argument(
        "--include-content",
        action="store_true",
        help="Explicitly include allowlisted task, model, and tool content in the external request.",
    )
    judge_parser.add_argument("--output")
    judge_parser.set_defaults(handler=_judge_trace)

    calibration_parser = subparsers.add_parser(
        "analyze-judge-calibration",
        help="Compare recorded judge ratings with human labels and apply calibration gates.",
    )
    calibration_parser.add_argument("manifest")
    calibration_parser.add_argument("--output")
    calibration_parser.add_argument("--summary")
    calibration_parser.set_defaults(handler=_analyze_judge_calibration)

    readiness_parser = subparsers.add_parser(
        "audit-deepresearch-evidence",
        help="Audit whether a DeepResearch benchmark can support diagnostic or release claims.",
    )
    readiness_parser.add_argument("manifest")
    readiness_parser.add_argument("report")
    readiness_parser.add_argument("--policy", required=True)
    readiness_parser.add_argument("--output")
    readiness_parser.add_argument("--summary")
    readiness_parser.set_defaults(handler=_audit_deepresearch_evidence)

    queue_review_parser = subparsers.add_parser(
        "queue-review",
        help="Queue the latest deterministic/Judge pair when it meets review rules.",
    )
    queue_review_parser.add_argument("trace_id")
    queue_review_parser.add_argument(
        "--force",
        action="store_true",
        help="Queue a deliberate audit sample even when the reports agree.",
    )
    queue_review_parser.set_defaults(handler=_queue_review)

    list_reviews_parser = subparsers.add_parser(
        "list-reviews",
        help="List human-review items, optionally filtered by state.",
    )
    list_reviews_parser.add_argument("--state", choices=[state.value for state in ReviewState])
    list_reviews_parser.add_argument("--output")
    list_reviews_parser.set_defaults(handler=_list_reviews)

    resolve_review_parser = subparsers.add_parser(
        "resolve-review",
        help="Resolve a pending review item from a validated JSON decision.",
    )
    resolve_review_parser.add_argument("review_id")
    resolve_review_parser.add_argument("resolution")
    resolve_review_parser.set_defaults(handler=_resolve_review)

    export_review_parser = subparsers.add_parser(
        "export-review-calibration",
        help="Export resolved reviews as a new immutable Judge calibration manifest.",
    )
    export_review_parser.add_argument("dataset_id")
    export_review_parser.add_argument("version")
    export_review_parser.add_argument("output")
    export_review_parser.add_argument(
        "--description",
        default="Human-resolved AgentTraceLab review items.",
    )
    export_review_parser.set_defaults(handler=_export_review_calibration)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
