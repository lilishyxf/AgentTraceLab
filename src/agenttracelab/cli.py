from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
from agenttracelab.datasets import run_dataset
from agenttracelab.judging import (
    ContentPolicy,
    JudgeConfig,
    JudgeError,
    JudgeVerdict,
    OpenAICompatibleJudge,
)
from agenttracelab.policy import apply_promotion_policy, load_promotion_policy, render_gate_markdown
from agenttracelab.replay import run_replay
from agenttracelab.reporting import render_comparison_html, render_comparison_junit
from agenttracelab.review import ReviewResolutionInput, ReviewState
from agenttracelab.service import AgentTraceService, TraceNotFoundError
from agenttracelab.storage import Database


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


def _compare(args: argparse.Namespace) -> int:
    report = _service(args.database_url).compare(args.baseline, args.candidate)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if report.recommendation == "promote" else 3


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

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare latest evaluations for two stored traces.",
    )
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.set_defaults(handler=_compare)

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
