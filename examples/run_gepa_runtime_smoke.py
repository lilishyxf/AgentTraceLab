from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agenttracelab.curation import (
    HardCaseResolution,
    HardCaseResolutionSet,
    curate_hard_cases,
)
from agenttracelab.gepa_runtime import (
    GepaValidationPolicy,
    run_gepa_experiment,
    validate_gepa_candidate_report,
)
from agenttracelab.mining import HardCaseMiningPolicy, mine_gepa_hard_cases
from agenttracelab.stability import GepaStabilityPolicy, run_gepa_stability_report

ROOT = Path(__file__).resolve().parents[1]


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))


def _replace_run_id(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    original = payload["runId"]
    return json.loads(json.dumps(payload).replace(original, run_id))


def _make_incomplete(payload: dict[str, Any]) -> dict[str, Any]:
    incomplete = copy.deepcopy(payload)
    incomplete["state"] = "active"
    incomplete["events"][0]["evidence"] = {}
    incomplete["events"][3]["outcome"] = "rejected"
    incomplete["events"][5]["summary"] = "Tool returned without validation"
    return incomplete


def _safe_proposer(
    candidate: dict[str, str],
    reflective_dataset: object,
    components_to_update: list[str],
) -> dict[str, str]:
    del candidate, reflective_dataset, components_to_update
    return {
        "system_prompt": (
            "Inspect first, require approval-before-commit, execute once, and validate by readback."
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--validation-output")
    parser.add_argument("--stability-output")
    parser.add_argument("--hard-cases-output")
    parser.add_argument("--resolutions-output")
    parser.add_argument("--curated-output")
    args = parser.parse_args()
    complete = _load_fixture("wasmhatch_complete.json")
    incomplete = _make_incomplete(complete)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["agent_config"]["system_prompt"]
        source = complete if "approval-before-commit" in prompt else incomplete
        trace = _replace_run_id(copy.deepcopy(source), f"run_journal_{calls:032x}")
        return httpx.Response(200, json={"trace": trace})

    manifest_path = ROOT / "evaluation" / "experiments" / "gepa-runtime-smoke" / "v1" / "manifest.json"
    validation_replay_path = (
        ROOT / "evaluation" / "experiments" / "gepa-runtime-smoke" / "v1" / "validation-replay.json"
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = run_gepa_experiment(
            manifest_path,
            custom_candidate_proposer=_safe_proposer,
            replay_client=client,
        )
        validation = validate_gepa_candidate_report(
            validation_id="official-gepa-runtime-smoke-held-out",
            validation_version="1.0.0",
            experiment_manifest_path=manifest_path,
            experiment_report=report,
            validation_replay_path=validation_replay_path,
            policy=GepaValidationPolicy(
                min_validation_score=1.0,
                min_task_count=2,
                max_train_validation_gap=0.0,
            ),
            replay_client=client,
        )
        stability = run_gepa_stability_report(
            stability_id="official-gepa-runtime-smoke-stability",
            stability_version="1.0.0",
            experiment_manifest_path=manifest_path,
            experiment_report=report,
            held_out_replay_path=validation_replay_path,
            policy=GepaStabilityPolicy(
                trials_per_task=5,
                min_candidate_mean_score=1.0,
                min_paired_mean_delta=1.0,
                min_candidate_success_lower_bound=0.7,
            ),
            replay_client=client,
        )
        hard_cases = mine_gepa_hard_cases(
            stability,
            dataset_id="official-gepa-runtime-smoke-challenge-set",
            dataset_version="1.0.0",
            policy=HardCaseMiningPolicy(max_cases=3, max_per_signature=3),
        )
        resolutions = HardCaseResolutionSet(
            schema_version="agenttracelab.hard-case-resolutions.v1",
            adjudication_id="official-gepa-runtime-smoke-curation",
            version="1.0.0",
            source_dataset_id=hard_cases.dataset_id,
            source_dataset_version=hard_cases.dataset_version,
            review_mode="generated-synthetic",
            reviewer_id="smoke-generator",
            reviewed_at=datetime.now(UTC),
            resolutions=tuple(
                HardCaseResolution(
                    case_id=case.case_id,
                    decision="regression_guard",
                    rationale_code="confirmed_fix",
                )
                for case in hard_cases.cases
            ),
        )
        curated = curate_hard_cases(hard_cases, resolutions)
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    validation_rendered = json.dumps(validation.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.validation_output:
        validation_output = Path(args.validation_output)
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(f"{validation_rendered}\n", encoding="utf-8")
    else:
        print(validation_rendered)
    stability_rendered = json.dumps(stability.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.stability_output:
        stability_output = Path(args.stability_output)
        stability_output.parent.mkdir(parents=True, exist_ok=True)
        stability_output.write_text(f"{stability_rendered}\n", encoding="utf-8")
    else:
        print(stability_rendered)
    hard_cases_rendered = json.dumps(hard_cases.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.hard_cases_output:
        hard_cases_output = Path(args.hard_cases_output)
        hard_cases_output.parent.mkdir(parents=True, exist_ok=True)
        hard_cases_output.write_text(f"{hard_cases_rendered}\n", encoding="utf-8")
    else:
        print(hard_cases_rendered)
    resolutions_rendered = json.dumps(resolutions.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.resolutions_output:
        resolutions_output = Path(args.resolutions_output)
        resolutions_output.parent.mkdir(parents=True, exist_ok=True)
        resolutions_output.write_text(f"{resolutions_rendered}\n", encoding="utf-8")
    else:
        print(resolutions_rendered)
    curated_rendered = json.dumps(curated.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.curated_output:
        curated_output = Path(args.curated_output)
        curated_output.parent.mkdir(parents=True, exist_ok=True)
        curated_output.write_text(f"{curated_rendered}\n", encoding="utf-8")
    else:
        print(curated_rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
