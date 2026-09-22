"""Create one deterministic DeepResearch durable run for adapter integration tests.

This uses the target repository's own persistence repositories. It does not call
an LLM or the network and must not be presented as a model-quality benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from backend.app.schemas import MessageCreate, RunCreate, RunEventCreate, SessionCreate
from deepresearch_agent.harness import SourceMode, WorkflowMode
from deepresearch_agent.harness.contracts import ContractCheckData, EvidenceData
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import (
    CheckpointRepository,
    ContractRepository,
    EventRepository,
    EvidenceRepository,
    PlanTaskToolRepository,
    RunRepository,
    SessionRepository,
)


async def build(output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    database = Database(f"sqlite+aiosqlite:///{output.as_posix()}")
    await database.create_schema()
    try:
        session = await SessionRepository(database).create(SessionCreate(title="AgentTraceLab fixture"))
        _, run, _ = await RunRepository(database).create_for_user_message(
            MessageCreate(
                session_id=session.session_id,
                role="user",
                content="Compare two documented agent-evaluation approaches and cite the evidence.",
                client_message_id="agenttracelab-deepresearch-fixture-v1",
            ),
            RunCreate(
                session_id=session.session_id,
                trigger_message_id="atomic",
                source_mode=SourceMode.WEB,
                workflow_mode=WorkflowMode.DEEP_RESEARCH,
            ),
        )
        runs = RunRepository(database)
        events = EventRepository(database)
        trajectory = PlanTaskToolRepository(database)
        await runs.update_status(run.run_id, status="planning", current_stage="planning")
        await trajectory.save_plan(
            run_id=run.run_id,
            plan_id=f"plan_{run.run_id}_1",
            version=1,
            status="executing",
            source_mode=SourceMode.WEB.value,
            plan={"goal": "collect independent evidence"},
            tasks=[
                {
                    "task_id": "task_search",
                    "task_type": "retrieval",
                    "description": "Find a primary source",
                    "status": "completed",
                }
            ],
        )
        await events.append(
            RunEventCreate(
                run_id=run.run_id,
                event_type="source.coverage_checked",
                stage="executing",
                payload={"passed": False, "evidence_count": 0, "minimum_evidence": 1},
            )
        )
        await events.append(
            RunEventCreate(
                run_id=run.run_id,
                event_type="plan.replanning",
                stage="replanning",
                payload={
                    "recovery_action": "replan",
                    "recovery_reason": "no_source_evidence",
                    "failures": ["min_evidence"],
                },
            )
        )
        call, _ = await trajectory.prepare_tool_call(
            tool_call_id="tool_fixture_search",
            run_id=run.run_id,
            task_id="task_search",
            tool_name="web_search",
            source_mode=SourceMode.WEB.value,
            args={"query": "agent evaluation primary source"},
        )
        await trajectory.complete_tool_call(call.tool_call_id, result={"result_ids": ["source-primary"]})
        await events.append(
            RunEventCreate(
                run_id=run.run_id,
                event_type="tool.completed",
                stage="executing",
                payload={"tool_call_id": call.tool_call_id, "tool_name": "web_search", "result_count": 1},
            )
        )
        await EvidenceRepository(database).upsert(
            EvidenceData(
                evidence_id="ev_fixture_primary",
                run_id=run.run_id,
                task_id="task_search",
                tool_call_id=call.tool_call_id,
                source_mode=SourceMode.WEB,
                provider="web_search",
                source_id="https://example.com/primary",
                summary="Primary-source fixture used only to validate the integration path.",
                content_hash="a" * 64,
                score=0.9,
            ),
            metadata={"domain": "example.com"},
        )
        await events.append(
            RunEventCreate(
                run_id=run.run_id,
                event_type="evidence.added",
                stage="executing",
                payload={"evidence_id": "ev_fixture_primary", "tool_call_id": call.tool_call_id},
            )
        )
        contracts = ContractRepository(database)
        for kind in ("source_match", "citation_integrity", "claim_support"):
            await contracts.upsert(
                ContractCheckData(
                    check_id=f"contract_{run.run_id}_{kind}",
                    run_id=run.run_id,
                    kind=kind,
                    verifier="fixture-deterministic",
                    verifier_version="1",
                    passed=True,
                )
            )
        await CheckpointRepository(database).save(run.run_id, "verifying", {"status": "verifying"})
        await events.append(
            RunEventCreate(
                run_id=run.run_id,
                event_type="verification.completed",
                stage="verifying",
                payload={"passed": True, "failures": [], "recovery_action": "complete"},
            )
        )
        await runs.complete_verified(
            run.run_id,
            assistant_content="Integration evidence is linked to the report [ev_fixture_primary]",
            usage={"usage": {"llm_tokens": 120, "tool_calls": 1}},
        )
        await CheckpointRepository(database).save(run.run_id, "completed", {"status": "completed"})
        await events.append(
            RunEventCreate(
                run_id=run.run_id,
                event_type="run.completed",
                stage="completed",
                payload={"verified": True},
            )
        )
        return run.run_id
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    print(asyncio.run(build(Path(args.output).resolve())))


if __name__ == "__main__":
    main()
