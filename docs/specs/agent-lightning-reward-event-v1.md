# Agent Lightning reward event v1 specification

## Purpose

Convert one stored AgentTraceLab trace and its latest deterministic evaluation into the current
Agent Lightning v1.0 reward-event request without sending it or starting a trainer.

## Source contract

Agent Lightning v1.0, pinned here to commit
`ff9457587fb6ec900e16e93be9ad2d77409afa08`, stores append-only events per rollout attempt. Its
`EventCreate` request body contains `event_type` and `data`; the well-known reward data requires a
scalar `value` and permits a message, source, and reason. The destination is:

```text
POST /rollouts/{rollout_id}/attempt/{attempt_id}/events
```

AgentTraceLab records the destination separately from the body so a caller can audit the mapping
before performing any external write.

## Reward policy

The exported scalar is `TrainingRewardRecord.recommended_reward`:

- passing deterministic evaluation: normalized process reward, normally `1.0`;
- non-safety failure: the dense process reward;
- approval-before-commit or secret-redaction failure: `0.0`.

The event reason is one of `deterministic_gate_passed`, `deterministic_gate_failed`, or
`safety_gate_blocked`. The message contains only pass counts and the normalized reward.

## Mapping and safety

- The caller must provide the ID of an existing Agent Lightning rollout.
- The attempt ID defaults to `0` but remains explicit in the export.
- Both IDs are restricted to ASCII letters, digits, period, underscore, and hyphen.
- Trace and evaluation IDs are retained for provenance.
- No task, prompt, response, observation, tool payload, or credential content is exported.
- Building or reading an export never performs an HTTP request.

## Interfaces

- Python: `build_agent_lightning_reward_event(...)`.
- Service: `AgentTraceService.get_agent_lightning_reward_event(...)`.
- CLI: `agent-lightning-reward-event TRACE_ID --rollout-id ID [--attempt-id ID]`.
- HTTP: `GET /v1/integrations/agent-lightning/reward-events/{trace_id}` with rollout query data.

## Non-goals

- creating or mutating an Agent Lightning rollout;
- authenticating to or posting into an Agent Lightning Gateway;
- exporting model-request token IDs or log probabilities;
- running VERL, GRPO, SFT, DPO, or any other trainer;
- claiming a model-quality improvement from an exported reward.
