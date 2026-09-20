# Provider-compatible LLM judge

AgentTraceLab v0.9 adds a qualitative trajectory judge without weakening the deterministic release
gate. The two report types remain separate:

- `agenttracelab.evaluation.v1` records reproducible lifecycle, authorization, privacy, linkage, and
  budget checks.
- `agenttracelab.llm-judge.v1` records model-scored trajectory quality with provider, model, prompt
  version, evidence citations, token usage, and limitations.

The qualitative score is never added to or averaged with the deterministic score. A judge verdict
cannot turn a failed safety check into a passing release gate.

## Supported provider contract

The client calls an OpenAI-compatible, non-streaming `POST {base_url}/chat/completions` endpoint
with `response_format: {"type": "json_object"}`. This works with providers that implement that
contract, including compatible Qwen and DeepSeek endpoints. Provider behavior still varies by model,
so a new endpoint/model combination must be calibrated on a recorded dataset before it is used as a
release signal.

The response must contain all five rubric dimensions exactly once:

1. `trajectory_coherence`
2. `tool_choice`
3. `result_grounding`
4. `safety_awareness`
5. `efficiency`

Each dimension is scored from 0 to 4 and may cite only span IDs supplied in the request. Unknown
citations, missing dimensions, extra fields, malformed JSON, or a non-chat-completions response are
protocol failures rather than silently accepted results.

## Privacy and prompt-injection boundary

The default `metadata_only` policy omits task text, observations, action reasons, validation text,
and model/tool input-output content. It retains structural span data and low-risk allowlisted
attributes needed to inspect the trajectory shape.

`--include-content` is an explicit opt-in. Even with that flag, only allowlisted content fields are
sent and individual values are bounded. Trace content is labelled as untrusted data in the system
prompt.

Before any network request, the stored deterministic evaluation must contain a passing
`privacy.no_raw_secret` check. A failed or missing privacy check blocks the provider call. This is a
credential-shaped secret gate, not a general-purpose PII detector; deployments handling personal or
regulated data still need an organization-specific redaction policy.

API keys are read from an environment variable and are never written to the report or database.

## Running a judgment

First import a trace into the configured AgentTraceLab database, then set provider configuration.

DeepSeek-compatible example:

```bash
export AGENTTRACELAB_JUDGE_BASE_URL=https://api.deepseek.com
export AGENTTRACELAB_JUDGE_MODEL=deepseek-flash
export AGENTTRACELAB_JUDGE_API_KEY=your-key
uv run agenttracelab judge-trace TRACE_ID --output evaluation-results/judge.json
```

Qwen-compatible deployments can provide their regional
`https://.../compatible-mode/v1` base URL and model name through the same variables.

To include allowlisted task and tool content after reviewing the data boundary:

```bash
uv run agenttracelab judge-trace TRACE_ID --include-content
```

The command exits with `0` only for a `pass` verdict and `8` for review, failure, blocked requests,
provider failures, or invalid responses. Retryable HTTP 408, 409, 429, 5xx responses, and the known
JSON-mode empty-content condition use bounded retries. Other schema or evidence failures stop
immediately. The latest stored report is available from:

```text
GET /v1/judgments/{trace_id}/latest
```

Automated tests use a deterministic mock HTTP transport. They prove request construction, privacy
blocking, strict output validation, evidence citation checks, bounded retry behavior, persistence,
and score calculation; they do not claim that a paid provider was called during CI.
