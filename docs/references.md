# References and reuse boundary

AgentTraceLab is a new implementation. The projects below are design references or optional
integration targets; their code is not copied into this repository.

| Project | What AgentTraceLab studies | License checked during initial research |
| --- | --- | --- |
| [Microsoft AgentPex](https://github.com/microsoft/agentpex) | Trace importers, transition/output specifications, forbidden-edge evaluation | MIT |
| [OpenInference](https://github.com/Arize-ai/openinference) | AI-specific span kinds and semantic-convention direction | Apache-2.0 |
| [DeepEval](https://github.com/confident-ai/deepeval) | Optional trajectory and tool-use judge metrics | Apache-2.0 |
| [Promptfoo](https://github.com/promptfoo/promptfoo) | CI output and red-team workflow patterns | MIT |
| [Opik](https://github.com/comet-ml/opik) | Dataset, experiment, trace, and online-evaluation product boundaries | Apache-2.0 |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | Reproducible Task/Solver/Scorer evaluation architecture | MIT |
| [OpenTelemetry OTLP specification](https://opentelemetry.io/docs/specs/otlp/) | OTLP/HTTP paths, media types, compression, responses, and receiver limits | Specification reference |
| [OpenTelemetry Protocol](https://github.com/open-telemetry/opentelemetry-proto) | Canonical trace export request and response messages | Apache-2.0 |
| [OpenTelemetry Python exporters](https://opentelemetry.io/docs/languages/python/exporters/) | Official OTLP/HTTP exporter and batching setup used by the SDK smoke example | Documentation reference |
| [Alibaba Cloud Model Studio OpenAI compatibility](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope) | Regional compatible base URLs and chat-completions request contract | Documentation reference |
| [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/) | Compatible JSON-object response mode and empty-output caveat | Documentation reference |
| [LangSmith evaluation types](https://docs.langchain.com/langsmith/evaluation-types) | Curated gold datasets, offline benchmarking, backtesting, and online-to-offline feedback loops | Documentation reference |
| [OpenAI Evals API](https://platform.openai.com/docs/api-reference/evals) | Versioned evaluation runs, labelled data items, grader results, and usage evidence | Documentation reference |

Before copying any implementation, re-check the exact file's current license and preserve required
copyright and attribution notices. Arize Phoenix is useful as a product reference but currently uses
the Elastic License 2.0, so this repository does not use it as a code source.
