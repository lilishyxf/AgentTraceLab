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
| [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Task/trial/grader/outcome distinctions, reference solutions, balanced sets, repeated trials, transcript review, and human calibration | Documentation reference |
| [Anthropic: Quantifying infrastructure noise](https://www.anthropic.com/engineering/infrastructure-noise) | Resource configuration and infrastructure failure rates as first-class evaluation variables | Documentation reference |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | Reproducible tasks, scorers, epochs, sandbox isolation, logs, and re-scoring boundaries | MIT |
| [Can Agent Benchmarks Support Their Scores?](https://arxiv.org/abs/2605.10448) | Outcome-evidence Pass/Fail/Unknown labels and evidence-supported score bounds for incomplete verification | Paper reference |
| [TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents](https://arxiv.org/abs/2602.21230) | Accuracy, efficiency, evidence grounding, reasoning quality, and scaffolded capability as distinct dimensions | Paper reference |
| [AgentEval: DAG-Structured Step-Level Evaluation](https://arxiv.org/abs/2604.23581) | Dependency-aware step evaluation, failure propagation, calibrated Judges, and root-cause attribution | Paper reference |
| [Inspect scoring policy](https://inspect.aisi.org.uk/scoring-policy.html) | Separating model, grader, scorer, and execution failures instead of collapsing them into one score | Documentation reference |
| [OpenAI Evals API](https://platform.openai.com/docs/api-reference/evals) | Versioned evaluation runs, labelled data items, grader results, and usage evidence | Documentation reference |
| [Agent Lightning v1.0 schemas](https://github.com/microsoft/agent-lightning/blob/ff9457587fb6ec900e16e93be9ad2d77409afa08/agentlightning/schemas.py) | Rollout `EventCreate` and scalar reward-event contract pinned during v1.4 development | MIT |
| [Agent Lightning v1.0 basics](https://github.com/microsoft/agent-lightning/blob/ff9457587fb6ec900e16e93be9ad2d77409afa08/docs/05-basics.md) | Gateway, rollout, event, Controller, and trainer boundaries pinned during v1.4 development | MIT |
| [GEPA `optimize_anything`](https://github.com/gepa-ai/gepa/blob/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/optimize_anything.py) | Pinned evaluator `(score, info)` contract and actionable side information | MIT |
| [GEPA FAQ](https://github.com/gepa-ai/gepa/blob/15ee314f9c7d34ec153b809d401f42f55c4dcd76/docs/docs/guides/faq.md) | Higher-is-better `info.scores` objectives and Pareto tracking guidance | MIT |
| [GEPA v0.1.4 FAQ](https://github.com/gepa-ai/gepa/blob/v0.1.4/docs/docs/guides/faq.md) | Tagged train/validation split and held-out validation guidance used by the v1.7 promotion gate | MIT |
| [GEPA 0.1.4 on PyPI](https://pypi.org/project/gepa/0.1.4/) | Exact optional runtime version used by the v1.6 candidate-search integration | MIT |
| [NIST/SEMATECH confidence limits for proportions](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) | Statistical reference for proportion confidence limits; v1.8 implements a bounded Wilson lower limit directly | Public technical reference |

Before copying any implementation, re-check the exact file's current license and preserve required
copyright and attribution notices. Arize Phoenix is useful as a product reference but currently uses
the Elastic License 2.0, so this repository does not use it as a code source.
