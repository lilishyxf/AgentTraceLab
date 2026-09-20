from __future__ import annotations

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


def send_sample_trace(endpoint: str, *, timeout: float = 5.0) -> str:
    resource = Resource.create(
        {
            SERVICE_NAME: "agenttracelab-otel-example",
            SERVICE_VERSION: "1.1.0",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, timeout=timeout)
    processor = BatchSpanProcessor(exporter, max_export_batch_size=64)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("agenttracelab.examples", "1.1.0")

    try:
        with tracer.start_as_current_span("office-agent-run") as agent_span:
            trace_id = f"{agent_span.get_span_context().trace_id:032x}"
            agent_span.set_attribute("openinference.span.kind", "AGENT")
            agent_span.set_attribute("input.value", "Inspect the table before answering.")
            agent_span.set_attribute("agent.role", "analyst")
            agent_span.set_attribute("agent.observation", "The table schema is unknown.")
            agent_span.set_attribute("agent.selected_action", "Inspect the table headers.")
            agent_span.set_attribute(
                "agent.reason",
                "The answer depends on the columns observed from the tool.",
            )

            with tracer.start_as_current_span("inspect_table") as tool_span:
                tool_call_id = f"{tool_span.get_span_context().span_id:016x}"
                tool_span.set_attribute("openinference.span.kind", "TOOL")
                tool_span.set_attribute("tool.name", "inspect_table")
                tool_span.set_attribute(
                    "tool.output",
                    "headers=order_id,total,status,owner",
                )
                tool_span.set_status(Status(StatusCode.OK))

            agent_span.set_attribute("agent.tool_call_id", tool_call_id)
            agent_span.set_attribute(
                "agent.validation",
                "The tool returned four named columns.",
            )
            agent_span.set_attribute("agent.disposition", "stop")
            agent_span.set_status(Status(StatusCode.OK))

        if not provider.force_flush(timeout_millis=int(timeout * 1_000)):
            raise RuntimeError("OpenTelemetry exporter did not flush before the timeout")
        return trace_id
    finally:
        provider.shutdown()
