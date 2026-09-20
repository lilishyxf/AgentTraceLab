from __future__ import annotations

import argparse
import os

from agenttracelab.otel_sample import send_sample_trace

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/traces"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a real OpenTelemetry SDK Agent trace to AgentTraceLab.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", DEFAULT_ENDPOINT),
        help="Full OTLP/HTTP traces endpoint.",
    )
    args = parser.parse_args()
    trace_id = send_sample_trace(args.endpoint)
    print(trace_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
