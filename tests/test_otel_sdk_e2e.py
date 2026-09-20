from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import uvicorn

from agenttracelab.api import create_app
from agenttracelab.otel_sample import send_sample_trace


@contextmanager
def _running_server(database_path: Path) -> Iterator[str]:
    app = create_app(f"sqlite+pysqlite:///{database_path.as_posix()}")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("AgentTraceLab test server did not start")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("AgentTraceLab test server did not stop")


def test_python_otel_sdk_exports_protobuf_trace_end_to_end(tmp_path: Path) -> None:
    with _running_server(tmp_path / "otel-sdk-e2e.db") as base_url:
        trace_id = send_sample_trace(f"{base_url}/v1/traces")

        trace_response = httpx.get(f"{base_url}/v1/traces/{trace_id}", timeout=5)
        evaluation_response = httpx.get(
            f"{base_url}/v1/evaluations/{trace_id}/latest",
            timeout=5,
        )

    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["source"]["kind"] == "openinference"
    assert [span["kind"] for span in trace["spans"]] == ["AGENT", "TOOL"]
    assert trace["decision_evidence"][0]["tool_call_id"] == trace["spans"][1]["span_id"]
    assert evaluation_response.status_code == 200
    assert evaluation_response.json()["passed"] is True
