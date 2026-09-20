from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def complete_journal() -> dict:
    path = Path(__file__).parent / "fixtures" / "wasmhatch_complete.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def otlp_payload() -> dict:
    path = Path(__file__).parent / "fixtures" / "openinference_otlp.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def incomplete_journal(complete_journal: dict) -> dict:
    payload = copy.deepcopy(complete_journal)
    payload["runId"] = "run_journal_22222222222222222222222222222222"
    payload["state"] = "active"
    payload["events"][0]["evidence"] = {}
    payload["events"][3]["outcome"] = "rejected"
    payload["events"][5]["summary"] = "Tool returned without validation"
    return payload
