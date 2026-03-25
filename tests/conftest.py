"""
Shared test helpers for ai01-eval.
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

DATASET_META = {
    "id": "general-single-topic-v1",
    "name": "General Single Topic",
    "num_queries": 3,
    "metrics": ["exact_match", "f1", "faithfulness"],
}

DATASET_ITEMS = [
    {"id": "q1", "query": "What is photosynthesis?"},
    {"id": "q2", "query": "What is mitosis?"},
    {"id": "q3", "query": "What is osmosis?"},
]

RAG_ITEMS_RESPONSE = {
    "dataset": "general-single-topic-v1",
    "items": DATASET_ITEMS,
    "total": 3,
    "text_corpus": "Biology is the study of living organisms...",
}

PLAIN_ITEMS_RESPONSE = {
    "dataset": "general-single-topic-v1",
    "items": DATASET_ITEMS,
    "total": 3,
}

DATASETS_LIST_RESPONSE = {
    "datasets": [DATASET_META],
}

RUN_REPORT_DATA = {
    "run_id": "abc12345",
    "scores": {"exact_match": 0.71, "f1": 0.84, "faithfulness": 0.79},
    "report_url": "https://ai01.dev/benchmark?run=abc12345",
    "submitted_at": "2024-01-01T00:00:00+00:00",
    "duration_seconds": 12.5,
}

SAMPLE_RESULTS = [
    {"id": "q1", "query": "What is photosynthesis?", "answer": "A process plants use to make food."},
    {"id": "q2", "query": "What is mitosis?", "answer": "Cell division producing two identical cells."},
]


# ---------------------------------------------------------------------------
# Response mock factory
# ---------------------------------------------------------------------------

def make_response(status_code: int = 200, json_data=None, text: str = "") -> MagicMock:
    """Return a mock that looks like a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.reason = "OK" if status_code < 400 else "Error"
    if json_data is not None:
        resp.json.return_value = copy.deepcopy(json_data)
    else:
        resp.json.side_effect = ValueError("No JSON body")
    return resp


def ok(json_data) -> MagicMock:
    """Shorthand for a 200 response with JSON body."""
    return make_response(200, json_data)


def err(status_code: int, detail: str = "Something went wrong") -> MagicMock:
    """Shorthand for an error response with a JSON detail field."""
    return make_response(status_code, json_data={"detail": detail})
