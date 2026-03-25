"""Tests for ai01_eval.submit — RunReport, experiment_timer, RunsClient, SubmitClient."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from ai01_eval.exceptions import AI01AuthError, AI01NotFoundError, AI01RateLimitError, AI01ServerError
from ai01_eval.submit import RunReport, RunsClient, SubmitClient, experiment_timer
from tests.conftest import RUN_REPORT_DATA, SAMPLE_RESULTS, err, ok

BASE_URL = "https://api.ai01.dev"
API_KEY = "test-key-abc"


# ---------------------------------------------------------------------------
# RunReport
# ---------------------------------------------------------------------------

class TestRunReport:
    def _make(self, overrides=None):
        data = dict(RUN_REPORT_DATA)
        if overrides:
            data.update(overrides)
        return RunReport(data)

    def test_id(self):
        assert self._make().id == "abc12345"

    def test_scores(self):
        scores = self._make().scores
        assert scores["exact_match"] == pytest.approx(0.71)
        assert scores["f1"] == pytest.approx(0.84)

    def test_report_url(self):
        assert self._make().report_url == "https://ai01.dev/benchmark?run=abc12345"

    def test_duration_seconds(self):
        assert self._make().duration_seconds == pytest.approx(12.5)

    def test_duration_seconds_none_when_missing(self):
        assert self._make({"duration_seconds": None}).duration_seconds is None

    def test_submitted_at(self):
        assert self._make().submitted_at == "2024-01-01T00:00:00+00:00"

    def test_repr_includes_id(self):
        assert "abc12345" in repr(self._make())

    def test_repr_includes_scores(self):
        assert "exact_match" in repr(self._make())

    def test_repr_includes_duration_when_present(self):
        assert "duration=" in repr(self._make())

    def test_repr_no_duration_when_missing(self):
        r = repr(self._make({"duration_seconds": None}))
        assert "duration=" not in r


# ---------------------------------------------------------------------------
# experiment_timer
# ---------------------------------------------------------------------------

class TestExperimentTimer:
    def test_populates_duration_seconds(self):
        with experiment_timer() as t:
            pass
        assert "duration_seconds" in t

    def test_duration_is_non_negative(self):
        with experiment_timer() as t:
            pass
        assert t["duration_seconds"] >= 0.0

    def test_duration_is_float(self):
        with experiment_timer() as t:
            pass
        assert isinstance(t["duration_seconds"], float)

    def test_duration_reflects_elapsed_time(self):
        with experiment_timer() as t:
            time.sleep(0.05)
        assert t["duration_seconds"] >= 0.05

    def test_duration_populated_even_on_exception(self):
        t = {}
        try:
            with experiment_timer() as t:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert "duration_seconds" in t
        assert t["duration_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# RunsClient
# ---------------------------------------------------------------------------

class TestRunsClient:
    def _client(self):
        return RunsClient(BASE_URL, API_KEY)

    def test_get_returns_run_report(self):
        with patch("ai01_eval.submit.requests.get", return_value=ok(RUN_REPORT_DATA)):
            report = self._client().get("abc12345")
        assert isinstance(report, RunReport)
        assert report.id == "abc12345"

    def test_get_calls_correct_url(self):
        with patch("ai01_eval.submit.requests.get", return_value=ok(RUN_REPORT_DATA)) as mock_get:
            self._client().get("abc12345")
        url = mock_get.call_args[0][0]
        assert url == f"{BASE_URL}/submissions/abc12345"

    def test_get_sends_auth_header(self):
        with patch("ai01_eval.submit.requests.get", return_value=ok(RUN_REPORT_DATA)) as mock_get:
            self._client().get("abc12345")
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {API_KEY}"

    def test_get_strips_trailing_slash(self):
        client = RunsClient(BASE_URL + "/", API_KEY)
        with patch("ai01_eval.submit.requests.get", return_value=ok(RUN_REPORT_DATA)) as mock_get:
            client.get("abc12345")
        url = mock_get.call_args[0][0]
        assert "//" not in url.replace("https://", "")

    def test_get_404_raises_not_found(self):
        with patch("ai01_eval.submit.requests.get", return_value=err(404, "Run not found")):
            with pytest.raises(AI01NotFoundError, match="Run not found"):
                self._client().get("bad-id")

    def test_get_401_raises_auth_error(self):
        with patch("ai01_eval.submit.requests.get", return_value=err(401)):
            with pytest.raises(AI01AuthError):
                self._client().get("abc12345")


# ---------------------------------------------------------------------------
# SubmitClient
# ---------------------------------------------------------------------------

class TestSubmitClient:
    def _client(self):
        return SubmitClient(BASE_URL, API_KEY)

    # --- success ---

    def test_submit_returns_run_report(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)):
            report = self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        assert isinstance(report, RunReport)

    def test_submit_calls_correct_url(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        url = mock_post.call_args[0][0]
        assert url == f"{BASE_URL}/submissions"

    def test_submit_sends_auth_header(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {API_KEY}"

    def test_submit_payload_contains_api_key(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["api_key"] == API_KEY

    def test_submit_payload_contains_dataset(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["dataset"] == "general-single-topic-v1"

    def test_submit_payload_contains_agent_name(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["agent_name"] == "My Agent"

    def test_submit_payload_contains_results(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["results"] == SAMPLE_RESULTS

    def test_submit_default_submitter_is_anonymous(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["metadata"]["submitter"] == "anonymous"

    def test_submit_custom_submitter(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
                submitter="alice",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["metadata"]["submitter"] == "alice"

    def test_submit_optional_experiment_name_included_when_given(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
                experiment_name="baseline",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["experiment_name"] == "baseline"

    def test_submit_optional_experiment_name_absent_when_not_given(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert "experiment_name" not in payload

    def test_submit_optional_description_included_when_given(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
                description="first run",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["description"] == "first run"

    def test_submit_optional_duration_included_when_given(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
                duration_seconds=42.5,
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["duration_seconds"] == pytest.approx(42.5)

    def test_submit_optional_duration_absent_when_not_given(self):
        with patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)) as mock_post:
            self._client().submit(
                dataset="general-single-topic-v1",
                results=SAMPLE_RESULTS,
                agent_name="My Agent",
            )
        payload = mock_post.call_args[1]["json"]
        assert "duration_seconds" not in payload

    # --- input validation ---

    def test_empty_results_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            self._client().submit(
                dataset="general-single-topic-v1",
                results=[],
                agent_name="My Agent",
            )

    def test_missing_id_raises_value_error(self):
        bad = [{"query": "Q?", "answer": "A."}]
        with pytest.raises(ValueError, match="id"):
            self._client().submit(
                dataset="ds",
                results=bad,
                agent_name="My Agent",
            )

    def test_missing_query_raises_value_error(self):
        bad = [{"id": "q1", "answer": "A."}]
        with pytest.raises(ValueError, match="query"):
            self._client().submit(
                dataset="ds",
                results=bad,
                agent_name="My Agent",
            )

    def test_missing_answer_raises_value_error(self):
        bad = [{"id": "q1", "query": "Q?"}]
        with pytest.raises(ValueError, match="answer"):
            self._client().submit(
                dataset="ds",
                results=bad,
                agent_name="My Agent",
            )

    def test_validation_reports_item_index(self):
        results = [
            {"id": "q1", "query": "Q1?", "answer": "A1."},
            {"id": "q2", "query": "Q2?"},  # missing answer
        ]
        with pytest.raises(ValueError, match=r"results\[1\]"):
            self._client().submit(dataset="ds", results=results, agent_name="My Agent")

    # --- HTTP errors ---

    def test_submit_401_raises_auth_error(self):
        with patch("ai01_eval.submit.requests.post", return_value=err(401)):
            with pytest.raises(AI01AuthError):
                self._client().submit(
                    dataset="ds",
                    results=SAMPLE_RESULTS,
                    agent_name="My Agent",
                )

    def test_submit_429_raises_rate_limit_error(self):
        with patch("ai01_eval.submit.requests.post", return_value=err(429)):
            with pytest.raises(AI01RateLimitError):
                self._client().submit(
                    dataset="ds",
                    results=SAMPLE_RESULTS,
                    agent_name="My Agent",
                )

    def test_submit_500_raises_server_error(self):
        with patch("ai01_eval.submit.requests.post", return_value=err(500)):
            with pytest.raises(AI01ServerError):
                self._client().submit(
                    dataset="ds",
                    results=SAMPLE_RESULTS,
                    agent_name="My Agent",
                )
