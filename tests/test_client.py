"""Tests for ai01_eval.AI01Eval — init, env vars, submit shortcut, public API."""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

import ai01_eval
from ai01_eval import (
    AI01AuthError,
    AI01Error,
    AI01Eval,
    AI01NotFoundError,
    AI01RateLimitError,
    AI01ServerError,
    RunReport,
    __version__,
    experiment_timer,
)
from ai01_eval.datasets import DatasetClient
from ai01_eval.submit import RunsClient, SubmitClient
from tests.conftest import DATASET_META, DATASETS_LIST_RESPONSE, PLAIN_ITEMS_RESPONSE, RUN_REPORT_DATA, SAMPLE_RESULTS, ok


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_is_defined(self):
        assert __version__ is not None

    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_is_semver_shaped(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Public exports (__all__)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_all_contains_ai01_eval(self):
        assert "AI01Eval" in ai01_eval.__all__

    def test_all_contains_run_report(self):
        assert "RunReport" in ai01_eval.__all__

    def test_all_contains_experiment_timer(self):
        assert "experiment_timer" in ai01_eval.__all__

    def test_all_contains_version(self):
        assert "__version__" in ai01_eval.__all__

    def test_all_contains_exception_classes(self):
        for name in ("AI01Error", "AI01AuthError", "AI01NotFoundError", "AI01RateLimitError", "AI01ServerError"):
            assert name in ai01_eval.__all__, f"{name} missing from __all__"

    def test_exceptions_importable_from_top_level(self):
        # Verify each exception class is accessible directly from the package.
        assert AI01Error is not None
        assert AI01AuthError is not None
        assert AI01NotFoundError is not None
        assert AI01RateLimitError is not None
        assert AI01ServerError is not None


# ---------------------------------------------------------------------------
# AI01Eval — initialisation
# ---------------------------------------------------------------------------

class TestAI01EvalInit:
    def test_explicit_api_key_accepted(self):
        client = AI01Eval(api_key="my-key")
        assert client is not None

    def test_no_api_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("AI01_API_KEY", raising=False)
        with pytest.raises(AI01AuthError, match="api_key"):
            AI01Eval()

    def test_empty_string_api_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("AI01_API_KEY", raising=False)
        with pytest.raises(AI01AuthError):
            AI01Eval(api_key="")

    def test_env_var_api_key_used_when_no_explicit_key(self, monkeypatch):
        monkeypatch.setenv("AI01_API_KEY", "env-key-xyz")
        client = AI01Eval()  # should not raise
        assert client is not None

    def test_explicit_key_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("AI01_API_KEY", "env-key")
        client = AI01Eval(api_key="explicit-key")
        # Verify the explicit key was used by inspecting the DatasetClient headers
        assert client.datasets._headers["Authorization"] == "Bearer explicit-key"

    def test_default_base_url_is_production(self, monkeypatch):
        monkeypatch.delenv("AI01_BASE_URL", raising=False)
        client = AI01Eval(api_key="k")
        assert "api.ai01.dev" in client.datasets._base_url

    def test_explicit_base_url_used(self, monkeypatch):
        monkeypatch.delenv("AI01_BASE_URL", raising=False)
        client = AI01Eval(api_key="k", base_url="http://localhost:8000")
        assert client.datasets._base_url == "http://localhost:8000"

    def test_env_var_base_url_used(self, monkeypatch):
        monkeypatch.setenv("AI01_BASE_URL", "http://localhost:9999")
        client = AI01Eval(api_key="k")
        assert "9999" in client.datasets._base_url

    def test_explicit_base_url_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("AI01_BASE_URL", "http://env-url:8000")
        client = AI01Eval(api_key="k", base_url="http://explicit-url:7000")
        assert "7000" in client.datasets._base_url
        assert "8000" not in client.datasets._base_url

    def test_datasets_attribute_is_dataset_client(self):
        client = AI01Eval(api_key="k")
        assert isinstance(client.datasets, DatasetClient)

    def test_runs_attribute_is_runs_client(self):
        client = AI01Eval(api_key="k")
        assert isinstance(client.runs, RunsClient)


# ---------------------------------------------------------------------------
# AI01Eval.submit() shortcut
# ---------------------------------------------------------------------------

class TestAI01EvalSubmit:
    def test_submit_delegates_to_submit_client(self):
        client = AI01Eval(api_key="k")
        mock_report = MagicMock(spec=RunReport)
        client._submit_client.submit = MagicMock(return_value=mock_report)

        result = client.submit(
            dataset="ds",
            results=SAMPLE_RESULTS,
            agent_name="My Agent",
        )

        client._submit_client.submit.assert_called_once_with(
            dataset="ds",
            results=SAMPLE_RESULTS,
            agent_name="My Agent",
            submitter="anonymous",
            experiment_name=None,
            description=None,
            duration_seconds=None,
        )
        assert result is mock_report

    def test_submit_passes_optional_params(self):
        client = AI01Eval(api_key="k")
        client._submit_client.submit = MagicMock(return_value=MagicMock(spec=RunReport))

        client.submit(
            dataset="ds",
            results=SAMPLE_RESULTS,
            agent_name="My Agent",
            submitter="alice",
            experiment_name="exp-1",
            description="notes",
            duration_seconds=5.0,
        )

        _, kwargs = client._submit_client.submit.call_args
        assert kwargs["submitter"] == "alice"
        assert kwargs["experiment_name"] == "exp-1"
        assert kwargs["description"] == "notes"
        assert kwargs["duration_seconds"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# End-to-end happy path (all HTTP calls mocked)
# ---------------------------------------------------------------------------

class TestEndToEndHappyPath:
    def test_full_workflow(self):
        """list → get → submit all succeed and return correct types."""
        client = AI01Eval(api_key="my-key", base_url="https://api.ai01.dev")

        with patch("ai01_eval.datasets.requests.get") as mock_ds_get, \
             patch("ai01_eval.submit.requests.post", return_value=ok(RUN_REPORT_DATA)):

            mock_ds_get.side_effect = [
                ok(DATASETS_LIST_RESPONSE),       # datasets.list()
                ok(DATASET_META),                  # datasets.get() → meta
                ok(PLAIN_ITEMS_RESPONSE),          # datasets.get() → items
            ]

            datasets = client.datasets.list()
            assert len(datasets) == 1

            dataset = client.datasets.get("general-single-topic-v1")
            assert len(dataset) == 3

            results = [
                {"id": item["id"], "query": item["query"], "answer": "answer"}
                for item in dataset
            ]
            run = client.submit(
                dataset="general-single-topic-v1",
                results=results,
                agent_name="Test Agent",
            )

        assert isinstance(run, RunReport)
        assert run.id == "abc12345"
        assert "f1" in run.scores
        assert run.report_url.startswith("https://")
