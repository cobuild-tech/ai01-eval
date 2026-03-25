"""
Submission helpers for ai01-eval.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

import requests

from ai01_eval.exceptions import raise_for_status


class RunReport:
    """Result of a submitted evaluation run."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def id(self) -> str:
        return self._data["run_id"]

    @property
    def scores(self) -> dict[str, float]:
        return self._data["scores"]

    @property
    def report_url(self) -> str:
        return self._data["report_url"]

    @property
    def duration_seconds(self) -> Optional[float]:
        return self._data.get("duration_seconds")

    @property
    def submitted_at(self) -> str:
        return self._data["submitted_at"]

    def __repr__(self) -> str:
        dur = (
            f" duration={self.duration_seconds:.1f}s"
            if self.duration_seconds is not None
            else ""
        )
        return f"<RunReport id={self.id!r} scores={self.scores}{dur}>"


@contextmanager
def experiment_timer() -> Generator[dict, None, None]:
    """
    Context manager that measures how long your agent loop takes.

    Usage::

        with experiment_timer() as t:
            for item in dataset:
                results.append(run_agent(item))

        run = client.submit(..., duration_seconds=t["duration_seconds"])
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duration_seconds"] = time.perf_counter() - start


class RunsClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def get(self, run_id: str) -> RunReport:
        """
        Retrieve a past submission report by run ID.

        :raises AI01NotFoundError: If the run ID does not exist.
        :raises AI01AuthError: If the API key is invalid.
        """
        resp = requests.get(
            f"{self._base_url}/submissions/{run_id}",
            headers=self._headers,
            timeout=30,
        )
        raise_for_status(resp)
        return RunReport(resp.json())


class SubmitClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def submit(
        self,
        *,
        dataset: str,
        results: list[dict[str, Any]],
        agent_name: str,
        submitter: str = "anonymous",
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> RunReport:
        """
        Submit a list of result dicts to the AI01 eval server.

        Each dict must contain:

        - ``id``     — matches the item ID from the dataset
        - ``query``  — the original query string
        - ``answer`` — your agent's answer

        Ground-truth references are looked up server-side; you do not need
        to include a ``reference`` field.

        Optional submit parameters:

        - ``experiment_name``  — label for this experiment run
        - ``description``      — free-text notes about this run
        - ``duration_seconds`` — pipeline wall-clock time; use
          :func:`experiment_timer` to measure this automatically

        :raises ValueError: If *results* is empty or any item is missing
            a required key.
        :raises AI01AuthError: If the API key is invalid.
        :raises AI01RateLimitError: If too many requests are sent.
        :raises AI01ServerError: For unexpected server errors.
        """
        if not results:
            raise ValueError("results must not be empty.")

        required_keys = {"id", "query", "answer"}
        for i, item in enumerate(results):
            missing = required_keys - item.keys()
            if missing:
                raise ValueError(
                    f"results[{i}] is missing required keys: {sorted(missing)}"
                )

        submitted_at = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "dataset": dataset,
            "agent_name": agent_name,
            "results": results,
            "api_key": self._api_key,
            "submitted_at": submitted_at,
            "metadata": {"submitter": submitter},
        }
        if experiment_name is not None:
            payload["experiment_name"] = experiment_name
        if description is not None:
            payload["description"] = description
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds

        resp = requests.post(
            f"{self._base_url}/submissions",
            json=payload,
            headers=self._headers,
            timeout=120,
        )
        raise_for_status(resp)
        return RunReport(resp.json())
