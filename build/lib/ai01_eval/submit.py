"""
Submission helpers for ai01-eval.
"""
from __future__ import annotations

from typing import Any
import requests


class RunReport:
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

    def __repr__(self) -> str:
        return f"<RunReport id={self.id!r} scores={self.scores}>"


class RunsClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def get(self, run_id: str) -> RunReport:
        resp = requests.get(
            f"{self._base_url}/submissions/{run_id}",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return RunReport(resp.json())


class SubmitClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
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
    ) -> RunReport:
        """
        Submit a list of result dicts to the AI01 eval server.

        Each dict must contain:
          - ``id``        : matches the item id from the dataset
          - ``query``     : the original query string
          - ``answer``    : your agent's answer
          - ``reference`` : the ground-truth answer

        Returns a :class:`RunReport` with scores and a URL to the full report.
        """
        payload = {
            "dataset": dataset,
            "agent_name": agent_name,
            "results": results,
            "metadata": {"submitter": submitter},
        }
        resp = requests.post(
            f"{self._base_url}/submissions",
            json=payload,
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        return RunReport(resp.json())
