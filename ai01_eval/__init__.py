"""
ai01-eval — Benchmark your AI agent against the AI01 leaderboard.

Quick start::

    from ai01_eval import AI01Eval, experiment_timer

    client = AI01Eval(api_key="your-api-key")

    # List datasets
    for ds in client.datasets.list():
        print(ds["id"], "-", ds["name"])

    # Download a dataset, time your pipeline, and submit results
    dataset = client.datasets.get("general-single-topic-v1")

    results = []
    with experiment_timer() as t:
        for item in dataset:
            answer = your_agent.run(item["query"], item.get("context"))
            results.append({
                "id":     item["id"],
                "query":  item["query"],
                "answer": answer,
            })

    # Submit and get scores — references are looked up server-side
    run = client.submit(
        dataset="general-single-topic-v1",
        results=results,
        agent_name="My Agent v1",
        experiment_name="RAG baseline",
        description="First run with basic chunking",
        duration_seconds=t["duration_seconds"],
    )
    print(run.scores)
    print(f"Time taken: {run.duration_seconds:.1f}s")
    print(run.report_url)

Environment variables
---------------------
``AI01_API_KEY``
    Your experiment API key — used when *api_key* is not passed explicitly.
``AI01_BASE_URL``
    Override the API base URL (e.g. ``http://localhost:8000`` for local dev).
"""
from __future__ import annotations

import os
from typing import Optional

from ai01_eval.datasets import DatasetClient
from ai01_eval.exceptions import (
    AI01AuthError,
    AI01Error,
    AI01NotFoundError,
    AI01RateLimitError,
    AI01ServerError,
)
from ai01_eval.submit import RunReport, RunsClient, SubmitClient, experiment_timer

__version__ = "0.2.0"

_DEFAULT_BASE_URL = "https://api.ai01.dev"


class AI01Eval:
    """
    Main entry point for the ai01-eval package.

    :param api_key: Your AI01 experiment API key. Falls back to the
        ``AI01_API_KEY`` environment variable when not provided.
    :param base_url: Override the API base URL. Falls back to
        ``AI01_BASE_URL``, then ``https://api.ai01.dev``.
    :raises AI01AuthError: If no API key is found.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("AI01_API_KEY")
        if not resolved_key:
            raise AI01AuthError(
                "api_key is required. Pass it explicitly or set the "
                "AI01_API_KEY environment variable."
            )
        resolved_url = base_url or os.environ.get("AI01_BASE_URL") or _DEFAULT_BASE_URL

        self.datasets = DatasetClient(resolved_url, resolved_key)
        self.runs = RunsClient(resolved_url, resolved_key)
        self._submit_client = SubmitClient(resolved_url, resolved_key)

    def submit(
        self,
        *,
        dataset: str,
        results: list[dict],
        agent_name: str,
        submitter: str = "anonymous",
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> RunReport:
        """Shortcut for :meth:`ai01_eval.submit.SubmitClient.submit`."""
        return self._submit_client.submit(
            dataset=dataset,
            results=results,
            agent_name=agent_name,
            submitter=submitter,
            experiment_name=experiment_name,
            description=description,
            duration_seconds=duration_seconds,
        )


__all__ = [
    "AI01Eval",
    "RunReport",
    "experiment_timer",
    "__version__",
    # exceptions
    "AI01Error",
    "AI01AuthError",
    "AI01NotFoundError",
    "AI01RateLimitError",
    "AI01ServerError",
]
