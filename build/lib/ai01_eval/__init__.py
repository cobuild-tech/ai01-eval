"""
ai01-eval — Benchmark your AI agent against the AI01 leaderboard.

Quick start::

    from ai01_eval import AI01Eval

    client = AI01Eval(api_key="your-api-key")

    # List datasets
    print(client.datasets.list())

    # Download a dataset and run your pipeline
    dataset = client.datasets.get("rag-retrieval-v1")
    results = []
    for item in dataset:
        answer = your_agent.run(item["query"], item.get("context"))
        results.append({
            "id":        item["id"],
            "query":     item["query"],
            "answer":    answer,
            "reference": item["reference"],
        })

    # Submit and get scores
    run = client.submit(
        dataset="rag-retrieval-v1",
        results=results,
        agent_name="My Agent v1",
    )
    print(run.scores)
    print(run.report_url)
"""
from __future__ import annotations

from ai01_eval.datasets import DatasetClient
from ai01_eval.submit import RunReport, RunsClient, SubmitClient

_DEFAULT_BASE_URL = "https://api.ai01.dev"


class AI01Eval:
    """
    Main entry point for the ai01-eval package.

    :param api_key: Your AI01 API key (create one at https://ai01.dev).
    :param base_url: Override the API base URL (useful for local testing).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self.datasets = DatasetClient(base_url, api_key)
        self.runs = RunsClient(base_url, api_key)
        self._submit_client = SubmitClient(base_url, api_key)

    def submit(
        self,
        *,
        dataset: str,
        results: list[dict],
        agent_name: str,
        submitter: str = "anonymous",
    ) -> RunReport:
        """Shortcut for :meth:`ai01_eval.submit.SubmitClient.submit`."""
        return self._submit_client.submit(
            dataset=dataset,
            results=results,
            agent_name=agent_name,
            submitter=submitter,
        )


__all__ = ["AI01Eval", "RunReport"]
