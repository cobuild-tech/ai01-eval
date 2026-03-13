"""
Dataset access helpers for ai01-eval.
Downloads dataset items from the AI01 API so users can iterate over them
and pass each query through their pipeline.
"""
from __future__ import annotations

from typing import Any, Iterator
import requests


class Dataset:
    """A lazily-loaded benchmark dataset."""

    def __init__(self, data: dict[str, Any], items: list[dict[str, Any]]) -> None:
        self._meta = data
        self._items = items

    # metadata shortcuts
    @property
    def id(self) -> str:
        return self._meta["id"]

    @property
    def name(self) -> str:
        return self._meta["name"]

    @property
    def num_queries(self) -> int:
        return self._meta["num_queries"]

    @property
    def metrics(self) -> list[str]:
        return self._meta["metrics"]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"<Dataset id={self.id!r} queries={len(self._items)}>"


class DatasetClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all available datasets."""
        resp = requests.get(f"{self._base_url}/datasets", headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["datasets"]

    def get(self, dataset_id: str) -> Dataset:
        """
        Download a dataset by ID.

        Returns a :class:`Dataset` that you can iterate over.
        Each item contains: ``id``, ``query``.
        RAG datasets also include a ``context`` field.
        References are kept server-side and used only for metric computation.
        """
        # Fetch metadata
        meta_resp = requests.get(
            f"{self._base_url}/datasets/{dataset_id}",
            headers=self._headers,
            timeout=30,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        # Fetch items (paginated endpoint)
        items_resp = requests.get(
            f"{self._base_url}/datasets/{dataset_id}/items",
            headers=self._headers,
            timeout=60,
        )
        items_resp.raise_for_status()
        items_data = items_resp.json()
        items = items_data.get("items", [])

        # For RAG datasets the corpus is returned once at the top level.
        # Inject it into each item so callers can use item["text_corpus"].
        text_corpus = items_data.get("text_corpus")
        if text_corpus:
            for item in items:
                item["text_corpus"] = text_corpus

        return Dataset(meta, items)
