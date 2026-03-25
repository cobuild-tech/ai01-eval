"""
Dataset access helpers for ai01-eval.

Downloads dataset items from the AI01 API so users can iterate over them
and pass each query through their pipeline.

Note: ground-truth references are kept server-side and are never returned
in the items payload. They are used only for metric computation at
submission time.
"""
from __future__ import annotations

from typing import Any, Iterator

import requests

from ai01_eval.exceptions import raise_for_status


class Dataset:
    """A downloaded benchmark dataset, ready to iterate over."""

    def __init__(self, data: dict[str, Any], items: list[dict[str, Any]]) -> None:
        self._meta = data
        self._items = items

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
        resp = requests.get(
            f"{self._base_url}/datasets",
            headers=self._headers,
            timeout=30,
        )
        raise_for_status(resp)
        return resp.json()["datasets"]

    def get(self, dataset_id: str) -> Dataset:
        """
        Download a dataset by ID.

        Returns a :class:`Dataset` that you can iterate over.
        Each item contains ``id`` and ``query``; RAG datasets also include
        a ``context`` field and a ``text_corpus`` field on the object.

        Ground-truth references are **not** included in items — they are
        looked up server-side when you submit results.

        :param dataset_id: The dataset ID string (e.g. ``"general-single-topic-v1"``).
        :raises AI01NotFoundError: If the dataset does not exist.
        :raises AI01AuthError: If the API key is invalid.
        """
        meta_resp = requests.get(
            f"{self._base_url}/datasets/{dataset_id}",
            headers=self._headers,
            timeout=30,
        )
        raise_for_status(meta_resp)
        meta = meta_resp.json()

        items_resp = requests.get(
            f"{self._base_url}/datasets/{dataset_id}/items",
            headers=self._headers,
            timeout=60,
        )
        raise_for_status(items_resp)
        items_data = items_resp.json()
        items = items_data.get("items", [])

        # For RAG datasets the corpus is returned once at the top level.
        # Inject it into each item so callers can use item["text_corpus"].
        text_corpus = items_data.get("text_corpus")
        if text_corpus:
            for item in items:
                item["text_corpus"] = text_corpus

        return Dataset(meta, items)
