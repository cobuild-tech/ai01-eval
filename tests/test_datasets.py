"""Tests for ai01_eval.datasets — Dataset model and DatasetClient."""
from __future__ import annotations

from unittest.mock import call, patch

import pytest

from ai01_eval.datasets import Dataset, DatasetClient
from ai01_eval.exceptions import AI01AuthError, AI01NotFoundError, AI01ServerError
from tests.conftest import (
    DATASET_ITEMS,
    DATASET_META,
    DATASETS_LIST_RESPONSE,
    PLAIN_ITEMS_RESPONSE,
    RAG_ITEMS_RESPONSE,
    err,
    ok,
)

BASE_URL = "https://api.ai01.dev"
API_KEY = "test-key-123"


# ---------------------------------------------------------------------------
# Dataset model
# ---------------------------------------------------------------------------

class TestDataset:
    def _make(self, items=None):
        return Dataset(DATASET_META, list(DATASET_ITEMS) if items is None else items)

    def test_id(self):
        assert self._make().id == "general-single-topic-v1"

    def test_name(self):
        assert self._make().name == "General Single Topic"

    def test_num_queries(self):
        assert self._make().num_queries == 3

    def test_metrics(self):
        assert self._make().metrics == ["exact_match", "f1", "faithfulness"]

    def test_len(self):
        assert len(self._make()) == 3

    def test_len_empty(self):
        assert len(self._make(items=[])) == 0

    def test_iter_yields_all_items(self):
        items = list(self._make())
        assert len(items) == 3
        assert items[0]["id"] == "q1"
        assert items[2]["id"] == "q3"

    def test_iter_item_has_query(self):
        for item in self._make():
            assert "query" in item

    def test_repr_contains_id(self):
        assert "general-single-topic-v1" in repr(self._make())

    def test_repr_contains_query_count(self):
        assert "3" in repr(self._make())

    def test_repr_format(self):
        assert repr(self._make()) == "<Dataset id='general-single-topic-v1' queries=3>"

    def test_iterable_multiple_times(self):
        ds = self._make()
        first = list(ds)
        second = list(ds)
        assert first == second


# ---------------------------------------------------------------------------
# DatasetClient — list()
# ---------------------------------------------------------------------------

class TestDatasetClientList:
    def _client(self):
        return DatasetClient(BASE_URL, API_KEY)

    def test_list_returns_datasets(self):
        with patch("ai01_eval.datasets.requests.get", return_value=ok(DATASETS_LIST_RESPONSE)):
            result = self._client().list()
        assert len(result) == 1
        assert result[0]["id"] == "general-single-topic-v1"

    def test_list_calls_correct_url(self):
        with patch("ai01_eval.datasets.requests.get", return_value=ok(DATASETS_LIST_RESPONSE)) as mock_get:
            self._client().list()
        mock_get.assert_called_once()
        url = mock_get.call_args[0][0]
        assert url == f"{BASE_URL}/datasets"

    def test_list_sends_auth_header(self):
        with patch("ai01_eval.datasets.requests.get", return_value=ok(DATASETS_LIST_RESPONSE)) as mock_get:
            self._client().list()
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {API_KEY}"

    def test_list_strips_trailing_slash_from_base_url(self):
        client = DatasetClient(BASE_URL + "/", API_KEY)
        with patch("ai01_eval.datasets.requests.get", return_value=ok(DATASETS_LIST_RESPONSE)) as mock_get:
            client.list()
        url = mock_get.call_args[0][0]
        assert "//" not in url.replace("https://", "")

    def test_list_401_raises_auth_error(self):
        with patch("ai01_eval.datasets.requests.get", return_value=err(401, "Invalid key")):
            with pytest.raises(AI01AuthError):
                self._client().list()

    def test_list_500_raises_server_error(self):
        with patch("ai01_eval.datasets.requests.get", return_value=err(500)):
            with pytest.raises(AI01ServerError):
                self._client().list()


# ---------------------------------------------------------------------------
# DatasetClient — get()
# ---------------------------------------------------------------------------

class TestDatasetClientGet:
    def _client(self):
        return DatasetClient(BASE_URL, API_KEY)

    def _patch_get(self, meta_resp, items_resp):
        """Patch requests.get to return meta_resp then items_resp."""
        return patch(
            "ai01_eval.datasets.requests.get",
            side_effect=[meta_resp, items_resp],
        )

    def test_get_returns_dataset(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)):
            ds = self._client().get("general-single-topic-v1")
        assert isinstance(ds, Dataset)

    def test_get_dataset_id(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)):
            ds = self._client().get("general-single-topic-v1")
        assert ds.id == "general-single-topic-v1"

    def test_get_dataset_items_count(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)):
            ds = self._client().get("general-single-topic-v1")
        assert len(ds) == 3

    def test_get_calls_meta_url(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)) as mock_get:
            self._client().get("general-single-topic-v1")
        first_url = mock_get.call_args_list[0][0][0]
        assert first_url == f"{BASE_URL}/datasets/general-single-topic-v1"

    def test_get_calls_items_url(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)) as mock_get:
            self._client().get("general-single-topic-v1")
        second_url = mock_get.call_args_list[1][0][0]
        assert second_url == f"{BASE_URL}/datasets/general-single-topic-v1/items"

    def test_get_sends_auth_header_for_both_requests(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)) as mock_get:
            self._client().get("general-single-topic-v1")
        for c in mock_get.call_args_list:
            assert c[1]["headers"]["Authorization"] == f"Bearer {API_KEY}"

    def test_get_items_do_not_contain_reference(self):
        """References must never be present in downloaded items."""
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)):
            ds = self._client().get("general-single-topic-v1")
        for item in ds:
            assert "reference" not in item

    def test_get_rag_corpus_injected_into_items(self):
        with self._patch_get(ok(DATASET_META), ok(RAG_ITEMS_RESPONSE)):
            ds = self._client().get("general-single-topic-v1")
        for item in ds:
            assert item["text_corpus"] == "Biology is the study of living organisms..."

    def test_get_no_corpus_items_unchanged(self):
        with self._patch_get(ok(DATASET_META), ok(PLAIN_ITEMS_RESPONSE)):
            ds = self._client().get("general-single-topic-v1")
        for item in ds:
            assert "text_corpus" not in item

    def test_get_meta_404_raises_not_found(self):
        with patch("ai01_eval.datasets.requests.get", return_value=err(404, "Dataset not found")):
            with pytest.raises(AI01NotFoundError, match="Dataset not found"):
                self._client().get("nonexistent")

    def test_get_items_404_raises_not_found(self):
        with self._patch_get(ok(DATASET_META), err(404, "Items not found")):
            with pytest.raises(AI01NotFoundError):
                self._client().get("general-single-topic-v1")

    def test_get_meta_401_raises_auth_error(self):
        with patch("ai01_eval.datasets.requests.get", return_value=err(401)):
            with pytest.raises(AI01AuthError):
                self._client().get("general-single-topic-v1")

    def test_get_items_500_raises_server_error(self):
        with self._patch_get(ok(DATASET_META), err(500)):
            with pytest.raises(AI01ServerError):
                self._client().get("general-single-topic-v1")
