"""Tests for ai01_eval.exceptions."""
from __future__ import annotations

import pytest

from ai01_eval.exceptions import (
    AI01AuthError,
    AI01Error,
    AI01NotFoundError,
    AI01RateLimitError,
    AI01ServerError,
    raise_for_status,
)
from tests.conftest import err, make_response


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_auth_error_is_ai01_error(self):
        assert issubclass(AI01AuthError, AI01Error)

    def test_not_found_error_is_ai01_error(self):
        assert issubclass(AI01NotFoundError, AI01Error)

    def test_rate_limit_error_is_ai01_error(self):
        assert issubclass(AI01RateLimitError, AI01Error)

    def test_server_error_is_ai01_error(self):
        assert issubclass(AI01ServerError, AI01Error)

    def test_all_are_exceptions(self):
        for cls in (AI01Error, AI01AuthError, AI01NotFoundError, AI01RateLimitError, AI01ServerError):
            assert issubclass(cls, Exception)


# ---------------------------------------------------------------------------
# raise_for_status — success cases
# ---------------------------------------------------------------------------

class TestRaiseForStatusSuccess:
    def test_200_does_not_raise(self):
        raise_for_status(make_response(200, json_data={}))

    def test_201_does_not_raise(self):
        raise_for_status(make_response(201, json_data={}))

    def test_204_does_not_raise(self):
        raise_for_status(make_response(204))

    def test_301_does_not_raise(self):
        raise_for_status(make_response(301))


# ---------------------------------------------------------------------------
# raise_for_status — error cases
# ---------------------------------------------------------------------------

class TestRaiseForStatusErrors:
    def test_401_raises_auth_error(self):
        with pytest.raises(AI01AuthError):
            raise_for_status(err(401, "Invalid API key"))

    def test_401_message_includes_detail(self):
        with pytest.raises(AI01AuthError, match="Invalid API key"):
            raise_for_status(err(401, "Invalid API key"))

    def test_404_raises_not_found(self):
        with pytest.raises(AI01NotFoundError):
            raise_for_status(err(404, "Dataset not found"))

    def test_404_message_is_detail(self):
        with pytest.raises(AI01NotFoundError, match="Dataset not found"):
            raise_for_status(err(404, "Dataset not found"))

    def test_429_raises_rate_limit(self):
        with pytest.raises(AI01RateLimitError):
            raise_for_status(err(429))

    def test_429_message_mentions_rate_limit(self):
        with pytest.raises(AI01RateLimitError, match="[Rr]ate limit"):
            raise_for_status(err(429))

    def test_500_raises_server_error(self):
        with pytest.raises(AI01ServerError):
            raise_for_status(err(500, "Internal server error"))

    def test_500_message_includes_status_code(self):
        with pytest.raises(AI01ServerError, match="500"):
            raise_for_status(err(500, "Internal server error"))

    def test_400_raises_server_error(self):
        with pytest.raises(AI01ServerError):
            raise_for_status(err(400, "Bad request"))

    def test_503_raises_server_error(self):
        with pytest.raises(AI01ServerError):
            raise_for_status(err(503))

    def test_non_json_response_falls_back_to_text(self):
        resp = make_response(500, text="plain error text")
        with pytest.raises(AI01ServerError, match="plain error text"):
            raise_for_status(resp)

    def test_non_json_response_falls_back_to_reason(self):
        resp = make_response(500)
        resp.text = ""
        resp.reason = "Internal Server Error"
        with pytest.raises(AI01ServerError):
            raise_for_status(resp)

    def test_401_is_also_catchable_as_ai01_error(self):
        with pytest.raises(AI01Error):
            raise_for_status(err(401))

    def test_404_is_also_catchable_as_ai01_error(self):
        with pytest.raises(AI01Error):
            raise_for_status(err(404))
