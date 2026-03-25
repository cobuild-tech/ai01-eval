"""
Custom exceptions for ai01-eval.

Catching ``AI01Error`` covers all errors raised by this library.
"""
from __future__ import annotations


class AI01Error(Exception):
    """Base class for all ai01-eval errors."""


class AI01AuthError(AI01Error):
    """Raised when the API key is missing or invalid (HTTP 401)."""


class AI01NotFoundError(AI01Error):
    """Raised when a requested resource does not exist (HTTP 404)."""


class AI01RateLimitError(AI01Error):
    """Raised when the server rate-limits the request (HTTP 429)."""


class AI01ServerError(AI01Error):
    """Raised for unexpected server-side errors (HTTP 5xx or unrecognised 4xx)."""


def raise_for_status(resp) -> None:
    """
    Inspect a :class:`requests.Response` and raise a domain-specific exception
    for any non-2xx status code.
    """
    if resp.status_code < 400:
        return
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text or resp.reason

    if resp.status_code == 401:
        raise AI01AuthError(f"Unauthorised: {detail}")
    if resp.status_code == 404:
        raise AI01NotFoundError(detail)
    if resp.status_code == 429:
        raise AI01RateLimitError("Rate limit exceeded. Please slow down your requests.")
    raise AI01ServerError(f"API error {resp.status_code}: {detail}")
