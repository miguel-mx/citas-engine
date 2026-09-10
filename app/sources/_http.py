"""Shared HTTP plumbing for the external bibliographic sources.

The notebook kept each source's "is it still usable" flags as module-level globals
(`_scopus_auth_ok`, `_wos_quota_ok`, …). That is fine for a notebook, which runs once
and dies, but wrong for a long-running service: a bad key would trip the flag and the
source would stay skipped for every later request, so an administrator fixing the key
in /admin/configuracion would see no effect until someone restarted uvicorn.

So the state lives on a client *instance* instead, and the pipeline builds one set of
clients per analysis run.
"""
from __future__ import annotations

import threading

import httpx
from tenacity import retry_if_exception

# A 429 asking us to wait longer than this is a daily quota, not a rate limit —
# sleeping on it would block the run on a response that will not come today.
MAX_RETRY_AFTER_SECONDS = 30


class QuotaExceededError(RuntimeError):
    """Raised instead of sleeping when Retry-After exceeds MAX_RETRY_AFTER_SECONDS."""


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


retry_on_transport = retry_if_exception(is_retryable)


def bare_orcid(orcid: str | None) -> str | None:
    """Strips the orcid.org prefix OpenAlex includes on ORCIDs."""
    if not orcid:
        return None
    for prefix in ("https://orcid.org/", "http://orcid.org/"):
        if orcid.startswith(prefix):
            return orcid[len(prefix):]
    return orcid


class SourceClient:
    """Per-run state for one external source.

    A client disables itself on an unrecoverable failure (bad key, exhausted quota)
    and records why. Callers keep calling; the methods return empty results once
    disabled, and the pipeline reports `disabled_reason` on the run so a partial
    analysis is never mistaken for a complete one.

    `disabled_severity` says which kind of report that is. "Not configured" is a
    deliberate choice by whoever set the deployment up and is merely worth stating;
    a rejected key or an exhausted quota is something a person has to fix, and only
    those should make a run ask to be reviewed.
    """

    name: str = "source"

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._disabled_reason: str | None = None
        self._disabled_severity: str = "warning"
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    @property
    def disabled_severity(self) -> str:
        """"warning" (someone must act) or "note" (expected, just worth saying)."""
        return self._disabled_severity

    def disable(self, reason: str, severity: str = "warning") -> None:
        """Stop using this source for the rest of the run. First reason wins — it is
        the one that actually caused the failure; later calls just observe it."""
        with self._lock:
            if self._enabled:
                self._enabled = False
                self._disabled_reason = reason
                self._disabled_severity = severity

    def _handle_failure(self, exc: BaseException, context: str) -> None:
        """Classifies a failure: quota and auth errors disable the source, anything
        else is left to the caller to re-raise."""
        if isinstance(exc, QuotaExceededError):
            self.disable(f"{self.name}: cuota agotada ({context}).")
            return

        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403):
            self.disable(
                f"{self.name}: la clave de API fue rechazada "
                f"({exc.response.status_code} en {context}). Revísala en la configuración."
            )
            return

        raise exc
