"""Web of Science Starter (Clarivate) client — ported from notebook cell 5b.

The Starter tier cannot list which documents cite a record, so WoS never
contributes citing works or Type A/B classification — only counts and metadata.

IMPORTANT (verified against the CCM key on 2026-07-30): on this subscription the
`citations` array comes back empty on every record, from both /documents and
/documents/{uid}, with or without detail=full. So WoS yields **no citation counts
at all** here; what it does contribute is an independent record of which papers
exist (UID, DOI, year, journal), useful for spotting coverage gaps in OpenAlex.
`wos_cited_by_count` is therefore left None rather than 0 — "unknown" and "zero"
are different claims, and a bibliometric report must not confuse them.

It also has a strict daily quota, which is why the client disables itself on the
first quota response rather than retrying per article.
"""
from __future__ import annotations

import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas import Article
from app.sources._http import (
    MAX_RETRY_AFTER_SECONDS,
    QuotaExceededError,
    SourceClient,
    bare_orcid,
    retry_on_transport,
)
from app.sources.openalex import normalize_doi

WOS_BASE = "https://api.clarivate.com/apis/wos-starter/v1"
PAGE_SIZE = 50


class WosClient(SourceClient):
    name = "Web of Science"

    def __init__(self, api_key: str | None) -> None:
        super().__init__(enabled=bool(api_key))
        self._api_key = api_key or ""
        if not api_key:
            self._disabled_reason = "Web of Science: sin clave de API configurada."
            # A deployment choice, not a fault: it must not make runs ask for review.
            self._disabled_severity = "note"

    @retry(
        retry=retry_on_transport,
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict) -> dict:
        with httpx.Client(base_url=WOS_BASE, timeout=30) as client:
            resp = client.get(
                path,
                params=params,
                headers={"X-ApiKey": self._api_key, "Accept": "application/json"},
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise QuotaExceededError(
                        f"WoS pidió esperar {retry_after}s (> {MAX_RETRY_AFTER_SECONDS}s)."
                    )
                time.sleep(retry_after)
            resp.raise_for_status()
            return resp.json()

    def fetch_articles_by_orcid(self, orcid: str) -> list[Article]:
        """All WoS records for an ORCID (the AI= field)."""
        articles: list[Article] = []
        page = 1
        identifier = bare_orcid(orcid)

        while self.enabled:
            try:
                data = self._get("/documents", {
                    "q": f"AI={identifier}",
                    "db": "WOS",
                    "limit": PAGE_SIZE,
                    "page": page,
                })
            except (QuotaExceededError, httpx.HTTPStatusError) as e:
                self._handle_failure(e, "búsqueda de artículos")
                break

            hits = data.get("hits", [])
            total = data.get("metadata", {}).get("total", 0)

            for record in hits:
                articles.append(_article_from_record(record))

            page += 1
            if len(articles) >= total or not hits:
                break

        return articles

    def fetch_cited_by_count(self, doi: str) -> int | None:
        """WoS times-cited for a DOI, or None when WoS has no record for it."""
        if not self.enabled:
            return None

        try:
            data = self._get("/documents", {
                "q": f'DO="{normalize_doi(doi)}"',
                "db": "WOS",
                "limit": 1,
                "page": 1,
            })
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                return None
            self._handle_failure(e, "conteo por DOI")
            return None
        except QuotaExceededError as e:
            self._handle_failure(e, "conteo por DOI")
            return None

        hits = data.get("hits", [])
        if not hits:
            return None

        # No `or 0` here: an absent count means "WoS did not tell us", which must
        # not be reported as a genuine zero.
        return _times_cited(hits[0])


def _times_cited(record: dict) -> int | None:
    """WoS Starter has spelled this field three ways across versions, and on some
    records the count only appears in the per-database `citations` list."""
    for key in ("times_cited", "timesCited", "citationCount"):
        value = record.get(key)
        if value is not None:
            return int(value)

    for citation in record.get("citations", []):
        if citation.get("db") == "WOS":
            return int(citation.get("count", 0))

    return None


def _article_from_record(record: dict) -> Article:
    names = record.get("names", {})
    authors = "; ".join(
        a.get("displayName", a.get("wosStandard", ""))
        for a in (names.get("authors") or [])
    )

    # publishYear is nested under `source`, not top-level. Reading it from the top
    # level (as the notebook does) silently yields None for every WoS record.
    source = record.get("source", {}) or {}

    return Article(
        wos_uid=record.get("uid") or None,
        doi=(record.get("identifiers", {}) or {}).get("doi") or None,
        title=record.get("title") or "(sin título)",
        year=source.get("publishYear"),
        journal=source.get("sourceTitle"),
        authors=authors or None,
        wos_cited_by_count=_times_cited(record),
    )
