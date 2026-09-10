"""Scopus (Elsevier) client — ported from notebook cell 5b.

Provides article lists, per-DOI citation counts, and — unlike Web of Science
Starter — the actual list of citing works, via the refeid() query operator.

Scopus search results do not expose author identifiers that can be compared with
OpenAlex author IDs, so citing records found only here are classified Type A
(external). That is stated in classify.py and holds for zbMATH and INSPIRE too.
"""
from __future__ import annotations

import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas import Article, CitingWork
from app.sources._http import (
    MAX_RETRY_AFTER_SECONDS,
    QuotaExceededError,
    SourceClient,
    bare_orcid,
    retry_on_transport,
)
from app.sources.openalex import normalize_doi

SCOPUS_BASE = "https://api.elsevier.com/content"
PAGE_SIZE = 200

_ARTICLE_FIELDS = (
    "dc:title,prism:coverDate,prism:doi,dc:identifier,eid,"
    "citedby-count,prism:publicationName,prism:volume,prism:issueIdentifier,"
    "prism:pageRange,prism:issn,author,subtypeDescription"
)

_CITING_FIELDS = (
    "dc:title,prism:coverDate,prism:doi,eid,dc:identifier,"
    "prism:publicationName,dc:creator,author,affiliation,subtypeDescription"
)


class ScopusClient(SourceClient):
    name = "Scopus"

    def __init__(self, api_key: str | None) -> None:
        super().__init__(enabled=bool(api_key))
        self._api_key = api_key or ""
        if not api_key:
            self._disabled_reason = "Scopus: sin clave de API configurada."
            # A deployment choice, not a fault: it must not make runs ask for review.
            self._disabled_severity = "note"

    @retry(
        retry=retry_on_transport,
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict) -> dict:
        with httpx.Client(base_url=SCOPUS_BASE, timeout=30) as client:
            resp = client.get(
                path,
                params=params,
                headers={"X-ELS-APIKey": self._api_key, "Accept": "application/json"},
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise QuotaExceededError(
                        f"Scopus pidió esperar {retry_after}s (> {MAX_RETRY_AFTER_SECONDS}s)."
                    )
                time.sleep(retry_after)
            resp.raise_for_status()
            return resp.json()

    # ── Articles ─────────────────────────────────────────────────────────────
    def fetch_articles_by_author_id(self, author_id: str) -> list[Article]:
        """All Scopus articles for a Scopus Author ID (AU-ID).

        Takes precedence over ORCID when both are known: an AU-ID identifies the
        Scopus profile exactly, whereas the ORCID query depends on Scopus having
        the ORCID recorded on each record.
        """
        return self._search_articles(f"AU-ID({author_id})")

    def fetch_articles_by_orcid(self, orcid: str) -> list[Article]:
        return self._search_articles(f"ORCID({bare_orcid(orcid)})")

    def _search_articles(self, query: str) -> list[Article]:
        articles: list[Article] = []
        start = 0

        while self.enabled:
            try:
                data = self._get("/search/scopus", {
                    "query": query,
                    "count": PAGE_SIZE,
                    "start": start,
                    "field": _ARTICLE_FIELDS,
                })
            except (QuotaExceededError, httpx.HTTPStatusError) as e:
                self._handle_failure(e, "búsqueda de artículos")
                break

            results = data.get("search-results", {})
            entries = results.get("entry", [])
            total = int(results.get("opensearch:totalResults", 0) or 0)

            for entry in entries:
                article = _article_from_entry(entry)
                if article is not None:
                    articles.append(article)

            start += len(entries)
            if start >= total or not entries:
                break

        return articles

    # ── Citing works ─────────────────────────────────────────────────────────
    def fetch_citing_works(self, eid: str) -> list[CitingWork]:
        """Every Scopus record citing the given EID, via refeid().

        Needs the cited-by permission on the API key; without it Scopus answers
        401 and this client disables itself with that reason.
        """
        citing: list[CitingWork] = []
        start = 0

        while self.enabled:
            try:
                data = self._get("/search/scopus", {
                    "query": f"refeid({eid})",
                    "count": 25,
                    "start": start,
                    "field": _CITING_FIELDS,
                })
            except (QuotaExceededError, httpx.HTTPStatusError) as e:
                self._handle_failure(e, "búsqueda de citas")
                break

            results = data.get("search-results", {})
            entries = results.get("entry", [])
            total = int(results.get("opensearch:totalResults", 0) or 0)

            for entry in entries:
                work = _citing_from_entry(entry)
                if work is not None:
                    citing.append(work)

            start += len(entries)
            if start >= total or not entries:
                break

            time.sleep(0.3)  # Scopus is unhappy with tight pagination loops

        return citing

    # ── Per-DOI count ────────────────────────────────────────────────────────
    def fetch_cited_by_count(self, doi: str) -> int | None:
        """Scopus citedby-count for a DOI, or None when it has no record."""
        if not self.enabled:
            return None

        try:
            data = self._get("/search/scopus", {
                "query": f'DOI("{normalize_doi(doi)}")',
                "count": 1,
                "field": "citedby-count",
            })
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            self._handle_failure(e, "conteo por DOI")
            return None
        except QuotaExceededError as e:
            self._handle_failure(e, "conteo por DOI")
            return None

        entries = data.get("search-results", {}).get("entry", [])
        if not entries or entries[0].get("error"):
            return None

        return int(entries[0].get("citedby-count", 0) or 0)


def _authors_of(entry: dict) -> list[str]:
    field = entry.get("author", entry.get("dc:creator", []))
    if isinstance(field, dict):
        field = [field]
    if isinstance(field, str):
        return [field]
    return [a.get("authname", "") for a in field if isinstance(a, dict) and a.get("authname")]


def _year_of(entry: dict) -> int | None:
    date = entry.get("prism:coverDate", "")
    return int(date[:4]) if date and date[:4].isdigit() else None


def _article_from_entry(entry: dict) -> Article | None:
    if entry.get("error") or not entry.get("dc:identifier"):
        return None

    count = entry.get("citedby-count")

    return Article(
        scopus_id=entry.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
        scopus_eid=entry.get("eid") or None,
        doi=entry.get("prism:doi") or None,
        title=entry.get("dc:title") or "(sin título)",
        year=_year_of(entry),
        journal=entry.get("prism:publicationName"),
        authors="; ".join(_authors_of(entry)) or None,
        scopus_cited_by_count=int(count) if count is not None else None,
    )


def _citing_from_entry(entry: dict) -> CitingWork | None:
    if entry.get("error") or not entry.get("eid"):
        return None

    affiliations = entry.get("affiliation", [])
    if isinstance(affiliations, dict):
        affiliations = [affiliations]
    countries = sorted({
        a.get("affiliation-country", "")
        for a in affiliations
        if isinstance(a, dict) and a.get("affiliation-country")
    })

    return CitingWork(
        scopus_eid=entry.get("eid"),
        doi=entry.get("prism:doi") or None,
        title=entry.get("dc:title") or "(sin título)",
        year=_year_of(entry),
        authors=_authors_of(entry),
        author_ids=[],  # Scopus search exposes none that map to OpenAlex ids
        countries=countries,
        source="scopus",
    )
