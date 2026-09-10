"""zbMATH Open client — ported from notebook cell 5c.

The pure-mathematics database, and the most relevant extra source for CCM. No API
key required, and unlike WoS Starter it *does* expose citing-work lists, via the
`rf:<document_id>` reference-search operator.

zbMATH author codes are not comparable with OpenAlex author IDs, so citing records
found only here are classified Type A (external) — the same fallback used for
Scopus and INSPIRE.
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
    retry_on_transport,
)

ZBMATH_BASE = "https://api.zbmath.org/v1"
PAGE_SIZE = 50


class ZbmathClient(SourceClient):
    name = "zbMATH"

    @retry(
        retry=retry_on_transport,
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict) -> dict:
        with httpx.Client(base_url=ZBMATH_BASE, timeout=30) as client:
            resp = client.get(path, params=params)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise QuotaExceededError(
                        f"zbMATH pidió esperar {retry_after}s (> {MAX_RETRY_AFTER_SECONDS}s)."
                    )
                time.sleep(retry_after)
            resp.raise_for_status()
            return resp.json()

    def resolve_author_code(self, display_name: str) -> str | None:
        """Author code by name, but only when the match is unambiguous.

        zbMATH author names collide (there is more than one 'Hrusak'), so anything
        other than exactly one candidate returns None. Guessing here would silently
        attribute a stranger's papers.
        """
        if not self.enabled:
            return None

        try:
            data = self._get("/author/_search", {
                "search_string": display_name,
                "results_per_page": 5,
                "page": 0,
            })
        except (QuotaExceededError, httpx.HTTPError):
            return None

        results = data.get("result") or []

        return results[0]["code"] if len(results) == 1 else None

    def fetch_articles_by_code(self, author_code: str) -> list[Article]:
        """Every zbMATH document authored by this author code."""
        articles: list[Article] = []
        page = 0

        while self.enabled:
            try:
                data = self._get("/document/_search", {
                    "search_string": f"ai:{author_code}",
                    "results_per_page": PAGE_SIZE,
                    "page": page,
                })
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:  # no documents for this code
                    break
                self._handle_failure(e, "búsqueda de artículos")
                break
            except QuotaExceededError as e:
                self._handle_failure(e, "búsqueda de artículos")
                break

            docs = data.get("result") or []
            total = (data.get("status") or {}).get("nr_total_results") or 0

            for doc in docs:
                articles.append(Article(
                    zbmath_id=str(doc["id"]) if doc.get("id") is not None else None,
                    doi=_doi(doc),
                    title=_title(doc),
                    year=_year(doc),
                    journal=_journal(doc),
                    authors="; ".join(_author_names(doc)) or None,
                ))

            page += 1
            if page * PAGE_SIZE >= total or not docs:
                break

        return articles

    def fetch_citing_works(self, document_id: str) -> list[CitingWork]:
        """Every zbMATH document whose reference list includes this document."""
        citing: list[CitingWork] = []
        page = 0

        while self.enabled:
            try:
                data = self._get("/document/_search", {
                    "search_string": f"rf:{document_id}",
                    "results_per_page": PAGE_SIZE,
                    "page": page,
                })
            except httpx.HTTPStatusError as e:
                # zbMATH answers 404 for a search that matches nothing, rather than
                # returning an empty result set. That is an ordinary "no citations
                # here", not a failure, and must not disable the source.
                if e.response.status_code == 404:
                    break
                self._handle_failure(e, "búsqueda de citas")
                break
            except QuotaExceededError as e:
                self._handle_failure(e, "búsqueda de citas")
                break

            docs = data.get("result") or []
            total = (data.get("status") or {}).get("nr_total_results") or 0

            for doc in docs:
                citing.append(CitingWork(
                    doi=_doi(doc),
                    title=_title(doc),
                    year=_year(doc),
                    authors=_author_names(doc),
                    author_ids=[],  # zbMATH codes do not map to OpenAlex ids
                    source="zbmath",
                ))

            page += 1
            if page * PAGE_SIZE >= total or not docs:
                break

        return citing


def _doi(doc: dict) -> str | None:
    for link in doc.get("links", []) or []:
        if link.get("type") == "doi":
            return link.get("identifier")
    return None


def _title(doc: dict) -> str:
    return (doc.get("title") or {}).get("title") or "(sin título)"


def _journal(doc: dict) -> str | None:
    source = doc.get("source") or {}
    series = source.get("series") or []
    if series and series[0].get("title"):
        return series[0]["title"]
    return source.get("source")


def _year(doc: dict) -> int | None:
    try:
        return int(doc.get("year"))
    except (TypeError, ValueError):
        return None


def _author_names(doc: dict) -> list[str]:
    authors = (doc.get("contributors") or {}).get("authors") or []
    return [a.get("name", "") for a in authors if a.get("name")]
