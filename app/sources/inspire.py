"""INSPIRE-HEP client — ported from notebook cell 5c.

High-energy-physics literature. No API key, but rate-limited to 15 requests per
5 seconds per IP, so this client paces itself.

MISATTRIBUTION WARNING (verified in the notebook, kept here deliberately):
INSPIRE's /authors endpoint is a loose full-text search, not a name match. A query
for "Michael Hrusak" returns "Holzbock, Michael" as the top hit — matched on the
first name alone. Accepting that would merge an unrelated physicist's papers into
a CCM mathematician's citation report. So name resolution only accepts a hit whose
*surname* matches, and callers should treat any name-resolved recid as unverified.
Most CCM mathematicians have no INSPIRE profile at all; leaving the recid unset is
the correct outcome for them, not a failure.
"""
from __future__ import annotations

import time
import unicodedata

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas import Article, CitingWork
from app.sources._http import (
    MAX_RETRY_AFTER_SECONDS,
    QuotaExceededError,
    SourceClient,
    retry_on_transport,
)

INSPIRE_BASE = "https://inspirehep.net/api"
PAGE_SIZE = 50

# 15 requests / 5s per IP; this keeps us comfortably under it.
_PACING_SECONDS = 0.35


class InspireClient(SourceClient):
    name = "INSPIRE-HEP"

    @retry(
        retry=retry_on_transport,
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict | None = None) -> dict:
        with httpx.Client(base_url=INSPIRE_BASE, timeout=30, follow_redirects=True) as client:
            resp = client.get(path, params=params or {}, headers={"Accept": "application/json"})
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise QuotaExceededError(
                        f"INSPIRE pidió esperar {retry_after}s (> {MAX_RETRY_AFTER_SECONDS}s)."
                    )
                time.sleep(retry_after)
            resp.raise_for_status()
            return resp.json()

    def resolve_recid_by_orcid(self, orcid: str) -> str | None:
        """Author recid from an ORCID. 404 simply means no INSPIRE profile, which
        is the normal case for mathematicians."""
        if not self.enabled:
            return None

        try:
            data = self._get(f"/orcid/{orcid}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            return None
        except (QuotaExceededError, httpx.HTTPError):
            return None

        control_number = (data.get("metadata") or {}).get("control_number")

        return str(control_number) if control_number is not None else None

    def resolve_recid_by_name(self, display_name: str) -> str | None:
        """Best-effort name lookup, rejected unless the surname matches.

        See the module docstring: without the surname check this returns confident
        nonsense for anyone whose first name is common in HEP.
        """
        if not self.enabled:
            return None

        try:
            data = self._get("/authors", {
                "q": display_name,
                "size": 1,
                "fields": "control_number,name",
            })
        except (QuotaExceededError, httpx.HTTPError):
            return None

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return None

        metadata = hits[0].get("metadata", {})
        control_number = metadata.get("control_number")
        if control_number is None:
            return None

        hit_name = (metadata.get("name") or {}).get("value", "")
        hit_surname = hit_name.split(",")[0] if "," in hit_name else hit_name
        query_tokens = display_name.strip().split()
        query_surname = query_tokens[-1] if query_tokens else ""

        if _fold(hit_surname) != _fold(query_surname):
            return None

        return str(control_number)

    def fetch_articles_by_recid(self, recid: str) -> list[Article]:
        """Every INSPIRE literature record authored by this author recid."""
        articles: list[Article] = []
        page = 1
        query = f'authors.record.$ref:"{INSPIRE_BASE}/authors/{recid}"'

        while self.enabled:
            try:
                data = self._get("/literature", {
                    "q": query,
                    "size": PAGE_SIZE,
                    "page": page,
                    "fields": "titles,control_number,dois,publication_info,citation_count,authors.full_name",
                })
            except (QuotaExceededError, httpx.HTTPStatusError) as e:
                self._handle_failure(e, "búsqueda de artículos")
                break

            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", 0)

            for hit in hits:
                metadata = hit.get("metadata", {})
                articles.append(Article(
                    inspire_recid=_recid(metadata),
                    doi=_doi(metadata),
                    title=_title(metadata),
                    year=_publication(metadata).get("year"),
                    journal=_publication(metadata).get("journal_title"),
                    authors="; ".join(_author_names(metadata)) or None,
                    inspire_cited_by_count=metadata.get("citation_count"),
                ))

            page += 1
            if len(articles) >= total or not hits:
                break

            time.sleep(_PACING_SECONDS)

        return articles

    def fetch_citing_works(self, recid: str) -> list[CitingWork]:
        """Every INSPIRE record citing the paper with this recid."""
        citing: list[CitingWork] = []
        page = 1

        while self.enabled:
            try:
                data = self._get("/literature", {
                    "q": f"refersto:recid:{recid}",
                    "size": PAGE_SIZE,
                    "page": page,
                    "fields": "titles,control_number,dois,publication_info,authors.full_name",
                })
            except (QuotaExceededError, httpx.HTTPStatusError) as e:
                self._handle_failure(e, "búsqueda de citas")
                break

            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", 0)

            for hit in hits:
                metadata = hit.get("metadata", {})
                citing.append(CitingWork(
                    doi=_doi(metadata),
                    title=_title(metadata),
                    year=_publication(metadata).get("year"),
                    authors=_author_names(metadata),
                    author_ids=[],  # INSPIRE recids do not map to OpenAlex ids
                    source="inspire",
                ))

            page += 1
            if len(citing) >= total or not hits:
                break

            time.sleep(_PACING_SECONDS)

        return citing


def _fold(value: str) -> str:
    """Accent- and case-insensitive comparison key (Hrušák == Hrusak)."""
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().strip().lower()


def _recid(metadata: dict) -> str | None:
    value = metadata.get("control_number")
    return str(value) if value is not None else None


def _doi(metadata: dict) -> str | None:
    dois = metadata.get("dois") or []
    return dois[0]["value"] if dois else None


def _title(metadata: dict) -> str:
    titles = metadata.get("titles") or []
    return titles[0]["title"] if titles else "(sin título)"


def _publication(metadata: dict) -> dict:
    return (metadata.get("publication_info") or [{}])[0]


def _author_names(metadata: dict) -> list[str]:
    return [a.get("full_name", "") for a in metadata.get("authors", []) if a.get("full_name")]
