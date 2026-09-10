"""Cross-source merging — ported from notebook cell 7.

Everything is keyed on the normalized DOI, the only identifier the five sources
share. OpenAlex wins on conflicts because it is the only source that exposes author
identifiers, and those are what the Type A/B/self classification is computed from.
"""
from __future__ import annotations

from app.schemas import Article, CitingWork
from app.sources.openalex import normalize_doi

# Which fields each source contributes to an article that OpenAlex already owns.
_CONTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "scopus": ("scopus_id", "scopus_eid", "scopus_cited_by_count"),
    "wos": ("wos_uid", "wos_cited_by_count"),
    "zbmath": ("zbmath_id", "zbmath_cited_by_count"),
    "inspire": ("inspire_recid", "inspire_cited_by_count"),
}


def merge_articles(
    openalex: list[Article],
    scopus: list[Article] | None = None,
    wos: list[Article] | None = None,
    zbmath: list[Article] | None = None,
    inspire: list[Article] | None = None,
) -> list[Article]:
    """Outer-merge the per-source article lists by normalized DOI.

    The OpenAlex record is kept whole and the other sources only fill in their own
    identifiers and counts. A record from another source that matches no OpenAlex
    DOI is added on its own — that is how coverage gaps in OpenAlex surface at all.
    Records with no DOI cannot be deduplicated, so they are appended as-is.
    """
    by_doi: dict[str, Article] = {}
    without_doi: list[Article] = []

    for article in openalex:
        key = normalize_doi(article.doi)
        if key:
            by_doi[key] = article
        else:
            without_doi.append(article)

    for source, articles in (
        ("scopus", scopus or []),
        ("wos", wos or []),
        ("zbmath", zbmath or []),
        ("inspire", inspire or []),
    ):
        fields = _CONTRIBUTIONS[source]

        for article in articles:
            key = normalize_doi(article.doi)

            if not key:
                without_doi.append(article)
                continue

            existing = by_doi.get(key)
            if existing is None:
                by_doi[key] = article
                continue

            # Only this source's own columns are copied over; the OpenAlex title,
            # year and author list stay authoritative.
            by_doi[key] = existing.model_copy(update={
                field: getattr(article, field) for field in fields
            })

    merged = list(by_doi.values()) + without_doi

    return sorted(merged, key=lambda a: a.year or 0, reverse=True)


def merge_citing_works(
    openalex: list[CitingWork],
    *others: list[CitingWork],
) -> list[CitingWork]:
    """Merge citing-work lists, OpenAlex taking precedence.

    Only OpenAlex records carry author_ids, so an OpenAlex record must never be
    replaced by the same work seen through another source — doing so would drop the
    evidence that makes a citation Type B or a self-citation, silently inflating
    Type A, the figure SECIHTI actually counts.

    Records from other sources without a DOI are dropped: there is no reliable way
    to tell whether they duplicate something OpenAlex already returned, and double
    counting a citation is worse than missing one.
    """
    by_doi: dict[str, CitingWork] = {}
    openalex_without_doi: list[CitingWork] = []

    for work in openalex:
        key = normalize_doi(work.doi)
        if key:
            by_doi[key] = work
        else:
            openalex_without_doi.append(work)

    for source_list in others:
        for work in source_list:
            key = normalize_doi(work.doi)
            if key and key not in by_doi:
                by_doi[key] = work

    return list(by_doi.values()) + openalex_without_doi
