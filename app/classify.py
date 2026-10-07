"""Deterministic citation classification (Cell 5 / Cell 6, notebook).

Type A/B/self is computed purely from author-id overlap — never an LLM
judgment. This mirrors the SECIHTI/Rizoma rules used by CCM/UNAM, which compare
each citing work against the authors *of the cited work*, not against everyone
the researcher has ever published with.
"""
from __future__ import annotations

from app.schemas import CitingWork


def classify_citation_type(
    cw: CitingWork,
    author_id: str,
    cited_coauthor_ids: set[str],
) -> str:
    """Returns 'self', 'B' (co-author), or 'A' (external). Precedence: self > B > A.

    `cited_coauthor_ids` are the OpenAlex author ids of the cited article itself,
    minus the researcher (Article.coauthor_ids of that one article).

    - self : the researcher's own OpenAlex id appears among the citing authors.
    - B    : an author of the cited article appears, but not the researcher.
    - A    : none of the citing authors is an author of the cited article (the
             citations that "count"). Someone who co-wrote a *different* paper
             with the researcher still yields Type A here.

    Citing works with no author_ids (Scopus/zbMATH/INSPIRE-only records), and
    cited articles with no author ids, fall through to Type A, since overlap
    cannot be established. Type A is therefore an upper bound.
    """
    ids = set(cw.author_ids)
    if author_id in ids:
        return "self"
    if ids & cited_coauthor_ids:
        return "B"
    return "A"
