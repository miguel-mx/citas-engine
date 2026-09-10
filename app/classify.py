"""Deterministic citation classification (Cell 5 / Cell 6, notebook).

Type A/B/self is computed purely from author-id overlap — never an LLM
judgment. This mirrors the SECIHTI/Rizoma rules used by CCM/UNAM.
"""
from __future__ import annotations

from app.schemas import CitingWork


def classify_citation_type(
    cw: CitingWork,
    author_id: str,
    all_coauthor_ids: set[str],
) -> str:
    """Returns 'self', 'B' (co-author), or 'A' (external). Precedence: self > B > A.

    - self : the researcher's own OpenAlex id appears among the citing authors.
    - B    : a known co-author of the researcher appears, but not the researcher.
    - A    : external — no shared authorship (the citations that "count").

    Citing works with no author_ids (Scopus/zbMATH/INSPIRE-only records) fall
    through to Type A, since overlap cannot be established.
    """
    ids = set(cw.author_ids)
    if author_id in ids:
        return "self"
    if ids & all_coauthor_ids:
        return "B"
    return "A"
