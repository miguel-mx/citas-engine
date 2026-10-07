"""Domain schemas — ported verbatim from the MVP notebook (Cell 4).

These are the contract between the engine and the Symfony app. Every tool
output validates against these; no raw dicts leak into a result. Multi-source
fields (Scopus/WoS/zbMATH/INSPIRE) are retained so those sources can plug in
later without a schema migration — the OpenAlex pipeline simply leaves them None.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Author(BaseModel):
    openalex_id: str
    display_name: str
    orcid: Optional[str] = None
    works_count: int


class CitingWork(BaseModel):
    openalex_id: Optional[str] = None   # None for Scopus/zbMATH/INSPIRE-only records
    scopus_eid: Optional[str] = None    # full EID e.g. "2-s2.0-84912345678"
    doi: Optional[str] = None
    title: str
    year: Optional[int] = None
    authors: list[str] = []
    author_ids: list[str] = []          # populated from OpenAlex; empty otherwise
    countries: list[str] = []           # populated from Scopus affiliation field
    source: str = "openalex"            # openalex | scopus | zbmath | inspire | openalex+scopus


class Article(BaseModel):
    # Source IDs — at least one will be set after merge
    openalex_id: Optional[str] = None
    scopus_id: Optional[str] = None
    scopus_eid: Optional[str] = None      # full EID, needed for refeid() citing query
    wos_uid: Optional[str] = None
    zbmath_id: Optional[str] = None       # zbMATH numeric document id, needed for rf: citing query
    inspire_recid: Optional[str] = None   # INSPIRE-HEP control_number, needed for refersto: citing query
    # Bibliographic fields
    doi: Optional[str] = None
    title: str
    year: Optional[int] = None
    journal: Optional[str] = None
    authors: Optional[str] = None
    # OpenAlex's work type ("article", "preprint", "book-chapter", "review"…),
    # normalised by openalex.classify_work so anything hosted as a submitted
    # manuscript reads "preprint" whatever OpenAlex called it. None where the record
    # came from a source that does not report one (Scopus, WoS, zbMATH, INSPIRE): the
    # absence means "not classified", never "not a preprint".
    work_type: Optional[str] = None
    # Which repository hosts it, for work_type == "preprint" only — arXiv, bioRxiv,
    # HAL, SSRN. It is what makes "18 preprints" answerable as "17 arXiv, 1 bioRxiv".
    repository: Optional[str] = None
    # Citation counts per source
    openalex_cited_by_count: int = 0
    scopus_cited_by_count: Optional[int] = None
    wos_cited_by_count: Optional[int] = None
    zbmath_cited_by_count: Optional[int] = None
    inspire_cited_by_count: Optional[int] = None
    # Classification
    citing_works: list[CitingWork] = []
    cites_type_a: int = 0
    cites_type_b: int = 0
    cites_self: int = 0
    # OpenAlex author ids of *this* work, minus the researcher. Type B is decided
    # against these alone. Empty for records from other sources.
    coauthor_ids: list[str] = []


class AnalysisResult(BaseModel):
    author: Author
    run_timestamp: str
    articles: list[Article] = []

    # Things a person should act on: a rejected key, a shortfall against OpenAlex's
    # own count, an author code that had to be guessed. A run with any of these is
    # the one worth reviewing.
    flags: list[str] = []

    # Context with nothing to act on: which source contributed what, why an
    # unconfigured source was skipped. Recorded so a partial run is never read as a
    # complete one, but never a reason to flag the run.
    notes: list[str] = []
