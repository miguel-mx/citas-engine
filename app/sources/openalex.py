"""OpenAlex client + the deterministic tools (ported from Cell 5).

The LLM is never in this path. Every Article / CitingWork produced here is
built directly from an OpenAlex API response, so every count and reference is
traceable to a tool output.
"""
from __future__ import annotations

import re
import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas import Article, Author, CitingWork

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
OA_AUTHOR_RE = re.compile(r"^(?:https://openalex\.org/)?(A\d+)$", re.IGNORECASE)


class QuotaExceededError(RuntimeError):
    """Raised instead of sleeping when Retry-After exceeds the configured cap."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get(path: str, params: dict | None = None) -> dict:
    """Single HTTP GET with retry/backoff and mandatory mailto."""
    p = {"mailto": settings.mailto, **(params or {})}
    with httpx.Client(base_url=settings.openalex_base, timeout=30) as client:
        resp = client.get(path, params=p)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            if retry_after > settings.max_retry_after_seconds:
                raise QuotaExceededError(
                    f"OpenAlex asked to wait {retry_after}s "
                    f"(> {settings.max_retry_after_seconds}s cap) — treating as quota exhaustion."
                )
            time.sleep(retry_after)
        resp.raise_for_status()
        return resp.json()


def _paginate(path: str, extra_params: dict | None = None) -> list[dict]:
    """Cursor-paginate through all results from an OpenAlex endpoint."""
    results: list[dict] = []
    cursor = "*"
    params = {"per-page": settings.page_size, **(extra_params or {})}
    while cursor:
        data = _get(path, {**params, "cursor": cursor})
        results.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
    return results


def normalize_doi(doi: str | None) -> str | None:
    """Strips common DOI URL prefixes and lowercases for deduplication."""
    if not doi:
        return None
    doi = doi.lower().strip()
    for prefix in (
        "https://doi.org/", "http://doi.org/",
        "https://dx.doi.org/", "http://dx.doi.org/",
        "doi:",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def _author_from_record(data: dict) -> Author:
    return Author(
        openalex_id=data["id"],
        display_name=data["display_name"],
        orcid=data.get("orcid"),
        works_count=data.get("works_count", 0),
    )


# ── Tool 1: resolve_author ────────────────────────────────────────────────────
# Accepted query forms:
#   • OpenAlex author ID  — "A5023888391" or "https://openalex.org/A5023888391"
#   • ORCID               — "0000-0002-1692-2216"
#   • Name                — "Michael Hrušák"  (returns a list for disambiguation)
def resolve_author(query: str) -> Author | list[Author]:
    query = query.strip()

    oa_match = OA_AUTHOR_RE.fullmatch(query)
    if oa_match:
        data = _get(f"/authors/{oa_match.group(1).upper()}")
        return _author_from_record(data)

    if re.match(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", query):
        if not ORCID_RE.fullmatch(query):
            raise ValueError(
                f"Malformed ORCID '{query}'. Expected format: XXXX-XXXX-XXXX-XXXX"
            )
        data = _get(f"/authors/orcid:{query}")
        return _author_from_record(data)

    data = _get("/authors", {"search": query, "per-page": 10})
    return [_author_from_record(a) for a in data.get("results", [])]


# ── Preprint detection ────────────────────────────────────────────────────────
# arXiv mints its own DataCite DOI for every submission; the published paper keeps
# the publisher's. A DOI under this prefix is therefore the arXiv record, never the
# version of record.
ARXIV_DOI_PREFIX = "10.48550/arxiv."


def classify_work(w: dict) -> tuple[str | None, str | None]:
    """(work_type, repository) for one OpenAlex work record.

    Two signals, both of which mean a preprint on their own:

      • OpenAlex typed it "preprint" — its own classification, and the reliable one.
      • The DOI sits under arXiv's DataCite prefix. An arXiv DOI *is* the arXiv
        record; the published paper keeps the publisher's, so this settles the
        question even when OpenAlex typed the work "article".

    `repository` is filled for preprints only, and names the hosting source (arXiv,
    bioRxiv…) so SIAB can report which servers the hidden works came from.

    Two tempting signals are deliberately NOT used, because checking this against
    a real author's record showed both misclassify published papers:

      • `primary_location.source.type == "repository"`. OpenAlex picks the best open
        location as primary, so a paper whose journal is not indexed — or that has a
        digital-library copy — has a repository there.
      • `primary_location.version == "submittedVersion"`. Green open access: the
        author's accepted manuscript in an institutional repository is the best open
        copy of a *published* paper. Hrušák's "Parametrized principles" is exactly
        this — type "article", version "submittedVersion", hosted by a university
        repository, and carrying DOI 10.1090/s0002-9947-03-03446-9, a published
        Transactions of the AMS paper. Either rule would have hidden it.

    The cost is under-detection: a preprint OpenAlex typed "article" and that has no
    arXiv DOI is not caught. That is the right way to be wrong here. Leaving a
    preprint in the list is the status quo and visible on the row; dropping a
    published paper out of a figure that feeds an evaluation is not.
    """
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}

    oa_type = (w.get("type") or "").strip().lower() or None
    doi = normalize_doi(w.get("doi")) or ""

    if oa_type == "preprint" or doi.startswith(ARXIV_DOI_PREFIX):
        return "preprint", _repository_name(src.get("display_name"))

    return oa_type, None


def _repository_name(display_name: str | None) -> str | None:
    """The repository as it should read on a badge.

    OpenAlex spells arXiv "arXiv (Cornell University)" and Trieste's repository
    "OpenstarTs (Univeristy of Trieste https://www.units.it/)" — the host institution
    in parentheses, sometimes with a URL and a typo. None of that helps someone
    scanning a list for which works are preprints, so only the name is kept.
    """
    if not display_name:
        return None

    return display_name.split(" (")[0].strip() or None


# ── Tool 2: fetch_author_works ────────────────────────────────────────────────
def fetch_author_works(author_id: str, max_works: int | None = None) -> list[Article]:
    """Retrieves all works for author from OpenAlex; respects the MAX_WORKS cap."""
    cap = settings.max_works if max_works is None else max_works
    raw = _paginate("/works", {"filter": f"author.id:{author_id}"})
    truncated = len(raw) > cap
    if truncated:
        raw = raw[:cap]

    articles: list[Article] = []
    for w in raw:
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        work_type, repository = classify_work(w)
        authorships = w.get("authorships", [])
        authors_str = "; ".join(
            a["author"]["display_name"]
            for a in authorships
            if a.get("author") and a["author"].get("display_name")
        )
        coauthor_ids = [
            a["author"]["id"]
            for a in authorships
            if a.get("author") and a["author"].get("id") and a["author"]["id"] != author_id
        ]
        articles.append(
            Article(
                openalex_id=w["id"],
                doi=w.get("doi"),
                title=w.get("title") or "(no title)",
                year=w.get("publication_year"),
                journal=src.get("display_name"),
                authors=authors_str,
                work_type=work_type,
                repository=repository,
                openalex_cited_by_count=w.get("cited_by_count", 0),
                coauthor_ids=coauthor_ids,
            )
        )
    return articles


# ── Tool 3: fetch_citing_works ────────────────────────────────────────────────
def fetch_citing_works(work_id: str, max_citing: int | None = None) -> list[CitingWork]:
    """Retrieves all works that cite work_id via OpenAlex; respects the cap."""
    cap = settings.max_citing_per_work if max_citing is None else max_citing
    raw = _paginate("/works", {"filter": f"cites:{work_id}"})
    if len(raw) > cap:
        raw = raw[:cap]

    citing: list[CitingWork] = []
    for w in raw:
        authors_names, author_ids = [], []
        for a in w.get("authorships", []):
            auth = a.get("author") or {}
            if auth.get("id"):
                author_ids.append(auth["id"])
            if auth.get("display_name"):
                authors_names.append(auth["display_name"])
        citing.append(
            CitingWork(
                openalex_id=w["id"],
                doi=w.get("doi"),
                title=w.get("title") or "(no title)",
                year=w.get("publication_year"),
                authors=authors_names,
                author_ids=author_ids,
                source="openalex",
            )
        )
    return citing
