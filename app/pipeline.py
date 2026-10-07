"""Analysis pipeline (ported from Cell 6 of the notebook).

Author resolution and disambiguation are handled by the /resolve endpoint (the
Symfony UI drives the human choice), so this module starts from an already-resolved
Author and runs the heavy analysis as a straight sequence:

    fetch_works → fetch_citing_works → classify_articles → validate_counts → assemble
                                                                                │
                                                          want_report? ──► generate_report

Nothing here is agentic: every step is deterministic Python, the order is fixed,
and the one branch (report or not) is a plain `if`. The stages are ordinary
functions over real values rather than nodes threading a shared state dict, so
each one is callable and testable on its own.

OpenAlex is always used; Scopus, WoS, zbMATH and INSPIRE join in when the caller
supplies the key or identifier each one needs (see SourceBundle).
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.classify import classify_citation_type
from app.config import settings
from app.merge import merge_articles, merge_citing_works
from app import progress
from app.report import generate_report
from app.schemas import AnalysisResult, Article, Author, CitingWork
from app.sources import openalex
from app.sources._http import SourceClient
from app.sources.inspire import InspireClient
from app.sources.scopus import ScopusClient
from app.sources.wos import WosClient
from app.sources.zbmath import ZbmathClient


@dataclass
class SourceBundle:
    """The extra sources for one run, plus the identifiers they are queried with.

    Built fresh per analysis (never module-level): each client carries its own
    "still usable" state, so a key fixed in SIAB's admin screen takes effect on the
    very next run instead of after a restart.
    """

    scopus: ScopusClient
    wos: WosClient
    zbmath: ZbmathClient
    inspire: InspireClient

    scopus_author_id: str | None = None
    zbmath_author_code: str | None = None
    inspire_author_recid: str | None = None

    # Two buckets, because they mean different things to whoever reads the run:
    # a warning is something to act on, a note is context worth recording.
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        scopus_api_key: str | None = None,
        wos_api_key: str | None = None,
        scopus_author_id: str | None = None,
        zbmath_author_code: str | None = None,
        inspire_author_recid: str | None = None,
        use_zbmath: bool = True,
        use_inspire: bool = True,
    ) -> SourceBundle:
        return cls(
            scopus=ScopusClient(scopus_api_key),
            wos=WosClient(wos_api_key),
            zbmath=ZbmathClient(enabled=use_zbmath),
            inspire=InspireClient(enabled=use_inspire),
            scopus_author_id=scopus_author_id,
            zbmath_author_code=zbmath_author_code,
            inspire_author_recid=inspire_author_recid,
        )

    def guarded(self, client: SourceClient, what: str, call, default):
        """Run one extra-source call, absorbing anything it throws.

        OpenAlex is the backbone of the analysis; Scopus, WoS, zbMATH and INSPIRE
        are enrichment. A surprise from any of them — an undocumented status code,
        a shape change — must degrade that one source and be reported, never abort
        an analysis that is otherwise complete.
        """
        try:
            return call()
        except Exception as e:  # noqa: BLE001 — deliberately broad, see docstring
            client.disable(f"{client.name}: error inesperado en {what} ({type(e).__name__}).")
            return default

    def _disabled(self, severity: str) -> list[str]:
        return [
            c.disabled_reason
            for c in (self.scopus, self.wos, self.zbmath, self.inspire)
            if c.disabled_reason and c.disabled_severity == severity
        ]

    def source_warnings(self) -> list[str]:
        """Sources that failed in a way somebody has to fix — a rejected key, an
        exhausted quota, an author code we had to guess."""
        return self._disabled("warning") + self.warnings

    def source_notes(self) -> list[str]:
        """Why a source contributed nothing when that was expected: not configured,
        no identifier to query with. Still recorded, so 'no citations found' is
        never confused with 'we never asked' — but nothing to act on."""
        return self._disabled("note") + self.notes


def _preprint_note(articles: list[Article]) -> list[str]:
    """How many of the works are preprints, and where they live.

    A note rather than a flag: preprints are a normal part of a publication record,
    and nothing here has to be fixed. It exists so the number is on the run itself
    even when nobody touches the "excluir preprints" toggle in SIAB — and so the
    duplication it implies is stated once, in writing, next to the figures.
    """
    preprints = [a for a in articles if a.work_type == "preprint"]

    if not preprints:
        return []

    by_repo = Counter(a.repository or "repositorio sin identificar" for a in preprints)
    breakdown = ", ".join(f"{name} {count}" for name, count in by_repo.most_common())

    return [
        f"{len(preprints)} de {len(articles)} obras son preprints ({breakdown}). "
        "OpenAlex registra el preprint y la versión publicada como obras distintas, "
        "con DOI distinto, así que no se fusionan: si un trabajo aparece dos veces, "
        "ésta es la razón. En el análisis pueden ocultarse."
    ]


# ── Stage 1: fetch the author's works from every available source ─────────────
def fetch_works(
    author: Author,
    max_works: int | None = None,
    sources: SourceBundle | None = None,
) -> tuple[list[Article], list[str], list[str]]:
    """Retrieve and merge the author's works. Returns (articles, warnings, notes)."""
    flags: list[str] = []
    notes: list[str] = []

    openalex_articles = openalex.fetch_author_works(author.openalex_id, max_works=max_works)
    cap = max_works or settings.max_works
    if len(openalex_articles) >= cap:
        flags.append(f"La lista de obras puede estar truncada en MAX_WORKS={cap}.")

    if sources is None:
        return openalex_articles, flags, notes + _preprint_note(openalex_articles)

    orcid = author.orcid

    # Each source is an independent network round trip, so they run concurrently.
    with ThreadPoolExecutor(max_workers=4) as pool:
        scopus_future = pool.submit(_scopus_articles, sources, orcid)
        wos_future = pool.submit(_wos_articles, sources, orcid)
        zbmath_future = pool.submit(_zbmath_articles, sources, author)
        inspire_future = pool.submit(_inspire_articles, sources, author)

        scopus_articles = scopus_future.result()
        wos_articles = wos_future.result()
        zbmath_articles = zbmath_future.result()
        inspire_articles = inspire_future.result()

    merged = merge_articles(
        openalex_articles,
        scopus=scopus_articles,
        wos=wos_articles,
        zbmath=zbmath_articles,
        inspire=inspire_articles,
    )

    # Provenance, not a problem: this fires on every multi-source run, so treating
    # it as a flag made every run ask to be reviewed and the status meant nothing.
    notes.append(
        f"Obras por fuente — OpenAlex {len(openalex_articles)}, Scopus {len(scopus_articles)}, "
        f"WoS {len(wos_articles)}, zbMATH {len(zbmath_articles)}, INSPIRE {len(inspire_articles)}; "
        f"{len(merged)} tras fusionar por DOI."
    )

    # Only OpenAlex exposes author identifiers, so a merged article with no OpenAlex
    # record can never yield a Type B or a self-citation: every citation to it is
    # counted as Type A by default. When that is a large share of the works, the
    # Type A total is an upper bound rather than a measurement, and whoever reads
    # the report has to know that.
    without_openalex = sum(1 for a in merged if not a.openalex_id)
    if without_openalex:
        share = round(100 * without_openalex / len(merged)) if merged else 0
        flags.append(
            f"{without_openalex} de {len(merged)} obras ({share} %) no tienen registro en OpenAlex. "
            "Sus citas se clasifican como Tipo A por defecto, porque sólo OpenAlex aporta "
            "identificadores de autor: el total de Tipo A es una cota superior."
        )

    return merged, flags, notes + _preprint_note(merged)


def _scopus_articles(sources: SourceBundle, orcid: str | None) -> list[Article]:
    if not sources.scopus.enabled:
        return []
    if sources.scopus_author_id:
        return sources.guarded(sources.scopus, "artículos",
                               lambda: sources.scopus.fetch_articles_by_author_id(sources.scopus_author_id), [])
    if orcid:
        return sources.guarded(sources.scopus, "artículos",
                               lambda: sources.scopus.fetch_articles_by_orcid(orcid), [])
    sources.notes.append("Scopus: sin AU-ID ni ORCID para consultar.")
    return []


def _wos_articles(sources: SourceBundle, orcid: str | None) -> list[Article]:
    if not sources.wos.enabled:
        return []
    if not orcid:
        sources.notes.append("Web of Science: sin ORCID para consultar.")
        return []
    articles = sources.guarded(sources.wos, "artículos",
                               lambda: sources.wos.fetch_articles_by_orcid(orcid), [])
    if articles and not any(a.wos_cited_by_count is not None for a in articles):
        # Documented limitation of this subscription, not an error — but it must be
        # visible so an empty WoS column is never read as "cero citas".
        # A documented property of this subscription, unchanged run to run.
        sources.notes.append(
            "Web of Science: la suscripción no devuelve conteos de citas; "
            "sus registros sólo sirven para contrastar cobertura."
        )
    return articles


def _zbmath_articles(sources: SourceBundle, author: Author) -> list[Article]:
    if not sources.zbmath.enabled:
        return []

    code = sources.zbmath_author_code
    if not code:
        code = sources.guarded(sources.zbmath, "resolución de autor",
                               lambda: sources.zbmath.resolve_author_code(author.display_name), None)
        if code:
            # A guess that decides whose papers are counted: it must be checked.
            sources.warnings.append(
                f"zbMATH: código de autor deducido del nombre ({code}); confírmalo en la ficha."
            )
        else:
            sources.notes.append(
                "zbMATH: no se pudo determinar un código de autor inequívoco; fuente omitida."
            )
            return []

    return sources.guarded(sources.zbmath, "artículos",
                           lambda: sources.zbmath.fetch_articles_by_code(code), [])


def _inspire_articles(sources: SourceBundle, author: Author) -> list[Article]:
    if not sources.inspire.enabled:
        return []

    recid = sources.inspire_author_recid
    if not recid and author.orcid:
        recid = sources.guarded(sources.inspire, "resolución de autor",
                                lambda: sources.inspire.resolve_recid_by_orcid(author.orcid), None)

    if not recid:
        # Normal for mathematicians. Name search is deliberately NOT used as a
        # fallback here: it can match a different person entirely (see inspire.py).
        return []

    return sources.guarded(sources.inspire, "artículos",
                           lambda: sources.inspire.fetch_articles_by_recid(recid), [])


# ── Stage 2: fetch citing works (parallel across articles) ────────────────────
def _fetch_one_article(
    art: Article,
    max_citing: int | None,
    sources: SourceBundle | None,
) -> Article:
    openalex_citing: list[CitingWork] = []
    if art.openalex_id:
        openalex_citing = openalex.fetch_citing_works(art.openalex_id, max_citing=max_citing)

    if sources is None:
        return art.model_copy(update={"citing_works": openalex_citing})

    scopus_citing: list[CitingWork] = []
    if art.scopus_eid and sources.scopus.enabled:
        scopus_citing = sources.guarded(sources.scopus, "citas",
                                        lambda: sources.scopus.fetch_citing_works(art.scopus_eid), [])

    zbmath_citing: list[CitingWork] = []
    if art.zbmath_id and sources.zbmath.enabled:
        zbmath_citing = sources.guarded(sources.zbmath, "citas",
                                        lambda: sources.zbmath.fetch_citing_works(art.zbmath_id), [])

    inspire_citing: list[CitingWork] = []
    if art.inspire_recid and sources.inspire.enabled:
        inspire_citing = sources.guarded(sources.inspire, "citas",
                                         lambda: sources.inspire.fetch_citing_works(art.inspire_recid), [])

    # WoS Starter cannot list citing documents at all, so it never appears here.
    merged = merge_citing_works(openalex_citing, scopus_citing, zbmath_citing, inspire_citing)

    return art.model_copy(update={"citing_works": merged})


def fetch_citing_works(
    articles: list[Article],
    max_citing: int | None = None,
    sources: SourceBundle | None = None,
    job_id: str | None = None,
) -> list[Article]:
    """Attach each article's citing works. I/O-bound, so fanned out over threads.

    Completions are consumed as they land so the caller can be told how far along
    this is — it is the long stage, and the only one with a real denominator — but
    results are written back by index, so the order of `articles` is preserved
    exactly as `pool.map` did.
    """
    out: list[Article | None] = [None] * len(articles)

    with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
        futures = {
            pool.submit(_fetch_one_article, art, max_citing, sources): i
            for i, art in enumerate(articles)
        }

        for done, future in enumerate(as_completed(futures), start=1):
            out[futures[future]] = future.result()
            progress.advance(job_id, done)

    return [art for art in out if art is not None]


# ── Stage 3: dedupe + classify ────────────────────────────────────────────────
def _dedupe_citing_works(citing_works: list[CitingWork]) -> list[CitingWork]:
    """Drop repeats within one article's citing list.

    Keyed on the normalized DOI when there is one and on a title prefix otherwise,
    since records coming from sources other than OpenAlex often carry no DOI.
    """
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[CitingWork] = []

    for cw in citing_works:
        doi_key = openalex.normalize_doi(cw.doi)
        if doi_key:
            if doi_key in seen_dois:
                continue
            seen_dois.add(doi_key)
        else:
            title_key = cw.title.lower().strip()[:100]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
        deduped.append(cw)

    return deduped


def classify_articles(author: Author, articles: list[Article]) -> list[Article]:
    """Dedupe each article's citing works and count them as Type A / Type B / self.

    Each citing work is compared with the authors of the article it cites
    (Article.coauthor_ids), as Rizoma defines Type B: someone who co-wrote a
    different paper with the researcher does not make a citation Type B.
    """
    updated: list[Article] = []
    for art in articles:
        if not art.citing_works:
            updated.append(art)
            continue

        deduped = _dedupe_citing_works(art.citing_works)

        cited_coauthor_ids = set(art.coauthor_ids)
        type_a = type_b = self_c = 0
        for cw in deduped:
            t = classify_citation_type(cw, author.openalex_id, cited_coauthor_ids)
            if t == "A":
                type_a += 1
            elif t == "B":
                type_b += 1
            else:
                self_c += 1

        updated.append(art.model_copy(update={
            "citing_works": deduped,
            "cites_type_a": type_a,
            "cites_type_b": type_b,
            "cites_self": self_c,
        }))

    return updated


# ── Stage 4: validate ─────────────────────────────────────────────────────────
def validate_counts(articles: list[Article], tolerance: float | None = None) -> list[str]:
    """Cross-check retrieved citations against reported counts. Returns new flags."""
    flags: list[str] = []
    tol = tolerance if tolerance is not None else settings.count_tolerance

    for art in articles:
        retrieved = len(art.citing_works)
        oa_expected = art.openalex_cited_by_count

        # retrieved >= oa_expected is normal; a shortfall beyond tolerance is a flag.
        if oa_expected > 0:
            gap = (retrieved - oa_expected) / oa_expected
            if gap < -tol:
                flags.append(
                    f"OpenAlex shortfall for '{art.title[:50]}': "
                    f"retrieved={retrieved} < OA reported={oa_expected} ({gap:.0%})."
                )

        total_classified = art.cites_type_a + art.cites_type_b + art.cites_self
        if total_classified != retrieved:
            flags.append(
                f"Classification mismatch for '{art.title[:50]}': "
                f"A+B+self={total_classified} ≠ retrieved={retrieved}."
            )

    return flags


# ── Entry point ───────────────────────────────────────────────────────────────
def run_analysis(
    author: Author,
    want_report: bool = False,
    report_language: str = "es",
    max_works: int | None = None,
    max_citing_per_work: int | None = None,
    job_id: str | None = None,
    comparison: dict | None = None,
    ollama_base: str | None = None,
    ollama_model: str | None = None,
    count_tolerance: float | None = None,
    sources: SourceBundle | None = None,
) -> tuple[AnalysisResult, str | None]:
    """Run the full pipeline for a resolved author. Returns (result, report_or_None).

    The Ollama arguments are only consulted when want_report is true; they come from
    the caller's own configuration (see EngineOverrides).
    """
    progress.set_phase(job_id, "works", detail=author.display_name)
    articles, flags, notes = fetch_works(author, max_works=max_works, sources=sources)

    # The long one: one round trip per article, so it reports n of N.
    progress.set_phase(job_id, "citing", done=0, total=len(articles))
    articles = fetch_citing_works(
        articles, max_citing=max_citing_per_work, sources=sources, job_id=job_id
    )

    progress.set_phase(job_id, "classify", total=len(articles))
    articles = classify_articles(author, articles)

    progress.set_phase(job_id, "validate")
    flags += validate_counts(articles, tolerance=count_tolerance)

    # Collected last: a source can disable itself part-way through the run, and the
    # reason only exists once that has happened.
    if sources is not None:
        flags += sources.source_warnings()
        notes += sources.source_notes()

    result = AnalysisResult(
        author=author,
        run_timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        articles=articles,
        flags=flags,
        notes=notes,
    )

    if want_report:
        progress.set_phase(job_id, "report")
        report = generate_report(
            result,
            language=report_language,
            comparison=comparison,
            ollama_base=ollama_base,
            ollama_model=ollama_model,
        )[0]
    else:
        report = None

    progress.finish(job_id)

    return result, report
