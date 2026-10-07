"""Network-free smoke tests: app wiring, classification, and report context.

Run: pytest -q  (from the citas-engine/ directory)
"""
import json

import httpx
from fastapi.testclient import TestClient

from app import health
from app.classify import classify_citation_type
from app.config import settings
from app.main import app
from app.pipeline import _preprint_note, classify_articles, validate_counts
from app.report import build_report_context, generate_report
from app.schemas import AnalysisResult, Article, Author, CitingWork
from app.sources.openalex import classify_work

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_classification_precedence():
    researcher = "A1"
    coauthors = {"A2"}
    # self-citation: researcher is among the citing authors
    assert classify_citation_type(CitingWork(title="x", author_ids=["A1", "A9"]), researcher, coauthors) == "self"
    # co-author (Type B)
    assert classify_citation_type(CitingWork(title="x", author_ids=["A2", "A9"]), researcher, coauthors) == "B"
    # external (Type A)
    assert classify_citation_type(CitingWork(title="x", author_ids=["A9"]), researcher, coauthors) == "A"
    # no author ids (e.g. non-OpenAlex record) falls through to Type A
    assert classify_citation_type(CitingWork(title="x", author_ids=[]), researcher, coauthors) == "A"


def test_report_context_percentages_sum():
    result = AnalysisResult(
        author=Author(openalex_id="A1", display_name="Test", works_count=1),
        run_timestamp="20260729T000000Z",
        articles=[Article(title="a", year=2020, openalex_cited_by_count=3,
                          cites_type_a=2, cites_type_b=1, cites_self=1)],
    )
    ctx = build_report_context(result)
    assert ctx["total_citas_tipo_a"] == 2
    assert ctx["total_citas"] == 4
    assert abs(ctx["porcentaje_tipo_a"] + ctx["porcentaje_tipo_b"] + ctx["porcentaje_autocitas"] - 100.0) < 0.2


def test_classify_articles_dedupes_and_counts():
    """The classify stage is a pure function of (author, articles), so it runs
    without touching the network — this is what the plain pipeline buys us."""
    author = Author(openalex_id="A1", display_name="Test", works_count=1)
    articles = [Article(
        title="paper",
        openalex_id="W1",
        coauthor_ids=["A2"],
        citing_works=[
            CitingWork(title="dup", doi="10.1/x", author_ids=["A9"]),
            CitingWork(title="dup again", doi="https://doi.org/10.1/X", author_ids=["A9"]),
            CitingWork(title="by a coauthor", author_ids=["A2"]),
            CitingWork(title="by the author", author_ids=["A1"]),
        ],
    )]

    [art] = classify_articles(author, articles)

    # The two DOI-identical records collapse to one despite differing titles/case.
    assert len(art.citing_works) == 3
    assert (art.cites_type_a, art.cites_type_b, art.cites_self) == (1, 1, 1)



def _classify_two_articles(citing: CitingWork) -> tuple[Article, Article]:
    """Researcher A1 wrote W1 with X (A2) and W2 with Y (A3); `citing` cites both."""
    author = Author(openalex_id="A1", display_name="Test", works_count=2)
    w1 = Article(title="one", openalex_id="W1", coauthor_ids=["A2"], citing_works=[citing])
    w2 = Article(title="two", openalex_id="W2", coauthor_ids=["A3"], citing_works=[citing])
    return tuple(classify_articles(author, [w1, w2]))


def _counts(art: Article) -> tuple[int, int, int]:
    return art.cites_type_a, art.cites_type_b, art.cites_self


def test_coauthor_of_another_article_is_type_a():
    """Rizoma compares against the authors of the cited work only: X co-wrote W1
    but not W2, so X citing W2 without the researcher is Type A there."""
    w1, w2 = _classify_two_articles(CitingWork(title="by X", author_ids=["A2", "A9"]))

    assert _counts(w1) == (0, 1, 0)
    assert _counts(w2) == (1, 0, 0)


def test_coauthor_of_the_cited_article_is_type_b():
    w1, w2 = _classify_two_articles(CitingWork(title="by Y", author_ids=["A3"]))

    assert _counts(w1) == (1, 0, 0)
    assert _counts(w2) == (0, 1, 0)


def test_researcher_on_the_citing_work_is_self_even_with_coauthors():
    w1, w2 = _classify_two_articles(
        CitingWork(title="by everyone", author_ids=["A2", "A1", "A3"])
    )

    assert _counts(w1) == (0, 0, 1)
    assert _counts(w2) == (0, 0, 1)


def test_cited_article_without_author_ids_is_type_a():
    """A record from a source other than OpenAlex carries no author ids, so no
    overlap can be shown: Type A, even for someone who co-wrote another paper."""
    author = Author(openalex_id="A1", display_name="Test", works_count=2)
    with_ids = Article(title="openalex", openalex_id="W1", coauthor_ids=["A2"])
    without_ids = Article(
        title="scopus only",
        doi="10.1/s",
        citing_works=[CitingWork(title="by X", doi="10.1/c", author_ids=["A2"])],
    )

    _, art = classify_articles(author, [with_ids, without_ids])

    assert _counts(art) == (1, 0, 0)

def test_validate_counts_flags_shortfall_against_openalex():
    articles = [Article(title="under-retrieved", openalex_cited_by_count=100,
                        citing_works=[], cites_type_a=0)]
    flags = validate_counts(articles)
    assert any("shortfall" in f for f in flags)

    # Retrieving at least the reported count, fully classified, is silent.
    ok = [Article(title="fine", openalex_cited_by_count=1,
                  citing_works=[CitingWork(title="c")], cites_type_a=1)]
    assert validate_counts(ok) == []


def test_generate_report_calls_ollama_chat(respx_mock):
    """Guards the hand-rolled Ollama POST that replaced ChatOllama: correct
    endpoint, non-streaming, and the reply read out of message.content."""
    route = respx_mock.post(f"{settings.ollama_base}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "# Informe"}})
    )
    result = AnalysisResult(
        author=Author(openalex_id="A1", display_name="Test", works_count=1),
        run_timestamp="20260729T000000Z",
        articles=[Article(title="a", year=2020, cites_type_a=1)],
    )

    text, context = generate_report(result, language="es")

    assert text == "# Informe"
    assert context["total_citas_tipo_a"] == 1

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == settings.model
    assert sent["stream"] is False
    # The prompt must carry the validated figures the model is forbidden to invent.
    assert "DATOS VALIDADOS" in sent["messages"][0]["content"]


def test_report_overrides_beat_engine_env(respx_mock):
    """SIAB owns the Ollama address and model, so a per-request override must win
    over this service's .env — that is what removes the need to restart uvicorn."""
    other = "http://10.0.0.9:11434"
    assert other != settings.ollama_base

    route = respx_mock.post(f"{other}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "ok"}})
    )
    result = AnalysisResult(
        author=Author(openalex_id="A1", display_name="Test", works_count=1),
        run_timestamp="20260730T000000Z",
        articles=[Article(title="a", cites_type_a=1)],
    )

    generate_report(result, ollama_base=other, ollama_model="otro-modelo")

    # Posted to the override's host, not settings.ollama_base, and asked for its model.
    assert route.called
    assert json.loads(route.calls.last.request.content)["model"] == "otro-modelo"


def test_resolve_rejects_malformed_orcid():
    # ORCID-shaped prefix with trailing garbage: caught by the format guard
    # before any network call, so this test needs no connectivity.
    r = client.post("/resolve", json={"query": "0000-0002-1692-2216-extra"})
    assert r.status_code == 422


def test_health_services_reports_every_dependency(monkeypatch):
    """The deep check must answer for every dependency even when nothing responds —
    a failed probe is a result, not an error. Probes are aimed at a closed local
    port so the test stays off the network."""
    dead = "http://127.0.0.1:9"
    monkeypatch.setattr(health.settings, "openalex_base", dead)
    monkeypatch.setattr(health.settings, "ollama_base", dead)

    r = client.get("/health/services")
    assert r.status_code == 200

    payload = r.json()
    assert "checked_at" in payload

    states = {s["key"]: s for s in payload["services"]}
    assert set(states) == {"openalex", "ollama", "scopus", "wos"}

    # Unreachable dependencies report down with a reason, never an HTTP 500.
    for key in ("openalex", "ollama"):
        assert states[key]["state"] == "down"
        assert states[key]["message"]

    # No keys in the test env, and no client implemented yet either.
    assert states["scopus"]["state"] == "not_configured"
    assert states["wos"]["state"] == "not_configured"


# ── Preprint detection ────────────────────────────────────────────────────────
def _work(**over) -> dict:
    """A minimal OpenAlex work record, shaped like the real payload."""
    work = {
        "id": "https://openalex.org/W1",
        "type": "article",
        "doi": "https://doi.org/10.1016/j.example.2024.01.001",
        "primary_location": {
            "version": "publishedVersion",
            "source": {"type": "journal", "display_name": "Journal of Examples"},
        },
    }
    work.update(over)
    return work


def test_published_article_is_not_a_preprint():
    assert classify_work(_work()) == ("article", None)


def test_openalex_own_preprint_type_is_honoured():
    work = _work(
        type="preprint",
        primary_location={"version": "submittedVersion",
                          "source": {"type": "repository",
                                     "display_name": "arXiv (Cornell University)"}},
    )
    # The host institution OpenAlex appends is dropped: the badge says "arXiv".
    assert classify_work(work) == ("preprint", "arXiv")


def test_arxiv_doi_beats_a_wrong_openalex_type():
    """arXiv records are routinely typed "article". The DOI prefix is arXiv's own
    DataCite namespace, so it settles the question on its own."""
    work = _work(
        type="article",
        doi="https://doi.org/10.48550/arXiv.2301.01234",
        primary_location={"version": None,
                          "source": {"type": "repository", "display_name": "arXiv"}},
    )
    assert classify_work(work) == ("preprint", "arXiv")


def test_green_oa_manuscript_of_a_published_paper_is_not_a_preprint():
    """The false positive a repository/version rule would produce, taken from real
    data: Hrušák's "Parametrized principles" is typed "article", its best open copy
    is the *submitted* manuscript in a university repository — and it is a published
    Transactions of the AMS paper, as its DOI says. Hiding it would drop a real
    publication out of a figure that feeds an evaluation."""
    work = _work(
        type="article",
        doi="https://doi.org/10.1090/s0002-9947-03-03446-9",
        primary_location={
            "version": "submittedVersion",
            "source": {"type": "repository",
                       "display_name": "UEA Digital Repository (University of East Anglia)"},
        },
    )
    assert classify_work(work) == ("article", None)


def test_digital_library_copy_is_not_a_preprint():
    """Same shape with no DOI at all: a digitised published paper in a national
    mathematics library. Typed "other" by OpenAlex, and left alone by us."""
    work = _work(
        type="other",
        doi=None,
        primary_location={"version": "submittedVersion",
                          "source": {"type": "repository",
                                     "display_name": "Czech digital mathematics library"}},
    )
    assert classify_work(work) == ("other", None)


def test_repository_hosting_alone_never_decides():
    """A published version sitting in a repository is a publication; the source being
    a repository says nothing on its own."""
    work = _work(
        type="article",
        primary_location={"version": "publishedVersion",
                          "source": {"type": "repository", "display_name": "HAL"}},
    )
    assert classify_work(work) == ("article", None)


def test_missing_fields_do_not_crash_the_classifier():
    assert classify_work({}) == (None, None)
    assert classify_work({"primary_location": None, "type": None}) == (None, None)


def test_preprint_note_counts_by_repository():
    articles = [
        Article(title="a", work_type="article"),
        Article(title="b", work_type="preprint", repository="arXiv"),
        Article(title="c", work_type="preprint", repository="arXiv"),
        Article(title="d", work_type="preprint", repository="bioRxiv"),
    ]
    note = _preprint_note(articles)[0]
    assert "3 de 4" in note
    assert "arXiv 2" in note
    assert "bioRxiv 1" in note


def test_preprint_note_is_silent_when_there_are_none():
    assert _preprint_note([Article(title="a", work_type="article")]) == []


def test_a_result_records_the_classification_rule(monkeypatch):
    """Every fresh result says which rule produced its A/B/self figures, so a
    caller can refuse to compare it with figures from another rule."""
    from app import pipeline
    from app.classify import CLASSIFICATION_RULE

    author = Author(openalex_id="A1", display_name="Test", works_count=0)
    monkeypatch.setattr(pipeline, "fetch_works", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(pipeline, "fetch_citing_works", lambda articles, **k: articles)

    result, _ = pipeline.run_analysis(author)

    assert result.classification_rule == CLASSIFICATION_RULE


def test_a_stored_snapshot_without_the_rule_is_not_read_as_current():
    """Snapshots stored before the field existed come back through /report; they
    were computed under the earlier rule and must not pass for current ones."""
    old = {"author": {"openalex_id": "A1", "display_name": "T", "works_count": 0},
           "run_timestamp": "20260729T000000Z"}

    assert AnalysisResult.model_validate(old).classification_rule is None
