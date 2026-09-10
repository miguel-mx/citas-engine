"""Progress reporting: network-free, like the rest of the suite.

The properties that matter are not "the numbers are pretty" but that this can
never damage a run — an unknown id is a normal answer, a missing job_id is a
no-op, and the fan-out still returns articles in the order it was given them.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app import progress
from app.main import app
from app.pipeline import fetch_citing_works
from app.schemas import Article

client = TestClient(app)


def teardown_function() -> None:
    progress.finish("job-under-test")


def test_unknown_job_is_not_an_error() -> None:
    # The run may not have started yet, or may already have finished and handed
    # the caller the real answer. Neither is a failure worth a 404.
    response = client.get("/progress/nobody-here")

    assert response.status_code == 200
    assert response.json()["running"] is False


def test_progress_reports_phase_and_fraction() -> None:
    progress.start("job-under-test")
    progress.set_phase("job-under-test", "citing", done=0, total=8)
    progress.advance("job-under-test", 2)

    body = client.get("/progress/job-under-test").json()

    assert body["running"] is True
    assert body["phase"] == "citing"
    assert body["done"] == 2
    assert body["total"] == 8
    assert body["percent"] == 25


def test_a_phase_without_a_denominator_reports_no_percentage() -> None:
    progress.start("job-under-test")
    progress.set_phase("job-under-test", "works")

    # Better to show the phase alone than to invent a fraction for it.
    assert client.get("/progress/job-under-test").json()["percent"] is None


def test_finishing_forgets_the_run() -> None:
    progress.start("job-under-test")
    progress.finish("job-under-test")

    assert progress.get("job-under-test") is None


def test_progress_calls_without_a_job_id_do_nothing() -> None:
    # SIAB always sends one, but /analyze may be called by anything.
    progress.start(None)
    progress.set_phase(None, "citing", total=3)
    progress.advance(None, 1)
    progress.finish(None)


def _article(n: int) -> Article:
    return Article(title=f"Artículo {n}", openalex_id=None, openalex_cited_by_count=0)


def test_the_fan_out_preserves_input_order_while_counting() -> None:
    # It used to be pool.map, which preserves order for free; consuming completions
    # as they land must not have changed the order articles come back in.
    articles = [_article(n) for n in range(12)]

    result = fetch_citing_works(articles, job_id="job-under-test")

    assert [a.title for a in result] == [a.title for a in articles]
