"""Which findings ask for a human, and which are merely worth recording.

The distinction is the point: SIAB marks a run "Por revisar" when there are flags,
so anything that fires on every run belongs in notes. Getting this wrong made every
multi-source analysis look like it needed attention, and a status that is always on
tells nobody anything.
"""
from __future__ import annotations

from app.pipeline import SourceBundle


def bundle(**kwargs) -> SourceBundle:
    return SourceBundle.build(**kwargs)


def test_an_unconfigured_source_is_a_note_not_a_warning() -> None:
    # Nobody entered a Scopus key. That is a deployment decision, not a fault.
    sources = bundle()

    assert any("Scopus" in n for n in sources.source_notes())
    assert not any("Scopus" in w for w in sources.source_warnings())


def test_a_rejected_key_is_a_warning() -> None:
    sources = bundle(scopus_api_key="wrong")
    sources.scopus.disable("Scopus: la clave de API fue rechazada (401 en búsqueda).")

    assert any("rechazada" in w for w in sources.source_warnings())
    assert not any("rechazada" in n for n in sources.source_notes())


def test_a_guessed_author_code_is_a_warning() -> None:
    # It decides whose papers get counted, so somebody has to confirm it.
    sources = bundle()
    sources.warnings.append("zbMATH: código de autor deducido del nombre (x.y); confírmalo en la ficha.")

    assert sources.source_warnings()
    assert not sources.source_notes() or "deducido" not in " ".join(sources.source_notes())


def test_source_counts_are_a_note() -> None:
    # "Obras por fuente — …" fires on every run with extra sources. As a flag it
    # made every single run ask to be reviewed.
    sources = bundle()
    sources.notes.append("Obras por fuente — OpenAlex 41, Scopus 1, WoS 0, zbMATH 33, INSPIRE 0; 52 tras fusionar.")

    assert any("Obras por fuente" in n for n in sources.source_notes())
    assert not any("Obras por fuente" in w for w in sources.source_warnings())


def test_a_clean_run_with_no_keys_configured_raises_no_warnings() -> None:
    # The ordinary CCM case today: OpenAlex plus zbMATH, no Scopus or WoS keys.
    # It must come back "Completo", which means no warnings at all.
    sources = bundle(use_zbmath=True, use_inspire=False)

    assert sources.source_warnings() == []
    assert sources.source_notes(), "the skipped sources are still recorded"


# ── Report comparison ─────────────────────────────────────────────────────────
def _result():
    from app.schemas import AnalysisResult, Author

    return AnalysisResult(
        author=Author(display_name="Michael Hrušák", openalex_id="A5065080063", works_count=166),
        run_timestamp="20260803T200000Z",
    )


def test_a_previous_analysis_reaches_the_report_context() -> None:
    from app.report import build_report_context

    context = build_report_context(_result(), {"fecha": "2026-08-03", "total_citas": 1467})

    assert context["comparacion_con_analisis_anterior"]["total_citas"] == 1467


def test_without_a_previous_analysis_the_key_is_absent_not_null() -> None:
    from app.report import build_report_context

    # Not `"...": None`: a null key still invites the model to narrate the absence,
    # which is how every report came to announce "este es el primer análisis".
    assert "comparacion_con_analisis_anterior" not in build_report_context(_result(), None)
