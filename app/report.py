"""Narrative report generation (Cell 6, write_report node).

The LLM writes prose only. Every figure it may state is precomputed in Python
and handed to it as validated JSON — including percentages — so the model never
does arithmetic and never invents a number. Ported verbatim from the notebook.
"""
from __future__ import annotations

import json

import httpx

from app.config import settings
from app.schemas import AnalysisResult

# ── Report language strings (one source of truth per language) ────────────────
_REPORT_STRINGS = {
    "es": {
        "role": "Eres un asistente bibliométrico del Centro de Ciencias Matemáticas (CCM), UNAM.",
        "style": "Escribe un informe bibliométrico formal en español, en el estilo de un informe SECIHTI/Rizoma.",
        "grounding": (
            "USA ÚNICAMENTE los datos del siguiente JSON — no inventes cifras, DOIs, ni referencias, "
            "y no calcules tus propios porcentajes o promedios; usa los que ya vienen en el JSON "
            "(porcentaje_tipo_a, porcentaje_tipo_b, porcentaje_autocitas)."
        ),
        "content": (
            "El informe debe incluir: resumen general, distribución por año, obras más citadas, "
            "y una nota sobre la distinción Tipo A / Tipo B según las reglas de la plataforma Rizoma (SECIHTI)."
        ),
        "comparison": (
            "Si aparece la clave 'comparacion_con_analisis_anterior', añade una sección breve titulada "
            "'Comparación con el análisis anterior' indicando la fecha previa y las citas y artículos "
            "nuevos identificados desde entonces. Si esa clave NO aparece en los datos, no escribas "
            "ninguna sección de comparación y no menciones análisis anteriores ni afirmes que este sea "
            "el primero: no dispones de esa información."
        ),
        "def_header": "DEFINICIÓN OFICIAL DE CITAS (plataforma Rizoma, SECIHTI):",
        "def_a": (
            "  - Tipo A: citas realizadas por autores externos; ni el investigador ni ninguno de sus "
            "coautores participa en el documento citante."
        ),
        "def_b": (
            "  - Tipo B: citas realizadas en documentos donde participa algún coautor del investigador, "
            "pero en los cuales el investigador mismo no es autor."
        ),
        "def_self": (
            "  - Autocitas: documentos citantes donde el propio investigador figura como autor; "
            "NO se contabilizan en ninguna categoría."
        ),
        "data_label": "DATOS VALIDADOS",
    },
    "en": {
        "role": "You are a bibliometrics assistant at the Centro de Ciencias Matemáticas (CCM), UNAM.",
        "style": "Write a formal bibliometric report in English, in the style of a SECIHTI/Rizoma report.",
        "grounding": (
            "USE ONLY the data in the following JSON — do not invent figures, DOIs, or references, "
            "and do not compute your own percentages or averages; use the ones already provided in "
            "the JSON (porcentaje_tipo_a, porcentaje_tipo_b, porcentaje_autocitas)."
        ),
        "content": (
            "The report must include: an overall summary, year-by-year distribution, most-cited works, "
            "and a note on the Type A / Type B distinction per the Rizoma platform's rules (SECIHTI)."
        ),
        "comparison": (
            "If the key 'comparacion_con_analisis_anterior' is present, add a short section titled "
            "'Comparison with the previous analysis' stating the previous date and the new citations "
            "and articles identified since then. If that key is NOT present in the data, write no "
            "comparison section and do not mention previous analyses or claim this is the first one: "
            "you do not have that information."
        ),
        "def_header": "OFFICIAL CITATION DEFINITION (Rizoma platform, SECIHTI):",
        "def_a": (
            "  - Type A: citations by external authors; neither the researcher nor any of their "
            "co-authors is an author of the citing document."
        ),
        "def_b": (
            "  - Type B: citations in documents where a co-author of the researcher participates, "
            "but the researcher themselves is not an author."
        ),
        "def_self": (
            "  - Self-citations: citing documents where the researcher themselves appears as an author; "
            "NOT counted in either category."
        ),
        "data_label": "VALIDATED DATA",
    },
}


def build_report_context(result: AnalysisResult, comparison: dict | None = None) -> dict:
    """Precompute every figure the report may state (percentages included).

    `comparison` is the caller's own earlier analysis of the same researcher; only
    SIAB knows about those, so the engine never invents one.
    """
    articles = result.articles
    total_a = sum(a.cites_type_a for a in articles)
    total_b = sum(a.cites_type_b for a in articles)
    total_self = sum(a.cites_self for a in articles)
    total_all = total_a + total_b + total_self

    def _pct(n: int) -> float:
        return round(100 * n / total_all, 1) if total_all else 0.0

    by_year: dict[int, int] = {}
    for art in articles:
        if art.year:
            by_year[art.year] = by_year.get(art.year, 0) + 1

    top_cited = sorted(articles, key=lambda a: a.openalex_cited_by_count, reverse=True)[:5]

    context = {
        "autor": result.author.display_name,
        "orcid": result.author.orcid,
        "total_articulos": len(articles),
        "total_citas_tipo_a": total_a,
        "total_citas_tipo_b": total_b,
        "autocitas": total_self,
        "total_citas": total_all,
        "porcentaje_tipo_a": _pct(total_a),
        "porcentaje_tipo_b": _pct(total_b),
        "porcentaje_autocitas": _pct(total_self),
        "distribucion_por_anio": by_year,
        "top_5_mas_citados": [
            {
                "titulo": a.title, "anio": a.year,
                "citas_openalex": a.openalex_cited_by_count,
                "citas_scopus": a.scopus_cited_by_count,
                "citas_wos": a.wos_cited_by_count,
                "citas_zbmath": a.zbmath_cited_by_count,
                "citas_inspire": a.inspire_cited_by_count,
            }
            for a in top_cited
        ],
        "fecha_analisis": result.run_timestamp,
    }

    # Omitted entirely rather than sent as null. A key whose value is null still
    # invites the model to narrate the absence — which is exactly how every report
    # came to announce "este es el primer análisis registrado para este
    # investigador", including for researchers with a dozen runs behind them.
    if comparison is not None:
        context["comparacion_con_analisis_anterior"] = comparison

    return context


def _ollama_chat(prompt: str, base: str, model: str) -> str:
    """One non-streaming completion from Ollama's native /api/chat.

    Posted directly rather than through a client library: this is the service's
    only LLM call, and it is a single request with a single user message.
    """
    response = httpx.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=settings.ollama_timeout,
    )
    response.raise_for_status()
    payload = response.json()

    try:
        return payload["message"]["content"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected Ollama response: {str(payload)[:200]}") from e


def generate_report(
    result: AnalysisResult,
    language: str = "es",
    comparison: dict | None = None,
    ollama_base: str | None = None,
    ollama_model: str | None = None,
) -> tuple[str, dict]:
    """Generate the narrative report. Returns (report_text, context).

    `ollama_base` / `ollama_model` come from SIAB, which owns those settings in its
    own database — an administrator can change the server or the model there and it
    applies to the next call, with no engine restart. They fall back to this
    service's .env when the caller sends nothing.

    Requires a reachable Ollama server with that model pulled; callers are expected
    to surface the failure (main.py turns it into a 503).
    """
    base = ollama_base or settings.ollama_base
    model = ollama_model or settings.model
    strings = _REPORT_STRINGS.get(language, _REPORT_STRINGS["es"])
    context = build_report_context(result, comparison)
    structured_context = json.dumps(context, ensure_ascii=False, indent=2)

    prompt = (
        f"{strings['role']}\n"
        f"{strings['style']}\n"
        f"{strings['grounding']}\n"
        f"{strings['content']}\n"
        f"{strings['comparison']}\n"
        f"{strings['def_header']}\n"
        f"{strings['def_a']}\n"
        f"{strings['def_b']}\n"
        f"{strings['def_self']}\n\n"
        f"{strings['data_label']}:\n{structured_context}"
    )

    return _ollama_chat(prompt, base, model), context
