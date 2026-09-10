"""FastAPI entrypoint for the CCM/UNAM citation engine.

Stateless by design: the engine computes, Symfony persists. Three endpoints
mirror the agreed contract — /resolve (with disambiguation), /analyze (the heavy
pipeline), and /report (Spanish/English narrative).
"""
from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException

from app.api_models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ReportRequest,
    ReportResponse,
    ResolveRequest,
    ResolveResponse,
    SourceKeyRequest,
    SourceKeyResponse,
)
from app.config import settings
from app import progress
from app.health import probe_all, probe_source_key
from app.pipeline import SourceBundle, run_analysis
from app.report import generate_report
from app.schemas import Author
from app.sources import openalex

app = FastAPI(
    title="CCM/UNAM Citation Engine",
    version="0.1.0",
    summary="Deterministic OpenAlex citation analysis with an optional LLM-written report.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "openalex_base": settings.openalex_base, "model": settings.model}


@app.get("/health/services")
async def health_services(
    ollama_base: str | None = None,
    ollama_model: str | None = None,
    scopus_key: bool | None = None,
    wos_key: bool | None = None,
) -> dict:
    """Deep health check: probes OpenAlex and Ollama and reports the optional
    sources' configuration. Feeds the Symfony dashboard's service panel, so it is
    time-boxed (see app.health.PROBE_TIMEOUT) and never fails as a whole.

    The Ollama query parameters let SIAB probe the server it will actually send
    reports to, rather than whatever this service's .env happens to name.

    scopus_key / wos_key say whether SIAB holds a key for those sources — it stores
    them, this service does not, so without being told it would report "sin clave"
    for a key that works. Booleans, never the keys: this is a GET and query strings
    are logged."""
    return await probe_all(
        ollama_base=ollama_base,
        ollama_model=ollama_model,
        scopus_key=scopus_key,
        wos_key=wos_key,
    )


@app.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest) -> ResolveResponse:
    """Resolve a researcher. A unique hit (OpenAlex ID / ORCID) returns kind=author;
    a name search returns kind=candidates for the UI to disambiguate."""
    try:
        outcome = openalex.resolve_author(req.query)
    except ValueError as e:  # malformed ORCID — reject before trusting the network result
        raise HTTPException(status_code=422, detail=str(e)) from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenAlex error: {e}") from e

    if isinstance(outcome, Author):
        return ResolveResponse(kind="author", author=outcome)
    return ResolveResponse(kind="candidates", candidates=outcome)


@app.post("/health/source-key", response_model=SourceKeyResponse)
async def check_source_key(req: SourceKeyRequest) -> SourceKeyResponse:
    """Does this API key work? Answered by asking the service itself."""
    return SourceKeyResponse(**await probe_source_key(req.source, req.api_key))


@app.get("/progress/{job_id}")
def analysis_progress(job_id: str) -> dict:
    """Where a running analysis has got to.

    Polled by SIAB while its own /analyze request is still open, so it answers in
    microseconds and never touches the network. An unknown id is not an error: the
    run may not have reached the pipeline yet, or may already have finished and
    handed the caller the real answer.
    """
    state = progress.get(job_id)

    return {"job_id": job_id, "running": state is not None, **(state or {})}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full citation pipeline for a resolved author id."""
    progress.start(req.job_id)
    try:
        resolved = openalex.resolve_author(req.author_id)
    except (ValueError, httpx.HTTPStatusError) as e:
        raise HTTPException(status_code=422, detail=f"Could not resolve author_id: {e}") from e
    if not isinstance(resolved, Author):
        raise HTTPException(
            status_code=422,
            detail="author_id must resolve to a single author (pass an OpenAlex ID, not a name).",
        )

    sources = SourceBundle.build(
        scopus_api_key=req.scopus_api_key,
        wos_api_key=req.wos_api_key,
        scopus_author_id=req.scopus_author_id,
        zbmath_author_code=req.zbmath_author_code,
        inspire_author_recid=req.inspire_author_recid,
        use_zbmath=req.use_zbmath,
        use_inspire=req.use_inspire,
    )

    result, report = run_analysis(
        resolved,
        want_report=req.want_report,
        report_language=req.report_language,
        max_works=req.max_works,
        max_citing_per_work=req.max_citing_per_work,
        job_id=req.job_id,
        comparison=req.comparison,
        ollama_base=req.ollama_base,
        ollama_model=req.ollama_model,
        count_tolerance=req.count_tolerance,
        sources=sources,
    )
    return AnalyzeResponse(result=result, report=report)


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest) -> ReportResponse:
    """Generate a narrative report from an already-computed result (needs Ollama)."""
    try:
        text, context = generate_report(
            req.result,
            language=req.language,
            comparison=req.comparison,
            ollama_base=req.ollama_base,
            ollama_model=req.ollama_model,
        )
    except Exception as e:  # Ollama unreachable / model missing
        raise HTTPException(status_code=503, detail=f"Report generation failed: {e}") from e
    return ReportResponse(report=text, context=context)
