"""Request/response contracts for the HTTP API (what Symfony sends/receives)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas import AnalysisResult, Author


class ResolveRequest(BaseModel):
    query: str = Field(..., description="OpenAlex ID (A…), ORCID, or a name")


class ResolveResponse(BaseModel):
    # "author" → resolved uniquely; "candidates" → disambiguation needed (name search)
    kind: Literal["author", "candidates"]
    author: Optional[Author] = None
    candidates: list[Author] = []


class EngineOverrides(BaseModel):
    """Configuration supplied by the caller instead of this service's .env.

    SIAB stores these in its own database and an administrator edits them at
    /admin/configuracion, so a change applies to the next request without
    restarting uvicorn. Anything left unset falls back to the engine's own config.
    """

    ollama_base: Optional[str] = Field(None, description="e.g. http://132.248.196.38:11434")
    ollama_model: Optional[str] = Field(None, description="must be pulled on that server")
    count_tolerance: Optional[float] = Field(
        None, ge=0, le=1, description="accepted shortfall vs OpenAlex's reported count"
    )


class AnalyzeRequest(EngineOverrides):
    author_id: str = Field(..., description="OpenAlex author ID, e.g. A5023888391")
    want_report: bool = False
    report_language: Literal["es", "en"] = "es"
    max_works: Optional[int] = None
    max_citing_per_work: Optional[int] = None

    # Opaque to the engine: whatever the caller wants to poll /progress/{job_id}
    # with while this request is still open. SIAB sends the analysis run's id.
    job_id: Optional[str] = Field(None, description="key for GET /progress/{job_id}")

    # Same as ReportRequest.comparison, for want_report runs that write the report
    # inline instead of asking for it afterwards.
    comparison: Optional[dict] = Field(None, description="previous run's figures for the report")

    # ── Extra bibliographic sources ──────────────────────────────────────────
    # Keys come from SIAB, where they are stored encrypted; a source with no key
    # is simply skipped and says so in the run's flags.
    scopus_api_key: Optional[str] = None
    wos_api_key: Optional[str] = None

    # Per-researcher identifiers. These are worth setting: an AU-ID pins the exact
    # Scopus profile, whereas the ORCID fallback depends on Scopus having recorded
    # the ORCID on each record.
    scopus_author_id: Optional[str] = Field(None, description="Scopus AU-ID, e.g. 6602738988")
    zbmath_author_code: Optional[str] = Field(None, description="e.g. hrusak.michael")
    inspire_author_recid: Optional[str] = Field(
        None,
        description="INSPIRE-HEP author recid. Leave unset for non-HEP authors — a "
                    "wrong recid silently merges another physicist's papers.",
    )

    use_zbmath: bool = True
    use_inspire: bool = True


class AnalyzeResponse(BaseModel):
    result: AnalysisResult
    report: Optional[str] = None


class SourceKeyRequest(BaseModel):
    """Check one API key against its service. The key is used for the request and
    nothing else: it is never logged, stored or returned."""

    source: Literal["scopus", "wos"]
    api_key: str


class SourceKeyResponse(BaseModel):
    ok: bool
    state: str
    message: str


class ReportRequest(EngineOverrides):
    result: AnalysisResult
    language: Literal["es", "en"] = "es"

    # The caller's previous analysis of this researcher, if it has one. Left unset
    # when there is none: the report then says nothing about earlier analyses,
    # rather than announcing that this is the first.
    comparison: Optional[dict] = Field(
        None, description="previous run's figures, e.g. {fecha, total_citas, total_articulos}"
    )


class ReportResponse(BaseModel):
    report: str
    context: dict
