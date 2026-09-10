# citas-engine

Stateless FastAPI service that performs the CCM/UNAM citation analysis, ported
from `citas/notebooks/citation_analysis_mvp.ipynb`. It is the "engine" half of
the hybrid architecture: the Symfony app (`../siab`) owns users, persistence,
jobs and UI, and calls these endpoints.

**Hard rule (inherited from the notebook):** the LLM never produces
bibliographic facts. All authors, counts and citing references come from
deterministic OpenAlex calls. Ollama/Gemma is used only to write the narrative
report, and only from figures already computed in Python.

## Layout

```
app/
├── main.py          # FastAPI app + routes
├── config.py        # env-driven settings singleton
├── schemas.py       # Author / CitingWork / Article / AnalysisResult (ported)
├── api_models.py    # request/response contracts
├── classify.py      # deterministic Type A/B/self classification
├── health.py        # dependency probes for /health/services
├── report.py        # es/en report strings + Ollama generation
├── pipeline.py      # plain pipeline (fetch → classify → validate → assemble)
└── sources/
    └── openalex.py  # OpenAlex client + resolve/fetch tools (ported)
tests/test_smoke.py  # network-free tests
```

The notebook orchestrates the same steps with LangGraph. That is not carried over
here: the flow is a fixed sequence with one `if` (report or not), no model-driven
control flow and no tool-calling loop, so it is written as plain functions over
real values. Each stage — `fetch_works`, `fetch_citing_works`, `classify_articles`,
`validate_counts` — is importable and testable on its own, and the service has no
LangChain dependency; the single LLM call is one POST to Ollama's `/api/chat`.

Scopus / WoS / zbMATH / INSPIRE live in the notebook and are intentionally left
out of this scaffold; they plug in as new modules under `sources/` plus extra
stages in `pipeline.py` without changing the schemas or the API.

## Run

```bash
cd citas-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # edit MAILTO / MODEL as needed
uvicorn app.main:app --reload --port 8001
```

Interactive docs at http://localhost:8001/docs.

## Endpoints

| Method | Path       | Purpose |
|--------|------------|---------|
| GET    | `/health`  | Liveness + effective config. |
| GET    | `/health/services` | Deep check: probes OpenAlex and Ollama (model present?) and reports the optional sources' configuration. Feeds the Symfony dashboard's service panel; time-boxed and never fails as a whole. |
| POST   | `/resolve` | `{query}` → `{kind:"author", author}` or `{kind:"candidates", candidates}` (name search → disambiguation). |
| POST   | `/analyze` | `{author_id, want_report?, report_language?, max_works?, max_citing_per_work?}` → `{result, report?}`. The heavy pipeline. |
| POST   | `/report`  | `{result, language}` → `{report, context}`. Needs Ollama. |

`/analyze` can take minutes for prolific authors — Symfony calls it from a
Messenger worker, not a web request.

## Test

```bash
pytest -q      # no network required
```

## Gold-standard cases (from the notebook)

- Michael Hrušák — ORCID `0000-0002-1692-2216`
- Salvador García-Ferreira — ORCID `0000-0002-6400-6394`
